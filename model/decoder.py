"""
attention_decoder.py
====================
Pointer decoder para el agente AM-Actor-Critic.

Recibe el vector de contexto h_t (de ContextNetwork) y los embeddings de nodos
(de AttentionEncoder) y produce la acción (destino j, opción de precio o), con
o ∈ {nivel 0 … K−1, VACÍO} (especificacion_fase0_ruteo_pricing.md §5).

Flujo interno
-------------
1. GlimpseAttention  — cross-attention (h_t → embeddings) → h_t refinado
2. PointerAttention  — compatibilidad dot-product + tanh(C) → logits por nodo
3. apply_mask        — logits de destinos inválidos → -inf; softmax → π(j|s)
4. PriceHead         — valor esperado por nivel + MLP sobre [h_t ; glimpse ; e_j ; cotización de la lane],
                       enmascarado con las opciones válidas de j → π(o|s, j)

Orden autorregresivo: primero j (máscara = el destino tiene alguna opción válida), después o.
    log π(j, o | s) = log π(j | s) + log π(o | s, j)
Si para j solo hay una opción válida (p. ej. solo VACÍO), log π(o | s, j) = 0. La entropía se devuelve por
separado (destino y precio) para poder ponderarlas distinto en el PPO. π(j | s) no depende de la cabeza de
precio, así que la comparación B puede fijar el destino del DRL y cambiar solo la regla de precio.
· Entrenamiento (PPO): act_cotizacion samplea; act_mixto samplea en unas filas y toma el argmax en el resto
  (rollout muestreado + greedy del self-critical en un solo batch); evaluate_cotizacion recalcula log π(j, o)
  y las entropías.
· Evaluación (greedy): greedy_cotizacion (argmax de j y de o); greedy_precio para un destino dado.

Compatibilidad
--------------
· La máscara de destinos es la del entorno (routing_env.VectorRoutingEnv): tensor bool (o 0/1), True = válido;
  la de opciones es env.opciones_validas (bool, True = válida).
· Sin dependencias de am_agent.py, am_training.py ni routing_env.py.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from config import CHECK_TENSORS
from model.debug_utils import check_tensor
from model.encoder import COT_P, COT_V, N_COT_FEATURES
from pricing import N_NIVELES


# ─────────────────────────────────────────────────────────────────────────────
# Bloque 1: Glimpse — cross-attention para refinar el contexto
# ─────────────────────────────────────────────────────────────────────────────

class GlimpseAttention(nn.Module):
    """
    Cross-attention multi-cabeza: el contexto h_t consulta los embeddings
    de nodos y produce un contexto refinado h_t'.

    Query  : h_t            — vector de contexto (batch, d_h)
    Keys   : embeddings     — (batch, N, d_h)
    Values : embeddings     — (batch, N, d_h)
    Salida : h_t'           — (batch, d_h)

    Parámetros
    ----------
    d_h     : int — dimensión del embedding.
    n_heads : int — cabezas de atención (debe dividir d_h).
    """

    def __init__(self, d_h: int, n_heads: int):
        super().__init__()
        assert d_h % n_heads == 0, "d_h debe ser divisible por n_heads"
        self.n_heads = n_heads
        self.d_k     = d_h // n_heads

        self.W_q = nn.Linear(d_h, d_h, bias=False)
        self.W_k = nn.Linear(d_h, d_h, bias=False)
        self.W_v = nn.Linear(d_h, d_h, bias=False)
        self.W_o = nn.Linear(d_h, d_h, bias=False)

    def forward(
        self,
        context:    torch.Tensor,   # (batch, d_h)
        embeddings: torch.Tensor,   # (batch, N, d_h)
        mask:       torch.Tensor,   # (batch, N) bool — True = inválido
    ) -> torch.Tensor:
        """Retorna el contexto refinado h_t' de forma (batch, d_h)."""
        B, N, _ = embeddings.shape
        H, dk   = self.n_heads, self.d_k

        # Query desde el contexto: (batch, 1, d_h) → (batch, H, 1, dk)
        Q = self.W_q(context).view(B, 1, H, dk).transpose(1, 2)
        # Keys/Values desde embeddings: (batch, N, d_h) → (batch, H, N, dk)
        K = self.W_k(embeddings).view(B, N, H, dk).transpose(1, 2)
        V = self.W_v(embeddings).view(B, N, H, dk).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / (dk ** 0.5)  # (B,H,1,N)

        # Enmascarar nodos inválidos antes del softmax
        if mask is not None:
            scores = scores.masked_fill(
                mask.unsqueeze(1).unsqueeze(2),  # (B,1,1,N)
                float("-inf")
            )

        attn = F.softmax(scores, dim=-1)          # (B, H, 1, N)
        check_tensor("GlimpseAttention attn", attn)
        out  = torch.matmul(attn, V)              # (B, H, 1, dk)
        out  = out.squeeze(2).transpose(1, 2).contiguous().view(B, -1)  # (B, d_h)
        return self.W_o(out)


# ─────────────────────────────────────────────────────────────────────────────
# Bloque 2: Pointer — logits finales con clipping tanh
# ─────────────────────────────────────────────────────────────────────────────

class PointerAttention(nn.Module):
    """
    Atención de puntero: compatibilidad entre el contexto refinado h_t' y
    los embeddings de nodos, con clipping tanh (C · tanh(score/sqrt(dk))).

    La constante C (default 10) controla la entropía de la distribución:
    valores altos producen distribuciones más concentradas.

    Entrada : h_t' (batch, d_h), embeddings (batch, N, d_h)
    Salida  : logits sin enmascarar (batch, N)
    """

    def __init__(self, d_h: int, clip_C: float = 10.0):
        super().__init__()
        self.clip_C = clip_C
        self.W_q    = nn.Linear(d_h, d_h, bias=False)
        self.W_k    = nn.Linear(d_h, d_h, bias=False)
        self.d_k    = d_h

    def forward(
        self,
        context:    torch.Tensor,   # (batch, d_h)
        embeddings: torch.Tensor,   # (batch, N, d_h)
    ) -> torch.Tensor:
        """Retorna logits sin enmascarar de forma (batch, N)."""
        q = self.W_q(context).unsqueeze(1)          # (batch, 1, d_h)
        k = self.W_k(embeddings)                     # (batch, N, d_h)
        scores = torch.bmm(q, k.transpose(1, 2))     # (batch, 1, N)
        scores = scores.squeeze(1) / (self.d_k ** 0.5)   # (batch, N)
        return self.clip_C * torch.tanh(scores)


# ─────────────────────────────────────────────────────────────────────────────
# Bloque 3: cabeza de precio (cotización conjunta)
# ─────────────────────────────────────────────────────────────────────────────

class PriceHead(nn.Module):
    """
    Logits de las opciones de precio (K niveles + VACÍO) para el destino ya elegido.

    Entrada : h_t (batch, d_h), glimpse h_t' (batch, d_h), e_j (batch, d_h) y las features de cotización de la
              lane (batch, N_COT_FEATURES): entre ellas v_k = P_k·(p_k·d − c)/p95 y P_k (model/encoder.py)
    Salida  : logits sin enmascarar (batch, n_opciones)

        logit_k = τ · (v_k + P_k · Δ) + MLP_k          (los niveles; VACÍO: solo MLP)

    Δ (aprendido, en unidades de p95) es lo que vale ganar la carga más allá del tramo: V(j) − V(perder). El
    nivel óptimo es argmax_k P_k·(p_k·d − c + V(j) − V(perder)), así que la cabeza solo tiene que aprender Δ; con
    Δ = 0 arranca en la cotización miope. El MLP corrige lo que falte.

    El embedding e_j ya trae, por la atención del encoder, el contexto de todo el grafo (cargas, competencia,
    índice de mercado, continuación esperada desde j); h_t trae el momento del ciclo (tiempo, K, rechazos).
    """

    _INIT_SEED = 20261124

    def __init__(self, d_h: int, n_opciones: int):
        super().__init__()
        n_in = 3 * d_h + N_COT_FEATURES
        # RNG propio: crear la cabeza no consume el RNG global de torch, así que el resto del agente y el
        # crítico se inicializan igual que antes de agregarla (mismo entrenamiento con la misma semilla).
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self._INIT_SEED)
            self.net = nn.Sequential(
                nn.Linear(n_in, d_h),
                nn.ReLU(),
                nn.Linear(d_h, n_opciones),
            )
            self.delta = nn.Sequential(nn.Linear(n_in, d_h), nn.ReLU(), nn.Linear(d_h, 1))
            nn.init.zeros_(self.delta[-1].weight)
            nn.init.zeros_(self.delta[-1].bias)
            self.log_tau = nn.Parameter(torch.tensor(float(np.log(10.0))))

    def forward(self, context: torch.Tensor, glimpse: torch.Tensor, e_j: torch.Tensor,
                cot_j: torch.Tensor) -> torch.Tensor:
        x = torch.cat([context, glimpse, e_j, cot_j], dim=-1)
        valor = cot_j[:, COT_V] + cot_j[:, COT_P] * self.delta(x)                # (batch, K)
        logits = self.net(x)
        K = valor.shape[1]
        return torch.cat([logits[:, :K] + self.log_tau.exp() * valor, logits[:, K:]], dim=-1)


# ─────────────────────────────────────────────────────────────────────────────
# Decoder principal
# ─────────────────────────────────────────────────────────────────────────────

class AttentionDecoder(nn.Module):
    """
    Pointer decoder completo para el agente AM-Actor-Critic.

    Parámetros
    ----------
    d_h            : int   — dimensión del embedding (igual que AttentionEncoder).
    n_heads_glimpse: int   — cabezas para el glimpse (default 8).
    clip_C         : float — constante de clipping tanh (default 10.0).

    Uso en entrenamiento (PPO)
    --------------------------
    destino, opcion, log_prob, H_destino, H_precio = decoder.act_cotizacion(context, embeddings, mask, opciones, cot)

    Uso en evaluación (greedy)
    --------------------------
    destino, opcion = decoder.greedy_cotizacion(context, embeddings, mask, opciones, cot)

    Recálculo para actualización PPO
    ---------------------------------
    log_prob, H_destino, H_precio = decoder.evaluate_cotizacion(context, embeddings, mask, opciones, cot, destino, opcion)
    """

    def __init__(
        self,
        d_h:             int,
        n_heads_glimpse: int   = 8,
        clip_C:          float = 10.0,
        n_opciones:      int   = N_NIVELES + 1,
    ):
        super().__init__()
        self.glimpse = GlimpseAttention(d_h, n_heads_glimpse)
        self.pointer = PointerAttention(d_h, clip_C)
        self.precio  = PriceHead(d_h, n_opciones)
        self.n_opciones = n_opciones

    # ── Auxiliares privados ───────────────────────────────────────────────────

    def _logits_y_glimpse(self, context, embeddings, mask) -> tuple:
        """(logits enmascarados (batch, N), glimpse h_t'). mask (batch, N) bool, True = inválido. El glimpse
        lo reusa la cabeza de precio."""
        h_refined = self.glimpse(context, embeddings, mask)    # (batch, d_h)
        logits    = self.pointer(h_refined, embeddings)         # (batch, N)
        check_tensor("decoder raw logits", logits)
        logits    = logits.masked_fill(mask, float("-inf"))
        # Verify no NaN crept in at valid positions after masking (solo con CHECK_TENSORS: fuerza una sincronización)
        if CHECK_TENSORS and torch.isnan(logits[~mask]).any():
            raise RuntimeError("decoder masked logits: NaN at valid positions after masking")
        return logits, h_refined

    @staticmethod
    def _categorica(logits: torch.Tensor, tag: str) -> Categorical:
        probs = F.softmax(logits, dim=-1)
        check_tensor(tag, probs)
        return Categorical(probs=torch.clamp(probs, min=1e-8))

    def _dist_precio(self, context, glimpse, embeddings, destinos, opciones, cot) -> Categorical:
        """Distribución de la opción de precio para el destino `destinos` (batch,), con la máscara
        opciones (batch, N, n_opciones) bool (True = válida) de ese destino."""
        B = embeddings.shape[0]
        filas = torch.arange(B, device=embeddings.device)
        validas = opciones[filas, destinos]                                   # (batch, n_opciones)
        if CHECK_TENSORS and (~validas).all(dim=-1).any():
            raise RuntimeError("cabeza de precio: destino sin opciones válidas")
        logits = self.precio(context, glimpse, embeddings[filas, destinos], cot[filas, destinos])  # (batch, n_opciones)
        check_tensor("price head raw logits", logits)
        return self._categorica(logits.masked_fill(~validas, float("-inf")), "price head probs")

    @staticmethod
    def _bool_mask(action_mask: torch.Tensor) -> torch.Tensor:
        """Convierte la máscara del entorno (True/1 = válido) a bool (True = inválido)."""
        return ~action_mask.bool()

    # ── API pública: acción (destino, opción de precio) ──────────────────────

    def _dist_destino(self, context, embeddings, action_mask, tag) -> tuple:
        mask = self._bool_mask(action_mask)
        logits, glimpse = self._logits_y_glimpse(context, embeddings, mask)
        return self._categorica(logits, tag), glimpse

    def act_cotizacion(
        self,
        context:     torch.Tensor,   # (batch, d_h)
        embeddings:  torch.Tensor,   # (batch, N, d_h)
        action_mask: torch.Tensor,   # (batch, N) int8: 1 = el destino tiene alguna opción válida
        opciones:    torch.Tensor,   # (batch, N, n_opciones) bool: True = opción válida
        cot:         torch.Tensor,   # (batch, N, N_COT_FEATURES): node_feats[..., COT_COLS]
    ):
        """Samplea (destino, opción). Retorna destino, opción, log π(j, o), entropía del destino y del precio."""
        dist_j, glimpse = self._dist_destino(context, embeddings, action_mask, "decoder softmax probs")
        destino = dist_j.sample()
        dist_o = self._dist_precio(context, glimpse, embeddings, destino, opciones, cot)
        opcion = dist_o.sample()
        return destino, opcion, dist_j.log_prob(destino) + dist_o.log_prob(opcion), dist_j.entropy(), dist_o.entropy()

    def act_mixto(self, context, embeddings, action_mask, opciones, cot, n_muestra: int) -> tuple:
        """Las primeras `n_muestra` filas samplean (igual que act_cotizacion) y el resto toma el argmax (igual que
        greedy_cotizacion), con un solo pase del glimpse, el puntero y la cabeza de precio. Lo usa el rollout de
        entrenamiento para correr el muestreado y el greedy del self-critical en el mismo batch.

        Retorna destino (batch,), opción (batch,) y, solo de las n_muestra filas muestreadas, log π(j, o),
        entropía del destino y del precio."""
        m = int(n_muestra)
        logits, glimpse = self._logits_y_glimpse(context, embeddings, self._bool_mask(action_mask))
        dist_j = self._categorica(logits[:m], "decoder softmax probs")
        destino = torch.cat([dist_j.sample(), logits[m:].argmax(dim=-1)])
        filas = torch.arange(embeddings.shape[0], device=embeddings.device)
        validas = opciones[filas, destino]
        if CHECK_TENSORS and (~validas).all(dim=-1).any():
            raise RuntimeError("cabeza de precio: destino sin opciones válidas")
        logits_o = self.precio(context, glimpse, embeddings[filas, destino], cot[filas, destino])
        check_tensor("price head raw logits", logits_o)
        logits_o = logits_o.masked_fill(~validas, float("-inf"))
        dist_o = self._categorica(logits_o[:m], "price head probs")
        opcion = torch.cat([dist_o.sample(), logits_o[m:].argmax(dim=-1)])
        log_prob = dist_j.log_prob(destino[:m]) + dist_o.log_prob(opcion[:m])
        return destino, opcion, log_prob, dist_j.entropy(), dist_o.entropy()

    def greedy_cotizacion(self, context, embeddings, action_mask, opciones, cot) -> tuple:
        """Argmax del destino y, para ese destino, argmax de la opción de precio. Retorna (destino, opción)."""
        if CHECK_TENSORS and (action_mask.sum(dim=-1) == 0).any():
            raise RuntimeError("AttentionDecoder.greedy_cotizacion: zero valid actions in mask")
        logits, glimpse = self._logits_y_glimpse(context, embeddings, self._bool_mask(action_mask))
        destino = logits.argmax(dim=-1)
        return destino, self.greedy_precio(context, glimpse, embeddings, destino, opciones, cot)

    def greedy_precio(self, context, glimpse, embeddings, destino, opciones, cot) -> torch.Tensor:
        """Argmax de la opción de precio para un destino dado (lo usa también la comparación B)."""
        B = embeddings.shape[0]
        filas = torch.arange(B, device=embeddings.device)
        logits = self.precio(context, glimpse, embeddings[filas, destino], cot[filas, destino])
        return logits.masked_fill(~opciones[filas, destino], float("-inf")).argmax(dim=-1)

    def probs_precio(self, context, embeddings, action_mask, opciones, cot, destino) -> torch.Tensor:
        """π(o | s, j): probabilidades de la cabeza de precio (batch, n_opciones) para el destino `destino`
        (la usa la evaluación para el rango de precio de cada arco)."""
        _, glimpse = self._logits_y_glimpse(context, embeddings, self._bool_mask(action_mask))
        return self._dist_precio(context, glimpse, embeddings, destino, opciones, cot).probs

    def evaluate_cotizacion(self, context, embeddings, action_mask, opciones, cot, destino, opcion) -> tuple:
        """Recalcula log π(j, o) y las dos entropías para un batch almacenado (actualización PPO)."""
        dist_j, glimpse = self._dist_destino(context, embeddings, action_mask, "decoder softmax probs (eval)")
        dist_o = self._dist_precio(context, glimpse, embeddings, destino, opciones, cot)
        return dist_j.log_prob(destino) + dist_o.log_prob(opcion), dist_j.entropy(), dist_o.entropy()
