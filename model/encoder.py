"""
attention_encoder.py
====================
Transformer encoder for the AM-Actor-Critic routing agent.

Componentes:
  NodeFeatureProjection  — proyecta features crudas de nodos a dimensión d_h
  MultiHeadSelfAttention — atención multi-cabeza estándar (scaled dot-product)
  TransformerEncoderLayer — MHA + FFN + LayerNorm con residual
  AttentionEncoder        — apila L capas, produce embeddings por nodo
                            y embedding global del grafo
  ContextNetwork          — combina embedding global + nodo actual + features
                            temporales → vector de contexto h_t

El vector de contexto h_t es compartido por:
  · AttentionDecoder  (decisión de ruteo)
  · CriticHead        (función de valor V(s))

Features por nodo j (desde la posición actual del camión):
  [0] reward_norm   — ganancia neta del arco actual→j (día actual), normalizada
  [1] time_norm     — tiempo de viaje actual→j, normalizado por max_duration
  [2] dist_norm     — distancia actual→j, normalizada por máximo
  [3] is_depot      — 1.0 si j es el nodo de inicio del episodio, 0.0 si no
  [4] avail_prior   — E[loads] histórico (train) del arco actual→j, normalizado por
                      LOADS_EXPECTED_CLIP y clipeado a [0, 1]. Preserva la magnitud
                      real (avail_expected_arr) en vez de colapsarla a una
                      probabilidad de existencia; es puramente descriptivo.
  [5] reward_abs    — reward crudo / REWARD_GLOBAL_P95, clipeado a [0, 1]; 0 si negativo

Features de continuación (solo con USE_CONTINUATION_FEATURES; i = nodo actual, h = casa, t = tiempo transcurrido,
est = EstadisticasLane de market_process.py: la media del generador por día de la semana, de los días de ENTRENAMIENTO).
Con t_j = llegada a j y dow_j = (dow0 + t_j // 14) % 7. Son datos del mercado, no la salida de ningún baseline: la red
combina sola estas estadísticas en una continuación.
  [6]  home_time     — T[j, h] / MAX_DURATION (tiempo de j a casa)
  [7]  slack_after   — clip((MAX − t_j − T[j,h]) / MAX, 0, 1) (holgura después de j)
  [8]  ret_margin    — clip(G[dow_j, j, h] / p95, −1, 1): margen de la lane de regreso j→casa al precio mediano
  [9]  ret_loads     — min(L[dow_j, j, h], CLIP) / CLIP: cargas esperadas de esa lane
  [10] out_best      — clip(max_{l ∈ F_j} P1·G / p95, −1, 1): valor de la mejor lane que sale de j
  [11] out_loads     — media de min(L, CLIP) / CLIP sobre las lanes salientes factibles
  [12] out_factible  — |F_j| / N: qué fracción de las lanes que salen de j se pueden usar con el tiempo que queda
  [13] m_j           — log(M[dow_j, j]) / log(M_MAX): competencia media en j ese día de la semana
  F_j = {l ∉ {i, j, h}: t_j + T[j,l] + T[l,h] ≤ MAX} (hay tiempo de ir de j a l y volver a casa). Para j == h (el ciclo
  termina) [8]-[13] valen 0, salvo la espera en casa antes de salir (j == i == h), que sí lleva las salientes [10]-[13].
  La fila j == i es la ESPERA (routing_env): su "arco" termina en fin_espera(t), el siguiente día de mercado, así que
  [1] son las horas de espera y [6-13] describen i mañana. Su reward [0] es 0.

Índice de mercado (columna INDEX_COL: [14] con continuación, [6] sin ella):
  [14] market_index — índice(j, día actual) / desviación del índice histórico (una sola para todos los
                      nodos, calculada en el ajuste). Estado del mercado de salida del nodo j neto del
                      nivel nacional, en desviaciones estándar (market_process.py).

Cotización conjunta (especificacion_fase0_ruteo_pricing.md §4.4; las ÚLTIMAS N_COT_FEATURES columnas, en cero
fuera del modo cotización). Van al final para que los checkpoints anteriores se carguen con esas columnas en
cero (transfer_learning._reconcile_state_dict copia las columnas viejas en las primeras posiciones):
  [15] dispersion   — (p_0.8 − p_0.2) / p_0.5 de la lane i→j, recortada a [0, 1]
  [16] break_even   — costo / (dist · p_0.5): qué tan cerca está el mínimo rentable de la mediana, en [0, 2] / 2
  [17] intentos     — cargas de la lane i→j que se pueden cotizar todavía en esta visita / COTIZACIONES_MAX_POR_NODO
                      (> 0 si y solo si se puede cotizar)
  [18 … 17+K]       — v_k = P_k·(p_k·d − c) / p95: valor esperado de cotizar cada nivel, recortado a [−1, 1]
  [18+K … 17+2K]    — P_k = (1 − u_k)^m: probabilidad de ganar con cada nivel
  [18+2K]           — valor de la lane con todos sus intentos (pricing.valor_reintentos) si perderlos vale 0, / p95
  [19+2K]           — lo mismo si perderlos es recorrer la lane vacía (la vuelta): base −c
  Las de [18] en adelante son 0 si la lane no se puede cotizar. Las columnas de cotización entran también a la
  cabeza de precio (model/decoder.PriceHead), que así ve el valor de cada nivel y no solo el embedding.
  Las features [0] y [5] ya son, en ese modo, el margen si se cobrara la mediana (env.reward_rows).

Condicionamiento en presupuesto de cardinalidad K:
  ContextNetwork recibe, además del bloque temporal y de mercado, un bloque
  separado de budget features (ver build_budget_features_batch): k_remaining
  normalizado por K_NORM (constante global) y un flag k_restricted. No forma
  parte de N_NODE_FEATURES (no es una feature por nodo).

Las funciones build_*_batch arman las entradas con tensores de torch en el dispositivo del entorno (la GPU en el
servidor), sin loops sobre B ni N: el estado del entorno nunca pasa por la CPU.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import (
    LTR_CLIP, LOADS_EXPECTED_CLIP, K_NORM, USE_CONTINUATION_FEATURES, COMPETENCIA_M_MAX, COTIZACIONES_MAX_POR_NODO,
    CHECK_TENSORS,
)
from pricing import N_NIVELES, U_NIVELES_ARR, indice_nivel, prob_ganar_t, valor_reintentos_t
from routing_env import fin_espera

# Número de features por nodo — debe coincidir con build_node_features_batch()
N_BASE_NODE_FEATURES  = 6
N_CONT_FEATURES       = 8   # home_time, slack_after + 6 estadísticas de las lanes que salen de j
N_INDEX_FEATURES      = 1   # índice de mercado (columna INDEX_COL)
N_COT_FEATURES        = 3 + 2 * N_NIVELES + 2   # cotización conjunta: las últimas columnas
COT_V = slice(3, 3 + N_NIVELES)                              # dentro de las de cotización: v_k
COT_P = slice(3 + N_NIVELES, 3 + 2 * N_NIVELES)              # … y P_k
INDEX_COL = N_BASE_NODE_FEATURES + (N_CONT_FEATURES if USE_CONTINUATION_FEATURES else 0)
COT_COLS  = slice(INDEX_COL + N_INDEX_FEATURES, INDEX_COL + N_INDEX_FEATURES + N_COT_FEATURES)
N_NODE_FEATURES = INDEX_COL + N_INDEX_FEATURES + N_COT_FEATURES

# Features de contexto de la cotización (último bloque de ContextNetwork): log(m) / log(M_MAX) del nodo actual y
# rechazos en esta visita / COTIZACIONES_MAX_POR_NODO. En cero fuera del modo cotización.
N_COT_CONTEXT = 2
_U_20, _U_50, _U_80 = indice_nivel(0.2), indice_nivel(0.5), indice_nivel(0.8)

# Día de la semana (one-hot, lunes = 0) del día en que está el camión: el proceso de cargas
# depende de él (factor s_d de 0,58 a 1,43), así que el agente lo necesita para estimar el
# valor de los nodos siguientes. Va como ÚLTIMO bloque de la entrada de ContextNetwork para que
# los checkpoints anteriores se carguen con esas columnas en cero (transfer_learning).
N_DOW_FEATURES = 7


# ─────────────────────────────────────────────────────────────────────────────
# Features de continuación
# ─────────────────────────────────────────────────────────────────────────────

def build_continuation_features_batch(
    current_node:      torch.Tensor,   # (B,) long — i
    start_node:        torch.Tensor,   # (B,) long — casa h
    time_elapsed:      torch.Tensor,   # (B,) horas transcurridas t
    dow0:              torch.Tensor,   # (B,) día de la semana (lunes = 0) del día de salida
    T:                 torch.Tensor,   # (N, N) f64 horas de viaje
    est,                               # EstadisticasLane en el dispositivo (EstadisticasLane.a(device))
    max_duration:      float,
    reward_global_p95: float,
) -> torch.Tensor:                     # (B, N, N_CONT_FEATURES) float32
    """Features de continuación por nodo j (ver docstring del módulo), sin loops sobre B ni N.

    Las 6 últimas son estadísticas (por día de la semana de llegada a j) de las lanes que SALEN de j: la red no ve
    esas lanes en ninguna otra entrada, porque sus features describen solo las lanes que salen del nodo actual i.
    """
    N    = T.shape[0]
    i, h = current_node, start_node
    t    = time_elapsed.double()
    MAX  = max(float(max_duration), 1e-6)
    p95  = max(float(reward_global_p95), 1e-6)
    jj   = torch.arange(N, device=T.device)

    is_h = jj[None, :] == h[:, None]                                         # (B, N)
    is_i = jj[None, :] == i[:, None]                                         # (B, N)
    # j = i es la espera: el "arco" i→i termina al siguiente día de mercado (fin_espera).
    t_j  = torch.where(is_i, fin_espera(t)[:, None], t[:, None] + T[i, :])  # (B, N) llegada a j
    T_jh = torch.where(is_h, torch.zeros_like(t_j), T[:, h].T)              # (B, N) tiempo de j a casa

    home_time = T_jh / MAX
    slack     = ((MAX - t_j - T_jh) / MAX).clamp(0.0, 1.0)

    dow_j = (dow0[:, None] + torch.div(t_j, 14.0, rounding_mode="floor").long()) % 7   # (B, N) día de la semana en j
    jx, hx = jj[None, :], h[:, None]

    # Lane de regreso j → casa.
    ret_margin = (est.G[dow_j, jx, hx] / p95).clamp(-1.0, 1.0)
    ret_loads  = est.L[dow_j, jx, hx].clamp(max=LOADS_EXPECTED_CLIP) / LOADS_EXPECTED_CLIP

    # Lanes salientes j → l con tiempo de llegar a l y volver a casa: l ∉ {i, j, h}.
    t_l  = t_j[:, :, None] + T[None, :, :]                                   # (B, N, N) llegada a l desde j
    T_lh = T[:, h].T[:, None, :]                                             # (B, 1, N) T[l, h]
    lx3  = jj[None, None, :]
    F    = ((lx3 != jj[None, :, None]) & (lx3 != i[:, None, None]) & (lx3 != h[:, None, None])
            & (t_l + T_lh <= max_duration))                                  # (B, N, N)
    dj, jx3 = dow_j[:, :, None], jj[None, :, None]
    n_fact    = F.sum(dim=2)                                                 # (B, N)
    hay       = n_fact > 0
    valor     = torch.where(F, est.P1[dj, jx3, lx3] * est.G[dj, jx3, lx3],
                            torch.full((), float("-inf"), device=T.device)).amax(dim=2)
    out_best  = torch.where(hay, (valor / p95).clamp(-1.0, 1.0), torch.zeros_like(valor))
    cargas    = est.L[dj, jx3, lx3].clamp(max=LOADS_EXPECTED_CLIP) / LOADS_EXPECTED_CLIP
    out_loads = torch.where(hay, torch.where(F, cargas, torch.zeros_like(cargas)).sum(dim=2) / n_fact.clamp_min(1),
                            torch.zeros_like(valor))
    out_fact  = n_fact.float() / max(N, 1)
    m_j = (torch.log(est.M[dow_j, jx].clamp_min(1.0)) / np.log(max(COMPETENCIA_M_MAX, 1.0 + 1e-9))).clamp(0.0, 1.0)

    # j == h es volver a casa: la lane de regreso no existe y el ciclo termina (sin salientes). La espera en casa antes de
    # salir (j == i == h) sí conserva las salientes de mañana. La holgura [7] se conserva en ambos casos.
    termina = is_h & ~is_i
    def cero(x, msk):
        return torch.where(msk, torch.zeros_like(x), x)
    ret_margin, ret_loads = cero(ret_margin, is_h), cero(ret_loads, is_h)
    out_best, out_loads, out_fact, m_j = (cero(x, termina) for x in (out_best, out_loads, out_fact, m_j))

    return torch.stack([home_time.float(), slack.float(), ret_margin, ret_loads, out_best, out_loads, out_fact, m_j],
                       dim=-1).float()


# ─────────────────────────────────────────────────────────────────────────────
# Bloques del Transformer Encoder
# ─────────────────────────────────────────────────────────────────────────────

class NodeFeatureProjection(nn.Module):
    """
    Proyección lineal de las features crudas de nodos a la dimensión de embedding.

    Entrada : (batch, N, d_input)
    Salida  : (batch, N, d_h)
    """

    def __init__(self, d_input: int, d_h: int):
        super().__init__()
        self.proj = nn.Linear(d_input, d_h)
        self.norm = nn.LayerNorm(d_h)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(F.relu(self.proj(x)))


class MultiHeadSelfAttention(nn.Module):
    """
    Atención multi-cabeza estándar (scaled dot-product).

    Entrada : (batch, N, d_h)
    Salida  : (batch, N, d_h)
    """

    def __init__(self, d_h: int, n_heads: int):
        super().__init__()
        assert d_h % n_heads == 0, "d_h debe ser divisible por n_heads"
        self.n_heads = n_heads
        self.d_k = d_h // n_heads

        self.W_q = nn.Linear(d_h, d_h, bias=False)
        self.W_k = nn.Linear(d_h, d_h, bias=False)
        self.W_v = nn.Linear(d_h, d_h, bias=False)
        self.W_o = nn.Linear(d_h, d_h, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, _ = x.shape

        def split_heads(t):
            return t.view(B, N, self.n_heads, self.d_k).transpose(1, 2)

        Q = split_heads(self.W_q(x))   # (B, H, N, d_k)
        K = split_heads(self.W_k(x))
        V = split_heads(self.W_v(x))

        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_k ** 0.5)
        attn   = F.softmax(scores, dim=-1)

        out = torch.matmul(attn, V)                         # (B, H, N, d_k)
        out = out.transpose(1, 2).contiguous().view(B, N, -1)  # (B, N, d_h)
        return self.W_o(out)


class TransformerEncoderLayer(nn.Module):
    """
    Una capa del encoder Transformer: MHA + FFN con residual y LayerNorm.

    Entrada : (batch, N, d_h)
    Salida  : (batch, N, d_h)
    """

    def __init__(self, d_h: int, n_heads: int, d_ff: int, pre_norm: bool = True, dropout: float = 0.0):
        super().__init__()
        self.pre_norm = pre_norm
        self.mha   = MultiHeadSelfAttention(d_h, n_heads)
        self.norm1 = nn.LayerNorm(d_h)
        self.ffn   = nn.Sequential(
            nn.Linear(d_h, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_h),
        )
        self.norm2 = nn.LayerNorm(d_h)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.pre_norm:
            x = x + self.dropout(self.mha(self.norm1(x)))
            x = x + self.dropout(self.ffn(self.norm2(x)))
        else:
            x = self.norm1(x + self.mha(x))
            x = self.norm2(x + self.ffn(x))
        return x


class AttentionEncoder(nn.Module):
    """
    Encoder Transformer: mapea N vectores de features a N embeddings ricos.

    Parámetros
    ----------
    d_input  : int — features por nodo (N_NODE_FEATURES: 8, o 12 con USE_CONTINUATION_FEATURES).
    d_h      : int — dimensión del embedding (ej. 128).
    n_heads  : int — cabezas de atención (ej. 8, debe dividir d_h).
    n_layers : int — capas del encoder (ej. 3).
    d_ff     : int — dimensión interna del FFN (ej. 512 = 4 × d_h).

    Forward
    -------
    node_features : Tensor (batch, N, d_input)

    Retorna
    -------
    embeddings      : Tensor (batch, N, d_h)  — embedding por nodo.
    graph_embedding : Tensor (batch, d_h)     — promedio de todos los embeddings.
    """

    def __init__(
        self,
        d_input:  int = N_NODE_FEATURES,
        d_h:      int = 128,
        n_heads:  int = 8,
        n_layers: int = 3,
        d_ff:     int = 512,
        pre_norm: bool = True,
        dropout:  float = 0.0,
    ):
        super().__init__()
        self.pre_norm = pre_norm
        self.input_proj = NodeFeatureProjection(d_input, d_h)
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_h, n_heads, d_ff, pre_norm=pre_norm, dropout=dropout)
            for _ in range(n_layers)
        ])
        self.norm_out = nn.LayerNorm(d_h)

    def forward(self, node_features: torch.Tensor):
        x = self.input_proj(node_features)   # (batch, N, d_h)
        for layer in self.layers:
            x = layer(x)                     # (batch, N, d_h)
        if self.pre_norm:
            x = self.norm_out(x)
        graph_emb = x.mean(dim=1)            # (batch, d_h)
        return x, graph_emb


# ─────────────────────────────────────────────────────────────────────────────
# Vector de contexto h_t
# ─────────────────────────────────────────────────────────────────────────────

class ContextNetwork(nn.Module):
    """
    Construye el vector de contexto h_t usado por el decoder (destino y precio) y el critic.

    Combina:
      · graph_embedding   (d_h)       — estado global del grafo
      · current_node_emb  (d_h)       — posición actual del camión
      · temporal_features (3)         — tiempo_norm, pasos_restantes_norm,
                                        progreso_norm
      · market_features   (n_market)  — MTI, dispersión, forecast (opcional,
                                        por defecto 0 = no se usa)
      · dow_features      (n_dow)     — día de la semana actual (one-hot), último bloque
      · budget_features   (n_budget)  — presupuesto de cardinalidad K:
                                        k_remaining_norm, k_restricted.
                                        Bloque SEPARADO de temporal_features
                                        (conceptualmente distinto: K acota el
                                        número de arcos, no el tiempo) — no se
                                        amplía n_temporal para incorporarlo.

      · cot_features      (n_cot)     — cotización conjunta: competencia del nodo actual y rechazos en esta
                                        visita (build_cotizacion_context_batch). Último bloque, en cero fuera
                                        de ese modo, para que los checkpoints anteriores carguen.

    Entrada total: 2·d_h + 3 + n_market + n_budget + n_dow + n_cot  →  MLP  →  d_h

    Parámetros
    ----------
    d_h      : int — dimensión del embedding (debe coincidir con AttentionEncoder).
    n_market : int — número de features de mercado adicionales (default 1).
    n_budget : int — número de features de presupuesto K (default 2, ver
               build_budget_features_batch()).
    n_dow    : int — one-hot del día de la semana (default N_DOW_FEATURES = 7).
    n_cot    : int — features de contexto de la cotización (default N_COT_CONTEXT = 2).
    """

    def __init__(self, d_h: int, n_market: int = 1, n_budget: int = 2, n_dow: int = N_DOW_FEATURES,
                 n_cot: int = N_COT_CONTEXT):
        super().__init__()
        self.n_dow = n_dow
        self.n_cot = n_cot
        n_temporal = 3          # time_norm, remaining_steps_norm, progress_norm
        d_in = 2 * d_h + n_temporal + n_market + n_budget + n_dow + n_cot
        self.proj = nn.Sequential(
            nn.Linear(d_in, d_h),
            nn.ReLU(),
            nn.Linear(d_h, d_h),
            nn.LayerNorm(d_h),
        )

    def forward(
        self,
        graph_emb:        torch.Tensor,   # (batch, d_h)
        current_node_emb: torch.Tensor,   # (batch, d_h)
        temporal_feats:   torch.Tensor,   # (batch, 3)
        market_feats:     torch.Tensor = None,  # (batch, n_market) | None
        budget_feats:     torch.Tensor = None,  # (batch, n_budget) | None
        dow_feats:        torch.Tensor = None,  # (batch, n_dow) | None (None ⇒ ceros)
        cot_feats:        torch.Tensor = None,  # (batch, n_cot) | None (None ⇒ ceros)
    ) -> torch.Tensor:
        """
        Retorna el vector de contexto h_t de forma (batch, d_h).
        """
        parts = [graph_emb, current_node_emb, temporal_feats]
        if market_feats is not None:
            parts.append(market_feats)
        if budget_feats is not None:
            parts.append(budget_feats)
        if self.n_dow:
            parts.append(dow_feats if dow_feats is not None
                         else torch.zeros(graph_emb.shape[0], self.n_dow, device=graph_emb.device))
        if self.n_cot:
            parts.append(cot_feats if cot_feats is not None
                         else torch.zeros(graph_emb.shape[0], self.n_cot, device=graph_emb.device))
        h = torch.cat(parts, dim=-1)
        return self.proj(h)



# ─────────────────────────────────────────────────────────────────────────────
# Features del agente desde el estado del entorno (tensores en el dispositivo del entorno)
# ─────────────────────────────────────────────────────────────────────────────

def build_budget_features_batch(
    k_remaining:  torch.Tensor,   # (B,) long — indefinido donde k_restricted es False
    k_restricted: torch.Tensor,   # (B,) bool
    k_norm:       float = K_NORM,
) -> torch.Tensor:                # (B, 2) float32
    """k_remaining_norm (1.0 si caso libre) y flag k_restricted, para ContextNetwork."""
    resto = torch.where(k_restricted, k_remaining.float() / float(k_norm),
                        torch.ones_like(k_remaining, dtype=torch.float32))
    return torch.stack([resto, k_restricted.float()], dim=-1)


def build_node_features_batch(
    current_node:      torch.Tensor,   # (B,) long
    start_node:        torch.Tensor,   # (B,) long
    T:                 torch.Tensor,   # (N, N) f64 horas
    dist:              torch.Tensor,   # (N, N) f64 millas
    max_duration:      float,
    reward_global_p95: float,
    avail_expected:    torch.Tensor,   # (N, N) f32 — E[loads] histórico (train), feature [4]; None ⇒ 0
    reward_rows:       torch.Tensor,   # (B, N) — env.reward_rows: margen de las lanes salientes a la mediana del mercado
    est,                               # EstadisticasLane en el dispositivo — features de continuación [6-13]
    time_elapsed:      torch.Tensor,   # (B,) horas transcurridas
    dow0:              torch.Tensor,   # (B,) día de la semana (lunes = 0) de salida
    market_index:      torch.Tensor,   # (B, N) índice de mercado de todos los nodos en el día actual (env.market_index)
    index_std:         float,          # desviación del índice histórico (env.index_std)
    cotizacion_feats:  torch.Tensor = None,   # (B, N, N_COT_FEATURES); None ⇒ esas columnas en cero (tests)
) -> torch.Tensor:                     # (B, N, N_NODE_FEATURES) float32
    """Matriz (N, N_NODE_FEATURES) de cada uno de los B episodios (ver el docstring del módulo), sin loops."""
    B, N = reward_rows.shape
    dev = reward_rows.device
    if USE_CONTINUATION_FEATURES and est is None:
        raise ValueError("USE_CONTINUATION_FEATURES requiere est (MarketProcess.estadisticas_dow(...).a(device))")
    filas = torch.arange(B, device=dev)

    # f0 (margen a la mediana) y f5 (el mismo en escala absoluta), recortados para evitar desbordes.
    raw = reward_rows.float().clamp(-1e4, 1e4)

    def _minmax(x):   # normalización por fila: preserva el ranking local entre las alternativas desde i
        lo, hi = x.amin(dim=1, keepdim=True), x.amax(dim=1, keepdim=True)
        rng = hi - lo
        return torch.where(rng > 0, (x - lo) / torch.where(rng > 0, rng, torch.ones_like(rng)), torch.zeros_like(x))

    f0 = _minmax(raw)
    f2 = _minmax(dist[current_node].float())
    # Fila propia (j = i) = la espera: f1 son las horas hasta el siguiente día de mercado y f2 = 0 (la diagonal de
    # distancias trae una distancia intra-hub, pero esperar no recorre nada).
    f2[filas, current_node] = 0.0
    f1 = T[current_node].clone()
    f1[filas, current_node] = fin_espera(time_elapsed) - time_elapsed.double()
    f1 = (f1 / max(max_duration, 1e-6)).clamp(0.0, 1.0).float()
    f3 = (torch.arange(N, device=dev)[None, :] == start_node[:, None]).float()
    f4 = (avail_expected[current_node].float().clamp(max=LOADS_EXPECTED_CLIP) / LOADS_EXPECTED_CLIP
          if avail_expected is not None else torch.zeros_like(raw))
    f5 = (raw / max(reward_global_p95, 1e-6)).clamp(0.0, 1.0)

    partes = [torch.stack([f0, f1, f2, f3, f4, f5], dim=-1)]
    if USE_CONTINUATION_FEATURES:
        partes.append(build_continuation_features_batch(current_node, start_node, time_elapsed, dow0, T, est,
                                                        max_duration, reward_global_p95))
    partes.append((market_index.double() / max(float(index_std), 1e-12)).float()[..., None])
    partes.append(cotizacion_feats if cotizacion_feats is not None
                  else torch.zeros(B, N, N_COT_FEATURES, dtype=torch.float32, device=dev))
    feats = torch.cat(partes, dim=-1)
    if CHECK_TENSORS and not torch.isfinite(feats).all():
        raise RuntimeError("features de nodo con NaN/Inf")
    return feats


def build_cotizacion_node_features_batch(
    current_node: torch.Tensor,   # (B,) long
    precios:      torch.Tensor,   # (B, N, K) USD/milla de cada nivel (env.precios_row)
    opciones:     torch.Tensor,   # (B, N, K+1) bool (env.opciones_validas)
    dist_rows:    torch.Tensor,   # (B, N) distancia i→j
    cost_rows:    torch.Tensor,   # (B, N) costo i→j
    m:            torch.Tensor,   # (B,) competidores efectivos del nodo actual (env.competencia)
    intentos:     torch.Tensor,   # (B, N) cargas cotizables que quedan por lane (env.intentos)
    reward_global_p95: float,
) -> torch.Tensor:                # (B, N, N_COT_FEATURES) float32
    """Features de cotización por lane i→j (ver docstring del módulo): dispersión del precio, break-even
    relativo a la mediana, si se puede cotizar y, por nivel, el valor esperado de cotizarlo y P(ganar). El
    self-loop (j = i) queda en cero."""
    B, N = precios.shape[:2]
    dev = precios.device
    precios = precios.double()
    p_med = precios[..., _U_50]
    ok = p_med > 1e-9
    seguro = torch.where(ok, p_med, torch.ones_like(p_med))
    d, c = dist_rows.double(), cost_rows.double()
    dispersion = torch.where(ok, (precios[..., _U_80] - precios[..., _U_20]) / seguro, torch.zeros_like(p_med))
    break_even = torch.where(ok, c / (d.clamp_min(1e-9) * seguro), torch.full_like(p_med, 2.0))
    cotizable = opciones[..., :N_NIVELES].any(dim=-1)                                     # (B, N)
    u_niv = torch.as_tensor(U_NIVELES_ARR, dtype=torch.float64, device=dev)
    P = prob_ganar_t(u_niv[None, :], m.double()[:, None])[:, None, :]                     # (B, 1, K)
    p95 = max(float(reward_global_p95), 1e-6)
    margen = precios * d[..., None] - c[..., None]                                        # (B, N, K)
    v = P * margen / p95
    A = COTIZACIONES_MAX_POR_NODO
    n = intentos.long().clamp(0, A)                                                       # (B, N)

    def _con_intentos(base):   # valor de la lane con sus intentos (pricing.valor_reintentos), / p95
        V = valor_reintentos_t(P, margen, base, A)                                        # (A+1, B, N)
        return torch.gather(V, 0, n[None]).squeeze(0)[..., None] / p95                    # (B, N, 1)

    abierto, cierre = _con_intentos(torch.zeros_like(c)), _con_intentos(-c)
    por_nivel = torch.cat([v.clamp(-1.0, 1.0), P.expand_as(v), abierto.clamp(-1.0, 1.0), cierre.clamp(-1.0, 1.0)],
                          dim=-1)
    out = torch.cat([
        torch.stack([dispersion.clamp(0.0, 1.0), break_even.clamp(0.0, 2.0) / 2.0,
                     torch.where(cotizable, n.double() / A, torch.zeros_like(p_med))], dim=-1),
        torch.where(cotizable[..., None], por_nivel, torch.zeros_like(por_nivel)),
    ], dim=-1).float()
    out[torch.arange(B, device=dev), current_node] = 0.0
    return out


def build_cotizacion_context_batch(m: torch.Tensor, n_rechazos: torch.Tensor) -> torch.Tensor:   # (B,), (B,) → (B, 2)
    """log(m) / log(M_MAX) (competencia en el nodo actual, en [0, 1]) y rechazos / COTIZACIONES_MAX_POR_NODO."""
    comp = (torch.log(m.double().clamp_min(1.0)) / np.log(max(COMPETENCIA_M_MAX, 1.0 + 1e-9))).clamp(0.0, 1.0)
    return torch.stack([comp, n_rechazos.double() / max(COTIZACIONES_MAX_POR_NODO, 1)], dim=-1).float()


def build_dow_features_batch(dow: torch.Tensor) -> torch.Tensor:   # (B,) long → (B, 7) float32
    """One-hot del día de la semana (lunes = 0) de cada episodio."""
    return F.one_hot(dow.long() % N_DOW_FEATURES, N_DOW_FEATURES).float()


def build_temporal_features_batch(
    time_elapsed: torch.Tensor,   # (B,) float32
    step_count:   torch.Tensor,   # (B,) long
    max_duration: float,
    num_nodes:    int,
) -> torch.Tensor:                # (B, 3) float32
    """Features temporales (elapsed/remaining/progreso) para ContextNetwork, batcheadas."""
    denom = max(max_duration, 1e-6)
    t = time_elapsed.float()
    return torch.stack([t.clamp(max=max_duration) / denom, (max_duration - t).clamp_min(0.0) / denom,
                        step_count.float() / max(num_nodes, 1)], dim=-1)


def build_market_features_batch(ltr: torch.Tensor) -> torch.Tensor:   # (B,) → (B, 1) float32
    """LTR del hub actual en el día actual (env.ltr_actual), normalizado a [0, 1], para ContextNetwork."""
    return (ltr.float().clamp(0.0, LTR_CLIP) / LTR_CLIP)[:, None]
