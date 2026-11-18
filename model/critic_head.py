"""
critic_head.py
==============
CriticHead: estima el valor del estado V(s_t) desde el vector de contexto h_t.

Usado en el loop PPO de am_training.py para calcular la ventaja:
    A_t = R_t - V(s_t)

Entrada : h_t  (batch, d_h)  — vector de contexto de ContextNetwork.
Salida  : V(s) (batch,)      — escalar por elemento del batch.

Compatibilidad
--------------
· Sin dependencias de am_agent.py, am_training.py ni routing_env.py.
· d_h debe coincidir con el d_h de AttentionEncoder y ContextNetwork.
"""

import torch
import torch.nn as nn


class CriticHead(nn.Module):
    """
    MLP de tres capas sobre el vector de contexto h_t → V(s).

    Arquitectura: d_h → d_h → ReLU → d_h//2 → ReLU → 1 (lineal)

    El output NO tiene activación final para permitir valores negativos
    (recompensas negativas son posibles en el problema de ruteo).

    Parámetros
    ----------
    d_h : int — dimensión del embedding (igual que AttentionEncoder).
    """

    def __init__(self, d_h: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_h, d_h),
            nn.ReLU(),
            nn.Linear(d_h, d_h // 2),
            nn.ReLU(),
            nn.Linear(d_h // 2, 1),
        )


    def forward(self, h_t: torch.Tensor) -> torch.Tensor:
        """
        Parámetros
        ----------
        h_t : Tensor (batch, d_h)

        Retorna
        -------
        Tensor (batch,) — estimación de V(s) por elemento.
        """
        return self.net(h_t).squeeze(-1)


class PostDecisionValueHead(nn.Module):
    """
    Valor post-decisión Ṽ(s, j): retorno esperado (en unidades escaladas por 1/S) de llegar
    al nodo j desde el estado s, ANTES de ver el tablero de cargas y tarifas de j.

    Arquitectura sobre [h_t, e_j]: 2·d_h → hidden → ReLU → hidden//2 → ReLU → 1 (lineal).
    Sin activación final (permite valores negativos), sin dropout, init xavier_uniform.

    La cabeza aprende una regresión sobre el estado congelado; no perturba al PPO: el
    forward hace `h_t.detach()` y `embeddings.detach()`. Hacen falta los dos: h_t.detach()
    corta el gradiente hacia context_net y decoder; embeddings.detach() lo corta hacia el
    encoder. Con cualquiera de los dos sin detach, loss_post contaminaría los parámetros
    de la política.

    La inicialización usa un RNG propio (torch.Generator con semilla `seed`): crear la cabeza no
    consume el RNG global de torch, así que el resto del agente y el crítico se inicializan
    igual con o sin ella (mismo reward con la misma semilla). Las dos cabezas del precio de
    reserva (A y B) reciben semillas distintas, si no nacerían idénticas.

    Parámetros
    ----------
    d_h    : int — dimensión del embedding (igual que AttentionEncoder).
    hidden : int — tamaño oculto (config.POST_HIDDEN).
    seed   : int — semilla de la inicialización (default `_INIT_SEED`).
    """

    _INIT_SEED = 20261021

    def __init__(self, d_h: int, hidden: int = None, seed: int = None):
        super().__init__()
        hidden = hidden or d_h
        gen = torch.Generator().manual_seed(self._INIT_SEED if seed is None else seed)
        with torch.random.fork_rng(devices=[]):   # nn.Linear sortea su init por defecto: no tocar el RNG global
            self.net = nn.Sequential(
                nn.Linear(2 * d_h, hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden // 2),
                nn.ReLU(),
                nn.Linear(hidden // 2, 1),
            )
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, generator=gen)
                nn.init.zeros_(m.bias)

    def forward(self, h_t: torch.Tensor, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Parámetros
        ----------
        h_t        : Tensor (B, d_h)
        embeddings : Tensor (B, N, d_h)

        Retorna
        -------
        Tensor (B, N) — Ṽ(s, j) por candidato j.
        """
        h = h_t.detach().unsqueeze(1).expand(-1, embeddings.shape[1], -1)   # (B, N, d_h)
        x = torch.cat([h, embeddings.detach()], dim=-1)                     # (B, N, 2·d_h)
        return self.net(x).squeeze(-1)

    def forward_pairs(self, h_t: torch.Tensor, e_a: torch.Tensor) -> torch.Tensor:
        """Ṽ para pares (estado, acción) ya emparejados: h_t (M, d_h), e_a (M, d_h) → (M,).

        Es la entrada del entrenamiento posterior al PPO (post_training.py), que guarda
        (h_t, e_a) congelados por ejemplo. Sin detach: las entradas ya vienen sin gradiente.
        """
        return self.net(torch.cat([h_t, e_a], dim=-1)).squeeze(-1)
