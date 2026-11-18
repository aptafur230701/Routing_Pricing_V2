"""
am_agent.py
===========
Agente AM-Actor-Critic mínimo para ruteo.

Combina AttentionEncoder + ContextNetwork + AttentionDecoder en un único
nn.Module. No construye rutas por sí mismo: el rollout (muestreado en
entrenamiento, argmax en evaluación) se ejecuta paso a paso sobre el entorno
(ver am_training.py y greedy_rollout.py).
"""

import torch
import torch.nn as nn

from model.encoder import (
    AttentionEncoder,
    ContextNetwork,
    N_NODE_FEATURES,
)
from model.decoder import AttentionDecoder
from model.critic_head import PostDecisionValueHead
from config import USE_POST_HEAD, POST_HIDDEN, POST_SEED_A, POST_SEED_B


class AMRoutingAgent(nn.Module):
    """
    Agente de ruteo basado en Attention Model.

    Parámetros
    ----------
    num_nodes : int   — tamaño del grafo (escalable).
    d_h       : int   — dimensión del embedding (default 128).
    n_heads   : int   — cabezas de atención (default 8, debe dividir d_h).
    n_layers  : int   — capas del encoder Transformer (default 3).
    d_ff      : int   — dimensión interna FFN (default 512).
    clip_C    : float — clipping tanh del pointer (default 10.0).
    device    : torch.device
    pre_norm  : bool  — pre-norm (True) vs post-norm (False) en el encoder (default True).
    dropout   : float — dropout en las sublayers del encoder (default 0.0).
    use_post_head : bool — crea las cabezas `post_value_a` y `post_value_b` (PostDecisionValueHead, estimador doble del
                    valor post-decisión); None si es False (default config.USE_POST_HEAD). El PPO las crea apagadas:
                    se entrenan después, con el actor congelado (post_training.py).
    """

    def __init__(
        self,
        num_nodes: int,
        d_h:       int   = 128,
        n_heads:   int   = 8,
        n_layers:  int   = 3,
        d_ff:      int   = 512,
        clip_C:    float = 10.0,
        device:    torch.device = None,
        pre_norm:  bool  = True,
        dropout:   float = 0.0,
        use_post_head: bool = USE_POST_HEAD,
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.d_h       = d_h
        self.device    = device or torch.device("cpu")

        self.encoder     = AttentionEncoder(
            N_NODE_FEATURES, d_h, n_heads, n_layers, d_ff,
            pre_norm=pre_norm, dropout=dropout,
        )
        self.context_net = ContextNetwork(d_h, n_market=1)
        self.decoder     = AttentionDecoder(d_h, n_heads_glimpse=n_heads, clip_C=clip_C)
        # Valor post-decisión Ṽ(s, j) para el precio de reserva (pricing.py): dos cabezas A y B de la misma
        # arquitectura, con inicializaciones distintas, entrenadas después del PPO sobre estados del actor
        # congelado (post_training.py). Ninguna participa en el PPO ni en la política (detach interno).
        self.post_value_a = PostDecisionValueHead(d_h, POST_HIDDEN, seed=POST_SEED_A) if use_post_head else None
        self.post_value_b = PostDecisionValueHead(d_h, POST_HIDDEN, seed=POST_SEED_B) if use_post_head else None

        self.post_entrenadas = False   # True cuando las cabezas Ṽ ya se entrenaron (post_training.py) o se cargaron entrenadas

        self.to(self.device)

    @property
    def has_post_heads(self) -> bool:
        return self.post_value_a is not None and self.post_value_b is not None

    def post_values(self, h_t: torch.Tensor, embeddings: torch.Tensor) -> tuple:
        """(Ṽ_A, Ṽ_B), cada una (B, N) en unidades escaladas por 1/S. El ancla Ṽ(s, depot) = 0 la impone el llamador."""
        return self.post_value_a(h_t, embeddings), self.post_value_b(h_t, embeddings)
