"""
critic_head.py
==============
CriticHead: estima el valor del estado V(s_t) con su PROPIO encoder.

Usado en el loop PPO de am_training.py para calcular la ventaja (GAE) y como objetivo de la pérdida de valor.

El crítico no comparte parámetros con la política: tiene su AttentionEncoder y su ContextNetwork (como los del
actor, con menos capas: config.CRITIC_N_LAYERS) y un MLP encima. Antes compartía el encoder y la pérdida de valor
empujaba la misma representación que la política; con redes separadas cada una aprende lo suyo (Andrychowicz et al., 2021, "What Matters
in On-Policy Reinforcement Learning?", ICLR: redes separadas para política y valor rinden mejor en PPO).

Entrada : el mismo dict de features que el agente (greedy_rollout.construir_features): node_feats, temporal, market,
          budget, dow, cot y current_node.
Salida  : V(s) (batch,) — escalar por elemento del batch, en las unidades del reward del entorno.

Compatibilidad
--------------
· Sin dependencias de am_agent.py, am_training.py ni routing_env.py.
"""

import torch
import torch.nn as nn

from model.encoder import AttentionEncoder, ContextNetwork, N_NODE_FEATURES


class CriticHead(nn.Module):
    """
    encoder propio → ContextNetwork propio → MLP de tres capas → V(s). En am_training el encoder tiene
    config.CRITIC_N_LAYERS capas (menos que el actor, para que el crítico propio no duplique el costo del update).

    MLP: d_h → d_h → ReLU → d_h//2 → ReLU → 1 (lineal). El output NO tiene activación final para permitir valores
    negativos (recompensas negativas son posibles en el problema de ruteo).

    Parámetros
    ----------
    d_h, n_heads, n_layers, d_ff, pre_norm, dropout : los del encoder (los del actor, salvo n_layers).
    """

    def __init__(self, d_h: int, n_heads: int = 8, n_layers: int = 3, d_ff: int = 512, pre_norm: bool = True,
                 dropout: float = 0.0):
        super().__init__()
        self.encoder = AttentionEncoder(N_NODE_FEATURES, d_h, n_heads, n_layers, d_ff, pre_norm=pre_norm,
                                        dropout=dropout)
        self.context_net = ContextNetwork(d_h, n_market=1)
        self.net = nn.Sequential(
            nn.Linear(d_h, d_h),
            nn.ReLU(),
            nn.Linear(d_h, d_h // 2),
            nn.ReLU(),
            nn.Linear(d_h // 2, 1),
        )

    def parametros_cabeza(self) -> list:
        """Los del MLP final (en am_training van con un LR 10× más alto que el encoder del crítico)."""
        return list(self.net.parameters())

    def parametros_encoder(self) -> list:
        """Encoder + ContextNetwork del crítico."""
        return list(self.encoder.parameters()) + list(self.context_net.parameters())

    def forward(self, feats: dict) -> torch.Tensor:
        """feats: dict de features del agente (ver docstring del módulo). Retorna (batch,) V(s)."""
        nf = feats["node_feats"]
        emb, graph_emb = self.encoder(nf)
        cur = emb[torch.arange(nf.shape[0], device=nf.device), feats["current_node"], :]
        h = self.context_net(graph_emb, cur, feats["temporal"], feats["market"], feats["budget"], feats["dow"],
                             feats["cot"])
        return self.net(h).squeeze(-1)
