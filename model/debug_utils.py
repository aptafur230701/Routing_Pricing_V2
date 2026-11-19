"""
debug_utils.py
==============
Utilidades de debugging numérico para el pipeline PPO + Attention Model.
Reemplaza torch.nan_to_num() como mecanismo silencioso de corrección
por fallos inmediatos con mensajes claros.
"""

import torch

from config import CHECK_TENSORS


def check_tensor(name: str, x: torch.Tensor) -> None:
    """
    Verifica que un tensor no contenga NaN ni Inf.
    Falla inmediatamente con un mensaje que identifica la etapa exacta.

    No-op si config.CHECK_TENSORS es False (default): cada check fuerza una
    sincronización CPU-GPU, y en PPO con batches chicos eso pesa (ver el
    comentario de CHECK_TENSORS en config.py). Activar con CHECK_TENSORS=1
    para diagnosticar un NaN/Inf.

    Parámetros
    ----------
    name : str          — nombre descriptivo de la etapa (p.ej. "embeddings").
    x    : torch.Tensor — tensor a verificar.

    Raises
    ------
    RuntimeError si x contiene NaN o Inf.
    """
    if not CHECK_TENSORS:
        return
    if torch.isnan(x).any():
        raise RuntimeError(
            f"[NaN] {name} — shape {tuple(x.shape)}, "
            f"dtype {x.dtype}, device {x.device}"
        )
    if torch.isinf(x).any():
        raise RuntimeError(
            f"[Inf] {name} — shape {tuple(x.shape)}, "
            f"dtype {x.dtype}, device {x.device}"
        )
