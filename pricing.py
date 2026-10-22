"""
pricing.py
==========
Precio de reserva p*(i→j, s) en USD/milla (PRICING_PLAN.md §2): el costo de oportunidad
por milla de ir a j, es decir, la tarifa a la que al camión le da igual aceptar la lane i→j
o su mejor alternativa. Es una recomendación del lado del transportista, NO una probabilidad
de aceptación del cliente.

    Q(s, a)   = r_obs(i→a)/S + Ṽ(s, a)                 valor de aceptar la lane i→a
    Q_alt(s)  = max_{a ≠ j válida} Q(s, a)               mejor alternativa a j
    p*(i→j)   = ( S · (Q_alt − Ṽ(s, j)) + costo(i→j) ) / dist(i→j)

con r_obs(i→a) = lane_reward sin redondear (ingreso solo si hay carga − costo), S =
REWARD_SCALE_FACTOR y Ṽ(s, a) la cabeza post-decisión del agente (agent.post_value), con
Ṽ(s, depot) = 0. Casos límite: si j es la única acción factible, p* no está definido y se
devuelve el piso costo/dist con only_option=True.
"""

import numpy as np
import torch

from config import MPG, MARGINAL_COST_SIN_DIESEL


def lane_cost_row(distance_row: np.ndarray, diesel_row: np.ndarray) -> np.ndarray:
    """(N,) costo en USD de recorrer cada lane saliente; misma fórmula que problem_data.lane_reward."""
    return distance_row * (diesel_row / MPG) + distance_row * MARGINAL_COST_SIN_DIESEL


def percentil_historico(valor: float, serie: np.ndarray) -> float:
    """Percentil (0-100) de `valor` dentro de `serie` (fracción de la serie menor o igual al valor)."""
    serie = np.asarray(serie, dtype=float)
    serie = serie[np.isfinite(serie)]
    return float("nan") if serie.size == 0 else 100.0 * float(np.mean(serie <= valor))


@torch.no_grad()
def reservation_price(
    agent,
    h_t:           torch.Tensor,    # (1, d_h)
    embeddings:    torch.Tensor,    # (1, N, d_h)
    action_mask:   np.ndarray,      # (N,) 1 = acción válida desde el nodo actual
    current_node:  int,
    j:             int,
    rate_row_obs:  np.ndarray,      # (N,) tarifa USD/milla observada al llegar a current_node
    distance_row:  np.ndarray,      # (N,) distancia current_node → a
    cost_row:      np.ndarray,      # (N,) costo de current_node → a (lane_cost_row)
    start_node:    int,
    S:             float,           # REWARD_SCALE_FACTOR
    loads_row:     np.ndarray = None,   # (N,) cargas observadas; ingreso solo donde > 0 (None ⇒ todas con carga)
    post_values:   np.ndarray = None,   # (N,) Ṽ ya calculado (evita recomputar al recorrer varias j)
) -> tuple:
    """Precio de reserva de la lane current_node→j. Retorna (p_estrella: float, only_option: bool).

    El ingreso de cada alternativa sigue el entorno (lane_reward): tarifa × distancia solo si hay
    carga en esa lane; si no, la acción cuesta el costo sin ingreso. El retorno al depot (start_node)
    es siempre una alternativa válida de la máscara y tiene Ṽ = 0.
    """
    if post_values is None:
        post_values = agent.post_value(h_t, embeddings).squeeze(0).cpu().numpy()
    v = np.array(post_values, dtype=np.float64)
    v[start_node] = 0.0                                           # ancla del retorno al depot

    break_even = float(cost_row[j] / distance_row[j])
    valid = np.asarray(action_mask).astype(bool).copy()
    valid[current_node] = False
    valid[j] = False
    if not valid.any():
        return break_even, True

    revenue = np.asarray(rate_row_obs, dtype=np.float64) * np.asarray(distance_row, dtype=np.float64)
    if loads_row is not None:
        revenue = np.where(np.asarray(loads_row) > 0, revenue, 0.0)
    q = (revenue - cost_row) / S + v                              # Q(s, a) para toda a
    q_alt = float(q[valid].max())
    p_star = (S * (q_alt - v[j]) + cost_row[j]) / distance_row[j]
    return float(p_star), False
