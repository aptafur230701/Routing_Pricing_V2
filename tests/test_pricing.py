"""
test_pricing.py
===============
Fase P2 de PRICING_PLAN.md: reservation_price.
"""

import numpy as np
import pytest
import torch

from config import REWARD_SCALE_FACTOR as S
from pricing import lane_cost_row, percentil_historico, reservation_price
from problem_data import lane_reward

N = 6
START, CUR = 0, 2


@pytest.fixture
def lane():
    rng = np.random.default_rng(0)
    dist = rng.uniform(200, 900, N)
    diesel = rng.uniform(3.5, 4.5, N)
    return dict(
        mask=np.array([1, 1, 0, 1, 1, 1], dtype=np.int8),          # nodo actual (2) fuera; depot (0) válido
        rate=rng.uniform(2.0, 3.0, N), loads=np.array([5, 3, 0, 2, 9, 1]),
        dist=dist, cost=lane_cost_row(dist, diesel), diesel=diesel,
    )


def price(lane, j, v, mask=None):
    return reservation_price(
        None, None, None, lane["mask"] if mask is None else mask, CUR, j,
        lane["rate"], lane["dist"], lane["cost"], START, S, loads_row=lane["loads"],
        post_values=np.asarray(v, dtype=float),
    )


def test_cost_row_matches_lane_reward(lane):
    ingreso_nulo = lane_reward(lane["rate"], np.zeros(N), lane["dist"], lane["diesel"])
    np.testing.assert_allclose(-lane["cost"], ingreso_nulo, atol=0.5)       # round(·, 0) de lane_reward


def test_return_to_depot_with_no_alternative_value_is_break_even(lane):
    """j = depot, Ṽ(depot) = 0 y Q_alt = 0 ⇒ p* = costo/dist."""
    # Todas las alternativas valen exactamente 0: Ṽ[a] = −r_obs[a]/S.
    rev = np.where(lane["loads"] > 0, lane["rate"] * lane["dist"], 0.0)
    v = -(rev - lane["cost"]) / S
    p, only = price(lane, START, v)
    assert not only
    assert p == pytest.approx(lane["cost"][START] / lane["dist"][START])


def test_post_value_equal_to_alt_gives_break_even(lane):
    j = 3
    base = np.zeros(N)
    _, _ = price(lane, j, base)
    rev = np.where(lane["loads"] > 0, lane["rate"] * lane["dist"], 0.0)
    q = (rev - lane["cost"]) / S + base
    valid = lane["mask"].astype(bool).copy(); valid[CUR] = False; valid[j] = False
    base[j] = q[valid].max()                                     # Ṽ(j) = Q_alt
    p, only = price(lane, j, base)
    assert not only
    assert p == pytest.approx(lane["cost"][j] / lane["dist"][j])


def test_valuable_position_is_below_break_even(lane):
    j = 3
    v = np.zeros(N); v[j] = 50.0                                 # j es una posición muy valiosa
    p, _ = price(lane, j, v)
    assert p < lane["cost"][j] / lane["dist"][j]


def test_worthless_position_is_above_break_even(lane):
    j = 3
    v = np.zeros(N); v[j] = -50.0
    p, _ = price(lane, j, v)
    assert p > lane["cost"][j] / lane["dist"][j]


def test_only_option(lane):
    mask = np.zeros(N, dtype=np.int8); mask[START] = 1           # K agotado: solo el retorno
    p, only = price(lane, START, np.zeros(N), mask=mask)
    assert only and p == pytest.approx(lane["cost"][START] / lane["dist"][START])


def test_formula_by_hand(lane):
    j = 4
    v = np.array([0.0, 1.0, 9.9, 0.5, 2.0, 0.3])
    p, _ = price(lane, j, v)
    q = {a: ((lane["rate"][a] * lane["dist"][a] if lane["loads"][a] > 0 else 0.0) - lane["cost"][a]) / S + (0.0 if a == START else v[a])
         for a in (0, 1, 3, 5)}
    q_alt = max(q.values())
    assert p == pytest.approx((S * (q_alt - v[j]) + lane["cost"][j]) / lane["dist"][j])


def test_uses_agent_head_when_post_values_not_given(lane):
    class Agent:
        class post_value:
            @staticmethod
            def __call__(h, e): return torch.zeros(1, N)
        post_value = post_value()
    p, only = reservation_price(Agent(), torch.zeros(1, 4), torch.zeros(1, N, 4), lane["mask"], CUR, 3,
                                lane["rate"], lane["dist"], lane["cost"], START, S, loads_row=lane["loads"])
    assert np.isfinite(p) and not only


def test_percentil_historico():
    assert percentil_historico(2.5, np.array([1.0, 2.0, 3.0, 4.0])) == 50.0
    assert np.isnan(percentil_historico(1.0, np.array([])))
