"""
test_pricing.py
===============
Pruebas internas del precio de reserva (PRICING_PLAN.md): reservation_price con las cabezas A/B y q_values.
"""

import numpy as np
import pytest

from config import REWARD_SCALE_FACTOR as S
from pricing import lane_cost_row, percentil_historico, q_values, r_obs, reservation_price
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


def price(lane, j, va, vb=None, mask=None, **kw):
    vb = va if vb is None else vb
    args = dict(rate_row_obs=lane["rate"], distance_row=lane["dist"], cost_row=lane["cost"], start_node=START, S=S,
                loads_row=lane["loads"])
    args.update(kw)
    return reservation_price(np.asarray(va, float), np.asarray(vb, float), lane["mask"] if mask is None else mask,
                             CUR, j, **args)


@pytest.fixture
def heads():
    rng = np.random.default_rng(1)
    return rng.normal(0, 1.5, N), rng.normal(0, 1.5, N)


def test_cost_row_matches_lane_reward(lane):
    ingreso_nulo = lane_reward(lane["rate"], np.zeros(N), lane["dist"], lane["diesel"])
    np.testing.assert_allclose(-lane["cost"], ingreso_nulo, atol=0.5)       # round(·, 0) de lane_reward


def test_q_depot_has_zero_post_value(lane):
    v = np.full(N, 3.0)
    q = q_values(lane["rate"], lane["loads"], lane["dist"], lane["cost"], S, v, START)
    r = r_obs(lane["rate"], lane["loads"], lane["dist"], lane["cost"])
    assert q[START] == pytest.approx(r[START] / S)
    assert q[3] == pytest.approx(r[3] / S + 3.0)
    qb = q_values(np.tile(lane["rate"], (2, 1)), np.tile(lane["loads"], (2, 1)), np.tile(lane["dist"], (2, 1)),
                  np.tile(lane["cost"], (2, 1)), S, np.tile(v, (2, 1)), np.array([START, 1]))
    assert qb[0, START] == pytest.approx(q[START]) and qb[1, 1] == pytest.approx(r[1] / S)


@pytest.mark.parametrize("j", [1, 3, 4, 5])
def test_q_of_j_equals_q_alt_at_pstar(lane, heads, j):
    """Con la tarifa de j igual a p*, Q(s, j) = Q_alt, para cada pareja (A elige / B evalúa y al revés)."""
    va, vb = heads
    pr = price(lane, j, va, vb)
    assert pr.definido and not pr.only_option and pr.has_load
    valid = lane["mask"].astype(bool).copy(); valid[CUR] = False; valid[j] = False
    for p, elige, evalua, alt in ((pr.p_ab, va, vb, pr.alt_a), (pr.p_ba, vb, va, pr.alt_b)):
        rate = lane["rate"].copy(); rate[j] = p
        q_ev = q_values(rate, lane["loads"], lane["dist"], lane["cost"], S, evalua, START)
        q_el = q_values(rate, lane["loads"], lane["dist"], lane["cost"], S, elige, START)
        assert alt == np.where(valid)[0][np.argmax(q_el[valid])]               # la pareja elige con su cabeza
        assert q_ev[j] == pytest.approx(q_ev[alt])                              # Q(s, j) = Q_alt (evaluado con la otra)


def test_pstar_increases_with_cost_of_j(lane, heads):
    j = 4
    base = price(lane, j, *heads).p
    alto = dict(lane, cost=lane["cost"].copy()); alto["cost"][j] += 150.0
    assert price(alto, j, *heads).p > base
    # exactamente: dp = Δcosto / dist
    assert price(alto, j, *heads).p - base == pytest.approx(150.0 / lane["dist"][j], rel=1e-9)


def test_pstar_does_not_depend_on_S(lane, heads):
    """Ṽ está en unidades de USD/S: con otro S y Ṽ' = Ṽ·S/S' el resultado en USD/milla es el mismo."""
    j = 3
    va, vb = heads
    p1 = price(lane, j, va, vb)
    S2 = 37.0
    p2 = price(lane, j, va * S / S2, vb * S / S2, S=S2)
    assert p2.p == pytest.approx(p1.p) and p2.p_ab == pytest.approx(p1.p_ab) and p2.p_min == pytest.approx(p1.p_min)


def test_range_is_min_max_of_ab_and_ba(lane, heads):
    pr = price(lane, 3, *heads)
    assert pr.p == pytest.approx((pr.p_ab + pr.p_ba) / 2)
    assert pr.p_min == min(pr.p_ab, pr.p_ba) and pr.p_max == max(pr.p_ab, pr.p_ba)
    same = price(lane, 3, heads[0], heads[0])
    assert same.p_ab == pytest.approx(same.p_ba) and same.p_min == pytest.approx(same.p_max)


def test_formula_by_hand(lane):
    j = 4
    v = np.array([0.0, 1.0, 9.9, 0.5, 2.0, 0.3])
    pr = price(lane, j, v)
    q = {a: ((lane["rate"][a] * lane["dist"][a] if lane["loads"][a] > 0 else 0.0) - lane["cost"][a]) / S + (0.0 if a == START else v[a])
         for a in (0, 1, 3, 5)}
    assert pr.p == pytest.approx((S * (max(q.values()) - v[j]) + lane["cost"][j]) / lane["dist"][j])


def test_valuable_position_is_below_break_even_and_worthless_above(lane):
    j = 3
    v = np.zeros(N); v[j] = 50.0
    assert price(lane, j, v).p < lane["cost"][j] / lane["dist"][j]
    v[j] = -50.0
    assert price(lane, j, v).p > lane["cost"][j] / lane["dist"][j]


def test_no_price_without_load_or_for_the_return(lane, heads):
    sin_carga = price(lane, 1, *heads, loads_row=np.array([5, 0, 0, 2, 9, 1]))
    assert not sin_carga.has_load and not sin_carga.definido and np.isnan(sin_carga.p)
    assert sin_carga.break_even == pytest.approx(lane["cost"][1] / lane["dist"][1])
    retorno = price(lane, START, *heads)
    assert not retorno.has_load and np.isnan(retorno.p) and retorno.break_even == pytest.approx(lane["cost"][START] / lane["dist"][START])


def test_only_option(lane, heads):
    mask = np.zeros(N, dtype=np.int8); mask[3] = 1                  # única acción factible: j = 3 (con carga)
    pr = price(lane, 3, *heads, mask=mask)
    assert pr.only_option and pr.has_load and np.isnan(pr.p)


def test_alternatives_without_load_pay_only_the_cost(lane):
    """Una alternativa sin carga (reposicionamiento válido en el depot) vale −costo/S + Ṽ."""
    mask = np.array([0, 1, 0, 1, 1, 0], dtype=np.int8)               # estado de depot: el retorno no es válido
    loads = np.array([0, 0, 0, 2, 9, 0])
    v = np.zeros(N); v[1] = 100.0                                    # j=1 sin carga: muy valiosa como alternativa
    pr = price(lane, 3, v, mask=mask, loads_row=loads, start_node=CUR)
    q1 = -lane["cost"][1] / S + 100.0
    assert pr.alt_a == 1
    assert pr.p == pytest.approx((S * (q1 - 0.0) + lane["cost"][3]) / lane["dist"][3])


def test_percentil_historico():
    assert percentil_historico(2.5, np.array([1.0, 2.0, 3.0, 4.0])) == 50.0
    assert np.isnan(percentil_historico(1.0, np.array([])))
    # solo cuentan los días con carga
    assert percentil_historico(2.5, np.array([1.0, 2.0, 3.0, 4.0]), cargas=np.array([1, 0, 0, 1])) == 50.0
    assert percentil_historico(2.5, np.array([1.0, 2.0, 9.0, 9.0]), cargas=np.array([0, 1, 1, 1])) == pytest.approx(100 / 3)
    assert np.isnan(percentil_historico(2.5, np.array([1.0, 2.0]), cargas=np.zeros(2)))
