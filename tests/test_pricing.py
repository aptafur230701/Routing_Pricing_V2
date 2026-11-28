"""
test_pricing.py
===============
Modelo de mercado de la cotización (pricing.py, especificacion_fase0_ruteo_pricing.md §3 y §6):

  · m(LTR): valores en el LTR de referencia, en los extremos y monotonía
  · con un competidor el percentil es el uniforme; la frecuencia de ganar converge a (1 − u_k)^m
  · ningún nivel gana siempre (el más bajo es p10); niveles en los cuantiles, escalados por δ, monótonos
  · reglas fijas: mediana, break-even + margen y miope; la recursión de reintentos
  · las versiones en torch (entorno, features y DP en el dispositivo) dan lo mismo que las de numpy
"""

import numpy as np
import pytest
import torch

from config import U_NIVELES, COMPETENCIA_KAPPA, COMPETENCIA_M_MAX, BE_MARGEN
from pricing import (
    N_NIVELES, U_NIVELES_ARR, competidores_efectivos, gana, interp_cuantiles, interp_cuantiles_t, lane_cost_row,
    lane_cost_t, lane_reward_t, nivel_regla, percentil_competidor, percentil_competidor_t, percentil_historico,
    precios_niveles, prob_ganar, prob_ganar_t, valor_reintentos, valor_reintentos_t,
)
from problem_data import lane_reward


@pytest.fixture
def cuantiles():
    return np.array([2.0, 3.0, 4.0, 5.0, 6.0, 8.0])          # min, q1, q2, q3, p90, max


# ── Competencia y aceptación ─────────────────────────────────────────────────

def test_competidores_en_ltr_de_referencia_y_extremos():
    ref = 25.0
    assert competidores_efectivos(ref, ref) == pytest.approx(1.0 + COMPETENCIA_KAPPA)
    assert competidores_efectivos(1e9, ref) == pytest.approx(1.0, abs=1e-6)
    assert competidores_efectivos(0.01, ref) == COMPETENCIA_M_MAX
    assert competidores_efectivos(0.0, ref) == COMPETENCIA_M_MAX          # hub sin dato ⇒ tope
    m = competidores_efectivos(np.array([0.5, 2.0, 10.0, 25.0, 100.0]), ref)
    assert np.all(np.diff(m) <= 0) and np.all((m >= 1.0) & (m <= COMPETENCIA_M_MAX))


def test_con_un_competidor_el_percentil_es_el_uniforme():
    U = np.random.default_rng(0).random(1000)
    np.testing.assert_allclose(percentil_competidor(U, 1.0), U)


@pytest.mark.parametrize("m", [1.0, 2.0, 7.5, COMPETENCIA_M_MAX])
def test_frecuencia_de_ganar_converge_a_la_formula(m):
    u_comp = percentil_competidor(np.random.default_rng(1).random(400_000), m)
    for u in U_NIVELES:
        # error estándar máximo sqrt(0.25 / 4e5) ≈ 8e-4; la tolerancia es ~4 errores estándar
        assert gana(u, u_comp).mean() == pytest.approx(float(prob_ganar(u, m)), abs=3e-3)


def test_ningun_nivel_gana_siempre_y_prob_baja_con_nivel_y_competencia():
    assert U_NIVELES_ARR[0] == pytest.approx(0.1)                       # el más bajo es p10
    assert not gana(U_NIVELES_ARR[0], np.array([0.0, 0.05, 0.1])).any()
    assert gana(U_NIVELES_ARR[0], np.array([0.11, 0.5, 0.999])).all()
    p = prob_ganar(U_NIVELES_ARR, 3.0)
    assert p[0] == pytest.approx(0.9 ** 3) and np.all(np.diff(p) < 0)
    assert prob_ganar(0.5, 1.0) > prob_ganar(0.5, 2.0) > prob_ganar(0.5, 10.0)


# ── Niveles de precio ────────────────────────────────────────────────────────

def test_niveles_en_los_cuantiles_y_escalados_por_delta(cuantiles):
    p = precios_niveles(cuantiles)
    assert p.shape == (N_NIVELES,) and np.all(np.diff(p) >= 0)
    assert 2.0 < p[0] < 3.0                                            # u = 0.1 → entre min y q1
    assert p[list(U_NIVELES).index(0.5)] == pytest.approx(4.0)        # u = 0.5 → q2
    assert p[list(U_NIVELES).index(0.9)] == pytest.approx(6.0)        # u = 0.9 → p90
    lote = np.stack([cuantiles, cuantiles * 2])
    p2 = precios_niveles(lote, np.log(np.array([1.1, 0.9])))
    np.testing.assert_allclose(p2[0], 1.1 * p)
    np.testing.assert_allclose(p2[1], 0.9 * 2 * p)


# ── Reglas fijas ─────────────────────────────────────────────────────────────

def test_reglas_mediana_y_break_even(cuantiles):
    p = precios_niveles(cuantiles)
    assert U_NIVELES[int(nivel_regla("mediana", p, 500.0, 1500.0, 3.0))] == 0.5
    dist, costo = 500.0, 500.0 * 3.1
    k = int(nivel_regla("be_margen", p, dist, costo, 3.0))
    piso = costo / dist + BE_MARGEN
    assert p[k] >= piso and (k == 0 or p[k - 1] < piso)
    assert int(nivel_regla("be_margen", p, dist, dist * 50.0, 3.0)) == N_NIVELES - 1   # piso inalcanzable


@pytest.mark.parametrize("m", [1.0, 3.0, 15.0])
def test_regla_miope_es_el_mejor_valor_esperado_del_tramo(cuantiles, m):
    p = precios_niveles(cuantiles)
    valor = [prob_ganar(u, m) * (p[k] * 600.0 - 600.0 * 2.9) for k, u in enumerate(U_NIVELES)]
    assert int(nivel_regla("miope", p, 600.0, 600.0 * 2.9, m)) == int(np.argmax(valor))
    assert int(nivel_regla("miope", p, 600.0, 1700.0, 15.0)) <= int(nivel_regla("miope", p, 600.0, 1700.0, 1.0))
    with pytest.raises(ValueError):
        nivel_regla("otra", p, 500.0, 1500.0, 2.0)


def test_valor_reintentos_es_la_recursion_y_sube_con_los_intentos():
    P = (1.0 - U_NIVELES_ARR) ** 2.0
    margen = -50.0 + 300.0 * U_NIVELES_ARR
    for base in (0.0, -900.0):
        V, ks = valor_reintentos(P, margen, base, 4)
        v = base
        for a in range(1, 5):
            q = P * margen + (1 - P) * v
            assert ks[a - 1] == int(np.argmax(q))
            v = float(q.max())
            assert V[a] == pytest.approx(v)
        assert (np.diff(V) > 0).all() and (np.diff(ks) >= 0).all()      # más cargas por cotizar: más valor y más alto


# ── torch = numpy ────────────────────────────────────────────────────────────

def test_versiones_torch_coinciden_con_numpy():
    rng = np.random.default_rng(3)
    q = np.sort(rng.uniform(1.5, 4.0, size=(50, 6)), axis=-1)
    u = rng.random(50)
    torch.testing.assert_close(interp_cuantiles_t(torch.as_tensor(u), torch.as_tensor(q)),
                               torch.as_tensor(interp_cuantiles(u, q)), rtol=1e-12, atol=1e-12)
    m = rng.uniform(1.0, COMPETENCIA_M_MAX, size=50)
    torch.testing.assert_close(prob_ganar_t(torch.as_tensor(u), torch.as_tensor(m)),
                               torch.as_tensor(prob_ganar(u, m)))
    torch.testing.assert_close(percentil_competidor_t(torch.as_tensor(u), torch.as_tensor(m)),
                               torch.as_tensor(percentil_competidor(u, m)))
    P = prob_ganar(U_NIVELES_ARR, m[:, None])
    margen = rng.uniform(-300, 900, size=(50, N_NIVELES))
    base = rng.uniform(-500, 0, size=50)
    torch.testing.assert_close(
        valor_reintentos_t(torch.as_tensor(P), torch.as_tensor(margen), torch.as_tensor(base), 4),
        torch.as_tensor(valor_reintentos(P, margen, base, 4)[0]))
    dist = rng.uniform(50, 900, size=(8, 8))
    diesel = rng.uniform(3.5, 4.5, size=(8, 8)).astype(np.float32)
    rate = rng.uniform(1.5, 3.5, size=(8, 8))
    loads = rng.integers(0, 3, size=(8, 8))
    torch.testing.assert_close(lane_cost_t(torch.as_tensor(dist), torch.as_tensor(diesel)),
                               torch.as_tensor(lane_cost_row(dist, diesel)))
    np.testing.assert_array_equal(
        lane_reward_t(torch.as_tensor(rate), torch.as_tensor(loads), torch.as_tensor(dist),
                      torch.as_tensor(diesel)).numpy(),
        lane_reward(rate, loads, dist, diesel))


# ── Utilidades ───────────────────────────────────────────────────────────────

def test_costo_de_la_lane_y_percentil_historico():
    rng = np.random.default_rng(0)
    dist, diesel = rng.uniform(200, 900, 6), rng.uniform(3.5, 4.5, 6)
    np.testing.assert_allclose(-lane_cost_row(dist, diesel), lane_reward(rng.uniform(2, 3, 6), np.zeros(6), dist, diesel),
                               atol=0.5)
    assert percentil_historico(2.5, np.array([1.0, 2.0, 3.0, 4.0])) == 50.0
    assert np.isnan(percentil_historico(1.0, np.array([])))
    assert percentil_historico(2.5, np.array([1.0, 2.0, 9.0, 9.0]), cargas=np.array([0, 1, 1, 1])) == pytest.approx(100 / 3)
    assert np.isnan(percentil_historico(2.5, np.array([1.0, 2.0]), cargas=np.zeros(2)))
