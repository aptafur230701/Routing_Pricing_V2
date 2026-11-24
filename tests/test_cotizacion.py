"""
test_cotizacion.py
==================
Fase 1 de especificacion_fase0_ruteo_pricing.md: modelo de mercado de la cotización conjunta.

  · m(LTR): valores en el LTR de referencia, en los extremos y monotonía (§3.3)
  · con m = 1 el competidor cotiza exactamente la tarifa sorteada del modelo anterior (test 1)
  · la frecuencia de ganar en el nivel k converge a (1 − u_k)^m (test 2)
  · el nivel 0 gana siempre y los niveles son monótonos en precio (test 3)
  · determinismo por (semilla, nodo, día), sin importar el lote ni el orden (test 4)
  · el tablero no cambia las cargas ni la tarifa de los métodos actuales
  · reglas de precio fijas: mediana, break-even + margen y miope óptima (§6)
"""

import numpy as np
import pytest

from config import U_NIVELES, COMPETENCIA_KAPPA, COMPETENCIA_M_MAX, BE_MARGEN, TRAIN_DAYS
from market_process import MarketProcess, Tablero, asegurar_parametros
from pricing import (
    N_NIVELES, U_NIVELES_ARR, competidores_efectivos, gana, interp_cuantiles, nivel_regla,
    percentil_competidor, precios_niveles, prob_ganar,
)

NUM_NODES = 10
SEMILLA = 12345
DIA = 10


@pytest.fixture(scope="module")
def mp():
    asegurar_parametros()
    return MarketProcess(NUM_NODES)


@pytest.fixture
def cuantiles():
    return np.array([2.0, 3.0, 4.0, 5.0, 6.0, 8.0])          # min, q1, q2, q3, p90, max


# ── pricing.py: competencia y aceptación ─────────────────────────────────────

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


def test_nivel_cero_gana_siempre():
    assert gana(0.0, np.array([0.0, 1e-12, 0.5, 0.999])).all()
    u_comp = percentil_competidor(np.random.default_rng(2).random(100_000), COMPETENCIA_M_MAX)
    assert gana(U_NIVELES_ARR[0], u_comp).all()


def test_prob_ganar_baja_con_el_nivel_y_con_la_competencia():
    p = prob_ganar(U_NIVELES_ARR, 3.0)
    assert p[0] == 1.0 and np.all(np.diff(p) < 0)
    assert prob_ganar(0.5, 1.0) > prob_ganar(0.5, 2.0) > prob_ganar(0.5, 10.0)


# ── pricing.py: niveles de precio ────────────────────────────────────────────

def test_niveles_en_los_cuantiles(cuantiles):
    p = precios_niveles(cuantiles)
    assert p.shape == (N_NIVELES,)
    assert p[0] == pytest.approx(2.0)                                  # u = 0   → min
    assert p[list(U_NIVELES).index(0.5)] == pytest.approx(4.0)        # u = 0.5 → q2
    assert p[list(U_NIVELES).index(0.9)] == pytest.approx(6.0)        # u = 0.9 → p90
    assert np.all(np.diff(p) >= 0)
    np.testing.assert_allclose(p, interp_cuantiles(U_NIVELES_ARR, np.tile(cuantiles, (N_NIVELES, 1))))


def test_niveles_escalan_con_delta_y_aceptan_lotes(cuantiles):
    lote = np.stack([cuantiles, cuantiles * 2])                       # (2, 6)
    delta = np.log(np.array([1.1, 0.9]))
    p = precios_niveles(lote, delta)
    assert p.shape == (2, N_NIVELES)
    np.testing.assert_allclose(p[0], 1.1 * precios_niveles(cuantiles))
    np.testing.assert_allclose(p[1], 0.9 * 2 * precios_niveles(cuantiles))


# ── pricing.py: reglas fijas ─────────────────────────────────────────────────

def test_regla_mediana(cuantiles):
    k = nivel_regla("mediana", precios_niveles(cuantiles), 500.0, 1500.0, 3.0)
    assert U_NIVELES[int(k)] == 0.5


def test_regla_break_even_mas_margen(cuantiles):
    p = precios_niveles(cuantiles)
    dist, costo = 500.0, 500.0 * 3.1                                   # break-even 3.1 USD/mi
    k = int(nivel_regla("be_margen", p, dist, costo, 3.0))
    piso = costo / dist + BE_MARGEN
    assert p[k] >= piso and (k == 0 or p[k - 1] < piso)
    assert int(nivel_regla("be_margen", p, dist, dist * 50.0, 3.0)) == N_NIVELES - 1   # piso inalcanzable


@pytest.mark.parametrize("m", [1.0, 3.0, 15.0])
def test_regla_miope_es_el_mejor_valor_esperado_del_tramo(cuantiles, m):
    p = precios_niveles(cuantiles)
    dist, costo = 600.0, 600.0 * 2.9
    valor = [prob_ganar(u, m) * (p[k] * dist - costo) for k, u in enumerate(U_NIVELES)]
    assert int(nivel_regla("miope", p, dist, costo, m)) == int(np.argmax(valor))


def test_regla_miope_cotiza_mas_bajo_con_mas_competencia(cuantiles):
    p = precios_niveles(cuantiles)
    k_poca = int(nivel_regla("miope", p, 600.0, 1700.0, 1.0))
    k_mucha = int(nivel_regla("miope", p, 600.0, 1700.0, 15.0))
    assert k_mucha <= k_poca


def test_reglas_en_lote_y_regla_desconocida(cuantiles):
    p = np.broadcast_to(precios_niveles(cuantiles), (4, 3, N_NIVELES))
    for regla in ("mediana", "be_margen", "miope"):
        assert nivel_regla(regla, p, 500.0, 1500.0, np.full((4, 3), 2.0)).shape == (4, 3)
    with pytest.raises(ValueError):
        nivel_regla("otra", p, 500.0, 1500.0, 2.0)


# ── market_process.py ────────────────────────────────────────────────────────

def _fila(mp, nodos, dias, semillas):
    return mp.tablero_por_mundo(np.asarray(nodos), np.asarray(dias), np.asarray(semillas))


def test_cuantiles_ordenados(mp):
    q = np.stack([mp._q[c] for c in ("min", "q1", "q2", "q3", "p90", "max")], axis=-1)
    assert np.all(np.diff(q, axis=-1) >= 0)


def test_ltr_de_referencia(mp):
    assert mp.ltr_ref == pytest.approx(25.5, rel=0.05)                # mediana jun-sep 2026, 194 hubs
    assert mp.ltr_hub.shape == (mp.num_dias_total, NUM_NODES)


def test_tablero_no_cambia_cargas_ni_tarifa(mp):
    nodos = np.arange(NUM_NODES)
    dias, semillas = np.full(NUM_NODES, DIA), np.full(NUM_NODES, SEMILLA)
    loads, rates = mp.sample_arrivals_por_mundo(nodos, dias, semillas)
    t = _fila(mp, nodos, dias, semillas)
    np.testing.assert_array_equal(t.loads, loads)
    np.testing.assert_array_equal(t.tarifa, rates)
    np.testing.assert_allclose(t.m, mp.competidores(nodos, dias))

    rng_a, rng_b = np.random.default_rng(7), np.random.default_rng(7)
    l2, r2 = mp.sample_arrivals(nodos, dias, rng_a)
    t2 = mp.tablero_sin_memoria(nodos, dias, rng_b)
    np.testing.assert_array_equal(t2.loads, l2)
    np.testing.assert_array_equal(t2.tarifa, r2)


def test_con_un_competidor_cotiza_la_tarifa_de_siempre(mp, monkeypatch):
    """Test 1 de la especificación: el modelo anterior es el caso m = 1."""
    monkeypatch.setattr(mp, "competidores", lambda nodos, dias: np.ones(len(nodos)))
    nodos = np.arange(NUM_NODES)
    t = _fila(mp, nodos, np.full(NUM_NODES, DIA), np.full(NUM_NODES, SEMILLA))
    np.testing.assert_allclose(t.precio_competidor, t.tarifa, rtol=1e-6)


def test_formas_y_niveles_monotonos(mp):
    nodos = np.arange(NUM_NODES)
    t = _fila(mp, nodos, np.full(NUM_NODES, DIA), np.full(NUM_NODES, SEMILLA))
    assert t.loads.shape == (NUM_NODES, NUM_NODES) and t.m.shape == (NUM_NODES,)
    assert t.precios.shape == (NUM_NODES, NUM_NODES, N_NIVELES)
    assert np.all(np.diff(t.precios, axis=-1) >= -1e-9)
    assert np.all(t.precios[..., 0] <= t.tarifa + 1e-4)                # nivel 0 = mínimo del mercado
    assert np.all((t.u_comp >= 0) & (t.u_comp <= 1))
    assert np.all((t.m >= 1) & (t.m <= COMPETENCIA_M_MAX))


def test_determinismo_sin_importar_lote_ni_orden(mp):
    """Test 4: la fila (semilla, nodo, día) es la misma sola, dentro de un lote o en otro orden."""
    nodos = np.array([3, 0, 7, 3, 5])
    dias = np.array([DIA, 4, DIA, 20, 0])
    semillas = np.array([SEMILLA, 99, SEMILLA, SEMILLA, 5])
    lote = _fila(mp, nodos, dias, semillas)
    orden = np.array([4, 2, 0, 3, 1])
    otro = _fila(mp, nodos[orden], dias[orden], semillas[orden])
    for b in range(len(nodos)):
        sola = _fila(mp, nodos[b:b + 1], dias[b:b + 1], semillas[b:b + 1])
        pos = int(np.flatnonzero(orden == b)[0])
        for campo in ("loads", "precios", "u_comp", "precio_competidor", "m", "tarifa"):
            np.testing.assert_array_equal(getattr(lote, campo)[b], getattr(sola, campo)[0])
            np.testing.assert_array_equal(getattr(otro, campo)[pos], getattr(sola, campo)[0])


def test_mundo_cotizacion_coincide_con_tablero_y_sample_world(mp):
    D = 3
    mundo = mp.mundo_cotizacion(SEMILLA, D, dia_inicial=DIA)
    loads_w, rate_w = mp.sample_world(SEMILLA, D, dia_inicial=DIA)
    assert mundo.precios.shape == (D, NUM_NODES, NUM_NODES, N_NIVELES) and mundo.m.shape == (D, NUM_NODES)
    np.testing.assert_array_equal(mundo.loads, loads_w)
    np.testing.assert_allclose(mundo.tarifa, rate_w, rtol=1e-6)
    d, i = 1, 4
    fila = _fila(mp, [i], [DIA + d], [SEMILLA])
    np.testing.assert_array_equal(mundo.u_comp[d, i], fila.u_comp[0])


def test_tablero_real_usa_el_dia_real_y_es_determinista(mp):
    nodos = np.arange(NUM_NODES)
    dias = np.full(NUM_NODES, TRAIN_DAYS + 2)                          # un día de prueba (octubre)
    semillas = np.full(NUM_NODES, 31)
    a = mp.tablero_real(nodos, dias, semillas)
    b = mp.tablero_real(nodos, dias, semillas)
    np.testing.assert_array_equal(a.loads, mp.count_loads[dias, nodos])
    np.testing.assert_array_equal(a.u_comp, b.u_comp)
    c = mp.tablero_real(nodos, dias, np.full(NUM_NODES, 32))
    assert not np.array_equal(a.u_comp, c.u_comp)                      # otra semilla, otra competencia
    assert isinstance(a, Tablero) and np.all(np.diff(a.precios, axis=-1) >= -1e-9)
