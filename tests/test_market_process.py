"""
test_market_process.py
======================
Proceso exógeno de cargas y tarifas (market_process.py, metodologia_realizaciones_mercado_reefer.md) y su ajuste:

  · Mundos sorteados en el dispositivo: formas, tipos, reproducibles con la semilla del generador
  · cargas Gamma-Poisson (= NegBin; Poisson si m < 1): media y ceros ≈ histórico por grupo de volumen; una lane sin
    cargas nunca sortea cargas; P(al menos una) = la de las estadísticas del agente
  · replay: cargas reales del día, δ = 0, días recortados al último con datos
  · tablas por día: precios con δ = 0 dentro de los cuantiles del día usado, niveles monótonos, m en [1, M_MAX]
  · estadísticas por lane del agente: solo días de entrenamiento, κ de entrenamiento
  · firma, índice histórico, día de la semana, ajuste (días excluidos) y la fórmula de lane_reward
"""

import copy
import datetime

import torch  # noqa: F401
import numpy as np
import pytest

from config import (DIAS_EXCLUIDOS, FECHA_INICIO, TRAIN_DAYS, MARKET_DIAS_EPISODIO, COTIZACIONES_MAX_POR_NODO,
                    COMPETENCIA_M_MAX, DEVICE)
from conftest import generador, mundos_sim
from fit_market_process import ajustar, dia_de_semana, params_path
from market_process import MarketProcess, asegurar_parametros, _CLAVES_AJUSTE
from pricing import N_NIVELES, indice_nivel, lane_cost_row, precios_niveles
from problem_data import cargar_dias_mercado, lane_reward

N = 10


# ── Mundos simulados ─────────────────────────────────────────────────────────

def test_mundos_formas_tipos_y_reproducibles(d10):
    mp = d10.market
    a = mundos_sim(mp, 40, semilla=3)
    b = mundos_sim(mp, 40, semilla=3)
    c = mundos_sim(mp, 40, semilla=4)
    D, C = MARKET_DIAS_EPISODIO, COTIZACIONES_MAX_POR_NODO
    assert a.loads.shape == (40, D, N, N) and a.loads.dtype == torch.int32 and int(a.loads.min()) >= 0
    assert a.u.shape == (40, D, N, N, C) and float(a.u.min()) >= 0 and float(a.u.max()) < 1
    assert a.delta.shape == (40,) and a.delta.dtype == torch.float64 and a.dia_max == TRAIN_DAYS - 1
    for x, y in ((a.loads, b.loads), (a.u, b.u), (a.delta, b.delta)):
        assert torch.equal(x, y)
    assert not torch.equal(a.u, c.u) and not torch.equal(a.delta, c.delta)
    sub = a.tomar(np.array([3, 3, 0]))
    assert sub.B == 3 and torch.equal(sub.loads[0], sub.loads[1]) and torch.equal(sub.u[2], a.u[0])


def test_delta_tiene_la_dispersion_del_ajuste(d10):
    mp = d10.market
    d = mp.sortear_deltas(20_000, generador(1))
    assert float(d.mean()) == pytest.approx(0.0, abs=4 * mp.s_delta / np.sqrt(20_000))
    assert float(d.std()) == pytest.approx(mp.s_delta, rel=0.03)


def test_lane_sin_cargas_nunca_genera_cargas(d10):
    mp = d10.market
    sin = mp.count_loads.sum(axis=0) == 0
    if not sin.any():
        pytest.skip("todas las lanes de estos nodos tuvieron carga alguna vez")
    m = mundos_sim(mp, 300, semilla=2)
    assert int(m.loads[:, :, torch.as_tensor(sin, device=DEVICE)].sum()) == 0


def test_media_y_ceros_simulados_se_parecen_al_historico_por_grupo():
    n = 35
    mp = MarketProcess(n)
    hist = mp.count_loads[:TRAIN_DAYS]
    universo = hist.sum(0) > 0
    media, ceros, reps = np.zeros((n, n)), np.zeros((n, n)), 30
    dias = np.arange(TRAIN_DAYS)
    for r in range(reps):
        L = mp.sortear_mundos(dias, 1, TRAIN_DAYS - 1, generador(100 + r)).loads[:, 0].cpu().numpy()   # (D, n, n)
        media += L.mean(axis=0)
        ceros += (L == 0).mean(axis=0)
    media /= reps
    ceros /= reps
    for h in range(len(mp.r)):
        sel = universo & (mp._grupo_lane == h)
        assert sel.sum() > 0
        assert media[sel].mean() == pytest.approx(hist.mean(0)[sel].mean(), rel=0.08, abs=0.02)
        assert ceros[sel].mean() == pytest.approx((hist == 0).mean(0)[sel].mean(), abs=0.05)


def test_p1_del_agente_es_la_frecuencia_del_generador(d10):
    """P(al menos una carga) de las estadísticas del agente es la que producen los mundos simulados."""
    mp, est = d10.market, d10.est
    k = 2
    dias = mp.dias_train_por_dow(k)
    reps = 400
    m = mp.sortear_mundos(np.repeat(dias, reps), 1, TRAIN_DAYS - 1, generador(7))
    freq = (m.loads[:, 0] > 0).double().mean(dim=0).cpu().numpy()
    np.testing.assert_allclose(est.P1[k], freq, atol=0.03)


# ── Replay ───────────────────────────────────────────────────────────────────

def test_replay_usa_las_cargas_reales_y_recorta_al_ultimo_dia(d10):
    mp = d10.market
    ultimo = mp.num_dias_total - 1
    dia0 = np.array([TRAIN_DAYS, TRAIN_DAYS + 5, ultimo - 1])
    m = mp.mundos_replay(dia0, MARKET_DIAS_EPISODIO, ultimo, generador(0))
    assert (m.delta == 0).all() and m.dia_max == ultimo
    for b, d0 in enumerate(dia0):
        for d in range(MARKET_DIAS_EPISODIO):
            np.testing.assert_array_equal(m.loads[b, d].cpu().numpy(), mp.count_loads[min(d0 + d, ultimo)])
    otra = mp.mundos_replay(dia0, MARKET_DIAS_EPISODIO, ultimo, generador(1))
    assert torch.equal(otra.loads, m.loads) and not torch.equal(otra.u, m.u)    # otra semilla: otro competidor


# ── Tablas por día ───────────────────────────────────────────────────────────

def test_tablas_por_dia(d10):
    mp = d10.market
    t = mp.tensores(DEVICE)
    D = mp.num_dias_total
    assert t["precios"].shape == (D, N, N, N_NIVELES) and t["cuant"].shape == (D, N, N, 6)
    assert t["m"].shape == (D, N) and t["indice"].shape == (D, N) and t["dow"].shape == (D,)
    assert (t["cuant"].diff(dim=-1) >= -1e-9).all() and (t["precios"].diff(dim=-1) >= -1e-9).all()
    assert (t["precios"] >= t["cuant"][..., :1] - 1e-9).all() and (t["precios"] <= t["cuant"][..., -1:] + 1e-9).all()
    assert (t["m"] >= 1).all() and (t["m"] <= COMPETENCIA_M_MAX).all()
    assert (t["precios"] > 0).all()
    np.testing.assert_array_equal(t["loads_reales"].cpu().numpy(), mp.count_loads)
    np.testing.assert_allclose(t["indice"].cpu().numpy(), mp.indice_historico(0, None))
    # Los cuantiles son los del propio día si la lane tuvo cargas, si no los del día más cercano con cargas.
    rng = np.random.default_rng(0)
    for _ in range(20):
        dia, i = int(rng.integers(0, TRAIN_DAYS)), int(rng.integers(0, N))
        dp = mp._dia_precio(dia, np.array([i]))[0]
        np.testing.assert_allclose(t["cuant"][dia, i, :, 0].cpu().numpy(), mp._q["min"][dp, i, np.arange(N)])


def test_la_firma_cambia_con_los_parametros_del_mercado(d10):
    """La tabla V de UB-DP y el caché de la evaluación se recalculan solos si cambian los datos del mercado."""
    from baselines.dp_relajada import _firma
    from config import UBDP_DT_HORAS, UBDP_S, UBDP_SEED
    mp = d10.market
    T = np.ones((N, N))
    args = (T, T, T, 5, UBDP_S, UBDP_DT_HORAS, UBDP_SEED)
    base, f0 = mp.firma_params(), _firma(mp, *args)
    assert MarketProcess(N).firma_params() == base
    for campo in _CLAVES_AJUSTE:
        otro = copy.copy(mp)
        otro._params_blob = dict(mp._params_blob)
        val = mp._params_blob[campo]
        otro._params_blob[campo] = val + (1 if np.issubdtype(val.dtype, np.integer) else 0.01)
        assert otro.firma_params() != base and _firma(otro, *args) != f0, campo
    otro = copy.copy(mp)
    otro.count_loads = mp.count_loads + 1
    assert otro.firma_params() != base


# ── Estadísticas por lane del agente ─────────────────────────────────────────

def test_estadisticas_forma_rangos_y_solo_dias_de_entrenamiento(d10):
    est, mp = d10.est, d10.market
    assert est.L.shape == est.P1.shape == est.G.shape == (7, N, N) and est.M.shape == (7, N)
    for x in (est.L, est.P1, est.G, est.M):
        assert np.isfinite(x).all()
    assert est.L.min() >= 0 and 0 <= est.P1.min() and est.P1.max() <= 1
    assert est.M.min() >= 1 and est.M.max() <= COMPETENCIA_M_MAX
    costo = lane_cost_row(d10.dist, d10.diesel)
    for k in range(7):
        dias = mp.dias_train_por_dow(k)
        assert dias.max() < TRAIN_DAYS
        np.testing.assert_allclose(est.L[k], mp._m_stack[dias].mean(axis=0))
        q = np.mean([mp.cuantiles_dia(int(x)) for x in dias], axis=0)
        np.testing.assert_allclose(est.G[k], precios_niveles(q)[..., indice_nivel(0.5)] * d10.dist - costo,
                                   rtol=1e-5, atol=1e-3)
    kappa = mp.kappa
    try:
        mp.kappa = 2.0 * kappa                                   # sensibilidad de κ: el agente no se entera
        otra = mp.estadisticas_dow(d10.dist, d10.diesel)
    finally:
        mp.kappa = kappa
    np.testing.assert_array_equal(otra.M, est.M)


# ── Índice, calendario, recompensa ───────────────────────────────────────────

def test_indice_historico_es_la_definicion_del_ajuste():
    mp = MarketProcess(N)
    with np.load(params_path()) as z:
        guardado = z["indice"][:, :N]
        media, uni = z["media_lane"], z["universo_indice"]
    assert np.allclose(mp.indice_historico(0, TRAIN_DAYS), guardado, atol=1e-9)
    ev = mp.indice_historico(TRAIN_DAYS, None)
    mean_rate = cargar_dias_mercado()["mean"][TRAIN_DAYS:].astype(np.float64)
    bruto = ((mean_rate - media[None]) * uni[None]).sum(axis=2) / uni.sum(axis=1)[None]
    assert np.allclose(ev, (bruto - bruto.mean(axis=1, keepdims=True))[:, :N])


def test_dia_de_semana_desde_fecha_inicio():
    fecha = datetime.date.fromisoformat(FECHA_INICIO)
    dias = np.arange(30)
    esperado = [(fecha + datetime.timedelta(days=int(d))).weekday() for d in dias]
    assert np.array_equal(dia_de_semana(dias, FECHA_INICIO), esperado)


def test_lane_reward_es_la_formula_de_siempre(d10):
    from config import MPG, MARGINAL_COST_SIN_DIESEL
    for d in (0, 5, 89):
        rev = d10.rate[d] * d10.dist
        rev[d10.loads[d] <= 0] = 0
        cost = d10.dist * (d10.diesel / MPG) + d10.dist * MARGINAL_COST_SIN_DIESEL
        assert np.array_equal(lane_reward(d10.rate[d], d10.loads[d], d10.dist, d10.diesel), np.round(rev - cost, 0))


# ── Ajuste ───────────────────────────────────────────────────────────────────

def _panel_sintetico(D=21, n=6, seed=0):
    rng = np.random.default_rng(seed)
    loads = rng.poisson(8, size=(D, n, n)).astype(np.float32)
    loads[:, 0, 1] = 0
    rate = rng.uniform(2, 4, size=(D, n, n)).astype(np.float32)
    return dict(count_loads=loads, mean=rate, fechas=np.datetime64(FECHA_INICIO, "D") + np.arange(D))


def test_ajuste_usa_todos_los_dias_o_excluye_los_indicados():
    assert list(DIAS_EXCLUIDOS) == []
    for D in (21, 35):
        p = ajustar(DIAS_EXCLUIDOS, ventana_lento=5, dias_data=_panel_sintetico(D=D), train_days=D)
        assert np.array_equal(p["dias_usados"], np.arange(D)) and p["indice"].shape[0] == D
        assert len(p["lento_escenarios"]) == D - 5 + 1 and p["fechas_ajuste"][0] == FECHA_INICIO
        assert p["f_w"].mean() == pytest.approx(1.0) and not p["universo"][0, 1]
    p = ajustar(["2026-06-05", "2026-06-06"], ventana_lento=5, dias_data=_panel_sintetico(), train_days=21)
    assert len(p["dias_usados"]) == 19 and 4 not in p["dias_usados"] and 5 not in p["dias_usados"]
    with pytest.raises(ValueError):
        ajustar(["2025-12-31"], dias_data=_panel_sintetico(), train_days=21)


def test_parametros_guardados_usan_solo_dias_de_entrenamiento():
    asegurar_parametros()
    with np.load(params_path()) as z:
        assert z["indice"].shape[0] == len(z["dias_usados"]) == TRAIN_DAYS
        assert np.array_equal(z["dias_usados"], np.arange(TRAIN_DAYS))
