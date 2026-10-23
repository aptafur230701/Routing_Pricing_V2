"""
test_b2p.py
===========
B2P (B2 con pronósticos del índice de mercado, baselines.py):

  · b_h = f + (1 − f) · ρ^h: b_0 = 1 y decrece hacia f.
  · los valores esperados por h y día de la semana son media_lane + b_h · índice (tabla precalculada).
  · con índice cero B2P coincide con B2: mismos valores de continuación y mismas acciones.
  · el valor de seguir es monótono en el índice (los pronósticos entran con coeficiente ≥ 0).
  · B2P corre paso a paso en el arnés (mundos simulados y replay), con K libre y con K = 5, sin violar el tiempo.
"""

import numpy as np
import pytest

from baselines import Observacion, PoliticaB2, PoliticaB2P
from config import MAX_DURATION, MPG, MARGINAL_COST_SIN_DIESEL, TRAIN_DAYS, UBDP_DT_HORAS
from evaluation import ejecutar_politica, make_replay_env, make_sim_env
from market_process import MarketProcess, asegurar_parametros
from problem_data import load_matrices

N = 10
DIA = 14.0


@pytest.fixture(scope="module")
def d():
    asegurar_parametros()
    tm, rate, loads, dist, diesel, *_ = load_matrices(N)
    mp = MarketProcess(N)
    return dict(tm=tm, T=np.asarray(tm, float), rate=rate, loads=loads, dist=dist, diesel=diesel, mp=mp,
                b2=PoliticaB2(mp, tm, dist, diesel), b2p=PoliticaB2P(mp, tm, dist, diesel))


def test_b_h_es_la_mejor_prediccion_lineal(d):
    p, b = d["mp"].params, d["b2p"].b
    f, rho = p.frac_var_lento, p.rho
    assert b[0] == pytest.approx(1.0)
    assert np.allclose(b, f + (1 - f) * rho ** np.arange(len(b)))
    assert (np.diff(b) < 0).all() and b[-1] > f


def test_tabla_por_h_es_media_lane_mas_b_h_por_indice(d):
    mp, pol = d["mp"], d["b2p"]
    p = mp.params
    phi = p.phi[p.grupo]
    costo = d["dist"] * (d["diesel"] / MPG) + d["dist"] * MARGINAL_COST_SIN_DIESEL
    rng = np.random.default_rng(0)
    idx = rng.normal(0, 0.1, N)
    for dia_hoy, dia in ((0, 0), (2, 5), (3, 3)):
        h = dia - dia_hoy
        dow = (4 + dia) % 7
        m = p.mu * p.s[dow]
        prob = 1 - (phi / (phi + m)) ** phi
        esperado = prob * (p.media_lane + pol.b[h] * idx[:, None]) * d["dist"] - costo
        np.testing.assert_allclose(pol._ev_dia({}, 4, dia_hoy, dia, idx), esperado, rtol=1e-12, atol=1e-9)


def _estados(d, B=60, pasos=1, seed=0):
    """Observaciones de B2 reales: B episodios simulados avanzados `pasos` pasos con acciones válidas al azar."""
    rng = np.random.default_rng(seed)
    env = make_sim_env(d["tm"], d["rate"][:TRAIN_DAYS], d["loads"][:TRAIN_DAYS], d["dist"], d["diesel"], N, d["mp"])
    starts = rng.integers(0, N, B)
    dias = rng.integers(0, TRAIN_DAYS, B)
    k = [None if b % 2 else 5 for b in range(B)]
    masks = env.reset(starts, dias, k_max_arr=k, world_seeds=rng.integers(0, 2**31 - 1, B))
    activo = np.ones(B, bool)
    for _ in range(pasos):
        acc = np.array([rng.choice(np.flatnonzero(m)) for m in masks])
        _, term, masks = env.step(acc)
        activo &= ~term
    obs = []
    for b in np.flatnonzero(activo):
        obs.append(Observacion(
            nodo=int(env.current_node[b]), casa=int(starts[b]), tiempo=float(env.time_elapsed[b]),
            k_restante=int(env.k_remaining[b]) if env.k_restricted[b] else None, mascara=masks[b].copy(),
            loads_row=env.current_loads_row[b], rate_row=env.current_rate_row[b],
            dia_semana_salida=int(env.start_dow[b]), indice=env.market_index[b].copy()))
    return obs


def test_con_indice_cero_coincide_con_b2_en_valores_y_acciones(d):
    obs = _estados(d)
    assert len(obs) > 10
    import dataclasses
    for o in obs:
        cero = dataclasses.replace(o, indice=np.zeros(N))
        assert d["b2p"].actuar(cero) == d["b2"].actuar(o)
        # valores de continuación: V de B2P en las franjas de llegada a cada acción vs la recursión memoizada de B2
        h, dow0, k_sig = o.casa, o.dia_semana_salida, None if o.k_restante is None else o.k_restante - 1
        acts = [a for a in np.flatnonzero(o.mascara) if a != h]
        if not acts:
            continue
        s_a = {a: min(int(np.floor((o.tiempo + d["T"][o.nodo, a]) / UBDP_DT_HORAS + 1e-9)), d["b2p"]._smax) for a in acts}
        dia_hoy = int(np.floor(o.tiempo / UBDP_DT_HORAS + 1e-9) * UBDP_DT_HORAS / DIA + 1e-9)
        V = d["b2p"]._valores(h, dow0, dia_hoy, np.zeros(N), min(s_a.values()), k_sig)
        memo = {}
        for a in acts:
            assert V[a, s_a[a]] == pytest.approx(d["b2"]._valor(h, dow0, int(a), s_a[a], k_sig, memo), abs=1e-6)


def test_valor_de_seguir_es_monotono_en_el_indice(d):
    o = _estados(d, seed=3)[0]
    h, dow0 = o.casa, o.dia_semana_salida
    k = None if o.k_restante is None else o.k_restante - 1
    dia_hoy = int(o.tiempo // DIA)
    s_lo = int(np.floor(o.tiempo / UBDP_DT_HORAS + 1e-9))
    bajo = d["b2p"]._valores(h, dow0, dia_hoy, np.full(N, -0.1), s_lo, k)
    cero = d["b2p"]._valores(h, dow0, dia_hoy, np.zeros(N), s_lo, k)
    alto = d["b2p"]._valores(h, dow0, dia_hoy, np.full(N, 0.1), s_lo, k)
    assert (bajo[:, s_lo:] <= cero[:, s_lo:] + 1e-9).all() and (cero[:, s_lo:] <= alto[:, s_lo:] + 1e-9).all()
    assert (alto[:, s_lo:] > bajo[:, s_lo:]).any()


def test_b2p_sin_indice_es_error(d):
    o = _estados(d, B=6)[0]
    import dataclasses
    with pytest.raises(ValueError):
        d["b2p"].actuar(dataclasses.replace(o, indice=None))


@pytest.mark.parametrize("K", [None, 5])
def test_b2p_corre_en_el_arnes_simulado_y_replay(d, K):
    env = make_sim_env(d["tm"], d["rate"][:TRAIN_DAYS], d["loads"][:TRAIN_DAYS], d["dist"], d["diesel"], N, d["mp"])
    for m in range(6):
        r = ejecutar_politica(d["b2p"], env, m, 10 + m, K, 100 + m, d["dist"], d["diesel"], (10 + m + 3) % 7)
        assert not r["violacion"] and np.isfinite(r["margen"]) and r["ruta"][-1] == m
    ev = make_replay_env(d["tm"], d["rate"][TRAIN_DAYS:], d["loads"][TRAIN_DAYS:], d["dist"], d["diesel"], N,
                         day_offset=TRAIN_DAYS, market=d["mp"])
    for m in range(4):
        r = ejecutar_politica(d["b2p"], ev, m, m, K, None, d["dist"], d["diesel"], d["mp"].dia_semana(TRAIN_DAYS + m),
                              d["loads"][TRAIN_DAYS:], d["rate"][TRAIN_DAYS:])
        assert not r["violacion"] and np.isfinite(r["margen"])
