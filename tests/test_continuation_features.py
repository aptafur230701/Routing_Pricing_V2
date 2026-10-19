"""
test_continuation_features.py
=============================
Features de continuación por nodo (f8–f11, USE_CONTINUATION_FEATURES), calcular_ev_esperado y
la expansión del checkpoint de 8 a 12 columnas de entrada.

  · calcular_ev_esperado da exactamente el (7, N, N) que PoliticaB2 tenía en __init__, y B2 no cambia.
  · (B, N, 12), sin NaN/Inf; f8–f11 coinciden con una referencia escalar con loops.
  · f10 coincide con PoliticaB2._ev en 5 estados elegidos al azar.
  · casos especiales: j == i (todo 0) y j == h (f8 = f10 = f11 = 0, f9 = holgura del retorno directo).
  · la versión escalar build_node_features coincide con la batcheada.
  · start_dow del entorno; transfer de 8 → 12 columnas (nuevas en cero, misma salida que antes).
"""

import numpy as np
import pytest
import torch

import attention_encoder as ae
from attention_encoder import (
    N_NODE_FEATURES, build_node_features, build_node_features_batch, build_continuation_features_batch,
)
from baselines import PoliticaB2, calcular_ev_esperado
from config import (
    MAX_DURATION, MPG, MARGINAL_COST_SIN_DIESEL, USE_CONTINUATION_FEATURES, TRAIN_DAYS,
)
from evaluation import ejecutar_politica, make_sim_env
from fit_market_process import dia_de_semana
from market_process import MarketProcess, asegurar_parametros
from problem_data import load_matrices

N = 10
pytestmark = pytest.mark.skipif(not USE_CONTINUATION_FEATURES, reason="USE_CONTINUATION_FEATURES = False")


@pytest.fixture(scope="module")
def datos():
    asegurar_parametros()
    (tm, rate, loads, dist, diesel, ltr, trucks, avail, p95) = load_matrices(N)
    market = MarketProcess(N)
    return dict(tm=tm, T=np.asarray(tm, dtype=float), rate=rate[:TRAIN_DAYS], loads=loads[:TRAIN_DAYS],
                dist=dist, diesel=diesel, ltr=ltr, trucks=trucks, avail=avail, p95=p95, market=market,
                ev=calcular_ev_esperado(market, dist, diesel))


def _estados(d, B=60, pasos=2, semilla=0):
    """Entorno simulado con B episodios avanzados `pasos` pasos con acciones válidas al azar.

    Devuelve (env, filas activas, start_nodes)."""
    rng = np.random.default_rng(semilla)
    env = make_sim_env(d["tm"], d["rate"], d["loads"], d["dist"], d["diesel"], N, d["market"])
    starts = rng.integers(0, N, size=B)
    dias = rng.integers(0, TRAIN_DAYS, size=B)
    masks = env.reset(starts, dias, k_max_arr=[5] * B, world_seeds=rng.integers(0, 2**31 - 1, size=B))
    activo = np.ones(B, dtype=bool)
    for _ in range(pasos):
        acc = np.array([rng.choice(np.flatnonzero(m)) if m.any() else 0 for m in masks])
        _, term, masks = env.step(acc)
        activo &= ~term
    return env, np.flatnonzero(activo), starts


def _features(d, env, starts):
    return build_node_features_batch(
        env.current_node, starts, env.visited_mask, None, d["T"], d["dist"], env.current_day_idx, N,
        MAX_DURATION, trucks_stack_raw=d["trucks"], reward_global_p95=d["p95"],
        avail_expected_arr=d["avail"], reward_rows=env.reward_rows,
        ev=d["ev"], time_elapsed=env.time_elapsed, dow0=env.start_dow,
    )


def _referencia(d, env, starts, b):
    """f8–f11 de la fila b con loops de Python (definición literal de la especificación)."""
    T, ev, p95 = d["T"], d["ev"], d["p95"]
    i, h, t = int(env.current_node[b]), int(starts[b]), float(env.time_elapsed[b])
    dow0, vis = int(env.start_dow[b]), env.visited_mask[b]
    out = np.zeros((N, 4))
    for j in range(N):
        if j == i:
            continue
        tj = t + T[i, j]
        dj = (dow0 + int(tj // 14)) % 7
        if j == h:
            out[j, 1] = np.clip((MAX_DURATION - tj) / MAX_DURATION, 0, 1)
            continue
        out[j, 0] = T[j, h] / MAX_DURATION
        out[j, 1] = np.clip((MAX_DURATION - tj - T[j, h]) / MAX_DURATION, 0, 1)
        out[j, 2] = np.clip(ev[dj, j, h] / p95, -1, 1)
        mejor = ev[dj, j, h]
        for l in range(N):
            if vis[l] or l in (i, j, h) or tj + T[j, l] + T[l, h] > MAX_DURATION:
                continue
            dl = (dow0 + int((tj + T[j, l]) // 14)) % 7
            mejor = max(mejor, ev[dj, j, l] + ev[dl, l, h])
        out[j, 3] = np.clip(mejor / p95, -1, 1)
    return out


def test_n_node_features_es_12():
    assert N_NODE_FEATURES == 12


def test_ev_esperado_identico_al_de_b2_y_b2_no_cambia(datos):
    d = datos
    p = d["market"].params
    phi = p.phi[p.grupo]                                   # fórmula original de PoliticaB2.__init__
    costo = d["dist"] * (d["diesel"] / MPG) + d["dist"] * MARGINAL_COST_SIN_DIESEL
    m = p.mu[None, :, :] * p.s[:, None, None]
    prob = 1.0 - (phi[None] / (phi[None] + m)) ** phi[None]
    ev_original = prob * p.tarifas.mean(axis=0).astype(float)[None] * d["dist"][None] - costo[None]
    assert d["ev"].shape == (7, N, N)
    assert np.array_equal(d["ev"], ev_original)

    b2 = PoliticaB2(d["market"], d["T"], d["dist"], d["diesel"])
    assert np.array_equal(b2._ev, ev_original)
    b2_ref = PoliticaB2(d["market"], d["T"], d["dist"], d["diesel"])
    b2_ref._ev = ev_original
    env = make_sim_env(d["tm"], d["rate"], d["loads"], d["dist"], d["diesel"], N, d["market"])
    for k, (nodo, dia, ws) in enumerate([(0, 3, 11), (4, 40, 12), (7, 77, 13), (9, 5, 14), (2, 60, 15), (5, 22, 16)]):
        dow = int(d["market"].dia_semana(dia))
        a = ejecutar_politica(b2, env, nodo, dia, 5, ws, d["dist"], d["diesel"], dow)
        r = ejecutar_politica(b2_ref, env, nodo, dia, 5, ws, d["dist"], d["diesel"], dow)
        assert a["ruta"] == r["ruta"] and a["margen"] == r["margen"]


def test_forma_y_finitud(datos):
    env, activas, starts = _estados(datos)
    feats = _features(datos, env, starts)
    assert feats.shape == (len(starts), N, 12) and feats.dtype == np.float32
    assert np.isfinite(feats).all()
    cont = feats[:, :, 8:]
    assert cont[:, :, 0].min() >= 0                       # f8 = T[j, h] / MAX no se recorta (puede pasar de 1)
    assert cont[:, :, 1].min() >= 0 and cont[:, :, 1].max() <= 1
    assert cont[:, :, [2, 3]].min() >= -1 and cont[:, :, [2, 3]].max() <= 1


@pytest.mark.parametrize("pasos", [0, 1, 2])
def test_f8_a_f11_coinciden_con_referencia_con_loops(datos, pasos):
    env, activas, starts = _estados(datos, pasos=pasos, semilla=pasos)
    feats = _features(datos, env, starts)
    assert len(activas) > 5
    for b in activas:
        np.testing.assert_allclose(feats[b, :, 8:], _referencia(datos, env, starts, b), atol=1e-6)


def test_f10_coincide_con_ev_de_b2_en_5_estados_al_azar(datos):
    env, activas, starts = _estados(datos, B=80, pasos=1, semilla=5)
    feats = _features(datos, env, starts)
    b2 = PoliticaB2(datos["market"], datos["T"], datos["dist"], datos["diesel"])
    rng = np.random.default_rng(123)
    for b in rng.choice(activas, size=5, replace=False):
        i, h, t = int(env.current_node[b]), int(starts[b]), float(env.time_elapsed[b])
        for j in range(N):
            if j in (i, h):
                continue
            llegada = t + datos["T"][i, j]
            dj = (int(env.start_dow[b]) + int(llegada // 14)) % 7           # mismo indexado que B2._continuacion
            esperado = np.clip(b2._ev[dj, j, h] / datos["p95"], -1, 1)
            assert feats[b, j, 10] == pytest.approx(float(esperado), abs=1e-6)


def test_casos_especiales_self_loop_y_casa(datos):
    env, activas, starts = _estados(datos, pasos=1, semilla=3)
    feats = _features(datos, env, starts)
    for b in activas:
        i, h = int(env.current_node[b]), int(starts[b])
        assert (feats[b, i, 8:] == 0).all()
        assert feats[b, h, 8] == 0 and feats[b, h, 10] == 0 and feats[b, h, 11] == 0
        directo = np.clip((MAX_DURATION - env.time_elapsed[b] - datos["T"][i, h]) / MAX_DURATION, 0, 1)
        assert feats[b, h, 9] == pytest.approx(float(directo), abs=1e-6)
    # en el primer paso i == h: todo en cero para el self-loop
    env0, _, starts0 = _estados(datos, pasos=0, semilla=9)
    f0 = _features(datos, env0, starts0)
    assert (f0[np.arange(len(starts0)), starts0, 8:] == 0).all()


def test_f11_es_al_menos_f10(datos):
    env, _, starts = _estados(datos, pasos=1, semilla=8)
    feats = _features(datos, env, starts)
    assert (feats[:, :, 11] >= feats[:, :, 10] - 1e-7).all()


def test_version_escalar_coincide_con_la_batcheada(datos):
    env, activas, starts = _estados(datos, pasos=1, semilla=2)
    feats = _features(datos, env, starts)
    for b in activas[:5]:
        i = int(env.current_node[b])
        rm_dia = np.asarray(env.reward_rows[b])
        esc = build_node_features(
            i, int(starts[b]), set(np.flatnonzero(env.visited_mask[b])), rm_dia[None].repeat(N, 0),
            datos["T"], datos["dist"], N, MAX_DURATION, trucks_stack=None,
            reward_global_p95=datos["p95"], avail_expected_arr=datos["avail"],
            ev=datos["ev"], time_elapsed=float(env.time_elapsed[b]), dow0=int(env.start_dow[b]),
        )
        assert esc.shape == (N, 12)
        np.testing.assert_allclose(esc[:, 8:], feats[b, :, 8:], atol=1e-6)


def test_start_dow_del_entorno(datos):
    for offset in (0, TRAIN_DAYS):
        env = make_sim_env(datos["tm"], datos["rate"], datos["loads"], datos["dist"], datos["diesel"], N,
                           datos["market"])
        env._day_offset = offset
        env.reset(np.array([0, 1, 2]), np.array([3, 10, 50]), world_seeds=np.array([1, 2, 3]))
        env.step(np.array([4, 5, 6]))                      # el tiempo avanza; start_dow no cambia
        assert (env.start_dow == dia_de_semana(np.array([3, 10, 50]) + offset)).all()


def test_transfer_8_a_12_columnas_deja_las_nuevas_en_cero():
    from transfer_learning import _reconcile_state_dict
    torch.manual_seed(0)
    viejo = torch.randn(32, 8)
    destino = {"encoder.input_proj.proj.weight": torch.randn(32, 12), "encoder.input_proj.proj.bias": torch.randn(32)}
    fuente = {"encoder.input_proj.proj.weight": viejo, "encoder.input_proj.proj.bias": torch.ones(32)}
    sd, transferidos, omitidos = _reconcile_state_dict(destino, fuente)
    w = sd["encoder.input_proj.proj.weight"]
    assert torch.equal(w[:, :8], viejo) and (w[:, 8:] == 0).all() and not omitidos
    x = torch.randn(4, 12)
    assert torch.allclose(x @ w.T, x[:, :8] @ viejo.T)    # las features nuevas no cambian la salida
