"""
test_continuation_features.py
=============================
Features de continuación por nodo (columnas 6–13, USE_CONTINUATION_FEATURES), índice de mercado (INDEX_COL),
MarketProcess.estadisticas_dow y la expansión del checkpoint a más columnas de entrada.

  · (B, N, N_NODE_FEATURES), sin NaN/Inf; las columnas 6–13 coinciden con una referencia escalar con loops.
  · Las estadísticas por lane (L, P1, G, M) son la media del generador sobre los días de ENTRENAMIENTO: P1 coincide
    con la frecuencia que da `sortear_cargas` y M usa el κ de entrenamiento aunque el mundo tenga otro.
  · El conjunto factible F_j excluye las lanes j→l desde las que ya no se alcanza a volver a casa.
  · casos especiales: j == i (la espera), j == h (el ciclo termina) y la espera en casa antes de salir.
  · la versión escalar build_node_features coincide con la batcheada; el encoder no importa de baselines.
  · start_dow del entorno; transfer de columnas de entrada (nuevas en cero, misma salida que antes).
"""

import inspect

import numpy as np
import pytest
import torch

import model.encoder as encoder_mod
from model.encoder import (
    INDEX_COL, N_NODE_FEATURES, build_continuation_features_batch, build_node_features, build_node_features_batch,
)
from pricing import indice_nivel, lane_cost_row, precios_niveles
from routing_env import fin_espera
from config import (
    MAX_DURATION, USE_CONTINUATION_FEATURES, TRAIN_DAYS, LOADS_EXPECTED_CLIP, COMPETENCIA_M_MAX,
)
from evaluation import make_sim_env, make_replay_env
from fit_market_process import dia_de_semana
from market_process import EstadisticasLane, MarketProcess, asegurar_parametros
from problem_data import load_matrices

N = 10
CONT = slice(6, 14)          # las 8 columnas de continuación
pytestmark = pytest.mark.skipif(not USE_CONTINUATION_FEATURES, reason="USE_CONTINUATION_FEATURES = False")


@pytest.fixture(scope="module")
def datos():
    asegurar_parametros()
    (tm, rate, loads, dist, diesel, ltr, avail, p95) = load_matrices(N)
    market = MarketProcess(N)
    return dict(tm=tm, T=np.asarray(tm, dtype=float), rate=rate[:TRAIN_DAYS], loads=loads[:TRAIN_DAYS],
                dist=dist, diesel=diesel, ltr=ltr, avail=avail, p95=p95, market=market,
                est=market.estadisticas_dow(dist, diesel))


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
        pares = [np.argwhere(o) for o in env.opciones_validas]
        acc = np.array([p[rng.integers(len(p))] if len(p) else (0, 0) for p in pares])
        _, term, masks = env.step(acc[:, 0], acc[:, 1])
        activo &= ~term
    return env, np.flatnonzero(activo), starts


def _features(d, env, starts):
    return build_node_features_batch(
        env.current_node, starts, d["T"], d["dist"], env.current_day_idx, N,
        MAX_DURATION, reward_global_p95=d["p95"],
        avail_expected_arr=d["avail"], reward_rows=env.reward_rows,
        est=d["est"], time_elapsed=env.time_elapsed, dow0=env.start_dow,
        market_index=env.market_index, index_std=env.index_std,
    )


def _referencia(d, env, starts, b):
    """Columnas 6–13 de la fila b con loops de Python (definición literal del docstring del encoder)."""
    T, est, p95 = d["T"], d["est"], d["p95"]
    i, h, t = int(env.current_node[b]), int(starts[b]), float(env.time_elapsed[b])
    dow0 = int(env.start_dow[b])
    clip = LOADS_EXPECTED_CLIP
    out = np.zeros((N, 8))
    for j in range(N):
        espera = j == i
        termina = j == h and not espera                    # volver a casa: el ciclo termina
        tj = float(fin_espera(t)) if espera else t + T[i, j]
        Tjh = 0.0 if j == h else T[j, h]
        dow_j = (dow0 + int(tj // 14)) % 7
        out[j, 0] = Tjh / MAX_DURATION
        out[j, 1] = np.clip((MAX_DURATION - tj - Tjh) / MAX_DURATION, 0, 1)
        if j != h:
            out[j, 2] = np.clip(est.G[dow_j, j, h] / p95, -1, 1)
            out[j, 3] = min(est.L[dow_j, j, h], clip) / clip
        if termina:
            continue
        F = [l for l in range(N) if l not in (i, j, h) and tj + T[j, l] + T[l, h] <= MAX_DURATION]
        if F:
            out[j, 4] = np.clip(max(est.P1[dow_j, j, l] * est.G[dow_j, j, l] for l in F) / p95, -1, 1)
            out[j, 5] = np.mean([min(est.L[dow_j, j, l], clip) / clip for l in F])
        out[j, 6] = len(F) / N
        out[j, 7] = np.clip(np.log(max(est.M[dow_j, j], 1.0)) / np.log(COMPETENCIA_M_MAX), 0, 1)
    return out


def test_n_node_features():
    assert N_NODE_FEATURES == 6 + 8 + 1 + 3 + 2 * 9 + 2    # 6 base + 8 continuación + índice + cotización (2 por nivel + 2)
    assert INDEX_COL == 14


def test_el_encoder_no_importa_de_baselines():
    """La continuación del DRL sale de datos de entrenamiento (market_process), no de la tabla de ningún baseline."""
    assert "baselines" not in inspect.getsource(encoder_mod)


# ─────────────────────────────────────────────────────────────────────────────
# MarketProcess.estadisticas_dow
# ─────────────────────────────────────────────────────────────────────────────

def test_estadisticas_forma_rangos_y_solo_dias_de_entrenamiento(datos):
    d = datos
    est, mp = d["est"], d["market"]
    assert est.L.shape == est.P1.shape == est.G.shape == (7, N, N) and est.M.shape == (7, N)
    for x in (est.L, est.P1, est.G, est.M):
        assert np.isfinite(x).all()
    assert est.L.min() >= 0 and 0 <= est.P1.min() and est.P1.max() <= 1
    assert est.M.min() >= 1 and est.M.max() <= COMPETENCIA_M_MAX
    mediana = indice_nivel(0.5)
    costo = lane_cost_row(d["dist"], d["diesel"])
    for k in range(7):
        dias = mp.dias_train_por_dow(k)
        assert dias.max() < TRAIN_DAYS                                  # nada de octubre
        np.testing.assert_allclose(est.L[k], mp._m_stack[dias].mean(axis=0))
        q = np.mean([mp.cuantiles_dia(int(x)) for x in dias], axis=0)
        np.testing.assert_allclose(est.G[k], precios_niveles(q)[..., mediana] * d["dist"] - costo,
                                   rtol=1e-5, atol=1e-3)           # cuantiles y distancias en float32
        ltr = mp.ltr_hub[dias]
        assert est.M[k].shape == (N,) and ltr.shape == (len(dias), N)


def test_p1_coincide_con_la_frecuencia_del_generador(datos):
    """P(al menos una carga) de la tabla es la que produce `sortear_cargas` (Poisson si m < 1, si no binomial negativa)."""
    mp, est = datos["market"], datos["est"]
    k, nodo, S = 2, 0, 4000
    rng = np.random.Generator(np.random.PCG64(1))
    dias = mp.dias_train_por_dow(k)
    freq = np.mean([(mp.sortear_cargas(np.full(S, nodo), np.full(S, int(x)), rng) > 0).mean(axis=0) for x in dias], axis=0)
    np.testing.assert_allclose(est.P1[k, nodo], freq, atol=0.02)


def test_competencia_media_usa_el_kappa_de_entrenamiento(datos):
    """Cambiar κ del mundo (sensibilidad) no cambia lo que el agente sabe de antemano."""
    d = datos
    kappa = d["market"].kappa
    try:
        d["market"].kappa = 2.0 * kappa
        otra = d["market"].estadisticas_dow(d["dist"], d["diesel"])
    finally:
        d["market"].kappa = kappa
    np.testing.assert_array_equal(otra.M, d["est"].M)


# ─────────────────────────────────────────────────────────────────────────────
# Features de continuación
# ─────────────────────────────────────────────────────────────────────────────

def test_forma_y_finitud(datos):
    env, activas, starts = _estados(datos)
    feats = _features(datos, env, starts)
    assert feats.shape == (len(starts), N, N_NODE_FEATURES) and feats.dtype == np.float32
    assert np.isfinite(feats).all()
    cont = feats[:, :, CONT]
    assert cont[:, :, 0].min() >= 0                       # T[j, h] / MAX no se recorta (puede pasar de 1)
    assert cont[:, :, 1].min() >= 0 and cont[:, :, 1].max() <= 1
    assert cont[:, :, [2, 4]].min() >= -1 and cont[:, :, [2, 4]].max() <= 1
    assert cont[:, :, [3, 5, 6, 7]].min() >= 0 and cont[:, :, [3, 5, 6, 7]].max() <= 1


@pytest.mark.parametrize("pasos", [0, 1, 2])
def test_columnas_6_a_13_coinciden_con_referencia_con_loops(datos, pasos):
    env, activas, starts = _estados(datos, pasos=pasos, semilla=pasos)
    feats = _features(datos, env, starts)
    assert len(activas) > 0
    for b in activas[:15]:
        np.testing.assert_allclose(feats[b, :, CONT], _referencia(datos, env, starts, b), atol=1e-6)


def test_conjunto_factible_excluye_lanes_desde_las_que_no_se_alcanza_a_volver():
    """Con N = 4: ir de j a l = 2 y volver a casa pasa del tiempo máximo, así que su valor (enorme) no cuenta."""
    T = np.full((4, 4), 10.0)
    np.fill_diagonal(T, 0.0)
    T[2, 0] = 70.0                                          # 2 → casa: 70 h
    G = np.zeros((7, 4, 4))
    G[:, 1, 2], G[:, 1, 3] = 100.0, 20.0
    est = EstadisticasLane(L=np.full((7, 4, 4), 5.0), P1=np.ones((7, 4, 4)), G=G, M=np.full((7, 4), 2.0))
    f = build_continuation_features_batch(
        np.array([0]), np.array([0]), np.array([0.0]), np.array([0]), T, est, MAX_DURATION, 100.0)[0]
    assert f[1, 4] == pytest.approx(0.2)                    # solo la lane 1→3 (20 / p95)
    assert f[1, 6] == pytest.approx(1 / 4)                  # una lane factible de N = 4
    assert f[1, 5] == pytest.approx(5.0 / LOADS_EXPECTED_CLIP)
    assert f[1, 7] == pytest.approx(np.log(2.0) / np.log(COMPETENCIA_M_MAX))


def test_casos_especiales_self_loop_y_casa(datos):
    env, activas, starts = _estados(datos, pasos=1, semilla=3)
    feats = _features(datos, env, starts)
    fuera = [b for b in activas if env.current_node[b] != starts[b]]   # tras un rechazo en casa sigue en casa
    assert fuera
    for b in fuera:
        i, h = int(env.current_node[b]), int(starts[b])
        # Fila propia = la espera: continuación desde i mañana (la referencia con loops lo cubre).
        np.testing.assert_allclose(feats[b, i, CONT], _referencia(datos, env, starts, b)[i], atol=1e-6)
        assert feats[b, i, 1] * MAX_DURATION == pytest.approx(float(fin_espera(env.time_elapsed[b]) - env.time_elapsed[b]), abs=1e-4)
        # Volver a casa termina el ciclo: sin lane de regreso ni salientes; solo queda la holgura del retorno directo.
        assert feats[b, h, 6] == 0 and (feats[b, h, 8:14] == 0).all()
        directo = np.clip((MAX_DURATION - env.time_elapsed[b] - datos["T"][i, h]) / MAX_DURATION, 0, 1)
        assert feats[b, h, 7] == pytest.approx(float(directo), abs=1e-6)
    # en el primer paso i == h: la espera en casa no tiene lane de regreso, pero sí las salientes de mañana
    env0, _, starts0 = _estados(datos, pasos=0, semilla=9)
    f0 = _features(datos, env0, starts0)[np.arange(len(starts0)), starts0, CONT]
    assert (f0[:, [0, 2, 3]] == 0).all()
    assert (f0[:, 6] > 0).any()                              # hay casas con lanes salientes factibles (no todas: a 10 nodos pocos pares tienen ida y vuelta)
    assert (f0[f0[:, 6] == 0][:, [4, 5]] == 0).all()        # sin lanes factibles, el valor y las cargas salientes son 0
    holgura = np.clip((MAX_DURATION - fin_espera(env0.time_elapsed)) / MAX_DURATION, 0, 1)
    np.testing.assert_allclose(f0[:, 1], holgura, atol=1e-6)


def test_version_escalar_coincide_con_la_batcheada(datos):
    env, activas, starts = _estados(datos, pasos=1, semilla=2)
    feats = _features(datos, env, starts)
    for b in activas[:5]:
        i = int(env.current_node[b])
        rm_dia = np.asarray(env.reward_rows[b])
        esc = build_node_features(
            i, int(starts[b]), rm_dia[None].repeat(N, 0),
            datos["T"], datos["dist"], N, MAX_DURATION,
            reward_global_p95=datos["p95"], avail_expected_arr=datos["avail"],
            est=datos["est"], time_elapsed=float(env.time_elapsed[b]), dow0=int(env.start_dow[b]),
            market_index=env.market_index[b], index_std=env.index_std,
        )
        assert esc.shape == (N, N_NODE_FEATURES)
        np.testing.assert_allclose(esc[:, 6:], feats[b, :, 6:], atol=1e-6)


def test_el_indice_es_el_de_mercado_normalizado(datos):
    """La columna INDEX_COL es índice(j, día actual) / desviación del índice histórico, sin NaN."""
    env, activas, starts = _estados(datos, pasos=1, semilla=4)
    feats = _features(datos, env, starts)
    idx = env.market_index
    assert idx.shape == (len(starts), N) and env.index_std == datos["market"].indice_std
    np.testing.assert_allclose(feats[:, :, INDEX_COL], idx / env.index_std, atol=1e-6)
    # el índice es el del mundo de cada episodio en su día actual (sin recortar al histórico)
    for b in activas[:10]:
        dia = int(env._start_day_idx[b]) + int(env.time_elapsed[b] // 14)
        assert np.array_equal(idx[b], datos["market"].indice_mundo(int(env.world_seed[b]), dia))


def test_replay_features_forma_sin_nan_e_indice_del_dia_real(datos):
    """Replay de días reales de evaluación: (B, N, N_NODE_FEATURES) sin NaN; INDEX_COL es el índice observado del día."""
    d = datos
    tm, rate, loads, dist, diesel, *_ = load_matrices(N)
    ev_dias = rate[TRAIN_DAYS:]
    env = make_replay_env(tm, ev_dias, loads[TRAIN_DAYS:], dist, diesel, N, day_offset=TRAIN_DAYS, market=d["market"])
    B = 12
    rng = np.random.default_rng(0)
    starts = rng.integers(0, N, size=B)
    dias = rng.integers(0, ev_dias.shape[0], size=B)
    env.reset(starts, dias, k_max_arr=[None] * B, world_seeds=np.arange(B))
    feats = build_node_features_batch(
        env.current_node, starts, d["T"], dist, env.current_day_idx, N, MAX_DURATION,
        reward_global_p95=d["p95"], avail_expected_arr=d["avail"],
        reward_rows=env.reward_rows, est=d["est"], time_elapsed=env.time_elapsed, dow0=env.start_dow,
        market_index=env.market_index, index_std=env.index_std,
    )
    assert feats.shape == (B, N, N_NODE_FEATURES) and np.isfinite(feats).all()
    esperado = d["market"].indice_historico(TRAIN_DAYS, None)[dias] / d["market"].indice_std
    np.testing.assert_allclose(feats[:, :, INDEX_COL], esperado, atol=1e-6)


def test_start_dow_del_entorno(datos):
    for offset in (0, TRAIN_DAYS):
        env = make_sim_env(datos["tm"], datos["rate"], datos["loads"], datos["dist"], datos["diesel"], N,
                           datos["market"])
        env._day_offset = offset
        env.reset(np.array([0, 1, 2]), np.array([3, 10, 50]), world_seeds=np.array([1, 2, 3]))
        par = [np.argwhere(o)[0] for o in env.opciones_validas]       # primera opción válida de cada fila
        env.step(np.array([p[0] for p in par]), np.array([p[1] for p in par]))   # el tiempo avanza; start_dow no cambia
        assert (env.start_dow == dia_de_semana(np.array([3, 10, 50]) + offset)).all()


# ─────────────────────────────────────────────────────────────────────────────
# Transfer de columnas de entrada
# ─────────────────────────────────────────────────────────────────────────────

def test_transfer_7_a_11_columnas_deja_las_nuevas_en_cero():
    from transfer_learning import _reconcile_state_dict
    torch.manual_seed(0)
    viejo = torch.randn(32, 7)
    destino = {"encoder.input_proj.proj.weight": torch.randn(32, 11), "encoder.input_proj.proj.bias": torch.randn(32)}
    fuente = {"encoder.input_proj.proj.weight": viejo, "encoder.input_proj.proj.bias": torch.ones(32)}
    sd, transferidos, omitidos = _reconcile_state_dict(destino, fuente)
    w = sd["encoder.input_proj.proj.weight"]
    assert torch.equal(w[:, :7], viejo) and (w[:, 7:] == 0).all() and not omitidos
    x = torch.randn(4, 11)
    assert torch.allclose(x @ w.T, x[:, :7] @ viejo.T)    # las features nuevas no cambian la salida


def test_transfer_11_a_12_columnas_deja_el_indice_en_cero():
    """Un checkpoint de 11 features se carga con la columna del índice de mercado (la última) en cero."""
    from transfer_learning import _reconcile_state_dict
    torch.manual_seed(1)
    viejo = torch.randn(32, 11)
    destino = {"encoder.input_proj.proj.weight": torch.randn(32, 12), "encoder.input_proj.proj.bias": torch.randn(32)}
    fuente = {"encoder.input_proj.proj.weight": viejo, "encoder.input_proj.proj.bias": torch.ones(32)}
    sd, _, omitidos = _reconcile_state_dict(destino, fuente)
    w = sd["encoder.input_proj.proj.weight"]
    assert torch.equal(w[:, :11], viejo) and (w[:, 11] == 0).all() and not omitidos


def test_un_checkpoint_con_las_columnas_de_nodo_de_antes_se_rechaza():
    """Con 34 columnas (antes del 2026-11-27) `_reconcile_state_dict` las copiaría desalineadas: se rechaza antes."""
    from transfer_learning import _tiene_entradas_actuales
    clave = "encoder.input_proj.proj.weight"
    assert _tiene_entradas_actuales({"agent": {clave: torch.zeros(32, N_NODE_FEATURES)}})
    assert not _tiene_entradas_actuales({"agent": {clave: torch.zeros(32, 34)}})
    assert not _tiene_entradas_actuales({"agent": {}})


def test_transfer_rechaza_forma_incompatible_con_error_claro():
    """Un checkpoint con MÁS features de nodo (12 → 11) no se omite en silencio: ValueError."""
    from transfer_learning import _reconcile_state_dict
    destino = {"encoder.input_proj.proj.weight": torch.zeros(32, 11), "encoder.input_proj.proj.bias": torch.zeros(32)}
    fuente = {"encoder.input_proj.proj.weight": torch.randn(32, 12), "encoder.input_proj.proj.bias": torch.ones(32)}
    with pytest.raises(ValueError, match=r"encoder\.input_proj\.proj\.weight.*\(32, 12\).*\(32, 11\)"):
        _reconcile_state_dict(destino, fuente)
    # Otra dimensión de salida, o un vector 1-D de otro tamaño, tampoco es una expansión válida.
    with pytest.raises(ValueError):
        _reconcile_state_dict({"w": torch.zeros(16, 11)}, {"w": torch.zeros(32, 7)})
    with pytest.raises(ValueError):
        _reconcile_state_dict({"b": torch.zeros(32)}, {"b": torch.zeros(16)})
    # Claves que el destino no tiene se omiten (no es un error); las que faltan en el checkpoint se conservan.
    sd, transferidos, omitidos = _reconcile_state_dict({"a": torch.zeros(2)}, {"a": torch.ones(2), "extra": torch.ones(3)})
    assert omitidos == ["extra"] and torch.equal(sd["a"], torch.ones(2))
