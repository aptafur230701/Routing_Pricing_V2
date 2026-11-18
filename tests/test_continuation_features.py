"""
test_continuation_features.py
=============================
Features de continuación por nodo (f6–f9, USE_CONTINUATION_FEATURES), índice de mercado (f10),
calcular_ev_esperado y la expansión del checkpoint a más columnas de entrada.

  · (B, N, 11), sin NaN/Inf; f6–f9 coinciden con una referencia escalar con loops.
  · f10 = índice de mercado del nodo en el día actual / desviación del índice histórico.
  · f6–f9 usan el valor esperado AJUSTADO por el índice (el de B2P): ev + índice(origen) · q[h, dow];
    con índice cero dan EXACTAMENTE lo de antes (copia congelada de la función anterior).
  · f8 coincide con el ev ajustado de PoliticaB2P en 5 estados elegidos al azar.
  · casos especiales: j == i (todo 0) y j == h (f6 = f8 = f9 = 0, f7 = holgura del retorno directo).
  · la versión escalar build_node_features coincide con la batcheada.
  · start_dow del entorno; transfer de columnas de entrada (nuevas en cero, misma salida que antes).

Nota: calcular_ev_esperado ya no es el modelo paramétrico de B2 (mu/s/phi/media_lane) sino un
promedio empírico sobre los días reales de entrenamiento (ver baselines/politicas.py); el test que
comparaba ev contra esa fórmula paramétrica (y MarketProcess.params, que ya no existe) quedó roto
por ese cambio — no por el de las features de camiones — y no se tocó aquí.
"""

import numpy as np
import pytest
import torch

from model.encoder import (
    N_NODE_FEATURES, build_continuation_features_batch, build_node_features, build_node_features_batch,
)
from baselines.politicas import PoliticaB2, PoliticaB2P, calcular_ev_esperado, calcular_ev_pronostico
from config import (
    MAX_DURATION, MPG, MARGINAL_COST_SIN_DIESEL, USE_CONTINUATION_FEATURES, TRAIN_DAYS,
)
from evaluation import make_sim_env, make_replay_env
from fit_market_process import dia_de_semana
from market_process import MarketProcess, asegurar_parametros
from problem_data import load_matrices

N = 10
pytestmark = pytest.mark.skipif(not USE_CONTINUATION_FEATURES, reason="USE_CONTINUATION_FEATURES = False")


@pytest.fixture(scope="module")
def datos():
    asegurar_parametros()
    (tm, rate, loads, dist, diesel, ltr, avail, p95) = load_matrices(N)
    market = MarketProcess(N)
    return dict(tm=tm, T=np.asarray(tm, dtype=float), rate=rate[:TRAIN_DAYS], loads=loads[:TRAIN_DAYS],
                dist=dist, diesel=diesel, ltr=ltr, avail=avail, p95=p95, market=market,
                ev=calcular_ev_pronostico(market, dist, diesel))


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
        env.current_node, starts, None, d["T"], d["dist"], env.current_day_idx, N,
        MAX_DURATION, reward_global_p95=d["p95"],
        avail_expected_arr=d["avail"], reward_rows=env.reward_rows,
        ev=d["ev"], time_elapsed=env.time_elapsed, dow0=env.start_dow,
        market_index=env.market_index, index_std=env.index_std,
    )


def _referencia(d, env, starts, b):
    """f6–f9 de la fila b con loops de Python (definición literal de la especificación)."""
    T, tab, p95 = d["T"], d["ev"], d["p95"]
    ev, q, H = tab.ev, tab.q, tab.h_max
    idx = env.market_index[b]
    i, h, t = int(env.current_node[b]), int(starts[b]), float(env.time_elapsed[b])
    dow0, hoy = int(env.start_dow[b]), int(t // 14)

    def ev_aj(origen, destino, t_salida):
        """ev + índice(origen) · q[h, dow] para un arco que sale en t_salida (h = días desde hoy)."""
        dia = int(t_salida // 14)
        dow, hh = (dow0 + dia) % 7, min(max(dia - hoy, 0), H)
        return ev[dow, origen, destino] + idx[origen] * q[hh, dow, origen, destino]

    out = np.zeros((N, 4))
    for j in range(N):
        if j == i:
            continue
        tj = t + T[i, j]
        if j == h:
            out[j, 1] = np.clip((MAX_DURATION - tj) / MAX_DURATION, 0, 1)
            continue
        out[j, 0] = T[j, h] / MAX_DURATION
        out[j, 1] = np.clip((MAX_DURATION - tj - T[j, h]) / MAX_DURATION, 0, 1)
        out[j, 2] = np.clip(ev_aj(j, h, tj) / p95, -1, 1)
        mejor = ev_aj(j, h, tj)
        for l in range(N):
            if l in (i, j, h) or tj + T[j, l] + T[l, h] > MAX_DURATION:
                continue
            mejor = max(mejor, ev_aj(j, l, tj) + ev_aj(l, h, tj + T[j, l]))
        out[j, 3] = np.clip(mejor / p95, -1, 1)
    return out


def test_n_node_features_es_11():
    assert N_NODE_FEATURES == 11


def test_ev_esperado_identico_al_de_b2(datos):
    """ev es el promedio empírico sobre los días de entrenamiento reales (ya no el modelo paramétrico
    mu/s/phi/media_lane): ver baselines.politicas._promedios_empiricos_por_dow."""
    from baselines.politicas import _promedios_empiricos_por_dow
    d = datos
    ev_bruto, _ = _promedios_empiricos_por_dow(d["market"], d["dist"])
    costo = d["dist"] * (d["diesel"] / MPG) + d["dist"] * MARGINAL_COST_SIN_DIESEL
    ev_original = ev_bruto - costo[None]
    assert d["ev"].ev.shape == (7, N, N)
    assert np.array_equal(d["ev"].ev, ev_original)

    b2 = PoliticaB2(d["market"], d["T"], d["dist"], d["diesel"])
    assert np.array_equal(b2._ev, ev_original)


def test_forma_y_finitud(datos):
    env, activas, starts = _estados(datos)
    feats = _features(datos, env, starts)
    assert feats.shape == (len(starts), N, 11) and feats.dtype == np.float32
    assert np.isfinite(feats).all()
    cont = feats[:, :, 6:10]
    assert cont[:, :, 0].min() >= 0                       # f6 = T[j, h] / MAX no se recorta (puede pasar de 1)
    assert cont[:, :, 1].min() >= 0 and cont[:, :, 1].max() <= 1
    assert cont[:, :, [2, 3]].min() >= -1 and cont[:, :, [2, 3]].max() <= 1


@pytest.mark.parametrize("pasos", [0, 1])
def test_f6_a_f9_coinciden_con_referencia_con_loops(datos, pasos):
    env, activas, starts = _estados(datos, pasos=pasos, semilla=pasos)
    feats = _features(datos, env, starts)
    assert len(activas) > 5
    for b in activas:
        np.testing.assert_allclose(feats[b, :, 6:10], _referencia(datos, env, starts, b), atol=1e-6)


def test_f8_coincide_con_el_ev_ajustado_de_b2p_en_5_estados_al_azar(datos):
    env, activas, starts = _estados(datos, B=80, pasos=1, semilla=5)
    feats = _features(datos, env, starts)
    b2p = PoliticaB2P(datos["market"], datos["T"], datos["dist"], datos["diesel"])
    rng = np.random.default_rng(123)
    for b in rng.choice(activas, size=5, replace=False):
        i, h, t = int(env.current_node[b]), int(starts[b]), float(env.time_elapsed[b])
        dow0, hoy, idx = int(env.start_dow[b]), int(t // 14), env.market_index[b]
        for j in range(N):
            if j in (i, h):
                continue
            dia = int((t + datos["T"][i, j]) // 14)                         # día en que sale el arco j→casa
            esperado = np.clip(b2p._ev_dia({}, dow0, hoy, dia, idx)[j, h] / datos["p95"], -1, 1)   # el mismo de B2P
            assert feats[b, j, 8] == pytest.approx(float(esperado), abs=1e-6)


def test_casos_especiales_self_loop_y_casa(datos):
    env, activas, starts = _estados(datos, pasos=1, semilla=3)
    feats = _features(datos, env, starts)
    for b in activas:
        i, h = int(env.current_node[b]), int(starts[b])
        assert (feats[b, i, 6:10] == 0).all()
        assert feats[b, h, 6] == 0 and feats[b, h, 8] == 0 and feats[b, h, 9] == 0
        directo = np.clip((MAX_DURATION - env.time_elapsed[b] - datos["T"][i, h]) / MAX_DURATION, 0, 1)
        assert feats[b, h, 7] == pytest.approx(float(directo), abs=1e-6)
    # en el primer paso i == h: todo en cero para el self-loop
    env0, _, starts0 = _estados(datos, pasos=0, semilla=9)
    f0 = _features(datos, env0, starts0)
    assert (f0[np.arange(len(starts0)), starts0, 6:10] == 0).all()


def test_f9_es_al_menos_f8(datos):
    env, _, starts = _estados(datos, pasos=1, semilla=8)
    feats = _features(datos, env, starts)
    assert (feats[:, :, 9] >= feats[:, :, 8] - 1e-7).all()


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
            ev=datos["ev"], time_elapsed=float(env.time_elapsed[b]), dow0=int(env.start_dow[b]),
            market_index=env.market_index[b], index_std=env.index_std,
        )
        assert esc.shape == (N, 11)
        np.testing.assert_allclose(esc[:, 6:], feats[b, :, 6:], atol=1e-6)


def test_f10_es_el_indice_de_mercado_normalizado(datos):
    """La última columna es índice(j, día actual) / desviación del índice histórico, sin NaN."""
    env, activas, starts = _estados(datos, pasos=1, semilla=4)
    feats = _features(datos, env, starts)
    idx = env.market_index
    assert idx.shape == (len(starts), N) and env.index_std == datos["market"].indice_std
    np.testing.assert_allclose(feats[:, :, 10], idx / env.index_std, atol=1e-6)
    # el índice es el del mundo de cada episodio en su día actual (sin recortar al histórico)
    for b in activas[:10]:
        dia = int(env._start_day_idx[b]) + int(env.time_elapsed[b] // 14)
        assert np.array_equal(idx[b], datos["market"].indice_mundo(int(env.world_seed[b]), dia))


def test_replay_features_forma_sin_nan_e_indice_del_dia_real(datos):
    """Replay de días reales de evaluación: (B, N, 11) sin NaN; f10 es el índice observado del día."""
    d = datos
    tm, rate, loads, dist, diesel, *_ = load_matrices(N)
    ev_dias = rate[TRAIN_DAYS:]
    env = make_replay_env(tm, ev_dias, loads[TRAIN_DAYS:], dist, diesel, N, day_offset=TRAIN_DAYS, market=d["market"])
    B = 12
    rng = np.random.default_rng(0)
    starts = rng.integers(0, N, size=B)
    dias = rng.integers(0, ev_dias.shape[0], size=B)
    env.reset(starts, dias, k_max_arr=[None] * B)
    feats = build_node_features_batch(
        env.current_node, starts, None, d["T"], dist, env.current_day_idx, N, MAX_DURATION,
        reward_global_p95=d["p95"], avail_expected_arr=d["avail"],
        reward_rows=env.reward_rows, ev=d["ev"], time_elapsed=env.time_elapsed, dow0=env.start_dow,
        market_index=env.market_index, index_std=env.index_std,
    )
    assert feats.shape == (B, N, 11) and np.isfinite(feats).all()
    esperado = d["market"].indice_historico(TRAIN_DAYS, None)[dias] / d["market"].indice_std
    np.testing.assert_allclose(feats[:, :, 10], esperado, atol=1e-6)


def test_start_dow_del_entorno(datos):
    for offset in (0, TRAIN_DAYS):
        env = make_sim_env(datos["tm"], datos["rate"], datos["loads"], datos["dist"], datos["diesel"], N,
                           datos["market"])
        env._day_offset = offset
        env.reset(np.array([0, 1, 2]), np.array([3, 10, 50]), world_seeds=np.array([1, 2, 3]))
        env.step(np.array([4, 5, 6]))                      # el tiempo avanza; start_dow no cambia
        assert (env.start_dow == dia_de_semana(np.array([3, 10, 50]) + offset)).all()


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


# ─────────────────────────────────────────────────────────────────────────────
# Valor esperado ajustado por el índice (el de B2P) en las features de continuación
# ─────────────────────────────────────────────────────────────────────────────

# Copia congelada de build_continuation_features_batch ANTES de usar el índice (ev sin ajustar), para comprobar
# que con índice cero las features de continuación valen exactamente lo de siempre.
def _continuacion_sin_indice(
    current_node:      np.ndarray,   # (B,) int64 — i
    start_node:        np.ndarray,   # (B,) int64 — casa h
    time_elapsed:      np.ndarray,   # (B,) horas transcurridas t
    dow0:              np.ndarray,   # (B,) día de la semana (lunes = 0) del día de salida
    time_matrix_arr:   np.ndarray,   # (N, N) T
    ev:                np.ndarray,   # (7, N, N) calcular_ev_esperado
    max_duration:      float,
    reward_global_p95: float,
) -> np.ndarray:                     # (B, N, 4) float32: f7, f8, f9, f10
    """Features de continuación por nodo j (ver docstring del módulo), sin loops sobre B ni N.

    f10 se calcula con un tensor (B, N, N) indexado por (episodio, j, l).
    """
    N    = np.asarray(time_matrix_arr).shape[0]
    T    = np.asarray(time_matrix_arr, dtype=np.float64)
    i    = np.asarray(current_node, dtype=np.int64)
    h    = np.asarray(start_node, dtype=np.int64)
    t    = np.asarray(time_elapsed, dtype=np.float64)
    dow0 = np.asarray(dow0, dtype=np.int64)
    MAX  = max(float(max_duration), 1e-6)
    p95  = max(float(reward_global_p95), 1e-6)
    jj   = np.arange(N)

    t_j  = t[:, None] + T[i, :]                              # (B, N) llegada a j
    T_jh = T[:, h].T                                         # (B, N) tiempo de j a casa
    is_h = jj[None, :] == h[:, None]                         # (B, N)
    is_i = jj[None, :] == i[:, None]                         # (B, N)
    T_jh = np.where(is_h, 0.0, T_jh)                         # casa → casa: sin tiempo

    f7 = T_jh / MAX
    f8 = np.clip((MAX - t_j - T_jh) / MAX, 0.0, 1.0)

    dow_j = (dow0[:, None] + (t_j // 14.0).astype(np.int64)) % 7          # (B, N)
    ev_jh = ev[dow_j, jj[None, :], h[:, None]]                            # (B, N) ev[dow_j, j, h]
    f9 = np.clip(ev_jh / p95, -1.0, 1.0)

    # Continuación a 2 saltos: j → l → casa, con l ∉ {i, j, h} y retorno factible.
    t_l    = t_j[:, :, None] + T[None, :, :]                              # (B, N, N) llegada a l desde j
    T_lh   = T[:, h].T[:, None, :]                                        # (B, 1, N) T[l, h]
    valid  = (
        (jj[None, None, :] != jj[None, :, None])                          # l ≠ j
        & (jj[None, None, :] != i[:, None, None])                         # l ≠ i
        & (jj[None, None, :] != h[:, None, None])                         # l ≠ h
        & (t_l + T_lh <= max_duration)
    )
    dow_l = (dow0[:, None, None] + (t_l // 14.0).astype(np.int64)) % 7    # (B, N, N)
    ev_jl = ev[dow_j[:, :, None], jj[None, :, None], jj[None, None, :]]   # (B, N, N) ev[dow_j, j, l]
    ev_lh = ev[dow_l, jj[None, None, :], h[:, None, None]]                # (B, N, N) ev[dow_l, l, h]
    dos   = np.where(valid, ev_jl + ev_lh, -np.inf).max(axis=2)           # (B, N); -inf si no hay l
    f10 = np.clip(np.maximum(ev_jh, dos) / p95, -1.0, 1.0)

    # Casos especiales: j == h → f7 = f9 = f10 = 0 (f8 = holgura del retorno directo);
    # j == i (self-loop) → las cuatro en 0 (tiene precedencia si i == h).
    f7[is_h] = 0.0
    f9[is_h] = 0.0
    f10[is_h] = 0.0
    for f in (f7, f8, f9, f10):
        f[is_i] = 0.0

    return np.stack([f7, f8, f9, f10], axis=-1).astype(np.float32)


def _argumentos_continuacion(d, env, starts):
    return (env.current_node, starts, env.time_elapsed, env.start_dow, d["T"], None, MAX_DURATION, d["p95"])


@pytest.mark.parametrize("pasos,semilla", [(0, 0), (1, 1), (2, 2)])
def test_con_indice_cero_las_features_de_continuacion_son_identicas_a_las_de_antes(datos, pasos, semilla):
    env, activas, starts = _estados(datos, B=120, pasos=pasos, semilla=semilla)
    cur, st, t, dow0, T, _, mx, p95 = _argumentos_continuacion(datos, env, starts)
    nuevo = build_continuation_features_batch(cur, st, t, dow0, T, datos["ev"], mx, p95, np.zeros((len(st), N)))
    anterior = _continuacion_sin_indice(cur, st, t, dow0, T, datos["ev"].ev, mx, p95)
    assert np.array_equal(nuevo, anterior)


def test_estados_que_cruzan_el_limite_de_un_dia_y_valores_de_h_distintos(datos):
    """Con tiempos cerca de múltiplos de 14 h el h de los arcos cambia: coincide con la referencia con loops,
    y se ven h = 0 y h > 0."""
    env, activas, starts = _estados(datos, B=200, pasos=1, semilla=11)
    t = env.time_elapsed
    cerca = np.flatnonzero((np.abs(t - 14.0 * np.round(t / 14.0)) < 4.0) & (t > 0))
    assert len(cerca) > 3
    feats = _features(datos, env, starts)
    for b in cerca[:10]:
        np.testing.assert_allclose(feats[b, :, 6:10], _referencia(datos, env, starts, b), atol=1e-6)
    llegadas = t[:, None] + datos["T"][env.current_node, :]
    h = (llegadas // 14).astype(int) - (t // 14).astype(int)[:, None]
    assert (h == 0).any() and (h >= 1).any()


def test_subir_el_indice_de_un_nodo_sube_su_valor_esperado_de_volver_a_casa(datos):
    env, activas, starts = _estados(datos, B=60, pasos=1, semilla=6)
    cur, st, t, dow0, T, _, mx, p95 = _argumentos_continuacion(datos, env, starts)
    bajo = build_continuation_features_batch(cur, st, t, dow0, T, datos["ev"], mx, p95, np.full((len(st), N), -0.05))
    alto = build_continuation_features_batch(cur, st, t, dow0, T, datos["ev"], mx, p95, np.full((len(st), N), 0.05))
    assert (alto[:, :, 2] >= bajo[:, :, 2] - 1e-7).all() and (alto[:, :, 3] >= bajo[:, :, 3] - 1e-7).all()
    assert (alto[:, :, 2] > bajo[:, :, 2]).any()


def test_la_tabla_q_es_la_misma_de_b2p(datos):
    b2p = PoliticaB2P(datos["market"], datos["T"], datos["dist"], datos["diesel"])
    assert np.array_equal(b2p._Q, datos["ev"].q) and np.array_equal(b2p._E0, datos["ev"].ev)
    assert np.array_equal(b2p.b, datos["ev"].b) and datos["ev"].q.shape[0] == datos["ev"].h_max + 1
