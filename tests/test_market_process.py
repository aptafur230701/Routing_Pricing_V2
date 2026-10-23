"""
test_market_process.py
======================
Pruebas del proceso exógeno de cargas y tarifas (proceso_exogeno_cargas_tarifas.md):

  · misma semilla y mismos parámetros ⇒ misma salida (cargas, tarifas, lento e índice del mundo)
  · la salida no cambia con las acciones del agente
  · el lento es fijo durante el episodio; el rápido arranca en su distribución estacionaria
  · un mismo (nodo, día) da el mismo índice desde el entorno, sample_world e indice_mundo
  · tarifas: nunca por debajo del mínimo histórico de la lane; sample_arrivals con los mismos marginales
  · las cargas son las del generador anterior (misma terna ⇒ mismas cargas)
  · formas, tipos y tarifa positiva en todas las lanes
  · con market_process=None el entorno reproduce las recompensas de siempre
  · la función de recompensa compartida coincide con rm_pen_stack
  · media y proporción de ceros simuladas ≈ histórico en cada grupo de volumen h
  · con DIAS_EXCLUIDOS vacía el ajuste usa todos los días de la matriz
  · smoke de entrenamiento con market_process
"""

import os

import numpy as np
import pytest

import am_training
from config import (
    MAX_DURATION, REWARD_SCALE_FACTOR, BIG_M_PENALTY, DIAS_EXCLUIDOS, FECHA_INICIO, TRAIN_DAYS,
    MARKET_DIA_MAX,
)
from fit_market_process import ajustar, dia_de_semana, cargar_matrices_completas, params_path
from market_process import MarketProcess, asegurar_parametros
from problem_data import load_matrices, build_rm_pen_stack, build_day_matrices, lane_reward
import routing_env
from routing_env import RoutingEnv, VectorRoutingEnv

NUM_NODES = 10


@pytest.fixture(scope="module")
def data():
    (time_matrix, rate_stack, loads_stack, distance_arr, diesel_arr,
     ltr_stack, trucks_stack, avail_expected_arr, reward_global_p95) = load_matrices(NUM_NODES)
    return dict(
        time_matrix=time_matrix, rate_stack=rate_stack, loads_stack=loads_stack,
        distance_arr=distance_arr, diesel_arr=diesel_arr,
        ltr_stack=ltr_stack, trucks_stack=trucks_stack,
        avail_expected_arr=avail_expected_arr, reward_global_p95=reward_global_p95,
    )


@pytest.fixture(scope="module")
def mp():
    asegurar_parametros()
    return MarketProcess(NUM_NODES)


def _scalar_env(data, market_process=None):
    return RoutingEnv(
        data["time_matrix"], data["rate_stack"], data["loads_stack"],
        data["distance_arr"], data["diesel_arr"], NUM_NODES, MAX_DURATION,
        market_process=market_process,
    )


def _vector_env(data, market_process=None, rm_pen_stack=None):
    return VectorRoutingEnv(
        time_matrix=data["time_matrix"], rate_stack=data["rate_stack"],
        loads_stack=data["loads_stack"], rm_pen_stack=rm_pen_stack,
        num_nodes=NUM_NODES, max_duration=MAX_DURATION,
        market_process=market_process,
        distance_arr=data["distance_arr"], diesel_arr=data["diesel_arr"],
    )


# ─────────────────────────────────────────────────────────────────────────────
# Reproducibilidad
# ─────────────────────────────────────────────────────────────────────────────

def test_misma_semilla_misma_salida(mp):
    nodos = np.arange(NUM_NODES).repeat(3)
    dias  = np.tile([0, 5, 40], NUM_NODES)

    a = mp.sample_arrivals(nodos, dias, np.random.default_rng(7))
    b = mp.sample_arrivals(nodos, dias, np.random.default_rng(7))
    c = mp.sample_arrivals(nodos, dias, np.random.default_rng(8))
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])
    assert not np.array_equal(a[0], c[0])

    w1 = mp.sample_world(123, 6, dia_inicial=2)
    w2 = mp.sample_world(123, 6, dia_inicial=2)
    w3 = mp.sample_world(124, 6, dia_inicial=2)
    assert np.array_equal(w1[0], w2[0]) and np.array_equal(w1[1], w2[1])
    assert not np.array_equal(w1[0], w3[0])


def test_sorteo_por_mundo_no_depende_del_lote(mp):
    """Cada fila es función pura de (semilla, nodo, día): invertir o recortar el lote no la cambia."""
    nodos = np.array([0, 3, 3, 7, 9]); dias = np.array([0, 1, 2, 3, 4]); sem = np.array([5, 5, 6, 7, 7])
    l, r = mp.sample_arrivals_por_mundo(nodos, dias, sem)
    l_inv, r_inv = mp.sample_arrivals_por_mundo(nodos[::-1], dias[::-1], sem[::-1])
    assert np.array_equal(l, l_inv[::-1]) and np.array_equal(r, r_inv[::-1])
    l_sub, r_sub = mp.sample_arrivals_por_mundo(nodos[2:3], dias[2:3], sem[2:3])
    assert np.array_equal(l[2], l_sub[0]) and np.array_equal(r[2], r_sub[0])


def _correr_politica(env, mp, seed_politica, world_seeds, starts, days):
    """Corre una política aleatoria y devuelve {(b, nodo, día): (loads_row, rate_row)} observados."""
    B = len(starts)
    rng = np.random.RandomState(seed_politica)
    masks = env.reset(starts, days, world_seeds=world_seeds)
    vistos = {}
    active = np.ones(B, dtype=bool)

    def registrar():
        for b in np.where(active)[0]:
            dia = int(env._start_day_idx[b]) + int(env.time_elapsed[b] // 14)
            vistos[(b, int(env.current_node[b]), dia)] = (
                env.current_loads_row[b].copy(), env.current_rate_row[b].copy(),
            )

    registrar()
    for _ in range(NUM_NODES):
        if not active.any():
            break
        actions = np.array([rng.choice(np.where(m == 1)[0]) for m in masks])
        _, terminated, masks = env.step(actions)
        active &= ~terminated
        registrar()
    return vistos


def test_salida_no_cambia_con_las_acciones_del_agente(data, mp):
    B = 40
    rng = np.random.RandomState(0)
    starts = rng.randint(0, NUM_NODES, B).astype(np.int64)
    days = rng.randint(0, 90, B).astype(np.int64)
    seeds = rng.randint(0, 2**31 - 1, B)

    vistos_a = _correr_politica(_vector_env(data, mp), mp, 1, seeds, starts, days)
    vistos_b = _correr_politica(_vector_env(data, mp), mp, 2, seeds, starts, days)

    # Dos políticas distintas recorren rutas distintas, pero donde coinciden en
    # (episodio, nodo, día) ven exactamente las mismas cargas y tarifas...
    comunes = set(vistos_a) & set(vistos_b)
    assert len(comunes) >= B   # al menos el nodo de inicio de cada episodio
    for k in comunes:
        assert np.array_equal(vistos_a[k][0], vistos_b[k][0])
        assert np.array_equal(vistos_a[k][1], vistos_b[k][1])
    assert set(vistos_a) != set(vistos_b)   # (y sí tomaron decisiones distintas)

    # ... y todo lo observado es el sorteo puro del proceso para esa terna.
    for (b, nodo, dia), (l, r) in {**vistos_a, **vistos_b}.items():
        l_ref, r_ref = mp.sample_arrivals_por_mundo([nodo], [dia], [seeds[b]])
        assert np.array_equal(l, l_ref[0]) and np.array_equal(r, r_ref[0])


def test_entorno_ve_el_mismo_mundo_que_sample_world(data, mp):
    w_loads, w_rate = mp.sample_world(99, 8, dia_inicial=3)
    env = _scalar_env(data, mp)
    env.reset(seed=0, options={"start_node": 2, "start_day_idx": 3, "world_seed": 99})
    assert np.array_equal(env.current_loads_row, w_loads[0, 2])
    assert np.allclose(env.current_rate_row, w_rate[0, 2])


# ─────────────────────────────────────────────────────────────────────────────
# Formas, tipos, positividad
# ─────────────────────────────────────────────────────────────────────────────

def test_formas_tipos_y_tarifa_positiva(mp):
    B = 64
    rng = np.random.default_rng(0)
    nodos = rng.integers(0, NUM_NODES, B); dias = rng.integers(0, TRAIN_DAYS, B)

    loads, rates = mp.sample_arrivals(nodos, dias, rng)
    for l, r in (
        (loads, rates),
        mp.sample_arrivals_por_mundo(nodos, dias, np.arange(B)),
    ):
        assert l.shape == (B, NUM_NODES) and r.shape == (B, NUM_NODES)
        assert np.issubdtype(l.dtype, np.integer) and l.min() >= 0
        assert np.issubdtype(r.dtype, np.floating)
        assert np.all(r > 0)                       # positiva en TODAS las lanes
        assert np.isfinite(r).all()

    lr, rr = mp.llegada_a_nodo(3, 10, np.random.default_rng(0))
    assert lr.shape == (NUM_NODES,) and rr.shape == (NUM_NODES,) and np.all(rr > 0)

    w_loads, w_rate = mp.sample_world(1, 5)
    assert w_loads.shape == (5, NUM_NODES, NUM_NODES) == w_rate.shape
    assert np.all(w_rate > 0)


def test_lane_sin_cargas_historicas_no_genera_cargas(mp):
    sin_cargas = mp.params.mu == 0
    if not sin_cargas.any():
        pytest.skip("todas las lanes de estos nodos tienen cargas históricas")
    loads, _ = mp.sample_arrivals(
        np.repeat(np.arange(NUM_NODES), 200), np.tile(np.arange(200) % 120, NUM_NODES),
        np.random.default_rng(0),
    )
    nodos = np.repeat(np.arange(NUM_NODES), 200)
    for i in range(NUM_NODES):
        assert np.all(loads[nodos == i][:, sin_cargas[i]] == 0)


def test_dia_de_semana_desde_fecha_inicio():
    dias = np.arange(120)
    assert FECHA_INICIO == "2026-01-01"                      # jueves
    assert np.array_equal(dia_de_semana(dias), (dias + 3) % 7)


# ─────────────────────────────────────────────────────────────────────────────
# Entorno: market_process=None reproduce lo de siempre
# ─────────────────────────────────────────────────────────────────────────────

# Recompensas capturadas del entorno ANTES de eliminar la regla de no revisitar (git tag pre-revisit),
# con los tiempos de viaje actuales de datos/duration.csv, la política fija indicada y
# NUM_NODES=10: (start_node, start_day_idx, k_max, acciones, recompensas). Si datos/duration.csv
# cambia, hay que recapturarlas con ese entorno.
RUTAS_REFERENCIA = [
    (3, 7,  None, [4, 3],       [6.02, 6.9361588215709125]),
    (0, 0,  5,    [1, 0],       [-0.93, 18.315254567221622]),
    (8, 40, None, [1, 8],       [-16.41, 15.652876498008624]),
    (5, 89, 5,    [3, 5],       [-2.88, 15.142161752486055]),
]


@pytest.fixture
def con_shaping(monkeypatch):
    """Las referencias de arriba se capturaron con el bono de retorno (USE_REWARD_SHAPING = True)."""
    monkeypatch.setattr(routing_env, "USE_REWARD_SHAPING", True)


@pytest.mark.parametrize("start,day,k,acciones,esperadas", RUTAS_REFERENCIA)
def test_market_none_reproduce_recompensas_de_antes(con_shaping, data, start, day, k, acciones, esperadas):
    env = _scalar_env(data, market_process=None)
    env.reset(seed=0, options={"start_node": start, "start_day_idx": day, "k_max": k})
    rewards = [env.step(a)[1] for a in acciones]
    assert rewards == pytest.approx(esperadas, abs=1e-9)


@pytest.mark.parametrize("start,day,k,acciones,esperadas", RUTAS_REFERENCIA)
def test_vector_env_market_none_coincide_con_referencia(con_shaping, data, start, day, k, acciones, esperadas):
    rm_pen = build_rm_pen_stack(data["rate_stack"], data["loads_stack"],
                                data["distance_arr"], data["diesel_arr"])
    env = _vector_env(data, None, rm_pen)
    env.reset(np.array([start]), np.array([day]), k_max_arr=[k])
    rewards = [float(env.step(np.array([a]))[0][0]) for a in acciones]
    assert rewards == pytest.approx(esperadas, abs=1e-5)   # float32


@pytest.mark.parametrize("start,day,k,acciones,esperadas", RUTAS_REFERENCIA)
def test_sin_shaping_el_retorno_exitoso_es_solo_margen(monkeypatch, data, start, day, k, acciones, esperadas):
    """USE_REWARD_SHAPING = False: el último paso (retorno a casa, a tiempo) no suma bono alguno:
    recompensa = lane / REWARD_SCALE_FACTOR, tanto en RoutingEnv como en VectorRoutingEnv."""
    rm_pen = build_rm_pen_stack(data["rate_stack"], data["loads_stack"],
                                data["distance_arr"], data["diesel_arr"])
    monkeypatch.setattr(routing_env, "USE_REWARD_SHAPING", False)
    env = _scalar_env(data, market_process=None)
    env.reset(seed=0, options={"start_node": start, "start_day_idx": day, "k_max": k})
    venv = _vector_env(data, None, rm_pen)
    venv.reset(np.array([start]), np.array([day]), k_max_arr=[k])
    for a in acciones:
        i, d = env.current_node, env.current_day_idx
        r_lane = float(rm_pen[d, i, a]) / REWARD_SCALE_FACTOR
        r = env.step(a)[1]
        rv = float(venv.step(np.array([a]))[0][0])
        assert r == pytest.approx(r_lane, abs=1e-9)
        assert rv == pytest.approx(r_lane, abs=1e-5)


def test_lane_reward_coincide_con_rm_pen_stack(data):
    rm_pen = build_rm_pen_stack(data["rate_stack"], data["loads_stack"],
                                data["distance_arr"], data["diesel_arr"])
    for d in (0, 17, 89):
        for i in (0, 4, 9):
            fila = lane_reward(
                data["rate_stack"][d, i], data["loads_stack"][d, i],
                data["distance_arr"][i], data["diesel_arr"][i],
            ).astype(np.float32)
            fila[i] = BIG_M_PENALTY
            assert np.array_equal(fila, rm_pen[d, i])
    # y build_day_matrices (que ahora la llama) sigue dando lo mismo que la fórmula original
    d = 5
    rev = data["rate_stack"][d] * data["distance_arr"]
    rev[data["loads_stack"][d] <= 0] = 0
    from config import MPG, MARGINAL_COST_SIN_DIESEL
    cost = data["distance_arr"] * (data["diesel_arr"] / MPG) + data["distance_arr"] * MARGINAL_COST_SIN_DIESEL
    assert np.array_equal(build_day_matrices(
        data["rate_stack"][d], data["loads_stack"][d], data["distance_arr"], data["diesel_arr"])[0],
        np.round(rev - cost, 0))


# ─────────────────────────────────────────────────────────────────────────────
# Entorno con market_process
# ─────────────────────────────────────────────────────────────────────────────

def test_mascara_y_recompensa_usan_las_filas_sorteadas(data, mp):
    env = _scalar_env(data, mp)
    _, info = env.reset(seed=0, options={"start_node": 1, "start_day_idx": 10, "world_seed": 5})
    rng = np.random.RandomState(0)
    for _ in range(30):
        mask = info["action_mask"]
        nodo_actual = env.current_node
        loads, rates = env.current_loads_row, env.current_rate_row
        if nodo_actual != env.start_node:
            # lanes sin carga ⇒ fuera de la máscara (salvo el retorno al depot)
            sin_carga = (loads == 0)
            sin_carga[env.start_node] = False
            assert np.all(mask[sin_carga] == 0)
        accion = int(rng.choice(np.where(mask == 1)[0]))
        esperado = lane_reward(
            rates[accion], loads[accion],
            data["distance_arr"][nodo_actual, accion], data["diesel_arr"][nodo_actual, accion],
        ) / REWARD_SCALE_FACTOR
        _, r, term, _, info = env.step(accion)
        if not term:
            assert r == pytest.approx(float(esperado), abs=1e-6)   # sin bono/penalización terminal
        else:
            break


def test_escalar_y_vectorial_coinciden_con_el_mismo_mundo(data, mp):
    starts = np.array([0, 3, 7]); days = np.array([4, 50, 88]); seeds = np.array([11, 22, 33])
    venv = _vector_env(data, mp)
    masks = venv.reset(starts, days, world_seeds=seeds)
    senvs = []
    for b in range(3):
        e = _scalar_env(data, mp)
        e.reset(seed=0, options={"start_node": int(starts[b]), "start_day_idx": int(days[b]),
                                 "world_seed": int(seeds[b])})
        senvs.append(e)
    activos = [True] * 3
    rng = np.random.RandomState(3)
    for _ in range(NUM_NODES):
        acts = np.array([rng.choice(np.where(m == 1)[0]) for m in masks])
        rewards, term, masks = venv.step(acts)
        for b in range(3):
            if not activos[b]:
                continue
            _, r, t, _, info = senvs[b].step(int(acts[b]))
            assert r == pytest.approx(float(rewards[b]), abs=1e-5)
            assert t == bool(term[b])
            if t:
                activos[b] = False
        if not any(activos):
            break


def test_rewards_row_vectorial_es_la_fila_observada(data, mp):
    venv = _vector_env(data, mp)
    venv.reset(np.array([2, 6]), np.array([0, 30]), world_seeds=np.array([1, 2]))
    for b, nodo in enumerate((2, 6)):
        esperado = lane_reward(
            venv.current_rate_row[b], venv.current_loads_row[b],
            data["distance_arr"][nodo], data["diesel_arr"][nodo],
        ).astype(np.float32)
        esperado[nodo] = BIG_M_PENALTY
        assert np.array_equal(venv.reward_rows[b], esperado)


# ─────────────────────────────────────────────────────────────────────────────
# Validación estadística: media y ceros por grupo de volumen h
# ─────────────────────────────────────────────────────────────────────────────

def test_media_y_ceros_simulados_se_parecen_al_historico_por_grupo():
    n = 35
    mp35 = MarketProcess(n)
    hist = np.load(os.path.join(os.path.dirname(__file__), "..", "datos", "load_availability.npy"))[:TRAIN_DAYS, :n, :n]   # días de ajuste
    D = hist.shape[0]
    grupo = mp35.params.grupo
    universo = hist.sum(0) > 0

    rng = np.random.default_rng(1)
    nodos = np.repeat(np.arange(n), D); dias = np.tile(np.arange(D), n)
    reps = 100
    media = np.zeros((n, n)); ceros = np.zeros((n, n))
    for _ in range(reps):
        l, _r = mp35.sample_arrivals(nodos, dias, rng)
        l = l.reshape(n, D, n)
        media += l.mean(axis=1); ceros += (l == 0).mean(axis=1)
    media /= reps; ceros /= reps

    for h in range(len(mp35.params.phi)):
        sel = universo & (grupo == h)
        assert sel.sum() > 0
        assert media[sel].mean() == pytest.approx(hist.mean(0)[sel].mean(), rel=0.03)
        assert ceros[sel].mean() == pytest.approx((hist == 0).mean(0)[sel].mean(), abs=0.03)


# ─────────────────────────────────────────────────────────────────────────────
# Modelo de tarifas: lento, rápido, índice del mundo
# ─────────────────────────────────────────────────────────────────────────────

# Cargas capturadas del generador ANTERIOR (remuestreo por lane) con NUM_NODES = 10: (semilla, nodo, día, cargas).
# Las cargas no cambian con el modelo de tarifas: misma terna ⇒ mismas cargas.
CARGAS_REFERENCIA = [
    (5, 0, 3, [215, 82, 73, 34, 8, 1, 19, 28, 1, 0]),
    (5, 1, 40, [256, 191, 53, 7, 39, 16, 3, 69, 10, 0]),
    (123, 3, 89, [18, 4, 18, 40, 27, 3, 0, 7, 0, 0]),
    (777, 9, 12, [0, 15, 0, 0, 0, 0, 0, 0, 0, 0]),
    (20260, 7, 60, [161, 25, 66, 11, 2, 0, 6, 30, 2, 0]),
    (31, 2, 94, [38, 15, 109, 26, 5, 7, 7, 32, 7, 0]),
]


@pytest.mark.parametrize("semilla,nodo,dia,cargas", CARGAS_REFERENCIA)
def test_cargas_son_las_del_generador_anterior(mp, semilla, nodo, dia, cargas):
    loads, _ = mp.sample_arrivals_por_mundo([nodo], [dia], [semilla])
    assert loads[0].tolist() == cargas


def test_mundo_determinista_por_semilla_nodo_dia(mp):
    """lento, índice y filas son función pura de la semilla (y del nodo y día), también con otro caché."""
    otro = MarketProcess(NUM_NODES)
    for semilla in (3, 4):
        assert np.array_equal(mp.lento_mundo(semilla), otro.lento_mundo(semilla))
        for dia in (0, 17, MARKET_DIA_MAX):
            assert np.array_equal(mp.indice_mundo(semilla, dia), otro.indice_mundo(semilla, dia))
    assert not np.array_equal(mp.indice_mundo(3, 5), mp.indice_mundo(4, 5))
    l1, r1 = mp.sample_arrivals_por_mundo([2, 2], [10, 10], [8, 8])
    assert np.array_equal(r1[0], r1[1]) and np.array_equal(l1[0], l1[1])
    # el índice de un nodo no depende de cuántos nodos tenga la corrida (el mundo se genera con los 194)
    mp5 = MarketProcess(5)
    assert np.array_equal(mp5.indice_mundo(3, 12), mp.indice_mundo(3, 12)[:5])


def test_indice_mundo_fuera_del_mundo_es_error(mp):
    with pytest.raises(ValueError):
        mp.indice_mundo(1, MARKET_DIA_MAX + 1)
    with pytest.raises(ValueError):
        mp.indice_mundo(1, -1)


def test_lento_es_el_mismo_todo_el_episodio(mp):
    """El lento del mundo es la fila de un día de escenario y no cambia con el día ni con las llamadas."""
    for semilla in (1, 2, 3):
        lento = mp.lento_mundo(semilla).copy()
        assert any(np.array_equal(lento, fila) for fila in mp.params.lento_escenarios)
        for dia in range(0, MARKET_DIA_MAX + 1, 7):
            mp.indice_mundo(semilla, dia)
            mp.sample_arrivals_por_mundo([0, 1], [dia, dia], [semilla, semilla])
        assert np.array_equal(mp.lento_mundo(semilla), lento)


def test_rapido_estacionario_y_con_la_persistencia_estimada():
    """índice − lento, entre mundos, tiene desviación σ_i / sqrt(1 − ρ²) desde el día 0 y autocorrelación ρ."""
    mp = MarketProcess(NUM_NODES)
    W = 1500
    rapido = np.stack([mp._mundo(s)[1] - mp._mundo(s)[0] for s in range(W)])    # (W, días, N)
    sd = mp.params.sigma / np.sqrt(1 - mp.params.rho ** 2)
    for dia in (0, 1, 30, MARKET_DIA_MAX):
        np.testing.assert_allclose(rapido[:, dia].std(axis=0), sd, rtol=0.12)
    x, y = rapido[:, 40:-1].ravel(), rapido[:, 41:].ravel()
    assert (x * y).sum() / (x * x).sum() == pytest.approx(mp.params.rho, abs=0.03)


def test_indice_coincide_entre_entorno_sample_world_e_indice_mundo(data, mp):
    semilla, dia0 = 99, 3
    _, _, ind_w = mp.sample_world(semilla, 8, dia_inicial=dia0, devolver_indice=True)      # (8, N)
    for d in range(8):
        assert np.array_equal(ind_w[d], mp.indice_mundo(semilla, dia0 + d))
    # entorno escalar y vectorial: mismo índice en el día de salida y en los días siguientes
    senv = _scalar_env(data, mp)
    senv.reset(seed=0, options={"start_node": 2, "start_day_idx": dia0, "world_seed": semilla})
    venv = _vector_env(data, mp)
    venv.reset(np.array([2]), np.array([dia0]), world_seeds=np.array([semilla]))
    assert np.array_equal(senv.market_index, ind_w[0]) and np.array_equal(venv.market_index[0], ind_w[0])
    assert senv.index_std == venv.index_std == mp.params.indice_std
    # y tras avanzar el tiempo (días de mercado de 14 h), con acciones válidas hasta que termine el episodio
    rng = np.random.RandomState(0)
    mask = senv._get_action_mask()
    for _ in range(NUM_NODES):
        destino = int(rng.choice(np.where(mask == 1)[0]))
        _, _, terminado, _, info = senv.step(destino)
        _, _, masks = venv.step(np.array([destino]))
        if terminado:
            break
        d = int(senv.time_elapsed // 14)
        assert np.array_equal(senv.market_index, ind_w[d]) and np.array_equal(venv.market_index[0], ind_w[d])
        mask = info["action_mask"]


def test_indice_del_entorno_no_depende_de_las_acciones(data, mp):
    B = 30
    rng = np.random.RandomState(1)
    starts = rng.randint(0, NUM_NODES, B).astype(np.int64)
    days = rng.randint(0, TRAIN_DAYS, B).astype(np.int64)
    seeds = rng.randint(0, 2**31 - 1, B)
    for politica in (1, 2):
        env = _vector_env(data, mp)
        masks = env.reset(starts, days, world_seeds=seeds)
        r = np.random.RandomState(politica)
        activos = np.ones(B, dtype=bool)
        for _ in range(NUM_NODES):
            if not activos.any():
                break
            for b in np.where(activos)[0]:
                dia = int(env._start_day_idx[b]) + int(env.time_elapsed[b] // 14)
                assert np.array_equal(env.market_index[b], mp.indice_mundo(int(seeds[b]), dia))
            acts = np.array([r.choice(np.where(m == 1)[0]) for m in masks])
            _, term, masks = env.step(acts)
            activos &= ~term


def test_tarifas_no_bajan_del_minimo_historico_y_se_cuentan_los_recortes(mp):
    mp.reiniciar_recorte()
    for semilla in range(30):
        _, rates = mp.sample_arrivals_por_mundo(
            np.arange(NUM_NODES), np.full(NUM_NODES, semilla % TRAIN_DAYS), np.full(NUM_NODES, semilla))
        assert (rates.astype(np.float64) >= mp.params.min_lane - 1e-12).all()
    rng = np.random.default_rng(0)
    nodos = rng.integers(0, NUM_NODES, 300)
    _, rates = mp.sample_arrivals(nodos, rng.integers(0, TRAIN_DAYS, 300), rng)
    assert (rates.astype(np.float64) >= mp.params.min_lane[nodos] - 1e-12).all()
    est = mp.estadisticas_recorte()
    assert est["tarifas"] == (30 * NUM_NODES + 300) * NUM_NODES
    assert 0 < est["recortadas"] < est["tarifas"]                  # el piso se activa en una fracción pequeña
    assert 0 <= est["pct_universo"] < 15


def test_sample_arrivals_tiene_los_mismos_marginales_que_los_mundos(mp):
    """Sin memoria pero con la misma distribución de la tarifa en un (nodo, día) cualquiera."""
    n = 3000
    nodo, dia = 4, 20
    _, r_mundos = mp.sample_arrivals_por_mundo(np.full(n, nodo), np.full(n, dia), np.arange(n))
    _, r_llegadas = mp.sample_arrivals(np.full(n, nodo), np.full(n, dia), np.random.default_rng(0))
    uni = mp._universo_lane[nodo]
    for f in (np.mean, np.std):
        a, b = f(r_mundos[:, uni], axis=0), f(r_llegadas[:, uni], axis=0)
        assert np.median(np.abs(a - b) / np.abs(a)) < 0.03


def test_la_firma_cambia_con_los_parametros_del_mercado(mp):
    """La tabla V de UB-DP y el caché de la evaluación se recalculan solos si cambian los datos del mercado."""
    import copy
    from dp_relajada import _firma
    from config import UBDP_DT_HORAS, UBDP_S, UBDP_SEED
    T = np.ones((NUM_NODES, NUM_NODES)); D = np.ones((NUM_NODES, NUM_NODES))
    args = (T, D, D, 5, UBDP_S, UBDP_DT_HORAS, UBDP_SEED)
    base, f0 = mp.firma_params(), _firma(mp, *args)
    assert MarketProcess(NUM_NODES).firma_params() == base
    for campo, cambio in (("media_lane", lambda a: a + 0.01), ("sigma", lambda a: a * 1.1),
                          ("residuos", lambda a: a + np.float32(0.01)), ("lento_escenarios", lambda a: a + 0.01),
                          ("min_lane", lambda a: a - 0.01), ("mu", lambda a: a * 1.1)):
        otro = copy.copy(mp); otro.params = copy.copy(mp.params)
        setattr(otro.params, campo, cambio(getattr(mp.params, campo)))
        assert otro.firma_params() != base and _firma(otro, *args) != f0, campo
    otro = copy.copy(mp); otro.params = copy.copy(mp.params); otro.params.rho = mp.params.rho + 0.01
    assert otro.firma_params() != base


def test_indice_historico_es_la_definicion_del_ajuste():
    """Días de entrenamiento: el índice guardado en el ajuste. Días de evaluación: misma definición con
    los 194 nodos y las tarifas reales, recortada a N."""
    mp = MarketProcess(NUM_NODES)
    with np.load(params_path()) as z:
        guardado = z["indice"][:, :NUM_NODES]
    assert np.allclose(mp.indice_historico(0, TRAIN_DAYS), guardado, atol=1e-9)
    ev = mp.indice_historico(TRAIN_DAYS, None)
    rate = np.load(os.path.join(os.path.dirname(__file__), "..", "datos", "rate.npy")).astype(np.float64)[TRAIN_DAYS:]
    assert ev.shape == (rate.shape[0], NUM_NODES)
    with np.load(params_path()) as z:
        media, uni = z["media_lane"], z["universo_indice"]
    bruto = ((rate - media[None]) * uni[None]).sum(axis=2) / uni.sum(axis=1)[None]
    assert np.allclose(ev, (bruto - bruto.mean(axis=1, keepdims=True))[:, :NUM_NODES])


# ─────────────────────────────────────────────────────────────────────────────
# Ajuste
# ─────────────────────────────────────────────────────────────────────────────

def _panel_sintetico(D=21, N=6, seed=0):
    rng = np.random.default_rng(seed)
    loads = rng.poisson(8, size=(D, N, N)).astype(float)
    loads[:, 0, 1] = 0    # una lane sin cargas en todo el histórico
    rate = rng.uniform(2, 4, size=(D, N, N))
    return loads, rate


def test_dias_excluidos_vacia_por_defecto_y_ajuste_usa_todos_los_dias():
    assert list(DIAS_EXCLUIDOS) == []
    for D in (21, 35):   # el número de días sale de la forma de la matriz
        loads, rate = _panel_sintetico(D=D)
        p = ajustar(loads, rate, FECHA_INICIO, DIAS_EXCLUIDOS, ventana_lento=5)
        assert len(p["dias_usados"]) == D
        assert np.array_equal(p["dias_usados"], np.arange(D))
        assert p["residuos"].shape[0] == p["indice"].shape[0] == D
        assert len(p["lento_escenarios"]) == D - 5 + 1                   # días con la ventana completa
        assert p["fechas_ajuste"][0] == FECHA_INICIO
        assert p["s"].mean() == pytest.approx(1.0)
        assert p["mu"][0, 1] == 0 and not p["universo"][0, 1]


def test_ajuste_excluye_solo_las_fechas_indicadas():
    loads, rate = _panel_sintetico(D=21)
    p = ajustar(loads, rate, FECHA_INICIO, ["2026-01-05", "2026-01-06"], ventana_lento=5)
    assert len(p["dias_usados"]) == 19
    assert 4 not in p["dias_usados"] and 5 not in p["dias_usados"]
    with pytest.raises(ValueError):
        ajustar(loads, rate, FECHA_INICIO, ["2025-12-31"])


def test_parametros_guardados_usan_solo_dias_de_entrenamiento():
    """El .npz guardado se ajusta con los primeros TRAIN_DAYS días, sin los de evaluación."""
    assert list(DIAS_EXCLUIDOS) == []
    asegurar_parametros()
    loads, rate = cargar_matrices_completas(TRAIN_DAYS)
    assert loads.shape[0] == TRAIN_DAYS
    with np.load(params_path()) as z:
        assert z["residuos"].shape[0] == z["indice"].shape[0] == len(z["dias_usados"]) == TRAIN_DAYS
        assert np.array_equal(z["dias_usados"], np.arange(TRAIN_DAYS))


# ─────────────────────────────────────────────────────────────────────────────
# Smoke de entrenamiento con market_process
# ─────────────────────────────────────────────────────────────────────────────

def test_smoke_entrenamiento_con_market_process(data, mp, monkeypatch):
    monkeypatch.setattr(am_training, "PPO_N_EPISODES_PER_UPDATE", 24)
    monkeypatch.setattr(am_training, "get_episodes_per_node", lambda n: 6)   # 60 ep, 2 updates
    agent, critic, ep_rewards, ep_losses, log = am_training.run_am_training(
        data["time_matrix"], data["rate_stack"][:90], data["loads_stack"][:90],
        data["distance_arr"], data["diesel_arr"], NUM_NODES,
        ltr_stack=data["ltr_stack"], trucks_stack=data["trucks_stack"],
        reward_global_p95=data["reward_global_p95"],
        avail_expected_arr=data["avail_expected_arr"],
        market_process=mp,
    )
    assert len(log) >= 1 and np.isfinite(ep_rewards).all()
    for entry in log:
        for d in entry["k_diagnostics"].values():
            if d is not None and "conformity_rate" in d:
                assert d["conformity_rate"] == 1.0
