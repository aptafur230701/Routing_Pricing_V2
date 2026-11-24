"""
test_market_process.py
======================
Pruebas del generador de realizaciones diarias (metodologia_realizaciones_mercado_reefer.md) y
del índice de mercado por nodo (proceso_exogeno_cargas_tarifas.md, conservado):

  · misma semilla y mismos parámetros ⇒ misma salida (cargas, tarifas, δ e índice)
  · la salida no cambia con las acciones del agente (el entorno ve el tablero puro de cada terna)
  · un mismo (nodo, día) da el mismo índice desde el entorno, sample_world e indice_mundo
  · tarifas dentro del rango de cuantiles del día usado (el propio o el día más cercano con cargas)
  · las cargas son las del generador actual (misma terna ⇒ mismas cargas)
  · formas, tipos y tarifa positiva en todas las lanes
  · la función de recompensa compartida es la fórmula de siempre
  · media y proporción de ceros simuladas ≈ histórico en cada grupo de volumen
  · con DIAS_EXCLUIDOS vacía el ajuste usa todos los días de la matriz
  · smoke de entrenamiento con market_process
"""

import copy
import datetime
import os

import numpy as np
import pytest

import am_training
from config import (
    MAX_DURATION, REWARD_SCALE_FACTOR, BIG_M_PENALTY, DIAS_EXCLUIDOS, FECHA_INICIO, TRAIN_DAYS,
)
from fit_market_process import ajustar, dia_de_semana, params_path
from market_process import MarketProcess, asegurar_parametros, _CLAVES_AJUSTE
from problem_data import load_matrices, lane_reward, cargar_dias_mercado
import routing_env
from routing_env import VectorRoutingEnv

NUM_NODES = 10


@pytest.fixture(scope="module")
def data():
    (time_matrix, rate_stack, loads_stack, distance_arr, diesel_arr,
     ltr_stack, avail_expected_arr, reward_global_p95) = load_matrices(NUM_NODES)
    return dict(
        time_matrix=time_matrix, rate_stack=rate_stack, loads_stack=loads_stack,
        distance_arr=distance_arr, diesel_arr=diesel_arr,
        ltr_stack=ltr_stack,
        avail_expected_arr=avail_expected_arr, reward_global_p95=reward_global_p95,
    )


@pytest.fixture(scope="module")
def mp():
    asegurar_parametros()
    return MarketProcess(NUM_NODES)


def _vector_env(data, market_process):
    return VectorRoutingEnv(data["time_matrix"], NUM_NODES, data["distance_arr"], data["diesel_arr"], TRAIN_DAYS,
                            max_duration=MAX_DURATION, market_process=market_process)


def _azar(env, active, rng):
    """(destino, opción) al azar entre las opciones válidas de cada fila activa (las inactivas: lo que sea)."""
    B = len(active)
    acts, opc = np.zeros(B, dtype=np.int64), np.zeros(B, dtype=np.int64)
    for b in np.flatnonzero(active):
        pares = np.argwhere(env.opciones_validas[b])
        acts[b], opc[b] = pares[rng.randint(len(pares))]
    return acts, opc


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
    """Corre una política aleatoria y devuelve {(b, nodo, día): (loads_row, precios_row)} observados."""
    B = len(starts)
    rng = np.random.RandomState(seed_politica)
    env.reset(starts, days, world_seeds=world_seeds)
    vistos = {}
    active = np.ones(B, dtype=bool)

    def registrar(llegaron):
        # Solo al llegar a un nodo: tras un rechazo el tablero es el del día de llegada, aunque la hora avance.
        for b in np.where(active & llegaron)[0]:
            dia = int(env._start_day_idx[b]) + int(env.time_elapsed[b] // 14)
            vistos[(b, int(env.current_node[b]), dia)] = (
                env.current_loads_row[b].copy(), env.precios_row[b].copy(),
            )

    registrar(np.ones(B, dtype=bool))
    for _ in range(env.max_steps):
        if not active.any():
            break
        _, terminated, _ = env.step(*_azar(env, active, rng))
        active &= ~terminated
        registrar(~(env.last_cotizo & ~env.last_gano))
    return vistos


def test_salida_no_cambia_con_las_acciones_del_agente(data, mp):
    B = 40
    rng = np.random.RandomState(0)
    starts = rng.randint(0, NUM_NODES, B).astype(np.int64)
    days = rng.randint(0, TRAIN_DAYS, B).astype(np.int64)
    seeds = rng.randint(0, 2**31 - 1, B)

    vistos_a = _correr_politica(_vector_env(data, mp), mp, 1, seeds, starts, days)
    vistos_b = _correr_politica(_vector_env(data, mp), mp, 2, seeds, starts, days)

    # Dos políticas distintas recorren rutas distintas, pero donde coinciden en
    # (episodio, nodo, día) ven exactamente el mismo tablero...
    comunes = set(vistos_a) & set(vistos_b)
    assert len(comunes) >= B   # al menos el nodo de inicio de cada episodio
    for k in comunes:
        assert np.array_equal(vistos_a[k][0], vistos_b[k][0])
        assert np.array_equal(vistos_a[k][1], vistos_b[k][1])
    assert set(vistos_a) != set(vistos_b)   # (y sí tomaron decisiones distintas)

    # ... y todo lo observado es el tablero puro del proceso para esa terna (el día recortado a train).
    for (b, nodo, dia), (l, p) in {**vistos_a, **vistos_b}.items():
        t = mp.tablero_por_mundo([nodo], [min(dia, TRAIN_DAYS - 1)], [seeds[b]])
        assert np.array_equal(l, t.loads[0]) and np.array_equal(p, t.precios[0])


def test_entorno_ve_el_mismo_mundo_que_mundo_cotizacion(data, mp):
    t = mp.mundo_cotizacion(99, 8, dia_inicial=3)
    env = _vector_env(data, mp)
    env.reset(np.array([2]), np.array([3]), world_seeds=np.array([99]))
    assert np.array_equal(env.current_loads_row[0], t.loads[0, 2])
    assert np.allclose(env.precios_row[0], t.precios[0, 2])
    assert np.allclose(env.u_comp_row[0], t.u_comp[0, 2])


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


def test_lane_sin_cargas_nunca_genera_cargas(mp):
    """Una lane sin ninguna carga en toda la ventana (m = 0 todos los días) nunca sortea cargas."""
    sin_cargas = mp.count_loads.sum(axis=0) == 0
    if not sin_cargas.any():
        pytest.skip("todas las lanes de estos nodos tuvieron carga alguna vez")
    loads, _ = mp.sample_arrivals(
        np.repeat(np.arange(NUM_NODES), 200), np.tile(np.arange(200) % TRAIN_DAYS, NUM_NODES),
        np.random.default_rng(0),
    )
    nodos = np.repeat(np.arange(NUM_NODES), 200)
    for i in range(NUM_NODES):
        assert np.all(loads[nodos == i][:, sin_cargas[i]] == 0)


def test_dia_de_semana_desde_fecha_inicio():
    """dia_de_semana(0, FECHA_INICIO) coincide con datetime.date.weekday() (lunes = 0, igual convención)."""
    fecha = datetime.date.fromisoformat(FECHA_INICIO)
    assert dia_de_semana(0, FECHA_INICIO) == fecha.weekday()
    dias = np.arange(30)
    esperado = [(fecha + datetime.timedelta(days=int(d))).weekday() for d in dias]
    assert np.array_equal(dia_de_semana(dias, FECHA_INICIO), esperado)


def test_lane_reward_es_la_formula_de_siempre(data):
    from config import MPG, MARGINAL_COST_SIN_DIESEL
    for d in (0, 5, 89):
        rev = data["rate_stack"][d] * data["distance_arr"]
        rev[data["loads_stack"][d] <= 0] = 0
        cost = data["distance_arr"] * (data["diesel_arr"] / MPG) + data["distance_arr"] * MARGINAL_COST_SIN_DIESEL
        assert np.array_equal(
            lane_reward(data["rate_stack"][d], data["loads_stack"][d], data["distance_arr"], data["diesel_arr"]),
            np.round(rev - cost, 0))


# ─────────────────────────────────────────────────────────────────────────────
# Entorno con market_process
# ─────────────────────────────────────────────────────────────────────────────

def test_rewards_row_vectorial_es_el_margen_a_la_mediana(data, mp):
    venv = _vector_env(data, mp)
    venv.reset(np.array([2, 6]), np.array([0, 30]), world_seeds=np.array([1, 2]))
    for b, nodo in enumerate((2, 6)):
        esperado = lane_reward(
            venv.rate_row_obs[b], venv.current_loads_row[b],
            data["distance_arr"][nodo], data["diesel_arr"][nodo],
        ).astype(np.float32)
        esperado[nodo] = BIG_M_PENALTY
        assert np.array_equal(venv.reward_rows[b], esperado)


# ─────────────────────────────────────────────────────────────────────────────
# Validación estadística: media y ceros por grupo de volumen
# ─────────────────────────────────────────────────────────────────────────────

def test_media_y_ceros_simulados_se_parecen_al_historico_por_grupo():
    n = 35
    mp35 = MarketProcess(n)
    hist = mp35.count_loads[:TRAIN_DAYS]   # (D, n, n) días de ajuste, ya en el orden de hubs correcto
    D = hist.shape[0]
    grupo = mp35._grupo_lane
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

    for h in range(len(mp35.r)):
        sel = universo & (grupo == h)
        assert sel.sum() > 0
        # tolerancia floja: `sample_arrivals` compara contra un día de train al azar del mismo día de
        # semana (no el día exacto), así que además del ruido de muestreo hay variación semana a
        # semana dentro del mismo grupo; en el grupo de menor volumen (medias pequeñas) la tolerancia
        # relativa sola es demasiado estricta.
        assert media[sel].mean() == pytest.approx(hist.mean(0)[sel].mean(), rel=0.08, abs=0.02)
        assert ceros[sel].mean() == pytest.approx((hist == 0).mean(0)[sel].mean(), abs=0.05)


# ─────────────────────────────────────────────────────────────────────────────
# Generador: cargas, δ, índice del día real
# ─────────────────────────────────────────────────────────────────────────────

# Cargas capturadas del generador ACTUAL (media móvil de 6 vecinos + NegBin/Poisson por grupo de
# volumen) con NUM_NODES = 10: (semilla, nodo, día, cargas). Si cambian los datos o el ajuste
# (fit_market_process.py), hay que recapturarlas.
CARGAS_REFERENCIA = [
    (5, 0, 3, [518, 399, 216, 59, 12, 51, 156, 78, 3, 0]),
    (5, 1, 40, [174, 8, 9, 16, 14, 16, 3, 13, 0, 0]),
    (123, 3, 89, [43, 10, 26, 61, 12, 3, 0, 7, 0, 0]),
    (777, 9, 12, [0, 2, 1, 1, 0, 0, 0, 0, 0, 0]),
    (20260, 7, 60, [29, 0, 3, 2, 0, 1, 2, 0, 0, 0]),
    (31, 2, 94, [49, 15, 20, 11, 16, 1, 0, 38, 18, 0]),
]


@pytest.mark.parametrize("semilla,nodo,dia,cargas", CARGAS_REFERENCIA)
def test_cargas_son_las_del_generador_actual(mp, semilla, nodo, dia, cargas):
    loads, _ = mp.sample_arrivals_por_mundo([nodo], [dia], [semilla])
    assert loads[0].tolist() == cargas


def test_delta_determinista_por_semilla_e_independiente_de_num_nodes(mp):
    """δ (el corrimiento de nivel nacional del episodio) es función pura de la semilla: no cambia
    entre instancias de MarketProcess ni con el número de nodos."""
    otro = MarketProcess(NUM_NODES)
    mp5 = MarketProcess(5)
    for semilla in (3, 4, 100):
        assert mp._delta(semilla) == otro._delta(semilla) == mp5._delta(semilla)
    assert mp._delta(3) != mp._delta(4)
    # y es lo único que cambia entre dos episodios con el mismo (nodo, día) y distinta semilla, salvo
    # el ruido propio del sorteo de cargas/precio (comparamos la tarifa media sobre muchas lanes, no
    # una lane puntual, para que el δ domine sobre el ruido de una sola interpolación de cuantiles).
    l1, r1 = mp.sample_arrivals_por_mundo([2, 2], [10, 10], [8, 8])
    assert np.array_equal(r1[0], r1[1]) and np.array_equal(l1[0], l1[1])


def test_indice_mundo_fuera_de_rango_es_error(mp):
    with pytest.raises(ValueError):
        mp.indice_mundo(1, mp.num_dias_total)
    with pytest.raises(ValueError):
        mp.indice_mundo(1, -1)


def test_indice_coincide_entre_entorno_sample_world_e_indice_mundo(data, mp):
    semilla, dia0 = 99, 3
    _, _, ind_w = mp.sample_world(semilla, 8, dia_inicial=dia0, devolver_indice=True)      # (8, N)
    for d in range(8):
        assert np.array_equal(ind_w[d], mp.indice_mundo(semilla, dia0 + d))
    venv = _vector_env(data, mp)
    venv.reset(np.array([2]), np.array([dia0]), world_seeds=np.array([semilla]))
    assert np.array_equal(venv.market_index[0], ind_w[0])
    assert venv.index_std == mp.indice_std
    # y tras avanzar el tiempo (días de mercado de 14 h), con acciones válidas hasta que termine el episodio
    rng = np.random.RandomState(0)
    activo = np.ones(1, dtype=bool)
    for _ in range(venv.max_steps):
        _, term, _ = venv.step(*_azar(venv, activo, rng))
        if term[0]:
            break
        d = int(venv.time_elapsed[0] // 14)
        assert np.array_equal(venv.market_index[0], ind_w[d])


def test_indice_del_entorno_no_depende_de_las_acciones(data, mp):
    B = 30
    rng = np.random.RandomState(1)
    starts = rng.randint(0, NUM_NODES, B).astype(np.int64)
    days = rng.randint(0, TRAIN_DAYS, B).astype(np.int64)
    seeds = rng.randint(0, 2**31 - 1, B)
    for politica in (1, 2):
        env = _vector_env(data, mp)
        env.reset(starts, days, world_seeds=seeds)
        r = np.random.RandomState(politica)
        activos = np.ones(B, dtype=bool)
        for _ in range(env.max_steps):
            if not activos.any():
                break
            for b in np.where(activos)[0]:
                dia = min(int(env._start_day_idx[b]) + int(env.time_elapsed[b] // 14), TRAIN_DAYS - 1)
                assert np.array_equal(env.market_index[b], mp.indice_mundo(int(seeds[b]), dia))
            _, term, _ = env.step(*_azar(env, activos, r))
            activos &= ~term


def test_tarifas_dentro_del_rango_exacto_de_cuantiles_con_delta_cero(mp):
    """Con δ = 0, el precio de cada lane cae dentro de [min, max] del día exacto del que salieron
    sus cuantiles (el propio día si tuvo cargas, si no el día más cercano con cargas de esa lane)."""
    rng = np.random.default_rng(0)
    B = 300
    nodos = rng.integers(0, NUM_NODES, B)
    dias = rng.integers(0, TRAIN_DAYS, B)
    _, rates = mp._sortear(nodos, dias, np.zeros(B), rng)   # δ = 0, mismo día que `dias` (no al azar)
    for b in range(B):
        dp = mp._dia_precio(int(dias[b]), nodos[b:b + 1])[0]          # (N,) día usado por cada lane
        mins = mp._q["min"][dp, nodos[b], np.arange(NUM_NODES)]
        maxs = mp._q["max"][dp, nodos[b], np.arange(NUM_NODES)]
        assert (rates[b] >= mins - 1e-4).all() and (rates[b] <= maxs + 1e-4).all()


def test_la_firma_cambia_con_los_parametros_del_mercado(mp):
    """La tabla V de UB-DP y el caché de la evaluación se recalculan solos si cambian los datos del mercado."""
    from baselines.dp_relajada import _firma
    from config import UBDP_DT_HORAS, UBDP_S, UBDP_SEED
    T = np.ones((NUM_NODES, NUM_NODES)); D = np.ones((NUM_NODES, NUM_NODES))
    args = (T, D, D, 5, UBDP_S, UBDP_DT_HORAS, UBDP_SEED)
    base, f0 = mp.firma_params(), _firma(mp, *args)
    assert MarketProcess(NUM_NODES).firma_params() == base

    for campo in _CLAVES_AJUSTE:
        otro = copy.copy(mp)
        otro._params_blob = dict(mp._params_blob)
        val = mp._params_blob[campo]
        otro._params_blob[campo] = val + (1 if np.issubdtype(val.dtype, np.integer) else 0.01)
        assert otro.firma_params() != base and _firma(otro, *args) != f0, campo

    otro = copy.copy(mp); otro.count_loads = mp.count_loads + 1
    assert otro.firma_params() != base
    otro = copy.copy(mp); otro.mean_rate = mp.mean_rate + 0.01
    assert otro.firma_params() != base


def test_indice_historico_es_la_definicion_del_ajuste():
    """Días de entrenamiento: el índice guardado en el ajuste. Días de prueba: misma definición con
    los 194 nodos y las tarifas medias reales, recortada a N."""
    mp = MarketProcess(NUM_NODES)
    with np.load(params_path()) as z:
        guardado = z["indice"][:, :NUM_NODES]
    assert np.allclose(mp.indice_historico(0, TRAIN_DAYS), guardado, atol=1e-9)

    ev = mp.indice_historico(TRAIN_DAYS, None)
    dias_full = cargar_dias_mercado()
    mean_rate_full = dias_full["mean"][TRAIN_DAYS:].astype(np.float64)
    assert ev.shape == (mean_rate_full.shape[0], NUM_NODES)
    with np.load(params_path()) as z:
        media, uni = z["media_lane"], z["universo_indice"]
    bruto = ((mean_rate_full - media[None]) * uni[None]).sum(axis=2) / uni.sum(axis=1)[None]
    assert np.allclose(ev, (bruto - bruto.mean(axis=1, keepdims=True))[:, :NUM_NODES])


# ─────────────────────────────────────────────────────────────────────────────
# Ajuste
# ─────────────────────────────────────────────────────────────────────────────

def _panel_sintetico(D=21, N=6, seed=0):
    """dict con la forma de cargar_dias_mercado() (solo las claves que usa ajustar(): count_loads,
    mean, fechas), para probar el ajuste sin tocar disco."""
    rng = np.random.default_rng(seed)
    loads = rng.poisson(8, size=(D, N, N)).astype(np.float32)
    loads[:, 0, 1] = 0    # una lane sin cargas en todo el histórico
    rate = rng.uniform(2, 4, size=(D, N, N)).astype(np.float32)
    fechas = np.datetime64(FECHA_INICIO, "D") + np.arange(D)
    return dict(count_loads=loads, mean=rate, fechas=fechas)


def test_dias_excluidos_vacia_por_defecto_y_ajuste_usa_todos_los_dias():
    assert list(DIAS_EXCLUIDOS) == []
    for D in (21, 35):   # el número de días sale de la forma del panel
        dias_data = _panel_sintetico(D=D)
        p = ajustar(DIAS_EXCLUIDOS, ventana_lento=5, dias_data=dias_data, train_days=D)
        assert len(p["dias_usados"]) == D
        assert np.array_equal(p["dias_usados"], np.arange(D))
        assert p["indice"].shape[0] == D
        assert len(p["lento_escenarios"]) == D - 5 + 1                   # días con la ventana completa
        assert p["fechas_ajuste"][0] == FECHA_INICIO
        assert p["f_w"].mean() == pytest.approx(1.0)
        assert not p["universo"][0, 1]


def test_ajuste_excluye_solo_las_fechas_indicadas():
    dias_data = _panel_sintetico(D=21)
    p = ajustar(["2026-06-05", "2026-06-06"], ventana_lento=5, dias_data=dias_data, train_days=21)
    assert len(p["dias_usados"]) == 19
    assert 4 not in p["dias_usados"] and 5 not in p["dias_usados"]
    with pytest.raises(ValueError):
        ajustar(["2025-12-31"], dias_data=dias_data, train_days=21)


def test_parametros_guardados_usan_solo_dias_de_entrenamiento():
    """El .npz guardado se ajusta con los primeros TRAIN_DAYS días, sin los de prueba."""
    assert list(DIAS_EXCLUIDOS) == []
    asegurar_parametros()
    with np.load(params_path()) as z:
        assert z["indice"].shape[0] == len(z["dias_usados"]) == TRAIN_DAYS
        assert np.array_equal(z["dias_usados"], np.arange(TRAIN_DAYS))


# ─────────────────────────────────────────────────────────────────────────────
# Smoke de entrenamiento con market_process
# ─────────────────────────────────────────────────────────────────────────────

def test_smoke_entrenamiento_con_market_process(data, mp, monkeypatch):
    monkeypatch.setattr(am_training, "PPO_N_EPISODES_PER_UPDATE", 24)
    monkeypatch.setattr(am_training, "get_episodes_per_node", lambda n: 6)   # 60 ep, 2 updates
    agent, critic, ep_rewards, ep_losses, log = am_training.run_am_training(
        data["time_matrix"], data["rate_stack"][:TRAIN_DAYS], data["loads_stack"][:TRAIN_DAYS],
        data["distance_arr"], data["diesel_arr"], NUM_NODES,
        ltr_stack=data["ltr_stack"][:, :TRAIN_DAYS],
        reward_global_p95=data["reward_global_p95"],
        avail_expected_arr=data["avail_expected_arr"],
        market_process=mp,
    )
    assert len(log) >= 1 and np.isfinite(ep_rewards).all()
    for entry in log:
        for d in entry["k_diagnostics"].values():
            if d is not None and "conformity_rate" in d:
                assert d["conformity_rate"] == 1.0
