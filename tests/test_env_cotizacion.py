"""
test_env_cotizacion.py
======================
VectorRoutingEnv con cotización, espera y reposicionamiento (especificacion_fase0_ruteo_pricing.md §4,
tests 5-7):

  · máscara = opciones_validas.any(-1); niveles solo en lanes j ≠ i con carga cotizable; VACÍO a j ≠ i solo
    para reposicionarse (≤ REPOSICION_MAX_MILLAS) o volver a casa desde un intermedio; VACÍO en j = i = esperar
  · ganar con el nivel k ⇔ u_k < u_comp; ningún nivel gana siempre
  · un rechazo no consume K, avanza COTIZACION_HORAS, no mueve el camión, deja el tablero igual y
    quita la cotización de esa lane (test 5)
  · después de COTIZACIONES_MAX_POR_NODO rechazos solo quedan VACÍO: casa, reposicionarse o esperar (test 6)
  · esperar salta al siguiente día de mercado, no consume K, revela el tablero del día nuevo y reinicia los
    rechazos
  · sin jugadas en reset() ⇒ no sale (margen 0, válido); sin jugadas después ⇒ el ciclo falla
  · margen = Σ lane_reward con el precio cotizado, en episodios completos (test 7)
  · determinismo y números aleatorios comunes; replay con cotización; opción inválida ⇒ error
"""

import numpy as np
import pytest

from config import (COTIZACION_HORAS, COTIZACIONES_MAX_POR_NODO, MAX_DURATION, TRAIN_DAYS, K_MAX,
                    REPOSICION_MAX_MILLAS, HORAS_DIA_MERCADO, TIME_VIOLATION_PENALTY)
from market_process import MarketProcess, asegurar_parametros
from pricing import N_NIVELES, U_NIVELES_ARR, lane_cost_row
from problem_data import lane_reward, load_matrices
from routing_env import OPCION_VACIO, VectorRoutingEnv, fin_espera

NUM_NODES = 10
B = 64
SEMILLAS = np.arange(1000, 1000 + B)


@pytest.fixture(scope="module")
def data():
    tm, rate, loads, dist, diesel, *_ = load_matrices(NUM_NODES)
    return dict(time_matrix=np.asarray(tm, dtype=float), rate=rate, loads=loads, dist=dist, diesel=diesel)


@pytest.fixture(scope="module")
def mp():
    asegurar_parametros()
    return MarketProcess(NUM_NODES)


def _env(data, mp, replay=False, n=NUM_NODES):
    if replay:
        return VectorRoutingEnv(
            data["time_matrix"], n, data["dist"], data["diesel"], data["rate"].shape[0] - TRAIN_DAYS,
            mercado_replay=mp, day_offset=TRAIN_DAYS,
        )
    return VectorRoutingEnv(data["time_matrix"], n, data["dist"], data["diesel"], TRAIN_DAYS, market_process=mp)


def _reset(env, k=None, semillas=SEMILLAS):
    n = len(semillas)
    return env.reset(np.arange(n) % env.num_nodes, np.arange(n) % 20, k_max_arr=[k] * n, world_seeds=semillas)


def _accion_al_azar(env, rng):
    """(destino, opción) uniforme entre las opciones válidas de cada fila."""
    op = env.opciones_validas
    acts, opc = np.zeros(env._B, dtype=np.int64), np.zeros(env._B, dtype=np.int64)
    for b in range(env._B):
        pares = np.argwhere(op[b])
        j, o = pares[rng.integers(len(pares))]
        acts[b], opc[b] = j, o
    return acts, opc


def _primera_opcion(env):
    """(destino, opción) de la primera opción válida de cada fila."""
    pares = [np.argwhere(env.opciones_validas[b])[0] for b in range(env._B)]
    return np.array([p[0] for p in pares]), np.array([p[1] for p in pares])


def _nivel_que_gana(env, b, j):
    """El nivel más bajo que gana la lane j de la fila b (conociendo el competidor), o None."""
    gana = np.flatnonzero(env.opciones_validas[b, j, :N_NIVELES] & (U_NIVELES_ARR < env.u_comp_row[b, j]))
    return int(gana[0]) if len(gana) else None


def _alejarse_de_casa(env):
    """Un paso en el que cada fila que puede gana una lane a un nodo intermedio (la más cercana donde algún
    nivel gana); las demás esperan si pueden. Devuelve las filas que quedaron en un nodo intermedio."""
    acts, opc = _primera_opcion(env)
    for b in range(env._B):
        i = env.current_node[b]
        if env.opciones_validas[b, i, OPCION_VACIO]:
            acts[b], opc[b] = i, OPCION_VACIO
        mejor = None
        for j in np.argsort(env._time_matrix_arr[i]):
            if j == env.start_node[b]:
                continue
            k = _nivel_que_gana(env, b, j)
            if k is not None:
                mejor = (j, k)
                break
        if mejor is not None:
            acts[b], opc[b] = mejor
    env.step(acts, opc)
    fuera = np.flatnonzero(env._active & (env.current_node != env.start_node))
    assert len(fuera) > env._B // 4, "pocas filas llegaron a un nodo intermedio"
    return fuera


def _lane_rechazable(env, b):
    """Destino j ≠ i con cotización válida donde el nivel más alto pierde (u_comp ≤ u_9), o None."""
    for j in range(env.num_nodes):
        if (j != env.current_node[b] and env.opciones_validas[b, j, N_NIVELES - 1]
                and env.u_comp_row[b, j] <= U_NIVELES_ARR[-1]):
            return j
    return None


def _paso_rechazando(env, filas=None):
    """Un step() donde cada fila activa (de `filas`) con una lane rechazable cotiza el nivel más alto ahí (y
    pierde); el resto espera si puede o toma su primera opción válida. Devuelve (casos, estado previo, salida)."""
    acts, opc = _primera_opcion(env)
    casos = []
    for b in (np.flatnonzero(env._active) if filas is None else filas):
        j = _lane_rechazable(env, b)
        if j is not None:
            acts[b], opc[b] = j, N_NIVELES - 1
            casos.append((b, j))
    antes = dict(nodo=env.current_node.copy(), t=env.time_elapsed.copy(), k=env.k_remaining.copy(),
                 precios=env.precios_row.copy(), loads=env.current_loads_row.copy(), perdidas=env.perdidas.copy())
    return casos, antes, env.step(acts, opc)


def _verificar_rechazo(env, casos, antes, salida):
    rewards, terminated, masks = salida
    for b, j in casos:
        assert env.last_cotizo[b] and not env.last_gano[b]
        assert env.current_node[b] == antes["nodo"][b]                                   # no se mueve
        assert env.time_elapsed[b] == pytest.approx(antes["t"][b] + COTIZACION_HORAS)
        assert env.k_remaining[b] == antes["k"][b]                                       # no consume K
        assert env.last_raw_reward[b] == 0.0
        if terminated[b]:                                                                # sin jugadas o fuera de tiempo
            continue
        assert rewards[b] == 0.0
        np.testing.assert_array_equal(env.precios_row[b], antes["precios"][b])           # mismo tablero
        np.testing.assert_array_equal(env.current_loads_row[b], antes["loads"][b])
        # Se pierde UNA carga: la lane se puede volver a cotizar si le quedan cargas (y hay tiempo e intentos).
        assert env.perdidas[b, j] == antes["perdidas"][b, j] + 1
        if env.current_loads_row[b, j] <= env.perdidas[b, j] or env.n_rechazos[b] >= COTIZACIONES_MAX_POR_NODO:
            assert not env.opciones_validas[b, j, :N_NIVELES].any()
        elif antes["t"][b] + COTIZACION_HORAS + 1.0 < MAX_DURATION - env._time_matrix_arr[env.current_node[b], j] \
                - (0.0 if j == env.start_node[b] else env._time_matrix_arr[j, env.start_node[b]]):
            assert env.opciones_validas[b, j, :N_NIVELES].all()
        i, h = env.current_node[b], env.start_node[b]
        puede_ir_vacio = (j == h and i != h) or env._reposicion[i, j]
        assert env.opciones_validas[b, j, OPCION_VACIO] == puede_ir_vacio


# ── Máscara y opciones ───────────────────────────────────────────────────────

def test_mascara_es_alguna_opcion_valida(data, mp):
    env = _env(data, mp)
    masks = _reset(env)
    op = env.opciones_validas
    assert op.shape == (B, NUM_NODES, N_NIVELES + 1)
    act = env.active
    np.testing.assert_array_equal(masks[act].astype(bool), op[act].any(axis=-1))
    filas = np.arange(B)
    i = env.current_node
    # Niveles: solo lanes j ≠ i con carga.
    con_carga = env.current_loads_row > 0
    assert not (op[:, :, :N_NIVELES].any(-1) & ~con_carga)[act].any()
    assert not op[filas, i, :N_NIVELES].any()
    # En casa (antes de salir), VACÍO a j ≠ i solo si es reposicionamiento (≤ 300 mi; a 10 nodos, ninguno).
    vacio_otro = op[:, :, OPCION_VACIO].copy()
    vacio_otro[filas, i] = False
    assert not (vacio_otro & ~env._reposicion[i])[act].any()
    assert not env._reposicion.any()                                                     # a 10 nodos, nada ≤ 300 mi


def test_sin_salida_es_exogeno_y_valido(data, mp):
    """La casa 9 a 10 nodos (ida y vuelta mínima 72.7 h): sin carga cotizable no tiene ninguna jugada."""
    env = _env(data, mp)
    sem = np.arange(5000, 5200)
    env.reset(np.full(200, 9), np.arange(200) % 100, world_seeds=sem)
    sin = env.sin_salida
    assert sin.any() and not sin.all()
    assert not env.active[sin].any() and env.active[~sin].all()
    assert not env.last_time_violation.any()


def test_en_nodo_intermedio_vacio_hacia_casa_o_esperar(data, mp):
    env = _env(data, mp)
    _reset(env)
    fuera = _alejarse_de_casa(env)
    op = env.opciones_validas
    for b in fuera:
        h, i = env.start_node[b], env.current_node[b]
        vacio = set(np.flatnonzero(op[b, :, OPCION_VACIO]))
        assert h in vacio                                                                # volver siempre se puede
        assert vacio <= {h, i} | set(np.flatnonzero(env._reposicion[i]))
        espera_ok = fin_espera(env.time_elapsed[b]) + data["time_matrix"][i, h] <= MAX_DURATION
        assert (i in vacio) == espera_ok


def test_reposicionamiento_solo_a_300_millas(mp):
    """A 20 nodos hay pares a ≤ 300 mi: VACÍO a j ≠ i fuera de casa solo hacia ellos (o hacia casa)."""
    tm, rate, loads, dist, diesel, *_ = load_matrices(20)
    mp20 = MarketProcess(20)
    env = VectorRoutingEnv(np.asarray(tm, dtype=float), 20, dist, diesel, TRAIN_DAYS, market_process=mp20)
    env.reset(np.arange(40) % 20, np.arange(40) % 20, world_seeds=np.arange(40))
    op = env.opciones_validas
    for b in np.flatnonzero(env.active):
        i = env.current_node[b]
        vacio = np.flatnonzero(op[b, :, OPCION_VACIO])
        otros = vacio[vacio != i]
        assert np.all(np.asarray(dist)[i, otros] <= REPOSICION_MAX_MILLAS)
    assert any(op[b, :, OPCION_VACIO].sum() > 1 for b in range(40))                     # alguien puede reposicionarse


# ── Ganar y perder ───────────────────────────────────────────────────────────

def test_gana_si_y_solo_si_queda_bajo_el_competidor(data, mp):
    env = _env(data, mp)
    _reset(env)
    rng = np.random.default_rng(3)
    acts, opc = _primera_opcion(env)
    for b in np.flatnonzero(env.active):
        pares = np.argwhere(env.opciones_validas[b, :, :N_NIVELES])
        if len(pares):
            acts[b], opc[b] = pares[rng.integers(len(pares))]
    u_comp = env.u_comp_row[np.arange(B), acts].copy()
    act = env.active.copy()
    env.step(acts, opc)
    cot = act & (opc < N_NIVELES)
    esperado = U_NIVELES_ARR[np.minimum(opc, N_NIVELES - 1)] < u_comp
    np.testing.assert_array_equal(env.last_gano[cot], esperado[cot])
    np.testing.assert_array_equal(env.last_cotizo[act], cot[act])


def test_ganar_cobra_el_precio_del_nivel(data, mp):
    env = _env(data, mp)
    _reset(env, k=K_MAX)
    acts, opc = _primera_opcion(env)
    filas = []
    for b in np.flatnonzero(env.active):
        for j in range(NUM_NODES):
            k = _nivel_que_gana(env, b, j)
            if k is not None:
                acts[b], opc[b] = j, k
                filas.append(b)
                break
    assert filas
    origen, precios = env.current_node.copy(), env.precios_row.copy()
    env.step(acts, opc)
    for b in filas:
        i, j, k = origen[b], acts[b], opc[b]
        assert env.last_gano[b] and env.current_node[b] == j
        esperado = lane_reward(precios[b, j, k], 1, data["dist"][i, j], data["diesel"][i, j])
        assert env.last_raw_reward[b] == pytest.approx(float(esperado))
        assert env.k_remaining[b] == K_MAX - 1 and env.n_rechazos[b] == 0


def test_volver_a_casa_vacio_paga_solo_el_costo(data, mp):
    env = _env(data, mp)
    _reset(env)
    fuera = _alejarse_de_casa(env)
    acts, opc = _primera_opcion(env)
    acts[fuera], opc[fuera] = env.start_node[fuera], OPCION_VACIO
    origen = env.current_node.copy()
    env.step(acts, opc)
    costo = lane_cost_row(data["dist"][origen, acts], data["diesel"][origen, acts])
    np.testing.assert_allclose(env.last_raw_reward[fuera], -np.round(costo[fuera]), atol=0.5)
    assert not env.last_cotizo[fuera].any() and (env.current_node[fuera] == env.start_node[fuera]).all()


def test_rechazo_en_casa(data, mp):
    """Test 5 en casa, antes de salir."""
    env = _env(data, mp)
    _reset(env, k=K_MAX)
    casos, antes, salida = _paso_rechazando(env)
    assert len(casos) > B // 4
    _verificar_rechazo(env, casos, antes, salida)
    assert all(env.n_rechazos[b] == 1 for b, _ in casos)


def test_rechazo_en_nodo_intermedio(data, mp):
    env = _env(data, mp)
    _reset(env, k=K_MAX)
    fuera = _alejarse_de_casa(env)
    casos, antes, salida = _paso_rechazando(env, fuera)
    assert casos, "ninguna fila tiene una lane donde el nivel más alto pierda"
    _verificar_rechazo(env, casos, antes, salida)


def test_cada_carga_de_la_lane_tiene_su_competidor(data, mp):
    """Al perder, la siguiente cotización de la lane se enfrenta al competidor de la carga siguiente; con una sola
    carga la lane se cierra."""
    env = _env(data, mp)
    _reset(env)
    casos, antes, _ = _paso_rechazando(env)
    vistos = 0
    for b, j in casos:
        if not env.active[b]:
            continue
        assert env.u_comp_row[b, j] == env.u_comp_cargas[b, j, 1]
        vistos += env.u_comp_cargas[b, j, 0] != env.u_comp_cargas[b, j, 1]
        esperado = min(antes["loads"][b, j] - 1, COTIZACIONES_MAX_POR_NODO - 1)
        assert env.intentos[b, j] == esperado
    assert vistos
    b, j = next((b, j) for b, j in casos if env.active[b])
    env._current_loads_row[b, j] = 1                                                     # la única que había se perdió
    env._compute_masks()
    assert not env.opciones_validas[b, j, :N_NIVELES].any()


def test_tope_de_cotizaciones_en_nodo_intermedio(data, mp):
    """Test 6 lejos de casa: con el tope alcanzado solo quedan VACÍO (casa y, si se puede, esperar)."""
    env = _env(data, mp)
    _reset(env)
    fuera = _alejarse_de_casa(env)
    env._n_rechazos[fuera] = COTIZACIONES_MAX_POR_NODO
    masks = env._compute_masks()
    for b in fuera:
        op = env.opciones_validas[b]
        assert not op[:, :N_NIVELES].any()
        assert op[env.start_node[b], OPCION_VACIO] and masks[b, env.start_node[b]]


# ── Esperar ──────────────────────────────────────────────────────────────────

def test_esperar_salta_al_dia_siguiente_con_tablero_nuevo(data, mp):
    env = _env(data, mp)
    _reset(env, k=K_MAX)
    puede = np.flatnonzero(env.active & env.opciones_validas[np.arange(B), env.current_node, OPCION_VACIO])
    assert len(puede) > B // 2
    # Primero un rechazo, para ver que la espera los reinicia.
    _paso_rechazando(env, puede)
    puede = [b for b in puede if env.active[b] and env.opciones_validas[b, env.current_node[b], OPCION_VACIO]]
    acts, opc = _primera_opcion(env)
    for b in puede:
        acts[b], opc[b] = env.current_node[b], OPCION_VACIO
    antes = dict(nodo=env.current_node.copy(), t=env.time_elapsed.copy(), k=env.k_remaining.copy(),
                 dia=env.current_day_idx.copy(), u=env.u_comp_row.copy())
    rewards, terminated, _ = env.step(acts, opc)
    cambio = 0
    for b in puede:
        assert env.last_espero[b] and not env.last_cotizo[b]
        assert env.current_node[b] == antes["nodo"][b] and env.k_remaining[b] == antes["k"][b]
        assert env.last_raw_reward[b] == 0.0 and rewards[b] == 0.0 and not terminated[b]
        assert env.time_elapsed[b] == pytest.approx(fin_espera(antes["t"][b]))
        assert env.time_elapsed[b] % HORAS_DIA_MERCADO == pytest.approx(0.0, abs=1e-4)
        assert env.current_day_idx[b] == min(antes["dia"][b] + 1, TRAIN_DAYS - 1)
        assert env.n_rechazos[b] == 0 and not env.perdidas[b].any()
        cambio += not np.array_equal(env.u_comp_row[b], antes["u"][b])
    assert cambio > len(puede) // 2                                                      # tablero del día nuevo
    assert (env.step_count[puede] == 0).all()                                            # no cuenta como arco


def test_quedarse_sin_jugadas_despues_es_falla(data, mp):
    """Casa 9 (ida y vuelta mínima 72.7 h): si tenía carga cotizable y la pierde, no puede esperar ni
    reposicionarse: el ciclo falla con la penalización (no es 'no sale')."""
    env = _env(data, mp)
    sem = np.arange(5000, 5200)
    env.reset(np.full(200, 9), np.arange(200) % 100, world_seeds=sem)
    filas = [b for b in np.flatnonzero(env.active) if _lane_rechazable(env, b) is not None]
    assert filas
    fallan = []
    for _ in range(COTIZACIONES_MAX_POR_NODO):
        activas = [b for b in filas if env.active[b]]
        casos, _, (rewards, terminated, _) = _paso_rechazando(env, activas)
        for b, _j in casos:
            if terminated[b] and env.step_count[b] == 0:                                # terminó en casa sin salir
                assert env.last_time_violation[b] and not env.sin_salida[b]
                assert rewards[b] == pytest.approx(TIME_VIOLATION_PENALTY)
                fallan.append(b)
    assert fallan


def test_opcion_invalida_lanza_error(data, mp):
    env = _env(data, mp)
    _reset(env)
    b = int(np.flatnonzero(env.active)[0])
    malo_j, malo_o = np.argwhere(~env.opciones_validas[b])[0]
    acts, opc = _primera_opcion(env)
    acts[b], opc[b] = malo_j, malo_o
    with pytest.raises(ValueError):
        env.step(acts, opc)


# ── Episodios completos ──────────────────────────────────────────────────────

def _episodio_al_azar(env, data, rng, k=None, semillas=SEMILLAS):
    """Corre episodios al azar y devuelve (margen por fila, margen recalculado a mano, pasos, rewards)."""
    _reset(env, k=k, semillas=semillas)
    n = len(semillas)
    margen, a_mano = np.zeros(n), np.zeros(n)
    traza = []
    for paso in range(env.max_steps):
        if not env._active.any():
            break
        act = env._active.copy()
        origen, precios = env.current_node.copy(), env.precios_row.copy()
        acts, opc = _accion_al_azar(env, rng)
        rewards, terminated, _ = env.step(acts, opc)
        traza.append((acts.copy(), opc.copy(), rewards.copy()))
        margen += np.where(act, env.last_raw_reward, 0.0)
        for b in np.flatnonzero(act):
            i, j, o = origen[b], acts[b], opc[b]
            if o == OPCION_VACIO and i == j:
                pass                                                                     # esperar: reward 0
            elif o == OPCION_VACIO:
                a_mano[b] += lane_reward(0.0, 0, data["dist"][i, j], data["diesel"][i, j])
            elif env.last_gano[b]:
                a_mano[b] += lane_reward(precios[b, j, o], 1, data["dist"][i, j], data["diesel"][i, j])
    return margen, a_mano, paso, traza


@pytest.mark.parametrize("k", [None, K_MAX])
def test_margen_es_suma_de_lane_reward_con_el_precio_cotizado(data, mp, k):
    """Test 7 de la especificación, con políticas al azar sobre las opciones válidas."""
    env = _env(data, mp)
    margen, a_mano, pasos, _ = _episodio_al_azar(env, data, np.random.default_rng(11), k=k)
    np.testing.assert_allclose(margen, a_mano)
    assert not env._active.any() and pasos < env.max_steps                             # todos terminan
    assert np.all(env.time_elapsed <= MAX_DURATION + 24)


def test_mismas_semillas_y_acciones_dan_lo_mismo(data, mp):
    m1, _, _, t1 = _episodio_al_azar(_env(data, mp), data, np.random.default_rng(5))
    m2, _, _, t2 = _episodio_al_azar(_env(data, mp), data, np.random.default_rng(5))
    np.testing.assert_array_equal(m1, m2)
    for (a1, o1, r1), (a2, o2, r2) in zip(t1, t2):
        np.testing.assert_array_equal(a1, a2)
        np.testing.assert_array_equal(r1, r2)


def test_misma_cotizacion_en_el_mismo_mundo_da_el_mismo_resultado(data, mp):
    """Números aleatorios comunes: dos métodos que cotizan igual en el mismo mundo ganan o pierden igual."""
    resultados = []
    for _ in range(2):
        env = _env(data, mp)
        _reset(env)
        acts, opc = _primera_opcion(env)
        for b in np.flatnonzero(env.active):
            pares = np.argwhere(env.opciones_validas[b, :, :N_NIVELES])
            if len(pares):
                acts[b], opc[b] = pares[0][0], N_NIVELES // 2 if env.opciones_validas[b, pares[0][0], N_NIVELES // 2] else pares[0][1]
        env.step(acts, opc)
        resultados.append(env.last_gano.copy())
    np.testing.assert_array_equal(*resultados)


def test_replay_con_cotizacion(data, mp):
    env = _env(data, mp, replay=True)
    _reset(env, semillas=SEMILLAS[:16])
    np.testing.assert_array_equal(
        env.current_loads_row, mp.count_loads[TRAIN_DAYS + np.arange(16) % 20, np.arange(16) % NUM_NODES])
    margen, a_mano, _, _ = _episodio_al_azar(env, data, np.random.default_rng(2), semillas=SEMILLAS[:16])
    np.testing.assert_allclose(margen, a_mano)


def test_constructor_exige_una_sola_fuente_de_tablero(data, mp):
    with pytest.raises(ValueError):
        VectorRoutingEnv(data["time_matrix"], NUM_NODES, data["dist"], data["diesel"], TRAIN_DAYS)
    with pytest.raises(ValueError):
        VectorRoutingEnv(data["time_matrix"], NUM_NODES, data["dist"], data["diesel"], TRAIN_DAYS,
                         market_process=mp, mercado_replay=mp)
