"""
test_env_cotizacion.py
======================
Fase 2 de especificacion_fase0_ruteo_pricing.md: VectorRoutingEnv con cotización (§4, tests 5-7).

  · máscara = opciones_validas.any(-1); niveles solo con carga cotizable; VACÍO solo hacia casa o
    desde casa y solo si esa lane no se puede cotizar (con carga siempre se cobra)
  · ganar con el nivel k ⇔ u_k < u_comp; el nivel 0 gana siempre
  · un rechazo no consume K, avanza COTIZACION_HORAS, no mueve el camión, deja el tablero igual y
    quita la cotización de esa lane (test 5)
  · después de COTIZACIONES_MAX_POR_NODO rechazos solo quedan opciones VACÍO (test 6)
  · margen = Σ lane_reward con el precio cotizado, en episodios completos (test 7)
  · determinismo y números aleatorios comunes; replay con cotización; opción inválida ⇒ error
"""

import numpy as np
import pytest

from config import COTIZACION_HORAS, COTIZACIONES_MAX_POR_NODO, MAX_DURATION, TRAIN_DAYS, K_MAX
from market_process import MarketProcess, asegurar_parametros
from pricing import N_NIVELES, U_NIVELES_ARR, lane_cost_row
from problem_data import lane_reward, load_matrices
from routing_env import OPCION_VACIO, VectorRoutingEnv

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


def _env(data, mp, replay=False):
    if replay:
        return VectorRoutingEnv(
            data["time_matrix"], NUM_NODES, data["dist"], data["diesel"], data["rate"].shape[0] - TRAIN_DAYS,
            mercado_replay=mp, day_offset=TRAIN_DAYS,
        )
    return VectorRoutingEnv(data["time_matrix"], NUM_NODES, data["dist"], data["diesel"], TRAIN_DAYS,
                            market_process=mp)


def _reset(env, k=None, semillas=SEMILLAS):
    n = len(semillas)
    return env.reset(np.arange(n) % NUM_NODES, np.arange(n) % 20, k_max_arr=[k] * n, world_seeds=semillas)


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


def _alejarse_de_casa(env):
    """Primer paso al destino factible más cercano de cada fila (para quedar en un nodo intermedio): VACÍO si
    la lane no tiene carga y nivel 0 (gana siempre) si la tiene. Devuelve (destinos, opciones)."""
    acts, opc = np.zeros(env._B, dtype=np.int64), np.zeros(env._B, dtype=np.int64)
    for b in range(env._B):
        cand = np.flatnonzero(env.opciones_validas[b].any(-1))
        cand = cand[cand != env.start_node[b]]
        j = cand[np.argmin(env._time_matrix_arr[env.current_node[b], cand])]
        acts[b], opc[b] = j, (OPCION_VACIO if env.opciones_validas[b, j, OPCION_VACIO] else 0)
    env.step(acts, opc)
    assert (env.current_node != env.start_node).all()
    return acts, opc


def _lane_rechazable(env, b):
    """Destino con cotización válida donde el nivel más alto pierde (u_comp ≤ u_9), o None.

    A 10 nodos, desde un nodo intermedio casi siempre la única lane cotizable es la de vuelta a casa
    (varios tiempos de viaje pasan de 50 h), así que la lane de casa cuenta."""
    for j in range(NUM_NODES):
        if env.opciones_validas[b, j, N_NIVELES - 1] and env.u_comp_row[b, j] <= U_NIVELES_ARR[-1]:
            return j
    return None


def _paso_rechazando(env):
    """Un step() donde cada fila activa con una lane rechazable cotiza el nivel 9 ahí (y pierde); el resto
    toma su primera opción válida. Devuelve (casos [(fila, lane)], estado previo, salida de step)."""
    acts, opc = _primera_opcion(env)
    casos = []
    for b in np.flatnonzero(env._active):
        j = _lane_rechazable(env, b)
        if j is not None:
            acts[b], opc[b] = j, N_NIVELES - 1
            casos.append((b, j))
    antes = dict(nodo=env.current_node.copy(), t=env.time_elapsed.copy(), k=env.k_remaining.copy(),
                 precios=env.precios_row.copy(), loads=env.current_loads_row.copy())
    return casos, antes, env.step(acts, opc)


def _verificar_rechazo(env, casos, antes, salida):
    rewards, terminated, masks = salida
    for b, j in casos:
        assert env.last_cotizo[b] and not env.last_gano[b]
        assert env.current_node[b] == antes["nodo"][b]                                   # no se mueve
        assert env.time_elapsed[b] == pytest.approx(antes["t"][b] + COTIZACION_HORAS)
        assert env.k_remaining[b] == antes["k"][b]                                       # no consume K
        assert env.last_raw_reward[b] == 0.0
        if terminated[b]:                                                                # se pasó de max_duration
            continue
        assert rewards[b] == 0.0
        np.testing.assert_array_equal(env.precios_row[b], antes["precios"][b])           # mismo tablero
        np.testing.assert_array_equal(env.current_loads_row[b], antes["loads"][b])
        assert env.rechazadas[b, j] and not env.opciones_validas[b, j, :N_NIVELES].any()
        puede_ir_vacio = (j == env.start_node[b]) or (env.current_node[b] == env.start_node[b])
        assert env.opciones_validas[b, j, OPCION_VACIO] == puede_ir_vacio
        assert bool(masks[b, j]) == puede_ir_vacio


# ── Máscara y opciones ───────────────────────────────────────────────────────

def test_mascara_es_alguna_opcion_valida(data, mp):
    env = _env(data, mp)
    masks = _reset(env)
    op = env.opciones_validas
    assert op.shape == (B, NUM_NODES, N_NIVELES + 1)
    np.testing.assert_array_equal(masks.astype(bool), op.any(axis=-1))
    con_carga = env.current_loads_row > 0
    factible = masks.astype(bool)
    # En casa: con carga se cotiza (niveles) y sin carga se reposiciona (VACÍO); nunca las dos.
    np.testing.assert_array_equal(op[:, :, :N_NIVELES].any(-1), con_carga & factible)
    np.testing.assert_array_equal(op[:, :, OPCION_VACIO], ~con_carga & factible)
    assert not (op[:, :, :N_NIVELES].any(-1) & op[:, :, OPCION_VACIO]).any()


def test_en_nodo_intermedio_vacio_solo_hacia_casa(data, mp):
    env = _env(data, mp)
    _reset(env)
    _alejarse_de_casa(env)
    act = env._active
    op = env.opciones_validas
    for b in np.flatnonzero(act):
        h = env.start_node[b]
        vacio = np.flatnonzero(op[b, :, OPCION_VACIO])
        # VACÍO solo hacia casa, y solo si la lane a casa no se puede cotizar.
        np.testing.assert_array_equal(vacio, [] if op[b, h, :N_NIVELES].any() else [h])
        assert op[b, h].any()                                                            # volver siempre se puede


# ── Ganar y perder ───────────────────────────────────────────────────────────

def test_nivel_cero_gana_y_cobra_su_precio(data, mp):
    env = _env(data, mp)
    _reset(env, k=K_MAX)
    filas = [b for b in range(B) if env.opciones_validas[b, :, 0].any()]
    acts, opc = _primera_opcion(env)
    for b in filas:
        acts[b], opc[b] = np.flatnonzero(env.opciones_validas[b, :, 0])[0], 0
    origen = env.current_node.copy()
    precios = env.precios_row.copy()
    env.step(acts, opc)
    for b in filas:
        i, j = origen[b], acts[b]
        assert env.last_gano[b] and env.current_node[b] == j
        esperado = lane_reward(precios[b, j, 0], 1, data["dist"][i, j], data["diesel"][i, j])
        assert env.last_raw_reward[b] == pytest.approx(float(esperado))
        assert env.k_remaining[b] == K_MAX - 1 and env.n_rechazos[b] == 0


def test_gana_si_y_solo_si_queda_bajo_el_competidor(data, mp):
    env = _env(data, mp)
    _reset(env)
    rng = np.random.default_rng(3)
    acts, opc = np.zeros(B, dtype=np.int64), np.zeros(B, dtype=np.int64)
    for b in range(B):
        pares = np.argwhere(env.opciones_validas[b, :, :N_NIVELES])
        acts[b], opc[b] = pares[rng.integers(len(pares))] if len(pares) else (
            np.flatnonzero(env.opciones_validas[b, :, OPCION_VACIO])[0], OPCION_VACIO)
    u_comp = env.u_comp_row[np.arange(B), acts].copy()
    env.step(acts, opc)
    cot = opc < N_NIVELES
    esperado = (U_NIVELES_ARR[np.minimum(opc, N_NIVELES - 1)] <= 0) | (U_NIVELES_ARR[np.minimum(opc, N_NIVELES - 1)] < u_comp)
    np.testing.assert_array_equal(env.last_gano[cot], esperado[cot])
    np.testing.assert_array_equal(env.last_cotizo, cot)


def test_vacio_desde_casa_paga_solo_el_costo(data, mp):
    env = _env(data, mp)
    _reset(env)
    origen = env.current_node.copy()
    acts, opc = _alejarse_de_casa(env)
    vacio = opc == OPCION_VACIO
    assert vacio.any()
    costo = lane_cost_row(data["dist"][origen, acts], data["diesel"][origen, acts])
    np.testing.assert_allclose(env.last_raw_reward[vacio], -np.round(costo[vacio]), atol=0.5)
    assert not env.last_cotizo[vacio].any() and env.last_gano[~vacio].all()            # nivel 0 gana siempre


def test_rechazo_en_casa(data, mp):
    """Test 5 en el primer movimiento: la lane rechazada sigue disponible como reposicionamiento VACÍO."""
    env = _env(data, mp)
    _reset(env, k=K_MAX)
    casos, antes, salida = _paso_rechazando(env)
    assert len(casos) > B // 2
    _verificar_rechazo(env, casos, antes, salida)
    assert all(env.n_rechazos[b] == 1 for b, _ in casos)


def test_rechazo_en_nodo_intermedio(data, mp):
    """Test 5 lejos de casa: la lane rechazada desaparece (salvo la de casa, que queda como VACÍO)."""
    env = _env(data, mp)
    _reset(env, k=K_MAX)
    _alejarse_de_casa(env)
    casos, antes, salida = _paso_rechazando(env)
    assert casos, "ninguna fila tiene una lane donde el nivel 9 pierda"
    _verificar_rechazo(env, casos, antes, salida)


def test_tope_de_cotizaciones_en_casa(data, mp):
    """Test 6 en casa: tras el tope ya no se cotiza; los reposicionamientos VACÍO siguen."""
    env = _env(data, mp)
    _reset(env)
    for _ in range(COTIZACIONES_MAX_POR_NODO):
        _paso_rechazando(env)
    llenas = np.flatnonzero(env._active & (env.n_rechazos == COTIZACIONES_MAX_POR_NODO))
    assert len(llenas), "ninguna fila llegó al tope de cotizaciones"
    for b in llenas:
        assert env.current_node[b] == env.start_node[b]
        assert not env.opciones_validas[b, :, :N_NIVELES].any()
        assert env.opciones_validas[b, :, OPCION_VACIO].any()


def test_tope_de_cotizaciones_en_nodo_intermedio(data, mp):
    """Test 6 lejos de casa: con el tope alcanzado la única opción es volver a casa VACÍO."""
    env = _env(data, mp)
    _reset(env)
    _alejarse_de_casa(env)
    act = np.flatnonzero(env._active)
    env._n_rechazos[act] = COTIZACIONES_MAX_POR_NODO
    masks = env._compute_masks()
    for b in act:
        np.testing.assert_array_equal(np.argwhere(env.opciones_validas[b]), [[env.start_node[b], OPCION_VACIO]])
        np.testing.assert_array_equal(np.flatnonzero(masks[b]), [env.start_node[b]])


def test_opcion_invalida_lanza_error(data, mp):
    env = _env(data, mp)
    _reset(env)
    b = 0
    malo = int(np.flatnonzero(~env.opciones_validas[b].any(-1))[0])                     # self-loop al menos
    acts, opc = _primera_opcion(env)
    acts[b] = malo
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
        margen += env.last_raw_reward
        for b in np.flatnonzero(act):
            i, j, o = origen[b], acts[b], opc[b]
            if o == OPCION_VACIO:
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
        acts = np.array([np.flatnonzero(env.opciones_validas[b].any(-1))[0] for b in range(B)])
        opc = np.array([N_NIVELES // 2 if env.opciones_validas[b, acts[b], N_NIVELES // 2] else OPCION_VACIO
                        for b in range(B)])
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
