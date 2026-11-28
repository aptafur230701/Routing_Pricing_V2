"""
test_env.py
===========
VectorRoutingEnv (routing_env.py) con cotización, espera y reposicionamiento, en tensores
(especificacion_fase0_ruteo_pricing.md §4):

  · máscara = opciones_validas.any(-1); niveles solo en lanes j ≠ i con carga cotizable; VACÍO a j ≠ i solo
    para reposicionarse (≤ REPOSICION_MAX_MILLAS) o volver a casa desde un intermedio; VACÍO en j = i = esperar
  · ganar con el nivel k ⇔ u_k < u_comp; ganar cobra el precio del nivel; volver vacío paga el costo
  · un rechazo no consume K, avanza COTIZACION_HORAS, no mueve el camión, deja el tablero igual y pierde UNA carga
    (la siguiente tiene su propio competidor); con COTIZACIONES_MAX_POR_NODO rechazos solo quedan VACÍO
  · esperar salta al siguiente día de mercado, no consume K, revela el tablero del día nuevo y reinicia los rechazos
  · sin jugadas en reset() ⇒ no sale (margen 0, válido); sin jugadas después ⇒ el ciclo falla
  · margen = Σ lane_reward con el precio cotizado; K se respeta; todos los episodios terminan
  · el tablero es el del mundo (no depende de las acciones): índice, día de la semana, margen a la mediana
  · revisitas: un intermedio ya recorrido sigue disponible; max_pasos acota los arcos
"""

import numpy as np
import pytest
import torch

import routing_env
from config import (COTIZACION_HORAS, COTIZACIONES_MAX_POR_NODO, MAX_DURATION, K_MAX, K_LEVELS,
                    REPOSICION_MAX_MILLAS, HORAS_DIA_MERCADO, TIME_VIOLATION_PENALTY, DEVICE)
from conftest import (Datos, accion_al_azar, correr_al_azar, escenario_sintetico, mundos_sim, primera_opcion)
from pricing import N_NIVELES, U_NIVELES_ARR, lane_cost_row, percentil_competidor
from problem_data import lane_reward, load_matrices
from routing_env import (OPCION_VACIO, VectorRoutingEnv, fin_espera, max_esperas, max_pasos, max_pasos_cotizacion)

N = 10
B = 64
U = torch.as_tensor(U_NIVELES_ARR, device=DEVICE)


def np_(x):
    return x.detach().cpu().numpy()


def _reset(d, k=None, B=B, semilla=0, nodos=None, replay=False):
    env = d.env()
    mundos = mundos_sim(d.market, B, semilla=semilla, replay=replay)
    nodos = np.arange(B) % N if nodos is None else nodos
    masks = env.reset(mundos, nodos, None if k is None else [k] * B)
    return env, mundos, masks


def _nivel_que_gana(env, b, j):
    """El nivel más bajo que gana la lane j de la fila b (conociendo el competidor), o None."""
    ok = np_(env.opciones_validas[b, j, :N_NIVELES]) & (U_NIVELES_ARR < float(env.u_comp_row[b, j]))
    return int(np.flatnonzero(ok)[0]) if ok.any() else None


def _alejarse_de_casa(env, T):
    """Un paso en que cada fila que puede gana una lane a un intermedio (la más cercana donde algún nivel gana); las
    demás esperan si pueden. Devuelve las filas que quedaron en un intermedio."""
    a, o = (np_(x).copy() for x in primera_opcion(env))
    op = np_(env.opciones_validas)
    cur, casa = np_(env.current_node), np_(env.start_node)
    for b in range(env.B):
        i = cur[b]
        if op[b, i, OPCION_VACIO]:
            a[b], o[b] = i, OPCION_VACIO
        for j in np.argsort(T[i]):
            if j != casa[b] and (k := _nivel_que_gana(env, b, j)) is not None:
                a[b], o[b] = j, k
                break
    env.step(torch.as_tensor(a, device=DEVICE), torch.as_tensor(o, device=DEVICE))
    fuera = np.flatnonzero(np_(env.active) & (np_(env.current_node) != casa))
    assert len(fuera) > env.B // 4, "pocas filas llegaron a un nodo intermedio"
    return fuera


def _lane_rechazable(env, b):
    """Destino j ≠ i con cotización válida donde el nivel más alto pierde, o None."""
    op, u = np_(env.opciones_validas[b]), np_(env.u_comp_row[b])
    i = int(env.current_node[b])
    for j in range(env.num_nodes):
        if j != i and op[j, N_NIVELES - 1] and u[j] <= U_NIVELES_ARR[-1]:
            return j
    return None


def _paso_rechazando(env, filas=None):
    """Un step() donde cada fila (de `filas`, o activa) con una lane rechazable cotiza ahí el nivel más alto (y
    pierde); el resto toma su primera opción válida. Devuelve (casos, estado previo, salida)."""
    a, o = (np_(x).copy() for x in primera_opcion(env))
    casos = []
    for b in (np.flatnonzero(np_(env.active)) if filas is None else filas):
        j = _lane_rechazable(env, b)
        if j is not None:
            a[b], o[b] = j, N_NIVELES - 1
            casos.append((b, j))
    antes = dict(nodo=np_(env.current_node).copy(), t=np_(env.time_elapsed).copy(), k=np_(env.k_remaining).copy(),
                 precios=np_(env.precios_row).copy(), loads=np_(env.current_loads_row).copy(),
                 perdidas=np_(env.perdidas).copy())
    salida = env.step(torch.as_tensor(a, device=DEVICE), torch.as_tensor(o, device=DEVICE))
    return casos, antes, salida


# ── Máscara y opciones ───────────────────────────────────────────────────────

def test_mascara_es_alguna_opcion_valida(d10):
    env, _, masks = _reset(d10)
    op, act = np_(env.opciones_validas), np_(env.active)
    assert op.shape == (B, N, N_NIVELES + 1) and masks.dtype == torch.bool
    np.testing.assert_array_equal(np_(masks)[act], op[act].any(axis=-1))
    filas, i = np.arange(B), np_(env.current_node)
    assert not (op[:, :, :N_NIVELES].any(-1) & ~(np_(env.current_loads_row) > 0))[act].any()   # niveles: con carga
    assert not op[filas, i, :N_NIVELES].any()                                                  # ni en el self-loop
    vacio_otro = op[:, :, OPCION_VACIO].copy()
    vacio_otro[filas, i] = False
    assert not vacio_otro[act].any()                       # en casa, VACÍO a otro nodo solo para reposicionarse…
    assert not np_(env._reposicion).any()                  # …y a 10 nodos ningún par está a ≤ 300 mi


def test_sin_salida_es_exogeno_y_valido(d10):
    """La casa 9 a 10 nodos (ida y vuelta mínima 72.7 h): sin carga cotizable no tiene ninguna jugada."""
    env, _, _ = _reset(d10, B=200, nodos=np.full(200, 9), semilla=5)
    sin, act = np_(env.sin_salida), np_(env.active)
    assert sin.any() and not sin.all()
    assert not act[sin].any() and act[~sin].all() and not np_(env.last_time_violation).any()


def test_en_nodo_intermedio_vacio_hacia_casa_o_esperar(d10):
    env, _, _ = _reset(d10)
    fuera = _alejarse_de_casa(env, d10.T)
    op = np_(env.opciones_validas)
    for b in fuera:
        h, i = int(env.start_node[b]), int(env.current_node[b])
        vacio = set(np.flatnonzero(op[b, :, OPCION_VACIO]))
        assert h in vacio and vacio <= {h, i}
        espera_ok = fin_espera(float(env.time_elapsed[b])) + d10.T[i, h] <= MAX_DURATION
        assert (i in vacio) == espera_ok


def test_reposicionamiento_solo_a_300_millas():
    """A 20 nodos hay pares a ≤ 300 mi: VACÍO a j ≠ i solo hacia ellos (o hacia casa desde un intermedio)."""
    d = Datos(20)
    env = d.env()
    env.reset(mundos_sim(d.market, 40, semilla=1), np.arange(40) % 20)
    op = np_(env.opciones_validas)
    for b in np.flatnonzero(np_(env.active)):
        i = int(env.current_node[b])
        vacio = np.flatnonzero(op[b, :, OPCION_VACIO])
        assert np.all(np.asarray(d.dist)[i, vacio[vacio != i]] <= REPOSICION_MAX_MILLAS)
    assert any(op[b, :, OPCION_VACIO].sum() > 1 for b in range(40))


# ── Ganar y perder ───────────────────────────────────────────────────────────

def test_gana_si_y_solo_si_queda_bajo_el_competidor(d10):
    env, _, _ = _reset(d10)
    gen = torch.Generator(device=DEVICE).manual_seed(3)
    a, o = accion_al_azar(env, gen)
    u_comp = env.u_comp_row[torch.arange(B, device=DEVICE), a].clone()
    act = env.active.clone()
    env.step(a, o)
    cot = act & (o < N_NIVELES)
    assert cot.any()
    esperado = U[o.clamp(max=N_NIVELES - 1)] < u_comp
    assert torch.equal(env.last_gano[cot], esperado[cot]) and torch.equal(env.last_cotizo[act], cot[act])


def test_ganar_cobra_el_precio_del_nivel(d10):
    env, _, _ = _reset(d10, k=K_MAX)
    a, o = (np_(x).copy() for x in primera_opcion(env))
    filas = []
    for b in np.flatnonzero(np_(env.active)):
        for j in range(N):
            if (k := _nivel_que_gana(env, b, j)) is not None:
                a[b], o[b] = j, k
                filas.append(b)
                break
    assert filas
    origen, precios = np_(env.current_node).copy(), np_(env.precios_row).copy()
    env.step(torch.as_tensor(a, device=DEVICE), torch.as_tensor(o, device=DEVICE))
    for b in filas:
        i, j, k = origen[b], a[b], o[b]
        assert env.last_gano[b] and int(env.current_node[b]) == j
        esperado = lane_reward(precios[b, j, k], 1, d10.dist[i, j], d10.diesel[i, j])
        assert float(env.last_raw_reward[b]) == pytest.approx(float(esperado))
        assert int(env.k_remaining[b]) == K_MAX - 1 and int(env.n_rechazos[b]) == 0


def test_volver_a_casa_vacio_paga_solo_el_costo(d10):
    env, _, _ = _reset(d10)
    fuera = _alejarse_de_casa(env, d10.T)
    a, o = (np_(x).copy() for x in primera_opcion(env))
    a[fuera], o[fuera] = np_(env.start_node)[fuera], OPCION_VACIO
    origen = np_(env.current_node).copy()
    env.step(torch.as_tensor(a, device=DEVICE), torch.as_tensor(o, device=DEVICE))
    costo = lane_cost_row(d10.dist[origen, a], d10.diesel[origen, a])
    np.testing.assert_allclose(np_(env.last_raw_reward)[fuera], -np.round(costo[fuera]), atol=0.5)
    assert not np_(env.last_cotizo)[fuera].any()
    assert (np_(env.current_node)[fuera] == np_(env.start_node)[fuera]).all()


def _verificar_rechazo(env, T, casos, antes, salida):
    rewards, terminated, _ = (np_(x) for x in salida)
    op = np_(env.opciones_validas)
    for b, j in casos:
        assert env.last_cotizo[b] and not env.last_gano[b]
        assert int(env.current_node[b]) == antes["nodo"][b]                                  # no se mueve
        assert float(env.time_elapsed[b]) == pytest.approx(antes["t"][b] + COTIZACION_HORAS)
        assert int(env.k_remaining[b]) == antes["k"][b] and float(env.last_raw_reward[b]) == 0.0
        if terminated[b]:
            continue
        assert rewards[b] == 0.0
        np.testing.assert_array_equal(np_(env.precios_row[b]), antes["precios"][b])          # mismo tablero
        np.testing.assert_array_equal(np_(env.current_loads_row[b]), antes["loads"][b])
        assert int(env.perdidas[b, j]) == antes["perdidas"][b, j] + 1                       # se pierde UNA carga
        if antes["loads"][b, j] <= int(env.perdidas[b, j]) or int(env.n_rechazos[b]) >= COTIZACIONES_MAX_POR_NODO:
            assert not op[b, j, :N_NIVELES].any()
        i, h = antes["nodo"][b], int(env.start_node[b])
        assert op[b, j, OPCION_VACIO] == (j == h and i != h)                                 # sin reposicionar a 10 nodos


def test_rechazo_en_casa_y_en_nodo_intermedio(d10):
    env, _, _ = _reset(d10, k=K_MAX)
    casos, antes, salida = _paso_rechazando(env)
    assert len(casos) > B // 4
    _verificar_rechazo(env, d10.T, casos, antes, salida)
    assert all(int(env.n_rechazos[b]) == 1 for b, _ in casos if env.active[b])
    env, _, _ = _reset(d10, k=K_MAX, semilla=1)
    fuera = _alejarse_de_casa(env, d10.T)
    casos, antes, salida = _paso_rechazando(env, fuera)
    assert casos, "ninguna fila tiene una lane donde el nivel más alto pierda"
    _verificar_rechazo(env, d10.T, casos, antes, salida)


def test_cada_carga_de_la_lane_tiene_su_competidor(d10):
    env, _, _ = _reset(d10)
    casos, antes, _ = _paso_rechazando(env)
    vistos = 0
    for b, j in casos:
        if not env.active[b]:
            continue
        assert float(env.u_comp_row[b, j]) == float(env.u_comp_cargas[b, j, 1])
        vistos += float(env.u_comp_cargas[b, j, 0]) != float(env.u_comp_cargas[b, j, 1])
        assert int(env.intentos[b, j]) == min(antes["loads"][b, j] - 1, COTIZACIONES_MAX_POR_NODO - 1)
    assert vistos


def test_tope_de_cotizaciones_en_nodo_intermedio(d10):
    env, _, _ = _reset(d10)
    fuera = _alejarse_de_casa(env, d10.T)
    env._n_rechazos[torch.as_tensor(fuera, device=DEVICE)] = COTIZACIONES_MAX_POR_NODO
    masks = env._compute_masks()
    for b in fuera:
        op, h = np_(env.opciones_validas[b]), int(env.start_node[b])
        assert not op[:, :N_NIVELES].any() and op[h, OPCION_VACIO] and masks[b, h]


# ── Esperar y fallar ─────────────────────────────────────────────────────────

def test_esperar_salta_al_dia_siguiente_con_tablero_nuevo(d10):
    env, mundos, _ = _reset(d10, k=K_MAX)
    filas = torch.arange(B, device=DEVICE)
    puede = np.flatnonzero(np_(env.active & env.opciones_validas[filas, env.current_node, OPCION_VACIO]))
    assert len(puede) > B // 2
    _paso_rechazando(env, puede)                                      # primero un rechazo: la espera los reinicia
    op = np_(env.opciones_validas)
    puede = [b for b in puede if env.active[b] and op[b, int(env.current_node[b]), OPCION_VACIO]]
    a, o = (np_(x).copy() for x in primera_opcion(env))
    for b in puede:
        a[b], o[b] = int(env.current_node[b]), OPCION_VACIO
    antes = dict(nodo=np_(env.current_node).copy(), t=np_(env.time_elapsed).copy(), k=np_(env.k_remaining).copy(),
                 dia=np_(env.current_day).copy(), u=np_(env.u_comp_row).copy())
    rewards, terminated, _ = (np_(x) for x in env.step(torch.as_tensor(a, device=DEVICE),
                                                      torch.as_tensor(o, device=DEVICE)))
    cambio = 0
    for b in puede:
        assert env.last_espero[b] and not env.last_cotizo[b]
        assert int(env.current_node[b]) == antes["nodo"][b] and int(env.k_remaining[b]) == antes["k"][b]
        assert float(env.last_raw_reward[b]) == 0.0 and rewards[b] == 0.0 and not terminated[b]
        assert float(env.time_elapsed[b]) == pytest.approx(fin_espera(antes["t"][b]))
        assert float(env.time_elapsed[b]) % HORAS_DIA_MERCADO == pytest.approx(0.0, abs=1e-4)
        assert int(env.current_day[b]) == min(antes["dia"][b] + 1, mundos.dia_max)
        assert int(env.n_rechazos[b]) == 0 and not np_(env.perdidas[b]).any()
        cambio += not np.array_equal(np_(env.u_comp_row[b]), antes["u"][b])
    assert cambio > len(puede) // 2
    assert (np_(env.step_count)[puede] == 0).all()                    # no cuenta como arco


def test_quedarse_sin_jugadas_despues_es_falla(d10):
    """Casa 9 (ida y vuelta mínima 72.7 h): si tenía carga cotizable y la pierde, no puede esperar ni
    reposicionarse: el ciclo falla con la penalización (no es 'no sale')."""
    env, _, _ = _reset(d10, B=200, nodos=np.full(200, 9), semilla=5)
    filas = [b for b in np.flatnonzero(np_(env.active)) if _lane_rechazable(env, b) is not None]
    assert filas
    fallan = []
    for _ in range(COTIZACIONES_MAX_POR_NODO):
        casos, _, (rewards, terminated, _) = _paso_rechazando(env, [b for b in filas if env.active[b]])
        for b, _j in casos:
            if terminated[b] and int(env.step_count[b]) == 0:
                assert env.last_time_violation[b] and not env.sin_salida[b]
                assert float(rewards[b]) == pytest.approx(TIME_VIOLATION_PENALTY)
                fallan.append(b)
    assert fallan


def test_opcion_invalida_lanza_error_con_check_tensors(d10, monkeypatch):
    monkeypatch.setattr(routing_env, "CHECK_TENSORS", True)
    env, _, _ = _reset(d10)
    b = int(np.flatnonzero(np_(env.active))[0])
    malo_j, malo_o = np.argwhere(~np_(env.opciones_validas[b]))[0]
    a, o = primera_opcion(env)
    a[b], o[b] = int(malo_j), int(malo_o)
    with pytest.raises(ValueError):
        env.step(a, o)


# ── Episodios completos ──────────────────────────────────────────────────────

def _episodio_al_azar(d, env, mundos, nodos, k=None, semilla=11):
    """Corre episodios al azar; devuelve (margen por fila, margen recalculado a mano, filas activas al final)."""
    n = mundos.B
    margen, a_mano = np.zeros(n), np.zeros(n)

    def al_paso(env, a, o, activo):
        al_paso.previo = (np_(env.current_node).copy(), np_(env.precios_row).copy(), np_(a), np_(o), np_(activo))

    def despues():
        i_, p_, a_, o_, act = al_paso.previo
        margen[act] += np_(env.last_raw_reward)[act]
        for b in np.flatnonzero(act):
            i, j, o = i_[b], a_[b], o_[b]
            if o == OPCION_VACIO and i != j:
                a_mano[b] += lane_reward(0.0, 0, d.dist[i, j], d.diesel[i, j])
            elif o < OPCION_VACIO and env.last_gano[b]:
                a_mano[b] += lane_reward(p_[b, j, o], 1, d.dist[i, j], d.diesel[i, j])

    env.reset(mundos, nodos, None if k is None else [k] * n)
    gen = torch.Generator(device=DEVICE).manual_seed(semilla)
    activo = env.active.clone()
    for _ in range(env.max_steps):
        if not bool(activo.any()):
            break
        a, o = accion_al_azar(env, gen)
        al_paso(env, a, o, activo)
        _, term, _ = env.step(a, o)
        despues()
        activo &= ~term
    return margen, a_mano, activo


@pytest.mark.parametrize("k", [None, K_MAX])
def test_margen_es_suma_de_lane_reward_y_todos_terminan(d10, k):
    env = d10.env()
    mundos = mundos_sim(d10.market, B, semilla=2)
    margen, a_mano, activo = _episodio_al_azar(d10, env, mundos, np.arange(B) % N, k=k)
    np.testing.assert_allclose(margen, a_mano)
    assert not activo.any() and (np_(env.time_elapsed) <= MAX_DURATION + 24).all()
    if k is not None:
        assert (np_(env.step_count) <= k).all()


def test_mismo_mundo_mismas_acciones_mismo_resultado(d10):
    mundos = mundos_sim(d10.market, B, semilla=4)
    m1, _, _ = _episodio_al_azar(d10, d10.env(), mundos, np.arange(B) % N, semilla=5)
    m2, _, _ = _episodio_al_azar(d10, d10.env(), mundos, np.arange(B) % N, semilla=5)
    np.testing.assert_array_equal(m1, m2)


def test_el_tablero_es_el_del_mundo_y_no_depende_de_las_acciones(d10):
    """En cada llegada el entorno lee la fila [b, día − dia0, i] del mundo: cargas y competidores (u → u_comp con el m
    del día), el índice y el LTR del día y el margen a la mediana; dos políticas distintas ven lo mismo. El día es el
    de la visita, aunque los rechazos hayan pasado el reloj al día siguiente."""
    mp, tab = d10.market, d10.market.tensores(DEVICE)
    mundos = mundos_sim(mp, 30, semilla=6)
    for semilla in (1, 2):
        def al_paso(env, a, o, activo):
            for b in np.flatnonzero(np_(activo))[:10]:
                dia, i = int(env.current_day[b]), int(env.current_node[b])
                slot = dia - int(mundos.dia0[b])
                np.testing.assert_array_equal(np_(env.current_loads_row[b]), np_(mundos.loads[b, slot, i]))
                m = float(tab["m"][dia, i])
                np.testing.assert_allclose(np_(env.u_comp_cargas[b]),
                                           percentil_competidor(np_(mundos.u[b, slot, i]).astype(np.float64), m))
                np.testing.assert_array_equal(np_(env.market_index[b]), np_(tab["indice"][dia]))
                assert float(env.competencia[b]) == m
                esperado = lane_reward(np_(env.rate_row_obs[b]), np_(env.current_loads_row[b]), d10.dist[i],
                                       d10.diesel[i]).astype(np.float32)
                esperado[i] = 0.0                                                # self-loop = esperar: margen 0
                np.testing.assert_array_equal(np_(env.reward_rows[b]), esperado)
                avance = int(env.dias_desde_salida[b])                       # el día de la visita, no el del reloj
                assert avance <= int(float(env.time_elapsed[b]) // 14) and dia == min(int(mundos.dia0[b]) + avance,
                                                                                    mundos.dia_max)
                assert int(env.current_dow[b]) == (int(tab["dow"][int(mundos.dia0[b])]) + avance) % 7
        correr_al_azar(d10.env(), mundos, np.arange(30) % N, semilla=semilla, al_paso=al_paso)


def test_cotizacion_perdida_que_cruza_de_dia_no_cambia_el_dia_de_la_visita():
    """El camión llega a 1 a las 13.9 h (día 0) y pierde una cotización: el reloj pasa a 14.15 h (día 1), pero el
    tablero, el día, el día de la semana, el índice, el LTR y el tablero de la red siguen siendo los del día 0. Al
    esperar se revela el día nuevo."""
    from market_process import Mundos
    n, D, C = 3, 6, COTIZACIONES_MAX_POR_NODO
    s = escenario_sintetico(n)
    T = np.full((n, n), 13.9)
    np.fill_diagonal(T, 0.0)
    mercado = s["mercado"]
    env = VectorRoutingEnv(T, s["dist"], s["diesel"], mercado, device=DEVICE)
    loads = torch.zeros((1, D, n, n), dtype=torch.int32, device=DEVICE)
    for d in range(D):
        loads[0, d] = d + 1                                                   # cada día, otro número de cargas
    loads[0, :, torch.arange(n), torch.arange(n)] = 0
    u = torch.zeros((1, D, n, n, C), device=DEVICE)                           # competidor u = 0: toda cotización pierde
    mundos = Mundos(loads=loads, u=u, delta=torch.zeros(1, dtype=torch.float64, device=DEVICE),
                    dia0=torch.zeros(1, dtype=torch.long, device=DEVICE), dia_max=D - 1)
    env.reset(mundos, [0])
    tab = mercado.tensores(DEVICE)
    a = lambda x: torch.tensor([x], device=DEVICE)
    env.step(a(1), a(OPCION_VACIO))                                           # vacío a 1: llega a las 13.9 h
    assert float(env.time_elapsed[0]) == pytest.approx(13.9) and int(env.current_day[0]) == 0
    antes = dict(cargas=np_(env.current_loads_row[0]).copy(), hoy=np_(env.cargas_hoy[0]).copy(),
                 indice=np_(env.market_index[0]).copy(), dow=int(env.current_dow[0]), ltr=float(env.ltr_actual[0]))
    env.step(a(2), a(0))                                                      # cotiza 1→2 y pierde
    assert env.last_cotizo[0] and not env.last_gano[0]
    assert float(env.time_elapsed[0]) == pytest.approx(13.9 + COTIZACION_HORAS)   # el reloj ya está en el día 1
    assert int(env.current_day[0]) == 0 and int(env.dias_desde_salida[0]) == 0
    np.testing.assert_array_equal(np_(env.current_loads_row[0])[[0]], antes["cargas"][[0]])   # lane no cotizada igual
    np.testing.assert_array_equal(np_(env.cargas_hoy[0]), antes["hoy"])
    np.testing.assert_array_equal(np_(env.cargas_hoy[0]), np_(loads[0, 0]))
    np.testing.assert_array_equal(np_(env.market_index[0]), np_(tab["indice"][0]))
    np.testing.assert_array_equal(np_(env.market_index[0]), antes["indice"])
    assert int(env.current_dow[0]) == antes["dow"] and float(env.ltr_actual[0]) == antes["ltr"]
    env.step(a(1), a(OPCION_VACIO))                                           # espera: tablero del día nuevo
    assert float(env.time_elapsed[0]) == pytest.approx(28.0) and int(env.current_day[0]) == 2
    np.testing.assert_array_equal(np_(env.current_loads_row[0]), np_(loads[0, 2, 1]))
    np.testing.assert_array_equal(np_(env.cargas_hoy[0]), np_(loads[0, 2]))


def test_replay_con_las_cargas_reales(d10):
    env = d10.env()
    mundos = mundos_sim(d10.market, 16, semilla=2, replay=True)
    nodos = np.arange(16) % N
    env.reset(mundos, nodos)
    np.testing.assert_array_equal(np_(env.current_loads_row), d10.market.count_loads[np_(mundos.dia0), nodos])
    margen, a_mano, _ = _episodio_al_azar(d10, env, mundos, nodos, semilla=2)
    np.testing.assert_allclose(margen, a_mano)


# ── Cardinalidad y revisitas (escenario sintético: 1 h por arco, K es la restricción que manda) ──

NS, MAX_SYN = 5, 30.0


@pytest.fixture(scope="module")
def sintetico():
    return escenario_sintetico(NS)


def _env_sint(s):
    return VectorRoutingEnv(s["T"], s["dist"], s["diesel"], s["mercado"], device=DEVICE, max_duration=MAX_SYN)


def _paso(env, a):
    """Viaja a `a` cotizando p10 (con u_min = 0.15 gana siempre)."""
    return env.step(torch.tensor([a], device=DEVICE), torch.tensor([0], device=DEVICE))


def test_mascara_permite_un_intermedio_ya_visitado(sintetico):
    s = sintetico
    env = _env_sint(s)
    m = env.reset(s["mercado"].mundos(s["loads"], 1, u_min=0.15), [0])[0]
    assert m[0] and np_(env.opciones_validas[0, 0]).tolist() == [False] * N_NIVELES + [True]   # en casa: esperar
    for a in (1, 2, 1, 3):                                         # 1 se recorre dos veces
        _, term, masks = _paso(env, a)
        assert not term[0] and env.last_gano[0]
        assert masks[0, a] and not env.opciones_validas[0, a, :N_NIVELES].any()   # self-loop: solo esperar
        assert masks[0, 0]                                         # el retorno siempre disponible
        if a == 2:
            assert masks[0, 1]                                     # 1 ya fue recorrido y sigue disponible
    _, term, _ = _paso(env, 0)
    assert term[0]


def test_retorno_forzado_por_k_con_revisita(sintetico):
    s = sintetico
    env = _env_sint(s)
    env.reset(s["mercado"].mundos(s["loads"], 1, u_min=0.15), [0], [4])
    for a in (1, 2, 1):                                            # 0→1→2→1: queda un arco
        _, _, m = _paso(env, a)
    assert np_(m[0]).tolist() == [True, True, False, False, False] and int(env.k_remaining[0]) == 1


@pytest.mark.parametrize("k", list(K_LEVELS) + [None])
def test_k_se_respeta_y_hay_episodios_con_mas_arcos_que_nodos(sintetico, k):
    s = sintetico
    env = _env_sint(s)
    assert max_pasos(s["T"], MAX_SYN) == 31
    assert env.max_steps == max_pasos_cotizacion(s["T"], MAX_SYN) == \
        (31 + max_esperas(MAX_SYN)) * (COTIZACIONES_MAX_POR_NODO + 1)
    B_ = 300
    mundos = s["mercado"].mundos(s["loads"], B_, semilla=1)
    activo = correr_al_azar(env, mundos, np.arange(B_) % NS, None if k is None else [k] * B_, semilla=3)
    assert not activo.any()
    pasos = np_(env.step_count)
    assert pasos.max() <= (max_pasos(s["T"], MAX_SYN) if k is None else k)
    if k is None:
        assert pasos.max() > NS                                    # solo posible si se revisitan nodos


def test_max_pasos_con_datos_reales():
    for n in (10, 20, 35):
        T = np.asarray(load_matrices(n)[0], dtype=float)
        assert max_pasos(T, MAX_DURATION) == int(MAX_DURATION // T[~np.eye(n, dtype=bool)].min()) + 1
