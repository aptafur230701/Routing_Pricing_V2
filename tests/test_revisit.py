"""
test_revisit.py
===============
El camión puede volver a pasar por un nodo intermedio dentro del tour. Pruebas de lo que cambia:

  · La máscara del entorno con cotización permite un intermedio ya recorrido; el self-loop es la espera,
    el retorno siempre está y con K = 1 restante solo quedan casa y esperar.
  · max_pasos acota los arcos de los episodios (reales y sintéticos), max_pasos_cotizacion los pasos
    (rechazos y esperas incluidos), y hay episodios con más arcos que N (se revisita).
  · UB-PI (DP hacia atrás) coincide con una enumeración por fuerza bruta de todas las rutas (cargadas,
    vacías ≤ 300 mi o a casa, y esperas), con revisitas, en mundos pequeños (N = 6, K = 4 y caso libre), y es
    cota del valor de cada ruta evaluada con tiempos exactos y el día real de cada salida.
  · B2 (recursión memoizada, con esperas) coincide con la enumeración ingenua de rutas (N = 5).
"""

import numpy as np
import pytest

from baselines.politicas import Observacion, PoliticaB2, valor_inmediato
from config import MAX_DURATION, COTIZACION_HORAS, COTIZACIONES_MAX_POR_NODO
from market_process import MarketProcess, asegurar_parametros
from pricing import N_NIVELES, U_NIVELES_ARR
from problem_data import lane_reward, load_matrices
from routing_env import (OPCION_VACIO, VectorRoutingEnv, max_pasos, max_pasos_cotizacion, max_esperas,
                         vacio_permitido, fin_espera)
from baselines.ub_pi import solve_ub_pi, franjas_por_dia
from mercado_sintetico import MercadoSintetico

# ─────────────────────────────────────────────────────────────────────────────
# Máscara con revisitas
# ─────────────────────────────────────────────────────────────────────────────

N = 5
MAX_SYN = 30.0   # con 1 h por arco caben hasta 30 arcos


@pytest.fixture(scope="module")
def sintetico():
    T = np.ones((N, N))
    np.fill_diagonal(T, 0.0)
    dist = np.full((N, N), 100.0)
    np.fill_diagonal(dist, 0.0)
    diesel = np.full((N, N), 3.0)
    loads = np.full((N, N), 5)
    np.fill_diagonal(loads, 0)
    return dict(T=T, dist=dist, diesel=diesel,
                mercado=MercadoSintetico(loads, np.full((N, N), 3.0), u_min=0.15))   # p10 gana siempre aquí


def _env(d):
    return VectorRoutingEnv(d["T"], N, d["dist"], d["diesel"], 3, max_duration=MAX_SYN, market_process=d["mercado"])


def _paso(env, a):
    """Viaja a `a` cotizando el nivel 0 (p10; gana siempre con u_min = 0.15)."""
    return env.step(np.array([a]), np.array([0]))


def test_mascara_permite_un_intermedio_ya_visitado(sintetico):
    env = _env(sintetico)
    m = env.reset(np.array([0]), np.array([0]), world_seeds=np.array([1]))[0]
    assert m[0] == 1 and env.opciones_validas[0, 0].tolist() == [False] * N_NIVELES + [True]   # en casa: esperar
    for a in (1, 2, 1, 3):                                         # 1 se recorre dos veces
        r, term, masks = _paso(env, a)
        m = masks[0]
        assert not term[0] and env.last_gano[0]
        assert m[a] == 1 and not env.opciones_validas[0, a, :N_NIVELES].any()   # self-loop: solo esperar
        assert m[0] == 1                                           # el retorno a casa siempre disponible
        if a == 2:
            assert m[1] == 1                                       # 1 ya fue recorrido y sigue disponible
    _, term, _ = _paso(env, 0)                                     # volver a casa termina el episodio
    assert term[0]


def test_retorno_forzado_por_k_con_revisita(sintetico):
    env = _env(sintetico)
    env.reset(np.array([0]), np.array([0]), k_max_arr=[4], world_seeds=np.array([1]))
    for a in (1, 2, 1):                                            # 0→1→2→1: queda un arco
        _, _, m = _paso(env, a)
    assert m[0].tolist() == [1, 1, 0, 0, 0]                        # el retorno o esperar en 1
    assert env.k_remaining[0] == 1


def test_max_pasos_acota_los_episodios_y_hay_episodios_con_mas_de_n_pasos(sintetico):
    d = sintetico
    B = 400
    env = _env(d)
    assert max_pasos(d["T"], MAX_SYN) == 31
    assert env.max_steps == max_pasos_cotizacion(d["T"], MAX_SYN) == (31 + max_esperas(MAX_SYN)) * (COTIZACIONES_MAX_POR_NODO + 1)
    rng = np.random.default_rng(0)
    env.reset(rng.integers(0, N, size=B), np.zeros(B, dtype=np.int64), world_seeds=np.arange(B))
    active = env.active.copy()
    for _ in range(env.max_steps):
        if not active.any():
            break
        pares = [np.argwhere(o) for o in env.opciones_validas]
        acc = np.array([p[rng.integers(len(p))] if len(p) else (0, OPCION_VACIO) for p in pares])
        _, term, _ = env.step(acc[:, 0], acc[:, 1])
        active &= ~term
    assert not active.any()
    assert env.step_count.max() > N                                # solo es posible si se revisitan nodos
    assert env.step_count.max() <= max_pasos(d["T"], MAX_SYN)


def test_max_pasos_con_datos_reales():
    for n in (10, 20, 35):
        tm = load_matrices(n)[0]
        T = np.asarray(tm, dtype=float)
        t_min = T[~np.eye(n, dtype=bool)].min()
        assert max_pasos(T, MAX_DURATION) == int(MAX_DURATION // t_min) + 1


# ─────────────────────────────────────────────────────────────────────────────
# UB-PI contra fuerza bruta
# ─────────────────────────────────────────────────────────────────────────────

def _fuerza_bruta_ub_pi(h, loads, rates, K, T, dist, diesel, max_d, dt, start_day=0, retrasos=0):
    """Mejor margen sobre TODAS las rutas (con revisitas) con las reglas del entorno y la misma
    discretización que solve_ub_pi: franjas de dt, tiempos redondeados hacia abajo; en cada nodo, ir cargado
    (si hay carga), vacío (≤ 300 mi, o a casa desde un intermedio) o esperar al siguiente día; y, para una
    salida en la franja s, la envolvente (máximo de la recompensa, unión de la disponibilidad) de los días
    start_day + floor(s·dt/14) y start_day + floor((s + max_pasos)·dt/14), recortados al último día del mundo.
    Sin ninguna jugada desde casa: 0 (no sale)."""
    n = T.shape[0]
    f = np.floor(T / dt + 1e-9).astype(int)
    smax = int(np.floor(max_d / dt + 1e-9))
    D = loads.shape[0]
    deriva = max_pasos(T, max_d)
    pd = franjas_por_dia(dt)
    tau = int(np.floor(COTIZACION_HORAS / dt + 1e-9))
    rep = vacio_permitido(dist)
    r0 = lane_reward(np.zeros_like(dist), 0, dist, diesel)
    vuelta = (f + f.T).astype(float)
    np.fill_diagonal(vuelta, np.inf)
    mejor = [-np.inf]

    def rec(i, s, acc, arcos, en_casa):
        d = min(start_day + int(s * dt / 14 + 1e-9), D - 1)
        e = min(start_day + int((s + deriva) * dt / 14 + 1e-9), D - 1)
        r = np.maximum(lane_reward(rates[d, i], loads[d, i], dist[i], diesel[i]),
                       lane_reward(rates[e, i], loads[e, i], dist[i], diesel[i]))
        existe = (loads[d, i] > 0) | (loads[e, i] > 0)
        if not en_casa:
            mejor[0] = max(mejor[0], acc + (r[h] if existe[h] else r0[i, h]))   # cerrar el ciclo
        sw = (s // pd + 1) * pd                                                # esperar al día siguiente
        if sw + (vuelta[h].min() if en_casa else f[i, h]) <= smax:
            rec(i, sw, acc, arcos, en_casa)
        if K is not None and arcos + 2 > K:
            return                                                 # otro arco no dejaría lugar al retorno
        for j in range(n):
            for q in range(retrasos + 1):                         # salir con q retrasos (mismo tablero, más tarde)
                sq = s + q * tau
                if j == i or j == h or sq + f[i, j] + f[j, h] > smax:
                    continue
                if existe[j]:
                    rec(j, sq + f[i, j], acc + r[j], arcos + 1, False)
                elif rep[i, j]:
                    rec(j, sq + f[i, j], acc + r0[i, j], arcos + 1, False)

    rec(h, 0, 0.0, 0, True)
    return mejor[0] if np.isfinite(mejor[0]) else 0.0


def _margen_de_ruta(ruta, loads, rates, T, dist, diesel, dt, start_day=0):
    """Suma de lane_reward de la ruta con la regla de franjas y el día de la franja (sin envolvente)."""
    f = np.floor(T / dt + 1e-9).astype(int)
    D = loads.shape[0]
    s, tot = 0, 0.0
    for a, b in zip(ruta[:-1], ruta[1:]):
        d = min(start_day + int(s * dt / 14 + 1e-9), D - 1)
        tot += float(lane_reward(rates[d, a], loads[d, a], dist[a], diesel[a])[b])
        s += int(f[a, b])
    return tot


@pytest.fixture(scope="module")
def mundo_chico():
    asegurar_parametros()
    n = 6
    (_, _, _, dist, diesel, *_) = load_matrices(n)
    rng = np.random.default_rng(5)
    T = rng.uniform(6.0, 12.0, size=(n, n))                        # viajes cortos: caben varias revisitas
    np.fill_diagonal(T, 0.0)
    return dict(n=n, T=T, dist=dist, diesel=diesel, market=MarketProcess(n))


@pytest.mark.parametrize("K", [4, None])
def test_ub_pi_coincide_con_fuerza_bruta(mundo_chico, K):
    m = mundo_chico
    max_d = 40.0 if K is None else 47.0                            # caso libre: hasta 6 arcos
    dt = 0.25
    for seed, dia in ((11, 3), (12, 40), (13, 77)):
        loads, rates = m["market"].sample_world(seed, 6, dia)
        for h in range(m["n"]):
            valor, ruta = solve_ub_pi(h, m["T"], rates, loads, m["dist"], m["diesel"], max_d,
                                      start_day_idx=0, k_max=K, dt=dt, retrasos=1)
            esperado = _fuerza_bruta_ub_pi(h, loads, rates, K, m["T"], m["dist"], m["diesel"], max_d, dt, retrasos=1)
            assert valor == pytest.approx(esperado, abs=1e-6), (seed, h, K)
            arcos = sum(a != b for a, b in zip(ruta[:-1], ruta[1:]))           # (i, i) = espera
            assert ruta[0] == ruta[-1] == h and (K is None or arcos <= K)


def test_ub_pi_con_dia_de_inicio_y_franja_de_un_minuto(mundo_chico):
    """start_day_idx desplaza la fila del día; con la franja por defecto (1 min) sigue igual a la fuerza bruta."""
    m = mundo_chico
    loads, rates = m["market"].sample_world(21, 6, 10)
    dt = 1.0 / 60.0
    valor, ruta = solve_ub_pi(1, m["T"], rates, loads, m["dist"], m["diesel"], 40.0, start_day_idx=2, k_max=4, retrasos=0)
    assert valor == pytest.approx(
        _fuerza_bruta_ub_pi(1, loads, rates, 4, m["T"], m["dist"], m["diesel"], 40.0, dt, start_day=2), abs=1e-6)


def _mejor_ruta_exacta(h, loads, rates, K, T, dist, diesel, max_d, start_day=0):
    """Mejor margen sobre todas las rutas con tiempos exactos (sin franjas) y las reglas del entorno:
    el día de cada salida es start_day + int(t // 14); se va cargado (con carga), vacío (≤ 300 mi o a casa)
    o se espera al siguiente día; ir a j exige poder volver a casa (t + T_ij + T_jh <= max_d) y cerrar exige
    llegar a tiempo."""
    n = T.shape[0]
    D = loads.shape[0]
    rep = vacio_permitido(dist)
    r0 = lane_reward(np.zeros_like(dist), 0, dist, diesel)
    vuelta = T + T.T
    np.fill_diagonal(vuelta, np.inf)
    mejor = [-np.inf]

    def rec(i, t, acc, arcos, en_casa):
        d = min(start_day + int(t // 14), D - 1)
        r = lane_reward(rates[d, i], loads[d, i], dist[i], diesel[i])
        if not en_casa and t + T[i, h] <= max_d:
            mejor[0] = max(mejor[0], acc + r[h])
        tw = float(fin_espera(t))
        if tw + (vuelta[h].min() if en_casa else T[i, h]) <= max_d:
            rec(i, tw, acc, arcos, en_casa)
        if K is not None and arcos + 2 > K:
            return
        for j in range(n):
            if j == i or j == h or t + T[i, j] + T[j, h] > max_d:
                continue
            if loads[d, i, j] > 0:
                rec(j, t + T[i, j], acc + r[j], arcos + 1, False)
            elif rep[i, j]:
                rec(j, t + T[i, j], acc + r0[i, j], arcos + 1, False)

    rec(h, 0.0, 0.0, 0, True)
    return mejor[0]


@pytest.mark.parametrize("K", [4, None])
def test_ub_pi_es_cota_de_la_ruta_exacta(mundo_chico, K):
    """Con la franja por defecto (1 min), UB-PI >= el mejor margen con tiempos exactos y el día real de cada salida."""
    m = mundo_chico
    max_d = 40.0
    for seed, dia in ((31, 5), (32, 50), (33, 80), (34, 11)):
        loads, rates = m["market"].sample_world(seed, 6, dia)
        for h in range(m["n"]):
            valor, _ = solve_ub_pi(h, m["T"], rates, loads, m["dist"], m["diesel"], max_d, k_max=K)
            exacto = _mejor_ruta_exacta(h, loads, rates, K, m["T"], m["dist"], m["diesel"], max_d)
            assert exacto == -np.inf or valor >= exacto - 1e-6, (seed, h, K, valor, exacto)


def test_ub_pi_elige_revisitar_cuando_conviene():
    """Dos lanes muy rentables (1→2 y 2→1): con K = 4 la mejor ruta es casa→1→2→1→casa."""
    n = 5
    T = np.full((n, n), 8.0)
    np.fill_diagonal(T, 0.0)
    dist = np.full((n, n), 100.0)
    dist[1, 2] = dist[2, 1] = 500.0
    diesel = np.full((n, n), 3.0)
    rates = np.full((3, n, n), 3.0)
    loads = np.full((3, n, n), 5.0)
    valor, ruta = solve_ub_pi(0, T, rates, loads, dist, diesel, MAX_DURATION, k_max=4)
    ruta = [n for k, n in enumerate(ruta) if k == 0 or n != ruta[k - 1]]   # sin esperas
    inter = ruta[1:-1]
    assert len(ruta) - 1 == 4 and len(set(inter)) < len(inter), ruta
    assert valor == pytest.approx(_margen_de_ruta(ruta, loads, rates, T, dist, diesel, 1.0 / 60.0))


def test_ub_pi_sin_ruta_factible():
    n = 4
    T = np.full((n, n), 60.0)
    np.fill_diagonal(T, 0.0)
    d = np.full((n, n), 100.0)
    v, ruta = solve_ub_pi(0, T, np.full((2, n, n), 3.0), np.full((2, n, n), 5.0), d, np.full((n, n), 3.0), 77.0)
    assert v == 0.0 and ruta == [0]                                # ida y vuelta = 120 h > 77 h: no sale
    v, ruta = solve_ub_pi(0, np.full((n, n), 5.0) - 5.0 * np.eye(n), np.full((2, n, n), 3.0),
                          np.full((2, n, n), 5.0), d, np.full((n, n), 3.0), 77.0, k_max=1)
    assert np.isnan(v) and ruta is None                            # K = 1 no permite salir y volver


# ─────────────────────────────────────────────────────────────────────────────
# B2: recursión memoizada contra enumeración ingenua
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def b2_chico():
    asegurar_parametros()
    n = 5
    (_, _, _, dist, diesel, *_) = load_matrices(n)
    rng = np.random.default_rng(2)
    T = rng.uniform(6.0, 10.0, size=(n, n))
    np.fill_diagonal(T, 0.0)
    market = MarketProcess(n)
    return dict(n=n, T=T, dist=dist, diesel=diesel, market=market,
                pol=PoliticaB2(market, T, dist, diesel, max_duration=40.0))


def _mejor_ruta_ingenua(pol, h, dow0, j, s, k):
    """Máximo sobre TODAS las secuencias j → l1 → … → casa (con revisitas y esperas al día siguiente) del
    valor esperado."""
    n, f, ev, dt = pol.N, pol._f, pol._ev, pol._dt
    pd = pol._por_dia
    mejor = [-np.inf]

    def rec(cur, s, acc, restantes):
        dow = (dow0 + int(s * dt / 14 + 1e-9)) % 7
        mejor[0] = max(mejor[0], acc + pol._ev_casa[dow, cur, h])  # cerrar el ciclo
        sw = (s // pd + 1) * pd
        if sw + f[cur, h] <= pol._smax:                            # esperar en cur (no consume k)
            rec(cur, sw, acc, restantes)
        if restantes is not None and restantes < 2:
            return
        for l in range(n):
            if l in (cur, h) or s + f[cur, l] + f[l, h] > pol._smax:
                continue
            rec(l, s + int(f[cur, l]), acc + ev[dow, cur, l], None if restantes is None else restantes - 1)

    rec(j, s, 0.0, k)
    return mejor[0]


def test_b2_memoizado_coincide_con_enumeracion_ingenua(b2_chico):
    pol, n = b2_chico["pol"], b2_chico["n"]
    rng = np.random.default_rng(7)
    for _ in range(60):
        h, j = rng.choice(n, size=2, replace=False)
        s = int(rng.integers(0, 80))
        k = [None, 2, 3, 4][int(rng.integers(0, 4))]
        dow0 = int(rng.integers(0, 7))
        memo = pol._memo.setdefault((int(h), dow0), {})
        assert pol._valor(int(h), dow0, int(j), s, k, memo) == pytest.approx(
            _mejor_ruta_ingenua(pol, int(h), dow0, int(j), s, k), abs=1e-9)


def test_b2_elige_el_primer_arco_de_la_enumeracion_ingenua(b2_chico):
    pol, n, T = b2_chico["pol"], b2_chico["n"], b2_chico["T"]
    rng = np.random.default_rng(9)
    for _ in range(40):
        h = int(rng.integers(0, n))
        i = int(rng.integers(0, n))
        t = float(rng.uniform(0.0, 14.0))
        k = [None, 2, 3, 4][int(rng.integers(0, 4))]
        dow0 = int(rng.integers(0, 7))
        mascara = np.ones(n, dtype=np.int8)
        mascara[i] = 0
        loads_row = rng.integers(0, 4, size=n).astype(float)
        rate_row = rng.uniform(1.5, 3.0, size=n)
        opciones = np.zeros((n, N_NIVELES + 1), dtype=bool)
        opciones[:, :N_NIVELES] = (loads_row > 0)[:, None]
        opciones[:, OPCION_VACIO] = loads_row == 0
        opciones[i] = False
        obs = Observacion(nodo=i, casa=h, tiempo=t, k_restante=k, mascara=mascara,
                          loads_row=loads_row, rate_row=rate_row,
                          precios=rate_row[:, None] * (0.8 + 0.5 * U_NIVELES_ARR), m=float(rng.uniform(1, 5)),
                          opciones=opciones, dia_semana_salida=dow0)
        r = valor_inmediato(obs, b2_chico["dist"], b2_chico["diesel"], cierre=True)
        mejor, mejor_a = -np.inf, -1
        for a in np.flatnonzero(mascara):
            if a == h:
                v = float(r[a])
            else:
                s_a = min(int(np.floor((t + T[i, a]) / pol._dt + 1e-9)), pol._smax)
                v = float(r[a]) + _mejor_ruta_ingenua(pol, h, dow0, int(a), s_a, None if k is None else k - 1)
            if v > mejor:
                mejor, mejor_a = v, int(a)
        assert pol.actuar(obs) == mejor_a
