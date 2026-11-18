"""
test_revisit.py
===============
El camión puede volver a pasar por un nodo intermedio dentro del tour. Pruebas de lo que cambia:

  · La máscara permite un intermedio ya recorrido, con el mismo resultado en RoutingEnv y en
    VectorRoutingEnv; el self-loop, el retorno y el retorno forzado por K siguen igual.
  · max_pasos acota la longitud de los episodios (reales y sintéticos) y hay episodios más largos
    que N (se revisita).
  · UB-PI (DP hacia atrás) coincide con una enumeración por fuerza bruta de todas las rutas,
    con revisitas, en mundos pequeños (N = 5 o 6, K = 4 y caso libre), y es cota del valor de
    cada ruta evaluada con tiempos exactos y el día real de cada salida.
  · B2 (recursión memoizada) coincide con la enumeración ingenua de rutas (N = 5).
"""

import numpy as np
import pytest

from baselines.politicas import Observacion, PoliticaB2
from config import MAX_DURATION
from market_process import MarketProcess, asegurar_parametros
from problem_data import build_rm_pen_stack, lane_reward, load_matrices
from routing_env import RoutingEnv, VectorRoutingEnv, max_pasos
from baselines.ub_pi import solve_ub_pi

# ─────────────────────────────────────────────────────────────────────────────
# Máscara con revisitas: RoutingEnv == VectorRoutingEnv
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
    rate = np.full((3, N, N), 3.0)
    loads = np.full((3, N, N), 5.0)
    return dict(T=T, dist=dist, diesel=diesel, rate=rate, loads=loads,
                rm=build_rm_pen_stack(rate, loads, dist, diesel))


def _envs(d, k_max=None):
    esc = RoutingEnv(d["T"], d["rate"], d["loads"], d["dist"], d["diesel"], N, max_duration=MAX_SYN)
    _, info = esc.reset(options={"start_node": 0, "start_day_idx": 0, "k_max": k_max})
    vec = VectorRoutingEnv(d["T"], d["rate"], d["loads"], d["rm"], N, max_duration=MAX_SYN)
    masks = vec.reset(np.array([0]), np.array([0]), k_max_arr=[k_max])
    return esc, info["action_mask"], vec, masks[0]


def test_mascara_permite_un_intermedio_ya_visitado_en_ambos_entornos(sintetico):
    esc, m_esc, vec, m_vec = _envs(sintetico)
    assert (m_esc == m_vec).all() and m_esc[0] == 0               # primer paso: self-loop en casa enmascarado
    for a in (1, 2, 1, 3):                                         # 1 se recorre dos veces
        _, r_esc, term_esc, _, info = esc.step(a)
        r_vec, term_vec, m = vec.step(np.array([a]))
        m_esc, m_vec = info["action_mask"], m[0]
        assert (m_esc == m_vec).all()
        assert r_esc == pytest.approx(float(r_vec[0]))
        assert not term_esc and not term_vec[0]
        assert m_esc[a] == 0                                       # self-loop siempre enmascarado
        assert m_esc[0] == 1                                       # el retorno a casa siempre disponible
        if a == 2:
            assert m_esc[1] == 1                                   # 1 ya fue recorrido y sigue disponible
    _, _, term, _, _ = esc.step(0)                                 # volver a casa termina el episodio
    _, term_vec, _ = vec.step(np.array([0]))
    assert term and term_vec[0]


def test_retorno_forzado_por_k_con_revisita(sintetico):
    esc, m_esc, vec, m_vec = _envs(sintetico, k_max=4)
    for a in (1, 2, 1):                                            # 0→1→2→1: queda un arco
        _, _, _, _, info = esc.step(a)
        _, _, m = vec.step(np.array([a]))
    assert (info["action_mask"] == m[0]).all()
    assert info["action_mask"].tolist() == [1, 0, 0, 0, 0]         # solo el retorno
    assert esc.k_remaining == 1 and vec.k_remaining[0] == 1


def test_max_pasos_acota_los_episodios_y_hay_episodios_con_mas_de_n_pasos(sintetico):
    d = sintetico
    B = 400
    vec = VectorRoutingEnv(d["T"], d["rate"], d["loads"], d["rm"], N, max_duration=MAX_SYN)
    assert vec.max_steps == max_pasos(d["T"], MAX_SYN) == 31
    rng = np.random.default_rng(0)
    masks = vec.reset(rng.integers(0, N, size=B), np.zeros(B, dtype=np.int64))
    active = np.ones(B, dtype=bool)
    for _ in range(vec.max_steps):
        if not active.any():
            break
        acc = np.array([rng.choice(np.flatnonzero(m)) for m in masks])
        _, term, masks = vec.step(acc)
        active &= ~term
    assert not active.any()
    assert vec.step_count.max() > N                                # solo es posible si se revisitan nodos
    assert vec.step_count.max() <= vec.max_steps


def test_max_pasos_con_datos_reales():
    for n in (10, 20, 35):
        tm = load_matrices(n)[0]
        T = np.asarray(tm, dtype=float)
        t_min = T[~np.eye(n, dtype=bool)].min()
        assert max_pasos(T, MAX_DURATION) == int(MAX_DURATION // t_min) + 1


# ─────────────────────────────────────────────────────────────────────────────
# UB-PI contra fuerza bruta
# ─────────────────────────────────────────────────────────────────────────────

def _fuerza_bruta_ub_pi(h, loads, rates, K, T, dist, diesel, max_d, dt, start_day=0):
    """Mejor margen sobre TODAS las rutas (con revisitas) con las reglas del entorno y la misma
    discretización que solve_ub_pi: franjas de dt, tiempos redondeados hacia abajo, lane con carga
    salvo el primer movimiento, y, para una salida en la franja s, la envolvente (máximo de la
    recompensa, unión de la disponibilidad) de los días start_day + floor(s·dt/14) y
    start_day + floor((s + max_pasos)·dt/14), recortados al último día del mundo."""
    n = T.shape[0]
    f = np.floor(T / dt + 1e-9).astype(int)
    smax = int(np.floor(max_d / dt + 1e-9))
    D = loads.shape[0]
    deriva = max_pasos(T, max_d)
    mejor = [-np.inf]

    def rec(i, s, acc, arcos, primero):
        d = min(start_day + int(s * dt / 14 + 1e-9), D - 1)
        e = min(start_day + int((s + deriva) * dt / 14 + 1e-9), D - 1)
        r = np.maximum(lane_reward(rates[d, i], loads[d, i], dist[i], diesel[i]),
                       lane_reward(rates[e, i], loads[e, i], dist[i], diesel[i]))
        existe = (loads[d, i] > 0) | (loads[e, i] > 0)
        if i != h:
            mejor[0] = max(mejor[0], acc + r[h])                   # cerrar el ciclo
        if K is not None and arcos + 2 > K:
            return                                                 # otro arco no dejaría lugar al retorno
        for j in range(n):
            if j == i or j == h:
                continue
            if not primero and not existe[j]:
                continue
            if s + f[i, j] + f[j, h] > smax:
                continue
            rec(j, s + f[i, j], acc + r[j], arcos + 1, False)

    rec(h, 0, 0.0, 0, True)
    return mejor[0]


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
                                      start_day_idx=0, k_max=K, dt=dt)
            esperado = _fuerza_bruta_ub_pi(h, loads, rates, K, m["T"], m["dist"], m["diesel"], max_d, dt)
            if ruta is None:
                assert esperado == -np.inf
                continue
            assert valor == pytest.approx(esperado, abs=1e-6), (seed, h, K)
            assert ruta[0] == ruta[-1] == h and (K is None or len(ruta) - 1 <= K)
            assert _margen_de_ruta(ruta, loads, rates, m["T"], m["dist"], m["diesel"], dt) <= valor + 1e-6


def test_ub_pi_con_dia_de_inicio_y_franja_de_un_minuto(mundo_chico):
    """start_day_idx desplaza la fila del día; con la franja por defecto (1 min) sigue igual a la fuerza bruta."""
    m = mundo_chico
    loads, rates = m["market"].sample_world(21, 6, 10)
    dt = 1.0 / 60.0
    valor, ruta = solve_ub_pi(1, m["T"], rates, loads, m["dist"], m["diesel"], 40.0, start_day_idx=2, k_max=4)
    assert valor == pytest.approx(
        _fuerza_bruta_ub_pi(1, loads, rates, 4, m["T"], m["dist"], m["diesel"], 40.0, dt, start_day=2), abs=1e-6)


def _mejor_ruta_exacta(h, loads, rates, K, T, dist, diesel, max_d, start_day=0):
    """Mejor margen sobre todas las rutas con tiempos exactos (sin franjas) y las reglas del entorno:
    el día de cada salida es start_day + int(t // 14), la lane necesita carga salvo en el primer
    movimiento, ir a j exige poder volver a casa (t + T_ij + T_jh <= max_d)."""
    n = T.shape[0]
    D = loads.shape[0]
    mejor = [-np.inf]

    def rec(i, t, acc, arcos, primero):
        d = min(start_day + int(t // 14), D - 1)
        r = lane_reward(rates[d, i], loads[d, i], dist[i], diesel[i])
        if i != h:
            mejor[0] = max(mejor[0], acc + r[h])
        if K is not None and arcos + 2 > K:
            return
        for j in range(n):
            if j == i or j == h or (not primero and loads[d, i, j] <= 0):
                continue
            if t + T[i, j] + T[j, h] > max_d:
                continue
            rec(j, t + T[i, j], acc + r[j], arcos + 1, False)

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
            assert (np.isnan(valor) and exacto == -np.inf) or valor >= exacto - 1e-6, (seed, h, K, valor, exacto)


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
    inter = ruta[1:-1]
    assert len(ruta) - 1 == 4 and len(set(inter)) < len(inter), ruta
    assert valor == pytest.approx(_margen_de_ruta(ruta, loads, rates, T, dist, diesel, 1.0 / 60.0))


def test_ub_pi_sin_ruta_factible():
    n = 4
    T = np.full((n, n), 60.0)
    np.fill_diagonal(T, 0.0)
    d = np.full((n, n), 100.0)
    v, ruta = solve_ub_pi(0, T, np.full((2, n, n), 3.0), np.full((2, n, n), 5.0), d, np.full((n, n), 3.0), 77.0)
    assert np.isnan(v) and ruta is None                            # ida y vuelta = 120 h > 77 h
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
    """Máximo sobre TODAS las secuencias j → l1 → … → casa (con revisitas) del valor esperado."""
    n, f, ev, dt = pol.N, pol._f, pol._ev, pol._dt
    mejor = [-np.inf]

    def rec(cur, s, acc, restantes):
        dow = (dow0 + int(s * dt / 14 + 1e-9)) % 7
        mejor[0] = max(mejor[0], acc + ev[dow, cur, h])            # cerrar el ciclo
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
        obs = Observacion(nodo=i, casa=h, tiempo=t, k_restante=k, mascara=mascara,
                          loads_row=loads_row, rate_row=rate_row, dia_semana_salida=dow0)
        r = lane_reward(rate_row, loads_row, b2_chico["dist"][i], b2_chico["diesel"][i])
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
