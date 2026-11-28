"""
test_baselines.py
=================
Baselines de baselines_drl_ruteo_estocastico.md (baselines/), todos en lote en el dispositivo:

  · UB-DP: la tabla V vectorizada coincide con una enumeración ingenua (dos intentos de cotización, mismo tablero);
    si no es tratable se deja el marcador y B3 / UB-DP se reportan así.
  · UB-PI (DP hacia atrás en lote): coincide con la fuerza bruta de todas las rutas (con revisitas, esperas y
    retrasos), es cota de la mejor ruta con tiempos exactos, revisita cuando conviene y vale 0 si no hay salida.
  · B1 desempata por el índice menor; B2 (tabla hacia atrás) coincide con la enumeración ingenua de rutas y elige
    el primer arco de esa enumeración; con K libre equivale a un K inalcanzable; B2P con índice cero es B2 y es
    monótono en el índice; B3.precio_en coincide con su propia opción.
  · Reglas de precio: en la vuelta a casa perder es volver vacío.
  · En el arnés: todas las políticas respetan la máscara y son deterministas; UB-PI ≥ cada política en cada mundo;
    B3 ≤ UB-DP (salvo Monte Carlo); un ciclo fallido vale la penalización.
  · Estadística (Holm, comparación pareada, notas) y limpieza del repositorio.
"""

import glob
import os
import re

import torch  # noqa: F401  (antes que pandas)
import numpy as np
import pandas as pd
import pytest

import evaluation as ev
import stats_analysis as st
from baselines.dp_relajada import (calcular_tabla, cargar_o_calcular_tabla, franjas_rechazo, _sortear_tableros,
                                   PENALIZACION_USD)
from baselines.politicas import (ObsLote, PoliticaB1, PoliticaB2, PoliticaB2P, PoliticaB3, _Estatico, _LIBRE,
                                 cotizar_regla, valor_inmediato)
from baselines.ub_pi import franjas_por_dia, solve_ub_pi_lote
from config import (COTIZACION_HORAS, COTIZACIONES_MAX_POR_NODO, DEVICE, K_MAX, MAX_DURATION, MARKET_SEED_EVAL,
                    MARKET_SEED_EVAL_FINAL, MARKET_SEED_TRAIN, TRAIN_DAYS)
from conftest import Datos, accion_al_azar, agente_chico, mundos_sim
from market_process import MarketProcess, asegurar_parametros
from pricing import N_NIVELES, U_NIVELES_ARR, lane_cost_row, nivel_regla, prob_ganar
from problem_data import lane_reward, load_matrices
from routing_env import OPCION_VACIO, fin_espera, max_pasos, vacio_permitido

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
N = 10


def np_(x):
    return x.detach().cpu().numpy()


@pytest.fixture(scope="module")
def sim10(d10, tabla10):
    return ev.evaluar_mundos_simulados(agente_chico(N), d10.market, tabla10, d10.tm, d10.dist, d10.diesel, N,
                                       d10.avail, d10.p95, M=100, verbose=False)


def _estados(d, B=60, pasos=1, semilla=0):
    """ObsLote de B episodios reales avanzados `pasos` pasos con acciones válidas al azar (K = 5 en las filas pares,
    libre en las impares), con la máscara del entorno."""
    env = d.env()
    mundos = mundos_sim(d.market, B, semilla=semilla)
    nodos = np.random.default_rng(semilla).integers(0, d.n, size=B)
    masks = env.reset(mundos, nodos, [None if b % 2 else K_MAX for b in range(B)])
    gen = torch.Generator(device=DEVICE).manual_seed(semilla)
    for _ in range(pasos):
        a, o = accion_al_azar(env, gen)
        _, _, masks = env.step(a, o)
    return env, ObsLote.del_entorno(env, masks)


def _obs(nodo, casa, n, precios, opciones, m=2.0, tiempo=0.0, k=None, intentos=None, dow0=0, indice=None):
    """ObsLote sintética de una fila."""
    t = lambda x, **kw: torch.as_tensor(x, device=DEVICE, **kw)
    opciones = np.asarray(opciones, dtype=bool)
    return ObsLote(
        nodo=t([nodo]), casa=t([casa]), tiempo=t([tiempo], dtype=torch.float64),
        k_rest=t([0 if k is None else k]), k_restr=t([k is not None]), mascara=t(opciones.any(-1)[None]),
        precios=t(np.asarray(precios, dtype=float)[None]), m=t([m], dtype=torch.float64), opciones=t(opciones[None]),
        dow0=t([dow0]), indice=t(np.zeros((1, n)) if indice is None else np.asarray(indice)[None], dtype=torch.float64),
        n_rech=t([0]), intentos=t((np.ones(n, dtype=np.int64) if intentos is None else np.asarray(intentos))[None]),
    )


# ─────────────────────────────────────────────────────────────────────────────
# UB-DP
# ─────────────────────────────────────────────────────────────────────────────

def test_tabla_dp_coincide_con_enumeracion_ingenua_de_los_dos_intentos():
    """Instancia chica (4 nodos, viajes de 6-10 h, franjas de 15 min, S = 4) donde sí hay revisitas y rechazos.

    La enumeración es la ecuación de especificacion_fase0_ruteo_pricing.md §7.3 escrita con loops: por cada
    tablero y r rechazos, la mejor de E (la mejor jugada sin carga: casa vacío, reposicionarse ≤ 300 mi o esperar
    al día siguiente) y (cotizar j al nivel k, si la lane tiene más de r cargas; si se pierde, lo mismo con r + 1). En casa sin haber salido (H): sin "volver"; sin ninguna jugada, 0 en s = 0 y la penalización
    después."""
    asegurar_parametros()
    n, S, dt, K = 4, 4, 0.25, 3
    _, _, _, dist, diesel, *_ = load_matrices(n)
    market = MarketProcess(n)
    rng = np.random.default_rng(3)
    T = rng.uniform(6.0, 10.0, size=(n, n))
    np.fill_diagonal(T, 0.0)
    tabla = calcular_tabla(market, T, dist, diesel, K=K, S=S, dt=dt, seed=11, device=torch.device("cpu"),
                           max_horas=None)

    tab = _sortear_tableros(market, dist, diesel, S, 11)
    A, D, Rk, P, R0 = tab["A"], tab["D"], tab["Rk"], tab["P"], tab["R0"]
    f = np.floor(T / dt + 1e-9).astype(int)
    smax = int(np.floor(MAX_DURATION / dt + 1e-9))
    tau = franjas_rechazo(dt)
    assert tau == 1
    C = COTIZACIONES_MAX_POR_NODO
    pd_ = franjas_por_dia(dt)
    rep = vacio_permitido(dist)
    niveles = range(Rk.shape[-1])
    memo, memo_h = {}, {}
    vuelta = f + f.T
    np.fill_diagonal(vuelta, smax + 1)

    def E_de(k, d0, h, i, t):
        """La mejor jugada sin carga desde el intermedio i en la franja t."""
        e = float(R0[i, h])
        sw = (t // pd_ + 1) * pd_
        if sw + f[i, h] <= smax:
            e = max(e, V(k, d0, h, i, sw))
        if k is None or k >= 2:
            for j in range(n):
                if rep[i, j] and j not in (i, h) and t + f[i, j] + f[j, h] <= smax:
                    e = max(e, float(R0[i, j]) + V(None if k is None else k - 1, d0, h, j, t + f[i, j]))
        return e

    def V(k, d0, h, i, s):
        clave = (k, d0, h, i, s)
        if clave in memo:
            return memo[clave]
        dw = (d0 + int((s * dt) // 14)) % 7
        ts = [min(s + r * tau, smax) for r in range(C + 1)]
        Es = [E_de(k, d0, h, i, t) for t in ts]
        vals = []
        for m in range(S):
            d = D[dw, i, m]

            def valido(j, t, r):
                if A[dw, i, m, j] <= r:
                    return False
                if t + tau + f[i, h] > smax:                        # si se pierde, no alcanza a volver
                    return False
                if j == h:
                    return True
                return (k is None or k >= 2) and j != i and t + f[i, j] + f[j, h] <= smax

            def G(j, lvl, t):
                cont = 0.0 if j == h else V(None if k is None else k - 1, d0, h, j, t + f[i, j])
                return float(Rk[dw, d, i, j, lvl]) + cont

            def intento(j, t, respaldo):
                return max(P[dw, d, i, l] * G(j, l, t) + (1 - P[dw, d, i, l]) * respaldo for l in niveles)

            b = Es[C]
            for r in range(C - 1, -1, -1):
                b = max([Es[r]] + [intento(j, ts[r], b) for j in range(n) if valido(j, ts[r], r)])
            vals.append(b)
        memo[clave] = float(np.mean(vals))
        return memo[clave]

    def H(k, d0, h, s):
        """Valor en casa sin haber salido, en la franja s (k arcos; None = libre)."""
        clave = (k, d0, h, s)
        if clave in memo_h:
            return memo_h[clave]
        ks = None if k is None else k - 1
        dw = (d0 + int((s * dt) // 14)) % 7
        ok = lambda j, t: j != h and t + f[h, j] + f[j, h] <= smax

        def sin_carga(t):
            e = -np.inf
            sw = (t // pd_ + 1) * pd_
            if sw + vuelta[h].min() <= smax:
                e = H(k, d0, h, sw)
            for j in range(n):
                if rep[h, j] and ok(j, t):
                    e = max(e, float(R0[h, j]) + V(ks, d0, h, j, t + f[h, j]))
            return e

        ts = [min(s + r * tau, smax) for r in range(C + 1)]
        Es = [sin_carga(t) for t in ts]
        falla = lambda x: x if np.isfinite(x) else PENALIZACION_USD   # después de perder, sin jugadas: falla
        vals = []
        for m in range(S):
            d = D[dw, h, m]
            G = lambda j, l, t: float(Rk[dw, d, h, j, l]) + V(ks, d0, h, j, t + f[h, j])
            intento = lambda j, t, resp: max(P[dw, d, h, l] * G(j, l, t) + (1 - P[dw, d, h, l]) * resp
                                             for l in niveles)
            mejor = falla(Es[C])
            for r in range(C - 1, -1, -1):
                mejor = max([Es[r]] + [intento(j, ts[r], mejor) for j in range(n)
                                       if A[dw, h, m, j] > r and ok(j, ts[r])])
                if r > 0:
                    mejor = falla(mejor)
            if not np.isfinite(mejor):
                mejor = 0.0 if s == 0 else PENALIZACION_USD        # no sale (s = 0) o falla (después)
            vals.append(mejor)
        memo_h[clave] = float(np.mean(vals))
        return memo_h[clave]

    def V0(k, d0, h):
        return H(k, d0, h, 0)

    for d0, h in ((0, 0), (3, 2)):
        for k in (2, 3):
            assert tabla.ub_dp(h, d0, k) == pytest.approx(V0(k, d0, h), rel=1e-4, abs=1e-2)
    assert tabla.ub_dp(1, 5, None) == pytest.approx(V0(None, 5, 1), rel=1e-4, abs=1e-2)
    for k, d0, h, i, s in ((1, 0, 0, 1, 150), (2, 2, 1, 3, 80), (3, 5, 2, 0, 0), (3, 4, 3, 2, 120)):
        assert float(tabla.V_k[k, d0, h, i, s]) == pytest.approx(V(k, d0, h, i, s), rel=1e-4, abs=1e-2)
    for d0, h, i, s in ((1, 0, 2, 60), (6, 2, 1, 160)):
        assert float(tabla.V_libre[d0, h, i, s]) == pytest.approx(V(None, d0, h, i, s), rel=1e-4, abs=1e-2)


def test_tabla_no_tratable_deja_el_marcador_y_b3_se_reporta_asi(tmp_path):
    import json
    d = Datos(4)
    ruta = str(tmp_path / "ubdp_tabla_4nodos.npz")
    assert cargar_o_calcular_tabla(d.market, d.tm, d.dist, d.diesel, S=20, path=ruta, verbose=False, max_horas=0.0) is None
    info = json.load(open(ruta.replace(".npz", ".no_tratable.json"), encoding="utf-8"))
    assert info["motivo"] == "tiempo" and info["num_nodes"] == 4 and 0 < info["fraccion"] <= 1
    assert cargar_o_calcular_tabla(d.market, d.tm, d.dist, d.diesel, S=20, path=ruta, verbose=False) is None
    df = ev.evaluar_mundos_simulados(agente_chico(4), d.market, None, d.tm, d.dist, d.diesel, 4, d.avail, d.p95,
                                     M=4, verbose=False)
    assert "B3" not in set(df["metodo"]) and df["ub_dp"].isna().all()
    res = ev.resumen_metodos(df)
    nt = res[res["metodo"].isin(["B3", "UB-DP"])]
    assert len(nt) == 2 * df["K"].nunique() and (nt["nota"] == ev.NOTA_NO_TRATABLE).all()


# ─────────────────────────────────────────────────────────────────────────────
# UB-PI contra fuerza bruta
# ─────────────────────────────────────────────────────────────────────────────

def _fuerza_bruta_ub_pi(h, loads, rates, K, T, dist, diesel, max_d, dt, retrasos=0):
    """Mejor margen sobre TODAS las rutas (con revisitas) con las reglas del entorno y la discretización de
    solve_ub_pi_lote: franjas de dt, tiempos redondeados hacia abajo; en cada nodo, ir cargado (si hay carga), vacío
    (≤ 300 mi, o a casa desde un intermedio) o esperar al siguiente día; para una salida en la franja s, la
    envolvente de los días floor(s·dt/14) y floor((s + max_pasos)·dt/14). Sin jugadas desde casa: 0 (no sale)."""
    n = T.shape[0]
    f = np.floor(T / dt + 1e-9).astype(int)
    smax = int(np.floor(max_d / dt + 1e-9))
    D = loads.shape[0]
    deriva = max_pasos(T, max_d)
    pd_ = franjas_por_dia(dt)
    tau = int(np.floor(COTIZACION_HORAS / dt + 1e-9))
    rep = vacio_permitido(dist)
    r0 = lane_reward(np.zeros_like(dist), 0, dist, diesel)
    vuelta = (f + f.T).astype(float)
    np.fill_diagonal(vuelta, np.inf)
    mejor = [-np.inf]

    def rec(i, s, acc, arcos, en_casa):
        d = min(int(s * dt / 14 + 1e-9), D - 1)
        e = min(int((s + deriva) * dt / 14 + 1e-9), D - 1)
        r = np.maximum(lane_reward(rates[d, i], loads[d, i], dist[i], diesel[i]),
                       lane_reward(rates[e, i], loads[e, i], dist[i], diesel[i]))
        existe = (loads[d, i] > 0) | (loads[e, i] > 0)
        if not en_casa:
            mejor[0] = max(mejor[0], acc + (r[h] if existe[h] else r0[i, h]))
        sw = (s // pd_ + 1) * pd_
        if sw + (vuelta[h].min() if en_casa else f[i, h]) <= smax:
            rec(i, sw, acc, arcos, en_casa)
        if K is not None and arcos + 2 > K:
            return
        for j in range(n):
            for q in range(retrasos + 1):
                sq = s + q * tau
                if j == i or j == h or sq + f[i, j] + f[j, h] > smax:
                    continue
                if existe[j]:
                    rec(j, sq + f[i, j], acc + r[j], arcos + 1, False)
                elif rep[i, j]:
                    rec(j, sq + f[i, j], acc + r0[i, j], arcos + 1, False)

    rec(h, 0, 0.0, 0, True)
    return mejor[0] if np.isfinite(mejor[0]) else 0.0


def _mejor_ruta_exacta(h, loads, rates, K, T, dist, diesel, max_d):
    """Mejor margen sobre todas las rutas con tiempos exactos y el día real de cada salida (sin franjas)."""
    n, D = T.shape[0], loads.shape[0]
    rep = vacio_permitido(dist)
    r0 = lane_reward(np.zeros_like(dist), 0, dist, diesel)
    vuelta = T + T.T
    np.fill_diagonal(vuelta, np.inf)
    mejor = [-np.inf]

    def rec(i, t, acc, arcos, en_casa):
        d = min(int(t // 14), D - 1)
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


@pytest.fixture(scope="module")
def mundo_chico():
    """6 nodos con viajes cortos (6-12 h: caben varias revisitas) y 3 mundos simulados por casa (loads, tarifas de
    evaluation.mundos_ub_pi)."""
    asegurar_parametros()
    n = 6
    _, _, _, dist, diesel, *_ = load_matrices(n)
    T = np.random.default_rng(5).uniform(6.0, 12.0, size=(n, n))
    np.fill_diagonal(T, 0.0)
    market = MarketProcess(n)
    mundos = mundos_sim(market, 3 * n, semilla=11)
    loads, rates = ev.mundos_ub_pi(market, mundos)
    return dict(n=n, T=T, dist=dist, diesel=diesel, loads=loads, rates=rates, casas=np.repeat(np.arange(n), 3))


@pytest.mark.parametrize("K", [4, None])
def test_ub_pi_coincide_con_fuerza_bruta(mundo_chico, K):
    m = mundo_chico
    max_d = 40.0 if K is None else 47.0                            # caso libre: hasta 6 arcos
    v = solve_ub_pi_lote(m["casas"], m["T"], m["rates"], m["loads"], m["dist"], m["diesel"], max_d, k_max=K,
                         dt=0.25, retrasos=1)
    for w, h in enumerate(m["casas"]):
        esperado = _fuerza_bruta_ub_pi(h, m["loads"][w], m["rates"][w], K, m["T"], m["dist"], m["diesel"], max_d,
                                       0.25, retrasos=1)
        assert v[w] == pytest.approx(esperado, abs=1e-6), (w, h, K)


@pytest.mark.parametrize("K", [4, None])
def test_ub_pi_es_cota_de_la_ruta_exacta(mundo_chico, K):
    """Con la franja por defecto (1 min), UB-PI ≥ el mejor margen con tiempos exactos y el día real de cada salida."""
    m = mundo_chico
    v = solve_ub_pi_lote(m["casas"], m["T"], m["rates"], m["loads"], m["dist"], m["diesel"], 40.0, k_max=K)
    for w, h in enumerate(m["casas"]):
        exacto = _mejor_ruta_exacta(h, m["loads"][w], m["rates"][w], K, m["T"], m["dist"], m["diesel"], 40.0)
        assert exacto == -np.inf or v[w] >= exacto - 1e-6, (w, h, K, v[w], exacto)


def test_ub_pi_revisita_cuando_conviene_y_sin_ruta_factible():
    """Dos lanes muy rentables (1→2 y 2→1): con K = 4 el valor es el de casa→1→2→1→casa."""
    n = 5
    T = np.full((n, n), 8.0)
    np.fill_diagonal(T, 0.0)
    dist = np.full((n, n), 100.0)
    dist[1, 2] = dist[2, 1] = 500.0
    diesel = np.full((n, n), 3.0)
    rates, loads = np.full((1, 3, n, n), 3.0), np.full((1, 3, n, n), 5)
    v = solve_ub_pi_lote([0], T, rates, loads, dist, diesel, MAX_DURATION, k_max=4)[0]
    ruta = [0, 1, 2, 1, 0]
    assert v == pytest.approx(sum(float(lane_reward(3.0, 5, dist[a, b], diesel[a, b])) for a, b in zip(ruta[:-1], ruta[1:])))
    assert v > solve_ub_pi_lote([0], T, rates, loads, dist, diesel, MAX_DURATION, k_max=3)[0]
    T = np.full((4, 4), 60.0)
    np.fill_diagonal(T, 0.0)
    d = np.full((4, 4), 100.0)
    args = (np.full((1, 2, 4, 4), 3.0), np.full((1, 2, 4, 4), 5), d, np.full((4, 4), 3.0), 77.0)
    assert solve_ub_pi_lote([0], T, *args)[0] == 0.0                       # ida y vuelta = 120 h > 77 h: no sale
    assert np.isnan(solve_ub_pi_lote([0], T, *args, k_max=1)[0])           # K = 1 no permite salir y volver


def test_ub_pi_regresion_con_datos_reales(d10):
    """Caso libre sobre los tableros reales de los días 0 … D−1 desde la casa 0 (capturado con el solver de antes:
    ruta 0 → 6 → 0). Si datos/ cambia hay que recapturarlo."""
    D = ev.MARKET_DIAS_EPISODIO
    v = solve_ub_pi_lote([0], d10.T, d10.rate[None, :D], d10.loads[None, :D], d10.dist, d10.diesel, MAX_DURATION)
    assert v[0] == 1328.0


def test_mundos_ub_pi_usa_el_competidor_de_cada_carga(d10):
    """La lane existe para UB-PI si alguno de los niveles le gana a alguna de sus cargas cotizables, y su tarifa es la
    del nivel más alto que gana (con e^δ)."""
    mp = d10.market
    mundos = mundos_sim(mp, 6, semilla=3)
    loads, rates = ev.mundos_ub_pi(mp, mundos)
    tab = mp.tensores(DEVICE)
    for b in range(6):
        for d in (0, 3):
            dia = min(int(mundos.dia0[b]) + d, mundos.dia_max)
            for i in (0, 4):
                m = float(tab["m"][dia, i])
                u = 1 - (1 - np_(mundos.u[b, d, i]).astype(np.float64)) ** (1 / m)       # (N, C)
                for j in range(N):
                    n = min(int(mundos.loads[b, d, i, j]), COTIZACIONES_MAX_POR_NODO)
                    mejor = u[j, :n].max() if n else 0.0
                    gana = U_NIVELES_ARR < mejor
                    assert (loads[b, d, i, j] > 0) == bool(gana.any())
                    if gana.any():
                        k = int(np.flatnonzero(gana).max())
                        p = float(tab["precios"][dia, i, j, k]) * np.exp(float(mundos.delta[b]))
                        assert rates[b, d, i, j] == pytest.approx(p)


# ─────────────────────────────────────────────────────────────────────────────
# B1, B2, B2P, B3
# ─────────────────────────────────────────────────────────────────────────────

def test_b1_es_el_mejor_valor_inmediato_y_desempata_por_indice_menor(d10):
    env, obs = _estados(d10, semilla=1)
    pol = PoliticaB1(d10.T, d10.dist, d10.diesel)
    a, o = pol.decidir(obs)
    v = np_(valor_inmediato(obs, pol._est(DEVICE)))
    mask = np_(obs.mascara)
    for b in range(obs.B):
        if mask[b].any():
            esperado = np.flatnonzero(mask[b])[np.argmax(v[b][mask[b]])]
            assert int(a[b]) == esperado
            assert np_(obs.opciones)[b, int(a[b]), int(o[b])]
    # todo empata (sin cargas, costos iguales): el índice menor
    n = 4
    op = np.zeros((n, N_NIVELES + 1), dtype=bool)
    op[[2, 3], OPCION_VACIO] = True
    plano = PoliticaB1(np.full((n, n), 10.0) - 10.0 * np.eye(n), np.ones((n, n)), np.ones((n, n)))
    a, o = plano.decidir(_obs(0, 0, n, np.ones((n, N_NIVELES)), op, intentos=np.zeros(n)))
    assert int(a[0]) == 2 and int(o[0]) == OPCION_VACIO


@pytest.fixture(scope="module")
def b2_chico():
    """B2 a 5 nodos (tiempos reales, MAX_DURATION): pocas rutas posibles, enumerables."""
    d = Datos(5)
    pol = PoliticaB2(d.market, d.T, d.dist, d.diesel)
    return d, pol, pol._est(DEVICE)


def _mejor_ruta_ingenua(pol, est, h, dow0, j, s, k):
    """Máximo sobre TODAS las secuencias j → l1 → … → casa (con revisitas y esperas al día siguiente) del valor
    esperado de B2 (ev de cada arco; ev_casa el de cierre)."""
    n, f, ev_, evc = est.N, np_(est.f), pol._ev_np, pol._ev_casa_np
    mejor = [-np.inf]

    def rec(cur, s, acc, restantes):
        dow = (dow0 + est.dia(s)) % 7
        mejor[0] = max(mejor[0], acc + evc[dow, cur, h])
        sw = (s // est.por_dia + 1) * est.por_dia
        if sw + f[cur, h] <= est.smax:
            rec(cur, sw, acc, restantes)
        if restantes is not None and restantes < 2:
            return
        for l in range(n):
            if l in (cur, h) or s + f[cur, l] + f[l, h] > est.smax:
                continue
            rec(l, s + int(f[cur, l]), acc + ev_[dow, cur, l], None if restantes is None else restantes - 1)

    rec(j, s, 0.0, k)
    return mejor[0]


def test_b2_tabla_coincide_con_enumeracion_ingenua(b2_chico):
    d, pol, est = b2_chico
    rng = np.random.default_rng(7)
    for _ in range(60):
        h, j = (int(x) for x in rng.choice(d.n, size=2, replace=False))
        s, dow0 = int(rng.integers(0, 120)), int(rng.integers(0, 7))
        k = [None, 1, 2, 3, 4][int(rng.integers(0, 5))]
        V = pol._tabla(_LIBRE if k is None else k, est)[0]
        assert float(V[h, dow0, j, s]) == pytest.approx(_mejor_ruta_ingenua(pol, est, h, dow0, j, s, k), abs=1e-9)


def test_b2_elige_el_primer_arco_de_la_enumeracion_ingenua(b2_chico):
    d, pol, est = b2_chico
    _, obs = _estados(d, B=80, pasos=1, semilla=2)
    a, _ = pol.decidir(obs)
    r = np_(valor_inmediato(obs, est, cierre=True))
    vistos = 0
    for b in range(obs.B):
        i, h, t = int(obs.nodo[b]), int(obs.casa[b]), float(obs.tiempo[b])
        mask = np_(obs.mascara[b])
        if i == h or not mask.any():
            continue                                    # en casa la espera usa la tabla de salida (S)
        k = int(obs.k_rest[b]) if bool(obs.k_restr[b]) else None
        dow0 = int(obs.dow0[b])
        v = np.full(d.n, -np.inf)
        for x in np.flatnonzero(mask):
            if x == h:
                v[x] = r[b, x]
            elif x == i:
                s_w = min(int(np.floor(float(fin_espera(t)) / est.dt + 1e-9)), est.smax)
                v[x] = _mejor_ruta_ingenua(pol, est, h, dow0, i, s_w, k)
            else:
                s_a = min(int(np.floor((t + d.T[i, x]) / est.dt + 1e-9)), est.smax)
                v[x] = r[b, x] + _mejor_ruta_ingenua(pol, est, h, dow0, int(x), s_a, None if k is None else max(k - 1, 1))
        orden = np.sort(v[np.isfinite(v)])[::-1]
        if len(orden) > 1 and orden[0] - orden[1] < 1e-6:
            continue                                    # empate numérico: cualquiera vale
        assert int(a[b]) == int(np.argmax(v)), (b, v)
        vistos += 1
    assert vistos > 5


def test_b2_libre_equivale_a_un_k_inalcanzable(d10):
    pol = PoliticaB2(d10.market, d10.T, d10.dist, d10.diesel)
    _, obs = _estados(d10, semilla=3)
    libre = ObsLote(**{**obs.__dict__, "k_restr": torch.zeros_like(obs.k_restr)})
    grande = ObsLote(**{**obs.__dict__, "k_restr": torch.ones_like(obs.k_restr),
                        "k_rest": torch.full_like(obs.k_rest, max_pasos(d10.T) + 1)})
    assert torch.equal(pol.decidir(libre)[0], pol.decidir(grande)[0])


def test_b2p_con_indice_cero_es_b2_y_es_monotono_en_el_indice(d10):
    b2 = PoliticaB2(d10.market, d10.T, d10.dist, d10.diesel)
    b2p = PoliticaB2P(d10.market, d10.T, d10.dist, d10.diesel)
    f, rho = d10.market.frac_var_lento, d10.market.rho
    assert b2p.b[0] == pytest.approx(1.0) and np.allclose(b2p.b, f + (1 - f) * rho ** np.arange(len(b2p.b)))
    assert (np.diff(b2p.b) < 0).all() and b2p.b[-1] > f
    _, obs = _estados(d10, B=40, semilla=4)
    cero = ObsLote(**{**obs.__dict__, "indice": torch.zeros_like(obs.indice)})
    a2, o2 = b2.decidir(obs)
    ap, op = b2p.decidir(cero)
    assert torch.equal(a2, ap) and torch.equal(o2, op)
    est = b2p._est(DEVICE)
    h, d0 = obs.casa[:4], obs.dow0[:4]
    dia_hoy = torch.zeros(4, dtype=torch.long, device=DEVICE)
    V = [b2p._valores(est, h, d0, dia_hoy, torch.full((4, N), x, dtype=torch.float64, device=DEVICE), 0, None)[_LIBRE]
         for x in (-0.1, 0.0, 0.1)]
    assert (V[0] <= V[1] + 1e-9).all() and (V[1] <= V[2] + 1e-9).all() and (V[2] > V[0]).any()


def test_b3_precio_en_coincide_con_su_opcion(d10, tabla10):
    b3 = PoliticaB3(tabla10, d10.T, d10.dist, d10.diesel)
    for semilla, pasos in ((5, 0), (6, 1), (7, 2)):
        _, obs = _estados(d10, B=40, pasos=pasos, semilla=semilla)
        a, o = b3.decidir(obs)
        act = obs.mascara.any(dim=1)
        assert torch.equal(b3.precio_en(obs, a)[act], o[act])
        assert obs.opciones[torch.arange(40, device=DEVICE), a, o][act].all()


def test_precio_de_la_vuelta_cuenta_volver_vacio():
    """En la vuelta a casa todas las reglas maximizan el ingreso esperado P·p (el costo se paga igual); en los
    demás tramos cada regla usa su nivel."""
    n = 3
    T, dist, diesel = np.full((n, n), 10.0) - 10.0 * np.eye(n), np.full((n, n), 500.0), np.full((n, n), 2.5)
    est = _Estatico(T, dist, diesel, DEVICE)
    precios = np.full((n, 1), 2.0) * (0.7 + 0.6 * U_NIVELES_ARR)
    op = np.zeros((n, N_NIVELES + 1), dtype=bool)
    op[[0, 2], :N_NIVELES] = True
    op[0, OPCION_VACIO] = True                                     # desde el intermedio 1 se puede volver vacío a 0
    obs = _obs(1, 0, n, precios, op)
    costo = float(lane_cost_row(dist[1, 0], diesel[1, 0]))
    P = prob_ganar(U_NIVELES_ARR, 2.0)
    for regla in ("mediana", "be_margen", "miope"):
        assert int(cotizar_regla(obs, torch.tensor([0], device=DEVICE), est, regla)[0]) == int(np.argmax(P * precios[0]))
        assert int(cotizar_regla(obs, torch.tensor([2], device=DEVICE), est, regla)[0]) == \
            int(nivel_regla(regla, precios[2], 500.0, costo, 2.0))
    v = np_(valor_inmediato(obs, est, cierre=True))[0]                                 # B2 / B2P
    assert v[0] == pytest.approx(float((P * precios[0] * 500.0).max() - costo))
    v1 = np_(valor_inmediato(obs, est))[0]                                              # B1: contra no moverse
    assert v1[0] == pytest.approx(float((P * (precios[0] * 500.0 - costo)).max()))


# ─────────────────────────────────────────────────────────────────────────────
# En el arnés
# ─────────────────────────────────────────────────────────────────────────────

class _Vigilada:
    """Una política del arnés que verifica que cada (destino, opción) esté entre las opciones válidas."""

    def __init__(self, p):
        self.p, self.nombre, self.precio = p, p.nombre, getattr(p, "precio", None)
        self.decisiones = 0

    def decidir(self, env, masks, activo):
        a, o, probs = self.p.decidir(env, masks, activo)
        ok = env.opciones_validas[torch.arange(env.B, device=env.device), a, o]
        assert bool(ok[activo].all()), f"{self.nombre}: decisión fuera de la máscara"
        self.decisiones += int(activo.sum())
        return a, o, probs


def test_politicas_respetan_mascara_y_son_deterministas(d10, tabla10):
    ctx = ev.ContextoEval(agente_chico(N), d10.market, tabla10, d10.tm, d10.dist, d10.diesel, N, d10.avail, d10.p95)
    assert set(ev._METODOS) == {"DRL", "DRL-dest+mediana", "DRL-dest+be_margen", "DRL-dest+miope",
                                "B1", "B2", "B2P", "B3"}
    mundos = mundos_sim(d10.market, 24, semilla=9)
    nodos = np.arange(24) % N
    k = [K_MAX if b % 2 else None for b in range(24)]
    for nombre in ev._METODOS + ("DRL-dest+dp",):
        pol = _Vigilada(ctx.politica(nombre))
        r1 = ev.ejecutar_lote(pol, ctx.env, mundos, nodos, k, d10.dist)
        r2 = ev.ejecutar_lote(ctx.politica(nombre), ctx.env, mundos, nodos, k, d10.dist)
        assert pol.decisiones > 0
        for a, b in zip(r1, r2):
            assert a["margen"] == b["margen"] and a["ruta"] == b["ruta"] and a["niveles"] == b["niveles"]
    sin_tabla = ev.ContextoEval(None, d10.market, None, d10.tm, d10.dist, d10.diesel, N, d10.avail, d10.p95)
    assert "B3" not in sin_tabla.politicas
    with pytest.raises(ValueError):
        ev.PoliticaDRL(sin_tabla, "dp")


def test_ub_pi_mayor_o_igual_que_cada_politica_y_b3_cerca_de_ub_dp(sim10):
    ok = ~sim10["violacion"]
    assert ok.any() and sim10["ub_pi"].notna().all()
    assert (sim10.loc[ok, "margen"] <= sim10.loc[ok, "ub_pi"] + 1e-6).all()
    assert set(sim10["metodo"]) == set(ev._METODOS)
    for k in (str(K_MAX), "libre"):
        g = sim10[(sim10["metodo"] == "B3") & (sim10["K"] == k)]
        dif = (g["margen"] - g["ub_dp"]).to_numpy()
        se = dif.std(ddof=1) / np.sqrt(len(dif))
        assert dif.mean() <= 3.0 * se, f"K={k}: B3 − UB-DP = {dif.mean():.1f} > 3·{se:.1f}"
        assert g["ub_dp"].mean() >= g["margen"].mean() - 3.0 * se
        assert g["ub_pi"].mean() >= g["ub_dp"].mean()


def test_ciclo_fallido_se_reporta_con_la_penalizacion(sim10):
    assert PENALIZACION_USD == -3000.0
    assert (sim10.loc[sim10["violacion"], "margen"] <= PENALIZACION_USD + 1e-6).all()
    sin_salir = sim10[(sim10["duracion"] == 0) & ~sim10["violacion"]]
    assert (sin_salir["margen"] == 0).all()


def test_diseno_de_mundos_balanceado_y_conjuntos_disjuntos(d10):
    n, M, mp = N, 100, d10.market
    dis = ev.disenar_mundos(n, M, mp)
    cn, cd = np.bincount(dis["nodo"], minlength=n), np.bincount(dis["dow"], minlength=7)
    assert cn.min() >= M // n and cn.max() <= -(-M // n) and cd.min() >= M // 7 and cd.max() <= -(-M // 7)
    assert (mp.dia_semana(dis["start_day"]) == dis["dow"]).all()
    assert (dis["start_day"] <= TRAIN_DAYS - ev.MARKET_DIAS_EPISODIO).all()
    assert all((dis[k] == ev.disenar_mundos(n, M, mp)[k]).all() for k in dis)
    assert len(set(zip(dis["nodo"], dis["dow"]))) > n                  # el día no queda atado al nodo
    assert MARKET_SEED_EVAL_FINAL not in (MARKET_SEED_EVAL, MARKET_SEED_TRAIN)
    a = ev.mundos_eval(mp, dis["start_day"], TRAIN_DAYS - 1, MARKET_SEED_EVAL)
    b = ev.mundos_eval(mp, dis["start_day"], TRAIN_DAYS - 1, MARKET_SEED_EVAL_FINAL)
    c = ev.mundos_eval(mp, dis["start_day"], TRAIN_DAYS - 1, MARKET_SEED_EVAL_FINAL)
    assert not torch.equal(a.u, b.u) and torch.equal(b.u, c.u) and torch.equal(b.loads, c.loads)


# ─────────────────────────────────────────────────────────────────────────────
# Estadística y limpieza
# ─────────────────────────────────────────────────────────────────────────────

def test_holm_y_comparacion_pareada():
    p = np.array([0.01, 0.04, 0.03])
    assert st.holm(p) == pytest.approx([0.03, 0.06, 0.06])
    rng = np.random.default_rng(0)
    base = rng.normal(1000, 100, size=60)
    filas = []
    for k in ("5", "libre"):
        for i, b in enumerate(base):
            filas.append(dict(modo="simulado", K=k, mundo=i, metodo="DRL", margen=b + 50 + rng.normal(0, 20)))
            filas.append(dict(modo="simulado", K=k, mundo=i, metodo="B1", margen=b))
            filas.append(dict(modo="simulado", K=k, mundo=i, metodo="B3", margen=b + 100))
            filas.append(dict(modo="simulado", K=k, mundo=i, metodo="B2", margen=b + 20))
            filas.append(dict(modo="simulado", K=k, mundo=i, metodo="DRL-dest+miope", margen=b + 40))
    df = pd.DataFrame(filas)
    tabla = st.comparar_con_baselines(df, B=2000)
    assert set(tabla["comparacion"]) == {"DRL − B1", "DRL − B2", "DRL − B3", "DRL − DRL-dest+miope"}
    assert len(tabla[tabla["K"] == "libre"]) == 4                 # B2 también con K libre
    # Comparación B: familia aparte, con su propio Holm (una sola comparación ⇒ p_holm = p).
    b = tabla[tabla["familia"] == "precio"]
    assert set(b["comparacion"]) == {"DRL − DRL-dest+miope"}
    assert np.allclose(b["p_holm"], b["p_una_cola"])
    assert set(tabla.loc[tabla["familia"] == "ruteo", "comparacion"]) == {"DRL − B1", "DRL − B2", "DRL − B3"}
    fila = tabla[(tabla["K"] == "5") & (tabla["comparacion"] == "DRL − B1")].iloc[0]
    assert fila["ic95_inf"] < fila["dif_media_usd"] < fila["ic95_sup"]
    assert fila["p_una_cola"] < 0.001 and fila["p_holm"] >= fila["p_una_cola"]
    assert tabla[tabla["comparacion"] == "DRL − B3"]["dif_media_usd"].max() < 0       # B3 le gana
    pd.testing.assert_frame_equal(tabla, st.comparar_con_baselines(df, B=2000))        # semilla fija

def test_marca_de_ub_dp_floja_y_cotas_separadas():
    filas = []
    for k, dp in (("5", 90.0), ("libre", 150.0)):          # en "libre" UB-DP > UB-PI
        for i in range(10):
            for metodo, m in (("DRL", 50.0), ("B3", 60.0)):
                filas.append(dict(modo="simulado", K=k, mundo=i, metodo=metodo, margen=m, violacion=False,
                                  frac_millas_cargadas=1.0, millas_totales=500.0, millas_cargadas=500.0, n_cargas=2, duracion=50.0, ms_decision=1.0,
                                  ruta=[0, 1, 0], n_cotizaciones=3, n_ganadas=2, horas_rechazo=0.25,
                                  niveles=[1, 4, 2], usd_mi_sobre_mediana=0.1, dinero_en_la_mesa=30.0,
                                  n_esperas=1, horas_espera=5.0, n_reposiciones=0, n_bajo_be=1, usd_bajo_be=-20.0,
                                  ub_dp=dp, ub_pi=100.0))
    df = pd.DataFrame(filas)
    res = ev.resumen_metodos(df)
    assert set(res.loc[res["K"] == "libre", "nota"].dropna()) == {st.NOTA_UB_DP_FLOJA}
    assert res.loc[res["K"] == "5", "nota"].isna().all()
    # Cada gap contra su propia cota; las cotas aparecen como filas separadas.
    b3 = res[(res["K"] == "5") & (res["metodo"] == "B3")].iloc[0]
    assert b3["gap_ub_dp_pct"] == pytest.approx((90 - 60) / 90 * 100)
    assert b3["gap_ub_pi_pct"] == pytest.approx((100 - 60) / 100 * 100)
    assert {"UB-PI", "UB-DP"} <= set(res["metodo"])
    assert b3["tasa_aceptacion_pct"] == pytest.approx(200 / 3) and b3["cotizaciones_por_arco"] == pytest.approx(1.5)
    assert b3["nivel_medio"] == pytest.approx(7 / 3) and b3["horas_rechazo"] == pytest.approx(0.25)
    # IC bootstrap de la media: contiene la media y es reproducible (semilla fija).
    f = res[(res["K"] == "5") & (res["metodo"] == "B3")].iloc[0]
    assert f["margen_ic95_low"] <= f["margen_medio"] <= f["margen_ic95_high"]
    pd.testing.assert_frame_equal(res, ev.resumen_metodos(df))
    rng = np.random.default_rng(1)
    x = rng.normal(500, 100, size=200)
    lo, hi = st.ic_bootstrap_media(x)
    assert lo < x.mean() < hi and hi - lo == pytest.approx(2 * 1.96 * 100 / np.sqrt(200), rel=0.25)

_NOMBRES_BORRADOS = [
    r"solve_HGA_LNS_metaheuristic", r"solve_label_setting_exact", r"solve_heuristic_rolling_horizon\w*",
    r"solve_mc_rollout_stochastic", r"run_solver_comparison\w*", r"N_DAYS_PER_NODE",
    r"solve_heuristic\b", r"_label_setting_core\b", r"_close_cycle_and_finalize", r"_rh_greedy_stoch_from_state",
    r"solve_label_setting_oracle", r"_label_setting_core_stochastic", r"simulate_route_reward",
    r"\bvisited_(set|mask)\b", r"_visited_(set|mask)\b",
    r"_repair_route_lns_best_position", r"_generate_route_nearest_neighbor", r"_generate_random_feasible_route",
    r"_crossover_routes", r"_apply_2opt_mutation", r"_apply_lns_mutation", r"\bRHResult\b", r"\bMetaResult\b",
    r"DRL Det\b", r"\bLS-Exact\b", r"\bHGA-LNS\b", r"\bMC-Rollout\b", r"RH-Greedy", r"RH-Lookahead",
    # reescritura para GPU (2026-11-27): la API por fila / numpy se reemplazó por la de lotes en el dispositivo
    r"\bclass Tablero\b", r"tablero_por_mundo", r"\bsample_world\b", r"mundo_cotizacion", r"tablero_real",
    r"sample_arrivals", r"\bObservacion\b", r"\bsolve_ub_pi\(", r"make_sim_env", r"make_replay_env",
    r"ejecutar_politica", r"world_seeds?\b", r"get_episodes_per_node", r"USE_MARKET_PROCESS",
    r"build_node_features\(", r"indice_mundo",
]


def test_ningun_nombre_borrado_sigue_referenciado():
    esta = os.path.abspath(__file__)
    archivos = [f for f in glob.glob(os.path.join(RAIZ, "**", "*.py"), recursive=True) if os.path.abspath(f) != esta]
    archivos.append(os.path.join(RAIZ, "README.md"))
    patron = re.compile("|".join(_NOMBRES_BORRADOS))
    hallazgos = []
    for f in archivos:
        with open(f, encoding="utf-8") as fh:
            for n, linea in enumerate(fh, 1):
                if patron.search(linea):
                    hallazgos.append(f"{os.path.relpath(f, RAIZ)}:{n}: {linea.strip()[:100]}")
    assert not hallazgos, "Referencias a baselines borrados:\n" + "\n".join(hallazgos)
