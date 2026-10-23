"""
evaluation.py
=============
Evaluación posterior al entrenamiento (baselines_drl_ruteo_estocastico.md):
  · rollout_drl_env              — rollout greedy (argmax por nodo) del agente en el entorno
  · evaluar_mundos_simulados     — DRL, B1, B2, B3 sobre los mismos mundos simulados + UB-PI (cota) y UB-DP
                                   (referencia: valor de la política que ignora el mercado, no es cota)
  · evaluar_replay               — el mismo arnés sobre los días reales de evaluación (market_process=None)
  · resumen_metodos              — márgenes, gaps y métricas operativas por método
  · save_results, plot_*         — Excel y figuras

Arnés único: todas las políticas se ejecutan paso a paso en `VectorRoutingEnv` (B = 1) con
la misma interfaz. Una política solo lee la máscara, la fila observada de cargas y tarifas,
el nodo actual, la hora y k restante (`baselines.Observacion`); el mismo
`world_seed` da números aleatorios comunes y la no anticipatividad queda garantizada por
construcción. `sample_world` se usa solo para UB-PI.

El margen de una ruta es la suma de `lane_reward` de cada arco, calculada con la fila
observada en el nodo de salida (no la recompensa del entorno, que trae bono, penalidad y
escala).
"""

import copy
import hashlib
import os
import pickle
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from config import (
    MAX_DURATION, K_MAX, TRAIN_DAYS, MARKET_DIAS_EPISODIO,
    MARKET_SEED_EVAL, MARKET_M_EVAL_WORLDS, REPLAY_MAX_CASOS,
)
from baselines import Observacion, PoliticaB1, PoliticaB2, calcular_ev_esperado
from dp_relajada import PoliticaB3
from greedy_rollout import greedy_rollout
from problem_data import build_rm_pen_stack, lane_reward
from routing_env import VectorRoutingEnv
from Solvers import solve_ub_pi
from stats_analysis import gap_pct, ic_bootstrap_media, ub_dp_floja, NOTA_UB_DP_FLOJA

_DIAS_SIMULADOS_UB_PI = MARKET_DIAS_EPISODIO   # días del mundo que cubren cualquier ruta (6: 77 h = 5,5 días de mercado)
_METODOS = ("DRL", "B1", "B2", "B3")


# ── DRL env-based rollout ─────────────────────────────────────
def make_replay_env(time_matrix, rate_stack, loads_stack, distance_arr, diesel_arr, num_nodes,
                    day_offset=0, market=None):
    """Entorno de replay de los días reales (market_process=None) sobre el stack dado.

    day_offset : índice absoluto del día 0 del stack (TRAIN_DAYS para los días de evaluación);
        define el día de la semana que ve el agente y los días de los que se toma el índice.
    market : MarketProcess del que sale el índice de mercado observado de cada día real
        (`indice_historico`: 194 nodos, media_lane de entrenamiento, tarifas reales del día, recortado a
        N nodos). Sin él el entorno no expone `market_index` (y el agente no puede usarlo).
    """
    indice = None if market is None else market.indice_historico(day_offset, day_offset + rate_stack.shape[0])
    return VectorRoutingEnv(
        time_matrix=time_matrix,
        rate_stack=rate_stack,
        loads_stack=loads_stack,
        rm_pen_stack=build_rm_pen_stack(rate_stack, loads_stack, distance_arr, diesel_arr),
        num_nodes=num_nodes,
        max_duration=MAX_DURATION,
        day_offset=day_offset,
        indice_stack=indice,
        indice_std=None if market is None else market.params.indice_std,
    )


def make_sim_env(time_matrix, rate_stack, loads_stack, distance_arr, diesel_arr, num_nodes, market):
    """Entorno de mundos simulados (cargas y tarifas sorteadas al llegar a cada nodo).

    rate_stack / loads_stack solo fijan el último día con datos para las features
    históricas (LTR, camiones); como en el entrenamiento, los días de entrenamiento.
    """
    return VectorRoutingEnv(
        time_matrix=time_matrix,
        rate_stack=rate_stack,
        loads_stack=loads_stack,
        rm_pen_stack=None,
        num_nodes=num_nodes,
        max_duration=MAX_DURATION,
        market_process=market,
        distance_arr=distance_arr,
        diesel_arr=diesel_arr,
    )


def rollout_drl_env(
    agent,
    env:            VectorRoutingEnv,
    start_node:     int,
    start_day_idx:  int,
    time_matrix_arr: np.ndarray,
    distance_arr:   np.ndarray,
    ltr_stack:      np.ndarray,
    trucks_stack:   np.ndarray,
    num_nodes:      int,
    reward_global_p95: float      = 1.0,
    k_max:          int        = None,
    avail_expected_arr: np.ndarray = None,
    world_seed:     int        = None,
    device:         torch.device = None,
    ev_esperado:    np.ndarray = None,   # (7, N, N) calcular_ev_esperado; obligatorio con USE_CONTINUATION_FEATURES
) -> tuple:
    """Una ruta del agente: paso a paso en el entorno, argmax en cada nodo.

    El entorno fija el mundo: replay del día `start_day_idx` (env sin
    market_process, ver make_replay_env) o un mundo simulado (env con
    market_process, `world_seed`). Máscara, disponibilidad de lanes y recompensa
    son las del entorno de entrenamiento; no hay búsqueda ni ranking por crítico.

    k_max : int | None — presupuesto de cardinalidad (nº de arcos, incluido
        el retorno). None (default) = caso libre (sin presupuesto K).

    Retorna (route, reward, duration) con el reward en unidades originales (suma
    de lanes); (None, -inf, inf) si no cierra el ciclo en el depot dentro de max_duration.
    """
    res = greedy_rollout(
        agent, env,
        np.array([start_node], dtype=np.int64), np.array([start_day_idx], dtype=np.int64),
        time_matrix_arr, distance_arr, trucks_stack, ltr_stack,
        reward_global_p95, num_nodes,
        k_max_arr=[k_max],
        avail_expected_arr=avail_expected_arr,
        world_seeds=None if world_seed is None else np.array([world_seed], dtype=np.int64),
        device=device,
        ev_esperado=ev_esperado,
    )
    if not res.valid[0]:
        return None, -np.inf, np.inf
    return res.routes[0], float(res.raw_reward[0]), float(res.duration[0])


# ── Arnés: una política, un mundo ─────────────────────────────

class _EnvCronometrado:
    """Envuelve un VectorRoutingEnv y acumula el tiempo gastado en reset/step.

    Sirve para separar el tiempo de decisión del DRL (features + red) del tiempo del
    entorno, que dentro de greedy_rollout va mezclado.
    """

    def __init__(self, env):
        self._env = env
        self.t_env = 0.0

    def reset(self, *args, **kwargs):
        t0 = time.perf_counter()
        out = self._env.reset(*args, **kwargs)
        self.t_env += time.perf_counter() - t0
        return out

    def step(self, *args, **kwargs):
        t0 = time.perf_counter()
        out = self._env.step(*args, **kwargs)
        self.t_env += time.perf_counter() - t0
        return out

    def __getattr__(self, name):
        return getattr(self._env, name)


class _PoliticaRuta:
    """Repite una ruta ya decidida (para medir sus métricas con el mismo arnés)."""

    def __init__(self, ruta):
        self._ruta, self._pos = ruta, 0

    def actuar(self, obs) -> int:
        self._pos += 1
        return int(self._ruta[self._pos])


def ejecutar_politica(politica, env, start_node, start_day_idx, k_max, world_seed,
                      distance_arr, diesel_arr, dow0, loads_stack=None, rate_stack=None) -> dict:
    """Corre `politica` paso a paso en `env` (B = 1) y devuelve las métricas de la ruta.

    Sin market_process (replay) las filas observadas salen de loads_stack / rate_stack en el
    día actual del entorno; con él, de las filas sorteadas al llegar al nodo.
    Métricas: margen (suma de lane_reward con la fila de salida), violacion (no cerró el
    ciclo en casa dentro de max_duration), frac_millas_cargadas, n_cargas, duracion,
    ms_decision (tiempo medio de `politica.actuar`) y ruta.
    """
    N = env.num_nodes
    seeds = None if world_seed is None else np.array([world_seed], dtype=np.int64)
    masks = env.reset(
        np.array([start_node], dtype=np.int64), np.array([start_day_idx], dtype=np.int64),
        k_max_arr=[k_max], world_seeds=seeds,
    )
    ruta = [int(start_node)]
    margen = millas_carg = millas_tot = 0.0
    n_cargas = n_dec = 0
    t_dec = 0.0
    cerro = False
    for _ in range(env.max_steps):
        mascara = masks[0]
        if not mascara.any():
            break                                         # sin acción factible (no cierra el ciclo)
        i = int(env.current_node[0])
        if env.current_loads_row is not None:
            loads_row, rate_row = env.current_loads_row[0], env.current_rate_row[0]
        else:
            d = int(env.current_day_idx[0])
            loads_row, rate_row = loads_stack[d, i], rate_stack[d, i]
        obs = Observacion(
            nodo=i, casa=int(start_node), tiempo=float(env.time_elapsed[0]),
            k_restante=int(env.k_remaining[0]) if env.k_restricted[0] else None,
            mascara=mascara.copy(),
            loads_row=loads_row, rate_row=rate_row, dia_semana_salida=int(dow0),
        )
        t0 = time.perf_counter()
        a = int(politica.actuar(obs))
        t_dec += time.perf_counter() - t0
        n_dec += 1
        if not mascara[a]:
            raise RuntimeError(f"{getattr(politica, 'nombre', politica)} eligió la acción {a}, fuera de la máscara")
        r = float(lane_reward(rate_row[a], loads_row[a], distance_arr[i, a], diesel_arr[i, a]))
        margen += r
        millas_tot += float(distance_arr[i, a])
        if loads_row[a] > 0:
            millas_carg += float(distance_arr[i, a])
            n_cargas += 1
        ruta.append(a)
        _, terminated, masks = env.step(np.array([a], dtype=np.int64))
        if terminated[0]:
            cerro = (a == start_node) and not bool(env.last_time_violation[0])
            break
    return dict(
        margen=margen,
        violacion=not cerro,
        frac_millas_cargadas=millas_carg / millas_tot if millas_tot > 0 else float("nan"),
        n_cargas=n_cargas,
        duracion=float(env.time_elapsed[0]),
        ms_decision=1000.0 * t_dec / n_dec if n_dec else float("nan"),
        ruta=ruta,
    )


class _UnHilo:
    """Contexto: fija torch en un solo hilo (tiempos en CPU comparables) y lo restaura."""

    def __enter__(self):
        self._prev = torch.get_num_threads()
        torch.set_num_threads(1)

    def __exit__(self, *exc):
        torch.set_num_threads(self._prev)


class ContextoEval:
    """Datos y políticas comunes de una evaluación (simulada o replay)."""

    def __init__(self, agent, market, tabla, time_matrix, distance_arr, diesel_arr, num_nodes,
                 ltr_stack, trucks_stack, avail_expected_arr, reward_global_p95):
        self.agent_cpu = copy.deepcopy(agent).to("cpu").eval()   # DRL en CPU, un hilo
        self.market, self.tabla = market, tabla
        self.T = np.asarray(time_matrix, dtype=float)
        self.dist, self.diesel, self.N = distance_arr, diesel_arr, num_nodes
        self.ltr, self.trucks = ltr_stack, trucks_stack
        self.avail, self.p95 = avail_expected_arr, reward_global_p95
        self.ev = calcular_ev_esperado(market, distance_arr, diesel_arr)   # una vez por evaluación
        self.politicas = {
            "B1": PoliticaB1(distance_arr, diesel_arr),
            "B2": PoliticaB2(market, self.T, distance_arr, diesel_arr),
            "B3": PoliticaB3(tabla, distance_arr, diesel_arr),
        }


def _ejecutar_drl(ctx, env, nodo, dia, K, world_seed, dow0, loads_stack=None, rate_stack=None) -> dict:
    """Rollout greedy del DRL (greedy_rollout, sin cambios) + métricas con el arnés común.

    El tiempo por decisión es el del rollout menos el del entorno (reset/step).
    """
    seeds = None if world_seed is None else np.array([world_seed], dtype=np.int64)
    if not env.reset(np.array([nodo]), np.array([dia]), k_max_arr=[K], world_seeds=seeds)[0].any():
        ruta, ms = [int(nodo)], float("nan")        # sin ninguna acción factible
    else:
        envc = _EnvCronometrado(env)
        t0 = time.perf_counter()
        res = greedy_rollout(
            ctx.agent_cpu, envc, np.array([nodo], dtype=np.int64), np.array([dia], dtype=np.int64),
            ctx.T, ctx.dist, ctx.trucks, ctx.ltr, ctx.p95, ctx.N,
            k_max_arr=[K], avail_expected_arr=ctx.avail, world_seeds=seeds, device=torch.device("cpu"),
            ev_esperado=ctx.ev,
        )
        total = time.perf_counter() - t0
        ruta = res.routes[0]
        ms = 1000.0 * (total - envc.t_env) / (len(ruta) - 1)
    m = ejecutar_politica(_PoliticaRuta(ruta), env, nodo, dia, K, world_seed,
                          ctx.dist, ctx.diesel, dow0, loads_stack, rate_stack)
    m["ms_decision"] = ms
    return m


def ub_pi(time_matrix, distance_arr, diesel_arr, start_node, K, mundo, start_day_idx=0) -> tuple:
    """UB-PI: mejor ruta con el mundo completo conocido, vía DP hacia atrás (Solvers.solve_ub_pi).

    Estado (nodo, franja de UBPI_DT_HORAS, k): el camión puede revisitar nodos, así que no hay
    conjunto de visitados. La recompensa de salir de un nodo en una franja usa la fila del día
    de esa franja del mundo, con las mismas reglas de factibilidad que el entorno.

    mundo = (loads [D, N, N], rate [D, N, N]) con el día 0 = día de salida (sample_world) o
    el stack histórico completo con start_day_idx. Devuelve (margen, ruta); (nan, None) si
    no hay ruta factible.
    """
    loads, rates = mundo
    valor, ruta = solve_ub_pi(
        start_node, np.asarray(time_matrix, dtype=float), rates, loads,
        distance_arr, diesel_arr, MAX_DURATION, start_day_idx=start_day_idx, k_max=K,
    )
    return (float(valor), ruta) if ruta is not None else (float("nan"), None)


def _etiqueta_k(K) -> str:
    return "libre" if K is None else str(int(K))


def _asignacion_balanceada(n_valores: int, M: int, rng) -> np.ndarray:
    """M valores en [0, n_valores) donde cada uno aparece ⌊M/n⌋ o ⌈M/n⌉ veces, en orden aleatorio."""
    base = np.tile(np.arange(n_valores), -(-M // n_valores))[:M]
    return rng.permutation(base)


def disenar_mundos(num_nodes: int, M: int, market, num_dias: int = TRAIN_DAYS,
                   semilla: int = MARKET_SEED_EVAL) -> dict:
    """Diseño de M mundos de evaluación a partir de MARKET_SEED_EVAL.

    Cada nodo aparece ⌊M/N⌋ o ⌈M/N⌉ veces y cada día de la semana ⌊M/7⌋ o ⌈M/7⌉ veces; los
    dos se barajan por separado, así que el día no queda atado al nodo. start_day_idx es un
    día del histórico (de los `num_dias` primeros) con ese día de la semana.
    """
    rng = np.random.default_rng(semilla)
    nodos = _asignacion_balanceada(num_nodes, M, rng)
    dows = _asignacion_balanceada(7, M, rng)
    dias = np.arange(num_dias)
    dow_dia = market.dia_semana(dias)
    inicio = np.array([rng.choice(dias[dow_dia == d]) for d in dows], dtype=np.int64)
    semillas = rng.integers(0, 2**31 - 1, size=M, dtype=np.int64)
    return dict(nodo=nodos.astype(np.int64), dow=dows.astype(np.int64),
                start_day=inicio, world_seed=semillas)


def _fila(modo, K, mundo, nodo, dia_abs, start_day, world_seed, dow, metodo, m, ub_dp, ub_pi_val) -> dict:
    return {
        "modo": modo, "K": _etiqueta_k(K), "mundo": mundo, "nodo_inicio": int(nodo),
        "dia_abs": int(dia_abs), "start_day_idx": int(start_day),
        "dia_semana": int(dow), "world_seed": -1 if world_seed is None else int(world_seed),
        "metodo": metodo, "margen": m["margen"], "violacion": m["violacion"],
        "frac_millas_cargadas": m["frac_millas_cargadas"], "n_cargas": m["n_cargas"],
        "duracion": m["duracion"], "ms_decision": m["ms_decision"], "ruta": m["ruta"],
        "ub_dp": ub_dp, "ub_pi": ub_pi_val,
    }


def _firma_cache(ctx, *partes) -> str:
    """Huella de lo que determina los baselines y las cotas: los parámetros del mercado, el ev, la tabla V
    de UB-DP y el diseño de la evaluación. Si cambia algo de eso, el caché no se reutiliza."""
    h = hashlib.sha1(ctx.market.firma_params().encode())
    h.update(np.ascontiguousarray(ctx.ev).tobytes())
    h.update(np.ascontiguousarray(ctx.tabla.V_k).tobytes())
    h.update(np.ascontiguousarray(ctx.tabla.V_libre).tobytes())
    h.update(repr((partes, MAX_DURATION)).encode())
    return h.hexdigest()


def _faltantes(df: pd.DataFrame, k_levels) -> set:
    """Pares (K, método) de B1, B2, B3 que `df` no trae para los niveles de K pedidos."""
    hay = set(zip(df["K"], df["metodo"]))
    return {(K, nombre) for K in k_levels for nombre in _METODOS[1:] if (_etiqueta_k(K), nombre) not in hay}


def _filas_con_cache(cache_path, firma, calcular, verbose=True, completar=None) -> pd.DataFrame:
    """Filas de B1, B2, B3 y cotas (no dependen del DRL): se calculan una vez y se reusan.

    cache_path None ⇒ sin caché. Un caché con otra firma se recalcula y se sobrescribe.
    completar : función df → df que añade lo que le falte al caché vigente (p. ej. B2 con K libre,
        que un caché anterior no traía) sin recalcular las cotas ni los demás baselines."""
    if cache_path and os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            c = pickle.load(f)
        if c.get("firma") == firma:
            df = c["df"]
            if completar is not None:
                df_nuevo = completar(df)
                if len(df_nuevo) != len(df):
                    with open(cache_path, "wb") as f:
                        pickle.dump({"firma": firma, "df": df_nuevo}, f)
                    df = df_nuevo
            if verbose:
                print(f"  baselines y cotas: reutilizados de {cache_path}")
            return df
    df = calcular()
    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, "wb") as f:
            pickle.dump({"firma": firma, "df": df}, f)
    return df


def _combinar(base: pd.DataFrame, drl: pd.DataFrame, k_levels) -> pd.DataFrame:
    """Une las filas de baselines y del DRL en el orden de siempre: (mundo, K, método)."""
    df = pd.concat([base, drl], ignore_index=True)
    orden_k = {_etiqueta_k(K): n for n, K in enumerate(k_levels)}
    orden_m = {m: n for n, m in enumerate(_METODOS)}
    clave = pd.DataFrame({"m": df["mundo"], "k": df["K"].map(orden_k), "n": df["metodo"].map(orden_m)})
    return df.iloc[clave.sort_values(["m", "k", "n"], kind="stable").index].reset_index(drop=True)


def evaluar_mundos_simulados(agent, market, tabla, time_matrix, rate_train, loads_train,
                             distance_arr, diesel_arr, num_nodes, ltr_stack, trucks_stack,
                             avail_expected_arr, reward_global_p95,
                             M=MARKET_M_EVAL_WORLDS, k_levels=(K_MAX, None), verbose=True,
                             cache_path=None, semilla_eval=MARKET_SEED_EVAL) -> pd.DataFrame:
    """DRL, B1, B2 y B3 sobre los mismos M mundos simulados, por nivel de K.

    Los mismos mundos (nodo, día, world_seed) se usan en todos los niveles de K. Devuelve un
    DataFrame largo: una fila por (K, mundo, método) con el margen en USD, la factibilidad,
    las métricas operativas y las cotas ub_dp / ub_pi del mundo.

    semilla_eval : semilla del diseño de los M mundos (MARKET_SEED_EVAL = conjunto de desarrollo;
        MARKET_SEED_EVAL_FINAL = conjunto B, test final). Usar un cache_path distinto por semilla.
    cache_path : B1, B2, B3, UB-PI y UB-DP no dependen del DRL, así que con una ruta se calculan una
        sola vez y se reusan entre corridas (mientras no cambie el proceso, la tabla V ni el diseño).
    """
    ctx = ContextoEval(agent, market, tabla, time_matrix, distance_arr, diesel_arr, num_nodes,
                       ltr_stack, trucks_stack, avail_expected_arr, reward_global_p95)
    env = make_sim_env(time_matrix, rate_train, loads_train, distance_arr, diesel_arr, num_nodes, market)
    dis = disenar_mundos(num_nodes, M, market, num_dias=rate_train.shape[0], semilla=semilla_eval)

    def cotas(m, nodo, dow, K, mundo_pi):
        pi, _ = ub_pi(time_matrix, distance_arr, diesel_arr, nodo, K, mundo_pi)
        dp = tabla.ub_dp(nodo, dow, K) if (K is None or 2 <= K <= tabla.K) else float("nan")
        return dp, pi

    def baselines(faltan=None, cota_previa=None):
        """faltan None: todo (cotas y baselines). Con un conjunto de (K, método): solo esos
        baselines, con las cotas ya calculadas (`cota_previa`)."""
        filas = []
        with _UnHilo():
            for m in range(M):
                nodo, dow = int(dis["nodo"][m]), int(dis["dow"][m])
                dia, ws = int(dis["start_day"][m]), int(dis["world_seed"][m])
                mundo_pi = market.sample_world(ws, _DIAS_SIMULADOS_UB_PI, dia) if faltan is None else None
                for K in k_levels:
                    if faltan is None:
                        dp, pi = cotas(m, nodo, dow, K, mundo_pi)
                    else:
                        dp, pi = cota_previa.loc[(_etiqueta_k(K), m)]
                    for nombre in _METODOS[1:]:
                        if faltan is not None and (K, nombre) not in faltan:
                            continue
                        res = ejecutar_politica(ctx.politicas[nombre], env, nodo, dia, K, ws,
                                                distance_arr, diesel_arr, dow)
                        filas.append(_fila("simulado", K, m, nodo, dia, dia, ws, dow, nombre, res, dp, pi))
                if verbose and (m + 1) % 100 == 0:
                    print(f"  mundos simulados (baselines y cotas): {m + 1}/{M}", flush=True)
        return pd.DataFrame(filas)

    def completar(df):
        faltan = _faltantes(df, k_levels)
        if not faltan:
            return df
        print(f"  completando el caché con {sorted(faltan, key=str)}", flush=True)
        extra = baselines(faltan, df.groupby(["K", "mundo"])[["ub_dp", "ub_pi"]].first())
        return pd.concat([df, extra], ignore_index=True)

    firma = _firma_cache(ctx, "simulado", num_nodes, M, semilla_eval, tuple(k_levels), rate_train.shape[0])
    base = _filas_con_cache(cache_path, firma, baselines, verbose, completar)
    cota = base.groupby(["K", "mundo"])[["ub_dp", "ub_pi"]].first()

    filas = []
    with _UnHilo():
        # Calentamiento de torch (primera llamada) fuera de la medición.
        _ejecutar_drl(ctx, env, int(dis["nodo"][0]), int(dis["start_day"][0]), k_levels[0],
                      int(dis["world_seed"][0]), int(dis["dow"][0]))
        for m in range(M):
            nodo, dow = int(dis["nodo"][m]), int(dis["dow"][m])
            dia, ws = int(dis["start_day"][m]), int(dis["world_seed"][m])
            for K in k_levels:
                dp, pi = cota.loc[(_etiqueta_k(K), m)]
                res = _ejecutar_drl(ctx, env, nodo, dia, K, ws, dow)
                filas.append(_fila("simulado", K, m, nodo, dia, dia, ws, dow, "DRL", res, dp, pi))
            if verbose and (m + 1) % 100 == 0:
                print(f"  mundos simulados (DRL): {m + 1}/{M}", flush=True)
    df = _combinar(base, pd.DataFrame(filas), k_levels)
    _avisar_si_supera_cota(df)
    return df


def casos_replay(num_nodes: int, num_dias: int, max_casos: int = REPLAY_MAX_CASOS,
                 semilla: int = MARKET_SEED_EVAL + 1) -> tuple:
    """Pares (nodo, día de evaluación): todos si caben en `max_casos`; si no, un subconjunto
    balanceado (cada nodo y cada día aparecen ⌊·⌋ o ⌈·⌉ veces, barajados por separado)."""
    if num_nodes * num_dias <= max_casos:
        nodos, dias = np.meshgrid(np.arange(num_nodes), np.arange(num_dias), indexing="ij")
        return nodos.ravel(), dias.ravel()
    rng = np.random.default_rng(semilla)
    return (_asignacion_balanceada(num_nodes, max_casos, rng),
            _asignacion_balanceada(num_dias, max_casos, rng))


def evaluar_replay(agent, market, tabla, time_matrix, rate_eval, loads_eval,
                   distance_arr, diesel_arr, num_nodes, ltr_eval, trucks_eval,
                   avail_expected_arr, reward_global_p95,
                   k_levels=(K_MAX, None), max_casos=REPLAY_MAX_CASOS, verbose=True,
                   cache_path=None) -> pd.DataFrame:
    """El mismo arnés con market_process=None sobre los días reales de evaluación (días 90-119).

    Fuera de muestra respecto del proceso (se ajusta solo con los TRAIN_DAYS de
    entrenamiento). ltr_eval / trucks_eval deben estar recortados a los mismos días que rate_eval.
    UB-PI se resuelve sobre el histórico; UB-DP no aplica (es la referencia del proceso simulado).
    cache_path : como en evaluar_mundos_simulados, para B1, B2, B3 y UB-PI.
    """
    ctx = ContextoEval(agent, market, tabla, time_matrix, distance_arr, diesel_arr, num_nodes,
                       ltr_eval, trucks_eval, avail_expected_arr, reward_global_p95)
    env = make_replay_env(time_matrix, rate_eval, loads_eval, distance_arr, diesel_arr, num_nodes,
                          day_offset=TRAIN_DAYS, market=market)
    nodos, dias = casos_replay(num_nodes, rate_eval.shape[0], max_casos)

    def baselines(faltan=None, cota_previa=None):
        filas = []
        with _UnHilo():
            for c, (nodo, dia) in enumerate(zip(nodos, dias)):
                nodo, dia = int(nodo), int(dia)
                dow = int(market.dia_semana(TRAIN_DAYS + dia))
                for K in k_levels:
                    if faltan is None:
                        pi, _ = ub_pi(time_matrix, distance_arr, diesel_arr, nodo, K,
                                      (loads_eval, rate_eval), start_day_idx=dia)
                    else:
                        pi = cota_previa.loc[(_etiqueta_k(K), c), "ub_pi"]
                    for nombre in _METODOS[1:]:
                        if faltan is not None and (K, nombre) not in faltan:
                            continue
                        res = ejecutar_politica(ctx.politicas[nombre], env, nodo, dia, K, None,
                                                distance_arr, diesel_arr, dow, loads_eval, rate_eval)
                        filas.append(_fila("replay", K, c, nodo, TRAIN_DAYS + dia, dia, None, dow,
                                           nombre, res, float("nan"), pi))
                if verbose and (c + 1) % 100 == 0:
                    print(f"  replay (baselines y cotas): {c + 1}/{len(nodos)}", flush=True)
        return pd.DataFrame(filas)

    def completar(df):
        faltan = _faltantes(df, k_levels)
        if not faltan:
            return df
        print(f"  completando el caché con {sorted(faltan, key=str)}", flush=True)
        extra = baselines(faltan, df.groupby(["K", "mundo"])[["ub_dp", "ub_pi"]].first())
        return pd.concat([df, extra], ignore_index=True)

    firma = _firma_cache(ctx, "replay", num_nodes, max_casos, tuple(k_levels), rate_eval.shape[0],
                         loads_eval.shape, float(loads_eval.sum()), float(rate_eval.sum()))
    base = _filas_con_cache(cache_path, firma, baselines, verbose, completar)
    cota = base.groupby(["K", "mundo"])[["ub_dp", "ub_pi"]].first()

    filas = []
    with _UnHilo():
        for c, (nodo, dia) in enumerate(zip(nodos, dias)):
            nodo, dia = int(nodo), int(dia)
            dow = int(market.dia_semana(TRAIN_DAYS + dia))
            for K in k_levels:
                dp, pi = cota.loc[(_etiqueta_k(K), c)]
                res = _ejecutar_drl(ctx, env, nodo, dia, K, None, dow, loads_eval, rate_eval)
                filas.append(_fila("replay", K, c, nodo, TRAIN_DAYS + dia, dia, None, dow,
                                   "DRL", res, dp, pi))
            if verbose and (c + 1) % 100 == 0:
                print(f"  replay (DRL): {c + 1}/{len(nodos)}", flush=True)
    df = _combinar(base, pd.DataFrame(filas), k_levels)
    _avisar_si_supera_cota(df)
    return df


def _avisar_si_supera_cota(df: pd.DataFrame, tol: float = 1e-6) -> None:
    """UB-PI debe ser ≥ el margen de toda política con ruta factible, mundo a mundo."""
    ok = ~df["violacion"] & df["ub_pi"].notna()
    malos = df[ok & (df["margen"] > df["ub_pi"] + tol)]
    if len(malos):
        print(f"  ⚠ {len(malos)} filas con margen > UB-PI (p. ej. {malos.iloc[0]['metodo']} "
              f"mundo {malos.iloc[0]['mundo']} K={malos.iloc[0]['K']}): revisar UB-PI frente al entorno.")


# ── Resumen ───────────────────────────────────────────────────

def resumen_metodos(df: pd.DataFrame) -> pd.DataFrame:
    """Margen medio, gaps contra UB-DP y UB-PI y métricas operativas, por (modo, K, método).

    Los gaps comparan promedios sobre los mismos mundos: (ref − método) / |ref| · 100, con ref = UB-PI
    (cota) o UB-DP (política que ignora el mercado; un gap negativo significa que el método la supera).
    margen_medio va con su IC del 95 % por bootstrap sobre mundos (BOOTSTRAP_B remuestreos,
    semilla fija): margen_ic95_low / margen_ic95_high. Cada grupo (modo, K, método) usa su
    propio Generator con la semilla fija, así que el IC no depende del orden de las filas.
    Se añaden las filas de UB-PI (cota) y UB-DP (valor de la política que ignora el mercado, NO es
    cota: el DRL, que observa el índice, puede superarla), siempre separadas: nunca se combinan ni se
    promedian. La columna `nota` marca los K donde UB-DP > UB-PI (la política que ignora el mercado no
    puede superar a la de información perfecta: señal de desajuste entre la DP y el entorno).
    """
    filas = []
    for (modo, K), g in df.groupby(["modo", "K"], sort=False):
        mundos = g.drop_duplicates("mundo")
        nota = NOTA_UB_DP_FLOJA if ub_dp_floja(mundos["ub_dp"], mundos["ub_pi"]) else None
        for metodo in [m for m in _METODOS if m in set(g["metodo"])]:
            gm = g[g["metodo"] == metodo]
            ic = ic_bootstrap_media(gm["margen"])
            filas.append({
                "modo": modo, "K": K, "metodo": metodo, "n_mundos": len(gm),
                "margen_medio": gm["margen"].mean(),
                "margen_ic95_low": ic[0], "margen_ic95_high": ic[1],
                "gap_ub_dp_pct": gap_pct(gm["ub_dp"], gm["margen"]),
                "gap_ub_pi_pct": gap_pct(gm["ub_pi"], gm["margen"]),
                "violacion_pct": 100.0 * gm["violacion"].mean(),
                "millas_cargadas_pct": 100.0 * gm["frac_millas_cargadas"].mean(),
                "cargas_medias": gm["n_cargas"].mean(),
                "duracion_media_h": gm["duracion"].mean(),
                "ms_por_decision": gm["ms_decision"].mean(),
                "nota": nota,
            })
        for nombre, col in (("UB-PI", "ub_pi"), ("UB-DP", "ub_dp")):
            if mundos[col].notna().any():
                ic = ic_bootstrap_media(mundos[col])
                filas.append({"modo": modo, "K": K, "metodo": nombre, "n_mundos": int(mundos[col].notna().sum()),
                              "margen_medio": mundos[col].mean(),
                              "margen_ic95_low": ic[0], "margen_ic95_high": ic[1],
                              "nota": nota if col == "ub_dp" else None})
    return pd.DataFrame(filas)


# ── Output ────────────────────────────────────────────────────
def save_results(sim_df, replay_df, resumen_rows, output_path):
    from excel_report import build_formatted_excel
    build_formatted_excel(sim_df, replay_df, resumen_rows, output_path)
    print(f"Results saved → {output_path}")


def plot_ppo_diagnostics(training_log: list, num_nodes: int, output_path: str):
    """Gráfica 2×3 con métricas internas de PPO por update.

    Métricas graficadas
    -------------------
    Fila 1 : Reward promedio | Explained Variance | Entropía
    Fila 2 : Policy Loss + Value Loss | KL Divergence | Clip Fraction
    """
    try:
        import pandas as pd
        df = pd.DataFrame(training_log)

        fig, axes = plt.subplots(2, 4, figsize=(24, 10))
        fig.suptitle(
            f'PPO Internal Diagnostics — {num_nodes} nodes', fontsize=14
        )

        updates = df["update"].values

        def _plot(ax, y, title, ylabel, color, hline=None, hline_label=None):
            ax.plot(updates, y, linewidth=0.9, color=color)
            ax.set_title(title); ax.set_xlabel("PPO Update"); ax.set_ylabel(ylabel)
            ax.grid(True, alpha=0.3)
            if hline is not None:
                ax.axhline(hline, color="red", linestyle="--", linewidth=0.8,
                           label=hline_label or f"{hline}")
                ax.legend(fontsize=8)

        # greedy_reward / greedy_valid_rate solo se registran cada log_freq
        # updates — el resto queda NaN en el DataFrame. dropna() evita que la
        # línea se corte en cada hueco.
        greedy_df = df.dropna(subset=["greedy_reward"]) if "greedy_reward" in df else None

        ax_reward = axes[0, 0]
        ax_reward.plot(updates, df["reward"], linewidth=0.9, color="steelblue", label="Sampled reward")
        if greedy_df is not None and not greedy_df.empty:
            ax_reward.plot(greedy_df["update"], greedy_df["greedy_reward"],
                           linewidth=1.4, color="darkorange", marker="o", markersize=3,
                           label="Greedy reward")
        ax_reward.set_title("Avg Reward per Update"); ax_reward.set_xlabel("PPO Update")
        ax_reward.set_ylabel("Reward"); ax_reward.grid(True, alpha=0.3); ax_reward.legend(fontsize=8)

        _plot(axes[0, 1], df["explained_var"], "Critic Explained Variance", "Expl. Var.", "mediumseagreen",
              hline=0.5, hline_label="threshold 0.5")
        _plot(axes[0, 2], df["entropy"],       "Policy Entropy",          "Entropy",      "mediumpurple")

        ax_valid = axes[0, 3]
        if greedy_df is not None and not greedy_df.empty:
            ax_valid.plot(greedy_df["update"], greedy_df["greedy_valid_rate"],
                          linewidth=1.4, color="teal", marker="o", markersize=3)
            ax_valid.set_ylim(-0.05, 1.05)
        ax_valid.set_title("Greedy Valid Route Rate"); ax_valid.set_xlabel("PPO Update")
        ax_valid.set_ylabel("Valid Rate"); ax_valid.grid(True, alpha=0.3)

        ax_loss = axes[1, 0]
        ax_loss.plot(updates, df["policy_loss"], linewidth=0.9, color="coral",    label="Policy loss")
        ax_loss.plot(updates, df["value_loss"],  linewidth=0.9, color="goldenrod", label="Value loss")
        ax_loss.set_title("Policy & Value Loss"); ax_loss.set_xlabel("PPO Update")
        ax_loss.set_ylabel("Loss"); ax_loss.grid(True, alpha=0.3); ax_loss.legend(fontsize=8)

        _plot(axes[1, 1], df["kl_divergence"], "Approx KL Divergence",   "KL",           "tomato",
              hline=0.02, hline_label="target 0.02")
        _plot(axes[1, 2], df["clip_fraction"], "PPO Clip Fraction",       "Clip Frac.",   "darkorange",
              hline=0.1, hline_label="ref 0.10")
        axes[1, 3].axis("off")

        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()
        print(f"PPO diagnostics plot saved → {output_path}")
    except Exception as e:
        print(f"Warning: could not generate PPO diagnostics plot: {e}")


def plot_diagnostics(episode_rewards, episode_losses, sim_df,
                     num_nodes, output_path):
    try:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(
            f'DRL Training Diagnostics — {num_nodes} nodes', fontsize=14)

        window = max(1, min(100, len(episode_rewards) // 5))

        ax1 = axes[0, 0]
        ax1.plot(pd.Series(episode_rewards).rolling(window).mean(),
                 linewidth=0.8, color='steelblue')
        ax1.set_title('Training rewards (rolling avg)')
        ax1.set_xlabel('Episode'); ax1.set_ylabel('Reward')
        ax1.grid(True, alpha=0.3)

        ax2 = axes[0, 1]
        ax2.plot(pd.Series(episode_losses).rolling(window).mean(),
                 linewidth=0.8, color='coral')
        ax2.set_title('Training loss (rolling avg)')
        ax2.set_xlabel('Episode'); ax2.set_ylabel('Loss')
        ax2.grid(True, alpha=0.3)

        # Margen medio por nodo de inicio (primer nivel de K de los mundos simulados).
        k0 = sim_df['K'].iloc[0]
        g = sim_df[sim_df['K'] == k0]
        x = np.arange(num_nodes); w = 0.35

        def _por_nodo(col, metodo=None):
            d = g if metodo is None else g[g['metodo'] == metodo]
            if metodo is None:
                d = d.drop_duplicates('mundo')
            return d.groupby('nodo_inicio')[col].mean().reindex(range(num_nodes)).fillna(0).to_numpy()

        drl = _por_nodo('margen', 'DRL')
        for ax, (label, vals, color) in zip(
            (axes[1, 0], axes[1, 1]),
            (('UB-PI', _por_nodo('ub_pi'), 'forestgreen'), ('B3', _por_nodo('margen', 'B3'), 'steelblue')),
        ):
            ax.bar(x - w/2, vals, w, label=label, color=color, alpha=0.8)
            ax.bar(x + w/2, drl,  w, label='DRL',  color='coral', alpha=0.8)
            ax.set_title(f'DRL vs {label} por nodo de inicio (K={k0}, mundos simulados)')
            ax.set_xlabel('Start node'); ax.set_ylabel('Margen (USD)')
            ax.set_xticks(x); ax.legend(); ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()
        print(f"Diagnostics plot saved → {output_path}")
    except Exception as e:
        print(f"Warning: could not generate diagnostics plot: {e}")
