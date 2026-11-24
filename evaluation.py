"""
evaluation.py
=============
Evaluación posterior al entrenamiento (baselines_drl_ruteo_estocastico.md, especificacion_fase0_ruteo_pricing.md §7-8):
  · rollout_drl_env              — rollout greedy del agente en el entorno (destino y precio)
  · evaluar_mundos_simulados     — DRL, comparación B y B1, B2, B2P, B3 sobre los mismos mundos simulados,
                                   + UB-PI (cota) y UB-DP (referencia de la DP)
  · evaluar_replay               — el mismo arnés sobre los días reales de evaluación
  · resumen_metodos              — márgenes, gaps, aceptación y métricas operativas por método
  · save_results, plot_*         — Excel y figuras

Arnés único: todas las políticas se ejecutan paso a paso en `VectorRoutingEnv` (B = 1), con
cotización: la política elige el destino (`actuar`) y el precio (`cotizar`); si la carga se pierde,
vuelve a decidir en el mismo nodo con la lane enmascarada. Una política solo lee la `Observacion`
(máscara, tablero observado, nodo, hora y k restante) y, el DRL, el estado observable del entorno;
nadie ve el competidor. La misma semilla da los mismos tableros y competidores a todos los métodos
(números aleatorios comunes).

Métodos (comparación B, §7.1):
  · DRL                  — destino y precio del DRL.
  · DRL-dest+<regla>     — destino del DRL, precio con la regla fija (mediana, be_margen, miope).
  · B1, B2, B2P, B3      — destino del baseline, precio con REGLA_PRECIO_BASELINES (miope).
Si la tabla de UB-DP es "no tratable" a ese tamaño (tabla None, baselines/dp_relajada.py), B3 y UB-DP no se
calculan y el resumen los reporta como "no tratable" (un resultado válido, no un error).

El margen de una ruta es la suma del reward de cada lane en USD con el precio cotizado
(env.last_raw_reward: p·d − c si se ganó la carga, −c si fue VACÍO, 0 en un rechazo).
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
    MARKET_SEED_EVAL, MARKET_M_EVAL_WORLDS, REPLAY_MAX_CASOS, REGLA_PRECIO_BASELINES,
)
from baselines.politicas import (
    Observacion, PoliticaB1, PoliticaB2, PoliticaB2P, PoliticaB3, calcular_ev_pronostico, cotizar_regla,
)
from greedy_rollout import greedy_rollout, construir_features, contexto_agente
from market_process import Tablero
from pricing import REGLAS_PRECIO, U_NIVELES_ARR, N_NIVELES, gana, prob_ganar
from routing_env import VectorRoutingEnv
from baselines.ub_pi import solve_ub_pi
from stats_analysis import gap_pct, ic_bootstrap_media, ub_dp_floja, NOTA_UB_DP_FLOJA

_DIAS_SIMULADOS_UB_PI = MARKET_DIAS_EPISODIO   # días del mundo que cubren cualquier ruta (6: 77 h = 5,5 días de mercado)
_BASELINES = ("B1", "B2", "B2P", "B3")   # no dependen del agente: se calculan una vez y se cachean
_DRL_REGLAS = tuple(f"DRL-dest+{r}" for r in REGLAS_PRECIO)   # comparación B
_METODOS = ("DRL",) + _DRL_REGLAS + _BASELINES
# Qué decide cada método (columnas "destino" y "precio" del resumen) y el grupo en que se reporta.
_NOMBRE_REGLA = {"mediana": "regla: mediana", "be_margen": "regla: break-even + margen", "miope": "regla: miope"}
DECISION = {
    "DRL": ("DRL", "DRL (cabeza de precio)"),
    **{f"DRL-dest+{r}": ("DRL", _NOMBRE_REGLA.get(r, f"regla: {r}")) for r in REGLAS_PRECIO},
    "B1": ("B1: greedy miope", "regla: miope"),
    "B2": ("B2: reoptimización con valores esperados", "regla: miope"),
    "B2P": ("B2P: B2 con pronóstico del índice", "regla: miope"),
    "B3": ("B3: tabla V de la DP (dos intentos)", "regla: miope"),
    "UB-PI": ("información perfecta (cota)", "nivel más alto que gana"),
    "UB-DP": ("DP que ignora el índice de mercado", "óptimo de la DP"),
}
GRUPO_DRL, GRUPO_BASELINES, GRUPO_COTAS = (
    "DRL y comparación B (mismos destinos del DRL, distinta regla de precio)",
    "Baselines de ruteo (cotizan con la regla miope)",
    "Cotas y referencias")


def _grupo(metodo: str) -> str:
    if metodo == "DRL" or metodo.startswith("DRL-dest+"):
        return GRUPO_DRL
    return GRUPO_COTAS if metodo.startswith("UB-") else GRUPO_BASELINES


def _describir(metodo: str) -> dict:
    destino, precio = DECISION.get(metodo, ("", ""))
    return {"grupo": _grupo(metodo), "destino": destino, "precio": precio}
NOTA_NO_TRATABLE = ("no tratable: la tabla de UB-DP no se calcula por encima de UBDP_MAX_NODOS nodos o no terminó "
                    "dentro del tiempo o la memoria disponibles")


# ── Entornos ─────────────────────────────────────────────────
def make_replay_env(time_matrix, rate_stack, loads_stack, distance_arr, diesel_arr, num_nodes,
                    day_offset=0, market=None):
    """Entorno de replay de los días reales: cargas y cuantiles del propio día, competidor sorteado.

    rate_stack / loads_stack solo fijan el número de días. day_offset : índice absoluto del día 0
    (TRAIN_DAYS para los días de evaluación). market : MarketProcess del que salen los tableros reales
    (`tablero_real`) y el índice de mercado observado de cada día.
    """
    if market is None:
        raise ValueError("make_replay_env requiere market (MarketProcess)")
    return VectorRoutingEnv(
        time_matrix, num_nodes, distance_arr, diesel_arr, rate_stack.shape[0],
        max_duration=MAX_DURATION, mercado_replay=market, day_offset=day_offset,
    )


def make_sim_env(time_matrix, rate_stack, loads_stack, distance_arr, diesel_arr, num_nodes, market):
    """Entorno de mundos simulados (tablero sorteado al llegar a cada nodo).

    rate_stack / loads_stack solo fijan el último día con datos; como en el entrenamiento, los días de
    entrenamiento.
    """
    return VectorRoutingEnv(
        time_matrix, num_nodes, distance_arr, diesel_arr, rate_stack.shape[0],
        max_duration=MAX_DURATION, market_process=market,
    )


def rollout_drl_env(
    agent,
    env:            VectorRoutingEnv,
    start_node:     int,
    start_day_idx:  int,
    time_matrix_arr: np.ndarray,
    distance_arr:   np.ndarray,
    ltr_stack:      np.ndarray,
    num_nodes:      int,
    reward_global_p95: float      = 1.0,
    k_max:          int        = None,
    avail_expected_arr: np.ndarray = None,
    world_seed:     int        = None,
    device:         torch.device = None,
    ev_esperado:    np.ndarray = None,   # EvEsperado, calcular_ev_pronostico; obligatorio con USE_CONTINUATION_FEATURES
) -> tuple:
    """Una ruta del agente: paso a paso en el entorno, argmax de destino y de precio en cada nodo.

    k_max : presupuesto de cardinalidad (nº de arcos, incluido el retorno); None = caso libre.
    Retorna (route, reward, duration) con el reward en USD (suma de lanes con el precio cotizado);
    (None, -inf, inf) si no cierra el ciclo en el depot dentro de max_duration.
    """
    res = greedy_rollout(
        agent, env,
        np.array([start_node], dtype=np.int64), np.array([start_day_idx], dtype=np.int64),
        time_matrix_arr, distance_arr, ltr_stack,
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

class PoliticaDRL:
    """El agente como política del arnés: destino con la cabeza de destino; precio con la cabeza de
    precio (precio="drl") o con una regla fija (comparación B). Lee el estado observable del entorno
    (las mismas features que en el entrenamiento)."""

    def __init__(self, ctx, env, precio: str = "drl"):
        if precio != "drl" and precio not in REGLAS_PRECIO:
            raise ValueError(f"precio desconocido: {precio!r}")
        self.ctx, self.env, self.precio = ctx, env, precio
        self.nombre = "DRL" if precio == "drl" else f"DRL-dest+{precio}"
        self._opcion = None

    def actuar(self, obs) -> int:
        c, env = self.ctx, self.env
        with torch.no_grad():
            feats = construir_features(env, env.start_node, c.T, c.dist, c.ltr, c.p95, c.N, c.avail, c.ev)
            h_t, emb = contexto_agente(c.agent_cpu, feats, env.current_node, torch.device("cpu"))
            mask_t = torch.from_numpy(np.asarray(obs.mascara, dtype=np.int8)[None])
            op_t = torch.from_numpy(feats["opciones"])
            a, o = c.agent_cpu.decoder.greedy_cotizacion(h_t, emb, mask_t, op_t)
        self._opcion = int(o[0])
        return int(a[0])

    def cotizar(self, obs, j: int) -> int:
        if self.precio == "drl":
            return self._opcion
        return cotizar_regla(obs, j, self.ctx.dist, self.ctx.diesel, self.precio)


def ejecutar_politica(politica, env, start_node, start_day_idx, k_max, world_seed,
                      distance_arr, diesel_arr, dow0) -> dict:
    """Corre `politica` paso a paso en `env` (B = 1) y devuelve las métricas de la ruta.

    En cada paso la política elige el destino (`actuar`) y la opción de precio (`cotizar`). Métricas:
    margen (suma de env.last_raw_reward), violacion (no cerró el ciclo en casa dentro de max_duration),
    frac_millas_cargadas, n_cargas, duracion, ms_decision (tiempo medio de actuar + cotizar), ruta (solo
    viajes), n_cotizaciones, n_ganadas, horas_rechazo, niveles (lista), usd_mi_sobre_mediana (media de
    precio cobrado − mediana de la lane en las cargas ganadas) y dinero_en_la_mesa (Σ (p_comp − p)·d en las
    cargas ganadas).
    Detalle por arco: `pasos` (una entrada por step: cotización ganada, cotización perdida o viaje vacío,
    con el nivel, el percentil u, el precio cotizado, la mediana, P(ganar), el precio del competidor, las
    millas y el reward del paso) y `ruta_precios`, la misma ruta en texto, p. ej.
    "3→7 cotiza 2.95 USD/mi (u=0.6) ✗ | 3→5 cotiza 2.80 USD/mi (u=0.4) ✓ | 5→3 vacío".
    """
    seeds = None if world_seed is None else np.array([world_seed], dtype=np.int64)
    masks = env.reset(
        np.array([start_node], dtype=np.int64), np.array([start_day_idx], dtype=np.int64),
        k_max_arr=[k_max], world_seeds=seeds,
    )
    ruta = [int(start_node)]
    margen = millas_carg = millas_tot = horas_rech = mesa = 0.0
    n_cargas = n_dec = n_cot = 0
    niveles, sobre_mediana, pasos = [], [], []
    t_dec = 0.0
    cerro = False
    for _ in range(env.max_steps):
        mascara = masks[0]
        if not mascara.any():
            break                                         # sin acción factible (no cierra el ciclo)
        i = int(env.current_node[0])
        obs = Observacion(
            nodo=i, casa=int(start_node), tiempo=float(env.time_elapsed[0]),
            k_restante=int(env.k_remaining[0]) if env.k_restricted[0] else None,
            mascara=mascara.copy(),
            loads_row=env.current_loads_row[0].copy(), rate_row=env.rate_row_obs[0].copy(),
            precios=env.precios_row[0].copy(), m=float(env.competencia[0]),
            opciones=env.opciones_validas[0].copy(), dia_semana_salida=int(dow0),
            indice=env.market_index[0].copy() if getattr(politica, "usa_indice", False) else None,
            n_rechazos=int(env.n_rechazos[0]),
        )
        t0 = time.perf_counter()
        a = int(politica.actuar(obs))
        o = int(politica.cotizar(obs, a))
        t_dec += time.perf_counter() - t0
        n_dec += 1
        if not mascara[a] or not obs.opciones[a, o]:
            raise RuntimeError(f"{getattr(politica, 'nombre', politica)} eligió ({a}, {o}), fuera de la máscara")
        mediana = float(obs.rate_row[a])
        t_antes = float(env.time_elapsed[0])
        _, terminated, masks = env.step(np.array([a], dtype=np.int64), np.array([o], dtype=np.int64))
        margen += float(env.last_raw_reward[0])
        cotizo, gano = bool(env.last_cotizo[0]), bool(env.last_gano[0])
        u = float(U_NIVELES_ARR[o]) if cotizo else float("nan")
        pasos.append(dict(
            paso=len(pasos) + 1, desde=i, hacia=a,
            accion=("cotiza y gana" if gano else "cotiza y pierde") if cotizo else "viaja vacío",
            nivel=o if cotizo else None, u=u,
            precio_cotizado_usd_mi=float(obs.precios[a, o]) if cotizo else float("nan"),
            mediana_usd_mi=mediana,
            p_ganar=float(prob_ganar(u, obs.m)) if cotizo else float("nan"),
            precio_competidor_usd_mi=float(env.last_precio_competidor[0]) if cotizo else float("nan"),
            millas=0.0 if (cotizo and not gano) else float(distance_arr[i, a]),
            reward_usd=float(env.last_raw_reward[0]),
            horas_transcurridas=t_antes,
        ))
        if cotizo:
            n_cot += 1
            niveles.append(o)
        if cotizo and not gano:
            horas_rech += float(env.time_elapsed[0]) - t_antes
        else:                                             # viajó (ganó o VACÍO)
            ruta.append(a)
            millas_tot += float(distance_arr[i, a])
            if gano:
                millas_carg += float(distance_arr[i, a])
                n_cargas += 1
                p = float(env.last_precio[0])
                sobre_mediana.append(p - mediana)
                mesa += (float(env.last_precio_competidor[0]) - p) * float(distance_arr[i, a])
        if terminated[0]:
            cerro = (len(ruta) > 1 and ruta[-1] == start_node) and not bool(env.last_time_violation[0])
            break
    return dict(
        margen=margen,
        violacion=not cerro,
        frac_millas_cargadas=millas_carg / millas_tot if millas_tot > 0 else float("nan"),
        n_cargas=n_cargas,
        duracion=float(env.time_elapsed[0]),
        ms_decision=1000.0 * t_dec / n_dec if n_dec else float("nan"),
        ruta=ruta,
        n_cotizaciones=n_cot,
        n_ganadas=n_cargas,
        horas_rechazo=horas_rech,
        niveles=niveles,
        usd_mi_sobre_mediana=float(np.mean(sobre_mediana)) if sobre_mediana else float("nan"),
        dinero_en_la_mesa=mesa,
        pasos=pasos,
        ruta_precios=ruta_con_precios(pasos),
    )


def ruta_con_precios(pasos) -> str:
    """La ruta en texto, un tramo por paso: precio cotizado y si se ganó (✓) o se perdió (✗), o viaje vacío."""
    partes = []
    for p in pasos:
        tramo = f"{p['desde']}→{p['hacia']}"
        if p["accion"] == "viaja vacío":
            partes.append(f"{tramo} vacío")
        else:
            marca = "✓" if p["accion"] == "cotiza y gana" else "✗"
            partes.append(f"{tramo} cotiza {p['precio_cotizado_usd_mi']:.2f} USD/mi (u={p['u']:.1f}) {marca}")
    return " | ".join(partes)


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
                 ltr_stack, avail_expected_arr, reward_global_p95):
        self.agent_cpu = None if agent is None else copy.deepcopy(agent).to("cpu").eval()   # DRL en CPU, un hilo
        self.market, self.tabla = market, tabla
        self.T = np.asarray(time_matrix, dtype=float)
        self.dist, self.diesel, self.N = distance_arr, diesel_arr, num_nodes
        self.ltr = ltr_stack
        self.avail, self.p95 = avail_expected_arr, reward_global_p95
        self.ev = calcular_ev_pronostico(market, distance_arr, diesel_arr)   # una vez por evaluación
        self.politicas = {
            "B1": PoliticaB1(distance_arr, diesel_arr),
            "B2": PoliticaB2(market, self.T, distance_arr, diesel_arr),
            "B2P": PoliticaB2P(market, self.T, distance_arr, diesel_arr),
        }
        if tabla is not None:                      # None: UB-DP "no tratable" ⇒ sin B3
            self.politicas["B3"] = PoliticaB3(tabla, distance_arr, diesel_arr)
        self.baselines = tuple(b for b in _BASELINES if b in self.politicas)

    def politicas_drl(self, env) -> dict:
        """DRL completo y destino del DRL con cada regla de precio (comparación B), sobre `env`."""
        return {"DRL": PoliticaDRL(self, env, "drl"),
                **{f"DRL-dest+{r}": PoliticaDRL(self, env, r) for r in REGLAS_PRECIO}}


# ── UB-PI ─────────────────────────────────────────────────────

def mundo_ub_pi(tablero: Tablero) -> tuple:
    """(loads, tarifas) para solve_ub_pi a partir de un Tablero (D, N, N, ...): con el competidor conocido, en
    cada lane se cotiza el nivel más alto que gana (el nivel 0 siempre gana), así que no hay rechazos."""
    gana_k = gana(U_NIVELES_ARR, tablero.u_comp[..., None])                       # (D, N, N, K)
    k_mejor = (N_NIVELES - 1) - np.argmax(gana_k[..., ::-1], axis=-1)            # el más alto que gana
    tarifa = np.take_along_axis(tablero.precios, k_mejor[..., None], axis=-1)[..., 0]
    return tablero.loads, tarifa


def tablero_replay(market, semilla: int, dias_abs) -> Tablero:
    """Tablero (D, N, N, ...) de los días reales `dias_abs` con el competidor de la semilla del caso: el mismo
    que ve el entorno de replay en cada (nodo, día)."""
    N = market.num_nodes
    nodos = np.arange(N)
    return Tablero.apilar([market.tablero_real(nodos, np.full(N, int(d)), np.full(N, semilla)) for d in dias_abs])


def ub_pi(time_matrix, distance_arr, diesel_arr, start_node, K, mundo, start_day_idx=0) -> tuple:
    """UB-PI: mejor ruta con el mundo completo conocido, vía DP hacia atrás (baselines.ub_pi.solve_ub_pi).

    mundo = (loads [D, N, N], tarifa [D, N, N]) de `mundo_ub_pi` (tarifa = la del nivel más alto que gana).
    Estado (nodo, franja de UBPI_DT_HORAS, k): el camión puede revisitar nodos. Devuelve (margen, ruta);
    (nan, None) si no hay ruta factible.
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


_COLS_METRICAS = ("margen", "violacion", "frac_millas_cargadas", "n_cargas", "duracion", "ms_decision", "ruta",
                  "ruta_precios", "pasos", "n_cotizaciones", "n_ganadas", "horas_rechazo", "niveles", "usd_mi_sobre_mediana",
                  "dinero_en_la_mesa")


def _fila(modo, K, mundo, nodo, dia_abs, start_day, world_seed, dow, metodo, m, ub_dp, ub_pi_val) -> dict:
    fila = {
        "modo": modo, "K": _etiqueta_k(K), "mundo": mundo, "nodo_inicio": int(nodo),
        "dia_abs": int(dia_abs), "start_day_idx": int(start_day),
        "dia_semana": int(dow), "world_seed": -1 if world_seed is None else int(world_seed),
        "metodo": metodo,
    }
    fila.update({c: m[c] for c in _COLS_METRICAS})
    if modo != "simulado":
        fila["dinero_en_la_mesa"] = float("nan")          # solo en mundos simulados (§8)
    fila.update({"ub_dp": ub_dp, "ub_pi": ub_pi_val})
    return fila


def _firma_cache(ctx, *partes) -> str:
    """Huella de lo que determina los baselines y las cotas: los parámetros del mercado, el ev, la tabla V
    de UB-DP, la regla de precio de los baselines y el diseño de la evaluación. Si cambia algo de eso, el
    caché no se reutiliza."""
    h = hashlib.sha1(ctx.market.firma_params().encode())
    h.update(np.ascontiguousarray(ctx.ev.ev).tobytes())   # q y b salen de los parámetros del mercado (ya en firma_params)
    if ctx.tabla is None:
        h.update(b"ub-dp no tratable")
    else:
        h.update(np.ascontiguousarray(ctx.tabla.V_k).tobytes())
        h.update(np.ascontiguousarray(ctx.tabla.V_libre).tobytes())
    h.update(repr((partes, MAX_DURATION, REGLA_PRECIO_BASELINES, "cotizacion", "pasos por arco")).encode())
    return h.hexdigest()


def _faltantes(df: pd.DataFrame, k_levels, nombres=_BASELINES) -> set:
    """Pares (K, método) de los baselines `nombres` que `df` no trae para los niveles de K pedidos."""
    hay = set(zip(df["K"], df["metodo"]))
    return {(K, nombre) for K in k_levels for nombre in nombres if (_etiqueta_k(K), nombre) not in hay}


def _completar_con_baselines(df: pd.DataFrame, k_levels, baselines_fn, nombres=_BASELINES) -> pd.DataFrame:
    """Añade al caché los pares (K, método) de B1/B2/B2P/B3 que le falten (p. ej. un nivel de K que
    el caché vigente no traía), con las cotas ya calculadas — sin recalcularlas ni recalcular los
    demás baselines. Usado como `completar` en evaluar_mundos_simulados y evaluar_replay."""
    faltan = _faltantes(df, k_levels, nombres)
    if not faltan:
        return df
    print(f"  completando el caché con {sorted(faltan, key=str)}", flush=True)
    extra = baselines_fn(faltan, df.groupby(["K", "mundo"])[["ub_dp", "ub_pi"]].first())
    return pd.concat([df, extra], ignore_index=True)


def _filas_con_cache(cache_path, firma, calcular, verbose=True, completar=None) -> pd.DataFrame:
    """Filas de B1, B2, B2P, B3 y cotas (no dependen del DRL): se calculan una vez y se reusan.

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


def _filas_drl(ctx, env, casos, k_levels, cota, modo, verbose) -> list:
    """Filas del DRL y de la comparación B para cada caso (m, nodo, día del env, día abs, semilla, dow)."""
    pols = ctx.politicas_drl(env)
    filas = []
    with _UnHilo():
        # Calentamiento de torch (primera llamada) fuera de la medición.
        m0, nodo0, dia0, _, ws0, dow0 = casos[0]
        ejecutar_politica(pols["DRL"], env, nodo0, dia0, k_levels[0], ws0, ctx.dist, ctx.diesel, dow0)
        for n, (m, nodo, dia, dia_abs, ws, dow) in enumerate(casos):
            for K in k_levels:
                dp, pi = cota.loc[(_etiqueta_k(K), m)]
                for nombre, pol in pols.items():
                    res = ejecutar_politica(pol, env, nodo, dia, K, ws, ctx.dist, ctx.diesel, dow)
                    filas.append(_fila(modo, K, m, nodo, dia_abs, dia, ws, dow, nombre, res, dp, pi))
            if verbose and (n + 1) % 100 == 0:
                print(f"  {modo} (DRL y comparación B): {n + 1}/{len(casos)}", flush=True)
    return filas


def evaluar_mundos_simulados(agent, market, tabla, time_matrix, rate_train, loads_train,
                             distance_arr, diesel_arr, num_nodes, ltr_stack,
                             avail_expected_arr, reward_global_p95,
                             M=MARKET_M_EVAL_WORLDS, k_levels=(K_MAX, None), verbose=True,
                             cache_path=None, semilla_eval=MARKET_SEED_EVAL) -> pd.DataFrame:
    """DRL, comparación B y B1, B2, B2P, B3 sobre los mismos M mundos simulados, por nivel de K.

    Los mismos mundos (nodo, día, world_seed) se usan en todos los niveles de K y todos los métodos.
    Devuelve un DataFrame largo: una fila por (K, mundo, método) con el margen en USD, la factibilidad,
    las métricas de cotización y operativas y las cotas ub_dp / ub_pi del mundo.

    semilla_eval : semilla del diseño de los M mundos (MARKET_SEED_EVAL = conjunto de desarrollo;
        MARKET_SEED_EVAL_FINAL = conjunto B, test final). Usar un cache_path distinto por semilla.
    cache_path : B1, B2, B2P, B3, UB-PI y UB-DP no dependen del DRL, así que con una ruta se calculan una
        sola vez y se reusan entre corridas (mientras no cambie el proceso, la tabla V ni el diseño).
    """
    ctx = ContextoEval(agent, market, tabla, time_matrix, distance_arr, diesel_arr, num_nodes,
                       ltr_stack, avail_expected_arr, reward_global_p95)
    env = make_sim_env(time_matrix, rate_train, loads_train, distance_arr, diesel_arr, num_nodes, market)
    dis = disenar_mundos(num_nodes, M, market, num_dias=rate_train.shape[0], semilla=semilla_eval)

    def cotas(nodo, dow, K, mundo_pi):
        pi, _ = ub_pi(time_matrix, distance_arr, diesel_arr, nodo, K, mundo_pi)
        dp = (tabla.ub_dp(nodo, dow, K) if tabla is not None and (K is None or 2 <= K <= tabla.K)
              else float("nan"))
        return dp, pi

    def baselines(faltan=None, cota_previa=None):
        """faltan None: todo (cotas y baselines). Con un conjunto de (K, método): solo esos
        baselines, con las cotas ya calculadas (`cota_previa`)."""
        filas = []
        with _UnHilo():
            for m in range(M):
                nodo, dow = int(dis["nodo"][m]), int(dis["dow"][m])
                dia, ws = int(dis["start_day"][m]), int(dis["world_seed"][m])
                mundo_pi = (mundo_ub_pi(market.mundo_cotizacion(ws, _DIAS_SIMULADOS_UB_PI, dia))
                            if faltan is None else None)
                for K in k_levels:
                    if faltan is None:
                        dp, pi = cotas(nodo, dow, K, mundo_pi)
                    else:
                        dp, pi = cota_previa.loc[(_etiqueta_k(K), m)]
                    for nombre in ctx.baselines:
                        if faltan is not None and (K, nombre) not in faltan:
                            continue
                        res = ejecutar_politica(ctx.politicas[nombre], env, nodo, dia, K, ws,
                                                distance_arr, diesel_arr, dow)
                        filas.append(_fila("simulado", K, m, nodo, dia, dia, ws, dow, nombre, res, dp, pi))
                if verbose and (m + 1) % 100 == 0:
                    print(f"  mundos simulados (baselines y cotas): {m + 1}/{M}", flush=True)
        return pd.DataFrame(filas)

    def completar(df):
        return _completar_con_baselines(df, k_levels, baselines, ctx.baselines)

    firma = _firma_cache(ctx, "simulado", num_nodes, M, semilla_eval, tuple(k_levels), rate_train.shape[0])
    base = _filas_con_cache(cache_path, firma, baselines, verbose, completar)
    cota = base.groupby(["K", "mundo"])[["ub_dp", "ub_pi"]].first()

    casos = [(m, int(dis["nodo"][m]), int(dis["start_day"][m]), int(dis["start_day"][m]),
              int(dis["world_seed"][m]), int(dis["dow"][m])) for m in range(M)]
    df = _combinar(base, pd.DataFrame(_filas_drl(ctx, env, casos, k_levels, cota, "simulado", verbose)), k_levels)
    df.attrs["ub_dp_no_tratable"] = tabla is None
    _avisar_si_supera_cota(df)
    return df


def casos_replay(num_nodes: int, num_dias: int, max_casos: int = REPLAY_MAX_CASOS,
                 semilla: int = MARKET_SEED_EVAL + 1) -> tuple:
    """Casos (nodo, día de evaluación, semilla del competidor): todos los pares si caben en `max_casos`; si
    no, un subconjunto balanceado (cada nodo y cada día aparecen ⌊·⌋ o ⌈·⌉ veces, barajados por separado).
    La semilla de cada caso fija el competidor que enfrentan todos los métodos en ese caso."""
    rng = np.random.default_rng(semilla)
    if num_nodes * num_dias <= max_casos:
        nodos, dias = np.meshgrid(np.arange(num_nodes), np.arange(num_dias), indexing="ij")
        nodos, dias = nodos.ravel(), dias.ravel()
    else:
        nodos = _asignacion_balanceada(num_nodes, max_casos, rng)
        dias = _asignacion_balanceada(num_dias, max_casos, rng)
    semillas = rng.integers(0, 2**31 - 1, size=len(nodos), dtype=np.int64)
    return nodos, dias, semillas


def evaluar_replay(agent, market, tabla, time_matrix, rate_eval, loads_eval,
                   distance_arr, diesel_arr, num_nodes, ltr_eval,
                   avail_expected_arr, reward_global_p95,
                   k_levels=(K_MAX, None), max_casos=REPLAY_MAX_CASOS, verbose=True,
                   cache_path=None) -> pd.DataFrame:
    """El mismo arnés sobre los días reales de prueba (oct 2026, fuera de muestra): cargas y cuantiles reales
    del día, competidor sorteado con la semilla del caso.

    Fuera de muestra respecto del proceso (se ajusta solo con los TRAIN_DAYS de entrenamiento). ltr_eval
    debe estar recortado a los mismos días que rate_eval. UB-PI se resuelve sobre los tableros reales con
    el competidor del caso; UB-DP no aplica (es la referencia del proceso simulado).
    cache_path : como en evaluar_mundos_simulados, para B1, B2, B2P, B3 y UB-PI.
    """
    ctx = ContextoEval(agent, market, tabla, time_matrix, distance_arr, diesel_arr, num_nodes,
                       ltr_eval, avail_expected_arr, reward_global_p95)
    env = make_replay_env(time_matrix, rate_eval, loads_eval, distance_arr, diesel_arr, num_nodes,
                          day_offset=TRAIN_DAYS, market=market)
    nodos, dias, semillas = casos_replay(num_nodes, rate_eval.shape[0], max_casos)
    D = rate_eval.shape[0]

    def baselines(faltan=None, cota_previa=None):
        filas = []
        with _UnHilo():
            for c, (nodo, dia, ws) in enumerate(zip(nodos, dias, semillas)):
                nodo, dia, ws = int(nodo), int(dia), int(ws)
                dow = int(market.dia_semana(TRAIN_DAYS + dia))
                mundo_pi = (mundo_ub_pi(tablero_replay(market, ws, TRAIN_DAYS + np.arange(D)))
                            if faltan is None else None)
                for K in k_levels:
                    if faltan is None:
                        pi, _ = ub_pi(time_matrix, distance_arr, diesel_arr, nodo, K, mundo_pi, start_day_idx=dia)
                    else:
                        pi = cota_previa.loc[(_etiqueta_k(K), c), "ub_pi"]
                    for nombre in ctx.baselines:
                        if faltan is not None and (K, nombre) not in faltan:
                            continue
                        res = ejecutar_politica(ctx.politicas[nombre], env, nodo, dia, K, ws,
                                                distance_arr, diesel_arr, dow)
                        filas.append(_fila("replay", K, c, nodo, TRAIN_DAYS + dia, dia, ws, dow,
                                           nombre, res, float("nan"), pi))
                if verbose and (c + 1) % 100 == 0:
                    print(f"  replay (baselines y cotas): {c + 1}/{len(nodos)}", flush=True)
        return pd.DataFrame(filas)

    def completar(df):
        return _completar_con_baselines(df, k_levels, baselines, ctx.baselines)

    firma = _firma_cache(ctx, "replay", num_nodes, max_casos, tuple(k_levels), rate_eval.shape[0],
                         loads_eval.shape, float(loads_eval.sum()), float(rate_eval.sum()))
    base = _filas_con_cache(cache_path, firma, baselines, verbose, completar)
    cota = base.groupby(["K", "mundo"])[["ub_dp", "ub_pi"]].first()

    casos = [(c, int(n), int(d), TRAIN_DAYS + int(d), int(ws), int(market.dia_semana(TRAIN_DAYS + int(d))))
             for c, (n, d, ws) in enumerate(zip(nodos, dias, semillas))]
    df = _combinar(base, pd.DataFrame(_filas_drl(ctx, env, casos, k_levels, cota, "replay", verbose)), k_levels)
    df.attrs["ub_dp_no_tratable"] = tabla is None
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

    Cada fila trae grupo, destino y precio (qué decide el método, ver DECISION) y dif_vs_drl_usd: margen del
    método − margen del DRL en los mismos mundos (media pareada; negativo = por debajo del DRL).
    Métricas de cotización (§8): tasa_aceptacion_pct (cargas ganadas / cotizaciones, sobre el total del
    grupo), cotizaciones_por_arco, horas_rechazo (media por ruta), nivel_medio (índice 0–9 del nivel
    cotizado) y u_medio (su percentil u, media por cotización), usd_mi_sobre_mediana (USD/mi cobrado − mediana de la lane, media por carga ganada) y
    dinero_en_la_mesa (media por ruta; solo mundos simulados).
    Los gaps comparan promedios sobre los mismos mundos: (ref − método) / |ref| · 100, con ref = UB-PI
    (cota) o UB-DP (política que ignora el mercado; un gap negativo significa que el método la supera).
    margen_medio va con su IC del 95 % por bootstrap sobre mundos (BOOTSTRAP_B remuestreos,
    semilla fija): margen_ic95_low / margen_ic95_high. Cada grupo (modo, K, método) usa su
    propio Generator con la semilla fija, así que el IC no depende del orden de las filas.
    Se añaden las filas de UB-PI (cota) y UB-DP (valor de la política que ignora el mercado, NO es
    cota: el DRL, que observa el índice, puede superarla), siempre separadas: nunca se combinan ni se
    promedian. Si df.attrs["ub_dp_no_tratable"], B3 y UB-DP aparecen como filas con nota "no tratable". La columna `nota` marca los K donde UB-DP > UB-PI (la política que ignora el mercado no
    puede superar a la de información perfecta: señal de desajuste entre la DP y el entorno).
    """
    filas = []
    for (modo, K), g in df.groupby(["modo", "K"], sort=False):
        mundos = g.drop_duplicates("mundo")
        nota = NOTA_UB_DP_FLOJA if ub_dp_floja(mundos["ub_dp"], mundos["ub_pi"]) else None
        drl = g[g["metodo"] == "DRL"].set_index("mundo")["margen"]

        def dif_vs_drl(serie_por_mundo):
            if drl.empty or (serie_por_mundo is drl):
                return float("nan") if drl.empty else 0.0
            d = (serie_por_mundo - drl).dropna()
            return float(d.mean()) if len(d) else float("nan")
        for metodo in [m for m in _METODOS if m in set(g["metodo"])]:
            gm = g[g["metodo"] == metodo]
            ic = ic_bootstrap_media(gm["margen"])
            n_cot, n_gan = gm["n_cotizaciones"].sum(), gm["n_ganadas"].sum()
            arcos = gm["ruta"].map(lambda r: len(r) - 1).sum()
            niv = [n for lista in gm["niveles"] for n in lista]
            filas.append({
                "modo": modo, "K": K, "metodo": metodo, **_describir(metodo), "n_mundos": len(gm),
                "margen_medio": gm["margen"].mean(),
                "margen_ic95_low": ic[0], "margen_ic95_high": ic[1],
                "dif_vs_drl_usd": 0.0 if metodo == "DRL" else dif_vs_drl(gm.set_index("mundo")["margen"]),
                "gap_ub_dp_pct": gap_pct(gm["ub_dp"], gm["margen"]),
                "gap_ub_pi_pct": gap_pct(gm["ub_pi"], gm["margen"]),
                "violacion_pct": 100.0 * gm["violacion"].mean(),
                "millas_cargadas_pct": 100.0 * gm["frac_millas_cargadas"].mean(),
                "cargas_medias": gm["n_cargas"].mean(),
                "duracion_media_h": gm["duracion"].mean(),
                "ms_por_decision": gm["ms_decision"].mean(),
                "tasa_aceptacion_pct": 100.0 * n_gan / n_cot if n_cot else float("nan"),
                "cotizaciones_por_arco": n_cot / arcos if arcos else float("nan"),
                "horas_rechazo": gm["horas_rechazo"].mean(),
                "nivel_medio": float(np.mean(niv)) if niv else float("nan"),
                "u_medio": float(np.mean(U_NIVELES_ARR[np.asarray(niv, dtype=int)])) if niv else float("nan"),
                "usd_mi_sobre_mediana": gm["usd_mi_sobre_mediana"].mean(),
                "dinero_en_la_mesa": gm["dinero_en_la_mesa"].mean(),
                "nota": nota,
            })
        if df.attrs.get("ub_dp_no_tratable"):
            for nombre in ("B3", "UB-DP"):
                filas.append({"modo": modo, "K": K, "metodo": nombre, **_describir(nombre), "n_mundos": 0,
                              "nota": NOTA_NO_TRATABLE})
        for nombre, col in (("UB-PI", "ub_pi"), ("UB-DP", "ub_dp")):
            if mundos[col].notna().any():
                ic = ic_bootstrap_media(mundos[col])
                filas.append({"modo": modo, "K": K, "metodo": nombre, **_describir(nombre),
                              "n_mundos": int(mundos[col].notna().sum()),
                              "margen_medio": mundos[col].mean(),
                              "margen_ic95_low": ic[0], "margen_ic95_high": ic[1],
                              "dif_vs_drl_usd": dif_vs_drl(mundos.set_index("mundo")[col]),
                              "nota": nota if col == "ub_dp" else None})
    return pd.DataFrame(filas)


# Columnas del resumen de consola: (columna, encabezado, formato).
_COLS_CONSOLA = (
    ("metodo", "método", "{}"), ("precio", "precio", "{}"),
    ("margen_medio", "margen USD", "{:,.0f}"), ("dif_vs_drl_usd", "Δ vs DRL", "{:+,.0f}"),
    ("gap_ub_pi_pct", "gap UB-PI %", "{:.1f}"), ("tasa_aceptacion_pct", "acept. %", "{:.0f}"),
    ("u_medio", "u cotizado", "{:.2f}"), ("usd_mi_sobre_mediana", "USD/mi vs mediana", "{:+.3f}"),
    ("horas_rechazo", "h rechazos", "{:.2f}"), ("violacion_pct", "viol. %", "{:.1f}"),
)


def imprimir_resumen(df: pd.DataFrame, titulo: str) -> None:
    """Resumen compacto para la consola, por K y por grupo (DRL y comparación B, baselines, cotas).

    El detalle completo (IC, gaps contra UB-DP, millas cargadas, dinero en la mesa, …) va al Excel."""
    res = resumen_metodos(df)
    print(f"\n  {titulo}")
    def fmt(v, f):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return "—"
        try:
            return f.format(v)
        except (ValueError, TypeError):
            return str(v)
    for K, gk in res.groupby("K", sort=False):
        print(f"\n  K = {K}")
        tabla = [[enc for _, enc, _ in _COLS_CONSOLA]]
        for grupo, gg in gk.groupby("grupo", sort=False):
            tabla.append([f"· {grupo}"])
            for _, r in gg.iterrows():
                fila = [fmt(r.get(c), f) for c, _, f in _COLS_CONSOLA]
                if isinstance(r.get("nota"), str) and r["nota"] == NOTA_NO_TRATABLE:
                    fila = [r["metodo"], "no tratable"] + ["—"] * (len(_COLS_CONSOLA) - 2)
                tabla.append(fila)
        anchos = [max(len(f[i]) for f in tabla if len(f) > 1) for i in range(len(_COLS_CONSOLA))]
        for f in tabla:
            if len(f) == 1:
                print(f"    {f[0]}")
            else:
                print("    " + "  ".join(v.ljust(a) if i < 2 else v.rjust(a) for i, (v, a) in enumerate(zip(f, anchos))))
    if (res["nota"] == NOTA_UB_DP_FLOJA).any():
        print(f"\n  * {NOTA_UB_DP_FLOJA}")
    if (res["nota"] == NOTA_NO_TRATABLE).any():
        print(f"\n  B3 / UB-DP: {NOTA_NO_TRATABLE}")


# ── Output ────────────────────────────────────────────────────
def save_results(sim_df, replay_df, resumen_rows, output_path):
    from excel_report import build_formatted_excel
    build_formatted_excel(sim_df, replay_df, resumen_rows, output_path)
    print(f"Results saved → {output_path}")


def plot_ppo_diagnostics(training_log: list, num_nodes: int, output_path: str):
    """Gráfica 2×3 con métricas internas de PPO por update.

    Métricas graficadas
    -------------------
    Fila 1 : Reward promedio | Explained Variance | Entropía (destino y precio) | Rutas válidas
    Fila 2 : Policy Loss + Value Loss | KL Divergence | Clip Fraction | Cotización (aceptación y nivel)
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
        ax_ent = axes[0, 2]
        ax_ent.plot(updates, df["entropy"], linewidth=0.9, color="mediumpurple", label="Destino")
        if "entropy_precio" in df:
            ax_ent.plot(updates, df["entropy_precio"], linewidth=0.9, color="seagreen", label="Precio")
        ax_ent.set_title("Policy Entropy"); ax_ent.set_xlabel("PPO Update"); ax_ent.set_ylabel("Entropy")
        ax_ent.grid(True, alpha=0.3); ax_ent.legend(fontsize=8)

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
        ax_cot = axes[1, 3]
        if "tasa_aceptacion" in df:
            ax_cot.plot(updates, df["tasa_aceptacion"], linewidth=0.9, color="steelblue", label="Aceptación")
            ax_cot.set_ylim(-0.05, 1.05)
            ax_cot.set_ylabel("Aceptación (rollout)")
            ax_niv = ax_cot.twinx()
            ax_niv.plot(updates, df["nivel_medio"], linewidth=0.9, color="darkorange", label="Nivel medio")
            ax_niv.set_ylim(-0.3, 9.3)
            ax_niv.set_ylabel("Nivel cotizado medio (0-9)")
            lineas = ax_cot.get_lines() + ax_niv.get_lines()
            ax_cot.legend(lineas, [l.get_label() for l in lineas], fontsize=8)
        ax_cot.set_title("Cotización (rollout)"); ax_cot.set_xlabel("PPO Update")
        ax_cot.grid(True, alpha=0.3)

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
