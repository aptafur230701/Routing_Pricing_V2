"""
evaluation.py
=============
Evaluación posterior al entrenamiento (baselines_drl_ruteo_estocastico.md, especificacion_fase0_ruteo_pricing.md §7-8):
  · ejecutar_lote                — corre una política sobre B episodios a la vez en el entorno (GPU)
  · evaluar_mundos_simulados     — DRL, comparación B y B1, B2, B2P, B3 sobre los mismos mundos simulados con día
                                   de salida en el período de PRUEBA (octubre), + UB-PI (cota)
  · resumen_metodos              — márgenes, gaps, aceptación y métricas operativas por método

Split temporal (config.periodos): el proceso de mercado, el DRL y las tablas de B2, B2P y B3 salen solo de los días
de entrenamiento (jun–ago); la validación (sep) elige el checkpoint; aquí se reporta sobre prueba (oct). La única
cota es UB-PI, que se resuelve mundo a mundo con el mundo evaluado.
  · save_results, plot_*         — Excel y figuras

Arnés único: cada política corre sobre TODOS los episodios (mundos × niveles de K) a la vez en
`VectorRoutingEnv`, en el dispositivo (la GPU en el servidor). En cada paso la política elige destino y precio
para las filas activas; si la carga se pierde, vuelve a decidir en el mismo nodo con la lane enmascarada. El DRL
decide en un solo forward batcheado y los baselines (B1, B2, B2P, B3, baselines/politicas.py) también deciden para
todas las filas a la vez, con la `ObsLote` del entorno (máscara, tablero observado, nodo, hora y k restante). Nadie
ve el competidor. Todos los métodos corren sobre el MISMO `Mundos` (los
mismos tableros y competidores: números aleatorios comunes).

Métodos (comparación B, §7.1):
  · DRL                  — destino y precio del DRL.
  · DRL-dest+<regla>     — destino del DRL, precio con la regla fija (mediana, be_margen, miope).
  · B1, B2, B2P, B3      — destino del baseline, precio con REGLA_PRECIO_BASELINES (miope).
Si la tabla de la DP es "no tratable" a ese tamaño (tabla None, baselines/dp_relajada.py), B3 no se calcula y el
resumen lo reporta como "no tratable" (un resultado válido, no un error).

El margen de una ruta es la suma del reward de cada lane en USD con el precio cotizado
(env.last_raw_reward: p·d − c si se ganó la carga, −c si fue VACÍO, 0 en un rechazo o una espera).
"""

import hashlib
import os
import pickle
import time
import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
from config import (
    DEVICE, MAX_DURATION, K_MAX, MARKET_DIAS_EPISODIO,
    MARKET_SEED_EVAL, MARKET_M_EVAL_WORLDS, REGLA_PRECIO_BASELINES, REPOSICION_MAX_MILLAS,
)
from baselines.politicas import (
    ObsLote, PoliticaB1, PoliticaB2, PoliticaB2P, PoliticaB3, calcular_ev_pronostico, cotizar_regla,
)
from greedy_rollout import EntradasAgente, construir_features, contexto_agente
from model.encoder import COT_COLS
from pricing import REGLAS_PRECIO, U_NIVELES_ARR, N_NIVELES, prob_ganar
from routing_env import VectorRoutingEnv
from baselines.dp_relajada import PENALIZACION_USD
from config import COTIZACIONES_MAX_POR_NODO, BANDA_PRECIO_MASA, BE_MARGEN
from baselines.ub_pi import solve_ub_pi_lote
from stats_analysis import gap_pct, ic_bootstrap_media

_BASELINES = ("B1", "B2", "B2P", "B3")   # no dependen del agente: se calculan una vez y se cachean
_DRL_REGLAS = tuple(f"DRL-dest+{r}" for r in REGLAS_PRECIO)   # comparación B
_METODOS = ("DRL",) + _DRL_REGLAS + _BASELINES
# Nombre visible de cada método en los reportes (los códigos internos — "DRL-dest+miope", "UB-PI", … — quedan en
# los datos y en mundos_AM.csv), qué decide (columnas "destino" y "precio" del resumen) y el grupo en que se reporta.
_PRECIO_TRAMO = "El que maximiza la ganancia esperada del tramo, sin mirar los siguientes"
NOMBRE = {
    "DRL": "DRL (modelo propuesto)",
    "DRL-dest+mediana": "Rutas del DRL + tarifa mediana",
    "DRL-dest+be_margen": f"Rutas del DRL + costo + {BE_MARGEN:.2f} USD/mi",
    "DRL-dest+miope": "Rutas del DRL + mejor precio del tramo",
    "B1": "B1 · Mejor tramo inmediato",
    "B2": "B2 · Plan con promedios",
    "B2P": "B2P · Plan con pronóstico",
    "B3": "B3 · Programación dinámica",
    "UB-PI": "Máximo teórico (información perfecta)",
}
_PRECIO_REGLA = {
    "mediana": "Siempre la tarifa mediana publicada del día",
    "be_margen": f"La tarifa publicada más baja que cubre el costo + {BE_MARGEN:.2f} USD/mi",
    "miope": _PRECIO_TRAMO,
}
DECISION = {
    "DRL": ("El DRL, pensando en todo el ciclo", "El DRL, según el mercado del día"),
    **{f"DRL-dest+{r}": ("Las mismas del DRL", _PRECIO_REGLA.get(r, r)) for r in REGLAS_PRECIO},
    "B1": ("La ciudad con mayor ganancia esperada en el próximo tramo", _PRECIO_TRAMO),
    "B2": ("En cada ciudad replanifica el resto de la ruta con cargas y tarifas promedio", _PRECIO_TRAMO),
    "B2P": ("Como B2, ajustando las tarifas promedio con un pronóstico del mercado", _PRECIO_TRAMO),
    "B3": ("Tabla de programación dinámica calculada con el mercado promedio",
           f"El de la misma tabla (hasta {COTIZACIONES_MAX_POR_NODO} intentos por ciudad)"),
    "UB-PI": ("Conoce de antemano todas las cargas y los precios de la competencia",
              "El más alto que todavía gana la carga"),
}
GRUPO_DRL, GRUPO_BASELINES, GRUPO_COTAS = (
    "Modelo propuesto, y sus mismas rutas con reglas de precio fijas (muestran cuánto aporta que el DRL elija el precio)",
    "Métodos de comparación",
    "Referencias (no son métodos aplicables en la práctica)")


def _grupo(metodo: str) -> str:
    if metodo == "DRL" or metodo.startswith("DRL-dest+"):
        return GRUPO_DRL
    return GRUPO_COTAS if metodo.startswith("UB-") else GRUPO_BASELINES


def _describir(metodo: str) -> dict:
    destino, precio = DECISION.get(metodo, ("", ""))
    return {"grupo": _grupo(metodo), "nombre": NOMBRE.get(metodo, metodo), "destino": destino, "precio": precio}
NOTA_NO_TRATABLE = ("no se calcula: con este número de ciudades el cálculo es demasiado largo o no cabe en memoria")


def _sync():
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()


# ── Políticas del arnés ──────────────────────────────────────

class PoliticaBaseline:
    """Una política baseline (B1, B2, B2P, B3) sobre el lote: lee la `ObsLote` del entorno y decide destino y precio de
    todas las filas a la vez, en el dispositivo."""

    def __init__(self, politica, nombre: str):
        self.p, self.nombre = politica, nombre
        self.precio = "regla"

    def decidir(self, env, masks, activo) -> tuple:
        a, o = self.p.decidir(ObsLote.del_entorno(env, masks), activo)
        return a, o, None


class PoliticaDRL:
    """El agente como política del arnés: destino con la cabeza de destino (argmax, batcheado en el dispositivo);
    precio con la cabeza de precio (precio="drl"), con una regla fija (comparación B) o con la tabla de B3 ("dp").
    Lee el estado observable del entorno (las mismas features que en el entrenamiento)."""

    def __init__(self, ctx, precio: str = "drl"):
        if precio not in ("drl", "dp") and precio not in REGLAS_PRECIO:
            raise ValueError(f"precio desconocido: {precio!r}")
        if precio == "dp" and "B3" not in ctx.politicas:
            raise ValueError('precio="dp" necesita la tabla de la DP (B3); a este tamaño no se calculó')
        self.ctx, self.precio = ctx, precio
        self.nombre = "DRL" if precio == "drl" else f"DRL-dest+{precio}"

    def decidir(self, env, masks, activo) -> tuple:
        c = self.ctx
        with torch.no_grad():
            feats = construir_features(env, c.entradas(env))
            h_t, emb = contexto_agente(c.agent, feats)
            cot = feats["node_feats"][..., COT_COLS]
            a, o = c.agent.decoder.greedy_cotizacion(h_t, emb, masks, feats["opciones"], cot)
            # π(o | s, j) del destino elegido: el rango de precio de la evaluación (solo la cabeza de precio).
            probs = (c.agent.decoder.probs_precio(h_t, emb, masks, feats["opciones"], cot, a)
                     if self.precio == "drl" else None)
        if self.precio != "drl":
            obs = ObsLote.del_entorno(env, masks)
            o = (c.politicas["B3"].precio_en(obs, a) if self.precio == "dp"
                 else cotizar_regla(obs, a, c.politicas["B1"]._est(env.device), self.precio))
        return a, o, probs


def banda_precio(probs, o: int, masa: float = BANDA_PRECIO_MASA) -> tuple:
    """(nivel más bajo, nivel más alto) del rango de precio: los niveles contiguos alrededor del elegido `o` que
    juntan al menos `masa` de la probabilidad de la cabeza de precio (renormalizada sobre los niveles, sin VACÍO).
    Se agrega siempre el vecino más probable; son los precios que el DRL considera casi igual de buenos."""
    p = np.asarray(probs[:N_NIVELES], dtype=np.float64)
    p = p / p.sum()
    lo = hi = int(o)
    acum = p[o]
    while acum < masa - 1e-12 and (lo > 0 or hi < N_NIVELES - 1):
        izq = p[lo - 1] if lo > 0 else -1.0
        der = p[hi + 1] if hi < N_NIVELES - 1 else -1.0
        if der >= izq:
            hi += 1
            acum += der
        else:
            lo -= 1
            acum += izq
    return lo, hi


# ── Arnés: una política, B episodios ─────────────────────────

def ejecutar_lote(politica, env, mundos, nodos, k_list, distance_arr) -> list:
    """Corre `politica` sobre los B episodios de `mundos` (nodos de inicio `nodos`, presupuesto `k_list`) y
    devuelve una lista de B dicts con las métricas de cada ruta.

    Métricas: margen (suma de env.last_raw_reward, más PENALIZACION_USD si el ciclo falla: la misma penalización del
    entrenamiento y de la DP de B3), violacion (no cerró el ciclo en casa dentro de max_duration; no salir
    de casa por no tener ninguna jugada, env.sin_salida, no es violación), frac_millas_cargadas, n_cargas,
    duracion, ms_decision (tiempo de decisión por paso: el del lote dividido por las filas activas), ruta (solo
    viajes), n_cotizaciones, n_ganadas, horas_rechazo, niveles (lista), usd_mi_sobre_mediana (media de precio cobrado
    − mediana de la lane en las cargas ganadas), dinero_en_la_mesa (Σ (p_comp − p)·d en las cargas ganadas),
    n_esperas, horas_espera, n_reposiciones (viajes vacíos a un nodo que no es casa), n_bajo_be (cargas ganadas a un
    precio menor que el costo por milla de la lane) y usd_bajo_be (Σ de lo que se perdió en ellas, ≤ 0).
    Detalle por arco: `pasos` (una entrada por step: cotización ganada, cotización perdida, viaje vacío o
    espera, con el nivel, el percentil u, el precio cotizado, la mediana, P(ganar), el precio del competidor,
    las millas, el reward del paso y el estado del mercado de la decisión) y `ruta_precios`, la misma ruta en texto,
    p. ej. "3→7 cotiza 2.95 USD/mi (u=0.6) ✗ | 3 espera 9.5 h | 3→5 cotiza 2.80 USD/mi (u=0.4) ✓ | 5→3 vacío".
    """
    masks = env.reset(mundos, nodos, k_list)
    B = env.B
    activo = env.active.clone()
    sin_salida = env.sin_salida.cpu().numpy()
    filas = torch.arange(B, device=env.device)
    log, t_dec, n_dec = [], 0.0, 0
    for _ in range(env.max_steps):
        if not bool(activo.any()):
            break
        _sync()
        t0 = time.perf_counter()
        a, o, probs = politica.decidir(env, masks, activo)
        _sync()
        t_dec += time.perf_counter() - t0
        n_dec += int(activo.sum())
        cur = env.current_node
        antes = dict(
            a=a, o=o, activo=activo, i=cur, t=env.time_elapsed.double(), precios_a=env.precios_row[filas, a],
            mediana=env.rate_row_obs[filas, a].double(), m=env.competencia, dia=env.current_day, ltr=env.ltr_actual,
            ind_i=env.market_index[filas, cur], ind_j=env.market_index[filas, a], n_rech=env.n_rechazos,
            k_rest=torch.where(env.k_restricted, env.k_remaining, torch.full_like(env.k_remaining, -1)),
            costo_a=env.cost_rows[filas, a],
        )
        if probs is not None:
            antes["probs"] = probs
        _, term, masks = env.step(a, o)
        despues = dict(raw=env.last_raw_reward.double(), cotizo=env.last_cotizo, gano=env.last_gano,
                       espero=env.last_espero, precio=env.last_precio, pcomp=env.last_precio_competidor,
                       t2=env.time_elapsed.double(), viol=env.last_time_violation, term=term)
        log.append({k: v.cpu().numpy() for k, v in {**antes, **despues}.items()})
        activo = activo & ~term
    ms = 1000.0 * t_dec / n_dec if n_dec else float("nan")
    dur = env.time_elapsed.double().cpu().numpy()
    nodos = env.start_node.cpu().numpy()
    con_rango = getattr(politica, "precio", None) == "drl"
    return [_metricas_episodio(b, log, int(nodos[b]), bool(sin_salida[b]), float(dur[b]), ms, con_rango,
                               distance_arr) for b in range(B)]


def _metricas_episodio(b, log, casa, sin_salida, duracion, ms, con_rango, distance_arr) -> dict:
    """Las métricas de `ejecutar_lote` para el episodio b, a partir del registro por paso (numpy)."""
    ruta = [casa]
    margen = millas_carg = millas_tot = horas_rech = mesa = horas_esp = usd_bajo_be = 0.0
    n_cargas = n_cot = n_esp = n_repos = n_bajo_be = 0
    niveles, sobre_mediana, pasos, cotizaciones = [], [], [], []
    cerro = sin_salida                                  # sin ninguna jugada desde casa: no sale (válido)
    for L in log:
        if not L["activo"][b]:
            break
        i, a, o = int(L["i"][b]), int(L["a"][b]), int(L["o"][b])
        cotizo, gano, espero = bool(L["cotizo"][b]), bool(L["gano"][b]), bool(L["espero"][b])
        raw = float(L["raw"][b])
        t_antes, t_despues = float(L["t"][b]), float(L["t2"][b])
        mediana = float(L["mediana"][b])
        d_ia = float(distance_arr[i, a])
        rango = None
        if con_rango and o < N_NIVELES and a != i:
            lo, hi = banda_precio(L["probs"][b], o)
            banda = L["precios_a"][b, lo:hi + 1]
            rango = dict(desde=i, hacia=a, millas=d_ia, piso=float(banda.min()), tope=float(banda.max()),
                         referencia=float(L["precios_a"][b, o]), nivel=o, nivel_piso=lo, nivel_tope=hi,
                         break_even=float(L["costo_a"][b] / d_ia), mediana=mediana, gano=gano)
        margen += raw
        u = float(U_NIVELES_ARR[o]) if cotizo else float("nan")
        pasos.append(dict(
            paso=len(pasos) + 1, desde=i, hacia=a,
            accion=(("cotiza y gana" if gano else "cotiza y pierde") if cotizo
                    else "espera" if espero else "viaja vacío"),
            nivel=o if cotizo else None, u=u,
            precio_cotizado_usd_mi=float(L["precios_a"][b, o]) if cotizo else float("nan"),
            mediana_usd_mi=mediana,
            p_ganar=float(prob_ganar(u, float(L["m"][b]))) if cotizo else float("nan"),
            precio_competidor_usd_mi=float(L["pcomp"][b]) if cotizo else float("nan"),
            millas=0.0 if ((cotizo and not gano) or espero) else d_ia,
            reward_usd=raw, horas_transcurridas=t_antes, horas_paso=t_despues - t_antes, rango=rango,
            # Estado del mercado en la decisión (validate_pricing lo usa para estudiar el precio).
            dia_abs=int(L["dia"][b]), ltr=float(L["ltr"][b]), m=float(L["m"][b]),
            indice_origen=float(L["ind_i"][b]), indice_destino=float(L["ind_j"][b]),
            rechazos_previos=int(L["n_rech"][b]), k_restante=int(L["k_rest"][b]) if L["k_rest"][b] >= 0 else None,
        ))
        if rango is not None:
            cotizaciones.append(rango)
        if cotizo:
            n_cot += 1
            niveles.append(o)
        if cotizo and not gano:
            horas_rech += t_despues - t_antes
        elif espero:
            n_esp += 1
            horas_esp += t_despues - t_antes
        else:                                             # viajó (ganó o VACÍO)
            ruta.append(a)
            millas_tot += d_ia
            if gano:
                millas_carg += d_ia
                n_cargas += 1
                p = float(L["precio"][b])
                sobre_mediana.append(p - mediana)
                mesa += (float(L["pcomp"][b]) - p) * d_ia
                if raw < 0:                               # ganó por debajo del costo de la lane
                    n_bajo_be += 1
                    usd_bajo_be += raw
            elif a != casa:
                n_repos += 1
        if L["term"][b]:
            cerro = (len(ruta) > 1 and ruta[-1] == casa) and not bool(L["viol"][b])
            break
    if not cerro:
        margen += PENALIZACION_USD
    return dict(
        margen=margen, violacion=not cerro,
        frac_millas_cargadas=millas_carg / millas_tot if millas_tot > 0 else float("nan"),
        millas_totales=millas_tot, millas_cargadas=millas_carg, n_cargas=n_cargas, duracion=duracion,
        ms_decision=ms, ruta=ruta, n_cotizaciones=n_cot, n_ganadas=n_cargas, horas_rechazo=horas_rech,
        niveles=niveles, usd_mi_sobre_mediana=float(np.mean(sobre_mediana)) if sobre_mediana else float("nan"),
        dinero_en_la_mesa=mesa, n_esperas=n_esp, horas_espera=horas_esp, n_reposiciones=n_repos,
        n_bajo_be=n_bajo_be, usd_bajo_be=usd_bajo_be, pasos=pasos, ruta_precios=ruta_con_precios(pasos),
        cotizaciones=cotizaciones, precios_por_arco=precios_por_arco(pasos) if con_rango else "",
    )


def precios_por_arco(pasos) -> str:
    """Los arcos recorridos con su rango de precio (USD/milla), una línea por arco, p. ej.
    "1-5: 2.15–2.48 (ref. 2.40)" o "5-3: vacío". La referencia es el precio al que se ganó la carga (el del
    margen de la ruta); el rango, los precios que el DRL consideraba casi igual de buenos en esa decisión."""
    lineas = []
    for p in pasos:
        tramo = f"{p['desde']}-{p['hacia']}"
        if p["accion"] == "viaja vacío":
            lineas.append(f"{tramo}: vacío")
        elif p["accion"] == "cotiza y gana":
            r = p.get("rango")
            if r is None:
                lineas.append(f"{tramo}: {p['precio_cotizado_usd_mi']:.2f}")
            else:
                lineas.append(f"{tramo}: {r['piso']:.2f}–{r['tope']:.2f} (ref. {r['referencia']:.2f})")
    return "\n".join(lineas)


def ruta_con_precios(pasos) -> str:
    """La ruta en texto, un tramo por paso: precio cotizado y si se ganó (✓) o se perdió (✗), viaje vacío o espera."""
    partes = []
    for p in pasos:
        tramo = f"{p['desde']}→{p['hacia']}"
        if p["accion"] == "espera":
            partes.append(f"{p['desde']} espera {p['horas_paso']:.1f} h")
        elif p["accion"] == "viaja vacío":
            partes.append(f"{tramo} vacío")
        else:
            marca = "✓" if p["accion"] == "cotiza y gana" else "✗"
            partes.append(f"{tramo} cotiza {p['precio_cotizado_usd_mi']:.2f} USD/mi (u={p['u']:.1f}) {marca}")
    return " | ".join(partes)


class ContextoEval:
    """Datos, entorno y políticas comunes de una evaluación."""

    def __init__(self, agent, market, tabla, time_matrix, distance_arr, diesel_arr, num_nodes,
                 avail_expected_arr, reward_global_p95):
        self.agent = None if agent is None else agent.to(DEVICE).eval()
        self.market, self.tabla = market, tabla
        self.T = np.asarray(time_matrix, dtype=float)
        self.dist, self.diesel, self.N = distance_arr, diesel_arr, num_nodes
        self.avail, self.p95 = avail_expected_arr, reward_global_p95
        self.est = market.estadisticas_dow(distance_arr, diesel_arr)            # entrada del DRL, una vez por evaluación
        self.ev = calcular_ev_pronostico(market, distance_arr, diesel_arr)   # la tabla de B2/B2P (solo entra a la huella del caché)
        self.env = VectorRoutingEnv(time_matrix, distance_arr, diesel_arr, market, device=DEVICE)
        self._entradas = None
        self.politicas = {
            "B1": PoliticaB1(self.T, distance_arr, diesel_arr),
            "B2": PoliticaB2(market, self.T, distance_arr, diesel_arr),
            "B2P": PoliticaB2P(market, self.T, distance_arr, diesel_arr),
        }
        if tabla is not None:                      # None: la DP es "no tratable" ⇒ sin B3
            self.politicas["B3"] = PoliticaB3(tabla, self.T, distance_arr, diesel_arr)
        self.baselines = tuple(b for b in _BASELINES if b in self.politicas)

    def entradas(self, env) -> EntradasAgente:
        if self._entradas is None:
            self._entradas = EntradasAgente.crear(env, self.avail, self.est, self.p95)
        return self._entradas

    def politica(self, nombre: str):
        """La política del arnés para el método `nombre` (DRL, DRL-dest+<regla>, DRL-dest+dp o un baseline)."""
        if nombre == "DRL":
            return PoliticaDRL(self, "drl")
        if nombre.startswith("DRL-dest+"):
            return PoliticaDRL(self, nombre.split("+", 1)[1])
        return PoliticaBaseline(self.politicas[nombre], nombre)


# ── Mundos de evaluación ─────────────────────────────────────

def _etiqueta_k(K) -> str:
    return "libre" if K is None else str(int(K))


def _asignacion_balanceada(n_valores: int, M: int, rng) -> np.ndarray:
    """M valores en [0, n_valores) donde cada uno aparece ⌊M/n⌋ o ⌈M/n⌉ veces, en orden aleatorio."""
    base = np.tile(np.arange(n_valores), -(-M // n_valores))[:M]
    return rng.permutation(base)


def dias_de_salida(num_dias: int) -> int:
    """Cuántos de los primeros días de un período de `num_dias` días sirven de salida: los que dejan ver
    MARKET_DIAS_EPISODIO días distintos sin salir del período. Más tarde el entorno se queda sin días y repite el
    último (esperar devolvería el mismo tablero con las cargas perdidas)."""
    return max(1, num_dias - MARKET_DIAS_EPISODIO + 1)


def disenar_mundos(num_nodes: int, M: int, market, periodo: str = "test",
                   semilla: int = MARKET_SEED_EVAL) -> dict:
    """Diseño de M episodios con día de salida en `periodo` ("val" o "test", `MarketProcess.periodo`) a partir
    de `semilla`.

    Cada nodo aparece ⌊M/N⌋ o ⌈M/N⌉ veces y cada día de la semana ⌊M/7⌋ o ⌈M/7⌉ veces; los
    dos se barajan por separado, así que el día no queda atado al nodo. start_day es un día absoluto del período
    con ese día de la semana, elegido entre los que dejan el episodio completo dentro del período (con octubre:
    oct 1–26). Los mundos (cargas, competidores y δ) se sortean después en el dispositivo con la misma semilla
    (`mundos_eval`, con dia_max = último día del período).
    """
    ini, fin = market.periodo(periodo)
    rng = np.random.default_rng(semilla)
    nodos = _asignacion_balanceada(num_nodes, M, rng)
    dows = _asignacion_balanceada(7, M, rng)
    dias = ini + np.arange(dias_de_salida(fin - ini))
    dow_dia = market.dia_semana(dias)
    inicio = np.array([rng.choice(dias[dow_dia == d]) for d in dows], dtype=np.int64)
    return dict(nodo=nodos.astype(np.int64), dow=dows.astype(np.int64), start_day=inicio, dia_max=fin - 1)


def mundos_eval(market, dia0_abs, dia_max_abs: int, semilla: int):
    """Los mundos simulados de los episodios (uno por día de salida `dia0_abs`): cargas, competidores y δ del
    proceso, sorteados en el dispositivo con `semilla`. Ningún día pasa de `dia_max_abs` (el fin del período)."""
    g = torch.Generator(device=DEVICE).manual_seed(int(semilla))
    return market.sortear_mundos(np.asarray(dia0_abs), MARKET_DIAS_EPISODIO, dia_max_abs, g)


# ── UB-PI ─────────────────────────────────────────────────────

def mundos_ub_pi(market, mundos, bloque: int = 64) -> tuple:
    """(loads, tarifas) (M, D, N, N) para solve_ub_pi a partir de los mundos: el día d es el tablero que ve el entorno
    al llegar el día dia0 + d (recortado al último día con datos). Con los competidores conocidos, en cada lane se toma
    la mejor de las cargas cotizables (a lo sumo COTIZACIONES_MAX_POR_NODO) y se cotiza el nivel más alto que gana. No
    cobra los 15 min de las cargas que habría que perder antes: sigue siendo cota. Si ningún nivel gana en ninguna, la
    lane queda sin carga: solo se puede recorrer vacía, si las reglas lo permiten."""
    tab = market.tensores(mundos.loads.device)
    dev = mundos.loads.device
    D, C, K = mundos.loads.shape[1], mundos.u.shape[-1], N_NIVELES
    u_niv = torch.as_tensor(U_NIVELES_ARR, dtype=torch.float64, device=dev)
    L_out, R_out = [], []
    for ini in range(0, mundos.B, bloque):
        b = torch.arange(ini, min(ini + bloque, mundos.B), device=dev)
        dias = (mundos.dia0[b, None] + torch.arange(D, device=dev)[None, :]).clamp(max=mundos.dia_max)   # (b, D)
        slot = dias - mundos.dia0[b, None]
        loads = mundos.loads[b[:, None], slot]                                              # (b, D, N, N)
        m = tab["m"][dias][..., None, None]                                                 # (b, D, N, 1, 1)
        u_comp = 1.0 - torch.pow(1.0 - mundos.u[b[:, None], slot].double(), 1.0 / m)       # (b, D, N, N, C)
        existe = torch.arange(C, device=dev) < loads[..., None]
        u_mejor = torch.where(existe, u_comp, torch.zeros_like(u_comp)).amax(dim=-1)       # competidor más caro
        gana_k = u_niv < u_mejor[..., None]                                                 # (b, D, N, N, K)
        k_mejor = (K - 1) - gana_k.flip(-1).long().argmax(dim=-1)                          # el más alto que gana
        precios = tab["precios"][dias] * torch.exp(mundos.delta[b])[:, None, None, None, None]
        tarifa = torch.gather(precios, -1, k_mejor[..., None])[..., 0]
        L_out.append(torch.where(gana_k.any(dim=-1), loads, torch.zeros_like(loads)).cpu().numpy())
        R_out.append(tarifa.cpu().numpy())
    return np.concatenate(L_out), np.concatenate(R_out)


def ub_pi(time_matrix, distance_arr, diesel_arr, start_nodes, K, loads, rates) -> np.ndarray:
    """UB-PI de M mundos: valor de la mejor ruta con el mundo completo conocido, vía DP hacia atrás en el dispositivo
    (baselines.ub_pi.solve_ub_pi_lote). loads, rates : (M, D, N, N) de `mundos_ub_pi` (tarifa = la del nivel más alto
    que gana), día 0 = salida. Estado (nodo, franja de UBPI_DT_HORAS, k): el camión puede revisitar nodos."""
    return solve_ub_pi_lote(start_nodes, np.asarray(time_matrix, dtype=float), rates, loads, distance_arr, diesel_arr,
                            MAX_DURATION, k_max=K)


# ── Filas, caché y combinación ────────────────────────────────

_COLS_METRICAS = ("margen", "violacion", "frac_millas_cargadas", "millas_totales", "millas_cargadas", "n_cargas", "duracion", "ms_decision", "ruta",
                  "ruta_precios", "precios_por_arco", "cotizaciones", "pasos", "n_cotizaciones", "n_ganadas", "horas_rechazo", "niveles", "usd_mi_sobre_mediana",
                  "dinero_en_la_mesa", "n_esperas", "horas_espera", "n_reposiciones", "n_bajo_be", "usd_bajo_be")


def _fila(modo, K, mundo, nodo, dia_abs, dow, metodo, m, ub_pi_val) -> dict:
    fila = {
        "modo": modo, "K": _etiqueta_k(K), "mundo": mundo, "nodo_inicio": int(nodo),
        "dia_abs": int(dia_abs), "start_day_idx": int(dia_abs), "dia_semana": int(dow), "metodo": metodo,
    }
    fila.update({c: m[c] for c in _COLS_METRICAS})
    fila["ub_pi"] = ub_pi_val
    return fila


def _firma_cache(ctx, *partes) -> str:
    """Huella de lo que determina los baselines y la cota: los parámetros del mercado, el ev, la tabla V
    de B3, la regla de precio de los baselines, el diseño de la evaluación y el dispositivo (los mundos se
    sortean con el generador del dispositivo). Si cambia algo de eso, el caché no se reutiliza."""
    h = hashlib.sha1(ctx.market.firma_params().encode())
    h.update(np.ascontiguousarray(ctx.ev.ev).tobytes())   # q y b salen de los parámetros del mercado (ya en firma_params)
    if ctx.tabla is None:
        h.update(b"dp no tratable")
    else:
        h.update(np.ascontiguousarray(ctx.tabla.V_k).tobytes())
        h.update(np.ascontiguousarray(ctx.tabla.V_libre).tobytes())
    h.update(repr((partes, DEVICE.type, MAX_DURATION, REGLA_PRECIO_BASELINES, "mundos en el dispositivo", "lote",
                   COTIZACIONES_MAX_POR_NODO, PENALIZACION_USD, REPOSICION_MAX_MILLAS, tuple(U_NIVELES_ARR))).encode())
    return h.hexdigest()


def _faltantes(df: pd.DataFrame, k_levels, nombres=_BASELINES) -> set:
    """Pares (K, método) de los baselines `nombres` que `df` no trae para los niveles de K pedidos."""
    hay = set(zip(df["K"], df["metodo"]))
    return {(K, nombre) for K in k_levels for nombre in nombres if (_etiqueta_k(K), nombre) not in hay}


def _completar_con_baselines(df: pd.DataFrame, k_levels, baselines_fn, nombres=_BASELINES) -> pd.DataFrame:
    """Añade al caché los pares (K, método) de B1/B2/B2P/B3 que le falten (p. ej. un nivel de K que
    el caché vigente no traía), con la cota ya calculada — sin recalcularla ni recalcular los
    demás baselines. Usado como `completar` en evaluar_mundos_simulados."""
    faltan = _faltantes(df, k_levels, nombres)
    if not faltan:
        return df
    print(f"  completando el caché con {sorted(faltan, key=str)}", flush=True)
    extra = baselines_fn(faltan, df.groupby(["K", "mundo"])["ub_pi"].first())
    return pd.concat([df, extra], ignore_index=True)


def _filas_con_cache(cache_path, firma, calcular, verbose=True, completar=None) -> pd.DataFrame:
    """Filas de B1, B2, B2P, B3 y la cota (no dependen del DRL): se calculan una vez y se reusan.

    cache_path None ⇒ sin caché. Un caché con otra firma se recalcula y se sobrescribe.
    completar : función df → df que añade lo que le falte al caché vigente (p. ej. B2 con K libre,
        que un caché anterior no traía) sin recalcular la cota ni los demás baselines."""
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
                print(f"  baselines y UB-PI: reutilizados de {cache_path}")
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


# ── Evaluación: mundos simulados ──────────────────────────────

class _Casos:
    """Los episodios de una evaluación: un mundo por caso, repetido por cada nivel de K (lote = casos × K)."""

    def __init__(self, nodos, dia0_abs, dows, mundos, k_levels):
        self.nodos, self.dia0, self.dows = nodos, dia0_abs, dows
        self.M, self.k_levels = len(nodos), list(k_levels)
        self.mundos = mundos
        self.lote_mundos = mundos.tomar(np.tile(np.arange(self.M), len(self.k_levels)))
        self.lote_nodos = np.tile(nodos, len(self.k_levels))
        self.lote_k = [K for K in self.k_levels for _ in range(self.M)]

    def filas(self, modo, metodo, res, cota) -> list:
        """Filas del DataFrame para los resultados `res` (lista del lote) de `metodo`; cota[(K, m)] = ub_pi."""
        out = []
        for n, K in enumerate(self.k_levels):
            for m in range(self.M):
                out.append(_fila(modo, K, m, self.nodos[m], self.dia0[m], self.dows[m], metodo,
                                 res[n * self.M + m], cota[(_etiqueta_k(K), m)]))
        return out


def _correr_baselines(ctx, casos, modo, nombres_k=None, verbose=True) -> dict:
    """{método: resultados del lote} de los baselines (todos, o solo los pares (K, método) de `nombres_k`)."""
    out = {}
    for nombre in ctx.baselines:
        if nombres_k is not None and not any(n == nombre for _, n in nombres_k):
            continue
        t0 = time.perf_counter()
        out[nombre] = ejecutar_lote(ctx.politica(nombre), ctx.env, casos.lote_mundos, casos.lote_nodos, casos.lote_k,
                                    ctx.dist)
        if verbose:
            print(f"  {modo}: {nombre} en {time.perf_counter() - t0:.1f} s", flush=True)
    return out


def _cotas(ctx, casos, modo, verbose=True) -> dict:
    """{(K, m): ub_pi} de cada caso y nivel de K: UB-PI resuelto con el mundo evaluado."""
    loads, rates = mundos_ub_pi(ctx.market, casos.mundos)
    cota = {}
    t0 = time.perf_counter()
    for K in casos.k_levels:
        pi = ub_pi(ctx.T, ctx.dist, ctx.diesel, casos.nodos, K, loads, rates)
        for m in range(casos.M):
            cota[(_etiqueta_k(K), m)] = float(pi[m])
    if verbose:
        print(f"  {modo}: UB-PI en {time.perf_counter() - t0:.1f} s", flush=True)
    return cota


def _evaluar(ctx, casos, modo, cache_path, firma, verbose) -> pd.DataFrame:
    """Baselines y UB-PI (con caché) + DRL y comparación B, sobre los mismos episodios."""
    def baselines(faltan=None, cota_previa=None):
        if faltan is None:
            cota = _cotas(ctx, casos, modo, verbose)
        else:
            cota = {(k, m): float(v) for (k, m), v in cota_previa.items()}
        res = _correr_baselines(ctx, casos, modo, faltan, verbose)
        filas = []
        for nombre, r in res.items():
            filas += [f for f in casos.filas(modo, nombre, r, cota)
                      if faltan is None or (None if f["K"] == "libre" else int(f["K"]), nombre) in faltan]
        return pd.DataFrame(filas)

    def completar(df):
        return _completar_con_baselines(df, casos.k_levels, baselines, ctx.baselines)

    base = _filas_con_cache(cache_path, firma, baselines, verbose, completar)
    cota = {(k, m): float(v) for (k, m), v in base.groupby(["K", "mundo"])["ub_pi"].first().items()}
    filas = []
    for nombre in ("DRL",) + _DRL_REGLAS:
        t0 = time.perf_counter()
        res = ejecutar_lote(ctx.politica(nombre), ctx.env, casos.lote_mundos, casos.lote_nodos, casos.lote_k, ctx.dist)
        filas += casos.filas(modo, nombre, res, cota)
        if verbose:
            print(f"  {modo}: {nombre} en {time.perf_counter() - t0:.1f} s", flush=True)
    df = _combinar(base, pd.DataFrame(filas), casos.k_levels)
    df.attrs["dp_no_tratable"] = ctx.tabla is None
    _avisar_si_supera_cota(df)
    return df


def evaluar_mundos_simulados(agent, market, tabla, time_matrix, distance_arr, diesel_arr, num_nodes,
                             avail_expected_arr, reward_global_p95, periodo="test",
                             M=MARKET_M_EVAL_WORLDS, k_levels=(K_MAX, None), verbose=True,
                             cache_path=None, semilla_eval=MARKET_SEED_EVAL) -> pd.DataFrame:
    """DRL, comparación B y B1, B2, B2P, B3 sobre los mismos M mundos simulados, por nivel de K.

    Los mundos salen el día `periodo` ("test" = octubre, lo que se reporta). El DRL y las tablas de los baselines
    se construyeron solo con días de entrenamiento. Los mismos mundos se usan en todos los niveles de K y todos los
    métodos. Devuelve un DataFrame largo: una fila por (K, mundo, método) con el margen en USD, la factibilidad, las
    métricas de cotización y operativas y la cota ub_pi del mundo.

    semilla_eval : semilla del diseño y del sorteo de los M mundos (MARKET_SEED_EVAL = conjunto de desarrollo;
        MARKET_SEED_EVAL_FINAL = conjunto B, test final). Usar un cache_path distinto por semilla.
    cache_path : B1, B2, B2P, B3 y UB-PI no dependen del DRL, así que con una ruta se calculan una
        sola vez y se reusan entre corridas (mientras no cambie el proceso, la tabla V ni el diseño).
    """
    ctx = ContextoEval(agent, market, tabla, time_matrix, distance_arr, diesel_arr, num_nodes,
                       avail_expected_arr, reward_global_p95)
    dis = disenar_mundos(num_nodes, M, market, periodo=periodo, semilla=semilla_eval)
    mundos = mundos_eval(market, dis["start_day"], dis["dia_max"], semilla_eval)
    casos = _Casos(dis["nodo"], dis["start_day"], dis["dow"], mundos, k_levels)
    firma = _firma_cache(ctx, "simulado", num_nodes, M, semilla_eval, tuple(k_levels), market.periodo(periodo))
    return _evaluar(ctx, casos, "simulado", cache_path, firma, verbose)


def _avisar_si_supera_cota(df: pd.DataFrame, tol: float = 1e-6) -> None:
    """UB-PI debe ser ≥ el margen de toda política con ruta factible, mundo a mundo."""
    ok = ~df["violacion"] & df["ub_pi"].notna()
    malos = df[ok & (df["margen"] > df["ub_pi"] + tol)]
    if len(malos):
        print(f"  ⚠ {len(malos)} filas con margen > UB-PI (p. ej. {malos.iloc[0]['metodo']} "
              f"mundo {malos.iloc[0]['mundo']} K={malos.iloc[0]['K']}): revisar UB-PI frente al entorno.")


# ── Resumen ───────────────────────────────────────────────────

def resumen_metodos(df: pd.DataFrame) -> pd.DataFrame:
    """Margen medio, gap contra UB-PI y métricas operativas, por (modo, K, método).

    Cada fila trae grupo, destino y precio (qué decide el método, ver DECISION) y dif_vs_drl_usd: margen del
    método − margen del DRL en los mismos mundos (media pareada; negativo = por debajo del DRL).
    Métricas de cotización (§8): tasa_aceptacion_pct (cargas ganadas / cotizaciones, sobre el total del
    grupo), cotizaciones_por_arco, horas_rechazo (media por ruta), nivel_medio (índice 0–8 del nivel
    cotizado) y u_medio (su percentil u, media por cotización), usd_mi_sobre_mediana (USD/mi cobrado − mediana de la lane, media por carga ganada) y
    dinero_en_la_mesa (media por ruta; solo mundos simulados). Movimientos sin carga: esperas_por_ruta,
    reposiciones_por_ruta (vacíos a un nodo que no es casa) y cargas_bajo_be_pct (% de las cargas ganadas
    cobradas por debajo del costo de la lane) con usd_bajo_be (lo perdido en ellas, media por ruta).
    El gap compara promedios sobre los mismos mundos: (UB-PI − método) / |UB-PI| · 100.
    margen_medio va con su IC del 95 % por bootstrap sobre mundos (BOOTSTRAP_B remuestreos,
    semilla fija): margen_ic95_low / margen_ic95_high. Cada grupo (modo, K, método) usa su
    propio Generator con la semilla fija, así que el IC no depende del orden de las filas.
    Se añade la fila de UB-PI (cota), separada de los métodos. Si df.attrs["dp_no_tratable"], B3 aparece como fila
    con nota "no tratable".
    """
    filas = []
    for (modo, K), g in df.groupby(["modo", "K"], sort=False):
        mundos = g.drop_duplicates("mundo")
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
                "gap_ub_pi_pct": gap_pct(gm["ub_pi"], gm["margen"]),
                "violacion_pct": 100.0 * gm["violacion"].mean(),
                "millas_cargadas_pct": 100.0 * gm["frac_millas_cargadas"].mean(),
                "millas_totales_medias": gm["millas_totales"].mean(),
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
                "esperas_por_ruta": gm["n_esperas"].mean(),
                "reposiciones_por_ruta": gm["n_reposiciones"].mean(),
                "cargas_bajo_be_pct": 100.0 * gm["n_bajo_be"].sum() / n_gan if n_gan else float("nan"),
                "usd_bajo_be": gm["usd_bajo_be"].mean(),
                "nota": None,
            })
        if df.attrs.get("dp_no_tratable"):
            filas.append({"modo": modo, "K": K, "metodo": "B3", **_describir("B3"), "n_mundos": 0,
                          "nota": NOTA_NO_TRATABLE})
        if mundos["ub_pi"].notna().any():
            ic = ic_bootstrap_media(mundos["ub_pi"])
            filas.append({"modo": modo, "K": K, "metodo": "UB-PI", **_describir("UB-PI"),
                          "n_mundos": int(mundos["ub_pi"].notna().sum()),
                          "margen_medio": mundos["ub_pi"].mean(),
                          "margen_ic95_low": ic[0], "margen_ic95_high": ic[1],
                          "dif_vs_drl_usd": dif_vs_drl(mundos.set_index("mundo")["ub_pi"]),
                          "nota": None})
    return pd.DataFrame(filas)


# Columnas del resumen de consola: (columna, encabezado, formato).
_COLS_CONSOLA = (
    ("metodo", "método", "{}"), ("precio", "precio", "{}"),
    ("margen_medio", "margen USD", "{:,.0f}"), ("dif_vs_drl_usd", "Δ vs DRL", "{:+,.0f}"),
    ("gap_ub_pi_pct", "gap UB-PI %", "{:.1f}"), ("tasa_aceptacion_pct", "acept. %", "{:.0f}"),
    ("u_medio", "u cotizado", "{:.2f}"), ("usd_mi_sobre_mediana", "USD/mi vs mediana", "{:+.3f}"),
    ("esperas_por_ruta", "esperas", "{:.2f}"), ("reposiciones_por_ruta", "reposic.", "{:.2f}"),
    ("cargas_bajo_be_pct", "bajo BE %", "{:.0f}"), ("violacion_pct", "viol. %", "{:.1f}"),
)


def imprimir_resumen(df: pd.DataFrame, titulo: str) -> None:
    """Resumen compacto para la consola, por K y por grupo (DRL y comparación B, baselines, cotas).

    El detalle completo (IC, millas cargadas, dinero en la mesa, …) va al Excel."""
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
    if (res["nota"] == NOTA_NO_TRATABLE).any():
        print(f"\n  B3: {NOTA_NO_TRATABLE}")


# ── Output ────────────────────────────────────────────────────
def save_results(sim_df, resumen_rows, output_path):
    from excel_report import build_formatted_excel
    build_formatted_excel(sim_df, resumen_rows, output_path)
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
