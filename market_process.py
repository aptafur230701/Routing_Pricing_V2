"""
market_process.py
=================
Generador de realizaciones diarias (metodologia_realizaciones_mercado_reefer.md) + índice de
mercado por nodo (nivel lento + rápido AR(1), estimado en fit_market_process.py).

Cuando el agente llega al nodo i el día real `dia`, el proceso sortea para cada lane (i, j) que
sale del nodo:
  1. cuántas cargas hay (Paso 4b-1): NegBin(m_{dia,i,j}, r_h) si m ≥ 1, Poisson(m) si m < 1.
     m es la media móvil de 6 vecinos ajustada por día de semana (`fit_market_process.estimar_m`);
     r_h es la dispersión del grupo de volumen h de la lane.
  2. la tarifa por milla (Paso 4b-2): un único precio por lane-día, sorteado por inversión de
     cuantiles [min, q1, q2, q3, p90, max] del día real (o del día más cercano con cargas de esa
     misma lane, si el día real tiene 0 cargas registradas), multiplicado por e^δ.
  3. δ (Paso 4a): un único corrimiento de nivel nacional por "mundo" (episodio), δ ~ N(0, s²),
     fijo durante todo el episodio — la única pieza de aleatorización por semilla del generador.

El índice de mercado por nodo (lento/rápido/ρ/f, usado por `routing_env.market_index` y por B2P) no
se simula: como los episodios recorren siempre días reales, el índice de cualquier día es el observado
(`tensores()["indice"]`, `indice_historico`). δ no entra al índice ni a las tablas de B2P.

El proceso NO depende de las acciones del agente. Modos de uso:

  · Entrenamiento y evaluación (en el dispositivo, la GPU en el servidor): `sortear_mundos(dia0, D, dia_max,
    generator)` sortea de una vez, para B episodios, las cargas de cada lane y día (Gamma-Poisson = NegBin, o
    Poisson si m < 1), los uniformes de los competidores de cada carga y δ (`Mundos`). `mundos_replay` hace lo
    mismo con las cargas reales de los días de prueba (solo se sortea el competidor). El entorno completa cada
    tablero con las tablas por día de `tensores()`.
  · Política que ignora el mercado (DP de UB-DP y B2): `sortear_cargas(nodos, días, rng)` con un día de
    ENTRENAMIENTO al azar del mismo día de la semana (sin anclar al día real del episodio, δ = 0): representa una
    política que conoce la distribución (con su estacionalidad semanal) pero ignora el estado del mercado.

Cotización conjunta (especificacion_fase0_ruteo_pricing.md §3): los precios de cada nivel son F⁻¹(u_k)·e^δ con
los cuantiles del día, m es el número efectivo de competidores del nodo de origen (LTR del día) y el percentil
del mejor competidor de cada carga es u_comp = 1 − (1 − U)^(1/m), con U el uniforme de esa carga en `Mundos`.
Los 6 cuantiles se ordenan al cargar (vienen desordenados en ~0.09 % de las lane-días con carga).

Toda la aleatoriedad sale de generadores propios (torch.Generator del dispositivo o numpy Generator), nunca
del estado global.
"""

import hashlib
import os
from dataclasses import dataclass

import numpy as np
import torch

from config import (
    TRAIN_DAYS, U_NIVELES, COMPETENCIA_KAPPA, COMPETENCIA_M_MAX, COTIZACIONES_MAX_POR_NODO, periodos,
)
from fit_market_process import (
    params_path, estimar_m, indice_observado, dia_de_semana as _dia_de_semana_fi,
)
from fit_market_process import main as _ajustar
from pricing import (
    interp_cuantiles as _interp_cuantiles,      # alias: validate_market_process lo importa con este nombre
    competidores_efectivos, indice_nivel, lane_cost_row, precios_niveles,
)
from problem_data import cargar_dias_mercado

_CLAVES_AJUSTE = ("f_w", "grupo_lane", "cortes_grupo", "r", "media_lane", "lento_escenarios",
                  "rho", "sigma", "indice_std", "frac_var_lento", "s_delta", "train_days")


def _ajuste_vigente(z) -> bool:
    """True si el .npz abierto `z` trae todas las claves del ajuste y se ajustó con los TRAIN_DAYS actuales."""
    return all(k in z.files for k in _CLAVES_AJUSTE) and int(z["train_days"]) == TRAIN_DAYS
_COLS_Q = ("min", "q1", "q2", "q3", "p90", "max")


@dataclass(frozen=True, eq=False)
class EstadisticasLane:
    """Media del generador por día de la semana y lane (entrada de las features de continuación del DRL).

    Todo sale de los días de ENTRENAMIENTO y de las mismas piezas que usa el generador (la media `m` de las
    cargas, la binomial negativa / Poisson, los cuantiles de la lane y el LTR del hub), así que es lo que el
    agente "sabe" del mercado promedio de cada día de la semana: la misma media que promedia la DP de B3
    (un día de entrenamiento al azar del mismo día de la semana), sin usar el día real ni el índice.

      L  (7, N, N)  cargas publicadas esperadas de la lane j→l (media de m)
      P1 (7, N, N)  P(al menos una carga) con la distribución del generador
      G  (7, N, N)  margen en USD de la lane con una carga al precio mediano: p50 · dist − costo
      M  (7, N)     competidores efectivos m medios del hub de origen (con el κ de entrenamiento)
    """
    L:  np.ndarray
    P1: np.ndarray
    G:  np.ndarray
    M:  np.ndarray

    def a(self, device) -> "EstadisticasLane":
        """Las mismas estadísticas como tensores float32 en `device` (las features se arman en el dispositivo)."""
        return EstadisticasLane(*(torch.as_tensor(getattr(self, c), dtype=torch.float32, device=device)
                                  for c in ("L", "P1", "G", "M")))


@dataclass
class Mundos:
    """Los mundos de B episodios, ya sorteados y en el dispositivo (especificación §3).

    El tablero que ve el camión al llegar al nodo i el día absoluto dia0[b] + d (d < D, recortado a dia_max) es la
    fila [b, d, i] de cada tensor. Todo lo que no depende del episodio (precios por nivel, cuantiles, m) lo pone el
    entorno a partir de `MarketProcess.tensores`. Como el mundo está sorteado de antemano, el tablero no depende de
    las decisiones del agente y todos los métodos evaluados sobre el mismo `Mundos` enfrentan lo mismo.

      loads (B, D, N, N) int32    cargas publicadas de la lane i→j
      u     (B, D, N, N, C) f32   un uniforme por carga cotizable; el competidor es u_comp = 1 − (1 − u)^(1/m)
      delta (B,) f64              corrimiento nacional de precios del episodio (0 en replay)
      dia0  (B,) int64            día absoluto de salida
      dia_max                     último día absoluto con datos: los días de llegada se recortan ahí
    """
    loads: torch.Tensor
    u: torch.Tensor
    delta: torch.Tensor
    dia0: torch.Tensor
    dia_max: int

    @property
    def B(self) -> int:
        return int(self.loads.shape[0])

    def tomar(self, idx) -> "Mundos":
        """Subconjunto (o repetición) de episodios: idx es un tensor/array de índices."""
        idx = torch.as_tensor(idx, dtype=torch.long, device=self.loads.device)
        return Mundos(self.loads[idx], self.u[idx], self.delta[idx], self.dia0[idx], self.dia_max)


class MarketProcess:
    """Generador de realizaciones diarias + índice de mercado por nodo, estimados del histórico.

    Parámetros
    ----------
    num_nodes : se usan los primeros `num_nodes` nodos (igual que load_matrices). Los parámetros se
                estiman con los 194 y se recortan aquí.
    path      : ruta del .npz del ajuste (por defecto datos/market_params.npz).
    """

    def __init__(self, num_nodes: int, path: str = None):
        path = path or params_path()
        if not os.path.exists(path):
            raise FileNotFoundError(f"{path} no existe. Corre `python fit_market_process.py` primero.")
        with np.load(path, allow_pickle=False) as z:
            falta = [k for k in _CLAVES_AJUSTE if k not in z.files]
            if falta:
                raise ValueError(f"{path} es de un ajuste anterior (faltan {falta}). "
                                 "Corre `python fit_market_process.py`.")
            if not _ajuste_vigente(z):
                raise ValueError(f"{path} se ajustó con {int(z['train_days'])} días de entrenamiento y TRAIN_DAYS = "
                                 f"{TRAIN_DAYS}. Corre `python fit_market_process.py`.")
            n_total = z["grupo_lane"].shape[0]
            if num_nodes > n_total:
                raise ValueError(f"num_nodes={num_nodes} excede los {n_total} nodos ajustados")
            n = num_nodes
            self.f_w = z["f_w"].astype(np.float64)
            self.r = z["r"].astype(np.float64)
            grupo_lane = z["grupo_lane"][:n, :n].astype(np.int64)
            media_lane_full = z["media_lane"].astype(np.float64)              # (194, 194) — para el índice
            universo_indice_full = z["universo_indice"].astype(bool)          # (194, 194) — ídem
            self.media_lane = media_lane_full[:n, :n]
            self.universo_indice = universo_indice_full[:n, :n]
            self.lento_escenarios = z["lento_escenarios"][:, :n].astype(np.float64)
            self.rho = float(z["rho"])
            self.sigma = z["sigma"][:n].astype(np.float64)
            self.indice_std = float(z["indice_std"])
            self.frac_var_lento = float(z["frac_var_lento"])
            self.s_delta = float(z["s_delta"])
            self.fecha_inicio = str(z["fecha_inicio"])
            self._params_blob = {k: np.ascontiguousarray(z[k]) for k in _CLAVES_AJUSTE}

        self.num_nodes = n
        n_grupos = len(self.r)
        self._grupo_lane = grupo_lane                                     # (N, N) int, -1 en la diagonal
        self._r_lane = self.r[np.clip(grupo_lane, 0, n_grupos - 1)]       # (N, N)

        # ── Stacks reales completos (jun-oct 2026), recortados a num_nodes ──
        dias = cargar_dias_mercado()
        self.fechas = dias["fechas"]
        self.num_dias_total = len(self.fechas)
        dow = (self.fechas.astype("datetime64[D]").astype(np.int64) + 3) % 7      # lunes = 0
        count_loads_full = dias["count_loads"]
        self._con_cargas = (count_loads_full[:, :n, :n] > 0)
        self._dia_cercano = dias["dia_con_cargas_cercano"][:, :n, :n]
        # Los 6 cuantiles se ordenan de menor a mayor: en ~0.09 % de las lane-días con carga del CSV vienen
        # desordenados (p. ej. q3 > p90), y sin ordenar un nivel de precio más alto podría salir más barato.
        q_orden = np.sort(np.stack([dias[c][:, :n, :n] for c in _COLS_Q], axis=-1), axis=-1).astype(np.float32)
        self._q = {c: np.ascontiguousarray(q_orden[..., k]) for k, c in enumerate(_COLS_Q)}
        del q_orden

        # LTR por hub de origen y día (la columna es constante en la fila; se toma la diagonal), y LTR_REF:
        # mediana en train sobre los 194 hubs, para que no dependa de num_nodes.
        ltr_full = dias["ltr"][:, np.arange(dias["ltr"].shape[1]), np.arange(dias["ltr"].shape[1])]   # (D, 194)
        ltr_train = ltr_full[:TRAIN_DAYS]
        self.ltr_ref = float(np.median(ltr_train[ltr_train > 0]))
        self.kappa = COMPETENCIA_KAPPA          # validate_pricing.py lo cambia para la sensibilidad a κ
        self.ltr_hub = ltr_full[:, :n].astype(np.float64)                     # (D, N)
        self.mean_rate = dias["mean"][:, :n, :n].astype(np.float64)    # (D, N, N) — tarifa media diaria
        self.count_loads = count_loads_full[:, :n, :n]                 # (D, N, N) — cargas publicadas

        # Media del generador de cada lane-día, calculada dentro de cada período (entrenamiento, validación, prueba):
        # la ventana de ±3 días no cruza de un período a otro.
        self._m_stack = estimar_m(self.count_loads, dow, self.f_w, por_periodo=True)

        # ── Índice de mercado observado en TODOS los días reales (train y prueba) ──
        # Se calcula a escala completa (194 nodos): el nivel nacional que se resta (Fase 3) es el
        # promedio sobre TODOS los nodos, no solo los `num_nodes` de esta corrida — si se calculara
        # ya recortado, el nivel nacional (y por tanto el índice de cada nodo) cambiaría con
        # `num_nodes`. Solo el resultado final se recorta.
        mean_rate_full = dias["mean"].astype(np.float64)                       # (D, 194, 194)
        self._indice_full = indice_observado(
            mean_rate_full, media_lane_full, universo_indice_full,
        )[:, :n]                                                              # (D, N)

        self._cols = np.arange(n)

        # Días de train por día de la semana (el modo sin memoria de la DP y B2).
        self.num_train_days = min(TRAIN_DAYS, self.num_dias_total)
        dow_train = self.dia_semana(np.arange(self.num_train_days))
        self._dias_train_por_dow = [np.flatnonzero(dow_train == k) for k in range(7)]

    # ── utilidades ───────────────────────────────────────────────────────────

    def periodo(self, nombre: str) -> tuple:
        """(día inicial, día final exclusivo) del período "train", "val" o "test" (config.periodos)."""
        ini, fin = periodos(self.num_dias_total)[nombre]
        if fin <= ini:
            raise ValueError(f"el período {nombre!r} no tiene días en el histórico ({self.num_dias_total} días)")
        return ini, fin

    def dia_semana(self, dias) -> np.ndarray:
        """Día de la semana (lunes = 0) de los índices de día `dias`."""
        return _dia_de_semana_fi(dias, self.fecha_inicio)

    def firma_params(self) -> str:
        """Huella de todo lo que determina el proceso (ajuste + datos reales usados para generar).

        La usan la tabla V de UB-DP y el caché de la evaluación para recalcularse solos cuando
        cambian los datos o el generador."""
        h = hashlib.sha256()
        for k in _CLAVES_AJUSTE:
            h.update(self._params_blob[k].tobytes())
        h.update(self.count_loads.tobytes())
        h.update(self.mean_rate.tobytes())
        for c in _COLS_Q:
            h.update(self._q[c].tobytes())
        h.update(self.ltr_hub.tobytes())
        h.update(repr((self.num_nodes, self.num_dias_total, self.fecha_inicio,
                       tuple(U_NIVELES), self.kappa, COMPETENCIA_M_MAX, self.ltr_ref)).encode())
        return h.hexdigest()

    def competidores(self, nodos, dias) -> np.ndarray:
        """(B,) competidores efectivos m del nodo de origen `nodos[b]` el día `dias[b]` (especificación §3.3)."""
        ltr = self.ltr_hub[np.asarray(dias, dtype=np.int64), np.asarray(nodos, dtype=np.int64)]
        return competidores_efectivos(ltr, self.ltr_ref, self.kappa)

    def cuantiles_dia(self, dia: int) -> np.ndarray:
        """(N, N, 6) cuantiles de precio de todas las lanes el día real `dia`, con la misma regla que los
        tableros (`_dia_precio`: el propio día si la lane tuvo cargas; si no, el día más cercano con cargas)."""
        nodos = np.arange(self.num_nodes)
        dia_p = self._dia_precio(int(dia), nodos)                                     # (N, N)
        return np.stack([self._q[c][dia_p, nodos[:, None], self._cols[None, :]] for c in _COLS_Q], axis=-1)

    def estadisticas_dow(self, distance_arr, diesel_arr) -> EstadisticasLane:
        """Estadísticas por día de la semana y lane de los días de ENTRENAMIENTO (ver `EstadisticasLane`).

        Se calcula una vez por número de nodos. M usa COMPETENCIA_KAPPA de config (el κ con el que se entrena), no
        `self.kappa`: el agente no conoce el κ del mundo, así que la sensibilidad a κ no cambia lo que sabe de antemano.
        """
        N = self.num_nodes
        dist = np.asarray(distance_arr, dtype=np.float64)
        costo = lane_cost_row(dist, np.asarray(diesel_arr, dtype=np.float64))
        u50 = indice_nivel(0.5)
        L, P1, G = (np.zeros((7, N, N)) for _ in range(3))
        M = np.zeros((7, N))
        r = self._r_lane                                                          # (N, N) dispersión de la binomial negativa
        for k in range(7):
            dias = self._dias_train_por_dow[k]
            m = self._m_stack[dias]                                               # (D_k, N, N) media del generador
            L[k] = m.mean(axis=0)
            m_pos = np.maximum(m, 1e-12)
            p0 = np.where(m < 1.0, np.exp(-m_pos), np.exp(-r * np.log1p(m_pos / r)))   # P(0 cargas): Poisson o NegBin
            P1[k] = (1.0 - p0).mean(axis=0)
            q_medio = np.mean([self.cuantiles_dia(int(d)) for d in dias], axis=0)       # (N, N, 6)
            G[k] = precios_niveles(q_medio)[..., u50] * dist - costo
            M[k] = competidores_efectivos(self.ltr_hub[dias], self.ltr_ref, COMPETENCIA_KAPPA).mean(axis=0)
        return EstadisticasLane(L=L, P1=P1, G=G, M=M)

    # ── En el dispositivo: tablas por día y mundos sorteados (entorno de entrenamiento y evaluación) ──

    def tensores(self, device) -> dict:
        """Tablas por día absoluto que el entorno lee en cada llegada, en `device` (se calculan una vez por dispositivo
        y κ):
          precios (D, N, N, K) f64  USD/milla de cada nivel con δ = 0 (cuantiles del propio día o del más cercano con
                                    cargas, `_dia_precio`); el entorno multiplica por e^δ
          cuant   (D, N, N, 6) f64  esos mismos cuantiles (precio del competidor)
          m       (D, N) f64        competidores efectivos del hub de origen con el κ actual (`self.kappa`)
          m_carg  (D, N, N) f32     media del generador de cargas; r (N, N) f32 su dispersión
          loads_reales (D, N, N)    cargas publicadas reales (replay); indice (D, N) f64; ltr (D, N) f32; dow (D,)
        """
        clave = (str(torch.device(device)), float(self.kappa))
        cache = getattr(self, "_tensores", {})
        if clave in cache:
            return cache[clave]
        D, N = self.num_dias_total, self.num_nodes
        dd = np.arange(D)[:, None, None]
        dia_p = np.where(self._con_cargas, dd, np.where(self._dia_cercano >= 0, self._dia_cercano, dd))   # (D, N, N)
        ii, jj = np.arange(N)[None, :, None], np.arange(N)[None, None, :]
        cuant = np.stack([self._q[c][dia_p, ii, jj] for c in _COLS_Q], axis=-1)                         # (D, N, N, 6)
        dev = torch.device(device)
        t = dict(
            precios=torch.as_tensor(precios_niveles(cuant), dtype=torch.float64, device=dev),
            cuant=torch.as_tensor(cuant, dtype=torch.float64, device=dev),
            m=torch.as_tensor(competidores_efectivos(self.ltr_hub, self.ltr_ref, self.kappa), dtype=torch.float64, device=dev),
            m_carg=torch.as_tensor(self._m_stack, dtype=torch.float32, device=dev),
            r=torch.as_tensor(self._r_lane, dtype=torch.float32, device=dev),
            loads_reales=torch.as_tensor(self.count_loads, dtype=torch.int32, device=dev),
            indice=torch.as_tensor(self._indice_full, dtype=torch.float64, device=dev),
            ltr=torch.as_tensor(self.ltr_hub, dtype=torch.float32, device=dev),
            dow=torch.as_tensor(self.dia_semana(np.arange(D)), dtype=torch.long, device=dev),
        )
        cache[clave] = t
        self._tensores = cache
        return t

    def sortear_mundos(self, dia0, num_dias: int, dia_max: int, generator: torch.Generator) -> Mundos:
        """B mundos simulados en el dispositivo de `generator` (Paso 4 de la metodología): para cada día d < num_dias
        y cada lane, cargas ~ NegBin(m, r) (Poisson(m) si m < 1, como `sortear_cargas`), C uniformes para los
        competidores de las cargas cotizables y un δ ~ N(0, s_δ²) por mundo. dia0 (B,) días absolutos de salida."""
        dev = generator.device
        dia0 = torch.as_tensor(np.asarray(dia0), dtype=torch.long, device=dev)
        B, N, C = len(dia0), self.num_nodes, COTIZACIONES_MAX_POR_NODO
        t = self.tensores(dev)
        loads = torch.empty((B, num_dias, N, N), dtype=torch.int32, device=dev)
        for d in range(num_dias):
            m = t["m_carg"][(dia0 + d).clamp(max=dia_max)].clamp_min(1e-12)                # (B, N, N)
            r = t["r"].expand(B, N, N).contiguous()
            gamma = torch._standard_gamma(r, generator=generator) * (m / r)               # mezcla Gamma-Poisson = NegBin
            loads[:, d] = torch.poisson(torch.where(m < 1.0, m, gamma), generator=generator).to(torch.int32)
        u = torch.rand((B, num_dias, N, N, C), generator=generator, device=dev)
        return Mundos(loads=loads, u=u, delta=self.sortear_deltas(B, generator), dia0=dia0, dia_max=int(dia_max))

    def sortear_deltas(self, n: int, generator: torch.Generator) -> torch.Tensor:
        """(n,) corrimientos de nivel nacional δ ~ N(0, s_δ²) (Paso 4a), uno por mundo, en float64."""
        return torch.randn(n, generator=generator, device=generator.device, dtype=torch.float64) * self.s_delta

    def mundos_replay(self, dia0, num_dias: int, dia_max: int, generator: torch.Generator) -> Mundos:
        """Replay de días reales (especificación §3.4): cargas reales del día, δ = 0; solo el competidor se sortea."""
        dev = generator.device
        dia0 = torch.as_tensor(np.asarray(dia0), dtype=torch.long, device=dev)
        B, N, C = len(dia0), self.num_nodes, COTIZACIONES_MAX_POR_NODO
        dias = (dia0[:, None] + torch.arange(num_dias, device=dev)[None, :]).clamp(max=dia_max)   # (B, D)
        loads = self.tensores(dev)["loads_reales"][dias]                                          # (B, D, N, N)
        u = torch.rand((B, num_dias, N, N, C), generator=generator, device=dev)
        return Mundos(loads=loads, u=u, delta=torch.zeros(B, dtype=torch.float64, device=dev), dia0=dia0,
                      dia_max=int(dia_max))

    def dias_train_por_dow(self, dow: int) -> np.ndarray:
        """Días de entrenamiento con ese día de la semana (de entre ellos sortean la DP de UB-DP y B2)."""
        return self._dias_train_por_dow[int(dow)]

    # ── Índice de mercado (lookup directo del día real) ───────────────────────

    def indice_historico(self, dia_ini: int = 0, dia_fin: int = None) -> np.ndarray:
        """(D, N) índice observado de los días reales [dia_ini, dia_fin)."""
        return self._indice_full[dia_ini:dia_fin]

    def _dia_precio(self, dia: int, nodos) -> np.ndarray:
        """(len(nodos), N) día del que se toman los cuantiles para cada lane saliente de `nodos` el
        día `dia`: el propio día si tuvo cargas, si no el día más cercano con cargas de esa lane (o
        el propio día si nunca tuvo cargas en toda la ventana)."""
        valido = self._con_cargas[dia, nodos]                   # (B, N)
        cercano = self._dia_cercano[dia, nodos]                  # (B, N)
        return np.where(valido, dia, np.where(cercano >= 0, cercano, dia))

    def sortear_cargas(self, nodos: np.ndarray, dias: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Cargas (B, N) de las lanes que salen de nodos[b] el día dias[b] (Poisson o binomial negativa).
        Es el primer sorteo de `_sortear_detalle` (mismo consumo de `rng`); la DP de UB-DP lo usa solo."""
        nodos = np.asarray(nodos, dtype=np.int64)
        dias = np.asarray(dias, dtype=np.int64)
        m = self._m_stack[dias, nodos]                            # (B, N)
        r = self._r_lane[nodos]                                   # (B, N)
        usar_poisson = m < 1.0
        m_pos = np.maximum(m, 1e-12)
        return np.where(
            usar_poisson,
            rng.poisson(m_pos),
            rng.negative_binomial(r, r / (r + m_pos)),
        ).astype(np.int64)


def asegurar_parametros() -> None:
    """Corre el ajuste (fit_market_process) si datos/market_params.npz no existe o es de un ajuste
    anterior (sin alguno de los parámetros que necesita el generador nuevo)."""
    if os.path.exists(params_path()):
        with np.load(params_path()) as z:
            if _ajuste_vigente(z):
                return
        print("market_params.npz no corresponde al ajuste actual (claves o TRAIN_DAYS): reajustando...")
    else:
        print("market_params.npz no existe: corriendo el ajuste del proceso exógeno...")
    _ajustar()
