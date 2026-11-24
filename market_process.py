"""
market_process.py
=================
Generador de realizaciones diarias (metodologia_realizaciones_mercado_reefer.md) + índice de
mercado por nodo (proceso_exogeno_cargas_tarifas.md, Fase 3 — conservado, repuntado a los datos
nuevos; ver decisión del plan).

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

El índice de mercado por nodo (lento/rápido/ρ/f, usado por `routing_env.market_index` y por B2P) ya
NO se simula con AR(1): como los episodios recorren siempre días reales, el índice de cualquier día
es un lookup directo (`indice_mundo` conserva la firma `(semilla, dia)` por compatibilidad con los
~30 llamadores existentes, pero ignora `semilla`). δ no entra al índice ni a las tablas de B2P.

El proceso NO depende de las acciones del agente. Tres modos de uso, igual que antes:

  · Entrenamiento : `sample_arrivals_por_mundo(nodos, días, semillas)` — cada sorteo es función pura
    de (semilla del "mundo" = episodio, nodo, día), así que no depende del resto del lote ni de las
    decisiones del agente.
  · Evaluación / baselines : `sample_world(semilla, num_días, día_inicial)` genera un mundo completo
    [D, N, N] con esa misma semilla.
  · Política que ignora el mercado (DP de UB-DP y B2) : `sample_arrivals(nodos, días, rng)` sortea,
    en cada llegada, un día de ENTRENAMIENTO al azar del mismo día de la semana (sin memoria, sin
    anclar al día real del episodio, δ = 0) y le aplica el mismo generador — representa una
    política que conoce la distribución (con su estacionalidad semanal) pero ignora el estado del
    mercado y el día real concreto.

Cotización conjunta (especificacion_fase0_ruteo_pricing.md §3): `tablero_sin_memoria`,
`tablero_por_mundo`, `mundo_cotizacion` y `tablero_real` (replay) devuelven un `Tablero` con los
precios de cada nivel (F⁻¹(u_k)·e^δ), el número efectivo de competidores m del nodo de origen (LTR
del día) y el percentil del mejor competidor u_comp = 1 − (1 − U)^(1/m), donde U es el MISMO uniforme
de la tarifa sorteada: las cargas y la tarifa no cambian y, con m = 1, el competidor cotiza la tarifa
del modelo anterior. Los 6 cuantiles se ordenan al cargar (vienen desordenados en ~0.09 % de las
lane-días con carga).

Toda la aleatoriedad sale de numpy Generators propios (nunca del estado global).
"""

import hashlib
import os
from collections import OrderedDict
from dataclasses import dataclass

import numpy as np

from config import MARKET_N_GRUPOS, TRAIN_DAYS, U_NIVELES, COMPETENCIA_KAPPA, COMPETENCIA_M_MAX
from fit_market_process import (
    params_path, estimar_m, indice_observado, dia_de_semana as _dia_de_semana_fi,
)
from fit_market_process import main as _ajustar
from pricing import (
    interp_cuantiles as _interp_cuantiles,      # alias: validate_market_process lo importa con este nombre
    PCT_CUANTILES as _PCT,
    competidores_efectivos, percentil_competidor, precios_niveles,
)
from problem_data import cargar_dias_mercado

# Etiqueta de la clave del generador de δ ([semilla, _TAG_DELTA]); distinta de la de las filas
# [semilla, nodo, día] para no coincidir nunca con ellas.
_TAG_DELTA = 1_000_000_007
_CACHE_DELTAS_MAX = 200_000          # tope de semillas de δ cacheadas (LRU)
_TAG_REPLAY = 1_000_000_009         # clave del competidor en replay ([semilla, nodo, día, _TAG_REPLAY])
_CLAVES_AJUSTE = ("f_w", "grupo_lane", "cortes_grupo", "r", "media_lane", "lento_escenarios",
                  "rho", "sigma", "indice_std", "frac_var_lento", "s_delta")
_COLS_Q = ("min", "q1", "q2", "q3", "p90", "max")


@dataclass
class Tablero:
    """Lo que trae una llegada a un nodo para la cotización conjunta (especificación §3).

    Formas para un lote de B llegadas con N lanes salientes y K niveles (en `sample_world`, B = D días
    y la segunda dimensión es el nodo de origen: loads (D, N, N), precios (D, N, N, K), m (D, N)):
      loads             (B, N) int64    cargas por lane
      precios           (B, N, K) f64   USD/milla de cada nivel: F⁻¹(u_k) · e^δ
      u_comp            (B, N) f64      percentil del mejor competidor (el agente no lo ve)
      precio_competidor (B, N) f64      F⁻¹(u_comp) · e^δ
      m                 (B,) f64        competidores efectivos del nodo de origen ese día
      tarifa            (B, N) f32      la tarifa sorteada del modelo anterior (= precio del competidor si m = 1)
    """
    loads: np.ndarray
    precios: np.ndarray
    u_comp: np.ndarray
    precio_competidor: np.ndarray
    m: np.ndarray
    tarifa: np.ndarray

    @staticmethod
    def apilar(tableros) -> "Tablero":
        return Tablero(*(np.stack([getattr(t, c) for t in tableros])
                         for c in ("loads", "precios", "u_comp", "precio_competidor", "m", "tarifa")))


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

        self._m_stack = estimar_m(self.count_loads, dow, self.f_w)

        # ── Índice de mercado observado en TODOS los días reales (train y prueba) ──
        # Se calcula a escala completa (194 nodos): el nivel nacional que se resta (Fase 3) es el
        # promedio sobre TODOS los nodos, no solo los `num_nodes` de esta corrida — si se calculara
        # ya recortado, el nivel nacional (y por tanto el índice de cada nodo) cambiaría con
        # `num_nodes`. Solo el resultado final se recorta.
        mean_rate_full = dias["mean"].astype(np.float64)                       # (D, 194, 194)
        self._indice_full = indice_observado(
            mean_rate_full, media_lane_full, universo_indice_full,
        )[:, :n]                                                              # (D, N)

        self._deltas = OrderedDict()           # semilla → δ (caché LRU)
        self._cols = np.arange(n)

        # Días de train por día de la semana (para `sample_arrivals`, el modo sin memoria).
        self.num_train_days = min(TRAIN_DAYS, self.num_dias_total)
        dow_train = self.dia_semana(np.arange(self.num_train_days))
        self._dias_train_por_dow = [np.flatnonzero(dow_train == k) for k in range(7)]

    # ── utilidades ───────────────────────────────────────────────────────────

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

    def dias_train_por_dow(self, dow: int) -> np.ndarray:
        """Días de entrenamiento con ese día de la semana (de entre ellos sortea `tablero_sin_memoria`)."""
        return self._dias_train_por_dow[int(dow)]

    # ── Índice de mercado (lookup directo del día real) ───────────────────────

    def indice_mundo(self, semilla: int, dia: int) -> np.ndarray:
        """(N,) índice de mercado de todos los nodos el día real `dia`.

        `semilla` se conserva por compatibilidad de firma con los llamadores existentes
        (routing_env.py, B2P, tests) pero se ignora: el índice ya no se simula por "mundo", es un
        dato observado del día real (los episodios siempre recorren días reales)."""
        dia = int(dia)
        if not 0 <= dia < self.num_dias_total:
            raise ValueError(f"día {dia} fuera de rango (0 … {self.num_dias_total - 1})")
        return self._indice_full[dia]

    def indice_mundo_lote(self, semillas, dias) -> np.ndarray:
        """(B, N) índice de todos los nodos para cada día (ver `indice_mundo`; `semillas` se ignora)."""
        return self._indice_full[np.asarray(dias, dtype=np.int64)]

    def indice_historico(self, dia_ini: int = 0, dia_fin: int = None) -> np.ndarray:
        """(D, N) índice observado de los días reales [dia_ini, dia_fin)."""
        return self._indice_full[dia_ini:dia_fin]

    # ── δ: corrimiento de nivel nacional por episodio (Paso 4a) ───────────────

    def _delta(self, semilla: int) -> float:
        semilla = int(semilla)
        hit = self._deltas.get(semilla)
        if hit is not None:
            self._deltas.move_to_end(semilla)
            return hit
        rng = np.random.Generator(np.random.PCG64([semilla, _TAG_DELTA]))
        d = float(rng.normal(0.0, self.s_delta))
        self._deltas[semilla] = d
        if len(self._deltas) > _CACHE_DELTAS_MAX:
            self._deltas.popitem(last=False)
        return d

    # ── Llegada a un nodo (Paso 4b) ────────────────────────────────────────────

    def _dia_precio(self, dia: int, nodos) -> np.ndarray:
        """(len(nodos), N) día del que se toman los cuantiles para cada lane saliente de `nodos` el
        día `dia`: el propio día si tuvo cargas, si no el día más cercano con cargas de esa lane (o
        el propio día si nunca tuvo cargas en toda la ventana)."""
        valido = self._con_cargas[dia, nodos]                   # (B, N)
        cercano = self._dia_cercano[dia, nodos]                  # (B, N)
        return np.where(valido, dia, np.where(cercano >= 0, cercano, dia))

    def _sortear(self, nodos: np.ndarray, dias: np.ndarray, deltas: np.ndarray,
                 rng: np.random.Generator) -> tuple:
        """Llegadas (loads, rates) de forma (B, N) para (nodos[b], dias[b], deltas[b]), con un único
        generador `rng` (consume su estado; para sorteos independientes de las decisiones del agente
        usa un rng derivado por fila, ver `sample_arrivals_por_mundo`)."""
        loads, cuantiles, u = self._sortear_detalle(nodos, dias, rng)
        rate = _interp_cuantiles(u, cuantiles) * np.exp(deltas[:, None])
        return loads, rate.astype(np.float32)

    def _tablero(self, nodos: np.ndarray, dias: np.ndarray, deltas: np.ndarray,
                 rng: np.random.Generator) -> Tablero:
        """Como `_sortear`, con el mismo consumo de `rng` (las cargas y la tarifa salen idénticas), pero
        devuelve el tablero de la cotización: el uniforme U de la tarifa se reutiliza para el mejor
        competidor, u_comp = 1 − (1 − U)^(1/m), así que con m = 1 el competidor cotiza la tarifa de siempre."""
        loads, cuantiles, u = self._sortear_detalle(nodos, dias, rng)
        return self._armar_tablero(loads, cuantiles, u, deltas, nodos, dias)

    def _armar_tablero(self, loads, cuantiles, u, deltas, nodos, dias) -> Tablero:
        escala = np.exp(deltas)[:, None]                                       # (B, 1)
        tarifa = (_interp_cuantiles(u, cuantiles) * escala).astype(np.float32)
        m_comp = self.competidores(nodos, dias)                               # (B,)
        u_comp = percentil_competidor(u, m_comp[:, None])                     # (B, N)
        return Tablero(
            loads=loads,
            precios=precios_niveles(cuantiles, deltas[:, None]),
            u_comp=u_comp,
            precio_competidor=_interp_cuantiles(u_comp, cuantiles.astype(np.float64)) * escala,
            m=m_comp,
            tarifa=tarifa,
        )

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

    def _sortear_detalle(self, nodos: np.ndarray, dias: np.ndarray, rng: np.random.Generator) -> tuple:
        """Cargas (B, N), cuantiles del día usado (B, N, 6) y el uniforme U de la tarifa (B, N)."""
        B, N = len(nodos), self.num_nodes
        loads = self.sortear_cargas(nodos, dias, rng)

        dia_p = np.stack([self._dia_precio(int(dias[b]), nodos[b:b + 1])[0] for b in range(B)])  # (B,N)
        nodos_col = nodos[:, None]
        cuantiles = np.stack([self._q[c][dia_p, nodos_col, self._cols[None, :]]
                              for c in _COLS_Q], axis=-1)                           # (B,N,6)
        u = rng.random((B, N))
        return loads, cuantiles, u

    def llegada_a_nodo(self, i: int, dia: int, rng: np.random.Generator) -> tuple:
        """(loads_row, rate_row) para las lanes que salen del nodo i el día `dia`, sin memoria
        (como `sample_arrivals` con un solo nodo; δ = 0)."""
        loads, rates = self.sample_arrivals([int(i)], [int(dia)], rng)
        return loads[0], rates[0]

    # ── Política que ignora el mercado (DP y B2) ─────────────────────────────

    def sample_arrivals(self, nodos, dias, rng: np.random.Generator) -> tuple:
        """Llegadas por lotes con un único Generator, SIN estado de mercado ni memoria del día real.

        En cada llegada se sortea un día de ENTRENAMIENTO al azar entre los que caen en el mismo
        día de la semana que `dias` (se conserva la estacionalidad semanal, igual que el modelo
        anterior) y se le aplica el mismo generador de cargas/tarifas, con δ = 0: representa una
        política que conoce la distribución (incluida la estacionalidad semanal) pero ignora el
        estado del mercado y el día real concreto (la que supone la DP de UB-DP y B2).

        nodos, dias : (B,) nodo de llegada y día de llegada (solo se usa su día de la semana).
        Devuelve loads (B, N) int64 y rates (B, N) float32. Depende del orden/tamaño del lote
        (consume `rng`); para sorteos independientes de las decisiones del agente usa
        `sample_arrivals_por_mundo`.
        """
        nodos = np.asarray(nodos, dtype=np.int64)
        dow = self.dia_semana(dias)
        dias_azar = np.array([
            self._dias_train_por_dow[int(k)][rng.integers(0, len(self._dias_train_por_dow[int(k)]))]
            for k in dow
        ])
        deltas = np.zeros(len(nodos))
        return self._sortear(nodos, dias_azar, deltas, rng)

    # ── Modo entrenamiento / evaluación con mundos ────────────────────────────

    def sample_arrivals_por_mundo(self, nodos, dias, semillas) -> tuple:
        """Llegadas por lotes, cada fila determinista dada (semilla del episodio, nodo, día).

        nodos, dias, semillas : (B,). Dos filas con la misma terna ven exactamente el mismo sorteo,
        sin importar el resto del lote, el orden de las llamadas ni las acciones del agente.
        Devuelve loads (B, N) int64 y rates (B, N) float32.
        """
        nodos    = np.asarray(nodos,    dtype=np.int64)
        dias     = np.asarray(dias,     dtype=np.int64)
        semillas = np.asarray(semillas, dtype=np.int64)
        B, N = len(nodos), self.num_nodes
        loads = np.empty((B, N), dtype=np.int64)
        rates = np.empty((B, N), dtype=np.float32)
        for b in range(B):
            rng = np.random.Generator(np.random.PCG64(
                [int(semillas[b]), int(nodos[b]), int(dias[b])]
            ))
            delta = self._delta(int(semillas[b]))
            l, r = self._sortear(nodos[b:b + 1], dias[b:b + 1], np.array([delta]), rng)
            loads[b], rates[b] = l[0], r[0]
        return loads, rates

    # ── Modo evaluación / baselines ──────────────────────────────────────────

    def sample_world(self, semilla: int, num_dias: int, dia_inicial: int = 0, devolver_indice: bool = False):
        """Mundo completo: loads_sim y rate_sim de forma [D, N, N], determinista dada la semilla.

        loads_sim[d, i] y rate_sim[d, i] son la fila que ve el agente al llegar al nodo i el día
        real `dia_inicial + d`: el mismo sorteo que entrega `sample_arrivals_por_mundo` con esa
        semilla. Con devolver_indice=True devuelve además el índice de mercado de esos días, (D, N).
        """
        N = self.num_nodes
        nodos = np.arange(N)
        loads_sim = np.empty((num_dias, N, N), dtype=np.float64)
        rate_sim  = np.empty((num_dias, N, N), dtype=np.float64)
        for d in range(num_dias):
            l, r = self.sample_arrivals_por_mundo(
                nodos, np.full(N, dia_inicial + d), np.full(N, semilla),
            )
            loads_sim[d], rate_sim[d] = l, r
        if not devolver_indice:
            return loads_sim, rate_sim
        indice = self._indice_full[dia_inicial:dia_inicial + num_dias]
        return loads_sim, rate_sim, indice

    # ── Cotización conjunta (especificacion_fase0_ruteo_pricing.md §3) ────────
    # Mismas tres formas de uso que arriba, pero devuelven un `Tablero` (cargas, precios por nivel,
    # competidor, m). Las cargas y la tarifa son idénticas a las de los métodos de arriba con la misma
    # semilla: el competidor reutiliza el uniforme de la tarifa, no consume más azar.

    def tablero_sin_memoria(self, nodos, dias, rng: np.random.Generator) -> Tablero:
        """Como `sample_arrivals` (DP de UB-DP y B2): día de train al azar del mismo día de la semana,
        δ = 0. El LTR (y por tanto m) es el de ese día de train sorteado."""
        nodos = np.asarray(nodos, dtype=np.int64)
        dow = self.dia_semana(dias)
        dias_azar = np.array([
            self._dias_train_por_dow[int(k)][rng.integers(0, len(self._dias_train_por_dow[int(k)]))]
            for k in dow
        ])
        return self._tablero(nodos, dias_azar, np.zeros(len(nodos)), rng)

    def tablero_por_mundo(self, nodos, dias, semillas) -> Tablero:
        """Como `sample_arrivals_por_mundo`: cada fila determinista dada (semilla, nodo, día)."""
        nodos    = np.asarray(nodos,    dtype=np.int64)
        dias     = np.asarray(dias,     dtype=np.int64)
        semillas = np.asarray(semillas, dtype=np.int64)
        filas = []
        for b in range(len(nodos)):
            rng = np.random.Generator(np.random.PCG64([int(semillas[b]), int(nodos[b]), int(dias[b])]))
            delta = self._delta(int(semillas[b]))
            t = self._tablero(nodos[b:b + 1], dias[b:b + 1], np.array([delta]), rng)
            filas.append(Tablero(*(getattr(t, c)[0] for c in
                                   ("loads", "precios", "u_comp", "precio_competidor", "m", "tarifa"))))
        return Tablero.apilar(filas)

    def mundo_cotizacion(self, semilla: int, num_dias: int, dia_inicial: int = 0) -> Tablero:
        """Como `sample_world`: Tablero con loads (D, N, N), precios (D, N, N, K), m (D, N), etc.
        La fila [d, i] es la que ve el agente al llegar al nodo i el día real `dia_inicial + d`."""
        N = self.num_nodes
        nodos = np.arange(N)
        return Tablero.apilar([
            self.tablero_por_mundo(nodos, np.full(N, dia_inicial + d), np.full(N, semilla))
            for d in range(num_dias)
        ])

    def tablero_real(self, nodos, dias, semillas) -> Tablero:
        """Replay de días reales (especificación §3.4): cargas y cuantiles del propio día, δ = 0.

        Lo único sorteado es el competidor, con un Generator propio por (semilla del caso, nodo, día),
        así que todos los métodos evaluados en el mismo caso enfrentan la misma competencia."""
        nodos    = np.asarray(nodos,    dtype=np.int64)
        dias     = np.asarray(dias,     dtype=np.int64)
        semillas = np.asarray(semillas, dtype=np.int64)
        B, N = len(nodos), self.num_nodes
        loads = self.count_loads[dias, nodos].astype(np.int64)                    # (B, N)
        dia_p = np.stack([self._dia_precio(int(dias[b]), nodos[b:b + 1])[0] for b in range(B)])
        cuantiles = np.stack([self._q[c][dia_p, nodos[:, None], self._cols[None, :]] for c in _COLS_Q], axis=-1)
        u = np.stack([
            np.random.Generator(np.random.PCG64(
                [int(semillas[b]), int(nodos[b]), int(dias[b]), _TAG_REPLAY])).random(N)
            for b in range(B)
        ])
        return self._armar_tablero(loads, cuantiles, u, np.zeros(B), nodos, dias)


def asegurar_parametros() -> None:
    """Corre el ajuste (fit_market_process) si datos/market_params.npz no existe o es de un ajuste
    anterior (sin alguno de los parámetros que necesita el generador nuevo)."""
    if os.path.exists(params_path()):
        with np.load(params_path()) as z:
            if all(k in z.files for k in _CLAVES_AJUSTE):
                return
        print("market_params.npz no corresponde al ajuste actual: reajustando...")
    else:
        print("market_params.npz no existe: corriendo el ajuste del proceso exógeno...")
    _ajustar()
