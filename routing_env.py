"""
routing_env.py
==============
VectorRoutingEnv: entorno del ruteo estocástico con cotización (especificacion_fase0_ruteo_pricing.md §4),
B episodios en paralelo.

MDP
---
En cada nodo la política elige (j, o): destino j y opción o ∈ {nivel 0 … nivel K−1, VACÍO}, con
VACÍO = OPCION_VACIO = K. step(actions, opciones). El objetivo es el margen total del ciclo: en cada nodo
(casa incluida, antes de salir) hay tres clases de jugada.
  · Cotizar (nivel k, percentiles p10 … p90): válido si j ≠ i, la lane i→j tiene alguna carga que
    no se haya perdido en esta visita y quedan cotizaciones (COTIZACIONES_MAX_POR_NODO). Se gana si u_k < u_comp
    (pricing.gana); ningún nivel gana con seguridad. Si se gana, el camión viaja a j y cobra:
    reward = round(p_k·d − c).
    Si se pierde: el camión se queda en i, el tiempo avanza COTIZACION_HORAS, reward 0, esa carga
    se pierde (la lane se puede volver a cotizar con OTRA carga, otro competidor independiente, hasta agotar las
    suyas), no se consume K y el resto del tablero no cambia.
  · VACÍO a j ≠ i (reward round(−c)): reposicionarse a un nodo a ≤ REPOSICION_MAX_MILLAS, o volver a casa
    desde un nodo intermedio a cualquier distancia. Vale aunque la lane tenga carga.
  · ESPERAR = (j = i, VACÍO): el camión se queda en i hasta el siguiente día de mercado (fin_espera:
    el próximo múltiplo de HORAS_DIA_MERCADO), reward 0, sin consumir K ni contar como arco. Al despertar
    se revela el tablero de i del día nuevo y se reinician los rechazos de la visita.
Todo movimiento (y la espera, y perder una cotización lejos de casa) exige poder volver a casa dentro de max_duration. Si un paso se pasa de
max_duration, termina con TIME_VIOLATION_PENALTY.
`opciones_validas` (B, N, K+1) es la máscara de opciones y la máscara de destinos es su `any(-1)`.

Revelación post-decisión: el tablero de un nodo (cargas, precios por nivel, competencia m y el
competidor oculto) se revela AL LLEGAR, y es función pura de (semilla del episodio, nodo, día de
llegada), con día de llegada = start_day_idx + int(time_elapsed // 14) recortado al último día con
datos. Dos fuentes de tablero:
  · mundos simulados: market_process=MarketProcess → `tablero_por_mundo` (cargas, cuantiles y δ
    sorteados del proceso exógeno);
  · replay de días reales: mercado_replay=MarketProcess → `tablero_real` (cargas y cuantiles del día
    real day_offset + día; solo el competidor se sortea, con la semilla del caso).
El agente observa los precios por nivel (`precios_row`), las cargas y m (`competencia`), nunca el
competidor (`u_comp_row` existe solo para la cota con información perfecta y los tests).
`reward_rows` y `rate_row_obs` son el margen y el precio en la mediana de cada lane: lo observable.

Terminación:
  A) el camión viaja de vuelta al start_node, o el tiempo pasa de max_duration;
  B) no queda ninguna acción factible. Fuera de casa no pasa (volver a casa siempre vale). En casa:
     · si ya en reset() no hay ninguna jugada (sin carga cotizable, sin reposicionamiento ni tiempo para
       esperar), el camión NO SALE: el ciclo termina con margen 0, no es violación (`sin_salida`) y la fila
       arranca inactiva (`active`). Es exógeno: le pasa igual a todos los métodos.
     · si la política tenía jugadas y se quedó sin ellas (perdió las cotizaciones o esperó hasta que no quedó
       nada), el ciclo FALLA: violación con TIME_VIOLATION_PENALTY, como pasarse del tiempo. El ciclo hay que
       hacerlo; cotizar alto a propósito no sirve para saltarse un ciclo malo.

Máscara de destinos: opciones_validas.any(-1). El self-loop es la espera. El camión puede revisitar
nodos intermedios.

Índice de mercado (feature del agente)
--------------------------------------
`market_index` (B, N): con market_process, el índice del día real de llegada
(`MarketProcess.indice_mundo_lote`); en replay, el índice observado del día real
(`MarketProcess.indice_historico`). `index_std` es la desviación con la que se normaliza.

Presupuesto de cardinalidad K (número de arcos, incluido el retorno)
----------------------------------------------------------------------
reset(k_max_arr=[K o None, …]). None = caso libre. Con K, k_remaining baja en 1 por cada arco
recorrido (un rechazo o una espera no consumen K) y, con k_remaining == 1, solo quedan volver a casa o
esperar.
Formulación ≤ K: la política puede cerrar el ciclo antes.
"""

import numpy as np

from config import (
    MAX_DURATION,
    REWARD_SCALE_FACTOR,
    RETURN_SUCCESS_BONUS,
    TIME_VIOLATION_PENALTY,
    USE_REWARD_SHAPING,
    MARKET_SEED_TRAIN,
    COTIZACION_HORAS,
    COTIZACIONES_MAX_POR_NODO,
    HORAS_DIA_MERCADO,
    REPOSICION_MAX_MILLAS,
)
from fit_market_process import dia_de_semana
from pricing import N_NIVELES, U_NIVELES_ARR, gana, indice_nivel, lane_cost_row
from problem_data import lane_reward

OPCION_VACIO = N_NIVELES                 # índice de la opción "ir sin cotizar" (tras los K niveles)
_NIVEL_MEDIANA = indice_nivel(0.5)


def fin_espera(t):
    """Hora en que termina una espera que empieza en t: el siguiente día de mercado (próximo múltiplo de
    HORAS_DIA_MERCADO, estrictamente mayor que t). Sirve con escalares o arrays."""
    return (np.floor(np.asarray(t, dtype=np.float64) / HORAS_DIA_MERCADO) + 1.0) * HORAS_DIA_MERCADO


def max_esperas(max_duration: float = MAX_DURATION) -> int:
    """Cota del número de esperas de un episodio: cada una salta a un día de mercado nuevo."""
    return int(max_duration // HORAS_DIA_MERCADO) + 1


def vacio_permitido(distance_arr, max_millas: float = REPOSICION_MAX_MILLAS) -> np.ndarray:
    """(N, N) bool: lanes i→j (i ≠ j) que se pueden recorrer VACÍAS para reposicionarse (≤ max_millas).
    Volver a casa vacío desde un nodo intermedio vale a cualquier distancia y va aparte."""
    ok = np.asarray(distance_arr, dtype=float) <= max_millas
    np.fill_diagonal(ok, False)
    return ok


def max_pasos(time_matrix, max_duration: float = MAX_DURATION) -> int:
    """Cota del número de arcos de un episodio.

    Sin self-loop, cada arco consume al menos el menor tiempo de viaje fuera de la diagonal;
    el arco que excede `max_duration` termina el episodio, así que como mucho caben
    floor(max_duration / T_min) arcos más ese último.
    """
    T = np.asarray(time_matrix, dtype=float)
    t_min = T[~np.eye(T.shape[0], dtype=bool)].min()
    return int(max_duration // t_min) + 1


def max_pasos_cotizacion(time_matrix, max_duration: float = MAX_DURATION) -> int:
    """Cota de pasos con cotización: en cada visita (una llegada o una espera), hasta COTIZACIONES_MAX_POR_NODO
    rechazos más el paso que viaja o espera."""
    return (max_pasos(time_matrix, max_duration) + max_esperas(max_duration)) * (COTIZACIONES_MAX_POR_NODO + 1)


class VectorRoutingEnv:
    """
    B episodios en paralelo; el estado son arrays (B, ...). Interfaz propia (no Gymnasium).

    Parámetros
    ----------
    time_matrix    : (N, N) horas de viaje.
    num_nodes      : N.
    distance_arr, diesel_arr : (N, N) para el reward y el costo de cada lane.
    num_dias       : días con datos del entorno (el día de llegada se recorta a num_dias − 1).
    market_process : MarketProcess para mundos simulados (exclusivo con mercado_replay).
    mercado_replay : MarketProcess para el replay de días reales (día absoluto = day_offset + día).
    market_seed    : semilla del generador que reparte las semillas de episodio cuando reset()
                     no recibe `world_seeds`.
    day_offset     : índice absoluto del día 0 del entorno (TRAIN_DAYS para el replay de los días de
                     evaluación). Define el día de la semana y, en replay, el día real del tablero.
    """

    def __init__(
        self,
        time_matrix,
        num_nodes:       int,
        distance_arr:    np.ndarray,
        diesel_arr:      np.ndarray,
        num_dias:        int,
        max_duration:    float = MAX_DURATION,
        market_process = None,
        mercado_replay = None,
        market_seed:     int = MARKET_SEED_TRAIN,
        day_offset:      int = 0,
    ):
        if (market_process is None) == (mercado_replay is None):
            raise ValueError("pasar exactamente uno de market_process (mundos simulados) o mercado_replay (replay)")
        self._market         = market_process
        self._mercado_replay = mercado_replay
        self._time_matrix_arr = (
            time_matrix.to_numpy(dtype=float)
            if hasattr(time_matrix, "to_numpy")
            else np.asarray(time_matrix, dtype=float)
        )
        self.num_nodes     = num_nodes
        self.max_duration  = max_duration
        self._distance_arr = np.asarray(distance_arr)
        self._diesel_arr   = np.asarray(diesel_arr)
        self._day_offset   = int(day_offset)
        self._max_day      = int(num_dias) - 1
        self._seed_rng     = np.random.default_rng(market_seed)   # reparte semillas de episodio
        mercado = market_process if market_process is not None else mercado_replay
        self._indice_std   = mercado.indice_std
        self._indice_stack = (None if market_process is not None
                              else mercado_replay.indice_historico(self._day_offset, self._day_offset + int(num_dias)))

        # Inicializados por reset()
        self._B:                   int        = 0
        self._start_node:          np.ndarray = None   # (B,) int64
        self._current_node:        np.ndarray = None   # (B,) int64
        self._time_elapsed:        np.ndarray = None   # (B,) float32
        self._step_count:          np.ndarray = None   # (B,) int64 — arcos recorridos
        self._start_day_idx:       np.ndarray = None   # (B,) int64
        self._active:              np.ndarray = None   # (B,) bool
        self._k_restricted:        np.ndarray = None   # (B,) bool
        self._k_max:               np.ndarray = None   # (B,) int64 — indefinido donde no restringido
        self._k_remaining:         np.ndarray = None   # (B,) int64 — indefinido donde no restringido
        self._last_time_violation: np.ndarray = None   # (B,) bool
        self._last_raw_reward:     np.ndarray = None   # (B,) float32 — reward de la lane del último step()
        self._world_seed:          np.ndarray = None   # (B,) int64 — semilla de cada episodio
        self._reward_row:          np.ndarray = None   # (B, N) float32 — margen a la mediana de las lanes salientes
        self._current_loads_row:   np.ndarray = None   # (B, N) int64 — cargas del tablero
        self._current_rate_row:    np.ndarray = None   # (B, N) float32 — precio en la mediana (USD/milla)
        self._precios:       np.ndarray = None   # (B, N, K) USD/milla de cada nivel en las lanes salientes
        self._u_comp:        np.ndarray = None   # (B, N, C) percentil del mejor competidor por carga (oculto al agente)
        self._precio_comp:   np.ndarray = None   # (B, N, C) precio del mejor competidor (USD/milla)
        self._m:             np.ndarray = None   # (B,) competidores efectivos del nodo actual
        self._perdidas:      np.ndarray = None   # (B, N) int64 — cargas perdidas por lane en esta visita al nodo
        self._n_rechazos:    np.ndarray = None   # (B,) int64 — rechazos en esta visita al nodo
        self._opciones:      np.ndarray = None   # (B, N, K+1) bool — opciones válidas del estado actual
        self._last_cotizo:   np.ndarray = None   # (B,) bool — el último step() fue una cotización
        self._last_gano:     np.ndarray = None   # (B,) bool — … y se ganó la carga
        self._last_precio:   np.ndarray = None   # (B,) USD/milla cobrado (NaN si no se cobró)
        self._last_precio_comp: np.ndarray = None   # (B,) USD/milla del competidor en esa lane (NaN si VACÍO)
        self._last_espero:   np.ndarray = None   # (B,) bool — el último step() fue una espera
        self._sin_salida:    np.ndarray = None   # (B,) bool — el ciclo terminó en casa sin salir (terminación B)
        self._reposicion = vacio_permitido(self._distance_arr)        # (N, N) vacío permitido por distancia
        T = self._time_matrix_arr
        vuelta = T + T.T
        np.fill_diagonal(vuelta, np.inf)
        self._t_vuelta_min = vuelta.min(axis=1)  # (N,) la ida y vuelta más corta desde cada nodo (para esperar en casa)

    # ─────────────────────────────────────────────────────────────────────────
    # Ciclo de vida
    # ─────────────────────────────────────────────────────────────────────────

    def reset(
        self,
        start_nodes:    np.ndarray,         # (B,) int64
        start_day_idxs: np.ndarray,         # (B,) int64
        k_max_arr:      list = None,        # length-B list of (int | None); None ⇒ todos libres
        world_seeds:    np.ndarray = None,  # (B,) semillas de episodio; None ⇒ se reparten desde market_seed.
                                            # Las MISMAS semillas reproducen los mismos tableros y competidores.
    ) -> np.ndarray:                        # máscaras (B, N) int8
        B, N, K = len(start_nodes), self.num_nodes, N_NIVELES
        self._B = B

        self._start_node    = np.asarray(start_nodes,    dtype=np.int64)
        self._current_node  = self._start_node.copy()
        self._time_elapsed  = np.zeros(B, dtype=np.float32)
        self._step_count    = np.zeros(B, dtype=np.int64)
        self._start_day_idx = np.asarray(start_day_idxs, dtype=np.int64)
        self._active        = np.ones(B, dtype=bool)

        if k_max_arr is None:
            self._k_restricted = np.zeros(B, dtype=bool)
            self._k_max        = np.zeros(B, dtype=np.int64)
        else:
            self._k_restricted = np.array([k is not None for k in k_max_arr], dtype=bool)
            self._k_max = np.array([int(k) if k is not None else 0 for k in k_max_arr], dtype=np.int64)
        self._k_remaining = self._k_max.copy()
        self._last_time_violation = np.zeros(B, dtype=bool)
        self._last_raw_reward     = np.zeros(B, dtype=np.float32)

        if world_seeds is None:
            world_seeds = self._seed_rng.integers(0, 2**31 - 1, size=B)
        self._world_seed = np.asarray(world_seeds, dtype=np.int64)
        self._reward_row        = np.zeros((B, N), dtype=np.float32)
        self._current_loads_row = np.zeros((B, N), dtype=np.int64)
        self._current_rate_row  = np.zeros((B, N), dtype=np.float32)
        self._precios      = np.zeros((B, N, K), dtype=np.float64)
        self._u_comp       = np.zeros((B, N, COTIZACIONES_MAX_POR_NODO), dtype=np.float64)
        self._precio_comp  = np.zeros((B, N, COTIZACIONES_MAX_POR_NODO), dtype=np.float64)
        self._m            = np.ones(B, dtype=np.float64)
        self._perdidas     = np.zeros((B, N), dtype=np.int64)
        self._n_rechazos   = np.zeros(B, dtype=np.int64)
        self._last_cotizo  = np.zeros(B, dtype=bool)
        self._last_gano    = np.zeros(B, dtype=bool)
        self._last_precio  = np.full(B, np.nan)
        self._last_precio_comp = np.full(B, np.nan)
        self._last_espero = np.zeros(B, dtype=bool)
        self._revelar_tablero(np.arange(B), np.minimum(self._start_day_idx, self._max_day))
        masks = self._compute_masks()
        self._sin_salida = masks.sum(axis=1) == 0                # no hay ninguna jugada desde casa: no sale
        self._active &= ~self._sin_salida
        masks[self._sin_salida, 0] = 1                           # filas inactivas: una opción ficticia válida
        self._opciones[self._sin_salida, 0, OPCION_VACIO] = True
        return masks

    def step(self, actions: np.ndarray, opciones: np.ndarray) -> tuple:
        """Un paso de los B episodios: actions (B,) destino, opciones (B,) nivel 0 … K−1 o VACÍO = K.

        Solo se procesan las filas activas; las inactivas reciben reward 0 y terminated False.
        Devuelve (rewards (B,), terminated (B,), next_masks (B, N)), con terminated True solo en las
        filas que terminan en ESTE paso.
        """
        actions  = np.asarray(actions,  dtype=np.int64)
        opciones = np.asarray(opciones, dtype=np.int64)
        B, K = self._B, N_NIVELES
        filas = np.arange(B)
        act = self._active
        if act.any() and not self._opciones[filas[act], actions[act], opciones[act]].all():
            malas = filas[act][~self._opciones[filas[act], actions[act], opciones[act]]]
            raise ValueError(f"(destino, opción) fuera de la máscara en las filas {malas.tolist()}")

        cotiza = act & (opciones < K)
        nivel = np.minimum(opciones, K - 1)
        intento = np.minimum(self._perdidas[filas, actions], COTIZACIONES_MAX_POR_NODO - 1)   # la carga que se cotiza
        gano = cotiza & gana(U_NIVELES_ARR[nivel], self._u_comp[filas, actions, intento])
        pierde = cotiza & ~gano
        espera = act & (actions == self._current_node)            # (i, VACÍO): esperar al día siguiente
        viaja = act & ~pierde & ~espera                           # ganó, o VACÍO a otro nodo

        # ── Tiempo ────────────────────────────────────────────────────────────
        t_viaje = self._time_matrix_arr[self._current_node, actions].astype(np.float32)
        next_time = np.where(viaja, self._time_elapsed + t_viaje,
                             np.where(espera, fin_espera(self._time_elapsed),
                                      self._time_elapsed + np.float32(COTIZACION_HORAS))).astype(np.float32)

        # ── Reward de la lane (unidades originales) ───────────────────────────
        nodos = self._current_node
        precio = np.where(gano, self._precios[filas, actions, nivel], 0.0)
        raw = lane_reward(precio, gano.astype(np.int64), self._distance_arr[nodos, actions],
                          self._diesel_arr[nodos, actions]).astype(np.float32)
        raw = np.where(viaja, raw, 0.0).astype(np.float32)
        self._last_raw_reward = raw
        self._last_espero = espera
        self._last_cotizo = cotiza
        self._last_gano = gano
        self._last_precio = np.where(gano, precio, np.nan)
        self._last_precio_comp = np.where(cotiza, self._precio_comp[filas, actions, intento], np.nan)

        # ── Terminación (condición A) ─────────────────────────────────────────
        vuelve = viaja & (actions == self._start_node)
        excede = act & (next_time > self.max_duration)
        cond_a = vuelve | excede
        success = vuelve & (next_time <= self.max_duration)
        time_fail = act & cond_a & ~success
        self._last_time_violation = time_fail.copy()

        terminal = np.zeros(B, dtype=np.float32)
        if USE_REWARD_SHAPING and success.any():
            denom = max(self.num_nodes - 1, 1)
            terminal[success] = (
                RETURN_SUCCESS_BONUS
                + (RETURN_SUCCESS_BONUS * 0.5) * (self._step_count[success].astype(np.float32) / denom)
                + (RETURN_SUCCESS_BONUS * 0.3) * (next_time[success] / self.max_duration)
            )
        terminal[time_fail] = TIME_VIOLATION_PENALTY
        rewards = np.where(act, raw / REWARD_SCALE_FACTOR + terminal, 0.0).astype(np.float32)

        # ── Estado ────────────────────────────────────────────────────────────
        self._current_node = np.where(viaja, actions, self._current_node).astype(np.int64)
        self._time_elapsed = np.where(act, next_time, self._time_elapsed).astype(np.float32)
        self._step_count = np.where(viaja, self._step_count + 1, self._step_count)          # arcos recorridos
        self._k_remaining = np.where(viaja, self._k_remaining - 1, self._k_remaining)      # rechazo y espera no consumen K
        self._perdidas[filas[pierde], actions[pierde]] += 1
        self._n_rechazos = np.where(pierde, self._n_rechazos + 1, self._n_rechazos)

        # ── Tablero del nodo de llegada, o del día nuevo tras esperar (nueva visita) ──
        llega = np.flatnonzero((viaja | espera) & ~cond_a)
        if len(llega):
            dia = self._start_day_idx[llega] + (self._time_elapsed[llega] // 14).astype(np.int64)
            self._perdidas[llega] = 0
            self._n_rechazos[llega] = 0
            self._revelar_tablero(llega, np.minimum(dia, self._max_day))

        self._active = act & ~cond_a
        next_masks = self._compute_masks()
        cond_b = (next_masks.sum(axis=1) == 0) & self._active
        if cond_b.any():                                         # se quedó sin jugadas en casa: el ciclo falla
            rewards = np.where(cond_b, rewards + np.float32(TIME_VIOLATION_PENALTY), rewards).astype(np.float32)
            self._last_time_violation = self._last_time_violation | cond_b
        self._active &= ~cond_b
        terminated = (act & cond_a) | cond_b

        vacias = next_masks.sum(axis=1) == 0                     # filas inactivas: una opción ficticia válida
        next_masks[vacias, 0] = 1
        self._opciones[vacias, 0, OPCION_VACIO] = True
        return rewards, terminated, next_masks

    # ─────────────────────────────────────────────────────────────────────────
    # Internos
    # ─────────────────────────────────────────────────────────────────────────

    def _revelar_tablero(self, rows: np.ndarray, arrival_day: np.ndarray) -> None:
        """Tablero del nodo actual de `rows` al llegar (función pura de semilla, nodo y día)."""
        nodes = self._current_node[rows]
        seeds = self._world_seed[rows]
        if self._market is not None:
            t = self._market.tablero_por_mundo(nodes, arrival_day, seeds)
        else:
            t = self._mercado_replay.tablero_real(nodes, self._day_offset + arrival_day, seeds)
        self._current_loads_row[rows] = t.loads
        self._precios[rows] = t.precios
        self._u_comp[rows] = t.u_comp
        self._precio_comp[rows] = t.precio_competidor
        self._m[rows] = t.m
        # Lo observable de cada lane: el precio y el margen si se cobrara la mediana del mercado.
        mediana = t.precios[..., _NIVEL_MEDIANA]
        self._current_rate_row[rows] = mediana.astype(np.float32)
        reward = lane_reward(mediana, t.loads, self._distance_arr[nodes], self._diesel_arr[nodes]).astype(np.float32)
        reward[np.arange(len(rows)), nodes] = 0.0           # self-loop = esperar: margen 0
        self._reward_row[rows] = reward

    def _puede_cotizar(self) -> np.ndarray:
        """(B, N) a la lane le queda alguna carga sin perder y quedan cotizaciones en la visita."""
        return self.intentos > 0

    def _compute_masks(self) -> np.ndarray:
        """Máscara de destinos (B, N) int8; deja en self._opciones (B, N, K+1) las opciones válidas.

        Por destino j desde el nodo actual i (ver el docstring del módulo):
          · niveles: j ≠ i y la lane se puede cotizar (`_puede_cotizar`);
          · VACÍO a j ≠ i: reposicionamiento (≤ REPOSICION_MAX_MILLAS) o, desde un intermedio, volver a casa;
          · VACÍO en j = i: esperar al siguiente día de mercado.
        Factibilidad de tiempo: después del movimiento (o de la espera) hay que poder volver a casa dentro de
        max_duration; en casa, antes de salir, la espera deja tiempo para la ida y vuelta más corta.
        Con k_remaining == 1 solo quedan casa y esperar.
        """
        B, N, K = self._B, self.num_nodes, N_NIVELES
        filas = np.arange(B)
        i, h, t = self._current_node, self._start_node, self._time_elapsed.astype(np.float64)
        jj = np.arange(N)[None, :]
        es_i, es_h = jj == i[:, None], jj == h[:, None]
        en_casa = i == h                                                           # (B,) aún no sale
        T = self._time_matrix_arr

        # Tiempo: llegar a j y volver a casa (casa misma: solo llegar).
        t_vuelta = np.where(es_h, 0.0, T[:, h].T)                                  # (B, N) T[j, h]
        # Volver a casa desde un intermedio vale siempre: si ya no alcanza, el paso termina con la penalización.
        a_tiempo = ((t[:, None] + T[i, :] + t_vuelta) <= self.max_duration) | (es_h & ~en_casa[:, None])
        t_desp = fin_espera(t)                                                     # (B,) fin de la espera
        espera_ok = np.where(en_casa, t_desp + self._t_vuelta_min[h], t_desp + T[i, h]) <= self.max_duration

        # Lejos de casa, cotizar exige que, si se pierde (COTIZACION_HORAS), todavía se pueda volver a casa.
        pierde_ok = en_casa | (t + COTIZACION_HORAS + T[i, h] <= self.max_duration)
        cotizable = self._puede_cotizar() & ~es_i & a_tiempo & pierde_ok[:, None]
        vacio = (self._reposicion[i] | (es_h & ~en_casa[:, None])) & a_tiempo & ~es_i
        if self._k_restricted.any():                                               # K: con un arco, solo casa
            solo_casa = (self._k_restricted & (self._k_remaining == 1))[:, None] & ~es_h
            cotizable &= ~solo_casa
            vacio &= ~solo_casa
        vacio |= es_i & espera_ok[:, None]

        op = np.zeros((B, N, K + 1), dtype=bool)
        op[:, :, :K] = cotizable[:, :, None]
        op[:, :, K] = vacio
        op &= self._active[:, None, None]
        self._opciones = op
        return op.any(axis=-1).astype(np.int8)

    # ─────────────────────────────────────────────────────────────────────────
    # Propiedades de solo lectura
    # ─────────────────────────────────────────────────────────────────────────

    @property
    def start_node(self) -> np.ndarray:
        return self._start_node

    @property
    def current_node(self) -> np.ndarray:
        return self._current_node

    @property
    def time_elapsed(self) -> np.ndarray:
        return self._time_elapsed

    @property
    def step_count(self) -> np.ndarray:
        """(B,) arcos recorridos (un rechazo no cuenta)."""
        return self._step_count

    @property
    def max_steps(self) -> int:
        """Cota del número de pasos de un episodio, rechazos incluidos (max_pasos_cotizacion)."""
        return max_pasos_cotizacion(self._time_matrix_arr, self.max_duration)

    @property
    def current_day_idx(self) -> np.ndarray:
        return np.minimum(self._start_day_idx + (self._time_elapsed // 14).astype(np.int64), self._max_day)

    @property
    def current_dow(self) -> np.ndarray:
        """(B,) día de la semana (lunes = 0) del día en que está cada episodio
        (day_offset + start_day_idx + int(time_elapsed // 14), sin recortar)."""
        dias = self._day_offset + self._start_day_idx + (self._time_elapsed // 14).astype(np.int64)
        return dia_de_semana(dias)

    @property
    def start_dow(self) -> np.ndarray:
        """(B,) día de la semana (lunes = 0) del día de salida de cada episodio."""
        return dia_de_semana(self._day_offset + self._start_day_idx)

    @property
    def reward_rows(self) -> np.ndarray:
        """(B, N) float32 — margen de las lanes salientes si se cobrara la mediana del mercado
        (f0 de las features del agente; el ingreso real depende de la cotización)."""
        return self._reward_row

    @property
    def world_seed(self) -> np.ndarray:
        """(B,) semillas de los episodios."""
        return self._world_seed

    @property
    def current_loads_row(self) -> np.ndarray:
        """(B, N) cargas del tablero del nodo actual."""
        return self._current_loads_row

    @property
    def rate_row_obs(self) -> np.ndarray:
        """(B, N) precio en la mediana del mercado de cada lane saliente (USD/milla)."""
        return self._current_rate_row

    @property
    def index_std(self) -> float:
        """Desviación del índice histórico (normaliza la feature de índice del agente)."""
        return float(self._indice_std)

    @property
    def market_index(self) -> np.ndarray:
        """(B, N) índice de mercado de todos los nodos en el día actual de cada episodio (ver docstring)."""
        if self._market is not None:
            return self._market.indice_mundo_lote(self._world_seed, self.current_day_idx)
        return np.asarray(self._indice_stack, dtype=np.float64)[self.current_day_idx]

    @property
    def k_remaining(self) -> np.ndarray:
        return self._k_remaining

    @property
    def k_restricted(self) -> np.ndarray:
        return self._k_restricted

    @property
    def last_raw_reward(self) -> np.ndarray:
        """(B,) float32 — reward de la lane del último step() en USD, sin REWARD_SCALE_FACTOR ni bonos ni
        penalizaciones (0 en un rechazo y en filas inactivas). Es la unidad en que se reporta el margen."""
        return self._last_raw_reward

    @property
    def last_time_violation(self) -> np.ndarray:
        """(B,) bool — el step() actual terminó por pasarse de max_duration."""
        return self._last_time_violation

    @property
    def opciones_validas(self) -> np.ndarray:
        """(B, N, K+1) bool — opciones válidas por destino: niveles 0 … K−1 y VACÍO (índice OPCION_VACIO)."""
        return self._opciones

    @property
    def precios_row(self) -> np.ndarray:
        """(B, N, K) USD/milla de cada nivel en las lanes salientes del nodo actual."""
        return self._precios

    @property
    def dist_rows(self) -> np.ndarray:
        """(B, N) distancia de las lanes salientes del nodo actual."""
        return self._distance_arr[self._current_node]

    @property
    def cost_rows(self) -> np.ndarray:
        """(B, N) costo USD de las lanes salientes del nodo actual (misma fórmula que lane_reward)."""
        return lane_cost_row(self._distance_arr[self._current_node], self._diesel_arr[self._current_node])

    @property
    def competencia(self) -> np.ndarray:
        """(B,) competidores efectivos m del nodo actual (LTR del día de llegada)."""
        return self._m

    @property
    def perdidas(self) -> np.ndarray:
        """(B, N) int64 — cargas perdidas por lane en la visita actual al nodo."""
        return self._perdidas

    @property
    def intentos(self) -> np.ndarray:
        """(B, N) int64 — cuántas cargas de cada lane se pueden cotizar todavía en esta visita:
        min(cargas − perdidas, COTIZACIONES_MAX_POR_NODO − rechazos)."""
        return np.minimum(self._current_loads_row - self._perdidas,
                          (COTIZACIONES_MAX_POR_NODO - self._n_rechazos)[:, None])

    @property
    def n_rechazos(self) -> np.ndarray:
        """(B,) rechazos en la visita actual al nodo."""
        return self._n_rechazos

    @property
    def u_comp_row(self) -> np.ndarray:
        """(B, N) percentil del competidor de la carga que se cotizaría ahora en cada lane. Solo para la cota con
        información perfecta y los tests: ninguna política lo puede leer."""
        c = np.minimum(self._perdidas, COTIZACIONES_MAX_POR_NODO - 1)
        return np.take_along_axis(self._u_comp, c[..., None], axis=-1)[..., 0]

    @property
    def u_comp_cargas(self) -> np.ndarray:
        """(B, N, C) percentil del competidor de cada carga (idem: solo cota y tests)."""
        return self._u_comp

    @property
    def active(self) -> np.ndarray:
        """(B,) bool — episodios aún en curso (tras reset(), False en las filas que no tienen ninguna jugada)."""
        return self._active

    @property
    def sin_salida(self) -> np.ndarray:
        """(B,) bool — el ciclo terminó en casa sin salir (ninguna jugada posible): margen 0, no es violación."""
        return self._sin_salida

    @property
    def last_espero(self) -> np.ndarray:
        """(B,) bool — el último step() fue una espera (self-loop VACÍO)."""
        return self._last_espero

    @property
    def last_cotizo(self) -> np.ndarray:
        """(B,) bool — el último step() fue una cotización (nivel, no VACÍO)."""
        return self._last_cotizo

    @property
    def last_gano(self) -> np.ndarray:
        """(B,) bool — el último step() cotizó y ganó la carga."""
        return self._last_gano

    @property
    def last_precio(self) -> np.ndarray:
        """(B,) USD/milla cobrado en el último step() (NaN si no se ganó una carga)."""
        return self._last_precio

    @property
    def last_precio_competidor(self) -> np.ndarray:
        """(B,) USD/milla del mejor competidor en la lane cotizada en el último step() (NaN si no cotizó)."""
        return self._last_precio_comp
