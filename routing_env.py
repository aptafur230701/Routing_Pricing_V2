"""
routing_env.py
==============
RoutingEnv: entorno Gymnasium para el problema de ruteo estocástico con
revelación post-decisión de disponibilidad de lanes (Stochastic DVRP).

MDP estocástico — revelación post-decisión
------------------------------------------
Cuando el camión decide moverse al nodo X, NO conoce qué lanes saldrán de X
en la fecha de llegada; esa disponibilidad se revela AL LLEGAR.
Una lane saliente de X existe sii el registro histórico del día de llegada
tiene carga (loads_stack[arrival_day, X, j] > 0). La disponibilidad es
determinista dado (node, arrival_day), donde
arrival_day = start_day_idx + int(time_elapsed // 14), clampado al último día
disponible. Esto garantiza que dos rutas que llegan al mismo nodo en la misma
fecha de calendario siempre ven la misma realización del mundo.

API Gymnasium moderna (estricta):
  obs, info                                = env.reset(seed=..., options={"start_node": k})
  obs, reward, terminated, truncated, info = env.step(action)

`obs` es siempre None: el agente AM-PPO construye sus features desde
attention_encoder.py y consume solo info["action_mask"] / info["k_remaining"].

Condiciones de terminación (exclusivamente):
  A) El camión regresa al start_node       → terminated=True
  B) No existe ninguna acción factible     → terminated=True (red de seguridad)
  truncated siempre es False (no hay límite de pasos).

La máscara de acciones (info["action_mask"]) excluye:
  · self-loops (no existe la acción de esperar)
  · nodos desde los que no es posible regresar al depot dentro de max_duration
  · lanes salientes que no existen según la disponibilidad histórica del día
  start_node siempre permanece válido (la violación de tiempo al regresar la
  detecta _check_termination, no la máscara).
El camión puede volver a pasar por un nodo intermedio dentro del tour: no se
lleva registro de nodos visitados.

Modo market_process (opcional, por defecto desactivado)
-------------------------------------------------------
Con `market_process=None` el entorno reproduce días reales del histórico (todo
lo descrito arriba). Si se pasa un `MarketProcess` (market_process.py), al
resetear y al llegar a cada nodo las cargas y tarifas de las lanes salientes se
SORTEAN del proceso exógeno (loads_row, rate_row) en vez de leerse de
loads_stack / rate_stack:
  · la máscara usa loads_row > 0 con las mismas reglas de hoy;
  · la recompensa del paso usa las filas sorteadas al llegar al nodo actual, con
    la fórmula compartida problem_data.lane_reward (no se usa rm_pen_stack);
  · el día de inicio del episodio solo aporta el día de la semana;
  · cada episodio es un "mundo" con semilla propia (`world_seed`): los sorteos
    son función de (world_seed, nodo, día) y no dependen de las acciones.

Índice de mercado (feature del agente)
--------------------------------------
`market_index` expone el índice de mercado de TODOS los nodos en el día actual del episodio
(ver market_process.py): (N,) en RoutingEnv y (B, N) en VectorRoutingEnv, y `index_std` la
desviación del índice histórico con la que se normaliza.
  · con market_process: el índice del mundo en (nodo, día) — `MarketProcess.indice_mundo`, con el día
    de llegada SIN recortar al histórico (start_day_idx + int(t // 14)); solo las filas ya terminadas
    pueden pasar de MARKET_DIA_MAX y se recortan (sus features no se usan);
  · sin market_process (replay): `indice_stack[día actual]`, el índice observado de ese día real
    (MarketProcess.indice_historico), con el mismo recorte de día que las tarifas del stack.
`indice_stack` / `indice_std` son opcionales en el constructor; leer `market_index` en replay sin
ellos es un error.

Presupuesto de cardinalidad K (número de arcos, incluido el retorno)
----------------------------------------------------------------------
Se pasa por episodio vía reset(options={"start_node": s, "k_max": K}).
  · k_max ausente o None → caso libre (comportamiento idéntico al actual):
    k_restricted=False, k_remaining_norm=1.0 constante.
  · k_max = K (int) → caso restringido: k_remaining se inicializa en K y se
    decrementa en 1 por cada arco recorrido (cada llamada a step()).
Formulación ≤ K (no = K): la política puede cerrar el ciclo antes de agotar
el presupuesto. La única regla de máscara adicional es: si k_restricted y
k_remaining == 1, la única acción válida es start_node (retorno forzado).
Esta regla se aplica DESPUÉS de toda la lógica de máscara existente (incluido
el self-loop), así que la conformidad ≤ K queda garantizada por construcción,
nunca por penalización de reward. Si el retorno forzado viola max_duration,
_check_termination aplica el mismo TIME_VIOLATION_PENALTY que ya existe hoy
(evento raro, instrumentado en el log de entrenamiento — ver am_training.py).
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from config import (
    MAX_DURATION,
    REWARD_SCALE_FACTOR,
    RETURN_SUCCESS_BONUS,
    TIME_VIOLATION_PENALTY,
    BIG_M_PENALTY,
    USE_REWARD_SHAPING,
    MARKET_SEED_TRAIN,
    MARKET_DIA_MAX,
)
from fit_market_process import dia_de_semana
from problem_data import build_day_matrices, draw_lane_availability, lane_reward


def max_pasos(time_matrix, max_duration: float = MAX_DURATION) -> int:
    """Cota del número de pasos (arcos) de un episodio.

    Sin self-loop, cada arco consume al menos el menor tiempo de viaje fuera de la diagonal;
    el arco que excede `max_duration` termina el episodio, así que como mucho caben
    floor(max_duration / T_min) arcos más ese último. Sirve de techo defensivo en los rollouts.
    """
    T = np.asarray(time_matrix, dtype=float)
    t_min = T[~np.eye(T.shape[0], dtype=bool)].min()
    return int(max_duration // t_min) + 1


class RoutingEnv(gym.Env):
    """
    Entorno Gymnasium para el ruteo estocástico de camiones (SDVRP).

    El agente selecciona el nodo destino en cada paso. El episodio termina
    cuando el camión cierra el ciclo regresando al nodo de inicio (condición A)
    o cuando no existe ninguna acción factible (condición B — red de seguridad).
    No hay truncación por número de pasos.

    Parámetros
    ----------
    time_matrix    : pd.DataFrame | np.ndarray  (num_nodes × num_nodes)
    rate_stack     : np.ndarray  [num_days, num_nodes, num_nodes]
    loads_stack    : np.ndarray  [num_days, num_nodes, num_nodes]
    distance_arr   : np.ndarray  (num_nodes × num_nodes)
    diesel_arr     : np.ndarray  (num_nodes × num_nodes)
    num_nodes      : int
    max_duration   : float, opcional (default: MAX_DURATION)
    market_process : MarketProcess, opcional (default: None = días reales del
                     histórico). Si se pasa, cargas y tarifas se sortean al llegar
                     a cada nodo (ver docstring del módulo).
    indice_stack   : np.ndarray [num_days, num_nodes], opcional — índice de mercado observado
                     de cada día del stack (solo replay; ver docstring del módulo).
    indice_std     : float, opcional — desviación del índice histórico (solo replay; con
                     market_process sale de sus parámetros).
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        time_matrix,
        rate_stack:     np.ndarray,
        loads_stack:    np.ndarray,
        distance_arr:   np.ndarray,
        diesel_arr:     np.ndarray,
        num_nodes:      int,
        max_duration:   float = MAX_DURATION,
        market_process = None,
        indice_stack:   np.ndarray = None,
        indice_std:     float = None,
    ):
        super().__init__()

        self._time_matrix   = time_matrix
        self._rate_stack    = rate_stack
        self._loads_stack   = loads_stack
        self._distance_arr  = distance_arr
        self._diesel_arr    = diesel_arr
        self.num_nodes      = num_nodes
        self.max_duration   = max_duration

        self.action_space = spaces.Discrete(num_nodes)

        self._start_node:         int      = None
        self._current_node:       int      = None
        self._time_elapsed:       float    = None
        self._step_count:         int      = None
        self._start_day_idx:      int      = None
        self._current_lane_exists: np.ndarray = None
        self._k_max:               int      = None
        self._k_remaining:         int      = None
        self._k_restricted:        bool     = False
        # Cache: day_idx → reward_matrix_penalized (np.ndarray).
        # Persists across episodes since build_day_matrices is deterministic.
        self._rm_pen_cache:       dict     = {}
        # Modo market_process
        self._market               = market_process
        self._indice_stack         = indice_stack
        self._indice_std           = market_process.params.indice_std if market_process is not None else indice_std
        self._world_seed:          int      = None
        self._reward_row:          np.ndarray = None   # (N,) recompensa penalizada de las lanes salientes
        self._current_loads_row:   np.ndarray = None   # (N,) cargas sorteadas al llegar al nodo actual
        self._current_rate_row:    np.ndarray = None   # (N,) tarifa por milla sorteada

    # ─────────────────────────────────────────────────────────────────────────
    # Ciclo de vida del entorno
    # ─────────────────────────────────────────────────────────────────────────

    def reset(self, seed: int = None, options: dict = None):
        super().reset(seed=seed)

        if options is not None and "start_node" in options:
            self._start_node = int(options["start_node"])
        else:
            self._start_node = int(self.np_random.integers(0, self.num_nodes))

        if options is not None and "start_day_idx" in options:
            self._start_day_idx = int(options["start_day_idx"])
        else:
            self._start_day_idx = 0

        self._current_node = self._start_node
        self._time_elapsed = 0.0
        self._step_count   = 0

        k_max = None
        if options is not None and "k_max" in options:
            k_max = options["k_max"]
        if k_max is not None:
            self._k_restricted = True
            self._k_max        = int(k_max)
            self._k_remaining   = self._k_max
        else:
            self._k_restricted = False
            self._k_max        = None
            self._k_remaining   = None

        # Disponibilidad inicial para las lanes salientes del nodo de inicio.
        # time_elapsed=0 → arrival_day = start_day_idx (sin offset).
        if self._market is not None:
            if options is not None and "world_seed" in options:
                self._world_seed = int(options["world_seed"])
            else:
                self._world_seed = int(self.np_random.integers(0, 2**31 - 1))
            self._revelar_mercado(self._start_node, self._start_day_idx)
        else:
            max_day = self._rate_stack.shape[0] - 1
            arrival_day = min(self._start_day_idx, max_day)
            self._current_lane_exists = draw_lane_availability(
                self._start_node, arrival_day, self._loads_stack,
            )

        info = {
            "action_mask": self._get_action_mask(),
            "k_remaining": self._k_remaining,
            "k_restricted": self._k_restricted,
        }
        return None, info

    def step(self, action: int):
        assert self._current_node is not None, (
            "El entorno no está inicializado. Llama env.reset() antes de env.step()."
        )

        next_node = int(action)

        # 1. Dinámica de transición
        next_time = self._transition(next_node)

        # 2. Condición A: retorno al depot o violación de tiempo
        terminated = self._check_termination(next_node, next_time)

        # 3. Recompensa del paso
        reward = self._compute_reward(next_node, next_time, terminated)

        # 4. Actualizar estado interno
        self._current_node = next_node
        self._time_elapsed = next_time
        self._step_count  += 1
        if self._k_restricted:
            self._k_remaining -= 1

        # 5. Revelar disponibilidad de lanes desde next_node (post-decisión).
        # Seed basado en (start_day_idx, node, arrival_day): dos rutas que llegan
        # al mismo nodo el mismo día de calendario ven la misma realización.
        if self._market is not None:
            # El día de llegada no se recorta al histórico. Como en VectorRoutingEnv, no se revela
            # nada tras una terminación por la condición A: sus filas no se usan y, con una violación
            # de tiempo, el día puede salirse del mundo simulado.
            if not terminated:
                self._revelar_mercado(next_node, self._start_day_idx + int(next_time // 14))
        else:
            max_day     = self._rate_stack.shape[0] - 1
            arrival_day = min(self._start_day_idx + int(next_time // 14), max_day)
            self._current_lane_exists = draw_lane_availability(
                next_node, arrival_day, self._loads_stack,
            )

        # 6. Máscara del siguiente estado
        next_mask = self._get_action_mask()

        # Condición B: si la máscara quedó vacía, terminar limpiamente
        if not terminated and next_mask.sum() == 0:
            terminated = True

        info = {
            "action_mask": next_mask,
            "k_remaining": self._k_remaining,
            "k_restricted": self._k_restricted,
        }
        return None, float(reward), terminated, False, info

    # ─────────────────────────────────────────────────────────────────────────
    # Componentes internos
    # ─────────────────────────────────────────────────────────────────────────

    def _transition(self, next_node: int) -> float:
        return self._time_elapsed + float(self._time_matrix[self._current_node, next_node])

    def _revelar_mercado(self, node: int, arrival_day: int) -> None:
        """Sortea (loads_row, rate_row) del nodo al llegar y fija máscara y recompensas.

        Las filas se observan solo al llegar al nodo. Es función pura de
        (world_seed, nodo, día): no depende de las acciones del agente.
        """
        loads, rates = self._market.sample_arrivals_por_mundo(
            [node], [arrival_day], [self._world_seed],
        )
        self._current_loads_row = loads[0]
        self._current_rate_row  = rates[0]
        self._current_lane_exists = (loads[0] > 0).astype(np.int8)
        # Recompensa de salir de `node` por cada lane: fórmula compartida con
        # build_day_matrices; diagonal penalizada igual que en rm_pen_stack.
        row = lane_reward(
            rates[0], loads[0], self._distance_arr[node], self._diesel_arr[node],
        ).astype(np.float32)
        row[node] = BIG_M_PENALTY
        self._reward_row = row

    def _get_current_day_idx(self) -> int:
        day_offset = int(self._time_elapsed // 14)
        raw = self._start_day_idx + day_offset
        return min(raw, self._rate_stack.shape[0] - 1)

    def _get_rm_pen(self, day_idx: int) -> np.ndarray:
        if day_idx not in self._rm_pen_cache:
            _, self._rm_pen_cache[day_idx] = build_day_matrices(
                self._rate_stack[day_idx],
                self._loads_stack[day_idx],
                self._distance_arr,
                self._diesel_arr,
            )
        return self._rm_pen_cache[day_idx]

    def _compute_reward(
        self, next_node: int, next_time: float, terminated: bool
    ) -> float:
        if self._market is not None:
            raw_reward = float(self._reward_row[next_node])
        else:
            current_day_idx = self._get_current_day_idx()
            rm_pen = self._get_rm_pen(current_day_idx)
            raw_reward = float(rm_pen[self._current_node, next_node])
        step_reward = raw_reward / REWARD_SCALE_FACTOR

        terminal_reward = 0.0
        if terminated:
            if next_node == self._start_node:
                if next_time <= self.max_duration:
                    if USE_REWARD_SHAPING:
                        n_arcs_before_return = self._step_count   # aún sin contar el arco de retorno
                        time_util = next_time / self.max_duration
                        terminal_reward = (
                            RETURN_SUCCESS_BONUS
                            + (RETURN_SUCCESS_BONUS * 0.5) * (n_arcs_before_return / (self.num_nodes - 1))
                            + (RETURN_SUCCESS_BONUS * 0.3) * time_util
                        )
                else:
                    terminal_reward = TIME_VIOLATION_PENALTY
            else:
                terminal_reward = TIME_VIOLATION_PENALTY

        return step_reward + terminal_reward

    def _check_termination(self, next_node: int, next_time: float) -> bool:
        """
        Condición A: el camión cerró el ciclo o excedió el tiempo.
        Retorna solo terminated (bool). truncated siempre es False.
        """
        return (
            next_node == self._start_node
            or next_time > self.max_duration
        )

    def _get_action_mask(self) -> np.ndarray:
        """
        Máscara binaria de acciones válidas para el estado actual.

        Reglas (en orden de aplicación):
        · 0 para self-loop (nodo actual).
        · 0 para nodos intermedios desde los que no se puede regresar al depot
          dentro de max_duration (lookahead temporal).
        · 0 para lanes salientes que no existen según la disponibilidad histórica de
          disponibilidad revelado al llegar al nodo actual.
        · start_node permanece siempre disponible (mask=1) para permitir retorno;
          si el tiempo al regresar excede max_duration, _check_termination lo detecta.
        """
        mask = np.ones(self.num_nodes, dtype=np.int8)

        # Self-loop
        mask[self._current_node] = 0

        # Lookahead temporal (vectorizado): excluir intermedios que agoten el presupuesto.
        # start_node queda exento — su retorno es siempre válido en la máscara.
        non_start   = np.arange(self.num_nodes) != self._start_node
        t_to_j      = self._time_matrix[self._current_node]      # (N,) current→j
        t_j_to_dep  = self._time_matrix[:, self._start_node]     # (N,) j→depot
        over_budget = (self._time_elapsed + t_to_j + t_j_to_dep) > self.max_duration
        mask[non_start & over_budget] = 0

        # Disponibilidad estocástica (vectorizado): excluir lanes que no existen.
        # start_node nunca se enmascara — retorno siempre válido.
        # Cuando current_node == start_node (primer paso) no se aplica.
        if self._current_lane_exists is not None and self._current_node != self._start_node:
            lane_absent = (self._current_lane_exists == 0)
            mask[non_start & lane_absent] = 0

        # Presupuesto de cardinalidad K (último, gana sobre todo lo anterior):
        # con un solo arco de margen, la única acción válida es el retorno.
        if self._k_restricted and self._k_remaining == 1:
            mask[:] = 0
            mask[self._start_node] = 1

        return mask

    # ─────────────────────────────────────────────────────────────────────────
    # Propiedades de solo lectura
    # ─────────────────────────────────────────────────────────────────────────

    @property
    def start_node(self) -> int:
        return self._start_node

    @property
    def current_node(self) -> int:
        return self._current_node

    @property
    def time_elapsed(self) -> float:
        return self._time_elapsed

    @property
    def current_day_idx(self) -> int:
        return self._get_current_day_idx()

    @property
    def k_remaining(self):
        return self._k_remaining

    @property
    def k_restricted(self) -> bool:
        return self._k_restricted

    @property
    def world_seed(self):
        """Semilla del mundo del episodio (None sin market_process)."""
        return self._world_seed

    @property
    def current_loads_row(self):
        """(N,) cargas sorteadas al llegar al nodo actual (None sin market_process)."""
        return self._current_loads_row

    @property
    def index_std(self) -> float:
        """Desviación del índice histórico (normaliza la feature de índice del agente)."""
        if self._indice_std is None:
            raise ValueError("index_std requiere market_process o indice_std en el constructor")
        return float(self._indice_std)

    @property
    def market_index(self) -> np.ndarray:
        """(N,) índice de mercado de todos los nodos en el día actual del episodio (ver docstring del módulo)."""
        if self._market is not None:
            dia = min(self._start_day_idx + int(self._time_elapsed // 14), MARKET_DIA_MAX)
            return self._market.indice_mundo(self._world_seed, dia)
        if self._indice_stack is None:
            raise ValueError("market_index en replay requiere indice_stack en el constructor")
        return np.asarray(self._indice_stack[self._get_current_day_idx()], dtype=np.float64)

    @property
    def current_rate_row(self):
        """(N,) tarifa por milla sorteada al llegar al nodo actual (None sin market_process)."""
        return self._current_rate_row


class VectorRoutingEnv:
    """
    Vectorized version of RoutingEnv: runs B episodes in parallel as a single batch.

    State is stored as numpy arrays of shape (B, ...) instead of scalars.
    Used exclusively by the PPO collection phase in am_training.py.
    Does NOT implement the Gymnasium API — it has a simpler custom interface.

    Parámetros
    ----------
    rm_pen_stack : np.ndarray (num_train_days, N, N) float32
                   Penalized reward matrices precomputed by build_rm_pen_stack().
                   Passed in from training to avoid recomputing per step.
                   Con market_process puede ser None (no se usa).
    market_process : MarketProcess, opcional (default None = días reales).
                   Si se pasa, cargas y tarifas de las lanes salientes se sortean
                   al llegar a cada nodo (ver docstring del módulo); requiere
                   distance_arr y diesel_arr para la recompensa.
    market_seed    : semilla del generador que reparte las semillas de mundo
                   cuando reset() no recibe `world_seeds`.
    day_offset     : índice absoluto (en el histórico completo) del día 0 de los stacks de este
                   entorno. Solo sirve para `current_dow`: 0 cuando los stacks arrancan en el día 0
                   (entrenamiento, mundos simulados); TRAIN_DAYS para el replay de los días de eval.
    indice_stack   : (días, N) índice de mercado observado de cada día del stack (replay; ver el
                   docstring del módulo). Con market_process no se usa.
    indice_std     : desviación del índice histórico (replay). Con market_process sale de sus parámetros.
    """

    def __init__(
        self,
        time_matrix,
        rate_stack:      np.ndarray,
        loads_stack:     np.ndarray,
        rm_pen_stack:    np.ndarray,
        num_nodes:       int,
        max_duration:    float = MAX_DURATION,
        market_process = None,
        distance_arr:    np.ndarray = None,
        diesel_arr:      np.ndarray = None,
        market_seed:     int = MARKET_SEED_TRAIN,
        day_offset:      int = 0,
        indice_stack:    np.ndarray = None,
        indice_std:      float = None,
    ):
        if market_process is None and rm_pen_stack is None:
            raise ValueError("rm_pen_stack es obligatorio sin market_process")
        if market_process is not None and (distance_arr is None or diesel_arr is None):
            raise ValueError("market_process requiere distance_arr y diesel_arr")
        self._time_matrix_arr = (
            time_matrix.to_numpy(dtype=float)
            if hasattr(time_matrix, "to_numpy")
            else np.asarray(time_matrix, dtype=float)
        )
        self._rate_stack     = rate_stack
        self._loads_stack    = loads_stack
        self._rm_pen_stack   = rm_pen_stack   # (num_train_days, N, N)
        self.num_nodes       = num_nodes
        self.max_duration    = max_duration
        self._market         = market_process
        self._day_offset     = int(day_offset)
        self._indice_stack   = indice_stack
        self._indice_std     = market_process.params.indice_std if market_process is not None else indice_std
        self._distance_arr   = distance_arr
        self._diesel_arr     = diesel_arr
        self._seed_rng       = np.random.default_rng(market_seed)   # reparte semillas de mundo
        # Último día con datos (para clamp de features históricas como LTR/camiones).
        self._max_day = (
            self._rm_pen_stack.shape[0] if self._rm_pen_stack is not None
            else rate_stack.shape[0]
        ) - 1

        # Initialized by reset()
        self._B:                   int        = 0
        self._start_node:          np.ndarray = None   # (B,) int64
        self._current_node:        np.ndarray = None   # (B,) int64
        self._time_elapsed:        np.ndarray = None   # (B,) float32
        self._step_count:         np.ndarray = None   # (B,) int64
        self._start_day_idx:       np.ndarray = None   # (B,) int64
        self._current_lane_exists: np.ndarray = None   # (B, N) int8
        self._active:              np.ndarray = None   # (B,) bool
        self._k_restricted:        np.ndarray = None   # (B,) bool
        self._k_max:               np.ndarray = None   # (B,) int64 — indefinido donde no restringido
        self._k_remaining:         np.ndarray = None   # (B,) int64 — indefinido donde no restringido
        self._last_time_violation: np.ndarray = None   # (B,) bool — poblado por step()
        self._last_raw_reward:     np.ndarray = None   # (B,) float32 — recompensa cruda de la lane del último step()
        # Modo market_process
        self._world_seed:          np.ndarray = None   # (B,) int64 — semilla del mundo de cada episodio
        self._reward_row:          np.ndarray = None   # (B, N) float32 — recompensa penalizada de las lanes salientes
        self._current_loads_row:   np.ndarray = None   # (B, N) int64 — cargas sorteadas al llegar
        self._current_rate_row:    np.ndarray = None   # (B, N) float32 — tarifa por milla sorteada

    # ─────────────────────────────────────────────────────────────────────────
    # Lifecycle
    # ─────────────────────────────────────────────────────────────────────────

    def reset(
        self,
        start_nodes:    np.ndarray,   # (B,) int64
        start_day_idxs: np.ndarray,   # (B,) int64
        k_max_arr:      list = None,  # length-B list of (int | None); None ⇒ all free
        world_seeds:    np.ndarray = None,  # (B,) semillas de mundo (solo market_process);
                                            # None ⇒ se reparten desde market_seed
    ) -> np.ndarray:                  # masks (B, N) int8
        B = len(start_nodes)
        N = self.num_nodes
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
            self._k_remaining  = np.zeros(B, dtype=np.int64)
        else:
            self._k_restricted = np.array(
                [k is not None for k in k_max_arr], dtype=bool
            )
            self._k_max = np.array(
                [int(k) if k is not None else 0 for k in k_max_arr], dtype=np.int64
            )
            self._k_remaining = self._k_max.copy()
        self._last_time_violation = np.zeros(B, dtype=bool)
        self._last_raw_reward     = np.zeros(B, dtype=np.float32)

        if self._market is not None:
            # Cada episodio es un mundo con semilla propia. Pasar las MISMAS
            # semillas a otro reset (p.ej. el rollout greedy del self-critical)
            # reproduce las mismas realizaciones de cargas y tarifas.
            if world_seeds is None:
                world_seeds = self._seed_rng.integers(0, 2**31 - 1, size=B)
            self._world_seed = np.asarray(world_seeds, dtype=np.int64)
            self._current_lane_exists = np.zeros((B, N), dtype=np.int8)
            self._reward_row          = np.zeros((B, N), dtype=np.float32)
            self._current_loads_row   = np.zeros((B, N), dtype=np.int64)
            self._current_rate_row    = np.zeros((B, N), dtype=np.float32)
            self._revelar_mercado(np.arange(B), self._start_day_idx)
        else:
            # Initial lane availability: one draw_lane_availability call per row
            max_day = self._rate_stack.shape[0] - 1
            lane_rows = []
            for b in range(B):
                arrival_day = min(int(self._start_day_idx[b]), max_day)
                lane_rows.append(draw_lane_availability(
                    int(self._start_node[b]), arrival_day, self._loads_stack,
                ))
            self._current_lane_exists = np.stack(lane_rows)   # (B, N) int8

        return self._compute_masks()

    def step(
        self,
        actions: np.ndarray,   # (B,) int64
    ) -> tuple:                # (rewards (B,), terminated (B,), next_masks (B, N))
        """
        Execute one step for all B episodes.

        Only active rows (self._active) are processed; inactive rows receive
        reward=0 and terminated=False (they are already done).

        Returns terminated = True only for rows that terminate in THIS step
        (condition A: depot return or time violation; condition B: empty mask).
        """
        B   = self._B
        N   = self.num_nodes
        actions = np.asarray(actions, dtype=np.int64)

        # ── Time transition ───────────────────────────────────────────────────
        t_step   = self._time_matrix_arr[self._current_node, actions].astype(np.float32)
        next_time = self._time_elapsed + t_step

        # ── Condition A termination (only for active rows) ────────────────────
        cond_a      = (actions == self._start_node) | (next_time > self.max_duration)
        terminated_a = self._active & cond_a

        # ── Reward (computed BEFORE state update, using current node/day) ──────
        if self._market is not None:
            # Filas sorteadas al llegar al nodo actual (fórmula compartida lane_reward).
            raw_reward = self._reward_row[np.arange(B), actions]
        else:
            max_rm_day = self._rm_pen_stack.shape[0] - 1
            day_idx = np.minimum(
                self._start_day_idx + (self._time_elapsed // 14).astype(np.int64),
                max_rm_day,
            )   # (B,)
            raw_reward = self._rm_pen_stack[day_idx, self._current_node, actions]
        step_reward = raw_reward.astype(np.float32) / REWARD_SCALE_FACTOR
        # Recompensa de la lane en unidades originales (sin escala ni bonos), 0 en filas inactivas.
        self._last_raw_reward = np.where(self._active, raw_reward, 0.0).astype(np.float32)

        # Terminal reward components (using step_count BEFORE the state update: arcs before the return)
        n_arcs_before_return = self._step_count.astype(np.float32)   # (B,)
        time_util            = next_time / self.max_duration

        success   = self._active & (actions == self._start_node) & (next_time <= self.max_duration)
        time_fail = self._active & cond_a & ~success
        # Instrumentación: violación de tiempo en el retorno (normal o forzado por K).
        self._last_time_violation = time_fail.copy()

        terminal_reward = np.zeros(B, dtype=np.float32)
        if USE_REWARD_SHAPING and success.any():
            denom = max(N - 1, 1)
            terminal_reward[success] = (
                RETURN_SUCCESS_BONUS
                + (RETURN_SUCCESS_BONUS * 0.5) * (n_arcs_before_return[success] / denom)
                + (RETURN_SUCCESS_BONUS * 0.3) * time_util[success]
            )
        if time_fail.any():
            terminal_reward[time_fail] = TIME_VIOLATION_PENALTY

        rewards = np.where(
            self._active,
            step_reward + terminal_reward,
            0.0,
        ).astype(np.float32)

        # ── State update (all active rows, including those terminating by cond A) ─
        self._current_node = np.where(self._active, actions, self._current_node).astype(np.int64)
        self._time_elapsed  = np.where(self._active, next_time, self._time_elapsed)
        self._step_count    = np.where(self._active, self._step_count + 1, self._step_count)
        self._k_remaining   = np.where(self._active, self._k_remaining - 1, self._k_remaining)

        # ── Lane revelation (active rows NOT terminated by condition A) ───────
        reveal = np.where(self._active & ~cond_a)[0]
        if self._market is not None:
            # El día de llegada no se recorta al histórico: el mundo es ilimitado.
            if len(reveal):
                arrival_day = self._start_day_idx[reveal] + (self._time_elapsed[reveal] // 14).astype(np.int64)
                self._revelar_mercado(reveal, arrival_day)
        else:
            max_day = self._rate_stack.shape[0] - 1
            for b in reveal:
                arrival_day = min(
                    int(self._start_day_idx[b]) + int(self._time_elapsed[b] // 14),
                    max_day,
                )
                self._current_lane_exists[b] = draw_lane_availability(
                    int(self._current_node[b]), arrival_day, self._loads_stack,
                )

        # ── Update active after condition A ───────────────────────────────────
        self._active = self._active & ~cond_a

        # ── Compute next masks ────────────────────────────────────────────────
        next_masks = self._compute_masks()

        # ── Condition B: zero valid actions for still-active rows ─────────────
        cond_b = (next_masks.sum(axis=1) == 0) & self._active
        self._active = self._active & ~cond_b

        terminated = terminated_a | cond_b

        # Ensure inactive rows have at least one valid action (prevents _forward crash)
        all_zero = next_masks.sum(axis=1) == 0
        next_masks[all_zero, 0] = 1

        return rewards, terminated, next_masks

    # ─────────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _revelar_mercado(self, rows: np.ndarray, arrival_day: np.ndarray) -> None:
        """Sortea (loads_row, rate_row) de las filas `rows` al llegar a su nodo actual.

        Actualiza la disponibilidad de lanes (máscara) y la fila de recompensas.
        Cada fila es función pura de (semilla de mundo, nodo, día): no depende de
        las acciones del agente ni de qué otras filas sigan activas.
        """
        nodes = self._current_node[rows]
        loads, rates = self._market.sample_arrivals_por_mundo(
            nodes, arrival_day, self._world_seed[rows],
        )
        self._current_loads_row[rows] = loads
        self._current_rate_row[rows]  = rates
        self._current_lane_exists[rows] = (loads > 0).astype(np.int8)
        # Recompensa de salir de `nodes` por cada lane: fórmula compartida con
        # build_day_matrices; diagonal penalizada igual que en rm_pen_stack.
        reward = lane_reward(
            rates, loads, self._distance_arr[nodes], self._diesel_arr[nodes],
        ).astype(np.float32)
        reward[np.arange(len(rows)), nodes] = BIG_M_PENALTY
        self._reward_row[rows] = reward

    def _compute_masks(self) -> np.ndarray:
        """Compute action masks for all B episodes. Returns (B, N) int8."""
        B = self._B
        N = self.num_nodes
        mask = np.ones((B, N), dtype=np.int8)

        # non-depot indicator per (b, j) pair
        non_depot = np.arange(N)[None, :] != self._start_node[:, None]   # (B, N)

        # Temporal lookahead
        t_to_j     = self._time_matrix_arr[self._current_node, :]         # (B, N)
        t_j_to_dep = self._time_matrix_arr[:, self._start_node].T         # (B, N)
        over_budget = (
            self._time_elapsed[:, None] + t_to_j + t_j_to_dep
        ) > self.max_duration                                              # (B, N)
        mask[over_budget & non_depot] = 0
        mask[np.arange(B), self._start_node] = 1

        # Stochastic lane availability (skip when at depot/start_node)
        not_at_start = (self._current_node != self._start_node)[:, None]   # (B, 1)
        lane_absent  = (self._current_lane_exists == 0)                     # (B, N)
        mask[not_at_start & lane_absent & non_depot] = 0
        mask[np.arange(B), self._start_node] = 1

        # Self-loop: applied LAST so it takes precedence over all depot restorations.
        # When current_node == start_node (first step of episode), this masks out
        # the depot itself, matching the scalar RoutingEnv._get_action_mask() behavior.
        mask[np.arange(B), self._current_node] = 0

        # Presupuesto de cardinalidad K: aplicado al final, gana sobre todo lo
        # anterior (incluido el self-loop). Con un solo arco de margen, la
        # única acción válida es el retorno forzado al depot.
        force_k = self._k_restricted & (self._k_remaining == 1)
        if force_k.any():
            mask[force_k, :] = 0
            mask[force_k, self._start_node[force_k]] = 1

        return mask

    # ─────────────────────────────────────────────────────────────────────────
    # Read-only properties
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
        return self._step_count

    @property
    def max_steps(self) -> int:
        """Cota del número de pasos de un episodio (ver max_pasos)."""
        return max_pasos(self._time_matrix_arr, self.max_duration)

    @property
    def current_day_idx(self) -> np.ndarray:
        return np.minimum(
            self._start_day_idx + (self._time_elapsed // 14).astype(np.int64),
            self._max_day,
        )

    @property
    def current_dow(self) -> np.ndarray:
        """(B,) día de la semana (lunes = 0) del día en que está cada episodio.

        Día absoluto = day_offset + start_day_idx + int(time_elapsed // 14), sin recortar al
        último día con datos (con market_process el mundo es ilimitado)."""
        dias = self._day_offset + self._start_day_idx + (self._time_elapsed // 14).astype(np.int64)
        return dia_de_semana(dias)

    @property
    def start_dow(self) -> np.ndarray:
        """(B,) día de la semana (lunes = 0) del día de salida de cada episodio
        (day_offset + start_day_idx, sin recortar al último día con datos)."""
        return dia_de_semana(self._day_offset + self._start_day_idx)

    @property
    def reward_rows(self) -> np.ndarray:
        """(B, N) float32 — recompensa penalizada de las lanes salientes del nodo
        actual (f0 de las features del agente). Con market_process sale de las
        filas sorteadas al llegar; sin él, de rm_pen_stack en el día actual."""
        if self._market is not None:
            return self._reward_row
        return self._rm_pen_stack[self.current_day_idx, self._current_node, :]

    @property
    def world_seed(self) -> np.ndarray:
        """(B,) semillas de mundo de los episodios (None sin market_process)."""
        return self._world_seed

    @property
    def current_loads_row(self) -> np.ndarray:
        """(B, N) cargas sorteadas al llegar al nodo actual (None sin market_process)."""
        return self._current_loads_row

    @property
    def current_rate_row(self) -> np.ndarray:
        """(B, N) tarifa por milla sorteada al llegar al nodo actual (None sin market_process)."""
        return self._current_rate_row

    @property
    def rate_row_obs(self) -> np.ndarray:
        """(B, N) tarifa por milla observada al llegar al nodo actual: la sorteada (market_process) o, en replay,
        la del stack histórico en el día actual. Es la que paga la recompensa del paso (junto con loads_row_obs)."""
        if self._market is not None:
            return self._current_rate_row
        return self._rate_stack[self.current_day_idx, self._current_node, :]

    @property
    def loads_row_obs(self) -> np.ndarray:
        """(B, N) cargas observadas al llegar al nodo actual (sorteadas, o del stack histórico en replay)."""
        if self._market is not None:
            return self._current_loads_row
        return self._loads_stack[self.current_day_idx, self._current_node, :]

    @property
    def index_std(self) -> float:
        """Desviación del índice histórico (normaliza la feature de índice del agente)."""
        if self._indice_std is None:
            raise ValueError("index_std requiere market_process o indice_std en el constructor")
        return float(self._indice_std)

    @property
    def market_index(self) -> np.ndarray:
        """(B, N) índice de mercado de todos los nodos en el día actual de cada episodio.

        Con market_process: el del mundo de cada episodio en su día de llegada sin recortar al histórico
        (solo las filas ya terminadas pueden pasar de MARKET_DIA_MAX; se recortan y no se usan).
        En replay: indice_stack[current_day_idx], el índice observado del día real."""
        if self._market is not None:
            dias = np.minimum(
                self._start_day_idx + (self._time_elapsed // 14).astype(np.int64), MARKET_DIA_MAX,
            )
            return self._market.indice_mundo_lote(self._world_seed, dias)
        if self._indice_stack is None:
            raise ValueError("market_index en replay requiere indice_stack en el constructor")
        return np.asarray(self._indice_stack, dtype=np.float64)[self.current_day_idx]

    @property
    def k_remaining(self) -> np.ndarray:
        return self._k_remaining

    @property
    def k_restricted(self) -> np.ndarray:
        return self._k_restricted

    @property
    def last_raw_reward(self) -> np.ndarray:
        """(B,) float32 — recompensa de la lane recorrida en el último step(), sin
        REWARD_SCALE_FACTOR ni bonos/penalizaciones terminales (0 en filas ya inactivas).
        Es la unidad en que los solvers reportan el reward de una ruta."""
        return self._last_raw_reward

    @property
    def last_time_violation(self) -> np.ndarray:
        """(B,) bool — True para filas cuyo step() actual terminó por violar
        max_duration en el arco de retorno (normal o forzado por K)."""
        return self._last_time_violation
