"""
Solvers.py
==========
Único solver que se conserva: el label-setting oráculo, que resuelve la mejor
ruta con el mundo completo conocido (UB-PI, ver dp_relajada.py y evaluation.py).
Los demás baselines viven en baselines.py y dp_relajada.py.
"""

import numpy as np
from collections import namedtuple

# ── Constantes de dominio ─────────────────────────────────────────────────────
DAYS_PER_PERIOD = 14            # día de mercado:  day_idx = start + int(t // 14)

# ── Tipos de resultado (namedtuple: soportan desempaquetado por tupla) ────────
RewardResult = namedtuple("RewardResult", "reward duration")


def _day_index(start_day_idx, time_elapsed, max_day):
    """Índice de día de mercado para un tiempo transcurrido."""
    return min(start_day_idx + int(time_elapsed // DAYS_PER_PERIOD), max_day)


def _make_rm_cache(rate_stack, loads_stack, distance_arr, diesel_arr,
                   start_day_idx, max_day, as_numpy=True):
    """Devuelve un closure get_rm(t) que construye y cachea (por llamada) la
    matriz de reward del día correspondiente a `t`.

    as_numpy=True  → np.ndarray float64 (copia explícita).
    as_numpy=False → np.ndarray directo de build_day_matrices (sin copia).
    """
    from problem_data import build_day_matrices
    cache = {}

    def get_rm(time_elapsed):
        day_idx = _day_index(start_day_idx, time_elapsed, max_day)
        if day_idx not in cache:
            _, rm_pen = build_day_matrices(
                rate_stack[day_idx], loads_stack[day_idx], distance_arr, diesel_arr
            )
            cache[day_idx] = np.array(rm_pen, dtype=float) if as_numpy else rm_pen
        return cache[day_idx]

    return get_rm


def simulate_route_reward(
    route, start_node, start_day_idx,
    time_matrix_np, rate_stack, loads_stack,
    distance_arr, diesel_arr,
    lane_filter=False,
):
    """Compute the actual reward for a fixed route using sequential arrival days
    and the same lane-availability filter as RoutingEnv.

    Replicates RoutingEnv exactly:
    - day_idx = min(start_day_idx + int(time_elapsed // 14), max_day)
    - draw_lane_availability(
        i, day_idx, loads_stack,
    )
      applied on every arc except those departing from or arriving at start_node,
      matching the skip-condition "current_node != start_node" in
      RoutingEnv._get_action_mask.
    - If a lane is blocked (lane_exists[j] == 0), that arc yields 0 reward
      (no cargo to haul).

    If lane_filter is False, no lane filtering is applied (deterministic world).
    """
    from problem_data import draw_lane_availability

    if route is None or len(route) < 2:
        return RewardResult(-np.inf, np.inf)

    max_day   = rate_stack.shape[0] - 1
    num_nodes = distance_arr.shape[0]
    # Caché compartido en modo DataFrame: simulate_route_reward indexa con
    # rm.iloc[i, j], por lo que necesita el DataFrame (as_numpy=False).
    _get_rm = _make_rm_cache(rate_stack, loads_stack, distance_arr, diesel_arr,
                             start_day_idx, max_day, as_numpy=False)

    time_elapsed = 0.0
    total_reward = 0.0

    for step in range(len(route) - 1):
        i       = route[step]
        j       = route[step + 1]
        day_idx = _day_index(start_day_idx, time_elapsed, max_day)
        rm      = _get_rm(time_elapsed)
        arc_r   = float(rm[i, j])

        # Mirror RoutingEnv: skip lane filtering when departing from or
        # arriving at start_node (those arcs are always available).
        if (lane_filter
                and i != start_node
                and j != start_node):
            lane_exists = draw_lane_availability(
                i, day_idx, loads_stack,
            )
            if lane_exists[j] == 0:
                arc_r = 0.0

        total_reward += arc_r
        time_elapsed += float(time_matrix_np[i][j])

    return RewardResult(total_reward, time_elapsed)


def _label_setting_core_stochastic(
    start_node, time_matrix_np, rate_stack, loads_stack,
    distance_arr, diesel_arr, max_d, num_n, start_day_idx,
    time_limit_seconds=None,
    k_max=None,
    dominancia=True,
):
    """Label Setting DP for the Orienteering Problem with lane filtering.

    Filters arc expansions using draw_lane_availability (same lane availability
    as RoutingEnv and simulate_route_reward). The return-to-depot closure is
    never filtered.

    State: (current_node, visited_bitmask, time_elapsed, reward).

    k_max : presupuesto de cardinalidad (nº de arcos, incluido el retorno). Se aplica
        filtrando la generación de labels: popcount(new_mask) es el total de arcos si se
        cerrara desde el nuevo label, y solo crece con cada expansión.

    dominancia : True (default) poda labels dominados en (tiempo, reward) dentro del mismo
        (nodo, visitados). Esa poda solo es exacta si la recompensa de un arco no depende
        del día de llegada; en un mundo simulado con filas distintas por día, un label más
        temprano puede ver peores filas después y la poda puede descartar el óptimo (en 1 de
        300 mundos de 35 nodos el oráculo con poda quedó 56 USD bajo el óptimo). Con
        dominancia=False se enumeran todas las rutas sin revisitar: exacto en cualquier
        mundo, y barato porque MAX_DURATION deja pocas rutas factibles.

    Returns (route, reward, duration, timed_out).
    """
    import time as _time
    from collections import deque
    from problem_data import build_day_matrices, draw_lane_availability

    max_day = rate_stack.shape[0] - 1
    deadline = (_time.time() + time_limit_seconds) if time_limit_seconds is not None else None

    _rm_cache = {}
    def _get_rm(t_elapsed):
        day_idx = min(start_day_idx + int(t_elapsed // DAYS_PER_PERIOD), max_day)
        if day_idx not in _rm_cache:
            _, rm_pen = build_day_matrices(
                rate_stack[day_idx], loads_stack[day_idx], distance_arr, diesel_arr
            )
            _rm_cache[day_idx] = np.array(rm_pen, dtype=float)
        return _rm_cache[day_idx]

    start_mask = 1 << start_node
    all_labels = [[start_node, start_mask, 0.0, 0.0, None]]
    open_q = deque([0])
    valid_ids = {0}

    pareto = {(start_node, start_mask): [(0.0, 0.0, 0)]}

    best_close = None
    timed_out = False

    while open_q:
        if deadline is not None and _time.time() > deadline:
            timed_out = True
            break

        idx = open_q.popleft()
        if idx not in valid_ids:
            continue

        cur_node, visited_mask, t_el, rew, _par = all_labels[idx]
        rm_cur = _get_rm(t_el)

        arrival_day = _day_index(start_day_idx, t_el, max_day)
        lane_exists = draw_lane_availability(
            cur_node, arrival_day, loads_stack,
        )

        for next_node in range(num_n):
            if (visited_mask >> next_node) & 1:
                continue
            if cur_node != start_node and lane_exists[next_node] != 1:
                continue

            travel = float(time_matrix_np[cur_node, next_node])
            new_t  = t_el + travel
            ret_t  = float(time_matrix_np[next_node, start_node])

            if new_t + ret_t > max_d:
                continue

            new_mask = visited_mask | (1 << next_node)
            if k_max is not None and bin(new_mask).count("1") > k_max:
                continue   # excedería K arcos incluso cerrando de inmediato

            arc_r    = rm_cur[cur_node, next_node]
            new_r    = rew + arc_r
            new_state = (next_node, new_mask)

            front = pareto.get(new_state, []) if dominancia else []
            dominated = False
            for (et, er, _) in front:
                if et <= new_t and er >= new_r and (et < new_t or er > new_r):
                    dominated = True
                    break
            if dominated:
                continue

            surviving = []
            for (et, er, eid) in front:
                if new_t <= et and new_r >= er and (new_t < et or new_r > er):
                    valid_ids.discard(eid)
                else:
                    surviving.append((et, er, eid))

            new_idx = len(all_labels)
            all_labels.append([next_node, new_mask, new_t, new_r, idx])
            valid_ids.add(new_idx)
            surviving.append((new_t, new_r, new_idx))
            pareto[new_state] = surviving
            open_q.append(new_idx)

            # Return to depot is never filtered — always available.
            rm_close = _get_rm(new_t)
            close_r  = new_r + rm_close[next_node, start_node]
            close_t  = new_t + ret_t
            if best_close is None or close_r > best_close[0]:
                best_close = (close_r, close_t, new_idx)

    if best_close is None:
        return None, -np.inf, np.inf, timed_out

    close_r, close_t, last_idx = best_close
    path = []
    cur = last_idx
    while cur is not None:
        path.append(all_labels[cur][0])
        cur = all_labels[cur][4]
    path.reverse()
    route = path + [start_node]

    return route, close_r, close_t, timed_out


def solve_label_setting_oracle(
    start_node, time_matrix_np, rate_stack, loads_stack,
    distance_arr, diesel_arr, max_d, num_n, start_day_idx,
    time_limit_seconds=None,
    k_max=None,
    dominancia=True,
):
    """Cota superior clarividente (posterior bound) para el mundo estocástico.

    Resuelve el Orienteering Problem de forma exacta vía Label Setting DP,
    pero con la disponibilidad de cada arco ya revelada de antemano
    (mismo criterio que RoutingEnv). Esto le da al solver información que ninguna
    política causal real posee al momento de decidir — por lo tanto este valor
    es un upper bound informacional, NO el óptimo de política (Bellman-óptimo)
    del MDP estocástico. Sirve como techo de referencia para el Bloque 2.

    k_max : int | None — presupuesto de cardinalidad (ver _label_setting_core_stochastic).
        None (default) = caso libre.
    dominancia : ver _label_setting_core_stochastic. UB-PI (evaluation.ub_pi) lo usa en
        False para que la cota sea exacta con recompensas que dependen del día.

    Returns (status, route, total_reward, total_duration).
    status: "Optimal" | "Time-Limited" | "Infeasible".
    """
    route, _label_r, _label_d, timed_out = _label_setting_core_stochastic(
        start_node, time_matrix_np, rate_stack, loads_stack,
        distance_arr, diesel_arr, max_d, num_n, start_day_idx,
        time_limit_seconds=time_limit_seconds,
        k_max=k_max,
        dominancia=dominancia,
    )

    if route is None:
        return "Infeasible", None, -np.inf, np.inf

    # Re-evaluar canónicamente con la MISMA semilla de disponibilidad,
    # para paridad bit-exact con el entorno.
    total_reward, total_duration = simulate_route_reward(
        route, start_node, start_day_idx,
        time_matrix_np, rate_stack, loads_stack, distance_arr, diesel_arr,
        lane_filter=True,
    )

    status = "Time-Limited" if timed_out else "Optimal"
    return status, route, total_reward, total_duration
