"""
test_label_setting_free_case_equivalence.py
==============================================
Criterio de aceptación: los 8 solvers activos de Solvers.py, en caso libre
(k_max=None, el default), reproducen EXACTAMENTE el mismo comportamiento que
antes de añadir soporte de K.

Los valores esperados se capturaron ejecutando cada solver sobre datos reales
de 10 nodos (start_node=0, day=0) con el código previo al cambio (vía
`git stash` al HEAD limpio, antes de tocar Solvers.py) y se hardcodean aquí
como regresión permanente. El nombre del archivo referencia el oráculo
Label-Setting por ser el más crítico, pero cubre los 8 solvers.
"""

import numpy as np
import pytest

from config import MAX_DURATION
from problem_data import load_matrices
from Solvers import (
    solve_heuristic_rolling_horizon,
    solve_heuristic_rolling_horizon_lookahead,
    solve_heuristic_rolling_horizon_stochastic,
    solve_heuristic_rolling_horizon_lookahead_stochastic,
    solve_mc_rollout_stochastic,
    solve_HGA_LNS_metaheuristic,
    solve_label_setting_exact,
    solve_label_setting_oracle,
)

NUM_NODES = 10
START_NODE = 0
DAY = 0

# Capturado con el código pre-condicionamiento-en-K (ver docstring arriba).
# route, reward
EXPECTED = {
    "rh_greedy":           ([0, 6, 1, 0], 1490.0),
    "rh_lookahead":        ([0, 6, 1, 0], 1490.0),
    "rh_greedy_stoch":     ([0, 6, 1, 0], 1490.0),
    "rh_lookahead_stoch":  ([0, 6, 1, 0], 1490.0),
    "mc_rollout":          ([0, 6, 1, 0], 1490.0),
    "hga_lns":             ([0, 6, 1, 0], 1490.0),
    "ls_exact":            ([0, 6, 1, 0], 1490.0),
    "ls_oracle":           ([0, 6, 1, 0], 1490.0),
}


@pytest.fixture(scope="module")
def data():
    return load_matrices(NUM_NODES)


def test_free_case_matches_pre_k_conditioning_baseline(data):
    (time_matrix, rate_stack, loads_stack, distance_arr, diesel_arr,
     ltr_stack, trucks_stack, avail_expected_arr, reward_global_p95) = data
    time_matrix_np = np.array(time_matrix, dtype=float)

    r = solve_heuristic_rolling_horizon(
        START_NODE, time_matrix, rate_stack, loads_stack, distance_arr, diesel_arr,
        MAX_DURATION, NUM_NODES, start_day_idx=DAY,
    )
    assert (r.route, r.reward) == EXPECTED["rh_greedy"]

    r = solve_heuristic_rolling_horizon_lookahead(
        START_NODE, time_matrix, rate_stack, loads_stack, distance_arr, diesel_arr,
        MAX_DURATION, NUM_NODES, start_day_idx=DAY,
    )
    assert (r.route, r.reward) == EXPECTED["rh_lookahead"]

    r = solve_heuristic_rolling_horizon_stochastic(
        START_NODE, time_matrix, rate_stack, loads_stack, distance_arr, diesel_arr,
        MAX_DURATION, NUM_NODES, start_day_idx=DAY,
    )
    assert (r.route, r.reward) == EXPECTED["rh_greedy_stoch"]

    r = solve_heuristic_rolling_horizon_lookahead_stochastic(
        START_NODE, time_matrix, rate_stack, loads_stack, distance_arr, diesel_arr,
        MAX_DURATION, NUM_NODES, start_day_idx=DAY,
    )
    assert (r.route, r.reward) == EXPECTED["rh_lookahead_stoch"]

    r = solve_mc_rollout_stochastic(
        START_NODE, time_matrix, rate_stack, loads_stack, distance_arr, diesel_arr,
        MAX_DURATION, NUM_NODES, start_day_idx=DAY,
    )
    assert (r.route, r.reward) == EXPECTED["mc_rollout"]

    r = solve_HGA_LNS_metaheuristic(
        START_NODE, time_matrix, MAX_DURATION, NUM_NODES,
        rate_stack=rate_stack, loads_stack=loads_stack,
        distance_arr=distance_arr, diesel_arr=diesel_arr,
        start_day_idx=DAY, seed=12345,
    )
    assert (r.route, r.reward) == EXPECTED["hga_lns"]

    status, route, reward, duration = solve_label_setting_exact(
        START_NODE, time_matrix_np, rate_stack, loads_stack,
        distance_arr, diesel_arr, MAX_DURATION, NUM_NODES,
        start_day_idx=DAY, time_limit_seconds=60,
    )
    assert (route, reward) == EXPECTED["ls_exact"]

    status, route, reward, duration = solve_label_setting_oracle(
        START_NODE, time_matrix_np, rate_stack, loads_stack,
        distance_arr, diesel_arr, MAX_DURATION, NUM_NODES,
        start_day_idx=DAY, time_limit_seconds=60,
    )
    assert (route, reward) == EXPECTED["ls_oracle"]
