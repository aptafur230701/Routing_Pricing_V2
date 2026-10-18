"""
test_label_setting_free_case_equivalence.py
==============================================
Criterio de aceptación: el label-setting oráculo, en caso libre (k_max=None, el default),
reproduce EXACTAMENTE el mismo resultado que antes de añadir soporte de K.

El valor esperado se capturó ejecutando el solver sobre datos reales de 10 nodos
(start_node=0, day=0) con el código previo al cambio y se hardcodea aquí como regresión
permanente. Es el único solver que conserva Solvers.py (los baselines viven en
baselines.py y dp_relajada.py).
"""

import numpy as np
import pytest

from config import MAX_DURATION
from problem_data import load_matrices
from Solvers import solve_label_setting_oracle

NUM_NODES = 10
START_NODE = 0
DAY = 0

# route, reward
EXPECTED_LS_ORACLE = ([0, 6, 1, 0], 1490.0)


@pytest.fixture(scope="module")
def data():
    return load_matrices(NUM_NODES)


def test_free_case_matches_pre_k_conditioning_baseline(data):
    (time_matrix, rate_stack, loads_stack, distance_arr, diesel_arr,
     ltr_stack, trucks_stack, avail_expected_arr, reward_global_p95) = data
    time_matrix_np = np.array(time_matrix, dtype=float)

    status, route, reward, duration = solve_label_setting_oracle(
        START_NODE, time_matrix_np, rate_stack, loads_stack,
        distance_arr, diesel_arr, MAX_DURATION, NUM_NODES,
        start_day_idx=DAY, time_limit_seconds=60,
    )
    assert (route, reward) == EXPECTED_LS_ORACLE
