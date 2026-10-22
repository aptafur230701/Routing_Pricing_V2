"""
test_ub_pi_caso_libre_real.py
=============================
Regresión del caso libre (k_max=None, el default) de UB-PI sobre datos reales de 10 nodos
(start_node=0, day=0): la ruta y el margen son los del óptimo con el día conocido.

El valor esperado se capturó con el solver anterior (label-setting sin revisitar, git tag
pre-revisit) sobre los tiempos de datos/duration.csv; coincide con UB-PI en 80 casos reales de
10 nodos (10 nodos de inicio × 4 días × K = 5 y libre). Si datos/duration.csv cambia hay que
recapturarlo con ese solver. UB-PI (Solvers.solve_ub_pi) es el único solver que conserva
Solvers.py; los baselines viven en baselines.py y dp_relajada.py.
"""

import numpy as np
import pytest

from config import MAX_DURATION
from problem_data import load_matrices
from Solvers import solve_ub_pi

NUM_NODES = 10
START_NODE = 0
DAY = 0

# route, reward
EXPECTED_UB_PI = ([0, 1, 0], 1106.0)


@pytest.fixture(scope="module")
def data():
    return load_matrices(NUM_NODES)


def test_ub_pi_caso_libre_coincide_con_el_optimo_de_referencia(data):
    (time_matrix, rate_stack, loads_stack, distance_arr, diesel_arr,
     ltr_stack, trucks_stack, avail_expected_arr, reward_global_p95) = data
    time_matrix_np = np.array(time_matrix, dtype=float)

    reward, route = solve_ub_pi(
        START_NODE, time_matrix_np, rate_stack, loads_stack,
        distance_arr, diesel_arr, MAX_DURATION, start_day_idx=DAY,
    )
    assert (route, reward) == EXPECTED_UB_PI
