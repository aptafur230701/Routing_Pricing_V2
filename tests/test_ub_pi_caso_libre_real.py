"""
test_ub_pi_caso_libre_real.py
=============================
Regresión del caso libre (k_max=None, el default) de UB-PI sobre datos reales de 10 nodos
(start_node=0, day=0): la ruta y el margen son los del óptimo con el día conocido.

El valor esperado se capturó con UB-PI (baselines.ub_pi.solve_ub_pi, label-setting sin
revisitar) sobre los datos de jun–oct 2026 (datos/duration.csv no cambió, pero rate/loads sí:
ver metodologia_realizaciones_mercado_reefer.md). Si datos/duration.csv o los CSV diarios
cambian hay que recapturarlo con ese mismo solver. UB-PI es el único solver que conserva
baselines/ub_pi.py; los baselines viven en baselines/politicas.py y baselines/dp_relajada.py.
"""

import numpy as np
import pytest

from config import MAX_DURATION
from problem_data import load_matrices
from baselines.ub_pi import solve_ub_pi

NUM_NODES = 10
START_NODE = 0
DAY = 0

# route, reward
EXPECTED_UB_PI = ([0, 6, 0], 1328.0)


@pytest.fixture(scope="module")
def data():
    return load_matrices(NUM_NODES)


def test_ub_pi_caso_libre_coincide_con_el_optimo_de_referencia(data):
    (time_matrix, rate_stack, loads_stack, distance_arr, diesel_arr,
     ltr_stack, avail_expected_arr, reward_global_p95) = data
    time_matrix_np = np.array(time_matrix, dtype=float)

    reward, route = solve_ub_pi(
        START_NODE, time_matrix_np, rate_stack, loads_stack,
        distance_arr, diesel_arr, MAX_DURATION, start_day_idx=DAY,
    )
    assert (route, reward) == EXPECTED_UB_PI
