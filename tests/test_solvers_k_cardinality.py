"""
test_solvers_k_cardinality.py
===============================
Criterio de aceptación: el label-setting oráculo (único solver que conserva Solvers.py)
honra el presupuesto de cardinalidad K (≤ K arcos, incluido el retorno) por construcción
— filtro de expansión en el DP —, con y sin poda por dominancia.

Usa un escenario sintético (no datos reales de 10 nodos) con tiempos de viaje
pequeños y max_duration holgado, para que K sea la restricción BINDING — con
los datos reales, MAX_DURATION ya limita las rutas a pocos arcos
independientemente de K (ver tests/test_greedy_k_conformity.py,
mismo hallazgo del lado DRL).
"""

import numpy as np
import pytest

from config import K_LEVELS
from Solvers import solve_label_setting_oracle

N = 9   # > max(K_LEVELS): un ciclo puede alcanzar hasta N arcos, así que todos
        # los niveles de K_LEVELS son físicamente alcanzables.


@pytest.fixture(scope="module")
def synthetic_data():
    time_matrix  = np.ones((N, N))
    np.fill_diagonal(time_matrix, 0.0)
    distance_arr = np.ones((N, N)) * 10.0
    np.fill_diagonal(distance_arr, 0.0)
    diesel_arr   = np.ones((N, N)) * 3.0
    rate_stack   = np.ones((3, N, N)) * 5.0
    loads_stack  = np.ones((3, N, N)) * 5.0
    max_duration = 30.0   # con time_matrix=1.0/arco, permite hasta 30 arcos
    return dict(
        time_matrix=time_matrix, rate_stack=rate_stack, loads_stack=loads_stack,
        distance_arr=distance_arr, diesel_arr=diesel_arr,
        max_duration=max_duration,
    )


def _n_arcs(route):
    return None if route is None else len(route) - 1


@pytest.mark.parametrize("dominancia", [True, False])
def test_label_setting_oracle_respects_k(synthetic_data, dominancia):
    d = synthetic_data
    time_matrix_np = np.asarray(d["time_matrix"], dtype=float)
    for k in K_LEVELS:
        status, route, reward, duration = solve_label_setting_oracle(
            0, time_matrix_np, d["rate_stack"], d["loads_stack"],
            d["distance_arr"], d["diesel_arr"], d["max_duration"], N,
            start_day_idx=0, time_limit_seconds=30, k_max=k, dominancia=dominancia,
        )
        n = _n_arcs(route)
        assert n == k, f"LS-Oracle K={k}: solución óptima debería usar exactamente K arcos (n_arcos={n})"
