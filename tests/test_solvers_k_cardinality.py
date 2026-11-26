"""
test_solvers_k_cardinality.py
===============================
Criterio de aceptación: la DP hacia atrás de UB-PI (único solver que conserva baselines/ub_pi.py)
honra el presupuesto de cardinalidad K (≤ K arcos, incluido el retorno) por construcción
— la dimensión k del estado —.

Usa un escenario sintético (no datos reales de 10 nodos) con tiempos de viaje
pequeños y max_duration holgado, para que K sea la restricción BINDING — con
los datos reales, MAX_DURATION ya limita las rutas a pocos arcos
independientemente de K (ver tests/test_greedy_k_conformity.py,
mismo hallazgo del lado DRL).
"""

import numpy as np
import pytest

from config import K_LEVELS
from baselines.ub_pi import solve_ub_pi

N = 9   # el camión puede revisitar nodos, así que K puede ser binding con cualquier N.


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
    """Arcos recorridos: pares consecutivos distintos (una espera aparece como el nodo repetido)."""
    return None if route is None else sum(a != b for a, b in zip(route[:-1], route[1:]))


def test_ub_pi_respects_k(synthetic_data):
    d = synthetic_data
    time_matrix_np = np.asarray(d["time_matrix"], dtype=float)
    for k in K_LEVELS:
        valor, route = solve_ub_pi(
            0, time_matrix_np, d["rate_stack"], d["loads_stack"],
            d["distance_arr"], d["diesel_arr"], d["max_duration"],
            start_day_idx=0, k_max=k,
        )
        n = _n_arcs(route)
        assert n == k, f"UB-PI K={k}: solución óptima debería usar exactamente K arcos (n_arcos={n})"
        assert route[0] == route[-1] == 0
