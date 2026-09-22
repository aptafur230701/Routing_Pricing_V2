"""
test_solvers_k_cardinality.py
===============================
Criterio de aceptación: los 8 solvers activos de Solvers.py (los que
evaluation.run_solver_comparison realmente llama) honran el presupuesto de
cardinalidad K (≤ K arcos, incluido el retorno) por construcción — filtro de
expansión en el DP de Label-Setting, cap en el loop de pasos en la familia
Rolling-Horizon / MC-Rollout, cap en max_route_len en la familia HGA-LNS.

Usa un escenario sintético (no datos reales de 10 nodos) con tiempos de viaje
pequeños y max_duration holgado, para que K sea la restricción BINDING — con
los datos reales, MAX_DURATION ya limita las rutas a pocos arcos
independientemente de K (ver tests/test_beam_and_sampling_k_conformity.py,
mismo hallazgo del lado DRL).
"""

import numpy as np
import pytest

from config import K_LEVELS
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

N = 9   # > K_MAX (8): un ciclo puede alcanzar hasta N arcos, así que todos
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
    avail_prob_arr = np.ones((N, N), dtype=np.float32)   # lanes siempre existen
    max_duration = 30.0   # con time_matrix=1.0/arco, permite hasta 30 arcos
    return dict(
        time_matrix=time_matrix, rate_stack=rate_stack, loads_stack=loads_stack,
        distance_arr=distance_arr, diesel_arr=diesel_arr,
        avail_prob_arr=avail_prob_arr, max_duration=max_duration,
    )


def _n_arcs(route):
    return None if route is None else len(route) - 1


# ─────────────────────────────────────────────────────────────────────────────
# Familia Rolling-Horizon / MC-Rollout (RHResult: status, route, reward, duration, is_valid)
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("solver_name,fn,extra_kwargs", [
    ("RH-Greedy", solve_heuristic_rolling_horizon, {}),
    ("RH-Lookahead", solve_heuristic_rolling_horizon_lookahead, {}),
])
def test_rh_deterministic_respects_k(synthetic_data, solver_name, fn, extra_kwargs):
    d = synthetic_data
    for k in K_LEVELS:
        status, route, reward, duration, is_valid = fn(
            0, d["time_matrix"], d["rate_stack"], d["loads_stack"],
            d["distance_arr"], d["diesel_arr"], d["max_duration"], N,
            start_day_idx=0, k_max=k, **extra_kwargs,
        )
        n = _n_arcs(route)
        if n is not None:
            assert n <= k, f"{solver_name} K={k}: {n} arcos > K"
        assert n == k, f"{solver_name} K={k}: K debería ser binding (n_arcos={n})"


@pytest.mark.parametrize("solver_name,fn", [
    ("RH-Greedy-Stoch", solve_heuristic_rolling_horizon_stochastic),
    ("RH-Lookahead-Stoch", solve_heuristic_rolling_horizon_lookahead_stochastic),
    ("MC-Rollout", solve_mc_rollout_stochastic),
])
def test_rh_stochastic_respects_k(synthetic_data, solver_name, fn):
    d = synthetic_data
    for k in K_LEVELS:
        status, route, reward, duration, is_valid = fn(
            0, d["time_matrix"], d["rate_stack"], d["loads_stack"],
            d["distance_arr"], d["diesel_arr"], d["max_duration"], N,
            start_day_idx=0, avail_prob_arr=d["avail_prob_arr"], k_max=k,
        )
        n = _n_arcs(route)
        if n is not None:
            assert n <= k, f"{solver_name} K={k}: {n} arcos > K"
        assert n == k, f"{solver_name} K={k}: K debería ser binding (n_arcos={n})"


# ─────────────────────────────────────────────────────────────────────────────
# HGA-LNS (MetaResult: status, route, reward, duration)
# ─────────────────────────────────────────────────────────────────────────────

def test_hga_lns_respects_k(synthetic_data):
    d = synthetic_data
    for k in K_LEVELS:
        status, route, reward, duration = solve_HGA_LNS_metaheuristic(
            0, d["time_matrix"], d["max_duration"], N,
            rate_stack=d["rate_stack"], loads_stack=d["loads_stack"],
            distance_arr=d["distance_arr"], diesel_arr=d["diesel_arr"],
            start_day_idx=0, seed=42, k_max=k,
        )
        n = _n_arcs(route)
        if n is not None:
            assert n <= k, f"HGA-LNS K={k}: {n} arcos > K"
        assert n == k, f"HGA-LNS K={k}: K debería ser binding (n_arcos={n})"


# ─────────────────────────────────────────────────────────────────────────────
# Label-Setting (oráculo) — (status, route, reward, duration)
# ─────────────────────────────────────────────────────────────────────────────

def test_label_setting_exact_respects_k(synthetic_data):
    d = synthetic_data
    time_matrix_np = np.asarray(d["time_matrix"], dtype=float)
    for k in K_LEVELS:
        status, route, reward, duration = solve_label_setting_exact(
            0, time_matrix_np, d["rate_stack"], d["loads_stack"],
            d["distance_arr"], d["diesel_arr"], d["max_duration"], N,
            start_day_idx=0, time_limit_seconds=30, k_max=k,
        )
        n = _n_arcs(route)
        assert n == k, f"LS-Exact K={k}: solución óptima debería usar exactamente K arcos (n_arcos={n})"


def test_label_setting_oracle_respects_k(synthetic_data):
    d = synthetic_data
    time_matrix_np = np.asarray(d["time_matrix"], dtype=float)
    for k in K_LEVELS:
        status, route, reward, duration = solve_label_setting_oracle(
            0, time_matrix_np, d["rate_stack"], d["loads_stack"],
            d["distance_arr"], d["diesel_arr"], d["max_duration"], N,
            start_day_idx=0, avail_prob_arr=d["avail_prob_arr"], time_limit_seconds=30, k_max=k,
        )
        n = _n_arcs(route)
        assert n == k, f"LS-Oracle K={k}: solución óptima debería usar exactamente K arcos (n_arcos={n})"


# ─────────────────────────────────────────────────────────────────────────────
# Caso libre — no debe verse afectado
# ─────────────────────────────────────────────────────────────────────────────

def test_free_case_unaffected(synthetic_data):
    d = synthetic_data
    status, route, reward, duration, is_valid = solve_heuristic_rolling_horizon(
        0, d["time_matrix"], d["rate_stack"], d["loads_stack"],
        d["distance_arr"], d["diesel_arr"], d["max_duration"], N,
        start_day_idx=0,   # k_max no pasado → default None
    )
    assert route is not None
