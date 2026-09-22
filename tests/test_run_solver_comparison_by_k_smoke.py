"""
test_run_solver_comparison_by_k_smoke.py
==========================================
Fase evaluation.py: smoke test de run_solver_comparison_by_k (función nueva,
opt-in) y verificación de que la refactorización de run_solver_comparison
(extracción de _run_comparison_cell) no cambió su salida ni su comportamiento
en el caso libre.

Usa datos reales pequeños (num_nodes=5) y un agente de pesos aleatorios —
suficiente para smoke-testing estructural (columnas, conformidad de K), no
evalúa calidad de las rutas.
"""

import numpy as np
import pytest
import torch

from config import DEVICE
from am_agent import AMRoutingAgent
from problem_data import load_matrices
from evaluation import run_solver_comparison, run_solver_comparison_by_k

NUM_NODES = 5

_ROUTE_COLUMNS = [
    'DRL Real Route', 'DRL Det Route', 'RH-Greedy Route', 'RH-Lookahead Route',
    'RH-Greedy Real Route', 'RH-Lookahead Real Route', 'MC-Rollout Route',
    'HGA-LNS Route', 'LS-Exact Route', 'LS-Oracle Route',
]


@pytest.fixture(scope="module")
def agent_and_data():
    torch.manual_seed(0)
    agent = AMRoutingAgent(num_nodes=NUM_NODES, d_h=32, n_heads=4, n_layers=1, d_ff=64, device=DEVICE)
    data = load_matrices(NUM_NODES)
    return agent, data


def test_run_solver_comparison_free_case_unchanged_shape(agent_and_data):
    """La refactorización (extracción de _run_comparison_cell) no debe
    cambiar la interfaz ni las columnas de run_solver_comparison."""
    agent, data = agent_and_data
    (time_matrix, rate_stack, loads_stack, distance_arr, diesel_arr,
     ltr_stack, trucks_stack, avail_prob_arr, reward_global_p95) = data

    df, timing = run_solver_comparison(
        agent, time_matrix, rate_stack, loads_stack, distance_arr, diesel_arr, NUM_NODES,
        ltr_stack=ltr_stack, trucks_stack=trucks_stack, avail_prob_arr=avail_prob_arr,
        n_days_per_node=1,
    )
    assert len(df) == NUM_NODES   # 1 rep por nodo
    assert 'K' not in df.columns   # caso libre: sin columna K
    for col in _ROUTE_COLUMNS:
        assert col in df.columns
    expected_timing_keys = {
        'drl_real_times', 'drl_det_times', 'hga_lns_times', 'ls_exact_times',
        'rh_greedy_times', 'rh_lookahead_times', 'rh_stoch_times',
        'rh_lookahead_stoch_times', 'mc_rollout_times', 'ls_oracle_times',
    }
    assert set(timing.keys()) == expected_timing_keys
    for key in expected_timing_keys:
        assert len(timing[key]) == NUM_NODES


def test_run_solver_comparison_by_k_smoke(agent_and_data):
    agent, data = agent_and_data
    (time_matrix, rate_stack, loads_stack, distance_arr, diesel_arr,
     ltr_stack, trucks_stack, avail_prob_arr, reward_global_p95) = data

    k_levels = [4, 5]
    df, timing = run_solver_comparison_by_k(
        agent, time_matrix, rate_stack, loads_stack, distance_arr, diesel_arr, NUM_NODES,
        k_levels=k_levels,
        ltr_stack=ltr_stack, trucks_stack=trucks_stack, avail_prob_arr=avail_prob_arr,
        n_days_per_node=1,
    )
    assert 'K' in df.columns
    assert set(df['K'].unique()) == set(k_levels)
    assert len(df) == NUM_NODES * len(k_levels)

    # Conformidad: toda ruta válida devuelta por cualquier solver debe
    # respetar su K (≤ K arcos, incluido el retorno).
    for col in _ROUTE_COLUMNS:
        for _, r in df.iterrows():
            route = r[col]
            if route is not None:
                n_arcs = len(route) - 1
                assert n_arcs <= r['K'], f"{col}: {route} tiene {n_arcs} arcos > K={r['K']}"
