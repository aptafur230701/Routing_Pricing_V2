"""
test_greedy_k_conformity.py
===========================
El rollout greedy paso a paso sobre el entorno (greedy_rollout) honra el
presupuesto de cardinalidad K (≤ K arcos, incluido el retorno), tanto en el
caso restringido como en el libre, y reporta la recompensa en las unidades de
los solvers.

Usa un escenario sintético (no los datos reales de 10 nodos) con tiempos de
viaje pequeños y max_duration holgado, para que K sea la restricción
BINDING — con los datos reales a 10 nodos, MAX_DURATION ya limita las rutas
a 2-3 arcos independientemente de K, lo que no ejercitaría el camino de
retorno forzado por K. Un agente de pesos aleatorios es suficiente: solo se
verifica conformidad estructural (impuesta por máscara), no calidad de ruta.
"""

import numpy as np
import pytest
import torch

from config import DEVICE, K_LEVELS
from am_agent import AMRoutingAgent
from greedy_rollout import greedy_rollout
from problem_data import build_rm_pen_stack
from routing_env import RoutingEnv, VectorRoutingEnv

N = 9   # > max(K_LEVELS): un ciclo puede visitar hasta N-1 intermedios + retorno = N arcos,
        # así que todos los niveles de K_LEVELS son físicamente alcanzables.
MAX_DURATION_SYN = 30.0   # con time_matrix=1.0/arco, permite hasta 30 arcos — K es el binding


@pytest.fixture(scope="module")
def synthetic():
    time_matrix  = np.ones((N, N))
    np.fill_diagonal(time_matrix, 0.0)
    distance_arr = np.ones((N, N)) * 10.0
    np.fill_diagonal(distance_arr, 0.0)
    diesel_arr   = np.ones((N, N)) * 3.0
    rate_stack   = np.ones((3, N, N)) * 5.0
    loads_stack  = np.ones((3, N, N)) * 5.0
    trucks_stack = np.ones((N, 3, 3), dtype=np.float32) * 2.0
    ltr_stack    = np.ones((N, 3), dtype=np.float32) * 10.0
    rm_pen_stack = build_rm_pen_stack(rate_stack, loads_stack, distance_arr, diesel_arr)
    return dict(
        time_matrix=time_matrix, rate_stack=rate_stack, loads_stack=loads_stack,
        distance_arr=distance_arr, diesel_arr=diesel_arr,
        trucks_stack=trucks_stack, ltr_stack=ltr_stack, rm_pen_stack=rm_pen_stack,
    )


@pytest.fixture
def env(synthetic):
    return VectorRoutingEnv(
        time_matrix=synthetic["time_matrix"], rate_stack=synthetic["rate_stack"],
        loads_stack=synthetic["loads_stack"], rm_pen_stack=synthetic["rm_pen_stack"],
        num_nodes=N, max_duration=MAX_DURATION_SYN,
    )


@pytest.fixture(scope="module")
def agent():
    torch.manual_seed(0)
    return AMRoutingAgent(num_nodes=N, d_h=32, n_heads=4, n_layers=1, d_ff=64, device=DEVICE)


def _run(agent, env, synthetic, k_max):
    """Un episodio por nodo de inicio, todos con el mismo presupuesto k_max."""
    starts = np.arange(N, dtype=np.int64)
    return greedy_rollout(
        agent, env, starts, np.zeros(N, dtype=np.int64),
        synthetic["time_matrix"], synthetic["distance_arr"],
        synthetic["trucks_stack"], synthetic["ltr_stack"],
        reward_global_p95=1.0, num_nodes=N,
        k_max_arr=[k_max] * N, max_duration=MAX_DURATION_SYN,
    )


def _never_return_voluntarily(start_nodes):
    """Política greedy que evita el depot mientras haya otra acción válida —
    obliga a que sea la máscara de K la que cierre el ciclo."""
    starts = torch.as_tensor(start_nodes, device=DEVICE)

    def pick(h_t, embeddings, mask_t):
        allowed = mask_t.bool().clone()
        rows = torch.arange(allowed.shape[0], device=allowed.device)
        non_depot = allowed.clone()
        non_depot[rows, starts] = False
        use = non_depot.any(dim=1, keepdim=True)
        return torch.where(use, non_depot, allowed).float().argmax(dim=1)

    return pick


def test_greedy_rollout_respects_k(agent, env, synthetic):
    for k in K_LEVELS:
        res = _run(agent, env, synthetic, k)
        for start, route, ok in zip(range(N), res.routes, res.valid):
            if ok:
                assert len(route) - 1 <= k, f"K={k} start={start}: ruta con {len(route) - 1} arcos > K"


def test_greedy_rollout_k_is_binding_when_policy_never_returns_early(agent, env, synthetic, monkeypatch):
    monkeypatch.setattr(agent.decoder, "greedy_action", _never_return_voluntarily(np.arange(N)))
    for k in K_LEVELS:
        res = _run(agent, env, synthetic, k)
        assert res.valid.all(), f"K={k}: rutas inválidas {np.where(~res.valid)[0]}"
        lens = [len(r) - 1 for r in res.routes]
        assert lens == [k] * N, f"K={k}: el retorno forzado debe cerrar en exactamente K arcos, got {lens}"


def test_greedy_rollout_free_case_closes_cycle(agent, env, synthetic):
    res = _run(agent, env, synthetic, None)
    assert res.valid.all()
    for start, route in zip(range(N), res.routes):
        assert route[0] == start and route[-1] == start


def test_greedy_rollout_raw_reward_is_sum_of_lane_rewards(agent, env, synthetic):
    """raw_reward está en unidades de los solvers: suma de rm_pen de las lanes
    recorridas, sin escala ni bonos terminales."""
    res = _run(agent, env, synthetic, None)
    rm = synthetic["rm_pen_stack"][0]
    for route, raw in zip(res.routes, res.raw_reward):
        expected = sum(float(rm[a, b]) for a, b in zip(route[:-1], route[1:]))
        assert raw == pytest.approx(expected)


def test_greedy_route_replays_in_scalar_routing_env(agent, env, synthetic):
    """La ruta del rollout vectorizado es válida paso a paso en RoutingEnv (Gymnasium):
    cada acción está en la máscara y el último paso termina el episodio."""
    res = _run(agent, env, synthetic, 3)
    for start, route, ok in zip(range(N), res.routes, res.valid):
        assert ok
        scalar = RoutingEnv(
            synthetic["time_matrix"], synthetic["rate_stack"], synthetic["loads_stack"],
            synthetic["distance_arr"], synthetic["diesel_arr"], N, max_duration=MAX_DURATION_SYN,
        )
        _, info = scalar.reset(options={"start_node": start, "start_day_idx": 0, "k_max": 3})
        for i, action in enumerate(route[1:]):
            assert info["action_mask"][action] == 1, f"start={start}: acción {action} fuera de máscara"
            _, _, terminated, _, info = scalar.step(action)
            assert terminated == (i == len(route) - 2)
