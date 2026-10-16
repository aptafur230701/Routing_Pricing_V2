"""
test_k_cardinality_env.py
==========================
Criterio de aceptación obligatorio del condicionamiento en cardinalidad de ruta (K):

    Con una política ALEATORIA, el 100% de los episodios en todos los niveles
    de K cumplen longitud (nº de arcos) <= K. Debe ser cierto por construcción
    de la máscara, nunca por penalización de reward.

Cubre tanto RoutingEnv (escalar) como VectorRoutingEnv, y el caso libre
(k_max=None) para verificar que no se ve afectado.
"""

import numpy as np
import pytest

from config import MAX_DURATION, K_LEVELS, K_TRAINING_LEVELS
from problem_data import load_matrices, build_rm_pen_stack
from routing_env import RoutingEnv, VectorRoutingEnv

NUM_NODES = 10
N_TRIALS_PER_LEVEL = 60   # episodios por nivel de K en el test escalar


@pytest.fixture(scope="module")
def data():
    (time_matrix, rate_stack, loads_stack, distance_arr, diesel_arr,
     ltr_stack, trucks_stack, avail_expected_arr, reward_global_p95) = load_matrices(NUM_NODES)
    return dict(
        time_matrix=time_matrix, rate_stack=rate_stack, loads_stack=loads_stack,
        distance_arr=distance_arr, diesel_arr=diesel_arr,
    )


def _random_valid_action(mask: np.ndarray, rng: np.random.RandomState) -> int:
    valid = np.where(mask == 1)[0]
    assert len(valid) > 0, "mask sin acciones válidas (condición B no debería activarse aquí)"
    return int(rng.choice(valid))


# ─────────────────────────────────────────────────────────────────────────────
# RoutingEnv (escalar)
# ─────────────────────────────────────────────────────────────────────────────

def test_scalar_env_random_policy_respects_k(data):
    env = RoutingEnv(
        time_matrix=data["time_matrix"],
        rate_stack=data["rate_stack"],
        loads_stack=data["loads_stack"],
        distance_arr=data["distance_arr"],
        diesel_arr=data["diesel_arr"],
        num_nodes=NUM_NODES,
        max_duration=MAX_DURATION,
    )
    rng = np.random.RandomState(0)

    for k_max in K_TRAINING_LEVELS:
        for trial in range(N_TRIALS_PER_LEVEL):
            start_node = int(rng.randint(0, NUM_NODES))
            options = {"start_node": start_node, "k_max": k_max}
            obs, info = env.reset(seed=int(rng.randint(0, 1_000_000)), options=options)

            assert info["k_restricted"] == (k_max is not None)
            if k_max is None:
                assert info["k_remaining"] is None

            n_arcs = 0
            terminated = False
            for _ in range(NUM_NODES + 2):   # techo defensivo
                action = _random_valid_action(info["action_mask"], rng)
                obs, reward, terminated, truncated, info = env.step(action)
                n_arcs += 1
                assert not truncated
                if terminated:
                    break

            assert terminated, "el episodio debe terminar dentro del techo defensivo"
            if k_max is not None:
                assert n_arcs <= k_max, (
                    f"K={k_max}: episodio terminó con {n_arcs} arcos (> K)"
                )
                assert env.k_remaining == k_max - n_arcs
                assert env.k_remaining >= 0


def test_scalar_env_backward_compatible_when_k_max_absent(data):
    """Sin 'k_max' en options, el comportamiento es el caso libre (compat. hacia atrás)."""
    env = RoutingEnv(
        time_matrix=data["time_matrix"],
        rate_stack=data["rate_stack"],
        loads_stack=data["loads_stack"],
        distance_arr=data["distance_arr"],
        diesel_arr=data["diesel_arr"],
        num_nodes=NUM_NODES,
        max_duration=MAX_DURATION,
    )
    obs, info = env.reset(seed=0, options={"start_node": 0})
    assert info["k_restricted"] is False
    assert info["k_remaining"] is None
    assert env.k_restricted is False


# ─────────────────────────────────────────────────────────────────────────────
# VectorRoutingEnv
# ─────────────────────────────────────────────────────────────────────────────

def test_vector_env_random_policy_respects_k(data):
    rm_pen_stack = build_rm_pen_stack(
        data["rate_stack"], data["loads_stack"], data["distance_arr"], data["diesel_arr"],
    )
    vec_env = VectorRoutingEnv(
        time_matrix=data["time_matrix"],
        rate_stack=data["rate_stack"],
        loads_stack=data["loads_stack"],
        rm_pen_stack=rm_pen_stack,
        num_nodes=NUM_NODES,
        max_duration=MAX_DURATION,
    )

    rng = np.random.RandomState(1)
    B = 600
    n_levels = len(K_TRAINING_LEVELS)
    per_level = B // n_levels
    k_max_arr = []
    for lvl in K_TRAINING_LEVELS:
        k_max_arr += [lvl] * per_level
    while len(k_max_arr) < B:
        k_max_arr.append(K_TRAINING_LEVELS[rng.randint(0, n_levels)])
    perm = rng.permutation(B)
    k_max_arr = [k_max_arr[i] for i in perm]

    start_nodes = rng.randint(0, NUM_NODES, size=B).astype(np.int64)
    start_days  = np.zeros(B, dtype=np.int64)

    masks = vec_env.reset(start_nodes, start_days, k_max_arr=k_max_arr)
    active = np.ones(B, dtype=bool)

    for _ in range(NUM_NODES + 2):
        if not active.any():
            break
        actions = np.zeros(B, dtype=np.int64)
        for b in np.where(active)[0]:
            actions[b] = _random_valid_action(masks[b], rng)
        rewards, terminated, next_masks = vec_env.step(actions)
        active = active & ~terminated
        masks = next_masks

    assert not active.any(), "todos los episodios deben terminar dentro del techo defensivo"

    step_counts = vec_env.step_count
    for b in range(B):
        k_max = k_max_arr[b]
        if k_max is not None:
            assert step_counts[b] <= k_max, (
                f"fila {b}: K={k_max} pero step_count={step_counts[b]}"
            )

    # Cada nivel restringido debe haber aparecido (balanceo aproximado).
    for lvl in K_LEVELS:
        assert any(k == lvl for k in k_max_arr)
