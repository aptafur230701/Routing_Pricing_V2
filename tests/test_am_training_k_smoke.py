"""
test_am_training_k_smoke.py
=============================
Fase 3 — §8 paso 3: smoke run corto de run_am_training en NUM_NODES=10 (pocos
updates), verificando que los niveles de K (K_TRAINING_LEVELS: K_MAX
restringido + caso libre) aparecen balanceados en el batch de recolección y
que training_log incluye las claves de diagnóstico nuevas, con conformidad
100% garantizada por máscara.

PPO_N_EPISODES_PER_UPDATE y get_episodes_per_node se monkeypatchean a valores
pequeños para que corra en segundos.
"""

import numpy as np
import pytest

import am_training
from config import K_TRAINING_LEVELS
from problem_data import load_matrices

NUM_NODES = 10


@pytest.fixture
def small_run(monkeypatch):
    monkeypatch.setattr(am_training, "PPO_N_EPISODES_PER_UPDATE", 24)
    monkeypatch.setattr(am_training, "get_episodes_per_node", lambda n: 6)   # 60 ep total, 2 updates

    (time_matrix, rate_stack, loads_stack, distance_arr, diesel_arr,
     ltr_stack, trucks_stack, avail_expected_arr, reward_global_p95) = load_matrices(NUM_NODES)

    agent, critic, ep_rewards, ep_losses, training_log = am_training.run_am_training(
        time_matrix, rate_stack, loads_stack, distance_arr, diesel_arr, NUM_NODES,
        ltr_stack=ltr_stack, trucks_stack=trucks_stack, reward_global_p95=reward_global_p95,
    )
    return training_log


def test_smoke_run_produces_k_diagnostics(small_run):
    training_log = small_run
    assert len(training_log) >= 1
    for entry in training_log:
        assert "k_diagnostics" in entry


def test_smoke_run_all_six_k_levels_represented(small_run):
    training_log = small_run
    keys_seen = set()
    for entry in training_log:
        keys_seen |= set(entry["k_diagnostics"].keys())
    expected_keys = {("free" if lvl is None else str(lvl)) for lvl in K_TRAINING_LEVELS}
    assert keys_seen == expected_keys, f"faltan niveles: {expected_keys - keys_seen}"


def test_smoke_run_conformity_is_always_100_percent(small_run):
    training_log = small_run
    for entry in training_log:
        for key, d in entry["k_diagnostics"].items():
            if d is None or key == "free":
                continue
            assert d["conformity_rate"] == 1.0, (
                f"conformidad < 100% en K={key}: {d['conformity_rate']}"
            )


def test_balanced_k_assignment_helper_is_balanced_and_independent():
    rng_a = np.random.RandomState(0)
    rng_b = np.random.RandomState(0)
    B = 240
    k_arr = am_training._balanced_k_assignment(B, rng_a)
    n_levels = len(K_TRAINING_LEVELS)
    per_level_expected = B // n_levels
    for lvl in K_TRAINING_LEVELS:
        count = sum(1 for k in k_arr if k == lvl)
        assert count >= per_level_expected

    # Con el mismo seed, dos llamadas independientes producen la misma
    # secuencia (determinismo), pero la asignación de K es una permutación
    # independiente de cualquier otra generada con el mismo rng.permutation
    # sobre nodos — no se prueba igualdad cruzada, solo reproducibilidad.
    k_arr_repeat = am_training._balanced_k_assignment(B, rng_b)
    assert k_arr == k_arr_repeat
