"""
test_post_value_training.py
===========================
Fase P1 de PRICING_PLAN.md: loss de Ṽ(s, j) dentro del loop PPO.
"""

import numpy as np
import pytest

import am_training
from am_training import RolloutBuffer
from baselines import calcular_ev_esperado
from market_process import MarketProcess, asegurar_parametros
from problem_data import load_matrices

NUM_NODES = 10


def test_target_post_is_return_minus_reward_and_zero_at_depot():
    """Episodio de 3 pasos: el último (retorno al depot) tiene target_post = 0."""
    buf = RolloutBuffer()
    rewards = [0.8, -0.3, 1.1]
    values  = [1.0, 0.7, 0.4]
    for t, (r, v) in enumerate(zip(rewards, values)):
        buf.add(np.zeros((NUM_NODES, 12), np.float32), np.zeros(3, np.float32), 0,
                np.ones(NUM_NODES, np.int8), 0, r, 0.0, v, t == 2, is_return=(t == 2))
    buf.episode_boundaries.append((0, 3))
    buf.compute_gae_per_episode(gamma=1.0, gae_lambda=0.95)
    assert buf.target_post[2] == pytest.approx(0.0, abs=1e-6)
    np.testing.assert_allclose(buf.target_post, buf.returns - np.asarray(rewards, np.float32), atol=1e-6)
    # con gamma=1 y GAE: G_0 − r_0 es lo que queda por cobrar tras la primera lane
    assert buf.target_post[0] == pytest.approx(buf.returns[0] - 0.8, abs=1e-6)
    assert buf.is_return == [False, False, True]


@pytest.fixture(scope="module")
def data():
    asegurar_parametros()
    (time_matrix, rate_stack, loads_stack, distance_arr, diesel_arr,
     ltr_stack, trucks_stack, avail_expected_arr, reward_global_p95) = load_matrices(NUM_NODES)
    ev = calcular_ev_esperado(MarketProcess(NUM_NODES), distance_arr, diesel_arr)
    return dict(
        args=(time_matrix, rate_stack, loads_stack, distance_arr, diesel_arr, NUM_NODES),
        kwargs=dict(ltr_stack=ltr_stack, trucks_stack=trucks_stack,
                    reward_global_p95=reward_global_p95, ev_esperado=ev),
    )


def _run(monkeypatch, data, use_post, n_updates, eps_per_update):
    monkeypatch.setattr(am_training, "PPO_N_EPISODES_PER_UPDATE", eps_per_update)
    monkeypatch.setattr(am_training, "get_episodes_per_node", lambda n: n_updates * eps_per_update // n)
    monkeypatch.setattr(am_training, "USE_POST_HEAD", use_post)
    agent, critic, _, _, log = am_training.run_am_training(*data["args"], **data["kwargs"])
    return agent, log


def test_smoke_post_mse_finite_and_decreases(monkeypatch, data):
    agent, log = _run(monkeypatch, data, True, n_updates=8, eps_per_update=140)
    assert agent.post_value is not None
    mse = [e["post_mse"] for e in log]
    assert all(np.isfinite(mse)) and all(np.isfinite([e["post_ev"] for e in log]))
    assert mse[-1] < mse[0], f"post_mse no decrece: {mse}"


def test_policy_reward_identical_with_and_without_post_head(monkeypatch, data):
    """El detach garantiza que la política no cambia: mismo reward con la misma semilla."""
    _, log_off = _run(monkeypatch, data, False, n_updates=3, eps_per_update=70)
    _, log_on  = _run(monkeypatch, data, True,  n_updates=3, eps_per_update=70)
    r_off = [e["reward"] for e in log_off]
    r_on  = [e["reward"] for e in log_on]
    assert "post_mse" not in log_off[0] and "post_mse" in log_on[0]
    np.testing.assert_allclose(r_on, r_off, rtol=0, atol=1e-6)
    np.testing.assert_allclose([e["policy_loss"] for e in log_on], [e["policy_loss"] for e in log_off], atol=1e-6)
