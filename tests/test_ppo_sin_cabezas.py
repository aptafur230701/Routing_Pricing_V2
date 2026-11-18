"""
test_ppo_sin_cabezas.py
=======================
El PPO no entrena Ṽ: la pérdida de la cabeza salió del entrenamiento. Con un agente que trae las cabezas (transfer)
el PPO no las toca, y el log ya no trae post_mse / post_ev.
"""

import numpy as np
import pytest
import torch

import am_training
from model.agent import AMRoutingAgent
from baselines.politicas import calcular_ev_pronostico
from config import AM_D_H, AM_N_HEADS, AM_N_LAYERS, AM_D_FF, DEVICE
from model.critic_head import CriticHead
from market_process import MarketProcess, asegurar_parametros
from problem_data import load_matrices

NUM_NODES = 10


@pytest.fixture(scope="module")
def data():
    asegurar_parametros()
    (time_matrix, rate_stack, loads_stack, distance_arr, diesel_arr,
     ltr_stack, avail_expected_arr, reward_global_p95) = load_matrices(NUM_NODES)
    ev = calcular_ev_pronostico(MarketProcess(NUM_NODES), distance_arr, diesel_arr)
    return dict(
        args=(time_matrix, rate_stack, loads_stack, distance_arr, diesel_arr, NUM_NODES),
        kwargs=dict(ltr_stack=ltr_stack,
                    reward_global_p95=reward_global_p95, ev_esperado=ev),
    )


def _run(monkeypatch, data, n_updates, eps_per_update, **extra):
    monkeypatch.setattr(am_training, "PPO_N_EPISODES_PER_UPDATE", eps_per_update)
    monkeypatch.setattr(am_training, "get_episodes_per_node", lambda n: n_updates * eps_per_update // n)
    return am_training.run_am_training(*data["args"], **data["kwargs"], **extra)


def test_ppo_agent_has_no_heads_and_log_has_no_post_metrics(monkeypatch, data):
    agent, _, _, _, log = _run(monkeypatch, data, n_updates=2, eps_per_update=70)
    assert agent.post_value_a is None and agent.post_value_b is None
    assert not any(k.startswith("post_value") for k in agent.state_dict())
    assert all("post_mse" not in e and "post_ev" not in e for e in log)
    assert all(np.isfinite(e["reward"]) for e in log)


def test_ppo_leaves_the_heads_of_a_pretrained_agent_untouched(monkeypatch, data):
    agent0 = AMRoutingAgent(NUM_NODES, AM_D_H, AM_N_HEADS, AM_N_LAYERS, AM_D_FF, device=DEVICE, use_post_head=True)
    critic0 = CriticHead(AM_D_H).to(DEVICE)
    antes = {k: v.clone() for k, v in agent0.state_dict().items() if k.startswith("post_value")}
    assert antes
    agent, _, _, _, _ = _run(monkeypatch, data, n_updates=2, eps_per_update=70,
                             pretrained_agent=agent0, pretrained_critic=critic0)
    for k, v in agent.state_dict().items():
        if k.startswith("post_value"):
            assert torch.equal(v, antes[k]), k
        else:
            assert k in agent0.state_dict()
