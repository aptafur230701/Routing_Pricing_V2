"""
test_ppo_sin_cabezas.py
=======================
Las cabezas Ṽ A/B del precio de reserva se eliminaron (Fase 3 de especificacion_fase0_ruteo_pricing.md):

  · el agente ya no tiene post_value_a / post_value_b ni claves post_value en su state_dict;
  · el PPO corre y su log no trae métricas de Ṽ;
  · un checkpoint viejo que sí trae las cabezas carga igual: sus claves post_value se omiten y el resto
    de los pesos se copia tal cual.
"""

import os

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
from transfer_learning import load_checkpoint_compatible

NUM_NODES = 10
RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


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


def _agente():
    return AMRoutingAgent(NUM_NODES, AM_D_H, AM_N_HEADS, AM_N_LAYERS, AM_D_FF, device=DEVICE)


def test_el_agente_no_tiene_cabezas_post():
    agent = _agente()
    assert not hasattr(agent, "post_value_a") and not hasattr(agent, "post_value_b")
    assert not any(k.startswith("post_value") for k in agent.state_dict())


def test_ppo_corre_y_el_log_no_trae_metricas_de_v_tilde(monkeypatch, data):
    agent, _, _, _, log = _run(monkeypatch, data, n_updates=2, eps_per_update=70)
    assert not any(k.startswith("post_value") for k in agent.state_dict())
    assert all("post_mse" not in e and "post_ev" not in e for e in log)
    assert all(np.isfinite(e["reward"]) for e in log)


def test_checkpoint_viejo_con_cabezas_carga_sin_ellas(tmp_path):
    torch.manual_seed(0)
    agent, critic = _agente(), CriticHead(AM_D_H).to(DEVICE)
    sd = dict(agent.state_dict())
    sd["post_value_a.net.0.weight"] = torch.zeros(AM_D_H, 2 * AM_D_H)                 # como los *_pricing.pt viejos
    sd["post_value_b.net.0.weight"] = torch.zeros(AM_D_H, 2 * AM_D_H)
    ruta = tmp_path / "viejo.pt"
    torch.save({"agent": sd, "critic": critic.state_dict()}, ruta)
    cargado, _ = load_checkpoint_compatible(str(ruta), NUM_NODES, AM_D_H, AM_N_HEADS, AM_N_LAYERS, AM_D_FF,
                                            verbose=False)
    for k, v in agent.state_dict().items():
        assert torch.equal(cargado.state_dict()[k], v), k


def test_checkpoint_anterior_a_la_cotizacion_es_error(tmp_path):
    """Sin la cabeza de precio el checkpoint es del modelo viejo: cargarlo (o usarlo para el transfer) dejaría la
    cabeza aleatoria sin aviso, así que debe fallar."""
    agent, critic = _agente(), CriticHead(AM_D_H).to(DEVICE)
    sd = {k: v for k, v in agent.state_dict().items() if not k.startswith("decoder.precio.")}
    ruta = tmp_path / "sin_precio.pt"
    torch.save({"agent": sd, "critic": critic.state_dict()}, ruta)
    with pytest.raises(ValueError, match="anterior a la cotización"):
        load_checkpoint_compatible(str(ruta), NUM_NODES, AM_D_H, AM_N_HEADS, AM_N_LAYERS, AM_D_FF, verbose=False)
