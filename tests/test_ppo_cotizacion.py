"""
test_ppo_cotizacion.py
======================
Fase 4 de especificacion_fase0_ruteo_pricing.md: el DRL decide (destino, nivel de precio).

  · la cabeza de precio respeta las opciones válidas del destino elegido; log π(j, o) = log π(j) + log π(o | j)
    y, con una sola opción válida, log π(o | j) = 0 (test 10)
  · evaluate_cotizacion reproduce la log-prob de act_cotizacion; greedy elige opciones válidas
  · crear la cabeza de precio no consume el RNG global de torch
  · features: 14 por nodo, las de cotización en rango y en cero en el self-loop
  · un checkpoint de la arquitectura anterior (11 features, contexto sin el bloque de cotización) carga
    con las columnas nuevas en cero
  · greedy_rollout: rutas solo con viajes, ganadas ≤ cotizadas, determinista con las mismas semillas
  · smoke de entrenamiento (test 11): reward finito, aceptación en (0, 1) y K respetado, en mundos simulados
    y en replay de días reales
"""

import os

import numpy as np
import pytest
import torch

import am_training
from baselines.politicas import calcular_ev_pronostico
from config import AM_D_H, AM_N_HEADS, AM_N_LAYERS, AM_D_FF, DEVICE, TRAIN_DAYS, K_MAX
from greedy_rollout import construir_features, greedy_rollout
from market_process import MarketProcess, asegurar_parametros
from model.agent import AMRoutingAgent
from model.critic_head import CriticHead
from model.decoder import PriceHead
from model.encoder import COT_COLS, INDEX_COL, N_NODE_FEATURES
from pricing import N_NIVELES
from problem_data import load_matrices
from routing_env import OPCION_VACIO, VectorRoutingEnv
from transfer_learning import load_checkpoint_compatible, es_checkpoint_con_cotizacion

NUM_NODES = 10
B = 32
SEMILLAS = np.arange(500, 500 + B)


@pytest.fixture(scope="module")
def datos():
    asegurar_parametros()
    tm, rate, loads, dist, diesel, ltr, avail, p95 = load_matrices(NUM_NODES)
    mp = MarketProcess(NUM_NODES)
    return dict(tm=np.asarray(tm, dtype=float), rate=rate, loads=loads, dist=dist, diesel=diesel, ltr=ltr,
                avail=avail, p95=p95, mp=mp, ev=calcular_ev_pronostico(mp, dist, diesel))


def _agente(seed=0):
    torch.manual_seed(seed)
    return AMRoutingAgent(NUM_NODES, AM_D_H, AM_N_HEADS, AM_N_LAYERS, AM_D_FF, device=DEVICE).eval()


def _env(d):
    return VectorRoutingEnv(d["tm"], NUM_NODES, d["dist"], d["diesel"], TRAIN_DAYS, market_process=d["mp"])


def _estado(d, agent, env):
    starts = np.arange(B) % NUM_NODES
    env.reset(starts, np.arange(B) % 20, world_seeds=SEMILLAS)
    feats = construir_features(env, starts, d["tm"], d["dist"], d["ltr"], d["p95"], NUM_NODES, d["avail"], d["ev"])
    t = {k: torch.from_numpy(v).to(DEVICE) for k, v in feats.items()}
    emb, g = agent.encoder(t["node_feats"])
    cur = torch.as_tensor(env.current_node, dtype=torch.long, device=DEVICE)
    h = agent.context_net(g, emb[torch.arange(B), cur], t["temporal"], t["market"], t["budget"], t["dow"], t["cot"])
    mask = torch.from_numpy(env.opciones_validas.any(-1).astype(np.int8)).to(DEVICE)
    return feats, h, emb, mask, t["opciones"]


# ── Decoder ──────────────────────────────────────────────────────────────────

def test_act_respeta_opciones_y_logprob_es_la_suma(datos):
    agent, env = _agente(), _env(datos)
    with torch.no_grad():
        feats, h, emb, mask, op = _estado(datos, agent, env)
        j, o, lp, h_j, h_o = agent.decoder.act_cotizacion(h, emb, mask, op)
        assert op[torch.arange(B), j, o].all()
        lp2, h_j2, h_o2 = agent.decoder.evaluate_cotizacion(h, emb, mask, op, j, o)
        torch.testing.assert_close(lp, lp2)
        # suma explícita de las dos log-probs
        logits, glimpse = agent.decoder._logits_y_glimpse(h, emb, mask == 0)
        lp_j = torch.log_softmax(logits, -1)[torch.arange(B), j]
        lp_o = agent.decoder._dist_precio(h, glimpse, emb, j, op).log_prob(o)
        torch.testing.assert_close(lp, lp_j + lp_o, atol=1e-5, rtol=1e-5)
        assert (h_j >= 0).all() and (h_o >= 0).all()


def test_una_sola_opcion_no_agrega_logprob(datos):
    agent, env = _agente(), _env(datos)
    with torch.no_grad():
        _, h, emb, mask, op = _estado(datos, agent, env)
        solo = torch.zeros_like(op)
        j = mask.float().argmax(dim=-1)                     # un destino válido por fila
        solo[torch.arange(B), j, OPCION_VACIO] = True
        o = torch.full((B,), OPCION_VACIO, dtype=torch.long, device=DEVICE)
        lp_o = agent.decoder._dist_precio(h, agent.decoder.glimpse(h, emb, mask == 0), emb, j, solo).log_prob(o)
        torch.testing.assert_close(lp_o, torch.zeros(B, device=DEVICE), atol=1e-6, rtol=0)


def test_greedy_elige_opciones_validas(datos):
    agent, env = _agente(), _env(datos)
    with torch.no_grad():
        _, h, emb, mask, op = _estado(datos, agent, env)
        j, o = agent.decoder.greedy_cotizacion(h, emb, mask, op)
    assert op[torch.arange(B), j, o].all()


def test_cabeza_de_precio_no_consume_el_rng_global():
    torch.manual_seed(123)
    antes = torch.get_rng_state().clone()
    PriceHead(AM_D_H, N_NIVELES + 1)
    assert torch.equal(antes, torch.get_rng_state())


# ── Features ─────────────────────────────────────────────────────────────────

def test_features_de_cotizacion(datos):
    agent = _agente()
    env = _env(datos)
    feats, *_ = _estado(datos, agent, env)
    nf = feats["node_feats"]
    assert nf.shape == (B, NUM_NODES, N_NODE_FEATURES) and np.isfinite(nf).all()
    cot = nf[:, :, COT_COLS]
    assert cot.min() >= 0 and cot.max() <= 1
    assert (cot[np.arange(B), env.current_node] == 0).all()                            # self-loop
    np.testing.assert_array_equal(cot[..., 2] > 0, env.opciones_validas[..., :N_NIVELES].any(-1)
                                  & (np.arange(NUM_NODES)[None] != env.current_node[:, None]))
    assert feats["cot"].shape == (B, 2) and (feats["cot"] >= 0).all() and (feats["cot"] <= 1).all()
    assert INDEX_COL == 10


@pytest.mark.skipif(not es_checkpoint_con_cotizacion(os.path.join(
                        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "checkpoints",
                        "am_checkpoint_10nodes.pt")),
                    reason="no hay un checkpoint de 10 nodos entrenado con la cotización")
def test_checkpoint_real_de_10_nodos_carga_y_rutea(datos):
    ruta = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "checkpoints", "am_checkpoint_10nodes.pt")
    agent, _ = load_checkpoint_compatible(ruta, NUM_NODES, AM_D_H, AM_N_HEADS, AM_N_LAYERS, AM_D_FF, verbose=False)
    res = greedy_rollout(agent, _env(datos), np.arange(B) % NUM_NODES, np.arange(B) % 20, datos["tm"], datos["dist"],
                         datos["ltr"], datos["p95"], NUM_NODES, avail_expected_arr=datos["avail"],
                         world_seeds=SEMILLAS, ev_esperado=datos["ev"])
    assert res.valid.all()


# ── Rollout greedy ───────────────────────────────────────────────────────────

def test_greedy_rollout(datos):
    env = _env(datos)
    pasos_margen = []

    def hook(c):
        pasos_margen.append(c["opciones"].copy())

    res = greedy_rollout(_agente(), env, np.arange(B) % NUM_NODES, np.arange(B) % 20, datos["tm"], datos["dist"],
                         datos["ltr"], datos["p95"], NUM_NODES, k_max_arr=[K_MAX] * B,
                         avail_expected_arr=datos["avail"], world_seeds=SEMILLAS, ev_esperado=datos["ev"],
                         step_hook=hook)
    assert pasos_margen and all(o is not None for o in pasos_margen)
    assert (res.n_ganadas <= res.n_cotizaciones).all()
    assert all(len(r) - 1 <= K_MAX for r in res.routes)                               # K se cuenta en viajes
    assert all(len(n) == c for n, c in zip(res.niveles, res.n_cotizaciones))
    assert all(r[-1] == r[0] for r, v in zip(res.routes, res.valid) if v)
    # Mismas semillas ⇒ misma competencia ⇒ mismo resultado
    res2 = greedy_rollout(_agente(), _env(datos), np.arange(B) % NUM_NODES, np.arange(B) % 20, datos["tm"],
                          datos["dist"], datos["ltr"], datos["p95"], NUM_NODES, k_max_arr=[K_MAX] * B,
                          avail_expected_arr=datos["avail"], world_seeds=SEMILLAS, ev_esperado=datos["ev"])
    np.testing.assert_array_equal(res.raw_reward, res2.raw_reward)


# ── Entrenamiento ────────────────────────────────────────────────────────────

@pytest.mark.parametrize("simulado", [True, False], ids=["mundos_simulados", "replay"])
def test_smoke_de_entrenamiento(monkeypatch, datos, simulado):
    monkeypatch.setattr(am_training, "PPO_N_EPISODES_PER_UPDATE", 70)
    monkeypatch.setattr(am_training, "get_episodes_per_node", lambda n: 2 * 70 // n)
    d = datos
    agent, _, rewards, _, log = am_training.run_am_training(
        d["tm"], d["rate"][:TRAIN_DAYS], d["loads"][:TRAIN_DAYS], d["dist"], d["diesel"], NUM_NODES,
        ltr_stack=d["ltr"], reward_global_p95=d["p95"], avail_expected_arr=d["avail"],
        market_process=d["mp"] if simulado else None, ev_esperado=d["ev"],
    )
    assert len(log) == 2 and all(np.isfinite(e["reward"]) and np.isfinite(e["total_loss"]) for e in log)
    for e in log:
        assert 0 < e["tasa_aceptacion"] < 1 and 0 <= e["nivel_medio"] <= N_NIVELES - 1
        assert e["entropy_precio"] > 0
        kd = e["k_diagnostics"].get(str(K_MAX))
        if kd is not None:
            assert kd["conformity_rate"] == 1.0
