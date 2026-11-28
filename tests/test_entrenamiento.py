"""
test_entrenamiento.py
=====================
PPO en el dispositivo (am_training.py) y su validación:

  · asignación balanceada de K; número de mundos de validación por tamaño
  · validación (ValidacionSimulada): mundos fijos distintos de los de evaluación, reproducible, métricas en rango
  · smoke de entrenamiento: reward y pérdidas finitas, aceptación en (0, 1), todos los niveles de K con conformidad
    100 %, sin métricas de Ṽ, guarda el mejor update {"agent", "critic"}
  · checkpoint por media móvil (el update central de la ventana), modo "max", LR coseno y LR constante
"""

import math
import os

import numpy as np
import pytest
import torch

import am_training
from config import (CKPT_WINDOW, DEVICE, K_TRAINING_LEVELS, LR_FINAL_FRAC, MARKET_SEED_EVAL,
                    PPO_LR, TRAIN_DAYS, get_val_worlds)
from conftest import agente_chico
from evaluation import disenar_mundos, mundos_eval
from pricing import N_NIVELES

N = 10


def test_asignacion_balanceada_de_k():
    k = am_training._balanced_k_assignment(240, np.random.RandomState(0))
    for lvl in K_TRAINING_LEVELS:
        assert sum(1 for x in k if x == lvl) >= 240 // len(K_TRAINING_LEVELS)
    assert k == am_training._balanced_k_assignment(240, np.random.RandomState(0))


def test_numero_de_mundos_de_validacion_por_tamano():
    assert get_val_worlds(10) == get_val_worlds(35) == 200
    assert get_val_worlds(50) == get_val_worlds(100) == 300


def test_validacion_fija_reproducible_y_distinta_de_la_evaluacion(d10):
    env = d10.env()
    val = am_training.ValidacionSimulada(d10.market, env, d10.entradas(env), TRAIN_DAYS, N)
    dis = disenar_mundos(N, 100, d10.market, semilla=MARKET_SEED_EVAL)
    ev_mundos = mundos_eval(d10.market, dis["start_day"], TRAIN_DAYS - 1, MARKET_SEED_EVAL)
    assert not torch.equal(val.mundos.u[:100], ev_mundos.u)
    agent = agente_chico(N)
    a, b = val.evaluar(agent), val.evaluar(agent)
    assert a == b
    assert set(a["por_k"]) == {"free" if K is None else str(K) for K in K_TRAINING_LEVELS}
    for d in a["por_k"].values():
        assert set(d) == {"margen", "valid_rate", "tasa_aceptacion", "nivel_medio"}      # sin B3 ni UB-DP
        assert 0 < d["valid_rate"] <= 1 and 0 < d["tasa_aceptacion"] <= 1 and 0 <= d["nivel_medio"] <= N_NIVELES - 1
    assert a["score"] == a["margen"]


def _entrenar(d, monkeypatch, tmp_path, n_updates, episodios=24, **cfg):
    monkeypatch.setattr(am_training, "PPO_N_EPISODES_PER_UPDATE", episodios)
    monkeypatch.setattr(am_training, "get_n_updates", lambda n: n_updates)
    for k, v in cfg.items():
        monkeypatch.setattr(am_training, k, v)
    ruta = str(tmp_path / "best.pt")
    agent, critic, rewards, losses, log = am_training.run_am_training(
        d.tm, d.dist, d.diesel, N, d.market, avail_expected_arr=d.avail, reward_global_p95=d.p95,
        best_checkpoint_path=ruta, est_lane=d.est)
    return agent, log, ruta


def test_smoke_de_entrenamiento(d10, monkeypatch, tmp_path):
    agent, log, ruta = _entrenar(d10, monkeypatch, tmp_path, n_updates=2, episodios=64)
    assert len(log) == 2
    vistos = set()
    for e in log:
        assert np.isfinite(e["reward"]) and np.isfinite(e["total_loss"]) and e["entropy_precio"] > 0
        assert 0 < e["tasa_aceptacion"] < 1 and 0 <= e["nivel_medio"] <= N_NIVELES - 1
        assert "post_mse" not in e and "post_ev" not in e
        for k, diag in e["k_diagnostics"].items():
            vistos.add(k)
            if diag is not None and "conformity_rate" in diag:
                assert diag["conformity_rate"] == 1.0
        assert "val" in e and "greedy_reward" in e
    assert vistos == {"free" if K is None else str(K) for K in K_TRAINING_LEVELS}
    assert not any(k.startswith("post_value") for k in agent.state_dict())
    assert os.path.exists(ruta) and set(torch.load(ruta, map_location="cpu")) == {"agent", "critic"}
    assert all(p.device.type == DEVICE.type for p in agent.parameters())


def test_media_movil_guarda_el_update_central_y_lr_coseno(d10, monkeypatch, tmp_path):
    n = 12
    _, log, ruta = _entrenar(d10, monkeypatch, tmp_path, n_updates=n, episodios=16)
    val = [e for e in log if "val_score" in e]
    assert len(val) == len(log) == n
    assert all(e["val_ma"] is None for e in val[:CKPT_WINDOW - 1])
    assert val[CKPT_WINDOW - 1]["val_ma"] == pytest.approx(np.mean([e["val_score"] for e in val[:CKPT_WINDOW]]))
    mejores = [e for e in val if e.get("val_best")]
    assert mejores
    for e in mejores:                                        # guarda el central, no el último de la ventana
        assert e["ckpt_update"] == e["update"] - CKPT_WINDOW // 2
        assert e["ckpt_score"] == next(x["val_score"] for x in val if x["update"] == e["ckpt_update"])
    mas = [e["val_ma"] for e in mejores]
    assert mas == sorted(mas) and len(set(mas)) == len(mas)
    assert set(torch.load(ruta, map_location="cpu")) == {"agent", "critic"}
    lr = [e["lr_actor"] for e in log]
    assert lr[0] == pytest.approx(PPO_LR) and all(a >= b for a, b in zip(lr, lr[1:]))
    assert lr[-1] == pytest.approx(PPO_LR * (LR_FINAL_FRAC + (1 - LR_FINAL_FRAC) * 0.5 * (1 + math.cos(math.pi * (n - 1) / n))))


def test_ckpt_max_y_lr_constante(d10, monkeypatch, tmp_path):
    _, log, _ = _entrenar(d10, monkeypatch, tmp_path, n_updates=5, episodios=16, CKPT_SELECTION="max",
                          USE_LR_COSINE=False)
    val = [e for e in log if "val_score" in e]
    mejores = [e for e in val if e.get("val_best")]
    assert [e["val_score"] for e in mejores] == sorted(e["val_score"] for e in mejores)
    assert mejores[-1]["val_score"] == max(e["val_score"] for e in val)
    assert all(e["ckpt_update"] == e["update"] for e in mejores)
    assert {e["lr_actor"] for e in log} == {PPO_LR}
