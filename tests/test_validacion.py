"""
test_validacion.py
==================
Validación en mundos simulados durante el entrenamiento (am_training.ValidacionSimulada):
mundos fijos y distintos de los de evaluación, resultados reproducibles, referencias B3 / UB-DP
coherentes y guardado del mejor update.
"""

import os

import numpy as np
import pytest
import torch

import am_training
from am_agent import AMRoutingAgent
from config import (
    DEVICE,
    K_TRAINING_LEVELS, MARKET_SEED_EVAL, MARKET_SEED_VAL, TRAIN_DAYS, get_val_worlds,
)
from baselines import calcular_ev_esperado
from evaluation import disenar_mundos
from market_process import MarketProcess, asegurar_parametros
from problem_data import load_matrices

N = 10


@pytest.fixture(scope="module")
def datos():
    asegurar_parametros()
    (tm, rate, loads, dist, diesel, ltr, trucks, avail, p95) = load_matrices(N)
    return dict(tm=tm, rate=rate[:TRAIN_DAYS], loads=loads[:TRAIN_DAYS], dist=dist, diesel=diesel,
                ltr=ltr, trucks=trucks, avail=avail, p95=p95, market=MarketProcess(N))


def _val(d):
    return am_training.ValidacionSimulada(
        d["market"], d["tm"], d["rate"], d["loads"], d["dist"], d["diesel"], N,
        np.asarray(d["tm"], dtype=float), d["ltr"], d["trucks"], d["p95"], d["avail"],
        ev_esperado=calcular_ev_esperado(d["market"], d["dist"], d["diesel"]),
    )


def test_numero_de_mundos_por_tamano():
    assert get_val_worlds(10) == get_val_worlds(35) == 200
    assert get_val_worlds(50) == get_val_worlds(100) == 300


def test_mundos_de_validacion_distintos_de_los_de_evaluacion(datos):
    mp = datos["market"]
    val = disenar_mundos(N, 200, mp, semilla=MARKET_SEED_VAL)
    ev_ = disenar_mundos(N, 100, mp, semilla=MARKET_SEED_EVAL)
    assert not np.isin(val["world_seed"], ev_["world_seed"]).any()


def test_validacion_estructura_y_reproducibilidad(datos):
    torch.manual_seed(0)
    agent = AMRoutingAgent(num_nodes=N, d_h=32, n_heads=4, n_layers=1, d_ff=64, device=DEVICE)
    val = _val(datos)
    a, b = val.evaluar(agent), val.evaluar(agent)
    assert a == b                                           # mismos mundos, misma política ⇒ mismo resultado
    assert set(a["por_k"]) == {"free" if K is None else str(K) for K in K_TRAINING_LEVELS}
    for d in a["por_k"].values():
        assert d["valid_rate"] == 1.0
        # B3 no supera a UB-DP salvo ruido de muestreo; el gap del agente es finito
        assert d["b3"] <= d["ub_dp"] * 1.05
        assert np.isfinite(d["gap_ub_dp_pct"])
    assert a["score"] == pytest.approx(a["margen"] * a["valid_rate"])


def test_entrenamiento_guarda_mejor_update_y_loguea_validacion(datos, monkeypatch, tmp_path):
    monkeypatch.setattr(am_training, "PPO_N_EPISODES_PER_UPDATE", 24)
    monkeypatch.setattr(am_training, "get_episodes_per_node", lambda n: 6)   # 60 ep ⇒ 2 updates
    ruta = str(tmp_path / "best.pt")
    _, _, _, _, log = am_training.run_am_training(
        datos["tm"], datos["rate"], datos["loads"], datos["dist"], datos["diesel"], N,
        ltr_stack=datos["ltr"], trucks_stack=datos["trucks"], reward_global_p95=datos["p95"],
        avail_expected_arr=datos["avail"], market_process=datos["market"], best_checkpoint_path=ruta,
    )
    con_val = [e for e in log if "val" in e]
    assert con_val and os.path.exists(ruta)
    assert any(e.get("val_best") for e in con_val)
    assert all("greedy_reward" in e for e in con_val)
    ck = torch.load(ruta, map_location="cpu")
    assert set(ck) == {"agent", "critic"}


# ─────────────────────────────────────────────────────────────────────────────
# Día de la semana como feature del contexto
# ─────────────────────────────────────────────────────────────────────────────

def test_dow_del_entorno_y_one_hot(datos):
    from attention_encoder import build_dow_features_batch, N_DOW_FEATURES
    from fit_market_process import dia_de_semana
    from routing_env import VectorRoutingEnv
    for offset in (0, TRAIN_DAYS):
        env = VectorRoutingEnv(
            time_matrix=datos["tm"], rate_stack=datos["rate"], loads_stack=datos["loads"], rm_pen_stack=None,
            num_nodes=N, market_process=datos["market"], distance_arr=datos["dist"],
            diesel_arr=datos["diesel"], day_offset=offset,
        )
        env.reset(np.array([0, 1]), np.array([3, 10]), world_seeds=np.array([1, 2]))
        assert (env.current_dow == dia_de_semana(np.array([3, 10]) + offset)).all()
        env.step(np.array([2, 3]))
        t = env.time_elapsed
        esperado = dia_de_semana(np.array([3, 10]) + offset + (t // 14).astype(int))
        assert (env.current_dow == esperado).all()
    oh = build_dow_features_batch(np.array([0, 6, 3]))
    assert oh.shape == (3, N_DOW_FEATURES) and (oh.sum(axis=1) == 1).all() and oh[1, 6] == 1


def test_dow_cambia_la_decision_y_checkpoint_viejo_equivale(datos):
    """Con columnas de dow en cero el contexto es el de antes; con ellas, el día importa."""
    from attention_encoder import ContextNetwork
    torch.manual_seed(0)
    ctx = ContextNetwork(32)
    g, c = torch.randn(4, 32), torch.randn(4, 32)
    t, m, b = torch.rand(4, 3), torch.rand(4, 1), torch.rand(4, 2)
    sin = ctx(g, c, t, m, b)                                    # dow None ⇒ ceros
    cero = ctx(g, c, t, m, b, torch.zeros(4, 7))
    assert torch.equal(sin, cero)
    uno = torch.zeros(4, 7); uno[:, 5] = 1
    assert not torch.allclose(sin, ctx(g, c, t, m, b, uno))
