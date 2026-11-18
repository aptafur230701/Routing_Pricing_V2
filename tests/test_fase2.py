"""
test_fase2.py
=============
Selección del checkpoint por media móvil, LR coseno, diferencia pareada contra B3 en validación,
y caché de baselines/cotas entre corridas.
"""

import math
import os

import numpy as np
import pytest
import torch

import am_training
from model.agent import AMRoutingAgent
from baselines.politicas import calcular_ev_pronostico
from config import TRAIN_DAYS, K_MAX, PPO_LR, LR_FINAL_FRAC, CKPT_WINDOW, DEVICE
import evaluation as ev
from market_process import MarketProcess, asegurar_parametros
from baselines.dp_relajada import calcular_tabla
from problem_data import load_matrices

N = 10


@pytest.fixture(scope="module")
def datos():
    asegurar_parametros()
    (tm, rate, loads, dist, diesel, ltr, avail, p95) = load_matrices(N)
    return dict(tm=tm, rate=rate, loads=loads, dist=dist, diesel=diesel, ltr=ltr,
                avail=avail, p95=p95, market=MarketProcess(N))


def _entrenar(d, monkeypatch, tmp_path, n_updates=20, **cfg):
    monkeypatch.setattr(am_training, "PPO_N_EPISODES_PER_UPDATE", 10)
    monkeypatch.setattr(am_training, "get_episodes_per_node", lambda n: n_updates)   # 10·N·n_updates/N ep
    for k, v in cfg.items():
        monkeypatch.setattr(am_training, k, v)
    ruta = str(tmp_path / "best.pt")
    *_, log = am_training.run_am_training(
        d["tm"], d["rate"][:TRAIN_DAYS], d["loads"][:TRAIN_DAYS], d["dist"], d["diesel"], N,
        ltr_stack=d["ltr"], reward_global_p95=d["p95"],
        avail_expected_arr=d["avail"], market_process=d["market"], best_checkpoint_path=ruta,
    )
    return log, ruta


def test_media_movil_guarda_el_update_central(datos, monkeypatch, tmp_path):
    log, ruta = _entrenar(datos, monkeypatch, tmp_path)
    val = [e for e in log if "val_score" in e]
    assert len(val) == len(log)                              # val_freq = 1 con 20 updates
    assert all(e["val_ma"] is None for e in val[:CKPT_WINDOW - 1])
    assert val[CKPT_WINDOW - 1]["val_ma"] == pytest.approx(np.mean([e["val_score"] for e in val[:CKPT_WINDOW]]))
    mejores = [e for e in val if e.get("val_best")]
    assert mejores
    for e in mejores:                                        # guarda el central, no el último de la ventana
        assert e["ckpt_update"] == e["update"] - CKPT_WINDOW // 2
        assert e["ckpt_score"] == next(x["val_score"] for x in val if x["update"] == e["ckpt_update"])
    mas = [e["val_ma"] for e in mejores]
    assert mas == sorted(mas) and len(set(mas)) == len(mas)  # cada guardado mejora la media móvil
    assert val[-1]["ckpt_update"] == mejores[-1]["ckpt_update"]
    assert set(torch.load(ruta, map_location="cpu")) == {"agent", "critic"}


def test_ckpt_max_conserva_el_comportamiento_anterior(datos, monkeypatch, tmp_path):
    log, _ = _entrenar(datos, monkeypatch, tmp_path, CKPT_SELECTION="max")
    val = [e for e in log if "val_score" in e]
    mejores = [e for e in val if e.get("val_best")]
    assert [e["val_score"] for e in mejores] == sorted(e["val_score"] for e in mejores)
    assert mejores[-1]["val_score"] == max(e["val_score"] for e in val)
    assert all(e["ckpt_update"] == e["update"] for e in mejores)


def test_lr_coseno_baja_hasta_la_fraccion_final(datos, monkeypatch, tmp_path):
    n = 20
    log, _ = _entrenar(datos, monkeypatch, tmp_path, n_updates=n)
    lr = [e["lr_actor"] for e in log]
    assert lr[0] == pytest.approx(PPO_LR)
    assert all(a >= b for a, b in zip(lr, lr[1:]))
    esperado_ultimo = PPO_LR * (LR_FINAL_FRAC + (1 - LR_FINAL_FRAC) * 0.5 * (1 + math.cos(math.pi * (n - 1) / n)))
    assert lr[-1] == pytest.approx(esperado_ultimo) and lr[-1] < 0.15 * PPO_LR


def test_sin_lr_coseno_el_lr_es_constante(datos, monkeypatch, tmp_path):
    log, _ = _entrenar(datos, monkeypatch, tmp_path, n_updates=6, USE_LR_COSINE=False)
    assert {e["lr_actor"] for e in log} == {PPO_LR}


def test_validacion_reporta_diferencia_pareada_contra_b3(datos):
    torch.manual_seed(0)
    agent = AMRoutingAgent(num_nodes=N, d_h=32, n_heads=4, n_layers=1, d_ff=64, device=DEVICE)
    d = datos
    val = am_training.ValidacionSimulada(
        d["market"], d["tm"], d["rate"][:TRAIN_DAYS], d["loads"][:TRAIN_DAYS], d["dist"], d["diesel"], N,
        np.asarray(d["tm"], dtype=float), d["ltr"], d["p95"], d["avail"],
        ev_esperado=calcular_ev_pronostico(d["market"], d["dist"], d["diesel"]),
    )
    for K, arr in val.b3_mundos.items():
        assert arr.shape == (val.M,)
        assert arr.mean() == pytest.approx(val.ref[K]["b3"])
    out = val.evaluar(agent)
    for K in val.niveles:
        k = "free" if K is None else str(K)
        if out["por_k"][k]["valid_rate"] == 1.0:           # margen = promedio de todos los mundos
            assert out["por_k"][k]["dif_b3"] == pytest.approx(out["por_k"][k]["margen"] - val.ref[K]["b3"])
        assert np.isfinite(out["por_k"][k]["dif_b3"])


def test_cache_de_baselines_reproduce_el_calculo_directo(datos, tmp_path):
    d = datos
    tabla = calcular_tabla(d["market"], d["tm"], d["dist"], d["diesel"], K=K_MAX, S=100)
    torch.manual_seed(0)
    a1 = AMRoutingAgent(num_nodes=N, d_h=32, n_heads=4, n_layers=1, d_ff=64, device=torch.device("cpu"))
    a2 = AMRoutingAgent(num_nodes=N, d_h=32, n_heads=4, n_layers=1, d_ff=64, device=torch.device("cpu"))
    args = (d["market"], tabla, d["tm"], d["rate"][:TRAIN_DAYS], d["loads"][:TRAIN_DAYS], d["dist"],
            d["diesel"], N, d["ltr"], d["avail"], d["p95"])
    cache = str(tmp_path / "c" / "sim.pkl")
    directo = ev.evaluar_mundos_simulados(a1, *args, M=8, verbose=False)
    primero = ev.evaluar_mundos_simulados(a1, *args, M=8, verbose=False, cache_path=cache)
    assert os.path.exists(cache)
    segundo = ev.evaluar_mundos_simulados(a2, *args, M=8, verbose=False, cache_path=cache)   # otro agente, mismo caché
    cols = ["modo", "K", "mundo", "metodo", "margen", "ruta", "ub_dp", "ub_pi"]
    for df in (primero, segundo):
        assert len(df) == len(directo)
        assert list(df["metodo"]) == list(directo["metodo"]) and list(df["mundo"]) == list(directo["mundo"])
    np.testing.assert_allclose(primero["margen"], directo["margen"])
    no_drl = directo["metodo"] != "DRL"
    np.testing.assert_allclose(segundo.loc[no_drl, "margen"], directo.loc[no_drl, "margen"])
    np.testing.assert_allclose(segundo["ub_pi"], directo["ub_pi"])
    np.testing.assert_allclose(segundo["ub_dp"], directo["ub_dp"], equal_nan=True)
    # firma distinta (M distinto) ⇒ recalcula
    otro = ev.evaluar_mundos_simulados(a1, *args, M=4, verbose=False, cache_path=cache)
    assert otro["mundo"].nunique() == 4
