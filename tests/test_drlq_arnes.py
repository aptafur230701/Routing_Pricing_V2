"""
test_drlq_arnes.py
==================
DRL-Q como un método más del arnés de evaluation.py, junto al DRL (actor) y los baselines.
"""

import numpy as np
import pytest
import torch

import evaluation as ev
from am_agent import AMRoutingAgent
from config import AM_D_H, AM_N_HEADS, AM_N_LAYERS, AM_D_FF, TRAIN_DAYS, configurar_k
from dp_relajada import cargar_o_calcular_tabla
from stats_analysis import comparar_referencias
import validate_pricing as vp

NUM_NODES = 10


def test_metodos_separan_baselines():
    assert ev._METODOS == ("DRL", "DRL-Q", "B1", "B2", "B2P", "B3")
    assert ev._BASELINES == ("B1", "B2", "B2P", "B3")


@pytest.fixture(scope="module")
def sim_df():
    configurar_k(False)
    torch.manual_seed(0)
    d = vp.cargar_datos(NUM_NODES)
    agent = AMRoutingAgent(NUM_NODES, AM_D_H, AM_N_HEADS, AM_N_LAYERS, AM_D_FF, use_post_head=True).eval()
    agent.post_entrenadas = True
    tabla = cargar_o_calcular_tabla(d["market"], d["time_matrix"], d["distance_arr"], d["diesel_arr"], verbose=False)
    return ev.evaluar_mundos_simulados(
        agent, d["market"], tabla, d["time_matrix"], d["rate_stack"][:TRAIN_DAYS], d["loads_stack"][:TRAIN_DAYS],
        d["distance_arr"], d["diesel_arr"], NUM_NODES, d["ltr_stack"], d["trucks_stack"], d["avail_expected_arr"],
        d["reward_global_p95"], M=14, k_levels=(None,), verbose=False)


def test_drl_q_esta_en_el_arnes(sim_df):
    assert set(sim_df["metodo"]) >= {"DRL", "DRL-Q", "B1", "B2", "B2P", "B3"}
    assert (sim_df.groupby("metodo").size() == 14).all()
    res = ev.resumen_metodos(sim_df)
    assert "DRL-Q" in set(res["metodo"])


def test_comparacion_pareada_de_drl_q(sim_df):
    c = comparar_referencias(sim_df)
    assert {"DRL − B1", "DRL-Q − DRL", "DRL-Q − B1", "DRL-Q − B2", "DRL-Q − B2P", "DRL-Q − B3"} <= set(c["comparacion"])
    pareada, rel = vp.utilidad_pareada(sim_df)
    assert len(rel) == 1 and rel.loc[0, "dif_rel_ic95_inf_pct"] <= rel.loc[0, "dif_rel_pct"] <= rel.loc[0, "dif_rel_ic95_sup_pct"]


def test_drl_q_sin_cabezas_no_se_evalua():
    configurar_k(False)
    d = vp.cargar_datos(NUM_NODES)
    agent = AMRoutingAgent(NUM_NODES, AM_D_H, AM_N_HEADS, AM_N_LAYERS, AM_D_FF, use_post_head=True).eval()
    tabla = cargar_o_calcular_tabla(d["market"], d["time_matrix"], d["distance_arr"], d["diesel_arr"], verbose=False)
    df = ev.evaluar_mundos_simulados(
        agent, d["market"], tabla, d["time_matrix"], d["rate_stack"][:TRAIN_DAYS], d["loads_stack"][:TRAIN_DAYS],
        d["distance_arr"], d["diesel_arr"], NUM_NODES, d["ltr_stack"], d["trucks_stack"], d["avail_expected_arr"],
        d["reward_global_p95"], M=7, k_levels=(None,), verbose=False)
    assert "DRL-Q" not in set(df["metodo"])
