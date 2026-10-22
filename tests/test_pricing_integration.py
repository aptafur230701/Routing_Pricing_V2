"""
test_pricing_integration.py
===========================
Fases P3-P5 de PRICING_PLAN.md: validate_pricing.py (corrida reducida), pestaña "Pricing" del Excel y
formato del bloque de precios de inference.py.
"""

import os

import numpy as np
import pandas as pd
import pytest
import torch

import inference
import validate_pricing as vp
from am_agent import AMRoutingAgent
from config import DEVICE, AM_D_H, AM_N_HEADS, AM_N_LAYERS, AM_D_FF

NUM_NODES = 10


def test_formatear_bloque_sigue_el_formato_del_plan():
    arcos = [
        dict(**{"from": 0, "to": 7}, rate_hoy=2.65, p_estrella=2.58, break_even=2.28, percentil_historico=42.0,
             only_option=False, is_return=False),
        dict(**{"from": 7, "to": 5}, rate_hoy=2.44, p_estrella=2.19, break_even=2.28, percentil_historico=18.0,
             only_option=False, is_return=False),
        dict(**{"from": 5, "to": 9}, rate_hoy=2.50, p_estrella=2.28, break_even=2.28, percentil_historico=float("nan"),
             only_option=True, is_return=False),
        dict(**{"from": 9, "to": 0}, rate_hoy=2.0, p_estrella=2.28, break_even=2.28, percentil_historico=float("nan"),
             only_option=False, is_return=True),
    ]
    assert inference.formatear_bloque([0, 7, 5, 9, 0], arcos) == "\n".join([
        "Ruta: 0 → 7 → 5 → 9 → 0",
        "Precio de reserva por arco:",
        "  0→7: p* = 2.58 /mi  (rate_hoy = 2.65, break-even = 2.28, percentil histórico 42 %)",
        "  7→5: p* = 2.19 /mi  (rate_hoy = 2.44, break-even = 2.28, percentil histórico 18 %)  ← posición valiosa",
        "  5→9: p* = cost/dist (única acción factible)",
        "  9→0: p* = cost/dist (retorno al depot, Ṽ = 0)",
    ])


@pytest.fixture(scope="module")
def resultado(tmp_path_factory):
    agent = AMRoutingAgent(NUM_NODES, AM_D_H, AM_N_HEADS, AM_N_LAYERS, AM_D_FF, device=DEVICE, use_post_head=True).eval()
    out = tmp_path_factory.mktemp("pricing")
    tablas = vp.run_pricing_validation(
        agent, NUM_NODES, **vp.cargar_datos(NUM_NODES), output_dir=str(out),
        n_worlds=40, n_mc_states=12, n_mc_rollouts=8, n_pos=30, n_no_tomadas=4, verbose=False,
    )
    return out, tablas


def test_validacion_escribe_tablas_y_figuras(resultado):
    out, tablas = resultado
    base = out / "pricing_val"
    assert len(list((base / "figures").glob("*.png"))) == 3
    assert (base / "tables" / "mc_calibracion_resumen.csv").exists()
    assert set(tablas) >= {"mc_calibracion_resumen", "posicion_muestras", "aceptacion_resumen"}


def test_validacion_resultados_coherentes(resultado):
    _, t = resultado
    assert np.isfinite(t["mc_calibracion_muestras"][["v_pred_usd", "v_mc_usd"]].to_numpy()).all()
    pos = t["posicion_muestras"]
    assert pos["percentil_p_estrella"].between(0, 100).all()
    assert (pos["posicion_valiosa"] == (pos["p_estrella"] < pos["break_even"])).all()
    acep = t["aceptacion_resumen"]
    assert len(acep) == 4 and (acep["validez"] > 0.9).all()
    # la regla "acepto si tarifa ≥ p*" con las mismas cargas no puede tener tours más largos que el límite
    assert (acep["largo_tour_medio"] <= NUM_NODES).all()


def test_pestana_pricing_en_excel(resultado, tmp_path):
    _, t = resultado
    xlsx = tmp_path / "res.xlsx"
    pd.DataFrame({"a": [1]}).to_excel(xlsx, sheet_name="Resumen", index=False)
    vp.append_pricing_sheet(str(xlsx), t)
    hojas = pd.ExcelFile(xlsx).sheet_names
    assert hojas == ["Resumen", "Pricing"]
