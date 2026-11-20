"""
test_pricing_integration.py
===========================
validate_pricing.py (corrida reducida), pestaña "Pricing" del Excel y formato del bloque de precios de
inference.py.
"""

import numpy as np
import pandas as pd
import pytest
import torch

import inference
import validate_pricing as vp
from model.agent import AMRoutingAgent
from config import AM_D_H, AM_N_HEADS, AM_N_LAYERS, AM_D_FF, TRAIN_DAYS, configurar_k
from stats_analysis import ETIQUETA_REPLAY_CORTA

NUM_NODES = 10


@pytest.fixture(scope="module", autouse=True)
def k_libre():
    configurar_k(False)


def _arco(de, a, rate, p, pmin, pmax, be, pct, only=False, ret=False, carga=True):
    return {"from": de, "to": a, "rate_hoy": rate, "p_estrella": p, "p_min": pmin, "p_max": pmax, "break_even": be,
            "percentil_historico": pct, "only_option": only, "is_return": ret, "has_load": carga}


def test_formatear_bloque_sigue_el_formato_del_plan():
    nan = float("nan")
    arcos = [
        _arco(0, 7, 2.65, 2.58, 2.50, 2.66, 2.28, 42.0),
        _arco(7, 5, 2.44, 2.19, 2.10, 2.30, 2.28, 18.0),
        _arco(5, 9, 2.50, nan, nan, nan, 2.28, nan, only=True),
        _arco(9, 8, 2.0, nan, nan, nan, 2.28, nan, carga=False),
        _arco(8, 0, 2.95, 2.70, 2.60, 2.80, 2.28, 55.0, ret=True),
    ]
    assert inference.formatear_bloque([0, 7, 5, 9, 8, 0], arcos) == "\n".join([
        "Ruta: 0 → 7 → 5 → 9 → 8 → 0",
        "Precio de reserva por arco:",
        "  0→7: p* = 2.58 /mi  [rango 2.50–2.66]  (rate_hoy = 2.65, break-even = 2.28, percentil histórico 42 %)",
        "  7→5: p* = 2.19 /mi  [rango 2.10–2.30]  (rate_hoy = 2.44, break-even = 2.28, percentil histórico 18 %)  ← posición valiosa",
        "  5→9: p* = cost/dist (única acción factible)",
        "  9→8: sin carga en esta lane (reposicionamiento): sin p*",
        "  8→0: p* = 2.70 /mi  [rango 2.60–2.80]  (rate_hoy = 2.95, break-even = 2.28, percentil histórico 55 %)  (regreso a casa, Ṽ = 0)",
    ])


def test_formatear_bloque_retorno_sin_carga():
    nan = float("nan")
    arcos = [_arco(8, 0, 2.0, nan, nan, nan, 2.28, nan, ret=True, carga=False)]
    assert inference.formatear_bloque([8, 0], arcos) == "\n".join([
        "Ruta: 8 → 0",
        "Precio de reserva por arco:",
        "  8→0: sin carga en este regreso: sin p*",
    ])


def test_csv_columns_incluyen_rango_y_carga():
    assert {"p_estrella", "p_min", "p_max", "has_load", "is_return", "only_option", "percentil_historico"} <= set(inference.CSV_COLUMNS)


@pytest.fixture(scope="module")
def resultado(tmp_path_factory):
    torch.manual_seed(0)
    agent = AMRoutingAgent(NUM_NODES, AM_D_H, AM_N_HEADS, AM_N_LAYERS, AM_D_FF, use_post_head=True).eval()
    agent.post_entrenadas = True
    out = tmp_path_factory.mktemp("pricing")
    tablas = vp.run_pricing_validation(
        agent, NUM_NODES, **vp.cargar_datos(NUM_NODES), output_dir=str(out),
        n_mundos_cf=80, n_estados_cf=30, m_eval=30, verbose=False,
    )
    return out, tablas


def test_validacion_escribe_tablas_y_figuras(resultado):
    out, tablas = resultado
    base = out / "pricing_val"
    assert {p.name for p in (base / "figures").glob("*.png")} == {"contrafactual.png", "plausibilidad.png"}
    for t in ("contrafactual_resumen", "contrafactual_estados", "arcos_resumen", "umbral_resumen", "umbral_diferencias"):
        assert (base / "tables" / f"{t}.csv").exists()
    assert set(tablas) >= {"contrafactual_resumen", "arcos_resumen", "umbral_resumen", "umbral_diferencias",
                           "arcos_simulado", "arcos_replay"}


def test_validacion_resultados_coherentes(resultado):
    _, t = resultado
    cf = t["contrafactual_resumen"]
    assert cf.loc[0, "grupo"] == "global" and cf.loc[0, "n_estados"] == t["contrafactual_estados"].shape[0]
    assert cf.loc[1:, "n_estados"].sum() == cf.loc[0, "n_estados"]
    ar = t["arcos_resumen"]
    assert len(ar) == 2 and ar["pct_p_bajo_breakeven"].between(0, 100).all()
    sim = t["arcos_simulado"]
    con_p = sim[sim["p"].notna()]
    assert len(con_p) and con_p["percentil"].dropna().between(0, 100).all() and (con_p["p_min"] <= con_p["p"] + 1e-9).all()
    assert con_p["tiene_carga"].all()                                                   # todo arco con p* definido tiene carga
    assert sim["es_retorno"].any()                                                      # el retorno aparece en la tabla
    sin_carga_retorno = sim[sim["es_retorno"] & ~sim["tiene_carga"]]
    assert sin_carga_retorno["p"].isna().all()                                          # el retorno sin carga, sin p*
    um = t["umbral_resumen"]
    assert len(um) == 6 and (um["validez"] > 0.9).all()
    assert len(t["umbral_diferencias"]) == 4


def test_pestana_pricing_en_excel(resultado, tmp_path):
    _, t = resultado
    xlsx = tmp_path / "res.xlsx"
    pd.DataFrame({"a": [1]}).to_excel(xlsx, sheet_name="Resumen", index=False)
    vp.append_pricing_sheet(str(xlsx), t)
    assert pd.ExcelFile(xlsx).sheet_names == ["Resumen", "Pricing", "Pricing arcos"]
    import openpyxl
    celdas = [c.value for fila in openpyxl.load_workbook(xlsx)["Pricing"].iter_rows() for c in fila if c.value is not None]
    textos = [c for c in celdas if isinstance(c, str)]
    assert "Nota" in textos
    # todas las secciones sobreviven en la hoja (no solo la última)
    for titulo in ("Contrafactual", "Arcos de la ruta", "Regla de aceptación por umbral"):
        assert any(t.startswith(titulo) for t in textos), titulo
    assert "sesgo_usd_milla" in textos and "pct_p_bajo_breakeven" in textos
    arcos = pd.read_excel(xlsx, sheet_name="Pricing arcos")
    assert {"conjunto", "ruta", "desde", "hacia", "rate_hoy", "p_estrella", "p_estrella_min", "p_estrella_max",
            "break_even", "percentil_historico", "posicion_valiosa", "nota"} <= set(arcos.columns)
    assert set(arcos["conjunto"]) == {"simulado", ETIQUETA_REPLAY_CORTA}
    n = len(t["arcos_simulado"]) + len(t["arcos_replay"])
    assert len(arcos) == n
    # cada ruta se ve completa: las filas sin p* llevan su explicación
    assert (arcos.loc[arcos["p_estrella"].isna(), "nota"].fillna("") != "").all()
    ok = arcos[arcos["p_estrella"].notna()]
    assert (ok["posicion_valiosa"] == (ok["p_estrella"] < ok["break_even"])).all()


def test_validacion_exige_cabezas_entrenadas():
    agent = AMRoutingAgent(NUM_NODES, AM_D_H, AM_N_HEADS, AM_N_LAYERS, AM_D_FF, use_post_head=True).eval()
    with pytest.raises(ValueError, match="cabezas"):
        vp.run_pricing_validation(agent, NUM_NODES, **vp.cargar_datos(NUM_NODES))
