"""
test_evaluacion.py
==================
Arnés de evaluación en lote (evaluation.py):

  · rango de precio del DRL (banda alrededor del nivel elegido) y su texto por arco
  · el caché de baselines y UB-PI reproduce el cálculo directo y se recalcula si cambia la firma
  · los mundos de prueba salen en octubre: corren todos los métodos, sin columna UB-DP, con UB-PI ≥ cada margen
  · las métricas de cada episodio son coherentes (ruta que cierra en casa, ganadas ≤ cotizaciones, margen = Σ pasos)
"""

import os

import numpy as np
import pytest

import evaluation as ev
from baselines.dp_relajada import PENALIZACION_USD
from config import K_MAX
from conftest import agente_chico, mundos_sim
from evaluation import banda_precio, precios_por_arco
from pricing import N_NIVELES

N = 10


def _probs(p):
    return np.array(list(p) + [0.0])                     # 9 niveles + VACÍO al final


def test_banda_de_precio():
    assert banda_precio(_probs([0, 0, 0.05, 0.25, 0.5, 0.15, 0.05, 0, 0]), 4) == (3, 5)   # 0.9 ≥ 0.8
    assert banda_precio(_probs([0, 0, 0, 0.05, 0.9, 0.05, 0, 0, 0]), 4) == (4, 4)
    rng = np.random.default_rng(0)
    for _ in range(200):
        p = np.append(rng.dirichlet(np.ones(N_NIVELES)), rng.random())
        o = int(np.argmax(p[:N_NIVELES]))
        lo, hi = banda_precio(p, o)
        q = p[:N_NIVELES] / p[:N_NIVELES].sum()
        assert lo <= o <= hi and q[lo:hi + 1].sum() >= 0.8 - 1e-9


def test_precios_por_arco_texto():
    pasos = [
        dict(desde=1, hacia=5, accion="cotiza y pierde", precio_cotizado_usd_mi=2.48, rango=None),
        dict(desde=1, hacia=5, accion="cotiza y gana", precio_cotizado_usd_mi=2.40,
             rango=dict(piso=2.15, tope=2.48, referencia=2.40)),
        dict(desde=5, hacia=1, accion="viaja vacío", precio_cotizado_usd_mi=float("nan")),
    ]
    assert precios_por_arco(pasos) == "1-5: 2.15–2.48 (ref. 2.40)\n5-1: vacío"


def test_metricas_por_episodio_coherentes(d10):
    ctx = ev.ContextoEval(agente_chico(N), d10.market, None, d10.tm, d10.dist, d10.diesel, N, d10.avail, d10.p95)
    mundos = mundos_sim(d10.market, 40, semilla=12)
    nodos = np.arange(40) % N
    for nombre in ("DRL", "B1"):
        res = ev.ejecutar_lote(ctx.politica(nombre), ctx.env, mundos, nodos, [K_MAX] * 40, d10.dist)
        assert len(res) == 40
        for b, r in enumerate(res):
            assert r["ruta"][0] == nodos[b] and r["n_ganadas"] <= r["n_cotizaciones"] == len(r["niveles"])
            if r["violacion"]:
                assert r["margen"] == pytest.approx(PENALIZACION_USD + sum(p["reward_usd"] for p in r["pasos"]))
            else:
                assert r["ruta"][-1] == nodos[b] and len(r["ruta"]) - 1 <= K_MAX
                assert r["margen"] == pytest.approx(sum(p["reward_usd"] for p in r["pasos"]))
            if nombre == "DRL":
                assert all(p["rango"] is not None for p in r["pasos"] if p["accion"].startswith("cotiza"))


def test_cache_de_baselines_reproduce_el_calculo_directo(d10, tabla10, tmp_path):
    args = (d10.market, tabla10, d10.tm, d10.dist, d10.diesel, N, d10.avail, d10.p95)
    a1, a2 = agente_chico(N, 0), agente_chico(N, 1)
    cache = str(tmp_path / "c" / "sim.pkl")
    directo = ev.evaluar_mundos_simulados(a1, *args, M=8, verbose=False)
    primero = ev.evaluar_mundos_simulados(a1, *args, M=8, verbose=False, cache_path=cache)
    assert os.path.exists(cache)
    segundo = ev.evaluar_mundos_simulados(a2, *args, M=8, verbose=False, cache_path=cache)   # otro agente
    for df in (primero, segundo):
        assert list(df["metodo"]) == list(directo["metodo"]) and list(df["mundo"]) == list(directo["mundo"])
    np.testing.assert_allclose(primero["margen"], directo["margen"])
    no_drl = ~directo["metodo"].str.startswith("DRL")
    np.testing.assert_allclose(segundo.loc[no_drl, "margen"], directo.loc[no_drl, "margen"])
    np.testing.assert_allclose(segundo["ub_pi"], directo["ub_pi"])
    otro = ev.evaluar_mundos_simulados(a1, *args, M=4, verbose=False, cache_path=cache)   # otra firma: recalcula
    assert otro["mundo"].nunique() == 4


CHECKPOINT_10 = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "checkpoints",
                             "am_checkpoint_10nodes.pt")


def test_checkpoint_real_de_10_nodos_carga_y_cierra_ciclos(d10):
    """Red de seguridad del checkpoint entrenado: carga y, en los mundos de prueba, cierra el ciclo en ≥ 90 % de los
    casos sin superar UB-PI. Se omite si no hay un checkpoint con la cabeza de precio y las entradas actuales."""
    from transfer_learning import es_checkpoint_con_cotizacion, load_checkpoint_compatible
    from config import AM_D_H, AM_N_HEADS, AM_N_LAYERS, AM_D_FF
    if not es_checkpoint_con_cotizacion(CHECKPOINT_10):
        pytest.skip(f"no hay un checkpoint de 10 nodos con la cotización: {CHECKPOINT_10}")
    agent, _ = load_checkpoint_compatible(CHECKPOINT_10, N, AM_D_H, AM_N_HEADS, AM_N_LAYERS, AM_D_FF, verbose=False)
    df = ev.evaluar_mundos_simulados(agent, d10.market, None, d10.tm, d10.dist, d10.diesel, N, d10.avail, d10.p95,
                                     M=100, k_levels=(None,), verbose=False)
    drl = df[df["metodo"] == "DRL"]
    assert (~drl["violacion"]).mean() >= 0.9
    ok = ~drl["violacion"]
    assert (drl.loc[ok, "margen"] <= drl.loc[ok, "ub_pi"] + 1e-6).all()


def test_evaluacion_en_los_mundos_de_prueba(d10):
    """Los mundos reportados salen en octubre (período de prueba), y el resumen solo trae UB-PI como cota."""
    df = ev.evaluar_mundos_simulados(agente_chico(N), d10.market, None, d10.tm, d10.dist, d10.diesel, N, d10.avail,
                                     d10.p95, M=20, verbose=False)
    ini, fin = d10.market.periodo("test")
    assert df["dia_abs"].between(ini, fin - 1).all()
    assert set(df["metodo"]) == set(ev._METODOS) - {"B3"} and df["mundo"].nunique() == 20
    assert "ub_dp" not in df.columns and df["ub_pi"].notna().all()
    ok = ~df["violacion"]
    assert (df.loc[ok, "margen"] <= df.loc[ok, "ub_pi"] + 1e-6).all()
    res = ev.resumen_metodos(df)
    assert "UB-PI" in set(res["metodo"]) and "UB-DP" not in set(res["metodo"])
    assert "gap_ub_dp_pct" not in res.columns
