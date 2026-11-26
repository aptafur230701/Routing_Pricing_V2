"""Rango de precio por arco del DRL — evaluation.banda_precio y precios_por_arco."""
import torch  # noqa: F401  (antes que pandas en Windows)
import numpy as np

from evaluation import banda_precio, precios_por_arco
from pricing import N_NIVELES


def _probs(p):
    """Probabilidades de las 9 opciones de nivel + VACÍO (VACÍO al final)."""
    return np.array(list(p) + [0.0])


def test_banda_crece_hacia_el_vecino_mas_probable_hasta_80pct():
    p = _probs([0, 0, 0.05, 0.25, 0.5, 0.15, 0.05, 0, 0])
    assert banda_precio(p, 4) == (3, 5)               # 0.5 + 0.25 + 0.15 = 0.9 ≥ 0.8


def test_banda_un_solo_nivel_si_el_modelo_esta_seguro():
    p = _probs([0, 0, 0, 0.05, 0.9, 0.05, 0, 0, 0])
    assert banda_precio(p, 4) == (4, 4)


def test_banda_ignora_vacio_y_contiene_el_nivel_elegido():
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
