"""
pricing.py
==========
Cotización conjunta (especificacion_fase0_ruteo_pricing.md §3 y §6): niveles de precio por lane en
escala de percentiles, número efectivo de competidores m(LTR), aceptación P(ganar | u_k) = (1 − u_k)^m
y las reglas de precio fijas (mediana, break-even + margen, miope óptima). Solo numpy y config: lo
usan market_process, el entorno, los baselines y las DP.

El precio ya no se calcula después del ruteo: en cada nodo la política elige destino y nivel de precio
(el precio de reserva p* con las cabezas Ṽ A/B se eliminó en la Fase 3).
"""

import numpy as np

from config import (
    MPG, MARGINAL_COST_SIN_DIESEL, U_NIVELES, COMPETENCIA_KAPPA, COMPETENCIA_M_MAX, BE_MARGEN,
)


# ═════════════════════════════════════════════════════════════════════════════
# Cotización conjunta
# ═════════════════════════════════════════════════════════════════════════════

PCT_CUANTILES = np.array([0.0, 0.25, 0.5, 0.75, 0.90, 1.0])   # percentiles de [min, q1, q2, q3, p90, max]
U_NIVELES_ARR = np.asarray(U_NIVELES, dtype=np.float64)       # (K,) percentil de cada nivel de precio
N_NIVELES = len(U_NIVELES_ARR)
REGLAS_PRECIO = ("mediana", "be_margen", "miope")


def interp_cuantiles(u: np.ndarray, cuantiles: np.ndarray) -> np.ndarray:
    """Inversión de cuantiles por interpolación lineal (Paso 4b-2 de la metodología).

    u : (...,) en [0, 1]. cuantiles : (..., 6) con los puntos [min, q1, q2, q3, p90, max], que se
    interpolan en los percentiles `PCT_CUANTILES` = [0, .25, .5, .75, .90, 1]. Devuelve (...,).
    Es la misma función que convierte en dólares el percentil sorteado de la tarifa
    (market_process) y los percentiles fijos de los niveles de precio (`precios_niveles`).
    """
    seg = np.clip(np.searchsorted(PCT_CUANTILES, u, side="right") - 1, 0, len(PCT_CUANTILES) - 2)
    p0, p1 = PCT_CUANTILES[seg], PCT_CUANTILES[seg + 1]
    v0 = np.take_along_axis(cuantiles, seg[..., None], axis=-1)[..., 0]
    v1 = np.take_along_axis(cuantiles, (seg + 1)[..., None], axis=-1)[..., 0]
    t = np.where(p1 > p0, (u - p0) / np.maximum(p1 - p0, 1e-12), 0.0)
    return v0 + t * (v1 - v0)


def precios_niveles(cuantiles: np.ndarray, delta=0.0, u_niveles: np.ndarray = U_NIVELES_ARR) -> np.ndarray:
    """USD/milla de cada nivel de precio: F⁻¹_{lane,día}(u_k) · e^δ.

    cuantiles : (..., 6) de la lane y el día (ordenados de menor a mayor).
    delta     : corrimiento nacional del episodio, escalar o broadcastable a cuantiles.shape[:-1].
    Devuelve (..., K), no decreciente en K si los cuantiles están ordenados.
    """
    c = np.asarray(cuantiles, dtype=np.float64)
    u = np.asarray(u_niveles, dtype=np.float64)
    forma = c.shape[:-1] + u.shape
    p = interp_cuantiles(np.broadcast_to(u, forma), np.broadcast_to(c[..., None, :], forma + (c.shape[-1],)))
    return p * np.exp(np.asarray(delta, dtype=np.float64))[..., None]


def competidores_efectivos(ltr, ltr_ref: float, kappa: float = COMPETENCIA_KAPPA,
                           m_max: float = COMPETENCIA_M_MAX) -> np.ndarray:
    """m = clip(1 + κ · LTR_REF / LTR, 1, M_MAX): competidores efectivos por carga en el nodo de origen.

    LTR alto (más cargas que camiones) ⇒ m → 1. LTR bajo ⇒ m alto, con tope M_MAX. Un LTR ≤ 0 (hub
    sin dato) queda en el tope.
    """
    ltr = np.asarray(ltr, dtype=np.float64)
    m = 1.0 + kappa * ltr_ref / np.maximum(ltr, 1e-12)
    return np.clip(m, 1.0, m_max)


def prob_ganar(u_nivel, m) -> np.ndarray:
    """P(ganar | u_k) = (1 − u_k)^m: probabilidad de que los m competidores coticen más caro."""
    return np.power(1.0 - np.asarray(u_nivel, dtype=np.float64), np.asarray(m, dtype=np.float64))


def percentil_competidor(U, m) -> np.ndarray:
    """Percentil del mejor de m competidores uniformes a partir de un uniforme U: u_comp ~ Beta(1, m).

    u_comp = 1 − (1 − U)^(1/m). Con m = 1, u_comp = U: el competidor cotiza la tarifa sorteada.
    """
    return 1.0 - np.power(1.0 - np.asarray(U, dtype=np.float64), 1.0 / np.asarray(m, dtype=np.float64))


def gana(u_nivel, u_comp) -> np.ndarray:
    """La cotización en el percentil u_nivel gana si queda por debajo del mejor competidor.

    Se compara en percentiles (no en dólares) para evitar empates en lanes con cuantiles repetidos.
    El nivel 0 (el mínimo del mercado) gana siempre.
    """
    u_nivel = np.asarray(u_nivel, dtype=np.float64)
    return (u_nivel <= 0.0) | (u_nivel < np.asarray(u_comp, dtype=np.float64))


def indice_nivel(u: float, u_niveles: np.ndarray = U_NIVELES_ARR) -> int:
    """Índice en U_NIVELES del nivel con percentil `u` (p. ej. 0.5 → el nivel de la mediana)."""
    idx = np.flatnonzero(np.isclose(u_niveles, u))
    if idx.size == 0:
        raise ValueError(f"U_NIVELES no tiene el percentil {u}")
    return int(idx[0])


def nivel_regla(regla: str, precios: np.ndarray, dist, costo, m,
                u_niveles: np.ndarray = U_NIVELES_ARR, margen: float = BE_MARGEN) -> np.ndarray:
    """Nivel (índice en U_NIVELES) que cotiza una regla fija de precio.

    precios : (..., K) USD/milla por nivel (`precios_niveles`).
    dist, costo, m : broadcastables a precios.shape[:-1].
    regla :
      · "mediana"   — u = 0.5.
      · "be_margen" — el nivel más bajo con precio ≥ break-even + margen; si ninguno llega, el último.
      · "miope"     — argmax_k (1 − u_k)^m · (p_k · dist − costo): optimiza contra el mercado solo el
                      tramo actual, sin mirar el futuro.
    Devuelve (...,) int64.
    """
    precios = np.asarray(precios, dtype=np.float64)
    forma = precios.shape[:-1]
    dist = np.broadcast_to(np.asarray(dist, dtype=np.float64), forma)
    costo = np.broadcast_to(np.asarray(costo, dtype=np.float64), forma)
    if regla == "mediana":
        return np.full(forma, indice_nivel(0.5, u_niveles), dtype=np.int64)
    if regla == "be_margen":
        piso = costo / np.maximum(dist, 1e-12) + margen
        ok = precios >= piso[..., None]
        return np.where(ok.any(axis=-1), ok.argmax(axis=-1), precios.shape[-1] - 1).astype(np.int64)
    if regla == "miope":
        m = np.broadcast_to(np.asarray(m, dtype=np.float64), forma)
        valor = prob_ganar(u_niveles, m[..., None]) * (precios * dist[..., None] - costo[..., None])
        return valor.argmax(axis=-1).astype(np.int64)
    raise ValueError(f"regla de precio desconocida: {regla!r} (opciones: {REGLAS_PRECIO})")


# ═════════════════════════════════════════════════════════════════════════════
# Utilidades compartidas
# ═════════════════════════════════════════════════════════════════════════════

def lane_cost_row(distance_row: np.ndarray, diesel_row: np.ndarray) -> np.ndarray:
    """Costo en USD de recorrer cada lane saliente (misma fórmula que problem_data.lane_reward); sirve con (N,) o (N, N)."""
    return distance_row * (diesel_row / MPG) + distance_row * MARGINAL_COST_SIN_DIESEL


def percentil_historico(valor: float, serie: np.ndarray, cargas: np.ndarray = None) -> float:
    """Percentil (0-100) de `valor` dentro de `serie` (fracción de la serie menor o igual al valor).

    cargas : si se pasa, solo cuentan los días con carga (> 0): el histórico trae tarifa también en días
        sin carga, que el camión nunca habría visto ofertada.
    """
    serie = np.asarray(serie, dtype=float)
    ok = np.isfinite(serie)
    if cargas is not None:
        ok &= np.asarray(cargas) > 0
    serie = serie[ok]
    return float("nan") if serie.size == 0 else 100.0 * float(np.mean(serie <= valor))

