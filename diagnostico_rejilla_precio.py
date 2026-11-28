"""
diagnostico_rejilla_precio.py
=============================
¿Cuánto margen se pierde por cotizar solo en los 9 niveles fijos U_NIVELES (p10 … p90)?

No usa ningún modelo entrenado: con los datos de mercado y la fórmula de aceptación P(u) = (1 − u)^m, compara, lane-día
por lane-día, el valor esperado de UNA cotización en

    · la rejilla de 9 niveles (la del DRL y de la regla miope),
    · una rejilla fina dentro del mismo rango [p10, p90]   → pérdida por RESOLUCIÓN,
    · una rejilla fina en todo (0, 1)                      → pérdida por resolución + RANGO (poder cotizar < p10 o > p90).

El valor de una cotización en el percentil u es  Q(u) = (1 − u)^m · (R(p(u)) + Δ), con R el margen de la lane al precio p(u) =
F⁻¹(u) (problems_data.lane_reward) y Δ lo que vale ganar la carga más allá del tramo (V(j) − V(perder); Δ = 0 es la
cotización miope). Como Δ no se conoce sin la tabla, se barren varios (DELTAS, en USD): con Δ alto conviene cotizar más
bajo, y la pérdida de una rejilla gruesa cambia.

Las cargas del mundo simulado se sortean desde estos mismos lane-días, así que las pérdidas por lane-día son las que
vería el modelo en la simulación. Se ponderan por min(cargas, COTIZACIONES_MAX_POR_NODO) (las cotizaciones posibles en la
lane). Se separan los días de entrenamiento (desde el día DIA_MIN_ENTRENAMIENTO, para dejar fuera los días 0-13 con
conteos atípicos), los de validación (septiembre) y los de evaluación (octubre, desde TEST_START).

Uso:   python diagnostico_rejilla_precio.py --num-nodes 20 [--num-nodes 35 ...]
Salida: results/<N>nodes/pricing_val/tables/rejilla_precio.csv   (y una tabla en consola)
"""

import argparse
import os

import torch  # noqa: F401  (antes que pandas: en algunos Windows importar pandas primero rompe c10.dll de torch)
import numpy as np
import pandas as pd

from config import TRAIN_DAYS, TEST_START, COMPETENCIA_KAPPA, COMPETENCIA_M_MAX, COTIZACIONES_MAX_POR_NODO, MPG, \
    MARGINAL_COST_SIN_DIESEL
from market_process import MarketProcess, asegurar_parametros
from pricing import competidores_efectivos, interp_cuantiles, U_NIVELES_ARR
from problem_data import load_matrices

DELTAS = (0.0, 100.0, 300.0)        # USD: valor de ganar la carga más allá del tramo (Δ); 0 = regla miope
DIA_MIN_ENTRENAMIENTO = 14          # los días 0-13 tienen 3-8 veces más cargas que el resto (artefacto probable)
PASO_FINO = 0.005
_CHUNK = 20_000                     # lane-días por bloque (memoria)
_COLS_Q = ("min", "q1", "q2", "q3", "p90", "max")


def _margen(precio, dist, diesel):
    """lane_reward con carga: ingreso − costo (sin el redondeo a dólares, que no importa aquí)."""
    return precio * dist - dist * (diesel / MPG) - dist * MARGINAL_COST_SIN_DIESEL


def _mejor(u_grid, q6, m, dist, diesel, delta):
    """(valor, u*) del mejor u de `u_grid` para cada lane-día. q6 (L, 6); m, dist, diesel (L,)."""
    p = interp_cuantiles(np.broadcast_to(u_grid, (len(q6), len(u_grid))),
                         np.broadcast_to(q6[:, None, :], (len(q6), len(u_grid), 6)))
    Q = np.power(1.0 - u_grid[None, :], m[:, None]) * (_margen(p, dist[:, None], diesel[:, None]) + delta)
    k = Q.argmax(axis=1)
    return Q[np.arange(len(q6)), k], u_grid[k]


def diagnosticar(n: int) -> pd.DataFrame:
    asegurar_parametros()
    market = MarketProcess(n)
    _, _, _, dist_nn, diesel_nn, *_ = load_matrices(n)
    D = market.count_loads.shape[0]
    cargas = market.count_loads                                            # (D, N, N)
    d_idx, i_idx, j_idx = np.nonzero(cargas > 0)                           # lane-días con cargas
    keep = i_idx != j_idx
    d_idx, i_idx, j_idx = d_idx[keep], i_idx[keep], j_idx[keep]
    q6 = np.stack([market._q[c][d_idx, i_idx, j_idx] for c in _COLS_Q], axis=-1).astype(np.float64)
    m = competidores_efectivos(market.ltr_hub[d_idx, i_idx], market.ltr_ref, COMPETENCIA_KAPPA, COMPETENCIA_M_MAX)
    dist = np.asarray(dist_nn, dtype=np.float64)[i_idx, j_idx]
    diesel = np.asarray(diesel_nn, dtype=np.float64)[i_idx, j_idx]
    peso = np.minimum(cargas[d_idx, i_idx, j_idx], COTIZACIONES_MAX_POR_NODO).astype(np.float64)
    periodo = np.where(d_idx >= TEST_START, "evaluación", np.where(d_idx >= TRAIN_DAYS, "validación",
                       np.where(d_idx >= DIA_MIN_ENTRENAMIENTO, "entrenamiento", "excluido")))

    u9 = np.asarray(U_NIVELES_ARR, dtype=np.float64)
    u_res = np.arange(u9.min(), u9.max() + 1e-9, PASO_FINO)                 # fino dentro de [p10, p90]
    u_all = np.arange(PASO_FINO, 1.0, PASO_FINO)                            # fino en (0, 1)
    filas = []
    for delta in DELTAS:
        v9 = np.empty(len(q6)); vres = np.empty(len(q6)); vall = np.empty(len(q6)); ustar = np.empty(len(q6))
        u9star = np.empty(len(q6))
        for a in range(0, len(q6), _CHUNK):
            s = slice(a, a + _CHUNK)
            v9[s], u9star[s] = _mejor(u9, q6[s], m[s], dist[s], diesel[s], delta)
            vres[s], _ = _mejor(u_res, q6[s], m[s], dist[s], diesel[s], delta)
            vall[s], ustar[s] = _mejor(u_all, q6[s], m[s], dist[s], diesel[s], delta)
        for per in ("entrenamiento", "evaluación"):
            k = (periodo == per) & (v9 > 0)         # solo donde cotizar vale la pena (valor esperado positivo)
            w = peso[k]
            wavg = lambda x: float(np.average(x[k], weights=peso[k]))
            filas.append(dict(
                nodos=n, periodo=per, delta_usd=delta, lane_dias=int(k.sum()),
                valor_9_niveles=wavg(v9), valor_fino_mismo_rango=wavg(vres), valor_fino_todo=wavg(vall),
                perdida_resolucion_usd=wavg(vres - v9), perdida_resolucion_pct=100 * wavg(vres - v9) / wavg(v9),
                perdida_resolucion_y_rango_usd=wavg(vall - v9),
                perdida_resolucion_y_rango_pct=100 * wavg(vall - v9) / wavg(v9),
                frac_optimo_fino_bajo_p10=float(np.average(ustar[k] < u9.min() - 1e-9, weights=w)),
                frac_optimo_fino_sobre_p90=float(np.average(ustar[k] > u9.max() + 1e-9, weights=w)),
                u_optimo_fino_mediana=float(np.median(ustar[k])),
                m_medio=float(np.average(m[k], weights=w))))
    return pd.DataFrame(filas)


def main():
    ap = argparse.ArgumentParser(description="Pérdida de margen por la rejilla de 9 niveles de precio.")
    ap.add_argument("--num-nodes", type=int, action="append", help="se puede repetir; default 20")
    args = ap.parse_args()
    cwd = os.path.dirname(os.path.abspath(__file__))
    for n in args.num_nodes or [20]:
        df = diagnosticar(n)
        out = os.path.join(cwd, "results", f"{n}nodes", "pricing_val", "tables")
        os.makedirs(out, exist_ok=True)
        df.to_csv(os.path.join(out, "rejilla_precio.csv"), index=False)
        with pd.option_context("display.width", 200, "display.max_columns", 20, "display.float_format", "{:.2f}".format):
            print(f"\n=== {n} nodos ===")
            print(df[["periodo", "delta_usd", "lane_dias", "valor_9_niveles", "perdida_resolucion_usd",
                      "perdida_resolucion_pct", "perdida_resolucion_y_rango_usd", "perdida_resolucion_y_rango_pct",
                      "frac_optimo_fino_bajo_p10", "frac_optimo_fino_sobre_p90", "u_optimo_fino_mediana",
                      "m_medio"]].to_string(index=False))
        print(f"→ {os.path.join(out, 'rejilla_precio.csv')}")


if __name__ == "__main__":
    main()
