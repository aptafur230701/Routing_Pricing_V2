"""
validate_market_process.py
==========================
Validación del generador de realizaciones diarias (Paso 6 de
metodologia_realizaciones_mercado_reefer.md) y del índice de mercado por nodo (conservado de
proceso_exogeno_cargas_tarifas.md).

Se simulan miles de realizaciones con δ = 0 de una muestra de días y lanes de ENTRENAMIENTO (con
lanes de volumen alto, medio y bajo) y se revisa:

  1. Precios      — cuantiles de los precios sorteados frente a min, q1, q2, q3, p90 y max del
                    archivo de cada día/lane. Como el sorteo es la inversión de esos mismos
                    cuantiles, es casi una identidad: valida la implementación de
                    `market_process._interp_cuantiles`, no una propiedad estadística.
  2. Dispersión de cargas — razón varianza/media de las cargas sorteadas frente a la observada,
                    por grupo de volumen (deciles de la m promedio de cada lane). Si la simulada
                    es mucho mayor, r es demasiado pequeño; si es mucho menor, r es demasiado grande.
  3. Días sin cargas — fracción de lane-días sorteados con 0 ofertas frente a la fracción real de
                    lane-días con count_loads = 0, por grupo de volumen.
  4. Nivel de precios (δ) — desviación y cuantiles de δ (sorteado con muchas semillas) frente a los
                    de ΔF_t real (el cambio diario del factor nacional, días de ajuste).
  5. Persistencia del índice — pendiente (con intercepto) de índice(i, d+h) sobre índice(i, d),
                    juntando nodos, para h = 1, 3, 5, 7, frente a la pendiente que implica el ajuste
                    AR(1) (b_h = frac_var_lento + (1 − frac_var_lento) · ρ^h, la misma que usa B2P).

La comparación de política (DRL y baselines en mundos simulados vs replay fuera de muestra) no se
hace aquí: está en evaluation.py.

Uso
---
    python validate_market_process.py                    # 35 nodos
    python validate_market_process.py --nodos 194        # toda la red (lento y pesado en memoria)
    python validate_market_process.py --sorteos 200      # sorteos por lane-día de prueba

Salida (results/market_validation/): tablas .csv.
"""

import argparse
import os

import numpy as np
import pandas as pd

from config import TRAIN_DAYS
from fit_market_process import estimar_m, factor_nacional
from market_process import MarketProcess, asegurar_parametros, _interp_cuantiles
from problem_data import cargar_dias_mercado

_CWD = os.path.dirname(os.path.abspath(__file__))
_Q_PCT = np.array([0, 25, 50, 75, 90, 100])


def _dentro(real, lo, hi) -> np.ndarray:
    return (real >= lo) & (real <= hi)


def pendiente(indice: np.ndarray, h: int) -> float:
    """Pendiente (con intercepto) de la regresión de índice(i, d + h) sobre índice(i, d), juntando nodos."""
    x, y = indice[:-h].ravel(), indice[h:].ravel()
    x = x - x.mean()
    return float((x * (y - y.mean())).sum() / (x * x).sum())


def validar_precios(mp: MarketProcess, sorteos: int, rng: np.random.Generator) -> pd.DataFrame:
    """Comparación 1: cuantiles de los precios sorteados (δ = 0) frente a los del archivo."""
    N = mp.num_nodes
    sin_diag = ~np.eye(N, dtype=bool)
    con_cargas = np.argwhere(mp._con_cargas[:TRAIN_DAYS] & sin_diag[None])
    muestra = con_cargas[rng.choice(len(con_cargas), size=min(30, len(con_cargas)), replace=False)]

    filas = []
    for dia, i, j in muestra:
        cuantiles = np.array([mp._q[c][dia, i, j] for c in ("min", "q1", "q2", "q3", "p90", "max")])
        u = rng.random(sorteos)
        precios = _interp_cuantiles(u, np.tile(cuantiles, (sorteos, 1)))
        sim_q = np.percentile(precios, _Q_PCT)
        for k, p in enumerate(_Q_PCT):
            filas.append({
                "dia": int(dia), "origen": int(i), "destino": int(j), "percentil": p,
                "archivo": cuantiles[k], "sorteado": sim_q[k], "dif": sim_q[k] - cuantiles[k],
            })
    return pd.DataFrame(filas)


def validar_cargas(mp: MarketProcess, sorteos: int, rng: np.random.Generator) -> pd.DataFrame:
    """Comparaciones 2 y 3: dispersión y fracción de ceros, por grupo de volumen de lane."""
    N = mp.num_nodes
    sin_diag = ~np.eye(N, dtype=bool)
    n_grupos = len(mp.r)
    m_prom_lane = mp._m_stack[:TRAIN_DAYS].mean(axis=0)
    cortes = np.quantile(m_prom_lane[sin_diag], np.linspace(0, 1, n_grupos + 1)[1:-1])
    grupo = np.full((N, N), -1, dtype=np.int8)
    grupo[sin_diag] = np.digitize(m_prom_lane[sin_diag], cortes).astype(np.int8)

    filas = []
    for h in range(n_grupos):
        lanes_h = np.argwhere(grupo == h)
        if len(lanes_h) == 0:
            continue
        sub = lanes_h[rng.choice(len(lanes_h), size=min(50, len(lanes_h)), replace=False)]
        y_real, y_sim = [], []
        for i, j in sub:
            dia = int(rng.integers(0, TRAIN_DAYS))
            y_real.append(int(mp.count_loads[dia, i, j]))
            m, r = mp._m_stack[dia, i, j], mp._r_lane[i, j]
            if m < 1.0:
                y_sim.append(rng.poisson(max(m, 1e-12), size=sorteos))
            else:
                y_sim.append(rng.negative_binomial(r, r / (r + max(m, 1e-12)), size=sorteos))
        y_real = np.asarray(y_real, dtype=np.float64)
        y_sim = np.concatenate(y_sim)
        real_cargas = mp.count_loads[:TRAIN_DAYS][:, grupo == h]
        filas.append({
            "grupo h": h, "lanes": int((grupo == h).sum()), "r_h": float(mp.r[h]),
            "var/media real (todo train)": real_cargas.var() / max(real_cargas.mean(), 1e-9),
            "var/media sorteado": y_sim.var() / max(y_sim.mean(), 1e-9),
            "% ceros real (todo train)": 100 * (real_cargas == 0).mean(),
            "% ceros sorteado": 100 * (y_sim == 0).mean(),
        })
    return pd.DataFrame(filas)


def validar_delta(mp: MarketProcess, n_semillas: int, rng: np.random.Generator) -> pd.DataFrame:
    """Comparación 4: δ sorteado (muchas semillas) frente a ΔF_t real de los días de ajuste.

    F_t (Paso 3) se calcula sobre las 37,442 lanes de los 194 nodos (igual que en el ajuste de δ en
    fit_market_process.estimar_dispersion_delta), no sobre los `--nodos` que se estén validando: δ
    es un único escalar nacional, compararlo contra un F_t de un subconjunto de nodos no sería
    la misma cantidad."""
    semillas = rng.integers(0, 2**31 - 1, size=n_semillas)
    deltas = np.array([mp._delta(int(s)) for s in semillas])

    dias_full = cargar_dias_mercado()
    count_loads_full = dias_full["count_loads"][:TRAIN_DAYS]
    mean_rate_full = dias_full["mean"][:TRAIN_DAYS]
    n_total = count_loads_full.shape[1]
    universo_full = count_loads_full.sum(axis=0) > 0
    universo_indice_full = universo_full & ~np.eye(n_total, dtype=bool)
    F = factor_nacional(count_loads_full, mean_rate_full, universo_indice_full)
    delta_F = np.diff(F)

    pct = [5, 25, 50, 75, 95]
    return pd.DataFrame({
        "estadístico": ["desviación"] + [f"percentil {p}" for p in pct],
        "δ sorteado": [deltas.std()] + list(np.percentile(deltas, pct)),
        "ΔF_t real": [delta_F.std()] + list(np.percentile(delta_F, pct)),
    })


def validar_persistencia_indice(mp: MarketProcess) -> pd.DataFrame:
    """Comparación 5: pendiente empírica de índice(d+h) sobre índice(d) frente a la que implica
    el ajuste AR(1) (b_h = frac_var_lento + (1 − frac_var_lento) · ρ^h, la de B2P/encoder)."""
    indice_train = mp.indice_historico(0, TRAIN_DAYS)
    filas = []
    for h in (1, 3, 5, 7):
        emp = pendiente(indice_train, h)
        modelo = mp.frac_var_lento + (1.0 - mp.frac_var_lento) * mp.rho ** h
        filas.append({"h (días)": h, "pendiente empírica": emp, "b_h del ajuste (ρ, f)": modelo,
                       "dif": modelo - emp})
    return pd.DataFrame(filas)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--nodos", type=int, default=35, help="primeros N nodos a validar (máx. 194)")
    ap.add_argument("--sorteos", type=int, default=2000, help="sorteos por lane-día / semillas de muestra")
    ap.add_argument("--seed", type=int, default=20261101)
    ap.add_argument("--salida", default=os.path.join(_CWD, "results", "market_validation"))
    args = ap.parse_args()

    asegurar_parametros()
    mp = MarketProcess(args.nodos)
    rng = np.random.default_rng(args.seed)

    print(f"Validación del generador: {args.nodos} nodos, {args.sorteos} sorteos por lane-día/semilla, "
          f"muestra de días de entrenamiento (0-{TRAIN_DAYS - 1})")

    precios = validar_precios(mp, args.sorteos, rng)
    cargas = validar_cargas(mp, args.sorteos, rng)
    delta = validar_delta(mp, args.sorteos, rng)
    persistencia = validar_persistencia_indice(mp)

    os.makedirs(args.salida, exist_ok=True)
    precios.to_csv(os.path.join(args.salida, "precios_cuantiles.csv"), index=False)
    cargas.to_csv(os.path.join(args.salida, "cargas_por_grupo.csv"), index=False)
    delta.to_csv(os.path.join(args.salida, "delta_vs_factor_nacional.csv"), index=False)
    persistencia.to_csv(os.path.join(args.salida, "persistencia_indice.csv"), index=False)

    pd.set_option("display.width", 200, "display.max_columns", 20, "display.float_format", "{:.4f}".format)
    print("\n== 1. Precios: cuantiles sorteados vs archivo (resumen por percentil) ==")
    print(precios.groupby("percentil")["dif"].agg(["mean", "std", "max"]).to_string())
    print("\n== 2-3. Cargas: dispersión y fracción de ceros por grupo de volumen ==")
    print(cargas.to_string(index=False))
    print("\n== 4. δ sorteado vs ΔF_t real (factor nacional, días de ajuste) ==")
    print(delta.to_string(index=False))
    print("\n== 5. Persistencia del índice: pendiente empírica vs la que implica el ajuste AR(1) ==")
    print(persistencia.to_string(index=False))
    print("\n== 6. Política: ver evaluation.py (mundos simulados y replay fuera de muestra) ==")
    print(f"\nTablas en: {args.salida}")


if __name__ == "__main__":
    main()
