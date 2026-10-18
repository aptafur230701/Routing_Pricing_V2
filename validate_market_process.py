"""
validate_market_process.py
==========================
Validación del simulador (fase 6 de proceso_exogeno_cargas_tarifas.md).

Como los parámetros se estiman con todos los días, la validación es DENTRO DE
MUESTRA: se simulan R réplicas (1.000 por defecto) del periodo completo con los
parámetros estimados y se comparan con el histórico.

  1. Cargas por lane  — media diaria y proporción de días en cero, agrupadas por
                        grupo de volumen h. Se lee: ¿el valor real cae dentro del
                        intervalo del 90 % de las réplicas?
  2. Tarifas          — cuantiles 10, 50 y 90 de la tarifa por milla por grupo de
                        distancia (cuartiles), con QQ-plot, y tarifa media real vs
                        simulada por mes (muestra la tendencia que el remuestreo no
                        reproduce).
  3. Política         — APLAZADA hasta que exista el modo de escenarios (Paso 6).

Uso
---
    python validate_market_process.py                    # 35 nodos, 1000 réplicas
    python validate_market_process.py --nodos 194        # toda la red (lento)
    python validate_market_process.py --replicas 200     # prueba rápida

Salida (results/market_validation/): tablas .csv y gráficos .png.
"""

import argparse
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import MARKET_SEED_EVAL, MARKET_N_GRUPOS
from fit_market_process import params_path
from market_process import MarketProcess, asegurar_parametros

_CWD = os.path.dirname(os.path.abspath(__file__))
_Q = (10, 50, 90)


def _dentro(real, lo, hi) -> np.ndarray:
    return (real >= lo) & (real <= hi)


def cargar_historico(num_nodes: int) -> dict:
    """Histórico recortado a num_nodes, restringido a los días usados en el ajuste."""
    data_dir = os.path.join(_CWD, "datos")
    with np.load(params_path()) as z:
        dias = z["dias_usados"]
        fechas = z["fechas_ajuste"]
    loads = np.load(os.path.join(data_dir, "load_availability.npy"))[dias, :num_nodes, :num_nodes]
    rate  = np.load(os.path.join(data_dir, "rate.npy"))[dias, :num_nodes, :num_nodes]
    dist  = pd.read_csv(os.path.join(data_dir, "distance.csv"), index_col=0)\
              .iloc[:num_nodes, :num_nodes].to_numpy(dtype=float)
    return dict(loads=loads.astype(np.float64), rate=rate.astype(np.float64), dist=dist,
                dias=dias, fechas=fechas.astype("datetime64[D]"))


def simular(mp: MarketProcess, hist: dict, replicas: int, seed: int) -> dict:
    """Simula `replicas` veces el periodo completo y acumula los estadísticos."""
    n, D = mp.num_nodes, len(hist["dias"])
    grupo_dist, uni = hist["grupo_dist"], hist["universo"]
    mes = hist["mes"]                                    # (D,) índice de mes de cada día
    n_mes = int(mes.max()) + 1
    rng = np.random.default_rng(seed)
    nodos = np.repeat(np.arange(n), D)
    dias  = np.tile(hist["dias"], n)

    media_lane = np.empty((replicas, n, n), dtype=np.float32)
    cero_lane  = np.empty((replicas, n, n), dtype=np.float32)
    cuant = np.empty((replicas, MARKET_N_GRUPOS, len(_Q)))
    tarifa_mes = np.empty((replicas, n_mes))
    muestras_qq = [[] for _ in range(MARKET_N_GRUPOS)]

    t0 = time.time()
    for r in range(replicas):
        loads, rates = mp.sample_arrivals(nodos, dias, rng)      # (n·D, n)
        loads = loads.reshape(n, D, n)
        rates = rates.reshape(n, D, n).transpose(1, 0, 2)        # → (D, origen, destino)
        media_lane[r] = loads.mean(axis=1)
        cero_lane[r]  = (loads == 0).mean(axis=1)
        for g in range(MARKET_N_GRUPOS):
            v = rates[:, uni & (grupo_dist == g)]                # (D, L_g) lane-días
            cuant[r, g] = np.percentile(v, _Q)
            if r < 10:                                           # submuestra para el QQ-plot
                muestras_qq[g].append(rng.choice(v.ravel(), size=20000, replace=False))
        rates_u = rates[:, uni]                                  # (D, L)
        tarifa_mes[r] = [rates_u[mes == m].mean() for m in range(n_mes)]
        if (r + 1) % max(1, replicas // 10) == 0:
            print(f"  réplica {r + 1:>5}/{replicas}  ({time.time() - t0:.0f} s)")
    return dict(media_lane=media_lane, cero_lane=cero_lane, cuant=cuant,
                tarifa_mes=tarifa_mes, muestras_qq=[np.concatenate(m) for m in muestras_qq])


def comparar_cargas(mp, hist, sim) -> pd.DataFrame:
    """Comparación 1: media diaria y proporción de ceros por grupo de volumen h."""
    grupo, uni = mp.params.grupo, hist["universo"]
    media_real = hist["loads"].mean(axis=0)
    cero_real  = (hist["loads"] == 0).mean(axis=0)
    filas = []
    for h in range(len(mp.params.phi)):
        sel = uni & (grupo == h)
        for nombre, real_lane, sim_lane in (
            ("media diaria de cargas", media_real, sim["media_lane"]),
            ("proporción de días en cero", cero_real, sim["cero_lane"]),
        ):
            real = real_lane[sel].mean()
            reps = sim_lane[:, sel].mean(axis=1)                 # estadístico de grupo por réplica
            lo, hi = np.percentile(reps, [5, 95])
            lane_lo, lane_hi = np.percentile(sim_lane[:, sel], [5, 95], axis=0)
            cobertura = _dentro(real_lane[sel], lane_lo, lane_hi).mean()
            filas.append({
                "grupo h": h, "lanes": int(sel.sum()), "phi_h": mp.params.phi[h],
                "estadístico": nombre, "real": real, "sim (media)": reps.mean(),
                "sim IC90 inf": lo, "sim IC90 sup": hi,
                "real dentro del IC90 (grupo)": bool(_dentro(real, lo, hi)),
                "% lanes con real dentro de su IC90": 100 * cobertura,
            })
    return pd.DataFrame(filas)


def comparar_tarifas(hist, sim) -> tuple:
    """Comparación 2: cuantiles por grupo de distancia y tarifa media por mes."""
    uni, gd = hist["universo"], hist["grupo_dist"]
    filas = []
    for g in range(MARKET_N_GRUPOS):
        v = hist["rate"][:, uni & (gd == g)]
        real = np.percentile(v, _Q)
        d = hist["dist"][uni & (gd == g)]
        for k, q in enumerate(_Q):
            lo, hi = np.percentile(sim["cuant"][:, g, k], [5, 95])
            filas.append({
                "grupo distancia": g + 1, "millas": f"{d.min():.0f}–{d.max():.0f}", "cuantil": q,
                "real": real[k], "sim (media)": sim["cuant"][:, g, k].mean(),
                "sim IC90 inf": lo, "sim IC90 sup": hi,
                "real dentro del IC90": bool(_dentro(real[k], lo, hi)),
            })
    cuantiles = pd.DataFrame(filas)

    mes = hist["mes"]
    real_mes = np.array([hist["rate"][mes == m][:, uni].mean() for m in range(mes.max() + 1)])
    etiquetas = pd.Series(hist["fechas"]).astype(str).str[:7].drop_duplicates().tolist()
    lo, hi = np.percentile(sim["tarifa_mes"], [5, 95], axis=0)
    mensual = pd.DataFrame({
        "mes": etiquetas, "real": real_mes, "sim (media)": sim["tarifa_mes"].mean(axis=0),
        "sim IC90 inf": lo, "sim IC90 sup": hi,
        "real dentro del IC90": _dentro(real_mes, lo, hi),
        "real vs sim (%)": 100 * (real_mes / sim["tarifa_mes"].mean(axis=0) - 1),
    })
    return cuantiles, mensual


def graficar(mp, hist, sim, cargas, cuantiles, mensual, salida: str) -> None:
    # 1. Cargas por grupo de volumen
    fig, axs = plt.subplots(1, 2, figsize=(11, 4))
    for ax, nombre in zip(axs, ("media diaria de cargas", "proporción de días en cero")):
        df = cargas[cargas["estadístico"] == nombre]
        x = df["grupo h"].to_numpy()
        yerr = np.vstack([df["sim (media)"] - df["sim IC90 inf"], df["sim IC90 sup"] - df["sim (media)"]])
        ax.errorbar(x, df["sim (media)"], yerr=yerr, fmt="o", capsize=5, label="simulado (media, IC 90 %)")
        ax.plot(x, df["real"], "r*", ms=12, label="histórico")
        ax.set_xticks(x); ax.set_xlabel("grupo de volumen h (0 = menor)"); ax.set_title(nombre)
        if "media" in nombre:
            ax.set_yscale("log")
        ax.legend()
    fig.suptitle("Cargas por lane — real vs réplicas simuladas")
    fig.tight_layout(); fig.savefig(os.path.join(salida, "cargas_por_grupo.png"), dpi=130); plt.close(fig)

    # 2. QQ-plot de tarifas por grupo de distancia
    qs = np.arange(1, 100)
    fig, axs = plt.subplots(1, MARKET_N_GRUPOS, figsize=(4 * MARKET_N_GRUPOS, 4))
    for g, ax in enumerate(axs):
        real = np.percentile(hist["rate"][:, hist["universo"] & (hist["grupo_dist"] == g)], qs)
        simq = np.percentile(sim["muestras_qq"][g], qs)
        ax.plot(real, simq, ".", ms=4)
        lim = [min(real.min(), simq.min()), max(real.max(), simq.max())]
        ax.plot(lim, lim, "k--", lw=1)
        ax.set_xlabel("cuantiles históricos ($/milla)"); ax.set_ylabel("cuantiles simulados")
        ax.set_title(f"Grupo de distancia {g + 1}")
    fig.suptitle("QQ-plot de la tarifa por milla (percentiles 1–99)")
    fig.tight_layout(); fig.savefig(os.path.join(salida, "tarifas_qq.png"), dpi=130); plt.close(fig)

    # 3. Tarifa media por mes
    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.arange(len(mensual))
    ax.fill_between(x, mensual["sim IC90 inf"], mensual["sim IC90 sup"], alpha=0.3, label="simulado IC 90 %")
    ax.plot(x, mensual["sim (media)"], "b-o", label="simulado (media)")
    ax.plot(x, mensual["real"], "r-*", ms=10, label="histórico")
    ax.set_xticks(x); ax.set_xticklabels(mensual["mes"]); ax.set_ylabel("tarifa media ($/milla)")
    ax.set_title("Tarifa media por mes: tendencia no reproducida")
    ax.legend(); fig.tight_layout()
    fig.savefig(os.path.join(salida, "tarifa_mensual.png"), dpi=130); plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--nodos", type=int, default=35, help="primeros N nodos a validar (máx. 194)")
    ap.add_argument("--replicas", type=int, default=1000, help="réplicas del periodo completo")
    ap.add_argument("--seed", type=int, default=MARKET_SEED_EVAL)
    ap.add_argument("--salida", default=os.path.join(_CWD, "results", "market_validation"))
    args = ap.parse_args()

    asegurar_parametros()
    mp = MarketProcess(args.nodos)
    hist = cargar_historico(args.nodos)
    n = args.nodos
    hist["universo"] = (hist["loads"].sum(axis=0) > 0) & ~np.eye(n, dtype=bool)   # sin la diagonal (i, i)
    # Grupos de distancia (cuartiles) entre las lanes del universo.
    cortes = np.quantile(hist["dist"][hist["universo"]], np.linspace(0, 1, MARKET_N_GRUPOS + 1)[1:-1])
    hist["grupo_dist"] = np.digitize(hist["dist"], cortes)
    meses = hist["fechas"].astype("datetime64[M]")
    hist["mes"] = np.searchsorted(np.unique(meses), meses)

    print(f"Validación dentro de muestra: {n} nodos, {len(hist['dias'])} días, {args.replicas} réplicas")
    sim = simular(mp, hist, args.replicas, args.seed)

    cargas = comparar_cargas(mp, hist, sim)
    cuantiles, mensual = comparar_tarifas(hist, sim)

    os.makedirs(args.salida, exist_ok=True)
    cargas.to_csv(os.path.join(args.salida, "cargas_por_grupo.csv"), index=False)
    cuantiles.to_csv(os.path.join(args.salida, "tarifas_cuantiles.csv"), index=False)
    mensual.to_csv(os.path.join(args.salida, "tarifa_mensual.csv"), index=False)
    graficar(mp, hist, sim, cargas, cuantiles, mensual, args.salida)

    pd.set_option("display.width", 200, "display.max_columns", 20, "display.float_format", "{:.4f}".format)
    print("\n== 1. Cargas por lane (por grupo de volumen h) ==")
    print(cargas.to_string(index=False))
    print("\n== 2a. Tarifas: cuantiles por grupo de distancia ==")
    print(cuantiles.to_string(index=False))
    print("\n== 2b. Tarifa media por mes ==")
    print(mensual.to_string(index=False))
    print("\n== 3. Política: APLAZADA (requiere el modo de escenarios del Paso 6) ==")
    print(f"\nTablas y gráficos en: {args.salida}")
    print("Nota: validación dentro de muestra (parámetros estimados con todos los días); "
          "el diagnóstico es de ajuste, no de generalización.")


if __name__ == "__main__":
    main()
