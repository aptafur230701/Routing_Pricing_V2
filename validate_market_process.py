"""
validate_market_process.py
==========================
Validación del simulador (fase 6 de proceso_exogeno_cargas_tarifas.md).

Los parámetros se estiman con los primeros TRAIN_DAYS días, así que esta validación es
DENTRO DE MUESTRA: se simulan R mundos (200 por defecto) del periodo de ajuste (`sample_world`, el
modelo con estado de mercado: lento fijo + rápido AR(1)) y se comparan con ese mismo histórico. La
prueba fuera de muestra es el replay de los días de evaluación.

  1. Cargas por lane  — media diaria y proporción de días en cero, agrupadas por
                        grupo de volumen h. Se lee: ¿el valor real cae dentro del
                        intervalo del 90 % de las réplicas?
  2. Tarifas          — cuantiles 10, 50 y 90 de la tarifa por milla POR LANE (real vs
                        simulado: sesgo, error absoluto, correlación y cobertura del IC90 de
                        las réplicas), por grupo de distancia (cuartiles) con QQ-plot, y tarifa
                        media real vs simulada por mes (muestra la tendencia del nivel, que el
                        modelo no reproduce). Se reporta el % de tarifas recortadas al mínimo
                        histórico de la lane.
  3. Persistencia     — pendiente de la regresión (con intercepto) de índice(i, d + h) sobre
                        índice(i, d), juntando nodos, para h = 1, 3, 5 y 7: histórico vs mundos
                        simulados. El histórico incluye la deriva del nivel lento durante los
                        días de ajuste; en un mundo el lento queda fijo, así que a h largo se
                        espera una pendiente simulada algo mayor.
  4. Política         — no se hace aquí: está en evaluation.py (mundos simulados y replay
                        fuera de muestra de los días de evaluación).

Uso
---
    python validate_market_process.py                    # 35 nodos, 200 réplicas
    python validate_market_process.py --nodos 194        # toda la red (lento y pesado en memoria)
    python validate_market_process.py --replicas 50      # prueba rápida

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
_H = (1, 3, 5, 7)   # horizontes (días) de la persistencia del índice


def _dentro(real, lo, hi) -> np.ndarray:
    return (real >= lo) & (real <= hi)


def cargar_historico(num_nodes: int) -> dict:
    """Histórico recortado a num_nodes, restringido a los días usados en el ajuste."""
    data_dir = os.path.join(_CWD, "datos")
    with np.load(params_path()) as z:
        dias = z["dias_usados"]
        fechas = z["fechas_ajuste"]
        indice = z["indice"][:, :num_nodes]
    loads = np.load(os.path.join(data_dir, "load_availability.npy"))[dias, :num_nodes, :num_nodes]
    rate  = np.load(os.path.join(data_dir, "rate.npy"))[dias, :num_nodes, :num_nodes]
    dist  = pd.read_csv(os.path.join(data_dir, "distance.csv"), index_col=0)\
              .iloc[:num_nodes, :num_nodes].to_numpy(dtype=float)
    return dict(loads=loads.astype(np.float64), rate=rate.astype(np.float64), dist=dist,
                dias=dias, fechas=fechas.astype("datetime64[D]"), indice=indice.astype(np.float64))


def pendiente(indice: np.ndarray, h: int) -> float:
    """Pendiente (con intercepto) de la regresión de índice(i, d + h) sobre índice(i, d), juntando nodos."""
    x, y = indice[:-h].ravel(), indice[h:].ravel()
    x = x - x.mean()
    return float((x * (y - y.mean())).sum() / (x * x).sum())


def simular(mp: MarketProcess, hist: dict, replicas: int, seed: int) -> dict:
    """Simula `replicas` mundos del periodo de ajuste (sample_world) y acumula los estadísticos."""
    n, D = mp.num_nodes, len(hist["dias"])
    if not np.array_equal(hist["dias"], np.arange(D)):
        raise ValueError("la validación simula los días 0 … D−1: los días de ajuste deben ser contiguos desde 0")
    grupo_dist, uni = hist["grupo_dist"], hist["universo"]
    mes = hist["mes"]                                    # (D,) índice de mes de cada día
    n_mes = int(mes.max()) + 1
    rng = np.random.default_rng(seed)
    semillas = rng.integers(0, 2**31 - 1, size=replicas)

    media_lane = np.empty((replicas, n, n), dtype=np.float32)
    cero_lane  = np.empty((replicas, n, n), dtype=np.float32)
    cuant = np.empty((replicas, MARKET_N_GRUPOS, len(_Q)))
    cuant_lane = np.empty((replicas, len(_Q), n, n), dtype=np.float32)
    tarifa_mes = np.empty((replicas, n_mes))
    pend = np.empty((replicas, len(_H)))
    muestras_qq = [[] for _ in range(MARKET_N_GRUPOS)]

    mp.reiniciar_recorte()
    t0 = time.time()
    for r in range(replicas):
        loads, rates, indice = mp.sample_world(int(semillas[r]), D, 0, devolver_indice=True)   # (D, n, n), (D, n)
        media_lane[r] = loads.mean(axis=0)
        cero_lane[r]  = (loads == 0).mean(axis=0)
        cuant_lane[r] = np.percentile(rates, _Q, axis=0)
        for g in range(MARKET_N_GRUPOS):
            v = rates[:, uni & (grupo_dist == g)]                # (D, L_g) lane-días
            cuant[r, g] = np.percentile(v, _Q)
            if r < 10:                                           # submuestra para el QQ-plot
                muestras_qq[g].append(rng.choice(v.ravel(), size=min(20000, v.size), replace=False))
        rates_u = rates[:, uni]                                  # (D, L)
        tarifa_mes[r] = [rates_u[mes == m].mean() for m in range(n_mes)]
        pend[r] = [pendiente(indice, h) for h in _H]
        if (r + 1) % max(1, replicas // 10) == 0:
            print(f"  réplica {r + 1:>5}/{replicas}  ({time.time() - t0:.0f} s)")
    return dict(media_lane=media_lane, cero_lane=cero_lane, cuant=cuant, cuant_lane=cuant_lane,
                tarifa_mes=tarifa_mes, pendientes=pend, recorte=mp.estadisticas_recorte(),
                muestras_qq=[np.concatenate(m) for m in muestras_qq])


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


def comparar_cuantiles_lane(hist, sim) -> tuple:
    """Comparación 2a: cuantiles 10, 50 y 90 de la tarifa POR LANE, real vs simulado.

    Cada réplica da un cuantil por lane con D días (como el histórico); `sim` es el promedio de las réplicas
    y [IC90 inf, IC90 sup] sus percentiles 5 y 95. Devuelve (resumen por cuantil, tabla por lane)."""
    uni = hist["universo"]
    real = np.percentile(hist["rate"], _Q, axis=0)                       # (3, n, n)
    cl = sim["cuant_lane"]
    media, lo, hi = cl.mean(axis=0), np.percentile(cl, 5, axis=0), np.percentile(cl, 95, axis=0)
    origen, destino = np.where(uni)
    filas, resumen = [], []
    for k, q in enumerate(_Q):
        r, m = real[k][uni], media[k][uni]
        dentro = (r >= lo[k][uni]) & (r <= hi[k][uni])
        resumen.append({
            "cuantil": q, "lanes": int(uni.sum()), "real (media)": r.mean(), "sim (media)": m.mean(),
            "sesgo (sim − real)": (m - r).mean(), "error absoluto medio": np.abs(m - r).mean(),
            "correlación entre lanes": np.corrcoef(r, m)[0, 1],
            "% lanes con |dif| < 0.05": 100 * (np.abs(m - r) < 0.05).mean(),
            "% lanes con real dentro del IC90 de las réplicas": 100 * dentro.mean(),
        })
        filas.append(pd.DataFrame({
            "origen": origen, "destino": destino, "millas": hist["dist"][uni], "cuantil": q,
            "real": r, "sim (media)": m, "sim IC90 inf": lo[k][uni], "sim IC90 sup": hi[k][uni],
        }))
    return pd.DataFrame(resumen), pd.concat(filas, ignore_index=True)


def comparar_persistencia(hist, sim) -> pd.DataFrame:
    """Comparación 3: pendiente de índice(i, d + h) sobre índice(i, d), histórico vs mundos simulados."""
    filas = []
    for k, h in enumerate(_H):
        real = pendiente(hist["indice"], h)
        lo, hi = np.percentile(sim["pendientes"][:, k], [5, 95])
        filas.append({
            "h (días)": h, "real": real, "sim (media)": sim["pendientes"][:, k].mean(),
            "sim IC90 inf": lo, "sim IC90 sup": hi, "real dentro del IC90": bool(_dentro(real, lo, hi)),
        })
    return pd.DataFrame(filas)


def graficar(mp, hist, sim, cargas, cuantiles, mensual, persistencia, salida: str) -> None:
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

    # 4. Persistencia del índice
    fig, ax = plt.subplots(figsize=(6, 4))
    x = persistencia["h (días)"].to_numpy()
    ax.fill_between(x, persistencia["sim IC90 inf"], persistencia["sim IC90 sup"], alpha=0.3, label="simulado IC 90 %")
    ax.plot(x, persistencia["sim (media)"], "b-o", label="simulado (media)")
    ax.plot(x, persistencia["real"], "r-*", ms=10, label="histórico")
    ax.set_xticks(x); ax.set_xlabel("h (días)"); ax.set_ylabel("pendiente de índice(d + h) sobre índice(d)")
    ax.set_title("Persistencia del índice de mercado"); ax.legend(); fig.tight_layout()
    fig.savefig(os.path.join(salida, "persistencia_indice.png"), dpi=130); plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--nodos", type=int, default=35, help="primeros N nodos a validar (máx. 194)")
    ap.add_argument("--replicas", type=int, default=200, help="mundos simulados del periodo completo")
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

    print(f"Validación dentro de muestra: {n} nodos, {len(hist['dias'])} días, {args.replicas} mundos simulados")
    sim = simular(mp, hist, args.replicas, args.seed)

    cargas = comparar_cargas(mp, hist, sim)
    cuantiles, mensual = comparar_tarifas(hist, sim)
    cuant_resumen, cuant_lane = comparar_cuantiles_lane(hist, sim)
    persistencia = comparar_persistencia(hist, sim)

    os.makedirs(args.salida, exist_ok=True)
    cargas.to_csv(os.path.join(args.salida, "cargas_por_grupo.csv"), index=False)
    cuantiles.to_csv(os.path.join(args.salida, "tarifas_cuantiles.csv"), index=False)
    mensual.to_csv(os.path.join(args.salida, "tarifa_mensual.csv"), index=False)
    cuant_resumen.to_csv(os.path.join(args.salida, "tarifas_cuantiles_lane_resumen.csv"), index=False)
    cuant_lane.to_csv(os.path.join(args.salida, "tarifas_cuantiles_por_lane.csv"), index=False)
    persistencia.to_csv(os.path.join(args.salida, "persistencia_indice.csv"), index=False)
    graficar(mp, hist, sim, cargas, cuantiles, mensual, persistencia, args.salida)

    pd.set_option("display.width", 200, "display.max_columns", 20, "display.float_format", "{:.4f}".format)
    print("\n== 1. Cargas por lane (por grupo de volumen h) ==")
    print(cargas.to_string(index=False))
    print("\n== 2a. Tarifas: cuantiles por lane, real vs simulado ==")
    print(cuant_resumen.to_string(index=False))
    print("\n== 2b. Tarifas: cuantiles por grupo de distancia ==")
    print(cuantiles.to_string(index=False))
    print("\n== 2c. Tarifa media por mes ==")
    print(mensual.to_string(index=False))
    rc = sim["recorte"]
    print(f"\n== 2d. Tarifas recortadas al mínimo histórico de la lane ==\n"
          f"  todas las lanes: {rc['recortadas']} de {rc['tarifas']} ({rc['pct']:.2f} %)   "
          f"lanes del universo: {rc['recortadas_universo']} de {rc['tarifas_universo']} ({rc['pct_universo']:.2f} %)")
    print("\n== 3. Persistencia del índice: pendiente de índice(d + h) sobre índice(d) ==")
    print(persistencia.to_string(index=False))
    print("\n== 4. Política: ver evaluation.py (mundos simulados y replay fuera de muestra) ==")
    print(f"\nTablas y gráficos en: {args.salida}")
    print("Nota: validación dentro de muestra (parámetros estimados con los días de ajuste); "
          "el diagnóstico es de ajuste, no de generalización.")


if __name__ == "__main__":
    main()
