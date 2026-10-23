"""
stats_analysis.py
=================
Comparación estadística pareada del DRL contra los baselines B1, B2 y B3, mundo a mundo
(baselines_drl_ruteo_estocastico.md, sección Pruebas estadísticas).

Unidad de análisis: la diferencia de margen (USD) DRL − baseline en un mismo mundo.
  · Diferencia media pareada con intervalo del 95 % por bootstrap sobre mundos
    (BOOTSTRAP_B remuestreos, semilla fija; Agarwal et al., 2021).
  · Wilcoxon de rangos con signo, pareado y de una cola (H1: DRL > baseline), con
    corrección de Holm por las comparaciones de cada (modo, K) (Accorsi, Lodi y Vigo).
  · B2 se compara en todos los niveles de K, incluido el libre.
  · Gap de cada método contra UB-DP y contra UB-PI: (ref − método) / |ref| · 100 sobre los
    promedios de los mismos mundos, con intervalo bootstrap. UB-PI es cota; UB-DP es el valor de la
    política que ignora el mercado (no es cota): un gap negativo contra UB-DP significa que el método la supera.

El replay (días reales 90-119, que ni el entrenamiento ni el ajuste del proceso vieron) se calcula
igual pero va etiquetado: "replay, fuera de muestra (días 90-119)".
"""

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from config import BOOTSTRAP_B, BOOTSTRAP_SEED

REFERENCIA = "DRL"
BASELINES = ("B1", "B2", "B3")
NOTA_REPLAY = "replay, fuera de muestra (días 90-119)"


def etiqueta_modo(modo: str) -> str:
    return NOTA_REPLAY if modo == "replay" else modo


NOTA_UB_DP = ("UB-DP es el valor de la política que ignora el mercado (supone la distribución de las tarifas, "
              "no su estado): NO es una cota, y un método que observa el índice de mercado puede superarla. "
              "UB-PI (información perfecta) sí es cota.")
NOTA_UB_DP_FLOJA = "UB-DP floja (> UB-PI)"
NOTA_UB_DP_FLOJA_LARGA = (
    "En este K la media de UB-DP supera a la de UB-PI. UB-DP es el valor aproximado de la política que "
    "ignora el mercado y UB-PI el de información perfecta, así que UB-PI ≥ UB-DP debe cumplirse: la "
    "diferencia indica sesgo de muestreo de la DP (máximo sobre S sorteos), la discretización de franjas "
    "o un desajuste entre la DP y el entorno. No leer UB-DP como referencia principal en este K."
)


def ub_dp_floja(ub_dp, ub_pi) -> bool:
    """True si la media de UB-DP supera a la de UB-PI (sobre los mundos con ambas cotas)."""
    ub_dp, ub_pi = np.asarray(ub_dp, dtype=float), np.asarray(ub_pi, dtype=float)
    ok = ~np.isnan(ub_dp) & ~np.isnan(ub_pi)
    return bool(ok.any() and ub_dp[ok].mean() > ub_pi[ok].mean())


def gap_pct(ub, x) -> float:
    """(media UB − media método) / |media UB| · 100 sobre los mundos con cota disponible."""
    ub, x = np.asarray(ub, dtype=float), np.asarray(x, dtype=float)
    ok = ~np.isnan(ub)
    if not ok.any() or ub[ok].mean() == 0:
        return float("nan")
    return float((ub[ok].mean() - x[ok].mean()) / abs(ub[ok].mean()) * 100.0)


def _indices_bootstrap(n: int, B: int, rng) -> np.ndarray:
    return rng.integers(0, n, size=(B, n))


def ic_bootstrap_media(valores, B: int = BOOTSTRAP_B, rng=None, semilla: int = BOOTSTRAP_SEED) -> tuple:
    """IC del 95 % (percentiles 2,5 y 97,5) de la media por bootstrap sobre mundos.

    Remuestrea con reemplazo B veces; con `rng` None usa un Generator nuevo de semilla fija.
    Devuelve (bajo, alto); (nan, nan) si no hay valores.
    """
    v = np.asarray(valores, dtype=float)
    v = v[~np.isnan(v)]
    if len(v) == 0:
        return float("nan"), float("nan")
    rng = rng if rng is not None else np.random.Generator(np.random.PCG64(semilla))
    medias = v[_indices_bootstrap(len(v), B, rng)].mean(axis=1)
    return float(np.percentile(medias, 2.5)), float(np.percentile(medias, 97.5))


def holm(p: np.ndarray) -> np.ndarray:
    """p-valores ajustados por Holm-Bonferroni (mismo orden que la entrada)."""
    p = np.asarray(p, dtype=float)
    m = len(p)
    orden = np.argsort(p)
    ajustado = np.empty(m)
    maximo = 0.0
    for rango, i in enumerate(orden):
        maximo = max(maximo, min(1.0, (m - rango) * p[i]))
        ajustado[i] = maximo
    return ajustado


def _wilcoxon_mayor(dif: np.ndarray) -> tuple:
    """Wilcoxon pareado de una cola (H1: la diferencia es positiva). (estadístico, p)."""
    if len(dif) < 1 or np.allclose(dif, 0):
        return float("nan"), 1.0 if len(dif) else float("nan")
    try:
        r = wilcoxon(dif, alternative="greater")
        return float(r.statistic), float(r.pvalue)
    except ValueError:
        return float("nan"), float("nan")


def comparar_con_baselines(df: pd.DataFrame, referencia: str = REFERENCIA, baselines=BASELINES,
                           B: int = BOOTSTRAP_B, semilla: int = BOOTSTRAP_SEED,
                           alpha: float = 0.05) -> pd.DataFrame:
    """Diferencia media pareada (referencia − baseline) con IC bootstrap y Wilcoxon + Holm.

    df : DataFrame largo de evaluation (columnas modo, K, mundo, metodo, margen).
    """
    rng = np.random.Generator(np.random.PCG64(semilla))
    filas = []
    for (modo, K), g in df.groupby(["modo", "K"], sort=False):
        ancho = g.pivot(index="mundo", columns="metodo", values="margen")
        if referencia not in ancho:
            continue
        bloque = []
        for b in baselines:
            if b not in ancho:
                continue
            par = ancho[[referencia, b]].dropna()
            dif = (par[referencia] - par[b]).to_numpy()
            n = len(dif)
            if n == 0:
                continue
            medias = dif[_indices_bootstrap(n, B, rng)].mean(axis=1)
            stat, p = _wilcoxon_mayor(dif)
            bloque.append({
                "modo": etiqueta_modo(modo), "K": K, "comparacion": f"{referencia} − {b}", "n_mundos": n,
                "dif_media_usd": float(dif.mean()),
                "ic95_inf": float(np.percentile(medias, 2.5)),
                "ic95_sup": float(np.percentile(medias, 97.5)),
                "wilcoxon_stat": stat, "p_una_cola": p,
            })
        if bloque:
            p_holm = holm(np.array([f["p_una_cola"] for f in bloque]))
            for f, ph in zip(bloque, p_holm):
                f["p_holm"] = float(ph)
                f[f"significativo (Holm, α={alpha})"] = bool(ph < alpha)
            filas.extend(bloque)
    return pd.DataFrame(filas)


def gaps_contra_cotas(df: pd.DataFrame, B: int = BOOTSTRAP_B, semilla: int = BOOTSTRAP_SEED) -> pd.DataFrame:
    """Gap (%) de cada método contra UB-DP y UB-PI, con intervalo bootstrap del 95 % sobre mundos."""
    rng = np.random.Generator(np.random.PCG64(semilla))
    filas = []
    for (modo, K), g in df.groupby(["modo", "K"], sort=False):
        floja = ub_dp_floja(g["ub_dp"], g["ub_pi"])
        for metodo, gm in g.groupby("metodo", sort=False):
            fila = {"modo": etiqueta_modo(modo), "K": K, "metodo": metodo, "n_mundos": len(gm),
                    "ub_dp_floja": floja}
            for col, nombre in (("ub_dp", "ub_dp"), ("ub_pi", "ub_pi")):
                ub, x = gm[col].to_numpy(float), gm["margen"].to_numpy(float)
                ok = ~np.isnan(ub)
                fila[f"gap_{nombre}_pct"] = gap_pct(ub, x)
                if ok.sum() > 1:
                    idx = _indices_bootstrap(int(ok.sum()), B, rng)
                    u, v = ub[ok][idx].mean(axis=1), x[ok][idx].mean(axis=1)
                    with np.errstate(divide="ignore", invalid="ignore"):
                        gaps = (u - v) / np.abs(u) * 100.0
                    fila[f"gap_{nombre}_ic95_inf"] = float(np.nanpercentile(gaps, 2.5))
                    fila[f"gap_{nombre}_ic95_sup"] = float(np.nanpercentile(gaps, 97.5))
            filas.append(fila)
    return pd.DataFrame(filas)


def append_stats_sheets_to_excel(excel_path: str, sim_df: pd.DataFrame, replay_df: pd.DataFrame = None) -> None:
    """Agrega 'Comparación pareada' y 'Gaps vs cotas' al Excel existente sin tocar las hojas previas."""
    df = sim_df if replay_df is None or replay_df.empty else pd.concat([sim_df, replay_df], ignore_index=True)
    pareada = comparar_con_baselines(df)
    gaps = gaps_contra_cotas(df)
    with pd.ExcelWriter(excel_path, engine="openpyxl", mode="a", if_sheet_exists="replace") as writer:
        pareada.to_excel(writer, sheet_name="Comparación pareada", index=False)
        gaps.to_excel(writer, sheet_name="Gaps vs cotas", index=False)
    print(f"Hojas estadísticas agregadas → {excel_path}")
