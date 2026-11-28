"""
stats_analysis.py
=================
Comparación estadística pareada del DRL, mundo a mundo (baselines_drl_ruteo_estocastico.md, sección Pruebas
estadísticas; especificacion_fase0_ruteo_pricing.md §8). Dos familias de comparaciones, cada una con su Holm:
  · "ruteo"  : DRL − B1, B2, B2P, B3 (el baseline elige destino y cotiza con REGLA_PRECIO_BASELINES).
  · "precio" (mismas rutas del DRL con reglas de precio fijas): DRL − DRL-dest+mediana, DRL-dest+be_margen, DRL-dest+miope. Mismos destinos
    del DRL con una regla fija de precio: la diferencia mide lo que aporta la cabeza de precio.

Unidad de análisis: la diferencia de margen (USD) DRL − método en un mismo mundo.
  · Diferencia media pareada con intervalo del 95 % por bootstrap sobre mundos
    (BOOTSTRAP_B remuestreos, semilla fija; Agarwal et al., 2021).
  · Wilcoxon de rangos con signo, pareado y de una cola (H1: DRL > método), con
    corrección de Holm por las comparaciones de cada (modo, K, familia) (Accorsi, Lodi y Vigo).
  · Si UB-DP es "no tratable" a ese tamaño, B3 no está en los datos y simplemente no se compara.
  · B2 se compara en todos los niveles de K, incluido el libre.
  · gap_pct: (ref − método) / |ref| · 100 sobre los promedios de los mismos mundos (lo usa el Resumen).

El replay (días reales desde TRAIN_DAYS, que ni el entrenamiento ni el ajuste del proceso vieron) se
calcula igual pero va etiquetado con la fecha real de inicio (NOTA_REPLAY), calculada a partir de
FECHA_INICIO y TRAIN_DAYS en vez de quemar un rango de días a mano.
"""

import datetime

import numpy as np
import torch  # noqa: F401  (antes que pandas: en Windows, pandas primero rompe la carga de las DLL de torch)
import pandas as pd
from scipy.stats import wilcoxon

from config import BOOTSTRAP_B, BOOTSTRAP_SEED, FECHA_INICIO, TRAIN_DAYS

REFERENCIA = "DRL"
BASELINES = ("B1", "B2", "B2P", "B3")
REGLAS_PRECIO = ("DRL-dest+mediana", "DRL-dest+be_margen", "DRL-dest+miope")
FAMILIAS = {"ruteo": BASELINES, "precio": REGLAS_PRECIO}
_FECHA_INICIO_REPLAY = (
    datetime.date.fromisoformat(FECHA_INICIO) + datetime.timedelta(days=TRAIN_DAYS)
).isoformat()
NOTA_REPLAY = f"replay, fuera de muestra (días reales desde {_FECHA_INICIO_REPLAY})"


def etiqueta_modo(modo: str) -> str:
    return NOTA_REPLAY if modo == "replay" else modo


NOTA_UB_DP = ("«Óptimo con mercado promedio» es lo que lograría la mejor política si el mercado de cada día fuera "
              "el promedio, sin memoria de los días anteriores. No es un tope: un método que observa el mercado del día "
              "puede superarlo. El «Máximo teórico (información perfecta)» sí es un tope: ningún método puede superarlo.")
NOTA_UB_DP_FLOJA = "supera al máximo teórico (ver nota)"
NOTA_UB_DP_FLOJA_LARGA = (
    "Aquí el «Óptimo con mercado promedio» supera al «Máximo teórico», lo que no debería pasar: es un error de "
    "aproximación de su cálculo (muestreo y discretización del tiempo). No usarlo como referencia en esta tabla."
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


def comparar_con_baselines(df: pd.DataFrame, referencia: str = REFERENCIA, familias=FAMILIAS,
                           B: int = BOOTSTRAP_B, semilla: int = BOOTSTRAP_SEED,
                           alpha: float = 0.05) -> pd.DataFrame:
    """Diferencia media pareada (referencia − método) con IC bootstrap y Wilcoxon + Holm por familia.

    df       : DataFrame largo de evaluation (columnas modo, K, mundo, metodo, margen).
    familias : {nombre: métodos}; Holm se aplica dentro de cada (modo, K, familia).
    """
    rng = np.random.Generator(np.random.PCG64(semilla))
    filas = []
    for (modo, K), g in df.groupby(["modo", "K"], sort=False):
        ancho = g.pivot(index="mundo", columns="metodo", values="margen")
        if referencia not in ancho:
            continue
        for familia, metodos in familias.items():
            filas.extend(_bloque_pareado(ancho, modo, K, familia, metodos, referencia, B, rng, alpha))
    return pd.DataFrame(filas)


def _bloque_pareado(ancho, modo, K, familia, metodos, referencia, B, rng, alpha) -> list:
    bloque = []
    for b in metodos:
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
            "modo": etiqueta_modo(modo), "K": K, "familia": familia,
            "comparacion": f"{referencia} − {b}", "n_mundos": n,
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
    return bloque


def append_comparacion_to_excel(excel_path: str, sim_df: pd.DataFrame, replay_df: pd.DataFrame = None) -> None:
    """Agrega al Excel la hoja 'Comparación': DRL − cada método, una fila por (modo, K, comparación).
    familia "ruteo" = B1, B2, B2P, B3; familia "precio" = mismas rutas del DRL con reglas de precio fijas (lo que aporta la cabeza de precio)."""
    df = sim_df if replay_df is None or replay_df.empty else pd.concat([sim_df, replay_df], ignore_index=True)
    pareada = comparar_con_baselines(df).drop(columns=["wilcoxon_stat", "p_una_cola"], errors="ignore")
    from excel_report import escribir_comparacion
    escribir_comparacion(excel_path, pareada)
    print(f"Hoja 'Comparación' agregada → {excel_path}")
