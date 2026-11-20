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
from pathlib import Path

import torch  # noqa: F401  (importar antes que pandas; ver nota de orden en main.py)
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import TRAIN_DAYS
from fit_market_process import estimar_m, factor_nacional
from market_process import MarketProcess, asegurar_parametros, _interp_cuantiles
from problem_data import cargar_dias_mercado, orden_hubs

_CWD = os.path.dirname(os.path.abspath(__file__))
_Q_PCT = np.array([0, 25, 50, 75, 90, 100])

_ESTADOS_US = {
    "AL": "Alabama", "AK": "Alaska", "AZ": "Arizona", "AR": "Arkansas", "CA": "California",
    "CO": "Colorado", "CT": "Connecticut", "DE": "Delaware", "FL": "Florida", "GA": "Georgia",
    "HI": "Hawaii", "ID": "Idaho", "IL": "Illinois", "IN": "Indiana", "IA": "Iowa", "KS": "Kansas",
    "KY": "Kentucky", "LA": "Louisiana", "ME": "Maine", "MD": "Maryland", "MA": "Massachusetts",
    "MI": "Michigan", "MN": "Minnesota", "MS": "Mississippi", "MO": "Missouri", "MT": "Montana",
    "NE": "Nebraska", "NV": "Nevada", "NH": "New Hampshire", "NJ": "New Jersey", "NM": "New Mexico",
    "NY": "New York", "NC": "North Carolina", "ND": "North Dakota", "OH": "Ohio", "OK": "Oklahoma",
    "OR": "Oregon", "PA": "Pennsylvania", "RI": "Rhode Island", "SC": "South Carolina",
    "SD": "South Dakota", "TN": "Tennessee", "TX": "Texas", "UT": "Utah", "VT": "Vermont",
    "VA": "Virginia", "WA": "Washington", "WV": "West Virginia", "WI": "Wisconsin", "WY": "Wyoming",
    "DC": "Washington D.C.",
}


def _estados_por_hub(data_dir: str = None) -> dict:
    """hub_id -> abreviatura de estado (columna `state`, el estado del destino), leída de un solo
    CSV diario — alcanza con uno porque los 194 hubs aparecen como destino todos los días."""
    data_dir = data_dir or os.path.join(_CWD, "datos")
    archivo = sorted((Path(data_dir) / "datos_finales_jun_oct_2026").glob("*.csv"))[0]
    df = pd.read_csv(archivo, usecols=["destination", "state"])
    return dict(zip(df["destination"], df["state"]))


def _nombre_hub(hub_id: int, estados: dict) -> str:
    abrev = estados.get(hub_id)
    return _ESTADOS_US.get(abrev, abrev) if abrev else f"hub {hub_id}"


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


def _graficar_precio_lane(mp: MarketProcess, rng: np.random.Generator, out_dir: str, i: int, j: int,
                          nombre_archivo: str, n_sorteos_dia: int = 300, nota_titulo: str = "") -> str:
    """Gráfico de serie de tiempo real vs. sintética (precio) para una ruta (i, j) puntual: tarifa
    media diaria real (línea negra, con la banda real [q1, q3] del día sombreada) contra el
    PROMEDIO de `n_sorteos_dia` sorteos por día (δ = 0, de los mismos cuantiles diarios del archivo
    para esa ruta), con su propia banda [q1, q3] entre sorteos. Se compara promedio contra
    promedio (no un sorteo suelto contra la media real) para que ambas líneas sean la misma
    cantidad y el gráfico no muestre picos sintéticos que la real, por construcción, no tiene.
    Compartida por `graficar_serie_tiempo` (la ruta con más datos) y `graficar_peor_lane` (la ruta
    con peor ajuste)."""
    hubs = orden_hubs()
    estados = _estados_por_hub()
    nombre_i = _nombre_hub(int(hubs[i]), estados)
    nombre_j = _nombre_hub(int(hubs[j]), estados)
    distancias = pd.read_csv(os.path.join(_CWD, "datos", "distance.csv"), index_col=0)
    dist_mi = float(distancias.iloc[i, j])

    fechas = mp.fechas[:TRAIN_DAYS]
    con_cargas = mp._con_cargas[:TRAIN_DAYS, i, j]
    real = np.where(con_cargas, mp.mean_rate[:TRAIN_DAYS, i, j], np.nan)
    real_q1 = np.where(con_cargas, mp._q["q1"][:TRAIN_DAYS, i, j], np.nan)
    real_q3 = np.where(con_cargas, mp._q["q3"][:TRAIN_DAYS, i, j], np.nan)

    cuantiles = np.stack([mp._q[c][:TRAIN_DAYS, i, j] for c in ("min", "q1", "q2", "q3", "p90", "max")], axis=-1)
    u = rng.random((n_sorteos_dia, TRAIN_DAYS))
    sorteos = _interp_cuantiles(u, np.broadcast_to(cuantiles, (n_sorteos_dia, TRAIN_DAYS, 6)))
    sint_media = np.where(con_cargas, sorteos.mean(axis=0), np.nan)
    sint_q1 = np.where(con_cargas, np.percentile(sorteos, 25, axis=0), np.nan)
    sint_q3 = np.where(con_cargas, np.percentile(sorteos, 75, axis=0), np.nan)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.fill_between(fechas, real_q1, real_q3, color="0.6", alpha=0.25, label="real: rango del día [q1, q3]")
    ax.plot(fechas, real, color="black", lw=1.6, marker="o", ms=3, label="real: tarifa media del día")
    ax.fill_between(fechas, sint_q1, sint_q3, color="tab:blue", alpha=0.20,
                    label=f"sintético: rango entre {n_sorteos_dia} sorteos [q1, q3]")
    ax.plot(fechas, sint_media, color="tab:blue", lw=1.6, marker="o", ms=3,
           label=f"sintético: promedio de {n_sorteos_dia} sorteos")

    fig.autofmt_xdate()
    titulo = (f"Precio real vs. sintético — {nombre_i} → {nombre_j} ({dist_mi:.0f} mi)\n"
             f"({int(con_cargas.sum())} días con cargas de {TRAIN_DAYS})")
    if nota_titulo:
        titulo += f" — {nota_titulo}"
    ax.set(xlabel="Fecha", ylabel="Precio (USD/mi)", title=titulo)
    ax.legend(fontsize=9, loc="upper left", ncol=2)

    fig.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, nombre_archivo)
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def graficar_serie_tiempo(mp: MarketProcess, rng: np.random.Generator, out_dir: str,
                          n_sorteos_dia: int = 300) -> str:
    """Serie de tiempo real vs. sintética para la ruta con más días con cargas en entrenamiento.
    Pensado para una audiencia no técnica: un solo gráfico, "¿se parecen la línea real y la
    sintética?", en vez de cuantiles sueltos."""
    N = mp.num_nodes
    sin_diag = ~np.eye(N, dtype=bool)
    con_cargas_train = mp._con_cargas[:TRAIN_DAYS] & sin_diag[None]
    conteo_por_lane = con_cargas_train.sum(axis=0)
    i, j = (int(x) for x in np.unravel_index(np.argmax(conteo_por_lane), conteo_por_lane.shape))
    path = _graficar_precio_lane(mp, rng, out_dir, i, j, "precios_serie_tiempo.png", n_sorteos_dia)
    return path, i, j


def _error_relativo_lane(mp: MarketProcess, rng: np.random.Generator, i: int, j: int,
                         n_sorteos_dia: int = 300) -> float:
    """Error relativo medio (MAPE) entre la tarifa media real del día y el promedio de
    `n_sorteos_dia` sorteos sintéticos, sobre los días con cargas en entrenamiento de esa ruta."""
    con_cargas = mp._con_cargas[:TRAIN_DAYS, i, j]
    n_dias = int(con_cargas.sum())
    real = mp.mean_rate[:TRAIN_DAYS, i, j][con_cargas]
    cuantiles = np.stack([mp._q[c][:TRAIN_DAYS, i, j] for c in ("min", "q1", "q2", "q3", "p90", "max")],
                         axis=-1)[con_cargas]
    u = rng.random((n_sorteos_dia, n_dias))
    sorteos = _interp_cuantiles(u, np.broadcast_to(cuantiles, (n_sorteos_dia, n_dias, 6)))
    sint = sorteos.mean(axis=0)
    return float(np.mean(np.abs(sint - real) / real))


def graficar_peor_lane(mp: MarketProcess, rng: np.random.Generator, out_dir: str,
                       n_sorteos_dia: int = 300, min_dias: int = 20):
    """Recorre todas las lanes con al menos `min_dias` días con cargas en entrenamiento (menos que
    eso, el error es puro ruido de muestra chica) y dibuja, con el mismo formato que
    `graficar_serie_tiempo`, la ruta con mayor error relativo medio entre el precio real y el
    promedio sintético — la ruta donde peor ajusta el generador. Útil para mostrar el caso más
    exigente, no solo el más favorable (el de más datos)."""
    N = mp.num_nodes
    sin_diag = ~np.eye(N, dtype=bool)
    con_cargas_train = mp._con_cargas[:TRAIN_DAYS] & sin_diag[None]
    conteo = con_cargas_train.sum(axis=0)
    candidatas = np.argwhere(conteo >= min_dias)
    if len(candidatas) == 0:
        return None, None, None, None

    peor_i = peor_j = -1
    peor_err = -1.0
    for i, j in candidatas:
        err = _error_relativo_lane(mp, rng, int(i), int(j), n_sorteos_dia)
        if err > peor_err:
            peor_i, peor_j, peor_err = int(i), int(j), err

    nota = f"error relativo medio = {100 * peor_err:.1f}%"
    path = _graficar_precio_lane(mp, rng, out_dir, peor_i, peor_j, "precios_peor_ajuste.png",
                                 n_sorteos_dia, nota_titulo=nota)
    return path, peor_i, peor_j, peor_err


def graficar_serie_cargas(mp: MarketProcess, i: int, j: int, rng: np.random.Generator, out_dir: str,
                          n_sorteos_dia: int = 300) -> str:
    """Serie de tiempo de la CANTIDAD de cargas por día (no el precio) en la misma ruta del gráfico
    de precios: conteo real día a día (mp.count_loads; 0 es un valor real válido, a diferencia del
    precio no hace falta descartar días sin cargas) contra el promedio de `n_sorteos_dia` sorteos
    por día del mismo modelo Poisson/binomial-negativa que usa el simulador para decidir cuántas
    cargas aparecen cada día (Paso 4b-1 de la metodología), con su banda [q1, q3] entre sorteos."""
    fechas = mp.fechas[:TRAIN_DAYS]
    real = mp.count_loads[:TRAIN_DAYS, i, j].astype(np.float64)
    m = mp._m_stack[:TRAIN_DAYS, i, j]
    r = float(mp._r_lane[i, j])
    bajo = m < 1.0

    forma = (n_sorteos_dia, TRAIN_DAYS)
    sim_pois = rng.poisson(np.broadcast_to(np.maximum(m, 1e-12), forma))
    p = r / (r + np.maximum(m, 1e-12))
    sim_nbinom = rng.negative_binomial(r, np.broadcast_to(p, forma))
    sim = np.where(np.broadcast_to(bajo, forma), sim_pois, sim_nbinom)

    sint_media = sim.mean(axis=0)
    sint_q1 = np.percentile(sim, 25, axis=0)
    sint_q3 = np.percentile(sim, 75, axis=0)

    hubs = orden_hubs()
    estados = _estados_por_hub()
    nombre_i, nombre_j = _nombre_hub(int(hubs[i]), estados), _nombre_hub(int(hubs[j]), estados)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(fechas, real, color="black", lw=1.4, marker="o", ms=3, label="real: cargas del día")
    ax.fill_between(fechas, sint_q1, sint_q3, color="tab:orange", alpha=0.25,
                    label=f"sintético: rango entre {n_sorteos_dia} sorteos [q1, q3]")
    ax.plot(fechas, sint_media, color="tab:orange", lw=1.4, marker="o", ms=3,
           label=f"sintético: promedio de {n_sorteos_dia} sorteos")

    fig.autofmt_xdate()
    ax.set(xlabel="Fecha", ylabel="Cantidad de cargas",
          title=f"Cantidad de cargas por día — {nombre_i} → {nombre_j}")
    ax.legend(fontsize=9, loc="upper left")

    fig.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "cargas_serie_tiempo.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def graficar_delta(mp: MarketProcess, rng: np.random.Generator, out_dir: str, n_semillas: int = 5000) -> str:
    """Histograma del ajuste de nivel de mercado: δ sintético (un único valor por "mundo" simulado,
    ver `MarketProcess._delta`) contra ΔF_t real (cambio diario del factor nacional, la misma
    comparación de `validar_delta` pero como distribución completa en vez de una tabla de
    percentiles). Si las dos campanas tienen un ancho parecido, el tamaño típico del "día/mundo más
    caro o más barato de lo normal" que genera el simulador es realista."""
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

    fig, ax = plt.subplots(figsize=(8, 5))
    lo, hi = min(deltas.min(), delta_F.min()), max(deltas.max(), delta_F.max())
    bins = np.linspace(lo, hi, 60)
    ax.hist(delta_F, bins=bins, color="0.3", alpha=0.5, density=True,
           label=f"real: cambio diario del mercado\n(desviación = {delta_F.std():.3f})")
    ax.hist(deltas, bins=bins, color="tab:green", alpha=0.5, density=True,
           label=f"sintético: ajuste por mundo simulado\n(desviación = {deltas.std():.3f})")
    ax.set(xlabel="Cambio de nivel del mercado (escala log)", ylabel="Densidad",
          title="¿El tamaño de los días/mundos \"más caros o más baratos de lo normal\" es realista?")
    ax.legend(fontsize=9)

    fig.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "delta_distribucion.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def tabla_resumen(mp: MarketProcess, rng: np.random.Generator) -> pd.DataFrame:
    """Tabla de 3 números simples, real vs. sintético, juntando TODAS las lanes y días de
    entrenamiento: precio promedio, cargas promedio por lane-día y % de lane-días sin cargas. Para
    mostrar en una diapositiva sin ejes ni leyendas que explicar."""
    N = mp.num_nodes
    sin_diag = ~np.eye(N, dtype=bool)

    con_cargas_mask = mp._con_cargas[:TRAIN_DAYS] & sin_diag[None]
    precio_real = float(mp.mean_rate[:TRAIN_DAYS][con_cargas_mask].mean())
    cuantiles = np.stack([mp._q[c][:TRAIN_DAYS] for c in ("min", "q1", "q2", "q3", "p90", "max")], axis=-1)
    cuantiles_validos = cuantiles[con_cargas_mask]
    u = rng.random(len(cuantiles_validos))
    precio_sint = float(_interp_cuantiles(u, cuantiles_validos).mean())

    m = mp._m_stack[:TRAIN_DAYS][:, sin_diag]
    r = np.broadcast_to(mp._r_lane[sin_diag], m.shape)
    real_counts = mp.count_loads[:TRAIN_DAYS][:, sin_diag].astype(np.float64)
    bajo = m < 1.0
    sim_pois = rng.poisson(np.maximum(m, 1e-12))
    p = r / (r + np.maximum(m, 1e-12))
    sim_nbinom = rng.negative_binomial(r, p)
    sim_counts = np.where(bajo, sim_pois, sim_nbinom).astype(np.float64)

    return pd.DataFrame([
        {"métrica": "Precio promedio (USD/mi)", "real": precio_real, "sintético": precio_sint},
        {"métrica": "Cargas promedio por lane-día", "real": float(real_counts.mean()),
         "sintético": float(sim_counts.mean())},
        {"métrica": "% lane-días sin cargas", "real": 100 * float((real_counts == 0).mean()),
         "sintético": 100 * float((sim_counts == 0).mean())},
    ])


def graficar_tabla_resumen(tabla: pd.DataFrame, out_dir: str) -> str:
    """Dibuja `tabla_resumen` como imagen (lista para pegar en una diapositiva)."""
    fig, ax = plt.subplots(figsize=(8.5, 0.9 + 0.6 * len(tabla)))
    ax.axis("off")
    celdas = [[row["métrica"], f"{row['real']:.2f}", f"{row['sintético']:.2f}"] for _, row in tabla.iterrows()]
    tbl = ax.table(cellText=celdas, colLabels=["Métrica", "Real", "Sintético"], loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(12)
    tbl.auto_set_column_width(col=list(range(3)))
    tbl.scale(1, 2.2)

    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "tabla_resumen.png")
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path


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
    figs_dir = os.path.join(args.salida, "figures")
    fig_precios, lane_i, lane_j = graficar_serie_tiempo(mp, rng, figs_dir)
    fig_cargas = graficar_serie_cargas(mp, lane_i, lane_j, rng, figs_dir)
    fig_peor, peor_i, peor_j, peor_err = graficar_peor_lane(mp, rng, figs_dir)
    fig_delta = graficar_delta(mp, rng, figs_dir)
    tabla = tabla_resumen(mp, rng)
    fig_tabla = graficar_tabla_resumen(tabla, figs_dir)

    precios.to_csv(os.path.join(args.salida, "precios_cuantiles.csv"), index=False)
    cargas.to_csv(os.path.join(args.salida, "cargas_por_grupo.csv"), index=False)
    delta.to_csv(os.path.join(args.salida, "delta_vs_factor_nacional.csv"), index=False)
    persistencia.to_csv(os.path.join(args.salida, "persistencia_indice.csv"), index=False)
    tabla.to_csv(os.path.join(args.salida, "tabla_resumen.csv"), index=False)

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
    print(f"Gráfico (precio, serie de tiempo real vs. sintética): {fig_precios}")
    print(f"Gráfico (cargas, serie de tiempo real vs. sintética): {fig_cargas}")
    if fig_peor:
        print(f"Gráfico (precio, ruta con PEOR ajuste, error relativo medio = {100 * peor_err:.1f}%): {fig_peor}")
    print(f"Gráfico (ajuste de nivel de mercado, real vs. sintético): {fig_delta}")
    print(f"Tabla resumen (imagen): {fig_tabla}")


if __name__ == "__main__":
    main()
