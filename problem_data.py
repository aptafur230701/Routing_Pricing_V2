"""
problem_data.py
==============
Data loading and matrix computation.

load_matrices(num_nodes)
    → time_matrix, rate_stack, loads_stack, distance_arr, diesel_arr,
      ltr_stack, avail_expected_arr, reward_global_p95

build_day_matrices(rate_day, loads_day, distance_arr, diesel_arr, num_nodes)
    → reward_matrix, reward_matrix_penalized

rate_stack  y loads_stack son arreglos numpy (num_days, num_nodes, num_nodes), uno por día real
jun-oct 2026 (ver metodologia_realizaciones_mercado_reefer.md, Paso 2). Se leen una sola vez de los
CSV diarios en datos/datos_finales_jun_oct_2026/ y se guardan en memoria con el MISMO orden de hubs
que usan distance.csv/duration.csv (el orden de su encabezado/índice, no el orden ascendente de
hub_id) para que rate_stack[d, i, j], distance_arr[i, j] y diesel_arr[i, j] refieran al mismo par
de hubs.
"""

import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

from config import (
    MPG,
    BIG_M_PENALTY, MARGINAL_COST_SIN_DIESEL,
    TRAIN_DAYS,
)

# ── Rutas de los datos nuevos (jun-oct 2026) ──────────────────────────────
_CWD = os.path.dirname(os.path.abspath(__file__))
DATOS_DIARIOS_DIR = "datos_finales_jun_oct_2026"
FUEL_CSV = "fuel_cost_hub_jun_oct_2026.csv"
_PATRON_FECHA = re.compile(r"(\d{4}-\d{2}-\d{2})")

# Columnas que se leen de cada CSV diario (además de origin/destination, que fijan la lane).
_COLS_DIARIAS = ["load_availability", "mean", "min", "q1", "q2", "q3", "p90", "max", "load_to_truck_ratio"]


def orden_hubs(data_dir: str = None) -> np.ndarray:
    """Orden canónico de los 194 hubs (hub_id por posición), el mismo que usan distance.csv/duration.csv.

    Es el orden de su encabezado (no es ascendente por hub_id); todo lo que se construya desde los
    CSV diarios nuevos debe reindexarse a este mismo orden para que las posiciones (i, j) coincidan
    entre rate_stack/loads_stack y distance_arr/diesel_arr.
    """
    data_dir = data_dir or os.path.join(_CWD, "datos")
    header = pd.read_csv(os.path.join(data_dir, "distance.csv"), index_col=0, nrows=0)
    return header.columns.astype(np.int64).to_numpy()


def _nearest_valid_index(valid: np.ndarray) -> np.ndarray:
    """Para cada posición a lo largo del eje 0 (días), el índice del día más cercano con `valid=True`
    en esa misma posición de los ejes restantes (empate → el día anterior). `valid` : (D, ...) bool.
    Devuelve (D, ...) int64; una posición sin ningún día válido devuelve -1 (el llamador debe manejarlo:
    en ese caso la lane nunca tuvo cargas en toda la ventana).
    """
    D = valid.shape[0]
    forma_dia = (D,) + (1,) * (valid.ndim - 1)
    dia_idx = np.arange(D).reshape(forma_dia)

    ultimo_valido = np.where(valid, dia_idx, -1)
    np.maximum.accumulate(ultimo_valido, axis=0, out=ultimo_valido)

    siguiente_valido = np.where(valid, dia_idx, D)
    siguiente_valido = np.minimum.accumulate(siguiente_valido[::-1], axis=0)[::-1]

    dist_atras = np.where(ultimo_valido >= 0, dia_idx - ultimo_valido, D + 1)
    dist_adelante = np.where(siguiente_valido < D, siguiente_valido - dia_idx, D + 1)
    usar_atras = dist_atras <= dist_adelante
    nearest = np.where(usar_atras, ultimo_valido, siguiente_valido)
    nearest = np.where((ultimo_valido < 0) & (siguiente_valido >= D), -1, nearest)
    return nearest


def cargar_dias_mercado(data_dir: str = None) -> dict:
    """Lee los CSV diarios de datos/datos_finales_jun_oct_2026 (uno por día, jun-oct 2026) una sola vez.

    Devuelve un dict con, todo en el orden canónico de hubs (`orden_hubs`), forma (D, 194, 194) float32:
        count_loads, mean, min, q1, q2, q3, p90, max   — por lane
        ltr                                            — por hub de origen (repetido en toda la fila)
        dia_con_cargas_cercano                         — (D,194,194) int64: índice del día más cercano
                                                          (con count_loads > 0 en esa lane) a usar para
                                                          los cuantiles (Paso 1, regla 1); -1 si la lane
                                                          nunca tuvo cargas en la ventana.
    y fechas (D,) datetime64[D].

    Limpieza (Paso 1 de la metodología):
      - lane-día con count_loads == 0: sus cuantiles no se usan (se resuelve con dia_con_cargas_cercano).
      - load_to_truck_ratio == 0 en un hub con cargas ese día: se trata como faltante y se llena con el
        día válido más cercano de ese mismo hub. Se imprime cuántos casos hubo.
    """
    data_dir = data_dir or os.path.join(_CWD, "datos")
    carpeta = Path(data_dir) / DATOS_DIARIOS_DIR
    archivos = sorted(carpeta.glob("*.csv"))
    if not archivos:
        raise FileNotFoundError(f"no hay CSV diarios en {carpeta}")

    hubs = orden_hubs(data_dir)
    n = len(hubs)
    pos = np.full(int(hubs.max()) + 1, -1, dtype=np.int64)
    pos[hubs] = np.arange(n)

    fechas = []
    for f in archivos:
        m = _PATRON_FECHA.search(f.name)
        if not m:
            raise ValueError(f"no pude leer la fecha del nombre de archivo: {f.name}")
        fechas.append(np.datetime64(m.group(1), "D"))
    fechas = np.array(fechas)
    if not (np.diff(fechas) == np.timedelta64(1, "D")).all():
        raise ValueError("los CSV diarios no cubren un rango continuo de días (falta alguno)")

    D = len(archivos)
    stacks = {c: np.zeros((D, n, n), dtype=np.float32) for c in _COLS_DIARIAS}
    for d, f in enumerate(archivos):
        df = pd.read_csv(f, usecols=["origin", "destination"] + _COLS_DIARIAS)
        po = pos[df["origin"].to_numpy(dtype=np.int64)]
        pd_ = pos[df["destination"].to_numpy(dtype=np.int64)]
        if (po < 0).any() or (pd_ < 0).any():
            raise ValueError(f"{f.name} trae un hub_id fuera del orden canónico de 194 hubs")
        for c in _COLS_DIARIAS:
            stacks[c][d, po, pd_] = df[c].to_numpy(dtype=np.float32)

    count_loads = stacks.pop("load_availability")
    ltr = stacks.pop("load_to_truck_ratio")

    # Regla 2: LT == 0 con cargas ese día en el hub de origen → faltante, llenar con el día válido más
    # cercano del hub (se mira solo la diagonal LTR[d, o, o] como representante del hub, ya que la
    # columna es constante en toda la fila o).
    loads_por_hub = count_loads.sum(axis=2)                       # (D, N) cargas salientes del hub
    ltr_hub = ltr[:, np.arange(n), np.arange(n)]                  # (D, N) — cualquier columna sirve
    ltr_valido = ~((ltr_hub == 0) & (loads_por_hub > 0))
    n_faltantes = int((~ltr_valido).sum())
    if n_faltantes:
        idx_cercano = _nearest_valid_index(ltr_valido)             # (D, N)
        malos = np.flatnonzero(~ltr_valido.ravel())
        d_malo, hub_malo = np.unravel_index(malos, ltr_valido.shape)
        d_cercano = idx_cercano[d_malo, hub_malo]
        ok = d_cercano >= 0
        ltr_hub[d_malo[ok], hub_malo[ok]] = ltr_hub[d_cercano[ok], hub_malo[ok]]
        ltr[d_malo[ok], hub_malo[ok], :] = ltr_hub[d_malo[ok], hub_malo[ok], None]
    print(f"L/T faltante (LT=0 con cargas): {n_faltantes} casos de {ltr_hub.size} hub-días, "
          "llenados con el día válido más cercano del hub.")

    # Regla 1: lane-día con count_loads == 0 → cuantiles no usables; día más cercano con cargas de esa
    # misma lane (para sortear precios cuando la realización sí ofrezca cargas en esa lane ese día).
    con_cargas = count_loads > 0
    dia_con_cargas_cercano = _nearest_valid_index(con_cargas)      # (D, N, N)

    return {
        **stacks,
        "count_loads": count_loads,
        "ltr": ltr,
        "dia_con_cargas_cercano": dia_con_cargas_cercano,
        "fechas": fechas,
        "hubs": hubs,
    }


def cargar_fuel_por_hub(data_dir: str = None) -> dict:
    """Lee datos/fuel_cost_hub_jun_oct_2026.csv (precio de diésel por hub, origin == destination siempre)
    y lo reindexa a cadencia diaria sobre el mismo rango de fechas que `cargar_dias_mercado`.

    Cadencia real: semanal en junio, diaria desde el 23 de junio (con un hueco el 21 de sep). Se rellena
    hacia adelante (forward-fill) para tener un valor por hub en cada día del rango.
    Devuelve {"precio": (D, 194) float32, "fechas": (D,) datetime64[D]} en el orden canónico de hubs.
    """
    data_dir = data_dir or os.path.join(_CWD, "datos")
    hubs = orden_hubs(data_dir)
    n = len(hubs)
    pos = np.full(int(hubs.max()) + 1, -1, dtype=np.int64)
    pos[hubs] = np.arange(n)

    df = pd.read_csv(os.path.join(data_dir, FUEL_CSV))
    df = df[df["origin"] == df["destination"]].copy()
    df["start_date"] = pd.to_datetime(df["start_date"]).dt.normalize()
    fecha_ini, fecha_fin = df["start_date"].min(), df["start_date"].max()
    fechas = pd.date_range(fecha_ini, fecha_fin, freq="D")

    precio = np.full((len(fechas), n), np.nan, dtype=np.float32)
    piv = df.pivot_table(index="start_date", columns="origin", values="value")
    piv = piv.reindex(fechas)
    for hub_id in piv.columns:
        p = pos[int(hub_id)]
        if p < 0:
            continue
        precio[:, p] = piv[hub_id].to_numpy(dtype=np.float32)
    # forward-fill (la cadencia semanal/el hueco del 21-sep deja NaN entre observaciones); un NaN inicial
    # (antes de la primera observación de ese hub) se llena hacia atrás con la primera observación.
    serie = pd.DataFrame(precio)
    serie = serie.ffill().bfill()
    return {"precio": serie.to_numpy(dtype=np.float32),
            "fechas": fechas.values.astype("datetime64[D]")}


def lane_reward(
    rate:     np.ndarray,
    loads:    np.ndarray,
    distance: np.ndarray,
    diesel:   np.ndarray,
) -> np.ndarray:
    """Recompensa de recorrer una lane (sin penalizar la diagonal ni escalar).

    Es la ÚNICA fórmula de recompensa del paso: la usan build_day_matrices (modo
    real, días del histórico) y el entorno con market_process (filas sorteadas
    al llegar al nodo), para que ambos modos no diverjan.

        ingreso = tarifa por milla × distancia, solo si hay carga (loads > 0)
        costo   = distancia × (diesel / MPG) + distancia × MARGINAL_COST_SIN_DIESEL
        recompensa = round(ingreso − costo, 0)

    Los cuatro argumentos son arrays con formas broadcast-compatibles: matrices
    (N, N) en modo real, o filas (N,) / lotes (B, N) en modo simulado.
    """
    revenue = np.where(loads > 0, rate * distance, 0.0)
    cost    = distance * (diesel / MPG) + distance * MARGINAL_COST_SIN_DIESEL
    return np.round(revenue - cost, 0)


def build_day_matrices(
    rate_day:     np.ndarray,
    loads_day:    np.ndarray,
    distance_arr: np.ndarray,
    diesel_arr:   np.ndarray,
) -> tuple:
    """Build reward matrix and its penalized version for a single day snapshot.

    Parameters
    ----------
    rate_day      : np.ndarray (num_nodes × num_nodes) — spot rates for one day.
    loads_day     : np.ndarray (num_nodes × num_nodes) — available loads for one day.
    distance_arr  : np.ndarray (num_nodes × num_nodes) — inter-node distances.
    diesel_arr    : np.ndarray (num_nodes × num_nodes) — fuel prices.

    Returns
    -------
    reward_matrix            : np.ndarray (num_nodes × num_nodes)
    reward_matrix_penalized  : np.ndarray (diagonal = BIG_M_PENALTY)

    Nota (existencia): esta función excluye un arco solo si el registro
    histórico de ESE día puntual no muestra ninguna carga (`loads_day <= 0`).
    `draw_lane_availability` aplica el mismo criterio sobre `loads_stack` para
    la máscara de acciones de los modos con filtrado de lanes, de modo que
    máscara y reward quedan consistentes. El umbral es `<= 0` (antes `<= 1`,
    sin justificación documentada — excluía también el caso `loads_day == 1`).
    La fórmula vive en `lane_reward` (compartida con el modo market_process).
    """
    reward_arr = lane_reward(rate_day, loads_day, distance_arr, diesel_arr)

    penalized_arr = reward_arr.copy()
    np.fill_diagonal(penalized_arr, BIG_M_PENALTY)

    return reward_arr, penalized_arr


def draw_lane_availability(
    node:        int,
    arrival_day: int,
    loads_stack: np.ndarray,
) -> np.ndarray:
    """Lane availability for outgoing lanes FROM node on arrival_day.

    Lee el registro histórico de ESE día: la lane existe sii hubo carga
    (`loads_stack[arrival_day, node, j] > 0`). Es el mismo criterio con que
    build_day_matrices pone ingreso cero, así que máscara de acciones y reward
    quedan consistentes: nunca se bloquea una lane que tuvo carga ni se deja
    abierta (con reward = -costo) una que no la tuvo.

    Es determinista dado (node, arrival_day), así que dos rutas que llegan al
    mismo nodo el mismo día ven la misma realización.

    Parameters
    ----------
    node         : node at which the truck just arrived.
    arrival_day  : min(start_day_idx + int(time_elapsed // 14), max_day).
    loads_stack  : [num_days, num_nodes, num_nodes] — historical loads.

    Returns
    -------
    np.ndarray of shape (num_nodes,) int8 — 1 = lane exists, 0 = lane absent.
    """
    return (loads_stack[arrival_day, node] > 0).astype(np.int8)


def build_rm_pen_stack(
    rate_stack:   np.ndarray,   # (num_days, N, N)
    loads_stack:  np.ndarray,   # (num_days, N, N)
    distance_arr: np.ndarray,   # (N, N)
    diesel_arr:   np.ndarray,   # (N, N)
) -> np.ndarray:                # (num_days, N, N) float32
    """Precompute the full stack of penalized reward matrices for all training days.

    Returns shape (num_days, N, N) float32 — one penalized reward matrix per day.
    Diagonal of each slice is BIG_M_PENALTY. Used by VectorRoutingEnv.
    """
    return np.stack([
        build_day_matrices(rate_stack[d], loads_stack[d], distance_arr, diesel_arr)[1]
        for d in range(len(rate_stack))
    ]).astype(np.float32)


def load_matrices(num_nodes: int) -> tuple:
    """Load data files and return multi-day stacks for rate/loads.

    Fixed matrices (time, distance) se leen de CSV únicos (no cambian con el día). El diésel sale del
    promedio por hub en los días de ENTRENAMIENTO del nuevo `fuel_cost_hub_jun_oct_2026.csv` (matriz
    constante por fila, igual que el fuel.csv viejo — ver metodologia_realizaciones_mercado_reefer.md y
    la nota de diseño en el plan: no se hace variar el diésel por día). `rate_stack`/`loads_stack` salen
    de los CSV diarios en datos/datos_finales_jun_oct_2026 (jun 1 – oct 31 2026); `rate_stack` es la
    tarifa MEDIA diaria de cada lane (usada tal cual en modo replay literal y para las features de
    disponibilidad esperada / P95 de recompensa; el generador de realizaciones usa los cuantiles crudos
    por separado, vía `market_process.py`).

    Returns
    -------
    time_matrix    : np.ndarray    (num_nodes × num_nodes) float64
    rate_stack     : np.ndarray    (num_days × num_nodes × num_nodes)
    loads_stack    : np.ndarray    (num_days × num_nodes × num_nodes)
    distance_arr   : np.ndarray    (num_nodes × num_nodes)
    diesel_arr     : np.ndarray    (num_nodes × num_nodes)
    ltr_stack      : np.ndarray    (num_nodes × num_days) float32 — load_to_truck_ratio diario del hub
                     de origen (sin pronóstico/delta; feature de entrada al agente).
    avail_expected_arr : np.ndarray (num_nodes × num_nodes) float32
                     E[loads] over train days per arc (valor esperado, no
                     umbralizado a existencia). Usado SOLO como feature de
                     entrada al agente (avail_prior, feature en model/encoder.py).
    reward_global_p95 : float — P95 de los rewards positivos en días de train.
    """
    cwd = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(cwd, "datos")

    # ── Fixed single-snapshot matrices (no cambian con el día) ───
    time_matrix_raw = pd.read_csv(os.path.join(data_dir, "duration.csv"), index_col=0)
    distance_raw    = pd.read_csv(os.path.join(data_dir, "distance.csv"), index_col=0)

    # ── Stacks diarios nuevos (jun-oct 2026) ──────────────────────
    dias = cargar_dias_mercado(data_dir)
    num_days = dias["count_loads"].shape[0]
    train_days = min(TRAIN_DAYS, num_days)
    print(f"Multi-day data  : {num_days} días totales (jun-oct 2026) | train={train_days} "
          f"| eval={num_days - train_days}")

    rate_stack_raw  = dias["mean"]
    loads_stack_raw = dias["count_loads"]
    ltr_stack_raw   = dias["ltr"][:, np.arange(dias["ltr"].shape[1]), np.arange(dias["ltr"].shape[1])]  # (D,N)

    # ── Diésel: promedio por hub en días de train, matriz constante por fila ──
    fuel = cargar_fuel_por_hub(data_dir)
    diesel_hub = fuel["precio"][:train_days].mean(axis=0)          # (194,) — alineado con orden_hubs
    diesel_raw_full = np.tile(diesel_hub[:, None], (1, len(diesel_hub)))  # (194, 194), fila = hub de origen

    # ── Slice fixed matrices / stacks to num_nodes ────────────────
    time_matrix  = time_matrix_raw.iloc[:num_nodes, :num_nodes].to_numpy(dtype=float)
    distance_arr = distance_raw.iloc[:num_nodes, :num_nodes].to_numpy(dtype=float)
    diesel_arr   = diesel_raw_full[:num_nodes, :num_nodes]

    rate_stack  = rate_stack_raw[:, :num_nodes, :num_nodes].astype(float)
    loads_stack = loads_stack_raw[:, :num_nodes, :num_nodes].astype(float)
    ltr_stack   = ltr_stack_raw[:, :num_nodes].T.astype(np.float32)   # (num_nodes, num_days)

    # ── Valor esperado de carga por lane (feature de entrada al agente) ──
    avail_expected_arr = loads_stack[:train_days].mean(axis=0).astype(np.float32)

    # P95 de los rewards positivos de las lanes en los días de entrenamiento.
    reward_train = lane_reward(
        rate_stack[:train_days], loads_stack[:train_days],
        distance_arr[None], diesel_arr[None],
    )
    _pos_vals = reward_train[reward_train > 0].ravel()
    reward_global_p95 = float(np.percentile(_pos_vals, 95)) if len(_pos_vals) > 0 else 1.0
    print(f"REWARD_GLOBAL_P95: {reward_global_p95:.4f}")

    return time_matrix, rate_stack, loads_stack, distance_arr, diesel_arr, \
           ltr_stack, avail_expected_arr, reward_global_p95
