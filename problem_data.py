"""
problem_data.py
==============
Data loading and matrix computation.

load_matrices(num_nodes)
    → time_matrix, rate_stack, loads_stack, distance_arr, diesel_arr,
      ltr_stack, trucks_stack, avail_expected_arr, reward_global_p95

build_day_matrices(rate_day, loads_day, distance_arr, diesel_arr, num_nodes)
    → reward_matrix, reward_matrix_penalized

rate_stack  and loads_stack are 3-D numpy arrays shaped [num_days, num_nodes, num_nodes].
A random day is sampled once per episode in training/tuning so the agent learns
from historical variability rather than a single reward distribution snapshot.
"""

import os
import numpy as np
import pandas as pd

from config import (
    MPG,
    BIG_M_PENALTY, MARGINAL_COST_SIN_DIESEL,
    TRAIN_DAYS,
)


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

    Fixed matrices (time, distance, fuel) are loaded from single CSV files.
    Rate and loads are loaded from .npy stacks (datos/) so the training loop
    can sample one day per episode.

    Expected file shapes
    --------------------
    datos/rate.npy               : [num_days, N, N]
    datos/load_availability.npy  : [num_days, N, N]
    datos/ltr.npy                : [194, 120]
    datos/trucks_forward.npy     : [194, 600]  — MultiIndex (fecha, delta), 120 días × 5 deltas

    Returns
    -------
    time_matrix    : np.ndarray    (num_nodes × num_nodes) float64
    rate_stack     : np.ndarray    (num_days × num_nodes × num_nodes)
    loads_stack    : np.ndarray    (num_days × num_nodes × num_nodes)
    distance_arr   : np.ndarray    (num_nodes × num_nodes)
    diesel_arr     : np.ndarray    (num_nodes × num_nodes)
    ltr_stack      : np.ndarray    (num_nodes × 120)
    trucks_stack   : np.ndarray    (num_nodes × 120 × 3)  — solo deltas 1,2,3
    avail_expected_arr : np.ndarray (num_nodes × num_nodes) float32
                     E[loads] over train days per arc (valor esperado, no
                     umbralizado a existencia). Usado SOLO como feature de
                     entrada al agente (avail_prior, feature [6] en
                     attention_encoder.py).
    """
    cwd = os.path.dirname(os.path.abspath(__file__))

    # ── Fixed single-snapshot matrices ────────────────────────────
    data_dir = os.path.join(cwd, "datos")
    time_matrix_raw = pd.read_csv(os.path.join(data_dir, "duration.csv"), index_col=0)
    distance_raw    = pd.read_csv(os.path.join(data_dir, "distance.csv"), index_col=0)
    diesel_raw      = pd.read_csv(os.path.join(data_dir, "fuel.csv"),     index_col=0)

    # ── Multi-day stacks — shape: [num_days, N, N] ───────────────
    rate_stack_raw  = np.load(os.path.join(data_dir, "rate.npy"))
    loads_stack_raw = np.load(os.path.join(data_dir, "load_availability.npy"))

    # ── Market signal stacks ──────────────────────────────────────
    # ltr: [194, 120] → slice a [num_nodes, 120]
    ltr_raw = np.load(os.path.join(data_dir, "ltr.npy"))
    ltr_stack = ltr_raw[:num_nodes, :].astype(np.float32)

    # trucks_forward: [194, fechas × 5 deltas] (orden: fecha es nivel outer); el nº de fechas sale de la forma
    # del archivo. reshape a [num_nodes, fechas, 5] y tomar solo deltas 1,2,3 (índices 0,1,2)
    trucks_raw = np.load(os.path.join(data_dir, "trucks_forward.npy"))
    n_deltas = 5
    trucks_stack = trucks_raw[:num_nodes, :].astype(np.float32).reshape(num_nodes, -1, n_deltas)[:, :, 0:3]

    # ── Slice fixed matrices to num_nodes ─────────────────────────
    time_matrix  = time_matrix_raw.iloc[:num_nodes, :num_nodes].to_numpy(dtype=float)
    distance_arr = distance_raw.iloc[:num_nodes, :num_nodes].to_numpy(dtype=float)
    diesel_arr   = diesel_raw.iloc[:num_nodes, :num_nodes].to_numpy(dtype=float)

    # ── Slice stacks to num_nodes; align days to the shorter stack ──
    num_days    = min(rate_stack_raw.shape[0], loads_stack_raw.shape[0])
    rate_stack  = rate_stack_raw[:num_days, :num_nodes, :num_nodes].astype(float)
    loads_stack = loads_stack_raw[:num_days, :num_nodes, :num_nodes].astype(float)

    train_days = min(TRAIN_DAYS, num_days)
    eval_days  = num_days - train_days
    print(f"Multi-day data  : {num_days} días totales | train={train_days} | eval={eval_days}")

    # ── Valor esperado de carga por lane (feature de entrada al agente) ──
    # E[loads] sobre días de train, sin umbralizar a existencia: preserva la
    # magnitud real (loads va de 0 a >1900) en vez de colapsarla a P(loads>0).
    avail_expected_arr = loads_stack[:train_days].mean(axis=0).astype(np.float32)

    # P95 de los rewards positivos de las lanes en los días de entrenamiento.
    # Usado como normalizador global de rewards en la feature [7] del encoder.
    # Debe estar en las MISMAS unidades que el reward del paso (dólares por lane,
    # vía lane_reward), no en tarifa por milla: feature [7] = reward / P95.
    reward_train = lane_reward(
        rate_stack[:train_days], loads_stack[:train_days],
        distance_arr[None], diesel_arr[None],
    )
    _pos_vals = reward_train[reward_train > 0].ravel()
    reward_global_p95 = float(np.percentile(_pos_vals, 95)) if len(_pos_vals) > 0 else 1.0
    print(f"REWARD_GLOBAL_P95: {reward_global_p95:.4f}")

    return time_matrix, rate_stack, loads_stack, distance_arr, diesel_arr, \
           ltr_stack, trucks_stack, avail_expected_arr, reward_global_p95



