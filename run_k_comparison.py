"""
run_k_comparison.py
====================
Script aparte (NO forma parte del pipeline de main.py) para correr la
comparación DRL-vs-8-baselines bajo presupuesto de cardinalidad K, usando
evaluation.run_solver_comparison_by_k (opt-in, ver plan de K en Solvers.py /
evaluation.py).

Carga un checkpoint ya entrenado (no reentrena nada) y evalúa, para cada
K en K_LEVELS (config.py: [K_MAX], el único nivel restringido — la
restricción real de negocio), todos los solvers — incluido el oráculo
LS-Oracle bajo K — para poder medir el gap real DRL-vs-óptimo bajo esa
restricción, no solo en el caso libre.

Uso
---
  python run_k_comparison.py                         # NUM_NODES=10, K_LEVELS completo
  python run_k_comparison.py --num-nodes 10 --k 5
  python run_k_comparison.py --n-days-per-node 2

Salida
------
  results/<N>nodes/K_Comparison_AM.xlsx   — una hoja con todas las filas (col. 'K')
                                              + un resumen agregado por K
  Devuelve también (df, timing) si se importa como módulo.
"""

import argparse
import os

import numpy as np
import pandas as pd

from config import (
    SEED, AM_D_H, AM_N_HEADS, AM_N_LAYERS, AM_D_FF, K_LEVELS, TRAIN_DAYS,
)
from problem_data import load_matrices
from transfer_learning import load_checkpoint_compatible
from evaluation import run_solver_comparison_by_k


def set_seeds(seed: int):
    import random
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def _summarize_by_k(df: pd.DataFrame) -> pd.DataFrame:
    """Resumen agregado: reward medio + gap vs LS-Oracle, por K."""
    rows = []
    for k, g in df.groupby("K"):
        def avg_valid(col, valid_col):
            mask = g[valid_col]
            return g.loc[mask, col].mean() if mask.any() else float("nan")

        ls_oracle_valid = g["LS-Oracle Valid"] & g["DRL Real Valid"]
        gap_series = g.loc[ls_oracle_valid].apply(
            lambda r: (r["LS-Oracle Reward"] - r["DRL Real Reward"]) / abs(r["LS-Oracle Reward"]) * 100
            if r["LS-Oracle Reward"] != 0 else float("nan"), axis=1,
        ).dropna()
        drl_real_gap = gap_series.mean() if len(gap_series) else float("nan")

        ls_exact_valid = g["LS-Exact Valid"] & g["DRL Det Valid"]
        gap_det_series = g.loc[ls_exact_valid].apply(
            lambda r: (r["LS-Exact Reward"] - r["DRL Det Reward"]) / abs(r["LS-Exact Reward"]) * 100
            if r["LS-Exact Reward"] != 0 else float("nan"), axis=1,
        ).dropna()
        drl_det_gap = gap_det_series.mean() if len(gap_det_series) else float("nan")

        rows.append({
            "K":                             k,
            "N":                             len(g),
            "DRL Det Avg Reward":            avg_valid("DRL Det Reward", "DRL Det Valid"),
            "DRL Real Avg Reward":           avg_valid("DRL Real Reward", "DRL Real Valid"),
            "LS-Exact Avg Reward":           avg_valid("LS-Exact Reward", "LS-Exact Valid"),
            "LS-Oracle Avg Reward":          avg_valid("LS-Oracle Reward", "LS-Oracle Valid"),
            "HGA-LNS Avg Reward":            avg_valid("HGA-LNS Reward", "HGA-LNS Valid"),
            "RH-Greedy Avg Reward":          avg_valid("RH-Greedy Reward", "RH-Greedy Valid"),
            "RH-Lookahead Avg Reward":       avg_valid("RH-Lookahead Reward", "RH-Lookahead Valid"),
            "RH-Greedy Real Avg Reward":     avg_valid("RH-Greedy Real Reward", "RH-Greedy Real Valid"),
            "RH-Lookahead Real Avg Reward":  avg_valid("RH-Lookahead Real Reward", "RH-Lookahead Real Valid"),
            "MC-Rollout Avg Reward":         avg_valid("MC-Rollout Reward", "MC-Rollout Valid"),
            "DRL Det Gap vs LS-Exact (%)":   drl_det_gap,
            "DRL Real Gap vs LS-Oracle (%)": drl_real_gap,
        })
    return pd.DataFrame(rows).sort_values("K").reset_index(drop=True)


def run(num_nodes: int, k_levels, n_days_per_node: int, checkpoint_dir: str = None,
        output_path: str = None):
    cwd = os.path.dirname(os.path.abspath(__file__))
    checkpoint_dir = checkpoint_dir or os.path.join(cwd, "checkpoints")
    checkpoint_path = os.path.join(checkpoint_dir, f"am_checkpoint_{num_nodes}nodes.pt")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint no encontrado: {checkpoint_path}")

    set_seeds(SEED)

    print(f"Cargando checkpoint: {checkpoint_path}")
    agent, critic = load_checkpoint_compatible(
        checkpoint_path, num_nodes, AM_D_H, AM_N_HEADS, AM_N_LAYERS, AM_D_FF, verbose=True,
    )
    agent.eval()
    critic.eval()

    time_matrix, rate_stack, loads_stack, distance_arr, diesel_arr, \
        ltr_stack, trucks_stack, avail_expected_arr, reward_global_p95 = load_matrices(num_nodes)

    # Mismo split train/eval que main.py — el modelo no vio estos días.
    rate_eval  = rate_stack[TRAIN_DAYS:]
    loads_eval = loads_stack[TRAIN_DAYS:]

    df, timing = run_solver_comparison_by_k(
        agent, time_matrix, rate_eval, loads_eval, distance_arr, diesel_arr, num_nodes,
        k_levels=k_levels,
        ltr_stack=ltr_stack, trucks_stack=trucks_stack, n_days_per_node=n_days_per_node, critic=critic,
        avail_expected_arr=avail_expected_arr,
    )

    summary = _summarize_by_k(df)

    print(f"\n{'='*70}")
    print(f"  Resumen por K — {num_nodes} nodos")
    print(f"{'='*70}")
    with pd.option_context("display.width", 200, "display.max_columns", None):
        print(summary.to_string(index=False, float_format=lambda x: f"{x:.1f}"))

    output_dir = os.path.join(cwd, "results", f"{num_nodes}nodes")
    os.makedirs(output_dir, exist_ok=True)
    output_path = output_path or os.path.join(output_dir, "K_Comparison_AM.xlsx")
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        summary.to_excel(writer, sheet_name="Summary by K", index=False)
        df.to_excel(writer, sheet_name="Raw cells", index=False)
    print(f"\nResultados guardados → {output_path}")

    return df, timing, summary


def _parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--num-nodes", type=int, default=10, help="Tamaño de la instancia (debe existir checkpoints/am_checkpoint_<N>nodes.pt)")
    p.add_argument("--k", type=int, nargs="+", default=None, help=f"Niveles de K a evaluar (default: K_LEVELS = {K_LEVELS})")
    p.add_argument("--n-days-per-node", type=int, default=1, help="Días de evaluación por nodo de inicio (default: 1, ver docstring de run_solver_comparison_by_k)")
    p.add_argument("--checkpoint-dir", type=str, default=None, help="Carpeta de checkpoints (default: ./checkpoints)")
    p.add_argument("--output", type=str, default=None, help="Ruta del .xlsx de salida (default: results/<N>nodes/K_Comparison_AM.xlsx)")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run(
        num_nodes=args.num_nodes,
        k_levels=args.k,
        n_days_per_node=args.n_days_per_node,
        checkpoint_dir=args.checkpoint_dir,
        output_path=args.output,
    )
