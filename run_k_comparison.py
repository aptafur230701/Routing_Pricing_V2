"""
run_k_comparison.py
====================
Script aparte (NO forma parte del pipeline de main.py) para correr la evaluación en mundos
simulados (DRL, B1, B2, B3 contra UB-PI y UB-DP) bajo uno o varios presupuestos de
cardinalidad K, con un checkpoint ya entrenado (no reentrena nada).

Uso
---
  python run_k_comparison.py                         # NUM_NODES=10, K = K_MAX y caso libre
  python run_k_comparison.py --num-nodes 10 --k 5
  python run_k_comparison.py --k 3 5 --sin-libre --n-worlds 50

Salida
------
  results/<N>nodes/K_Comparison_AM.xlsx   — hoja 'Resumen por K' + hoja 'Mundos' (todas las filas)
  Devuelve también (df, resumen) si se importa como módulo.
"""

import argparse
import os

import pandas as pd

from config import (
    SEED, AM_D_H, AM_N_HEADS, AM_N_LAYERS, AM_D_FF, K_MAX, TRAIN_DAYS, MARKET_M_EVAL_WORLDS, set_seeds,
)
from baselines.dp_relajada import cargar_o_calcular_tabla
from evaluation import evaluar_mundos_simulados, resumen_metodos
from market_process import MarketProcess, asegurar_parametros
from problem_data import load_matrices
from stats_analysis import comparar_con_baselines
from transfer_learning import load_checkpoint_compatible


def run(num_nodes: int, k_levels, n_worlds: int, checkpoint_dir: str = None,
        output_path: str = None):
    """k_levels: lista de enteros y/o None (caso libre)."""
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

    time_matrix, rate_stack, loads_stack, distance_arr, diesel_arr, \
        ltr_stack, avail_expected_arr, reward_global_p95 = load_matrices(num_nodes)

    asegurar_parametros()
    market = MarketProcess(num_nodes)
    k_tabla = max([K_MAX] + [k for k in k_levels if k is not None])
    tabla = cargar_o_calcular_tabla(market, time_matrix, distance_arr, diesel_arr, K=k_tabla)

    df = evaluar_mundos_simulados(
        agent, market, tabla, time_matrix,
        rate_stack[:TRAIN_DAYS], loads_stack[:TRAIN_DAYS], distance_arr, diesel_arr, num_nodes,
        ltr_stack, avail_expected_arr, reward_global_p95,
        M=n_worlds, k_levels=tuple(k_levels),
    )
    resumen = resumen_metodos(df)

    print(f"\n{'='*70}")
    print(f"  Resumen por K — {num_nodes} nodos, {n_worlds} mundos")
    print(f"{'='*70}")
    with pd.option_context("display.width", 220, "display.max_columns", None):
        print(resumen.to_string(index=False, float_format=lambda x: f"{x:.2f}"))

    output_dir = os.path.join(cwd, "results", f"{num_nodes}nodes")
    os.makedirs(output_dir, exist_ok=True)
    output_path = output_path or os.path.join(output_dir, "K_Comparison_AM.xlsx")
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        resumen.to_excel(writer, sheet_name="Resumen por K", index=False)
        comparar_con_baselines(df).to_excel(writer, sheet_name="Comparación pareada", index=False)
        df.assign(ruta=df["ruta"].astype(str)).to_excel(writer, sheet_name="Mundos", index=False)
    print(f"\nResultados guardados → {output_path}")

    return df, resumen


def _parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--num-nodes", type=int, default=10, help="Tamaño de la instancia (debe existir checkpoints/am_checkpoint_<N>nodes.pt)")
    p.add_argument("--k", type=int, nargs="+", default=[K_MAX], help=f"Niveles de K finitos a evaluar (default: {K_MAX})")
    p.add_argument("--sin-libre", action="store_true", help="No evaluar el caso libre (sin K)")
    p.add_argument("--n-worlds", type=int, default=MARKET_M_EVAL_WORLDS, help="Mundos simulados por nivel de K")
    p.add_argument("--checkpoint-dir", type=str, default=None, help="Carpeta de checkpoints (default: ./checkpoints)")
    p.add_argument("--output", type=str, default=None, help="Ruta del .xlsx de salida (default: results/<N>nodes/K_Comparison_AM.xlsx)")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run(
        num_nodes=args.num_nodes,
        k_levels=list(args.k) + ([] if args.sin_libre else [None]),
        n_worlds=args.n_worlds,
        checkpoint_dir=args.checkpoint_dir,
        output_path=args.output,
    )
