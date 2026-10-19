"""
main.py
=======
Entry point for the AM-PPO pipeline.

  1. Load data          (problem_data.py)
  2. Training           (am_training.py)
  3. Evaluation         (evaluation.py: mundos simulados + replay; baselines B1-B3, UB-PI, UB-DP)
  4. Save results       (evaluation.py, excel_report.py, stats_analysis.py)

Uso
---
  Cambia NUM_NODES en main() (opciones: 10 · 20 · 35 · 50 · 75 · 100).

  Luego ejecuta simplemente:
    python main.py

Checkpoints (transfer learning)
--------------------------------
  Los .pt se guardan en  checkpoints/  (raíz del proyecto, ignorada por git).
  Para usar transfer learning, copia el .pt del tamaño anterior en esa carpeta
  antes de entrenar (o entrena en orden y se guarda sólo).
"""

import os
import random
import time
import warnings
import numpy as np
import pandas as pd
import torch

warnings.filterwarnings("ignore")

from config import (
    SEED, DEVICE, RUN_TAG,
    get_episodes_per_node, TRAIN_DAYS,
    AM_D_H, AM_N_HEADS, AM_N_LAYERS, AM_D_FF,
)
from baselines import calcular_ev_esperado
from problem_data import load_matrices
from dp_relajada import cargar_o_calcular_tabla
from evaluation import (
    evaluar_mundos_simulados, evaluar_replay, resumen_metodos,
    save_results, plot_diagnostics, plot_ppo_diagnostics,
)
from market_process import MarketProcess, asegurar_parametros
from stats_analysis import append_stats_sheets_to_excel, NOTA_REPLAY
from transfer_learning import build_agent_for_training, load_checkpoint_compatible


def set_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def _run_am(
    NUM_NODES, time_matrix, rate_stack, loads_stack,
    distance_arr, diesel_arr, cwd,
    pretrained_agent=None, pretrained_critic=None,
    ltr_stack=None, trucks_stack=None,
    reward_global_p95=1.0, avail_expected_arr=None,
    market_process=None, best_checkpoint_path=None, ev_esperado=None,
):
    """Pipeline AM: entrenamiento PPO → checkpoint.

    market_process: MarketProcess opcional; si se pasa, el entrenamiento sortea
    cargas y tarifas del proceso exógeno en vez de reproducir días del histórico.
    """
    from am_training import run_am_training

    episodes_per_node = get_episodes_per_node(NUM_NODES)
    num_episodes      = episodes_per_node * NUM_NODES

    print(f"Device           : {DEVICE}")
    print(f"Training episodes: {num_episodes}")

    set_seeds(SEED)
    t0 = time.time()
    agent_am, critic, ep_rewards, ep_losses, training_log = run_am_training(
        time_matrix, rate_stack, loads_stack,
        distance_arr, diesel_arr, NUM_NODES,
        pretrained_agent=pretrained_agent,
        pretrained_critic=pretrained_critic,
        ltr_stack=ltr_stack, trucks_stack=trucks_stack,
        reward_global_p95=reward_global_p95,
        avail_expected_arr=avail_expected_arr,
        market_process=market_process,
        best_checkpoint_path=best_checkpoint_path,
        ev_esperado=ev_esperado,
    )
    train_time = time.time() - t0
    print(f"Training time: {train_time:.1f} s")

    torch.save(
        {"agent": agent_am.state_dict(), "critic": critic.state_dict()},
        os.path.join(cwd, f"am_checkpoint_{NUM_NODES}nodes.pt"),
    )
    return agent_am, critic, ep_rewards, ep_losses, train_time, training_log


def _checkpoint_path(checkpoint_dir: str, num_nodes: int, best: bool = False) -> str:
    """Último update (am_checkpoint_<N>nodes.pt) o mejor update de validación (..._best.pt)."""
    sufijo = "_best" if best else ""
    return os.path.join(checkpoint_dir, f"am_checkpoint_{num_nodes}nodes{sufijo}.pt")


def _load_agent(checkpoint_dir: str, num_nodes: int, best: bool = False) -> tuple:
    """Carga agente + critic entrenados desde checkpoint sin reentrenar.

    Usa load_checkpoint_compatible (transfer_learning.py) en vez de un
    load_state_dict estricto: tolera checkpoints guardados antes del
    condicionamiento en K (expansión en cero de context_net.proj), y es idéntico a un load estricto para
    checkpoints ya compatibles con la arquitectura actual.
    """
    path = _checkpoint_path(checkpoint_dir, num_nodes, best)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint no encontrado: {path}")
    agent, critic = load_checkpoint_compatible(
        path, num_nodes, AM_D_H, AM_N_HEADS, AM_N_LAYERS, AM_D_FF, verbose=False
    )
    agent.eval()
    critic.eval()
    print(f"Checkpoint cargado: {path}")
    return agent, critic


def _evaluate_and_report(
    agent, NUM_NODES, time_matrix, rate_stack, loads_stack,
    distance_arr, diesel_arr, ep_rewards, ep_losses,
    train_time, summary_rows, cwd, label,
    ltr_stack=None, trucks_stack=None,
    avail_expected_arr=None, reward_global_p95=1.0, market_process=None,
):
    """Evaluación en mundos simulados y replay + resumen + guardado de archivos.

    Mundos simulados: DRL, B1, B2 (K finito) y B3 sobre M mundos por nivel de K, con UB-PI y
    UB-DP. Replay: el mismo arnés sobre los días reales de evaluación (TRAIN_DAYS en adelante),
    descriptivo y no independiente del simulador.
    """
    tabla = cargar_o_calcular_tabla(market_process, time_matrix, distance_arr, diesel_arr)
    # B1, B2, B3, UB-PI y UB-DP no dependen del DRL: se calculan una vez por nº de nodos y se reusan
    # entre corridas (semillas, ablaciones) mientras no cambien el proceso, la tabla V ni el diseño.
    cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "datos", "cache_eval")

    print("\n--- Evaluación en mundos simulados ---")
    sim_df = evaluar_mundos_simulados(
        agent, market_process, tabla, time_matrix,
        rate_stack[:TRAIN_DAYS], loads_stack[:TRAIN_DAYS], distance_arr, diesel_arr, NUM_NODES,
        ltr_stack, trucks_stack, avail_expected_arr, reward_global_p95,
        cache_path=os.path.join(cache_dir, f"baselines_simulado_{NUM_NODES}nodos.pkl"),
    )
    print("\n--- Replay de los días reales de evaluación ---")
    replay_df = evaluar_replay(
        agent, market_process, tabla, time_matrix,
        rate_stack[TRAIN_DAYS:], loads_stack[TRAIN_DAYS:], distance_arr, diesel_arr, NUM_NODES,
        ltr_stack[:, TRAIN_DAYS:], trucks_stack[:, TRAIN_DAYS:], avail_expected_arr, reward_global_p95,
        cache_path=os.path.join(cache_dir, f"baselines_replay_{NUM_NODES}nodos.pkl"),
    )

    print(f"\n{'='*60}")
    print(f"  SUMMARY — {NUM_NODES} nodes  [{label}]")
    print(f"{'='*60}")
    for titulo, df in (("Mundos simulados", sim_df), (f"Replay ({NOTA_REPLAY})", replay_df)):
        print(f"\n  {titulo}")
        with pd.option_context("display.width", 220, "display.max_columns", None):
            print(resumen_metodos(df).to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    print(f"\n  Entrenamiento del DRL        : {train_time:.1f} s")
    print(f"  Tabla V de UB-DP             : {tabla.tiempo_calculo:.1f} s")
    print(f"{'='*60}")

    summary_rows.append({
        "Model":                  label,
        "Node Size":              NUM_NODES,
        "DRL Training Time (s)":  train_time,
        "UB-DP Table Time (s)":   tabla.tiempo_calculo,
    })

    excel_path = os.path.join(cwd, f"DRL_Routing_Summary_{label}.xlsx")
    plot_path  = os.path.join(cwd, f"DRL_Training_Diagnostics_{label}.png")

    save_results(sim_df, replay_df, summary_rows, excel_path)
    append_stats_sheets_to_excel(excel_path, sim_df, replay_df)
    if ep_rewards:
        plot_diagnostics(ep_rewards, ep_losses, sim_df, NUM_NODES, plot_path)

    return sim_df, replay_df


def main():
    NUM_NODES       = 35     # opciones: 10 · 20 · 35 · 50 · 75 · 100
    EVAL_ONLY       = False   # True: carga checkpoint y salta entrenamiento
    USE_TRANSFER    = True   # True: warm-start desde el checkpoint del tamaño anterior en
                              # NODE_SEQUENCE (requiere mismo AM_D_H/AM_N_LAYERS que la fuente).
                              # False: entrena desde pesos aleatorios con la arquitectura de config.py.
    EVAL_BEST_CHECKPOINT = True  # True: evalúa el update con mejor margen en los mundos de validación
                                 # (checkpoints/am_checkpoint_<N>nodes_best.pt, solo si se entrenó con el
                                 # proceso exógeno); False: evalúa el último update. El checkpoint
                                 # am_checkpoint_<N>nodes.pt (último) se guarda siempre, y es el que usa
                                 # el transfer learning.
    USE_MARKET_PROCESS = True  # True: el entrenamiento sortea cargas y tarifas del proceso exógeno
                                # (proceso_exogeno_cargas_tarifas.md) en vez de reproducir días del
                                # histórico. False: entrena con días reales. La evaluación siempre
                                # usa el proceso (mundos simulados) y además el replay de los días
                                # de eval. Si datos/market_params.npz no existe, corre el ajuste.
    # ──────────────────────────────────────────────────────────────────────────

    cwd          = os.path.dirname(os.path.abspath(__file__))
    summary_rows = []

    print(f"\n{'='*60}")
    print(f"  AM — {NUM_NODES} nodes")
    print(f"{'='*60}")

    set_seeds(SEED)

    etiqueta = f"_{RUN_TAG}" if RUN_TAG else ""    # RUN_TAG (config.py): ablaciones sin pisar otras corridas
    output_dir = os.path.join(cwd, "results", f"{NUM_NODES}nodes{etiqueta}")
    os.makedirs(output_dir, exist_ok=True)

    # Carpeta compartida de checkpoints (para transfer learning entre ramas)
    checkpoint_dir = os.path.join(cwd, "checkpoints", RUN_TAG) if RUN_TAG else os.path.join(cwd, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    time_matrix, rate_stack, loads_stack, distance_arr, diesel_arr, \
        ltr_stack, trucks_stack, avail_expected_arr, reward_global_p95 = load_matrices(NUM_NODES)

    # ── Proceso exógeno de cargas y tarifas ──────────────────────────────────
    # La evaluación (UB-PI, UB-DP, B2, B3 y los mundos simulados) lo necesita siempre;
    # el entrenamiento lo usa solo si USE_MARKET_PROCESS.
    asegurar_parametros()
    market_process = MarketProcess(NUM_NODES)
    ev_esperado = calcular_ev_esperado(market_process, distance_arr, diesel_arr)   # una vez por nº de nodos

    # ── Train / eval split (temporal — el modelo no ve los días de eval) ──────
    rate_train  = rate_stack[:TRAIN_DAYS]
    loads_train = loads_stack[:TRAIN_DAYS]


    if EVAL_ONLY:
        usar_best = EVAL_BEST_CHECKPOINT and os.path.exists(_checkpoint_path(checkpoint_dir, NUM_NODES, True))
        agent_am, critic  = _load_agent(checkpoint_dir, NUM_NODES, best=usar_best)
        ep_r, ep_l, t, training_log = [], [], 0.0, []
    else:
        agent_pretrained, critic_pretrained = build_agent_for_training(
            checkpoint_dir, NUM_NODES, use_transfer=USE_TRANSFER,
            d_h=AM_D_H, n_heads=AM_N_HEADS, n_layers=AM_N_LAYERS, d_ff=AM_D_FF,
        )
        agent_am, critic, ep_r, ep_l, t, training_log = _run_am(
            NUM_NODES, time_matrix, rate_train, loads_train,
            distance_arr, diesel_arr, checkpoint_dir,
            agent_pretrained, critic_pretrained,
            ltr_stack=ltr_stack, trucks_stack=trucks_stack,
            reward_global_p95=reward_global_p95,
            avail_expected_arr=avail_expected_arr,
            market_process=market_process if USE_MARKET_PROCESS else None,
            best_checkpoint_path=_checkpoint_path(checkpoint_dir, NUM_NODES, True),
            ev_esperado=ev_esperado,
        )
        best_path = _checkpoint_path(checkpoint_dir, NUM_NODES, True)
        if EVAL_BEST_CHECKPOINT and USE_MARKET_PROCESS and os.path.exists(best_path):
            print("Se evalúa el mejor update de validación (EVAL_BEST_CHECKPOINT).")
            agent_am, critic = _load_agent(checkpoint_dir, NUM_NODES, best=True)
        else:
            print("Se evalúa el último update.")
        ppo_plot_path = os.path.join(output_dir, "PPO_Diagnostics_AM.png")
        plot_ppo_diagnostics(training_log, NUM_NODES, ppo_plot_path)

        import pickle
        tlog_path = os.path.join(output_dir, f"training_log_{NUM_NODES}nodes.pkl")
        with open(tlog_path, "wb") as _f:
            pickle.dump(training_log, _f)
        print(f"training_log guardado en: {tlog_path}")

        tlog_xlsx_path = os.path.join(output_dir, f"training_log_{NUM_NODES}nodes.xlsx")
        pd.DataFrame(training_log).to_excel(tlog_xlsx_path, index=False)
        print(f"training_log (Excel) guardado en: {tlog_xlsx_path}")

    _evaluate_and_report(
        agent_am, NUM_NODES, time_matrix, rate_stack, loads_stack,
        distance_arr, diesel_arr, ep_r, ep_l, t,
        summary_rows, output_dir, label="AM",
        ltr_stack=ltr_stack, trucks_stack=trucks_stack,
        avail_expected_arr=avail_expected_arr,
        reward_global_p95=reward_global_p95,
        market_process=market_process,
    )

    print(f"\nDone! Checkpoint guardado en: checkpoints/am_checkpoint_{NUM_NODES}nodes.pt")


if __name__ == "__main__":
    main()
