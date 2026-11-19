"""
hpo_optuna.py
=============
Búsqueda de hiperparámetros PPO con Optuna (TPE + pruning), para reemplazar el
ajuste "a ojo" de:

    PPO_LR, PPO_CLIP_EPS, PPO_GAE_LAMBDA, PPO_ENTROPY_COEF_START,
    PPO_ENTROPY_COEF_END, PPO_GRAD_CLIP, PPO_BATCH_SIZE,
    SELF_CRITICAL_COEF, PPO_WEIGHT_DECAY

Qué NO se tunea acá (deliberado)
--------------------------------
· El presupuesto de entrenamiento (episodes_per_node / n_updates) no es un
  hiperparámetro de performance — más entrenamiento casi nunca empeora el
  resultado, solo cuesta tiempo. Se fija en la corrida FINAL con los
  hiperparámetros ganadores y presupuesto completo, no buscando el valor que
  da mejor score en una corrida corta.
· Arquitectura (AM_D_H, AM_N_HEADS, AM_N_LAYERS, AM_D_FF) — cambia el tamaño
  del modelo, se deja fuera de este estudio por alcance explícito.
· USE_SELF_CRITICAL (bool) — es un switch de ablación, no un hiperparámetro
  continuo; se deja fijo en True y solo se tunea su coeficiente.

Diseño del objective
---------------------
Cada trial entrena con un presupuesto FIJO y corto (--updates-per-trial),
no hasta convergencia — sería tan caro como la corrida final, multiplicado
por n_trials. En su lugar:
  · Se usa el `greedy_reward` que am_training.run_am_training ya loguea
    periódicamente (cada log_freq updates) como métrica de progreso.
  · Se reporta a Optuna vía trial.report() en cada punto de log, habilitando
    pruning (MedianPruner) de trials claramente malos antes de terminar.
  · El valor final del trial es el promedio de los últimos 3 greedy_reward
    reportados (reduce ruido vs. tomar solo el último punto).
  · SELF_CRITICAL_WARMUP_UPDATES se reescala proporcionalmente al presupuesto
    del trial (25%), para que SELF_CRITICAL_COEF realmente tenga efecto
    dentro de una corrida corta (con el valor de producción, 50 updates,
    nunca se activaría en un trial de ~40 updates).
  · Cada trial corre --n-seeds-per-trial semillas (default 2: SEED, SEED+1, ...)
    con los MISMOS hiperparámetros, y el valor del trial es el promedio entre
    semillas — una sola semilla en PPO da rankings poco fiables (Colas et al.
    2018; Henderson et al. 2018). El pruning opera tanto dentro de cada
    semilla (corta trials claramente malos antes de terminarla) como entre
    semillas (no corre la 2a semilla si la 1a ya se podó).
  · am_training.SEED (no solo el seed de numpy/torch global) se sobreescribe
    por semilla, porque run_am_training se reseedea internamente con esa
    constante — y de paso selecciona eval_start_nodes/eval_start_days
    (greedy eval periódico), así que varían con la semilla. Eso es
    intencional: para la MISMA semilla, todos los trials ven exactamente las
    mismas instancias de evaluación (comparación justa entre hiperparámetros);
    promediar entre semillas de paso cubre más combinaciones nodo/día.

Dónde correr esto
------------------
Solo en la instancia más chica (--num-nodes 10 por default): la arquitectura
es invariante a permutación y los hiperparámetros de optimización (lr, clip,
entropía, gae) transfieren razonablemente entre tamaños de problema — no hace
falta (ni es viable en costo) repetir el estudio completo por cada N.

Uso
---
  python hpo_optuna.py                                  # 50 trials, 10 nodos, 2 semillas/trial
  python hpo_optuna.py --n-trials 100 --updates-per-trial 60
  python hpo_optuna.py --sampler random --n-trials 50     # control: TPE debe ganarle a esto
  python hpo_optuna.py --study-name mi_estudio --storage sqlite:///foo.db

Salida
------
  results/<N>nodes/optuna_study.db           — storage SQLite (resumible)
  results/<N>nodes/optuna_trials.csv         — todos los trials
  results/<N>nodes/best_ppo_hyperparams.json — mejores hiperparámetros
  results/<N>nodes/optuna_optimization_history.png
  results/<N>nodes/optuna_param_importances.png
"""

import argparse
import json
import os

import numpy as np
import optuna
import torch

import am_training
from baselines.politicas import calcular_ev_pronostico
from config import SEED, set_seeds
from market_process import MarketProcess, asegurar_parametros
from problem_data import load_matrices

# Hiperparámetros congelados de referencia (para restaurar entre trials y
# para fijar lo que NO se tunea, p.ej. PPO_N_EPISODES_PER_UPDATE).
_BASELINE = {
    name: getattr(am_training, name)
    for name in (
        "PPO_LR", "PPO_CLIP_EPS", "PPO_GAE_LAMBDA",
        "PPO_ENTROPY_COEF_START", "PPO_ENTROPY_COEF_END",
        "PPO_GRAD_CLIP", "PPO_BATCH_SIZE", "SELF_CRITICAL_COEF",
        "PPO_WEIGHT_DECAY", "SELF_CRITICAL_WARMUP_UPDATES",
        "get_episodes_per_node", "PPO_N_EPISODES_PER_UPDATE", "SEED",
    )
}
_SEED_STEP_OFFSET = 100_000  # separa el espacio de `step` por semilla dentro
                              # de un mismo trial (ver nota de diseño arriba)


def _restore_baseline():
    for name, value in _BASELINE.items():
        setattr(am_training, name, value)


def make_objective(num_nodes: int, updates_per_trial: int, data: dict, seeds: list):
    n_episodes_per_update = _BASELINE["PPO_N_EPISODES_PER_UPDATE"]
    episodes_per_node_trial = int(np.ceil(
        updates_per_trial * n_episodes_per_update / num_nodes
    ))
    warmup_trial = max(5, int(0.25 * updates_per_trial))

    def objective(trial: optuna.Trial) -> float:
        entropy_start = trial.suggest_float("ppo_entropy_coef_start", 0.01, 0.2, log=True)
        entropy_end_ratio = trial.suggest_float("entropy_end_ratio", 0.05, 0.9)

        am_training.PPO_LR                    = trial.suggest_float("ppo_lr", 1e-5, 1e-3, log=True)
        am_training.PPO_CLIP_EPS              = trial.suggest_float("ppo_clip_eps", 0.1, 0.3)
        am_training.PPO_GAE_LAMBDA            = trial.suggest_float("ppo_gae_lambda", 0.90, 0.99)
        am_training.PPO_ENTROPY_COEF_START    = entropy_start
        am_training.PPO_ENTROPY_COEF_END      = entropy_start * entropy_end_ratio
        am_training.PPO_GRAD_CLIP             = trial.suggest_float("ppo_grad_clip", 0.3, 2.0)
        am_training.PPO_BATCH_SIZE            = trial.suggest_categorical("ppo_batch_size", [64, 128, 256, 512])
        am_training.SELF_CRITICAL_COEF        = trial.suggest_float("self_critical_coef", 0.0, 1.0)
        am_training.PPO_WEIGHT_DECAY          = trial.suggest_float("ppo_weight_decay", 1e-6, 1e-3, log=True)
        am_training.SELF_CRITICAL_WARMUP_UPDATES = warmup_trial
        am_training.get_episodes_per_node     = lambda n: episodes_per_node_trial

        pruned_holder = {"pruned": False}
        seed_values = []

        try:
            for seed_idx, seed in enumerate(seeds):
                reported = []

                def _cb(step, n_updates, log_entry, _seed_idx=seed_idx):
                    val = log_entry.get("greedy_reward")
                    if val is None:
                        return False
                    reported.append(val)
                    trial.report(val, _seed_idx * _SEED_STEP_OFFSET + step)
                    if trial.should_prune():
                        pruned_holder["pruned"] = True
                        return True
                    return False

                am_training.SEED = seed
                set_seeds(seed)
                agent, critic, _, _, _ = am_training.run_am_training(
                    data["time_matrix"], data["rate_stack"], data["loads_stack"],
                    data["distance_arr"], data["diesel_arr"], num_nodes,
                    ltr_stack=data["ltr_stack"],
                    reward_global_p95=data["reward_global_p95"],
                    avail_expected_arr=data["avail_expected_arr"],
                    ev_esperado=data["ev_esperado"],
                    trial_callback=_cb,
                )
                del agent, critic
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                if pruned_holder["pruned"]:
                    break
                if reported:
                    seed_values.append(float(np.mean(reported[-3:])))
        finally:
            _restore_baseline()

        if pruned_holder["pruned"]:
            raise optuna.TrialPruned()
        if not seed_values:
            return float("nan")
        return float(np.mean(seed_values))

    return objective


def run(num_nodes: int, n_trials: int, updates_per_trial: int,
        storage: str, study_name: str, n_seeds_per_trial: int = 2,
        sampler_name: str = "tpe"):
    cwd = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(cwd, "results", f"{num_nodes}nodes")
    os.makedirs(out_dir, exist_ok=True)
    storage = storage or f"sqlite:///{os.path.join(out_dir, 'optuna_study.db')}"

    print(f"Cargando datos de {num_nodes} nodos...")
    (time_matrix, rate_stack, loads_stack, distance_arr, diesel_arr,
     ltr_stack, avail_expected_arr,
     reward_global_p95) = load_matrices(num_nodes)
    data = dict(
        time_matrix=time_matrix, rate_stack=rate_stack, loads_stack=loads_stack,
        distance_arr=distance_arr, diesel_arr=diesel_arr,
        ltr_stack=ltr_stack,
        avail_expected_arr=avail_expected_arr,
        reward_global_p95=reward_global_p95,
    )
    # HPO entrena con días reales (sin market_process): el ev sale del proceso ajustado, una vez.
    asegurar_parametros()
    data["ev_esperado"] = calcular_ev_pronostico(MarketProcess(num_nodes), distance_arr, diesel_arr)

    n_episodes_per_update = _BASELINE["PPO_N_EPISODES_PER_UPDATE"]
    episodes_per_node_trial = int(np.ceil(updates_per_trial * n_episodes_per_update / num_nodes))
    seeds = [SEED + i for i in range(n_seeds_per_trial)]
    print(f"Presupuesto por trial: {updates_per_trial} updates x {n_seeds_per_trial} semillas "
          f"{seeds} ({episodes_per_node_trial * num_nodes} episodios/semilla)")

    if sampler_name == "random":
        sampler = optuna.samplers.RandomSampler(seed=SEED)
    else:
        # multivariate=True: modela la distribución conjunta de los 9
        # hiperparámetros (captura interacciones, p.ej. LR con batch_size),
        # en vez de tratarlos como independientes (default de Optuna).
        sampler = optuna.samplers.TPESampler(seed=SEED, multivariate=True)

    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        load_if_exists=True,
        direction="maximize",
        sampler=sampler,
        # n_warmup_steps está en unidades de `step` (= número de update, no de
        # reportes) — tiene que cubrir la fase ruidosa inicial del greedy_reward
        # (ver nota en make_objective) antes de permitir podar un trial.
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=5,
            n_warmup_steps=max(10, int(0.3 * updates_per_trial)),
        ),
    )

    objective = make_objective(num_nodes, updates_per_trial, data, seeds)
    try:
        study.optimize(objective, n_trials=n_trials)
    finally:
        _restore_baseline()

    print(f"\n{'='*70}")
    print(f"  Mejor trial: #{study.best_trial.number}  (greedy_reward={study.best_value:.2f})")
    print(f"{'='*70}")
    for k, v in study.best_params.items():
        print(f"  {k:28s} = {v}")

    study.trials_dataframe().to_csv(os.path.join(out_dir, "optuna_trials.csv"), index=False)

    best_params = dict(study.best_params)
    entropy_start = best_params.pop("ppo_entropy_coef_start")
    entropy_end_ratio = best_params.pop("entropy_end_ratio")
    config_mapping = {
        "PPO_LR":                 best_params["ppo_lr"],
        "PPO_CLIP_EPS":           best_params["ppo_clip_eps"],
        "PPO_GAE_LAMBDA":         best_params["ppo_gae_lambda"],
        "PPO_ENTROPY_COEF_START": entropy_start,
        "PPO_ENTROPY_COEF_END":   entropy_start * entropy_end_ratio,
        "PPO_GRAD_CLIP":          best_params["ppo_grad_clip"],
        "PPO_BATCH_SIZE":         best_params["ppo_batch_size"],
        "SELF_CRITICAL_COEF":     best_params["self_critical_coef"],
        "PPO_WEIGHT_DECAY":       best_params["ppo_weight_decay"],
    }
    with open(os.path.join(out_dir, "best_ppo_hyperparams.json"), "w") as f:
        json.dump(config_mapping, f, indent=2)

    print(f"\nHiperparámetros ganadores → config.py (copiar a mano, no se auto-aplica):")
    for k, v in config_mapping.items():
        print(f"  {k:24s} = {v}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    try:
        from optuna.visualization.matplotlib import plot_optimization_history
        plot_optimization_history(study)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "optuna_optimization_history.png"), dpi=150)
        plt.close()
        print(f"\nGráfico guardado: {out_dir}/optuna_optimization_history.png")
    except Exception as e:
        print(f"\n[aviso] no se pudo generar optimization_history: {e}")

    try:
        from optuna.visualization.matplotlib import plot_param_importances
        plot_param_importances(study)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "optuna_param_importances.png"), dpi=150)
        plt.close()
        print(f"Gráfico guardado: {out_dir}/optuna_param_importances.png")
    except Exception as e:
        print(f"[aviso] no se pudo generar param_importances (requiere 'pip install scikit-learn'): {e}")

    return study


def _parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--num-nodes", type=int, default=10)
    p.add_argument("--n-trials", type=int, default=50)
    p.add_argument("--updates-per-trial", type=int, default=60,
                   help="Updates por trial (> SELF_CRITICAL_WARMUP_UPDATES de producción=50, "
                        "para que el self-critical llegue a activarse en el trial)")
    p.add_argument("--n-seeds-per-trial", type=int, default=2,
                   help="Semillas por trial, promediadas (default 2 — una sola semilla "
                        "en PPO da rankings poco fiables, ver Colas et al. 2018)")
    p.add_argument("--sampler", type=str, default="tpe", choices=["tpe", "random"],
                   help="'random' sirve como control: si TPE no le gana claramente con el "
                        "mismo presupuesto, el ruido domina sobre la señal del sampler")
    p.add_argument("--storage", type=str, default=None,
                   help="Default: sqlite:///results/<N>nodes/optuna_study.db")
    p.add_argument("--study-name", type=str, default=None,
                   help="Default: ppo_hpo_<N>nodes_<sampler>")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run(
        num_nodes=args.num_nodes,
        n_trials=args.n_trials,
        updates_per_trial=args.updates_per_trial,
        storage=args.storage,
        study_name=args.study_name or f"ppo_hpo_{args.num_nodes}nodes_{args.sampler}",
        n_seeds_per_trial=args.n_seeds_per_trial,
        sampler_name=args.sampler,
    )
