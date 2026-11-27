"""
hpo_optuna.py
=============
Búsqueda de hiperparámetros PPO con Optuna (TPE + pruning). Ajusta:

    PPO_LR, PPO_CLIP_EPS, PPO_GAE_LAMBDA, PPO_ENTROPY_COEF_START,
    PPO_ENTROPY_COEF_END, PPO_ENTROPY_COEF_PRECIO_START, PPO_ENTROPY_COEF_PRECIO_END,
    PPO_GRAD_CLIP, PPO_BATCH_SIZE, SELF_CRITICAL_COEF, PPO_WEIGHT_DECAY

main.py lo llama antes de entrenar (ver obtener_hiperparametros); también se puede correr solo.

Qué NO se tunea acá (deliberado)
--------------------------------
· El presupuesto de entrenamiento (episodes_per_node / n_updates) no es un
  hiperparámetro de desempeño: más entrenamiento casi nunca empeora el resultado,
  solo cuesta tiempo. Se fija por tamaño en config.get_episodes_per_node.
· Arquitectura (AM_D_H, AM_N_HEADS, AM_N_LAYERS, AM_D_FF): fuera de alcance.
· PPO_GAMMA = 1.0 (horizonte finito de 2-5 pasos) y PPO_N_EPOCHS = 4 (Schulman et al. 2017).
· USE_SELF_CRITICAL (bool): switch de ablación; se tunea solo su coeficiente.

Objetivo
--------
Cada trial entrena en MUNDOS SIMULADOS (market_process, igual que el entrenamiento de
producción) con un presupuesto fijo y corto (--updates-per-trial), y se mide con el
`val_score` de am_training.ValidacionSimulada: margen USD del ciclo completo del greedy,
con las rutas que no vuelven a casa penalizadas (PENALIZACION_USD), promediado sobre
los mundos fijos de MARKET_SEED_VAL y los niveles de K. Es el mismo criterio con el que
se elige el checkpoint en producción. Los mundos de test (MARKET_SEED_EVAL y
MARKET_SEED_EVAL_FINAL) no se usan nunca aquí.

  · El val_score se reporta a Optuna en cada punto de log (trial.report) para el
    pruning (MedianPruner).
  · El valor de una semilla es el promedio de sus últimos 3 val_score (menos ruido que
    el último punto); el valor del trial es el promedio entre semillas: una sola semilla
    en PPO da rankings poco fiables (Colas et al. 2018; Henderson et al. 2018).
  · SELF_CRITICAL_WARMUP_UPDATES se reescala al 25% del presupuesto del trial, para que
    SELF_CRITICAL_COEF tenga efecto dentro de una corrida corta.
  · am_training.SEED se sobreescribe por semilla. Para la MISMA semilla todos los trials
    ven la misma secuencia de mundos de entrenamiento y los mismos mundos de validación.

Dónde correr esto
------------------
En el tamaño más chico de la cadena (--num-nodes 20 por default). Los hiperparámetros
ganadores se usan en todos los tamaños: la arquitectura es invariante a permutación y lo
que se tunea son hiperparámetros del optimizador, no del tamaño del problema (como en
Kool et al. 2019, que usa los mismos para 20, 50 y 100 nodos).

Uso
---
  python hpo_optuna.py                                  # 50 trials, 20 nodos, 2 semillas/trial
  python hpo_optuna.py --n-trials 1                     # medir cuánto tarda un trial
  python hpo_optuna.py --sampler random --n-trials 50   # control: TPE debe ganarle a esto

  El estudio es resumible: si se corta, al relanzarlo continúa desde el SQLite y solo
  corre los trials que faltan para llegar a --n-trials.

Salida
------
  results/<N>nodes/optuna_study.db           — storage SQLite (resumible)
  results/<N>nodes/optuna_trials.csv         — todos los trials
  results/<N>nodes/best_ppo_hyperparams.json — mejores hiperparámetros (nombres de config.py)
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
import config
from config import SEED, TRAIN_DAYS, set_seeds
from market_process import MarketProcess, asegurar_parametros
from problem_data import load_matrices

HPO_NUM_NODES = 20   # tamaño en el que se corre el HPO (el primero de transfer_learning.NODE_SEQUENCE)

# Nombres (de config.py / am_training) de los hiperparámetros que salen del HPO.
HP_TUNEADOS = (
    "PPO_LR", "PPO_CLIP_EPS", "PPO_GAE_LAMBDA",
    "PPO_ENTROPY_COEF_START", "PPO_ENTROPY_COEF_END",
    "PPO_ENTROPY_COEF_PRECIO_START", "PPO_ENTROPY_COEF_PRECIO_END",
    "PPO_GRAD_CLIP", "PPO_BATCH_SIZE", "SELF_CRITICAL_COEF", "PPO_WEIGHT_DECAY",
)

# Valores de referencia (para restaurar entre trials y para fijar lo que NO se tunea).
_BASELINE = {
    name: getattr(am_training, name)
    for name in HP_TUNEADOS + (
        "SELF_CRITICAL_WARMUP_UPDATES", "get_episodes_per_node", "PPO_N_EPISODES_PER_UPDATE", "SEED",
    )
}
_SEED_STEP_OFFSET = 100_000  # separa el espacio de `step` por semilla dentro de un mismo trial


def _restore_baseline():
    for name, value in _BASELINE.items():
        setattr(am_training, name, value)


def _params_a_config(p: dict) -> dict:
    """Parámetros de Optuna (nombres del trial) → nombres de config.py."""
    return {
        "PPO_LR":                        p["ppo_lr"],
        "PPO_CLIP_EPS":                  p["ppo_clip_eps"],
        "PPO_GAE_LAMBDA":                p["ppo_gae_lambda"],
        "PPO_ENTROPY_COEF_START":        p["ppo_entropy_coef_start"],
        "PPO_ENTROPY_COEF_END":          p["ppo_entropy_coef_start"] * p["entropy_end_ratio"],
        "PPO_ENTROPY_COEF_PRECIO_START": p["ppo_entropy_coef_precio_start"],
        "PPO_ENTROPY_COEF_PRECIO_END":   p["ppo_entropy_coef_precio_start"] * p["entropy_precio_end_ratio"],
        "PPO_GRAD_CLIP":                 p["ppo_grad_clip"],
        "PPO_BATCH_SIZE":                p["ppo_batch_size"],
        "SELF_CRITICAL_COEF":            p["self_critical_coef"],
        "PPO_WEIGHT_DECAY":              p["ppo_weight_decay"],
    }


def aplicar_hiperparametros(hps: dict) -> None:
    """Escribe los hiperparámetros en am_training (lo que lee run_am_training) y en config."""
    for name, value in hps.items():
        if name not in HP_TUNEADOS:
            raise KeyError(f"{name} no es un hiperparámetro del HPO ({HP_TUNEADOS})")
        setattr(am_training, name, value)
        setattr(config, name, value)


def _sugerir(trial: optuna.Trial) -> dict:
    """Espacio de búsqueda. Rangos en torno a los valores de la literatura de PPO
    (Schulman et al. 2017; Andrychowicz et al. 2021; Huang et al. 2022)."""
    p = {
        "ppo_entropy_coef_start":        trial.suggest_float("ppo_entropy_coef_start", 0.01, 0.2, log=True),
        "entropy_end_ratio":             trial.suggest_float("entropy_end_ratio", 0.05, 0.9),
        # Cabeza de precio: entropía máxima log(10) ≈ 2.3 (9 niveles + VACÍO), entre 0.005 y 0.2.
        "ppo_entropy_coef_precio_start": trial.suggest_float("ppo_entropy_coef_precio_start", 0.005, 0.2, log=True),
        "entropy_precio_end_ratio":      trial.suggest_float("entropy_precio_end_ratio", 0.05, 0.9),
        "ppo_lr":                        trial.suggest_float("ppo_lr", 1e-5, 1e-3, log=True),
        "ppo_clip_eps":                  trial.suggest_float("ppo_clip_eps", 0.1, 0.3),
        "ppo_gae_lambda":                trial.suggest_float("ppo_gae_lambda", 0.90, 0.99),
        "ppo_grad_clip":                 trial.suggest_float("ppo_grad_clip", 0.3, 2.0),
        "ppo_batch_size":                trial.suggest_categorical("ppo_batch_size", [64, 128, 256, 512]),
        "self_critical_coef":            trial.suggest_float("self_critical_coef", 0.0, 1.0),
        "ppo_weight_decay":              trial.suggest_float("ppo_weight_decay", 1e-6, 1e-3, log=True),
    }
    return _params_a_config(p)


def make_objective(num_nodes: int, updates_per_trial: int, data: dict, seeds: list):
    n_episodes_per_update = _BASELINE["PPO_N_EPISODES_PER_UPDATE"]
    episodes_per_node_trial = int(np.ceil(updates_per_trial * n_episodes_per_update / num_nodes))
    warmup_trial = max(5, int(0.25 * updates_per_trial))

    def objective(trial: optuna.Trial) -> float:
        aplicar_hiperparametros(_sugerir(trial))
        am_training.SELF_CRITICAL_WARMUP_UPDATES = warmup_trial
        am_training.get_episodes_per_node = lambda n: episodes_per_node_trial

        pruned = False
        seed_values = []
        try:
            for seed_idx, seed in enumerate(seeds):
                reported = []

                def _cb(step, n_updates, log_entry, _seed_idx=seed_idx):
                    nonlocal pruned
                    val = log_entry.get("val_score")
                    if val is None or not np.isfinite(val):
                        return False
                    reported.append(val)
                    trial.report(val, _seed_idx * _SEED_STEP_OFFSET + step)
                    if trial.should_prune():
                        pruned = True
                        return True
                    return False

                am_training.SEED = seed
                set_seeds(seed)
                agent, critic, _, _, _ = am_training.run_am_training(
                    data["time_matrix"], data["rate_train"], data["loads_train"],
                    data["distance_arr"], data["diesel_arr"], num_nodes,
                    ltr_stack=data["ltr_stack"],
                    reward_global_p95=data["reward_global_p95"],
                    avail_expected_arr=data["avail_expected_arr"],
                    market_process=data["market_process"],
                    est_lane=data["est_lane"],
                    trial_callback=_cb,
                )
                del agent, critic
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                if pruned:
                    break
                if reported:
                    seed_values.append(float(np.mean(reported[-3:])))
        finally:
            _restore_baseline()

        if pruned:
            raise optuna.TrialPruned()
        if not seed_values:
            return float("nan")
        return float(np.mean(seed_values))

    return objective


def run(num_nodes: int = HPO_NUM_NODES, n_trials: int = 50, updates_per_trial: int = 60,
        storage: str = None, study_name: str = None, n_seeds_per_trial: int = 2,
        sampler_name: str = "tpe", k_restringido: bool = False) -> dict:
    """Corre (o continúa) el estudio y devuelve los mejores hiperparámetros (nombres de config.py).

    k_restringido: igual que K_RESTRINGIDO de main.py (niveles de K de entrenamiento y validación).
    """
    cwd = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(cwd, "results", f"{num_nodes}nodes")
    os.makedirs(out_dir, exist_ok=True)
    storage = storage or f"sqlite:///{os.path.join(out_dir, 'optuna_study.db')}"
    # Sufijo "valsim": objetivo sobre ValidacionSimulada. No mezclar con estudios del objetivo anterior.
    study_name = study_name or f"ppo_hpo_{num_nodes}nodes_{sampler_name}_valsim"

    config.configurar_k(k_restringido)
    # Los overrides por tamaño (config.HP_POR_NODOS) no deben pisar lo que propone cada trial.
    config.HP_OVERRIDES.clear()

    print(f"Cargando datos de {num_nodes} nodos...")
    (time_matrix, rate_stack, loads_stack, distance_arr, diesel_arr,
     ltr_stack, avail_expected_arr, reward_global_p95) = load_matrices(num_nodes)
    asegurar_parametros()
    market_process = MarketProcess(num_nodes)
    data = dict(
        time_matrix=time_matrix,
        rate_train=rate_stack[:TRAIN_DAYS], loads_train=loads_stack[:TRAIN_DAYS],   # como main.py
        distance_arr=distance_arr, diesel_arr=diesel_arr,
        ltr_stack=ltr_stack, avail_expected_arr=avail_expected_arr,
        reward_global_p95=reward_global_p95,
        market_process=market_process,
        est_lane=market_process.estadisticas_dow(distance_arr, diesel_arr),
    )

    n_episodes_per_update = _BASELINE["PPO_N_EPISODES_PER_UPDATE"]
    episodes_per_node_trial = int(np.ceil(updates_per_trial * n_episodes_per_update / num_nodes))
    seeds = [SEED + i for i in range(n_seeds_per_trial)]
    print(f"Presupuesto por trial: {updates_per_trial} updates x {n_seeds_per_trial} semillas "
          f"{seeds} ({episodes_per_node_trial * num_nodes} episodios/semilla)")

    if sampler_name == "random":
        sampler = optuna.samplers.RandomSampler(seed=SEED)
    else:
        # multivariate=True: modela la distribución conjunta (interacciones, p.ej. LR con batch_size).
        sampler = optuna.samplers.TPESampler(seed=SEED, multivariate=True)

    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        load_if_exists=True,
        direction="maximize",
        sampler=sampler,
        # n_warmup_steps en unidades de `step` (= número de update): cubre la fase ruidosa inicial.
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=5,
            n_warmup_steps=max(10, int(0.3 * updates_per_trial)),
        ),
    )

    terminados = (optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.PRUNED)
    hechos = sum(t.state in terminados for t in study.trials)
    faltan = max(0, n_trials - hechos)
    print(f"Estudio '{study_name}': {hechos} trials terminados, faltan {faltan} para llegar a {n_trials}")

    objective = make_objective(num_nodes, updates_per_trial, data, seeds)
    try:
        if faltan:
            study.optimize(objective, n_trials=faltan)
    finally:
        _restore_baseline()

    print(f"\n{'='*70}")
    print(f"  Mejor trial: #{study.best_trial.number}  (val_score={study.best_value:.2f} USD)")
    print(f"{'='*70}")

    study.trials_dataframe().to_csv(os.path.join(out_dir, "optuna_trials.csv"), index=False)

    mejores = _params_a_config(study.best_params)
    with open(os.path.join(out_dir, "best_ppo_hyperparams.json"), "w") as f:
        json.dump(mejores, f, indent=2)
    print("Hiperparámetros ganadores (guardados en best_ppo_hyperparams.json):")
    for k, v in mejores.items():
        print(f"  {k:30s} = {v}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    try:
        from optuna.visualization.matplotlib import plot_optimization_history
        plot_optimization_history(study)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "optuna_optimization_history.png"), dpi=150)
        plt.close()
    except Exception as e:
        print(f"[aviso] no se pudo generar optimization_history: {e}")

    try:
        from optuna.visualization.matplotlib import plot_param_importances
        plot_param_importances(study)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "optuna_param_importances.png"), dpi=150)
        plt.close()
    except Exception as e:
        print(f"[aviso] no se pudo generar param_importances (requiere 'pip install scikit-learn'): {e}")

    return mejores


def obtener_hiperparametros(modo: str = "auto", num_nodes: int = HPO_NUM_NODES, **kwargs_run) -> dict:
    """Hiperparámetros para el entrenamiento, según `modo`:

    "auto" : si existe results/<N>nodes/best_ppo_hyperparams.json lo carga; si no, corre el HPO.
    "force": corre (o continúa) el HPO aunque el JSON exista.
    "off"  : no hace nada y devuelve {} (se usan los valores de config.py).
    """
    if modo == "off":
        return {}
    cwd = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(cwd, "results", f"{num_nodes}nodes", "best_ppo_hyperparams.json")
    if modo == "auto" and os.path.exists(path):
        with open(path) as f:
            hps = json.load(f)
        faltan = set(HP_TUNEADOS) - set(hps)
        if not faltan:
            print(f"HPO: se reusan los hiperparámetros de {path}")
            return hps
        print(f"HPO: {path} no trae {sorted(faltan)} (es de una versión anterior); se corre el HPO")
    return run(num_nodes=num_nodes, **kwargs_run)


def _parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--num-nodes", type=int, default=HPO_NUM_NODES)
    p.add_argument("--n-trials", type=int, default=50)
    p.add_argument("--updates-per-trial", type=int, default=60,
                   help="Updates por trial (el warmup del self-critical se reescala al 25%%)")
    p.add_argument("--n-seeds-per-trial", type=int, default=2,
                   help="Semillas por trial, promediadas (una sola semilla en PPO da rankings poco fiables)")
    p.add_argument("--sampler", type=str, default="tpe", choices=["tpe", "random"],
                   help="'random' sirve como control: si TPE no le gana con el mismo presupuesto, domina el ruido")
    p.add_argument("--storage", type=str, default=None,
                   help="Default: sqlite:///results/<N>nodes/optuna_study.db")
    p.add_argument("--study-name", type=str, default=None,
                   help="Default: ppo_hpo_<N>nodes_<sampler>_valsim")
    p.add_argument("--k-restringido", action="store_true",
                   help="Entrena y valida con K libre y K = K_MAX (como K_RESTRINGIDO=True en main.py)")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run(
        num_nodes=args.num_nodes,
        n_trials=args.n_trials,
        updates_per_trial=args.updates_per_trial,
        storage=args.storage,
        study_name=args.study_name,
        n_seeds_per_trial=args.n_seeds_per_trial,
        sampler_name=args.sampler,
        k_restringido=args.k_restringido,
    )
