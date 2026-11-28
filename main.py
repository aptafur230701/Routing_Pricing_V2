"""
main.py
=======
Entry point for the AM-PPO pipeline.

  1. Load data          (problem_data.py)
  2. Training           (am_training.py, con los hiperparámetros PPO fijos de config.py)
  3. Evaluation         (evaluation.py: mundos simulados + replay; DRL, baselines B1-B3 y B2P, UB-PI, UB-DP)
  4. Save results       (Excel con hojas Resumen y Comparación + mundos_AM.csv con los datos por mundo)
  5. Validación del precio (validate_pricing.py: results/<N>nodes/pricing_val/)

  La validación de la decisión de precio va aparte: python validate_pricing.py --num-nodes N.

Uso
---
  Edita NODE_SIZES en main() (la cadena completa de transfer_learning.NODE_SEQUENCE es
  20 · 35 · 50 · 75 · 100). Luego ejecuta simplemente:
    python main.py

  Los hiperparámetros de PPO son fijos (config.py; salen del HPO de hpo_optuna.py, que se corrió una vez
  en 20 nodos y ya no forma parte del pipeline).

  Corre cada tamaño en orden: entrena, guarda checkpoint y resultados, y pasa al siguiente usando
  ese checkpoint recién guardado como punto de partida (transfer learning automático, ver
  transfer_learning.get_source_checkpoint). Si un tamaño falla, se imprime el error y se continúa
  con el siguiente en vez de abortar toda la corrida.

  Para correr un solo tamaño, deja NODE_SIZES con un único elemento, p. ej. NODE_SIZES = [10].

Checkpoints (transfer learning)
--------------------------------
  Los .pt se guardan en  checkpoints/  (raíz del proyecto; sí se versionan en git).
  Para usar transfer learning, copia el .pt del tamaño anterior en esa carpeta
  antes de entrenar (o entrena en orden y se guarda sólo).
"""

import os
import time
import warnings
import traceback
import torch
import pandas as pd

warnings.filterwarnings("ignore")

from config import (
    SEED, DEVICE, K_TRAINING_LEVELS, configurar_k, MARKET_SEED_EVAL, MARKET_SEED_EVAL_FINAL,
    HP_OVERRIDES, configurar_hp,
    get_n_updates, PPO_N_EPISODES_PER_UPDATE, set_seeds,
    AM_D_H, AM_N_HEADS, AM_N_LAYERS, AM_D_FF,
)
from problem_data import load_matrices
from baselines.dp_relajada import cargar_o_calcular_tabla
from evaluation import (
    evaluar_mundos_simulados, evaluar_replay, imprimir_resumen,
    save_results, plot_diagnostics, plot_ppo_diagnostics,
)
from market_process import MarketProcess, asegurar_parametros
from excel_report import guardar_mundos, agregar_rutas_y_precios
from stats_analysis import append_comparacion_to_excel, comparar_con_baselines, FAMILIAS, NOTA_REPLAY
from transfer_learning import (
    NODE_SEQUENCE, build_agent_for_training, load_checkpoint_compatible,
)


def _run_am(
    NUM_NODES, time_matrix, distance_arr, diesel_arr, cwd, market_process,
    pretrained_agent=None, pretrained_critic=None,
    reward_global_p95=1.0, avail_expected_arr=None,
    best_checkpoint_path=None, est_lane=None, resume_path=None, reanudar=False,
):
    """Pipeline AM: entrenamiento PPO (mundos simulados del proceso exógeno, en DEVICE) → checkpoint.

    resume_path / reanudar: ver am_training.run_am_training. Con reanudación, el tiempo de entrenamiento reportado
    suma el de las sesiones anteriores (el trabajo perdido entre el último guardado y el corte no cuenta)."""
    from am_training import run_am_training, tiempo_previo_reanudacion

    print(f"Device           : {DEVICE}")
    print(f"Training episodes: {get_n_updates(NUM_NODES) * PPO_N_EPISODES_PER_UPDATE}")

    set_seeds(SEED)
    t_previo = tiempo_previo_reanudacion(resume_path) if reanudar else 0.0
    t0 = time.time()
    agent_am, critic, ep_rewards, ep_losses, training_log = run_am_training(
        time_matrix, distance_arr, diesel_arr, NUM_NODES, market_process,
        avail_expected_arr=avail_expected_arr,
        reward_global_p95=reward_global_p95,
        pretrained_agent=pretrained_agent,
        pretrained_critic=pretrained_critic,
        best_checkpoint_path=best_checkpoint_path,
        est_lane=est_lane,
        resume_path=resume_path,
        reanudar=reanudar,
    )
    train_time = time.time() - t0 + t_previo
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


def _resume_path(checkpoint_dir: str, num_nodes: int) -> str:
    """Estado completo del entrenamiento para reanudarlo (am_training.run_am_training)."""
    return os.path.join(checkpoint_dir, f"am_resume_{num_nodes}nodes.pt")


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


def _tiempo_entrenamiento_previo(output_dir: str) -> float:
    """Tiempo de entrenamiento del DRL que ya trae el Excel de resultados (0.0 si no hay).

    Con EVAL_ONLY no se entrena; sin esto el Excel regrabado perdería el tiempo de la corrida
    que produjo el checkpoint."""
    import openpyxl
    path = os.path.join(output_dir, "DRL_Routing_Summary_AM.xlsx")
    if not os.path.exists(path):
        return 0.0
    try:
        for fila in openpyxl.load_workbook(path, read_only=True, data_only=True)["Resumen"].iter_rows(values_only=True):
            if fila and fila[0] == "Entrenamiento del DRL (s)" and isinstance(fila[1], (int, float)):
                return float(fila[1])
    except Exception:
        pass
    return 0.0


def _evaluate_and_report(
    agent, NUM_NODES, time_matrix,
    distance_arr, diesel_arr, ep_rewards, ep_losses,
    train_time, summary_rows, cwd, label,
    avail_expected_arr=None, reward_global_p95=1.0, market_process=None,
    semilla_eval=MARKET_SEED_EVAL, sufijo_cache="",
):
    """Evaluación en mundos simulados y replay + resumen + guardado de archivos.

    Mundos simulados: DRL, B1, B2, B2P y B3 sobre M mundos por nivel de K, con UB-PI y
    UB-DP. Replay: el mismo arnés sobre los días reales de evaluación (TRAIN_DAYS en adelante),
    fuera de muestra: ni el entrenamiento ni el ajuste del proceso exógeno usaron esos días.
    """
    # Antes de evaluar (decenas de minutos): los archivos de salida tienen que poderse sobrescribir. En Windows
    # un Excel abierto bloquea el archivo y la evaluación se perdería al final.
    for nombre in (f"DRL_Routing_Summary_{label}.xlsx", f"mundos_{label}.csv"):
        ruta = os.path.join(cwd, nombre)
        if os.path.exists(ruta):
            try:
                open(ruta, "r+b").close()
            except PermissionError:
                raise PermissionError(f"{ruta} está abierto en otro programa (¿Excel?): ciérralo y vuelve a correr")
    # La tabla solo calcula los K que se evalúan: con K libre (K_RESTRINGIDO = False) es un solo barrido
    # (K = 0) en vez de K = 1…K_MAX más el libre.
    k_tabla = max([k for k in K_TRAINING_LEVELS if k is not None], default=0)
    tabla = cargar_o_calcular_tabla(market_process, time_matrix, distance_arr, diesel_arr, K=k_tabla)
    # B1, B2, B2P, B3, UB-PI y UB-DP no dependen del DRL: se calculan una vez por nº de nodos y se reusan
    # entre corridas mientras no cambien el proceso, la tabla V ni el diseño.
    cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "datos", "cache_eval")

    print("\n--- Evaluación en mundos simulados ---")
    sim_df = evaluar_mundos_simulados(
        agent, market_process, tabla, time_matrix, distance_arr, diesel_arr, NUM_NODES,
        avail_expected_arr, reward_global_p95,
        k_levels=tuple(K_TRAINING_LEVELS),
        cache_path=os.path.join(cache_dir, f"baselines_simulado_{NUM_NODES}nodos{sufijo_cache}.pkl"),
        semilla_eval=semilla_eval,
    )
    print("\n--- Replay de los días reales de evaluación ---")
    replay_df = evaluar_replay(
        agent, market_process, tabla, time_matrix, distance_arr, diesel_arr, NUM_NODES,
        avail_expected_arr, reward_global_p95,
        k_levels=tuple(K_TRAINING_LEVELS),
        cache_path=os.path.join(cache_dir, f"baselines_replay_{NUM_NODES}nodos.pkl"),
    )

    print(f"\n{'='*60}")
    print(f"  SUMMARY — {NUM_NODES} nodes  [{label}]")
    print(f"{'='*60}")
    for titulo, df in (("Mundos simulados", sim_df), (f"Replay ({NOTA_REPLAY})", replay_df)):
        imprimir_resumen(df, titulo)
    _imprimir_comparacion_b(sim_df, replay_df)
    print(f"\n  Entrenamiento del DRL        : {train_time:.1f} s")
    print(f"  Tabla V de UB-DP             : " + ("no tratable" if tabla is None else f"{tabla.tiempo_calculo:.1f} s"))
    print(f"{'='*60}")

    summary_rows.append({
        "Model":                  label,
        "Node Size":              NUM_NODES,
        "DRL Training Time (s)":  train_time,
        "UB-DP Table Time (s)":   float("nan") if tabla is None else tabla.tiempo_calculo,
    })

    excel_path = os.path.join(cwd, f"DRL_Routing_Summary_{label}.xlsx")
    plot_path  = os.path.join(cwd, f"DRL_Training_Diagnostics_{label}.png")

    save_results(sim_df, replay_df, summary_rows, excel_path)
    append_comparacion_to_excel(excel_path, sim_df, replay_df)
    agregar_rutas_y_precios(excel_path, sim_df, replay_df)
    guardar_mundos(sim_df, replay_df, os.path.join(cwd, f"mundos_{label}.csv"))
    if ep_rewards:
        plot_diagnostics(ep_rewards, ep_losses, sim_df, NUM_NODES, plot_path)

    # Validación de la decisión de precio (nivel según el estado, sensibilidad a κ): results/<N>nodes/pricing_val/
    from validate_pricing import validar_pricing
    validar_pricing(agent, NUM_NODES, time_matrix, distance_arr, diesel_arr,
                    avail_expected_arr, reward_global_p95, market_process, cwd, tabla=tabla)

    return sim_df, replay_df


def _imprimir_comparacion_b(sim_df, replay_df) -> None:
    """Comparación B en consola: DRL − (destino del DRL + regla fija), con IC bootstrap y Holm."""
    df = sim_df if replay_df is None or replay_df.empty else pd.concat([sim_df, replay_df], ignore_index=True)
    b = comparar_con_baselines(df, familias={"precio": FAMILIAS["precio"]})
    if b.empty:
        return
    sig = [c for c in b.columns if c.startswith("significativo")][0]
    print("\n  Lo que aporta el precio del DRL (DRL − mismas rutas con una regla de precio fija):")
    for _, r in b.iterrows():
        modo = "replay" if str(r["modo"]).startswith("replay") else "simulado"
        print(f"    {modo:<9} K={r['K']:<6} {r['comparacion']:<26} "
              f"{r['dif_media_usd']:+9.1f} USD  IC95 [{r['ic95_inf']:+.1f}, {r['ic95_sup']:+.1f}]  "
              f"p_Holm={r['p_holm']:.3g}{'  *' if r[sig] else ''}")


def _run_one_size(
    NUM_NODES, cwd,
    EVAL_ONLY, USE_TRANSFER, EVAL_BEST_CHECKPOINT,
    EVAL_SET, K_RESTRINGIDO, REANUDAR=False,
):
    """Pipeline completo (entrenamiento + evaluación) para un solo tamaño de grafo.

    Aislado en su propia función para que main() pueda llamarlo en secuencia sobre varios
    NUM_NODES: cada llamada guarda su checkpoint y sus resultados antes de que empiece la
    siguiente, y build_agent_for_training (transfer_learning.py) recoge automáticamente el
    checkpoint del tamaño anterior en NODE_SEQUENCE.
    """
    configurar_k(K_RESTRINGIDO)    # antes de entrenar/evaluar: fija K_TRAINING_LEVELS en el sitio
    configurar_hp(NUM_NODES)       # hiperparámetros del tamaño (config.HP_POR_NODOS)

    summary_rows = []

    print(f"\n{'='*60}")
    print(f"  AM — {NUM_NODES} nodes")
    print(f"  Niveles de K: {['libre' if k is None else k for k in K_TRAINING_LEVELS]}")
    print(f"  Hiperparámetros propios de este tamaño: {dict(HP_OVERRIDES) or 'ninguno (los de config.py)'}")
    print(f"  Semilla de entrenamiento (SEED): {SEED}")
    print(f"{'='*60}")

    set_seeds(SEED)

    output_dir = os.path.join(cwd, "results", f"{NUM_NODES}nodes")
    os.makedirs(output_dir, exist_ok=True)

    # Carpeta compartida de checkpoints (para transfer learning entre tamaños).
    checkpoint_dir = os.path.join(cwd, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    time_matrix, _, _, distance_arr, diesel_arr, _, avail_expected_arr, reward_global_p95 = load_matrices(NUM_NODES)

    # ── Proceso exógeno de cargas y tarifas ──────────────────────────────────
    # El entrenamiento (mundos simulados de los TRAIN_DAYS) y la evaluación (mundos simulados, replay de los días
    # de prueba, UB-PI, UB-DP, B2, B2P y B3) salen de él. El modelo no ve los días de prueba.
    asegurar_parametros()
    market_process = MarketProcess(NUM_NODES)
    est_lane = market_process.estadisticas_dow(distance_arr, diesel_arr)   # una vez por nº de nodos

    usar_best = False
    if EVAL_ONLY:
        usar_best = EVAL_BEST_CHECKPOINT and os.path.exists(_checkpoint_path(checkpoint_dir, NUM_NODES, True))
        agent_am, critic  = _load_agent(checkpoint_dir, NUM_NODES, best=usar_best)
        ep_r, ep_l, t, training_log = [], [], _tiempo_entrenamiento_previo(output_dir), []
    else:
        agent_pretrained, critic_pretrained = build_agent_for_training(
            checkpoint_dir, NUM_NODES, use_transfer=USE_TRANSFER,
            d_h=AM_D_H, n_heads=AM_N_HEADS, n_layers=AM_N_LAYERS, d_ff=AM_D_FF,
        )
        agent_am, critic, ep_r, ep_l, t, training_log = _run_am(
            NUM_NODES, time_matrix, distance_arr, diesel_arr, checkpoint_dir, market_process,
            agent_pretrained, critic_pretrained,
            reward_global_p95=reward_global_p95,
            avail_expected_arr=avail_expected_arr,
            best_checkpoint_path=_checkpoint_path(checkpoint_dir, NUM_NODES, True),
            est_lane=est_lane,
            resume_path=_resume_path(checkpoint_dir, NUM_NODES),
            reanudar=REANUDAR,
        )
        best_path = _checkpoint_path(checkpoint_dir, NUM_NODES, True)
        if EVAL_BEST_CHECKPOINT and os.path.exists(best_path):
            usar_best = True
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

    label = "AM" if EVAL_SET == "dev" else "AM_final"
    _evaluate_and_report(
        agent_am, NUM_NODES, time_matrix,
        distance_arr, diesel_arr, ep_r, ep_l, t,
        summary_rows, output_dir, label=label,
        avail_expected_arr=avail_expected_arr,
        reward_global_p95=reward_global_p95,
        market_process=market_process,
        semilla_eval=MARKET_SEED_EVAL if EVAL_SET == "dev" else MARKET_SEED_EVAL_FINAL,
        sufijo_cache="" if EVAL_SET == "dev" else "_final",
    )

    print(f"\nDone! Checkpoint guardado en: {os.path.relpath(checkpoint_dir, cwd)}/am_checkpoint_{NUM_NODES}nodes.pt")


def main():
    # Tamaños a correr, en orden. La cadena completa de transfer_learning.NODE_SEQUENCE es
    # (20 → 35 → 50 → 75): cada tamaño arranca con el checkpoint recién guardado del
    # anterior. Para correr uno solo, deja aquí un único elemento, p. ej. NODE_SIZES = [20].
    # NODE_SIZES      = list(NODE_SEQUENCE)   # cadena completa: [20, 35, 50, 75]
    # NODE_SIZES      = [75]   # solo el que falta (20, 35 y 50 ya tienen checkpoint)
    NODE_SIZES      = [20]   # solo 20 nodos
    EVAL_ONLY       = os.environ.get("EVAL_ONLY", "0") == "1"   # True (EVAL_ONLY=1): carga checkpoint y salta entrenamiento
    REANUDAR        = os.environ.get("RESUME", "0") == "1"      # True (RESUME=1): sigue el entrenamiento desde
                                                                 # checkpoints/am_resume_<N>nodes.pt si existe (si ya
                                                                 # había terminado, pasa directo a la evaluación)
    USE_TRANSFER    = True   # True: inicializa los pesos desde el checkpoint del tamaño anterior. False: entrena desde pesos aleatorios con la
                              # arquitectura de config.py.
    EVAL_BEST_CHECKPOINT = False  # False (por defecto): evalúa el último update, el modelo que dejó el
                                 # entrenamiento ya con el LR decaído. True: evalúa el update con mejor
                                 # margen en los mundos de validación (checkpoints/am_checkpoint_<N>nodes_best.pt).
                                 # Se dejó en False porque elegir
                                 # el máximo entre ~80 validaciones de 200 mundos sobreajusta esa validación:
                                 # en 10, 20 y 35 nodos el último update rindió igual o mejor en los mundos
                                 # de test (8-32 USD). El checkpoint am_checkpoint_<N>nodes.pt (último) se
                                 # guarda siempre, y es el que usa el transfer learning.
    EVAL_SET = os.environ.get("EVAL_SET", "dev")   # "dev": mundos de MARKET_SEED_EVAL (desarrollo, resultados de siempre).
                                                    # "final": mundos de MARKET_SEED_EVAL_FINAL (test final, un solo uso con
                                                    # la configuración congelada); escribe DRL_Routing_Summary_AM_final.xlsx.
    K_RESTRINGIDO = False      # True: entrena, valida y evalúa con K libre y K = K_MAX (5).
                                # False: solo K libre (sin presupuesto de cardinalidad).
    # ──────────────────────────────────────────────────────────────────────────

    cwd = os.path.dirname(os.path.abspath(__file__))

    fallidos = []
    for NUM_NODES in NODE_SIZES:
        try:
            _run_one_size(
                NUM_NODES, cwd,
                EVAL_ONLY, USE_TRANSFER, EVAL_BEST_CHECKPOINT,
                EVAL_SET, K_RESTRINGIDO, REANUDAR,
            )
        except Exception:
            fallidos.append(NUM_NODES)
            print(f"\n{'!'*60}")
            print(f"  FALLÓ el tamaño {NUM_NODES} nodos — se continúa con el siguiente de NODE_SIZES")
            print(f"{'!'*60}")
            traceback.print_exc()
        finally:
            # Libera la memoria GPU del tamaño recién corrido (o del intento fallido) antes de
            # pasar al siguiente: todos los tamaños comparten el mismo proceso, y sin esto la
            # memoria reservada (y su fragmentación) se acumula de un tamaño al siguiente.
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    print(f"\n{'='*60}")
    print(f"  Secuencia terminada: {NODE_SIZES}")
    if fallidos:
        print(f"  Tamaños que fallaron: {fallidos} (revisa el traceback de cada uno arriba)")
    else:
        print("  Todos los tamaños terminaron sin errores.")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
