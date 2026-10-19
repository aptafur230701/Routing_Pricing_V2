"""
inference.py
============
Inferencia en tiempo real con el modelo AM entrenado.

Carga el checkpoint am_checkpoint_<NUM_NODES>nodes.pt y genera una ruta desde
el nodo que el usuario ingrese, usando los datos del día más reciente. La ruta
se construye paso a paso en el entorno (argmax de la política en cada nodo).

Uso
---
    python inference.py
"""

import os

import numpy as np

from config import DEVICE, MAX_DURATION, AM_D_H, AM_N_HEADS, AM_N_LAYERS, AM_D_FF
from am_agent import AMRoutingAgent
from baselines import calcular_ev_esperado
from evaluation import make_replay_env, rollout_drl_env
from market_process import MarketProcess, asegurar_parametros
from problem_data import load_matrices
from transfer_learning import load_checkpoint_compatible

# ── Configuración ─────────────────────────────────────────────────────────────

NUM_NODES       = 10
CHECKPOINT_NAME = f"am_checkpoint_{NUM_NODES}nodes.pt"


# ── Carga del modelo ──────────────────────────────────────────────────────────

def load_agent(checkpoint_path: str) -> AMRoutingAgent:
    """Instancia el agente y carga los pesos del checkpoint.

    Usa load_checkpoint_compatible (transfer_learning.py) en vez de un
    load_state_dict estricto: tolera checkpoints guardados antes del
    condicionamiento en K (expansión en cero de context_net.proj), y es
    idéntico a un load estricto para checkpoints ya compatibles con la
    arquitectura actual. Usa la arquitectura de config.py (la misma que main.py).
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"No se encontró el checkpoint: {checkpoint_path}\n"
            f"Asegúrate de haber ejecutado main.py antes de correr inference.py."
        )

    agent, _critic = load_checkpoint_compatible(
        checkpoint_path, NUM_NODES, AM_D_H, AM_N_HEADS, AM_N_LAYERS, AM_D_FF, verbose=False,
    )
    agent.eval()
    return agent


# ── Carga de datos ────────────────────────────────────────────────────────────

def load_latest_day():
    """Carga las matrices y toma el último día disponible como snapshot actual.

    Devuelve un entorno de replay de ese único día (day_idx 0) y las features de
    mercado recortadas a ese mismo día, para que el índice 0 apunte a él.
    """
    time_matrix, rate_stack, loads_stack, distance_arr, diesel_arr, \
        ltr_stack, trucks_stack, avail_expected_arr, reward_global_p95 = load_matrices(NUM_NODES)

    # Último día del stack histórico = día más reciente disponible
    last = rate_stack.shape[0] - 1
    env = make_replay_env(
        time_matrix, rate_stack[last:], loads_stack[last:], distance_arr, diesel_arr, NUM_NODES,
        day_offset=last,
    )
    asegurar_parametros()
    ev_esperado = calcular_ev_esperado(MarketProcess(NUM_NODES), distance_arr, diesel_arr)   # una vez
    return dict(
        env=env,
        ev_esperado=ev_esperado,
        time_matrix_arr=np.asarray(time_matrix, dtype=float),
        distance_arr=distance_arr,
        ltr_stack=ltr_stack[:, last:],
        trucks_stack=trucks_stack[:, last:],
        avail_expected_arr=avail_expected_arr,
        reward_global_p95=reward_global_p95,
    )


# ── Inferencia ────────────────────────────────────────────────────────────────

def run_inference(agent: AMRoutingAgent, start_node: int, day: dict):
    """Genera una ruta greedy desde start_node y muestra el resultado."""
    import time

    print(f"\n  Generando ruta desde nodo {start_node}...")

    t0 = time.perf_counter()
    route, total_reward, time_elapsed = rollout_drl_env(
        agent, day["env"], start_node, 0,
        day["time_matrix_arr"], day["distance_arr"], day["ltr_stack"], day["trucks_stack"],
        NUM_NODES,
        reward_global_p95=day["reward_global_p95"],
        avail_expected_arr=day["avail_expected_arr"],
        ev_esperado=day["ev_esperado"],
    )
    inference_ms = (time.perf_counter() - t0) * 1000

    print("\n" + "=" * 50)
    if route is None:
        print("  ⚠  No se encontró una ruta válida desde ese nodo.")
        print(f"     (Tiempo máximo permitido: {MAX_DURATION:.1f} h)")
    else:
        ruta_str = " → ".join(str(n) for n in route)
        print(f"  Ruta      : {ruta_str}")
        print(f"  Reward    : {total_reward:.0f}")
        print(f"  Duración  : {time_elapsed:.2f} h  (máx {MAX_DURATION:.1f} h)")
        print(f"  Nodos vis.: {len(route) - 1} de {NUM_NODES}")
    print(f"  Inferencia: {inference_ms:.2f} ms")
    print("=" * 50)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    cwd             = os.path.dirname(os.path.abspath(__file__))
    checkpoint_path = os.path.join(cwd, CHECKPOINT_NAME)

    print("=" * 50)
    print(f"  AM Inference — {NUM_NODES} nodos")
    print(f"  Device     : {DEVICE}")
    print(f"  Checkpoint : {CHECKPOINT_NAME}")
    print("=" * 50)

    # Cargar modelo y datos una sola vez
    print("\nCargando modelo...")
    agent = load_agent(checkpoint_path)
    print("Cargando datos del día más reciente...")
    day = load_latest_day()
    print("Listo.\n")

    # Loop interactivo: el usuario elige el nodo de inicio
    while True:
        print(f"Ingresa el nodo de inicio (0 a {NUM_NODES - 1}), o 'q' para salir:")
        entrada = input("  > ").strip()

        if entrada.lower() == "q":
            print("\nSaliendo. ¡Hasta luego!")
            break

        if not entrada.isdigit():
            print(f"  ✗ Entrada inválida. Ingresa un número entre 0 y {NUM_NODES - 1}.\n")
            continue

        start_node = int(entrada)
        if start_node < 0 or start_node >= NUM_NODES:
            print(f"  ✗ Nodo fuera de rango. Debe estar entre 0 y {NUM_NODES - 1}.\n")
            continue

        run_inference(agent, start_node, day)
        print()


if __name__ == "__main__":
    main()
