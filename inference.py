"""
inference.py
============
Inferencia en tiempo real con el modelo AM entrenado.

Carga el checkpoint del PPO (am_checkpoint_<NUM_NODES>nodes.pt) y genera, desde el nodo que el usuario
ingrese y con los datos del día más reciente, la RUTA del DRL: en cada nodo, el argmax del destino y del nivel
de precio. Por cada cotización muestra el precio cotizado frente a la mediana de la lane, la probabilidad de
ganar (1 − u)^m y si se ganó la carga; el competidor es simulado (semilla fija), las cargas y los cuantiles de
precio son los del día real. Escribe results/<N>nodes/rutas_<run_tag>.csv con una fila por paso (cotización
o viaje).

Uso
---
    python inference.py [--start-node K] [--num-nodes N] [--checkpoint RUTA]

  --start-node   corre un solo tour desde ese nodo, sin el loop interactivo
"""

import argparse
import os

import torch  # noqa: F401  (antes que pandas: en algunos Windows importar pandas primero rompe c10.dll de torch)
import numpy as np
import pandas as pd

from config import (
    DEVICE, MAX_DURATION, AM_D_H, AM_N_HEADS, AM_N_LAYERS, AM_D_FF, RUN_TAG, SEED,
)
from model.agent import AMRoutingAgent
from baselines.politicas import calcular_ev_pronostico
from evaluation import make_replay_env
from greedy_rollout import construir_features, contexto_agente
from market_process import MarketProcess, asegurar_parametros
from pricing import U_NIVELES_ARR, lane_cost_row, prob_ganar
from problem_data import load_matrices
from routing_env import OPCION_VACIO
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
        ltr_stack, avail_expected_arr, reward_global_p95 = load_matrices(NUM_NODES)

    # Último día del stack histórico = día más reciente disponible
    last = rate_stack.shape[0] - 1
    asegurar_parametros()
    market = MarketProcess(NUM_NODES)
    env = make_replay_env(
        time_matrix, rate_stack[last:], loads_stack[last:], distance_arr, diesel_arr, NUM_NODES,
        day_offset=last, market=market,
    )
    ev_esperado = calcular_ev_pronostico(market, distance_arr, diesel_arr)   # una vez
    return dict(
        env=env,
        ev_esperado=ev_esperado,
        time_matrix_arr=np.asarray(time_matrix, dtype=float),
        distance_arr=distance_arr,
        diesel_arr=diesel_arr,
        ltr_stack=ltr_stack[:, last:],
        avail_expected_arr=avail_expected_arr,
        reward_global_p95=reward_global_p95,
    )


# ── Inferencia ────────────────────────────────────────────────────────────────

CSV_COLUMNS = ["episode_id", "step", "from", "to", "opcion", "precio_cotizado", "mediana", "p_ganar",
               "gano", "break_even", "margen", "is_return"]


def _rollout(agent, start_node, day) -> tuple:
    """Rollout greedy del DRL paso a paso. Retorna (route, reward, duración, pasos); route es None si el tour
    no cierra el ciclo dentro de max_duration. Cada paso es una cotización (ganada o perdida) o un viaje VACÍO."""
    env = day["env"]
    T, dist, diesel = day["time_matrix_arr"], day["distance_arr"], day["diesel_arr"]
    starts = np.array([start_node], dtype=np.int64)
    masks = env.reset(starts, np.array([0], dtype=np.int64), k_max_arr=[None],
                      world_seeds=np.array([SEED], dtype=np.int64))
    route, pasos, total = [int(start_node)], [], 0.0
    valido = False
    with torch.no_grad():
        for _ in range(env.max_steps):
            i = int(env.current_node[0])
            feats = construir_features(env, starts, T, dist, day["ltr_stack"], day["reward_global_p95"], NUM_NODES,
                                       day["avail_expected_arr"], day["ev_esperado"], MAX_DURATION)
            h_t, emb = contexto_agente(agent, feats, env.current_node, DEVICE)
            a_t, o_t = agent.decoder.greedy_cotizacion(h_t, emb, torch.from_numpy(masks).to(DEVICE),
                                                       torch.from_numpy(feats["opciones"]).to(DEVICE))
            j, o = int(a_t[0]), int(o_t[0])
            vacio = o == OPCION_VACIO
            precio = float("nan") if vacio else float(env.precios_row[0][j, o])
            mediana = float(env.rate_row_obs[0][j])
            p_ganar = 1.0 if vacio else float(prob_ganar(U_NIVELES_ARR[o], env.competencia[0]))
            _, terminated, masks = env.step(np.array([j]), np.array([o]))
            gano = bool(env.last_gano[0])
            r = float(env.last_raw_reward[0])
            total += r
            if vacio or gano:
                route.append(j)
            pasos.append({"from": i, "to": j, "opcion": "VACIO" if vacio else int(o), "precio_cotizado": precio,
                          "mediana": mediana, "p_ganar": p_ganar, "gano": gano,
                          "break_even": float(lane_cost_row(dist[i, j], diesel[i, j]) / dist[i, j]),
                          "margen": r, "is_return": (vacio or gano) and j == start_node})
            if terminated[0]:
                valido = route[-1] == start_node and not bool(env.last_time_violation[0])
                break
    if not valido:
        return None, -np.inf, np.inf, pasos
    return route, total, float(env.time_elapsed[0]), pasos


def formatear_bloque(route, pasos) -> str:
    """Una línea por paso: cotización (nivel, precio frente a la mediana, P(ganar), resultado) o viaje VACÍO."""
    lineas = ["Ruta: " + " → ".join(str(n) for n in route), "Pasos:"]
    for a in pasos:
        retorno = "  (regreso a casa)" if a["is_return"] else ""
        if a["opcion"] == "VACIO":
            lineas.append(f"  {a['from']}→{a['to']}: sin carga cotizable (VACÍO), margen {a['margen']:.0f} USD{retorno}")
            continue
        cab = (f"  {a['from']}→{a['to']}: cotiza nivel {a['opcion']} = {a['precio_cotizado']:.2f} /mi "
               f"(mediana {a['mediana']:.2f}, break-even {a['break_even']:.2f}, P(ganar) {a['p_ganar']:.0%})")
        lineas.append(cab + (f" → GANA, margen {a['margen']:.0f} USD{retorno}" if a["gano"]
                             else " → pierde, sigue en el nodo"))
    return "\n".join(lineas)


def run_inference(agent: AMRoutingAgent, start_node: int, day: dict, filas_csv: list = None):
    """Genera la ruta del DRL desde start_node y muestra el resultado por arco."""
    import time

    print(f"\n  Generando ruta desde nodo {start_node}...")

    t0 = time.perf_counter()
    route, total_reward, time_elapsed, pasos = _rollout(agent, start_node, day)
    inference_ms = (time.perf_counter() - t0) * 1000

    print("\n" + "=" * 50)
    if route is None:
        print("  ⚠  No se encontró una ruta válida desde ese nodo.")
        print(f"     (Tiempo máximo permitido: {MAX_DURATION:.1f} h)")
    else:
        print(f"  Ruta      : {' → '.join(str(n) for n in route)}   (DRL)")
        print(f"  Reward    : {total_reward:.0f}")
        print(f"  Duración  : {time_elapsed:.2f} h  (máx {MAX_DURATION:.1f} h)")
        print(f"  Nodos vis.: {len(route) - 1} de {NUM_NODES}")
        print()
        print(formatear_bloque(route, pasos))
        if filas_csv is not None:
            ep = 1 + max((f["episode_id"] for f in filas_csv), default=-1)
            filas_csv.extend(dict(a, episode_id=ep, step=k) for k, a in enumerate(pasos))
    print(f"  Inferencia: {inference_ms:.2f} ms")
    print("=" * 50)


# ── Main ──────────────────────────────────────────────────────────────────────

def _escribir_csv(filas, cwd):
    out = os.path.join(cwd, "results", f"{NUM_NODES}nodes" + (f"_{RUN_TAG}" if RUN_TAG else ""))
    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, f"rutas_{RUN_TAG or 'inference'}.csv")
    pd.DataFrame(filas, columns=CSV_COLUMNS).to_csv(path, index=False)
    print(f"  Pasos de la ruta → {path}")


def main():
    global NUM_NODES, CHECKPOINT_NAME
    ap = argparse.ArgumentParser(description="Inferencia del AM entrenado (ruta y cotización del DRL por paso).")
    ap.add_argument("--start-node", type=int, help="un solo tour desde este nodo (sin loop interactivo)")
    ap.add_argument("--num-nodes", type=int, default=NUM_NODES)
    ap.add_argument("--checkpoint", help="ruta del .pt (default: checkpoints[/RUN_TAG]/am_checkpoint_<N>nodes.pt)")
    args = ap.parse_args()
    NUM_NODES = args.num_nodes
    CHECKPOINT_NAME = f"am_checkpoint_{NUM_NODES}nodes.pt"

    cwd             = os.path.dirname(os.path.abspath(__file__))
    checkpoint_path = args.checkpoint or os.path.join(cwd, "checkpoints", RUN_TAG, CHECKPOINT_NAME)

    print("=" * 50)
    print(f"  AM Inference — {NUM_NODES} nodos")
    print(f"  Device     : {DEVICE}")
    print(f"  Checkpoint : {checkpoint_path}")
    print("=" * 50)

    # Cargar modelo y datos una sola vez
    print("\nCargando modelo...")
    agent = load_agent(checkpoint_path)
    print("Cargando datos del día más reciente...")
    day = load_latest_day()
    print("Listo.\n")
    filas_csv = []

    if args.start_node is not None:
        run_inference(agent, args.start_node, day, filas_csv)
        if filas_csv:
            _escribir_csv(filas_csv, cwd)
        return

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

        run_inference(agent, start_node, day, filas_csv)
        if filas_csv:
            _escribir_csv(filas_csv, cwd)
        print()


if __name__ == "__main__":
    main()
