"""
inference.py
============
Inferencia en tiempo real con el modelo AM entrenado.

Carga el checkpoint con las cabezas Ṽ A/B (am_checkpoint_<NUM_NODES>nodes_pricing.pt, ver post_training.py) y
genera, desde el nodo que el usuario ingrese y con los datos del día más reciente, dos salidas:

  · la RUTA del DRL (actor): en cada nodo, el argmax de la política entrenada (el único método de ruteo);
  · el PRECIO DE RESERVA p* por arco (pricing.py): la tarifa mínima por milla a la que al camión le conviene
    aceptar la lane frente a su mejor alternativa. Se imprime también su rango [mín, máx] entre las parejas AB/BA
    (variabilidad del estimador por partición de datos y ruido de etiquetas, NO es un intervalo de confianza) y se
    escribe results/<N>nodes/pricing_<run_tag>.csv con una fila por arco.

p* es un costo de oportunidad del transportista; no es una probabilidad de aceptación del cliente. Solo se calcula
para lanes con carga: un arco sin carga (reposicionamiento) o el retorno a casa no llevan p*.

Uso
---
    python inference.py [--show-board] [--start-node K] [--num-nodes N] [--checkpoint RUTA]

  --show-board   en cada nodo, tabla con el p* de cada lane saliente con carga (además de la acción del DRL)
  --start-node   corre un solo tour desde ese nodo, sin el loop interactivo
"""

import argparse
import os

import numpy as np
import pandas as pd

from config import (
    DEVICE, MAX_DURATION, AM_D_H, AM_N_HEADS, AM_N_LAYERS, AM_D_FF, TRAIN_DAYS, RUN_TAG,
    REWARD_SCALE_FACTOR,
)
from model.agent import AMRoutingAgent
from baselines.politicas import calcular_ev_pronostico
from evaluation import make_replay_env
from greedy_rollout import greedy_rollout
from market_process import MarketProcess, asegurar_parametros
from pricing import lane_cost_row, percentil_historico, reservation_price
from problem_data import load_matrices
from transfer_learning import checkpoint_tiene_post_head, load_checkpoint_compatible

# ── Configuración ─────────────────────────────────────────────────────────────

NUM_NODES       = 10
CHECKPOINT_NAME = f"am_checkpoint_{NUM_NODES}nodes_pricing.pt"


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
        rate_train=rate_stack[:TRAIN_DAYS],            # serie histórica para el percentil de p*
        loads_train=loads_stack[:TRAIN_DAYS],          # (solo cuentan los días con carga)
        rate_last=rate_stack[last:], loads_last=loads_stack[last:],   # tablero del día (replay)
        ltr_stack=ltr_stack[:, last:],
        avail_expected_arr=avail_expected_arr,
        reward_global_p95=reward_global_p95,
    )


# ── Inferencia ────────────────────────────────────────────────────────────────

CSV_COLUMNS = ["episode_id", "step", "from", "to", "rate_hoy", "p_estrella", "p_min", "p_max", "break_even",
               "percentil_historico", "only_option", "is_return", "has_load"]


def _rollout_con_pricing(agent, start_node, day, with_board) -> tuple:
    """Rollout del DRL (actor, argmax de la política) con el p* de cada arco. Retorna (route, reward, duración,
    arcos, tableros).

    El actor elige la acción de cada paso (el único método de ruteo) y un hook de greedy_rollout calcula el
    precio de reserva del arco elegido antes de que el entorno lo ejecute — p* nunca re-decide la ruta, solo la
    valora. tableros[k] = lanes con carga ofertadas en el paso k. route es None si el tour no cierra el ciclo
    dentro de max_duration.
    """
    env, N = day["env"], NUM_NODES
    arcos, tableros = [], []

    def hook(c):
        h_t, emb = c["h_t"][:1], c["embeddings"][:1]
        i, j, mask = int(c["cur_nodes"][0]), int(c["actions"][0]), c["masks"][0]
        rate, loads = env.rate_row_obs[0].astype(np.float64), env.loads_row_obs[0]
        dist, cost = day["distance_arr"][i], lane_cost_row(day["distance_arr"][i], day["diesel_arr"][i])
        va, vb = (v.squeeze(0).detach().cpu().numpy().astype(np.float64) for v in agent.post_values(h_t, emb))

        def fila(a):
            pr = reservation_price(va, vb, mask, i, a, rate, dist, cost, start_node, REWARD_SCALE_FACTOR, loads_row=loads)
            return dict(
                **{"from": i, "to": a}, rate_hoy=float(rate[a]), p_estrella=pr.p, p_min=pr.p_min, p_max=pr.p_max,
                break_even=pr.break_even, only_option=pr.only_option, is_return=(a == start_node), has_load=pr.has_load,
                percentil_historico=percentil_historico(pr.p, day["rate_train"][:, i, a], day["loads_train"][:, i, a])
                if pr.definido else float("nan"),
            )

        arcos.append(fila(j))
        if with_board:
            ofertas = [a for a in range(N) if mask[a] and a != i and a != start_node and loads[a] > 0]
            tableros.append([dict(fila(a), elegida=(a == j)) for a in ofertas])

    res = greedy_rollout(
        agent, env, np.array([start_node], dtype=np.int64), np.array([0], dtype=np.int64),
        day["time_matrix_arr"], day["distance_arr"], day["ltr_stack"],
        day["reward_global_p95"], NUM_NODES,
        k_max_arr=[None], avail_expected_arr=day["avail_expected_arr"],
        ev_esperado=day["ev_esperado"], step_hook=hook,
    )
    if not res.valid[0]:
        return None, -np.inf, np.inf, arcos, tableros
    return res.routes[0], float(res.raw_reward[0]), float(res.duration[0]), arcos, tableros


def formatear_bloque(route, arcos) -> str:
    """Bloque de precio de reserva por arco (PRICING_PLAN.md §8), una línea por arco del tour."""
    lineas = ["Ruta: " + " → ".join(str(n) for n in route), "Precio de reserva por arco:"]
    for a in arcos:
        cab = f"  {a['from']}→{a['to']}: "
        if not a["has_load"]:
            nota_sin_carga = "sin carga en este regreso: sin p*" if a["is_return"] else "sin carga en esta lane (reposicionamiento): sin p*"
            lineas.append(cab + nota_sin_carga)
        elif a["only_option"]:
            lineas.append(cab + "p* = cost/dist (única acción factible)")
        else:
            marca = "  ← posición valiosa" if a["p_estrella"] < a["break_even"] else ""
            nota_retorno = "  (regreso a casa, Ṽ = 0)" if a["is_return"] else ""
            lineas.append(
                cab + f"p* = {a['p_estrella']:.2f} /mi  [rango {a['p_min']:.2f}–{a['p_max']:.2f}]  "
                f"(rate_hoy = {a['rate_hoy']:.2f}, break-even = {a['break_even']:.2f}, "
                f"percentil histórico {a['percentil_historico']:.0f} %)" + nota_retorno + marca
            )
    return "\n".join(lineas)


def formatear_tablero(i, tablero) -> str:
    """Tabla del p* de cada lane saliente con carga ofertada en el nodo i (la elegida por el DRL lleva *)."""
    lineas = [f"    Tablero en el nodo {i}:   lane    rate_hoy      p*      rango p*       break-even  percentil"]
    for f in sorted(tablero, key=lambda f: f["to"]):
        if f["only_option"] or not np.isfinite(f["p_estrella"]):
            p, rango, pct = "     --", "       --      ", "   --"
        else:
            p, rango, pct = f"{f['p_estrella']:7.2f}", f"[{f['p_min']:5.2f}–{f['p_max']:5.2f}]", f"{f['percentil_historico']:5.0f}"
        marca = "←" if np.isfinite(f["p_estrella"]) and f["p_estrella"] < f["break_even"] else " "
        lineas.append(
            f"      {'*' if f['elegida'] else ' '} {f['from']:>3}→{f['to']:<3}  {f['rate_hoy']:9.2f}  {p}  {rango}"
            f"  {f['break_even']:10.2f}  {pct} {marca}"
        )
    return "\n".join(lineas)


def run_inference(agent: AMRoutingAgent, start_node: int, day: dict, show_board: bool = False, filas_csv: list = None):
    """Genera la ruta del DRL (actor) desde start_node, muestra el resultado y el p* por arco."""
    import time

    print(f"\n  Generando ruta desde nodo {start_node}...")

    t0 = time.perf_counter()
    route, total_reward, time_elapsed, arcos, tableros = _rollout_con_pricing(agent, start_node, day, show_board)
    inference_ms = (time.perf_counter() - t0) * 1000

    print("\n" + "=" * 50)
    if route is None:
        print("  ⚠  No se encontró una ruta válida desde ese nodo.")
        print(f"     (Tiempo máximo permitido: {MAX_DURATION:.1f} h)")
    else:
        ruta_str = " → ".join(str(n) for n in route)
        print(f"  Ruta      : {ruta_str}   (DRL)")
        print(f"  Reward    : {total_reward:.0f}")
        print(f"  Duración  : {time_elapsed:.2f} h  (máx {MAX_DURATION:.1f} h)")
        print(f"  Nodos vis.: {len(route) - 1} de {NUM_NODES}")
        if show_board:
            for k, tab in enumerate(tableros):
                print(formatear_tablero(arcos[k]["from"], tab))
        print()
        print(formatear_bloque(route, arcos))
        if filas_csv is not None:
            ep = 1 + max((f["episode_id"] for f in filas_csv), default=-1)
            for k, a in enumerate(arcos):
                filas_csv.append(dict(
                    episode_id=ep, step=k, rate_hoy=a["rate_hoy"], p_estrella=a["p_estrella"], p_min=a["p_min"],
                    p_max=a["p_max"], break_even=a["break_even"], percentil_historico=a["percentil_historico"],
                    only_option=a["only_option"], is_return=a["is_return"], has_load=a["has_load"],
                    **{"from": a["from"], "to": a["to"]},
                ))
    print(f"  Inferencia: {inference_ms:.2f} ms")
    print("=" * 50)


# ── Main ──────────────────────────────────────────────────────────────────────

def _escribir_csv(filas, cwd):
    out = os.path.join(cwd, "results", f"{NUM_NODES}nodes" + (f"_{RUN_TAG}" if RUN_TAG else ""))
    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, f"pricing_{RUN_TAG or 'inference'}.csv")
    pd.DataFrame(filas, columns=CSV_COLUMNS).to_csv(path, index=False)
    print(f"  Precios de reserva → {path}")


def main():
    global NUM_NODES, CHECKPOINT_NAME
    ap = argparse.ArgumentParser(description="Inferencia del AM entrenado (ruta + precio de reserva por arco).")
    ap.add_argument("--show-board", action="store_true", help="tabla de p* por lane saliente en cada nodo")
    ap.add_argument("--start-node", type=int, help="un solo tour desde este nodo (sin loop interactivo)")
    ap.add_argument("--num-nodes", type=int, default=NUM_NODES)
    ap.add_argument("--checkpoint", help="ruta del .pt con las cabezas Ṽ (default: checkpoints[/RUN_TAG]/am_checkpoint_<N>nodes_pricing.pt)")
    args = ap.parse_args()
    NUM_NODES = args.num_nodes
    CHECKPOINT_NAME = f"am_checkpoint_{NUM_NODES}nodes_pricing.pt"

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
    if not checkpoint_tiene_post_head(checkpoint_path):
        raise SystemExit(f"{checkpoint_path} no trae las cabezas Ṽ A/B: corre post_training.py primero.")
    print("Cargando datos del día más reciente...")
    day = load_latest_day()
    print("Listo.\n")
    filas_csv = []

    if args.start_node is not None:
        run_inference(agent, args.start_node, day, args.show_board, filas_csv)
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

        run_inference(agent, start_node, day, args.show_board, filas_csv)
        if filas_csv:
            _escribir_csv(filas_csv, cwd)
        print()


if __name__ == "__main__":
    main()
