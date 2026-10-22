"""
greedy_rollout.py
=================
Evaluación del agente paso a paso en el entorno: en cada nodo se elige el
argmax de la política (sobre las acciones válidas de la máscara del entorno).

No hay búsqueda: una sola ruta por episodio. El mismo rollout sirve para
  · mundos simulados  (env con market_process, un mundo por `world_seeds[b]`),
  · replay de días reales (env con market_process=None, `start_day_idxs[b]`),
y para el baseline self-critical y el monitor de entrenamiento (am_training.py),
de modo que recompensa, máscara y disponibilidad de lanes son SIEMPRE las del
entorno que se usó para entrenar.
"""

from dataclasses import dataclass

import numpy as np
import torch

from config import MAX_DURATION, DEVICE
from attention_encoder import (
    build_node_features_batch,
    build_temporal_features_batch,
    build_market_features_batch,
    build_budget_features_batch,
    build_dow_features_batch,
)


@dataclass
class GreedyResult:
    """Resultado de un rollout greedy sobre B episodios.

    env_reward : (B,) recompensa del entorno (escalada por REWARD_SCALE_FACTOR, con la penalización
                 terminal por violar el tiempo y, solo con USE_REWARD_SHAPING, los bonos de retorno) — la que compara el self-critical con el rollout muestreado.
    raw_reward : (B,) suma de las recompensas de las lanes recorridas, en unidades
                 originales (las de los solvers). Solo significativa si valid[b].
    routes     : list[list[int]] — nodos visitados, desde start_node hasta donde terminó
                 el episodio.
    duration   : (B,) horas acumuladas al terminar.
    valid      : (B,) bool — el episodio cerró el ciclo en start_node dentro de max_duration.
    """
    env_reward: np.ndarray
    raw_reward: np.ndarray
    routes:     list
    duration:   np.ndarray
    valid:      np.ndarray


def greedy_rollout(
    agent,
    env,
    start_nodes:        np.ndarray,          # (B,) int64
    start_day_idxs:     np.ndarray,          # (B,) int64
    time_matrix_arr:    np.ndarray,
    distance_arr:       np.ndarray,
    trucks_stack:       np.ndarray,
    ltr_stack:          np.ndarray,
    reward_global_p95:  float,
    num_nodes:          int,
    k_max_arr:          list       = None,   # length-B list of (int | None); None ⇒ todos libres
    avail_expected_arr: np.ndarray = None,
    world_seeds:        np.ndarray = None,   # (B,) — solo con market_process; las MISMAS que el
                                             # rollout muestreado hacen que ambos vean el mismo mundo
    max_duration:       float      = MAX_DURATION,
    device:             torch.device = None,   # dispositivo del agente (default: DEVICE de config)
    ev_esperado:        np.ndarray = None,   # (7, N, N) calcular_ev_esperado (baselines.py); obligatorio con
                                             # USE_CONTINUATION_FEATURES (features de continuación f8–f11)
    step_hook:          callable   = None,   # fn(dict) invocada en cada paso ANTES de env.step, para inspeccionar la
                                             # decisión sin alterarla (p. ej. el precio de reserva en inference.py).
                                             # Claves: env, h_t (B, d_h), embeddings (B, N, d_h), masks (B, N),
                                             # actions (B,), active (B,) bool, cur_nodes (B,)
) -> GreedyResult:
    """Rollout greedy (argmax por nodo) vectorizado sobre un VectorRoutingEnv.

    `env` fija el mundo: con market_process las cargas y tarifas se sortean al
    llegar a cada nodo; sin él, se reproducen los días del histórico de su
    rm_pen_stack / loads_stack. Las features del agente salen de
    env.reward_rows (la recompensa de las lanes salientes ya observada), por lo
    que no hace falta pasar la matriz de recompensas.

    No samplea ni consume RNG global (greedy_action es argmax puro). Deja el
    agente en el modo (train/eval) en que estaba.
    """
    device = device or DEVICE
    B = len(start_nodes)
    start_nodes = np.asarray(start_nodes, dtype=np.int64)

    env_reward = np.zeros(B, dtype=np.float32)
    raw_reward = np.zeros(B, dtype=np.float64)
    valid      = np.zeros(B, dtype=bool)
    routes     = [[int(s)] for s in start_nodes]

    masks  = env.reset(start_nodes, start_day_idxs, k_max_arr=k_max_arr, world_seeds=world_seeds)
    active = np.ones(B, dtype=bool)

    was_training = agent.training
    agent.eval()
    try:
        with torch.no_grad():
            for _ in range(num_nodes):   # techo defensivo: cada paso visita un nodo nuevo o cierra
                if not active.any():
                    break

                cur_nodes = env.current_node
                day_idx   = env.current_day_idx

                node_feats = build_node_features_batch(
                    cur_nodes, start_nodes, env.visited_mask,
                    None, time_matrix_arr, distance_arr,
                    day_idx, num_nodes, max_duration,
                    trucks_stack_raw=trucks_stack,
                    reward_global_p95=reward_global_p95,
                    avail_expected_arr=avail_expected_arr,
                    reward_rows=env.reward_rows,
                    ev=ev_esperado, time_elapsed=env.time_elapsed, dow0=env.start_dow,
                )
                temporal = build_temporal_features_batch(
                    env.time_elapsed, env.step_count, max_duration, num_nodes,
                )
                market = build_market_features_batch(cur_nodes, day_idx, ltr_stack) \
                    if ltr_stack is not None else np.zeros((B, 1), dtype=np.float32)
                budget = build_budget_features_batch(env.k_remaining, env.k_restricted)
                dow    = build_dow_features_batch(env.current_dow)

                node_feats_t = torch.from_numpy(node_feats).to(device)
                temporal_t   = torch.from_numpy(temporal).to(device)
                market_t     = torch.from_numpy(market).to(device)
                budget_t     = torch.from_numpy(budget).to(device)
                dow_t        = torch.from_numpy(dow).to(device)
                cur_nodes_t  = torch.tensor(cur_nodes, dtype=torch.long, device=device)
                mask_t       = torch.from_numpy(masks).to(device)

                embeddings, graph_emb = agent.encoder(node_feats_t)
                current_emb = embeddings[torch.arange(B), cur_nodes_t, :]
                h_t = agent.context_net(graph_emb, current_emb, temporal_t, market_t, budget_t, dow_t)
                actions = agent.decoder.greedy_action(h_t, embeddings, mask_t).cpu().numpy()

                if step_hook is not None:
                    step_hook(dict(env=env, h_t=h_t, embeddings=embeddings, masks=masks, actions=actions,
                                   active=active.copy(), cur_nodes=cur_nodes.copy()))

                rewards, terminated, masks = env.step(actions)

                for b in np.where(active)[0]:
                    routes[b].append(int(actions[b]))
                env_reward += np.where(active, rewards, 0.0)
                raw_reward += np.where(active, env.last_raw_reward, 0.0)
                # Éxito: este paso cerró el ciclo en start_node sin violar el tiempo.
                valid |= active & terminated & (actions == start_nodes) & ~env.last_time_violation

                active = active & ~terminated
    finally:
        agent.train(was_training)

    return GreedyResult(
        env_reward=env_reward,
        raw_reward=raw_reward,
        routes=routes,
        duration=np.asarray(env.time_elapsed, dtype=np.float64),
        valid=valid,
    )
