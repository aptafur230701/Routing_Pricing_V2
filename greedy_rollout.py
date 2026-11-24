"""
greedy_rollout.py
=================
Evaluación del agente paso a paso en el entorno: en cada nodo se elige el argmax del destino y, para ese
destino, el argmax de la cabeza de precio (especificacion_fase0_ruteo_pricing.md §5), siempre sobre las
opciones válidas del entorno.

No hay búsqueda: una sola ruta por episodio. El mismo rollout sirve para
  · mundos simulados  (env con market_process, un mundo por `world_seeds[b]`),
  · replay de días reales (env con mercado_replay, `start_day_idxs[b]`),
y para el baseline self-critical y el monitor de entrenamiento (am_training.py),
de modo que recompensa, máscara y disponibilidad de lanes son SIEMPRE las del
entorno que se usó para entrenar.

`construir_features` arma las entradas del agente desde el estado del entorno y la usan también los
rollouts muestreados del entrenamiento.
"""

from dataclasses import dataclass

import numpy as np
import torch

from config import MAX_DURATION, DEVICE
from model.encoder import (
    build_node_features_batch,
    build_temporal_features_batch,
    build_market_features_batch,
    build_budget_features_batch,
    build_dow_features_batch,
    build_cotizacion_node_features_batch,
    build_cotizacion_context_batch,
)


@dataclass
class GreedyResult:
    """Resultado de un rollout greedy sobre B episodios.

    env_reward : (B,) recompensa del entorno (escalada por REWARD_SCALE_FACTOR, con la penalización
                 terminal por violar el tiempo y, solo con USE_REWARD_SHAPING, los bonos de retorno) — la que compara el self-critical con el rollout muestreado.
    raw_reward : (B,) suma de las recompensas de las lanes recorridas, en unidades
                 originales (las de los solvers). Solo significativa si valid[b].
    routes     : list[list[int]] — nodos visitados, desde start_node hasta donde terminó
                 el episodio (solo los viajes: un rechazo no agrega nodo).
    duration   : (B,) horas acumuladas al terminar.
    valid      : (B,) bool — el episodio cerró el ciclo en start_node dentro de max_duration.
    n_cotizaciones, n_ganadas : (B,) cotizaciones hechas y cargas ganadas.
    niveles    : list[list[int]] — nivel cotizado en cada cotización.
    """
    env_reward: np.ndarray
    raw_reward: np.ndarray
    routes:     list
    duration:   np.ndarray
    valid:      np.ndarray
    n_cotizaciones: np.ndarray
    n_ganadas:      np.ndarray
    niveles:        list


def construir_features(
    env, start_nodes, time_matrix_arr, distance_arr, ltr_stack, reward_global_p95, num_nodes,
    avail_expected_arr=None, ev_esperado=None, max_duration=MAX_DURATION,
) -> dict:
    """Entradas del agente (numpy) para el estado actual de cada fila de `env`.

    Claves: node_feats (B, N, F), temporal (B, 3), market (B, 1), budget (B, 2), dow (B, 7),
    cot (B, N_COT_CONTEXT) y opciones (B, N, K+1) bool.
    """
    B = len(start_nodes)
    cur_nodes = env.current_node
    day_idx = env.current_day_idx
    cot_nodo = build_cotizacion_node_features_batch(
        cur_nodes, env.precios_row, env.opciones_validas, env.dist_rows, env.cost_rows,
    )
    node_feats = build_node_features_batch(
        cur_nodes, start_nodes,
        time_matrix_arr, distance_arr,
        day_idx, num_nodes, max_duration,
        reward_global_p95=reward_global_p95,
        avail_expected_arr=avail_expected_arr,
        reward_rows=env.reward_rows,
        ev=ev_esperado, time_elapsed=env.time_elapsed, dow0=env.start_dow,
        market_index=env.market_index, index_std=env.index_std,
        cotizacion_feats=cot_nodo,
    )
    return dict(
        node_feats=node_feats,
        temporal=build_temporal_features_batch(env.time_elapsed, env.step_count, max_duration, num_nodes),
        market=build_market_features_batch(cur_nodes, day_idx, ltr_stack) if ltr_stack is not None
        else np.zeros((B, 1), dtype=np.float32),
        budget=build_budget_features_batch(env.k_remaining, env.k_restricted),
        dow=build_dow_features_batch(env.current_dow),
        cot=build_cotizacion_context_batch(env.competencia, env.n_rechazos),
        opciones=env.opciones_validas.copy(),
    )


def contexto_agente(agent, feats: dict, cur_nodes, device) -> tuple:
    """encoder + ContextNetwork: (h_t (B, d_h), embeddings (B, N, d_h))."""
    t = {k: torch.from_numpy(v).to(device) for k, v in feats.items() if k != "opciones"}
    B = t["node_feats"].shape[0]
    embeddings, graph_emb = agent.encoder(t["node_feats"])
    cur_t = torch.as_tensor(np.asarray(cur_nodes), dtype=torch.long, device=device)
    current_emb = embeddings[torch.arange(B, device=device), cur_t, :]
    h_t = agent.context_net(graph_emb, current_emb, t["temporal"], t["market"], t["budget"], t["dow"], t["cot"])
    return h_t, embeddings


def greedy_rollout(
    agent,
    env,
    start_nodes:        np.ndarray,          # (B,) int64
    start_day_idxs:     np.ndarray,          # (B,) int64
    time_matrix_arr:    np.ndarray,
    distance_arr:       np.ndarray,
    ltr_stack:          np.ndarray,
    reward_global_p95:  float,
    num_nodes:          int,
    k_max_arr:          list       = None,   # length-B list of (int | None); None ⇒ todos libres
    avail_expected_arr: np.ndarray = None,
    world_seeds:        np.ndarray = None,   # (B,) — solo con market_process; las MISMAS que el
                                             # rollout muestreado hacen que ambos vean el mismo mundo
    max_duration:       float      = MAX_DURATION,
    device:             torch.device = None,   # dispositivo del agente (default: DEVICE de config)
    ev_esperado:        np.ndarray = None,   # EvEsperado, calcular_ev_pronostico (baselines/politicas.py); obligatorio con
                                             # USE_CONTINUATION_FEATURES (features de continuación f7–f10)
    step_hook:          callable   = None,   # fn(dict) invocada en cada paso ANTES de env.step, para inspeccionar la
                                             # decisión sin alterarla (p. ej. la ruta por arco en inference.py).
                                             # Claves: env, h_t (B, d_h), embeddings (B, N, d_h), masks (B, N),
                                             # actions (B,), opciones (B,), active (B,) bool, cur_nodes (B,)
) -> GreedyResult:
    """Rollout greedy (argmax por nodo) vectorizado sobre un VectorRoutingEnv.

    `env` fija el mundo (mundos simulados o replay, ver routing_env.py) y las semillas fijan los tableros
    y el competidor de cada episodio.

    No samplea ni consume RNG global (greedy_cotizacion es argmax puro). Deja el
    agente en el modo (train/eval) en que estaba.
    """
    device = device or DEVICE
    B = len(start_nodes)
    start_nodes = np.asarray(start_nodes, dtype=np.int64)

    env_reward = np.zeros(B, dtype=np.float32)
    raw_reward = np.zeros(B, dtype=np.float64)
    valid      = np.zeros(B, dtype=bool)
    routes     = [[int(s)] for s in start_nodes]
    n_cot = np.zeros(B, dtype=np.int64)
    n_gan = np.zeros(B, dtype=np.int64)
    niveles = [[] for _ in range(B)]

    masks  = env.reset(start_nodes, start_day_idxs, k_max_arr=k_max_arr, world_seeds=world_seeds)
    active = np.ones(B, dtype=bool)
    techo = env.max_steps

    was_training = agent.training
    agent.eval()
    try:
        with torch.no_grad():
            for _ in range(techo):   # techo defensivo (ver max_pasos_cotizacion)
                if not active.any():
                    break

                cur_nodes = env.current_node.copy()
                feats = construir_features(
                    env, start_nodes, time_matrix_arr, distance_arr, ltr_stack, reward_global_p95, num_nodes,
                    avail_expected_arr, ev_esperado, max_duration,
                )
                h_t, embeddings = contexto_agente(agent, feats, cur_nodes, device)
                mask_t = torch.from_numpy(masks).to(device)
                opciones_t = torch.from_numpy(feats["opciones"]).to(device)
                a_t, o_t = agent.decoder.greedy_cotizacion(h_t, embeddings, mask_t, opciones_t)
                actions, opciones = a_t.cpu().numpy(), o_t.cpu().numpy()

                if step_hook is not None:
                    step_hook(dict(env=env, h_t=h_t, embeddings=embeddings, masks=masks, actions=actions,
                                   opciones=opciones, active=active.copy(), cur_nodes=cur_nodes))

                rewards, terminated, masks = env.step(actions, opciones)
                cotizo = active & env.last_cotizo
                viajo = active & ~(env.last_cotizo & ~env.last_gano)
                n_cot += cotizo
                n_gan += cotizo & env.last_gano
                for b in np.flatnonzero(cotizo):
                    niveles[b].append(int(opciones[b]))

                for b in np.flatnonzero(viajo):
                    routes[b].append(int(actions[b]))
                env_reward += np.where(active, rewards, 0.0)
                raw_reward += np.where(active, env.last_raw_reward, 0.0)
                # Éxito: este paso viajó a start_node y cerró el ciclo sin violar el tiempo.
                valid |= viajo & terminated & (actions == start_nodes) & ~env.last_time_violation

                active = active & ~terminated
    finally:
        agent.train(was_training)

    return GreedyResult(
        env_reward=env_reward,
        raw_reward=raw_reward,
        routes=routes,
        duration=np.asarray(env.time_elapsed, dtype=np.float64),
        valid=valid,
        n_cotizaciones=n_cot,
        n_ganadas=n_gan,
        niveles=niveles,
    )
