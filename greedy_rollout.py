"""
greedy_rollout.py
=================
Evaluación del agente paso a paso en el entorno: en cada nodo se elige el argmax del destino y, para ese
destino, el argmax de la cabeza de precio (especificacion_fase0_ruteo_pricing.md §5), siempre sobre las
opciones válidas del entorno.

No hay búsqueda: una sola ruta por episodio. El mismo rollout sirve para mundos simulados y replay de días
reales (los dos son un `market_process.Mundos`), para el baseline self-critical y para la validación del
entrenamiento (am_training.py), de modo que recompensa, máscara y disponibilidad de lanes son SIEMPRE las del
entorno que se usó para entrenar.

Todo corre en el dispositivo del entorno (la GPU en el servidor): las features se arman con tensores
(`construir_features`) y nada vuelve a la CPU hasta el final.
"""

from dataclasses import dataclass

import numpy as np
import torch

from config import MAX_DURATION
from routing_env import OPCION_VACIO
from model.encoder import (
    COT_COLS,
    build_node_features_batch,
    build_temporal_features_batch,
    build_market_features_batch,
    build_budget_features_batch,
    build_dow_features_batch,
    build_cotizacion_node_features_batch,
    build_cotizacion_context_batch,
)


@dataclass
class EntradasAgente:
    """Lo fijo de las entradas del agente para un tamaño de grafo, en el dispositivo del entorno."""
    avail: torch.Tensor          # (N, N) E[loads] histórico (feature [4]) o None
    est: object                  # EstadisticasLane en el dispositivo (features de continuación)
    p95: float                   # REWARD_GLOBAL_P95 (normaliza los márgenes)
    max_duration: float = MAX_DURATION

    @staticmethod
    def crear(env, avail_expected_arr, est_lane, reward_global_p95, max_duration=MAX_DURATION) -> "EntradasAgente":
        dev = env.device
        avail = (None if avail_expected_arr is None
                 else torch.as_tensor(np.asarray(avail_expected_arr), dtype=torch.float32, device=dev))
        est = None if est_lane is None else est_lane.a(dev)
        return EntradasAgente(avail=avail, est=est, p95=float(reward_global_p95), max_duration=max_duration)


@dataclass
class GreedyResult:
    """Resultado de un rollout greedy sobre B episodios (tensores en el dispositivo del entorno).

    env_reward : (B,) recompensa del entorno (escalada por REWARD_SCALE_FACTOR, con la penalización
                 terminal por violar el tiempo) — la que compara el self-critical con el rollout muestreado.
    raw_reward : (B,) suma de las recompensas de las lanes recorridas, en USD. Solo significativa si valid[b].
    duration   : (B,) horas acumuladas al terminar.
    valid      : (B,) bool — el episodio cerró el ciclo en start_node dentro de max_duration, o no salió
                 de casa porque no tenía ninguna jugada (env.sin_salida, margen 0).
    n_cotizaciones, n_ganadas : (B,) cotizaciones hechas y cargas ganadas.
    suma_niveles : (B,) suma de los niveles cotizados (nivel medio = suma / n_cotizaciones).
    pasos      : con registrar=True, dict de tensores (P, B) por paso: actions, opciones, activo, cotizo, gano,
                 espero, viajo, raw (las rutas se arman en la CPU solo cuando se piden, ver `rutas`).
    """
    env_reward: torch.Tensor
    raw_reward: torch.Tensor
    duration:   torch.Tensor
    valid:      torch.Tensor
    n_cotizaciones: torch.Tensor
    n_ganadas:      torch.Tensor
    suma_niveles:   torch.Tensor
    start_nodes:    torch.Tensor
    pasos:          dict = None

    def rutas(self) -> list:
        """list[list[int]] — nodos visitados por episodio (solo los viajes: un rechazo o una espera no agregan nodo)."""
        if self.pasos is None:
            raise ValueError("greedy_rollout(..., registrar=True) para tener las rutas")
        a = self.pasos["actions"].cpu().numpy()
        v = self.pasos["viajo"].cpu().numpy()
        return [[int(s)] + [int(x) for x in a[v[:, b], b]] for b, s in enumerate(self.start_nodes.cpu().numpy())]


def construir_features(env, entradas: EntradasAgente) -> dict:
    """Entradas del agente (tensores en el dispositivo del entorno) para el estado actual de cada fila de `env`.

    Claves: node_feats (B, N, F) (sus últimas columnas, COT_COLS, entran también a la cabeza de precio), temporal (B, 3),
    market (B, 1), budget (B, 2), dow (B, 7), cot (B, N_COT_CONTEXT) y opciones (B, N, K+1) bool.
    """
    cur = env.current_node
    cot_nodo = build_cotizacion_node_features_batch(
        cur, env.precios_row, env.opciones_validas, env.dist_rows, env.cost_rows,
        env.competencia, env.intentos, entradas.p95,
    )
    node_feats = build_node_features_batch(
        cur, env.start_node, env.T, env.dist, entradas.max_duration, entradas.p95, entradas.avail,
        env.reward_rows, entradas.est, env.time_elapsed, env.start_dow, env.market_index, env.index_std,
        cotizacion_feats=cot_nodo, cargas_hoy=env.cargas_hoy, margen_hoy=env.margen_hoy,
    )
    return dict(
        node_feats=node_feats,
        temporal=build_temporal_features_batch(env.time_elapsed, env.step_count, entradas.max_duration, env.num_nodes),
        market=build_market_features_batch(env.ltr_actual),
        budget=build_budget_features_batch(env.k_remaining, env.k_restricted),
        dow=build_dow_features_batch(env.current_dow),
        cot=build_cotizacion_context_batch(env.competencia, env.n_rechazos),
        opciones=env.opciones_validas,
        current_node=cur,
    )


def contexto_agente(agent, feats: dict) -> tuple:
    """encoder + ContextNetwork: (h_t (B, d_h), embeddings (B, N, d_h))."""
    nf = feats["node_feats"]
    B = nf.shape[0]
    embeddings, graph_emb = agent.encoder(nf)
    current_emb = embeddings[torch.arange(B, device=nf.device), feats["current_node"], :]
    h_t = agent.context_net(graph_emb, current_emb, feats["temporal"], feats["market"], feats["budget"],
                            feats["dow"], feats["cot"])
    return h_t, embeddings


def greedy_rollout(
    agent,
    env,
    mundos,                               # market_process.Mundos de los B episodios
    start_nodes,                          # (B,) nodos de inicio
    entradas:   EntradasAgente,
    k_max                = None,          # lista de (int | None) o None ⇒ todos libres
    registrar:  bool     = False,         # True: guarda (P, B) por paso para armar rutas y detalle por arco
    step_hook:  callable = None,          # fn(dict) invocada en cada paso ANTES de env.step, para inspeccionar la
                                          # decisión sin alterarla. Claves: env, filas (índices de las filas
                                          # activas) y, solo de esas filas, h_t, embeddings, masks, actions,
                                          # opciones, feats
) -> GreedyResult:
    """Rollout greedy (argmax por nodo) vectorizado sobre un VectorRoutingEnv, en su dispositivo.

    No samplea ni consume RNG (greedy_cotizacion es argmax puro). Deja el agente en el modo (train/eval) en que
    estaba.
    """
    dev = env.device
    masks = env.reset(mundos, start_nodes, k_max)
    B = env.B
    active = env.active.clone()                    # sin ninguna jugada desde casa: no sale (margen 0, válido)
    env_reward = torch.zeros(B, dtype=torch.float32, device=dev)
    raw_reward = torch.zeros(B, dtype=torch.float64, device=dev)
    valid = torch.zeros(B, dtype=torch.bool, device=dev)
    n_cot = torch.zeros(B, dtype=torch.long, device=dev)
    n_gan = torch.zeros(B, dtype=torch.long, device=dev)
    suma_niv = torch.zeros(B, dtype=torch.long, device=dev)
    log = {k: [] for k in ("actions", "opciones", "activo", "cotizo", "gano", "espero", "viajo", "raw")} if registrar else None

    was_training = agent.training
    agent.eval()
    try:
        with torch.no_grad():
            for _ in range(env.max_steps):   # techo defensivo (ver max_pasos_cotizacion)
                idx = active.nonzero().squeeze(1)
                if idx.numel() == 0:
                    break
                # Solo las filas activas pasan por la red; las que terminaron reciben una acción ficticia válida
                # (nodo 0, VACÍO) que el entorno ignora.
                feats = {k: x[idx] for k, x in construir_features(env, entradas).items()}
                h_t, embeddings = contexto_agente(agent, feats)
                a_act, o_act = agent.decoder.greedy_cotizacion(
                    h_t, embeddings, masks[idx], feats["opciones"], feats["node_feats"][..., COT_COLS])
                actions = torch.zeros(B, dtype=torch.long, device=dev)
                opciones = torch.full((B,), OPCION_VACIO, dtype=torch.long, device=dev)
                actions[idx] = a_act
                opciones[idx] = o_act
                if step_hook is not None:
                    step_hook(dict(env=env, filas=idx, h_t=h_t, embeddings=embeddings, masks=masks[idx],
                                   actions=a_act, opciones=o_act, feats=feats))

                rewards, terminated, masks = env.step(actions, opciones)
                cotizo = active & env.last_cotizo
                gano = cotizo & env.last_gano
                viajo = active & ~(env.last_cotizo & ~env.last_gano) & ~env.last_espero
                n_cot += cotizo.long()
                n_gan += gano.long()
                suma_niv += torch.where(cotizo, opciones, torch.zeros_like(opciones))
                env_reward += torch.where(active, rewards, torch.zeros_like(rewards))
                raw_reward += torch.where(active, env.last_raw_reward.double(), torch.zeros_like(raw_reward))
                # Éxito: este paso viajó a start_node y cerró el ciclo sin violar el tiempo.
                valid |= viajo & terminated & (actions == env.start_node) & ~env.last_time_violation
                if registrar:
                    for k, x in (("actions", actions), ("opciones", opciones), ("activo", active), ("cotizo", cotizo),
                                 ("gano", gano), ("espero", active & env.last_espero), ("viajo", viajo),
                                 ("raw", torch.where(active, env.last_raw_reward, torch.zeros_like(rewards)))):
                        log[k].append(x)
                active = active & ~terminated
    finally:
        agent.train(was_training)
    valid |= env.sin_salida                         # no salió de casa: ciclo válido con margen 0

    return GreedyResult(
        env_reward=env_reward, raw_reward=raw_reward, duration=env.time_elapsed.double(), valid=valid,
        n_cotizaciones=n_cot, n_ganadas=n_gan, suma_niveles=suma_niv, start_nodes=env.start_node,
        pasos=None if log is None else {k: torch.stack(v) if v else torch.zeros((0, B), device=dev)
                                         for k, v in log.items()},
    )
