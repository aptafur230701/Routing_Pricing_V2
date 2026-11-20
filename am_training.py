"""
am_training.py
==============
Loop de entrenamiento PPO para AMRoutingAgent + CriticHead.

Algoritmo: Proximal Policy Optimization (PPO-Clip)
  1. Recolectar N episodios con la política actual (sin gradientes).
  2. Calcular ventajas con GAE (Generalized Advantage Estimation).
  3. Actualizar actor + critic durante K épocas con mini-batches.
  4. Repetir hasta completar el número de updates.

Interfaz análoga a run_training() de training.py:

    agent, critic, episode_rewards, episode_losses = run_am_training(
        time_matrix, rate_stack, loads_stack, distance_arr, diesel_arr,
        num_nodes
    )

· Usa RoutingEnv como entorno estándar (Siguiendo practicas de API Gymnasium).
"""

import math
import time
from collections import defaultdict, deque
from contextlib import contextmanager

import numpy as np
import torch
import torch.nn as nn

from config import (
    MARKET_SEED_VAL, VAL_CHECKS, get_val_worlds,
    MAX_DURATION,
    DEVICE,
    SEED,
    TRAIN_DAYS,
    N_EVAL_EPISODES,
    get_episodes_per_node,
    AM_D_H, AM_N_HEADS, AM_N_LAYERS, AM_D_FF, AM_PRE_NORM, AM_DROPOUT,
    PPO_N_EPISODES_PER_UPDATE, PPO_N_EPOCHS, PPO_BATCH_SIZE,
    PPO_LR, PPO_GAMMA, PPO_GAE_LAMBDA, PPO_CLIP_EPS,
    PPO_ENTROPY_COEF_START, PPO_ENTROPY_COEF_END, PPO_GRAD_CLIP,
    PPO_WEIGHT_DECAY,
    USE_SELF_CRITICAL, SELF_CRITICAL_COEF, SELF_CRITICAL_WARMUP_UPDATES,
    K_TRAINING_LEVELS,
    MARKET_SEED_TRAIN,
    USE_CONTINUATION_FEATURES,
    CKPT_SELECTION, CKPT_WINDOW, USE_LR_COSINE, LR_FINAL_FRAC, hp,
)
from routing_env import VectorRoutingEnv, max_pasos
from greedy_rollout import greedy_rollout
from problem_data import build_rm_pen_stack
from model.encoder import (
    build_node_features_batch,
    build_temporal_features_batch,
    build_market_features_batch,
    build_budget_features_batch,
    build_dow_features_batch,
    N_DOW_FEATURES,
)
from model.agent import AMRoutingAgent
from model.critic_head import CriticHead
from baselines.politicas import calcular_ev_pronostico, PoliticaB3
from model.debug_utils import check_tensor
from baselines.dp_relajada import cargar_o_calcular_tabla
from evaluation import disenar_mundos, ejecutar_politica
from market_process import MarketProcess, asegurar_parametros


# ─────────────────────────────────────────────────────────────────────────────
# Muestreo balanceado de niveles de K
# ─────────────────────────────────────────────────────────────────────────────

def _balanced_k_assignment(n: int, rng) -> list:
    """
    Reparte K_TRAINING_LEVELS (K_MAX restringido + caso libre None) de
    forma balanceada sobre n episodios y permuta. Mismo patrón que el
    balanceo ya existente de start_nodes_arr, pero independiente (cruce
    aproximadamente balanceado).

    rng : np.random.Generator | np.random.RandomState — debe exponer
          .permutation(n) y (si hace falta relleno) .choice/.randint.
    """
    n_levels  = len(K_TRAINING_LEVELS)
    per_level = n // n_levels
    remainder = n % n_levels

    k_arr = []
    for lvl in K_TRAINING_LEVELS:
        k_arr += [lvl] * per_level
    if remainder:
        if hasattr(rng, "integers"):
            extra_idx = rng.integers(0, n_levels, size=remainder)
        else:
            extra_idx = rng.randint(0, n_levels, size=remainder)
        k_arr += [K_TRAINING_LEVELS[i] for i in extra_idx]

    perm = rng.permutation(n)
    return [k_arr[i] for i in perm]


# ─────────────────────────────────────────────────────────────────────────────
# Buffer de rollout para PPO
# ─────────────────────────────────────────────────────────────────────────────

class RolloutBuffer:
    """
    Almacena transiciones de un ciclo de recolección PPO.

    Cada transición guarda las entradas crudas (numpy) necesarias para
    reconstruir el forward pass durante la fase de actualización.
    """

    def __init__(self):
        self.node_feats    = []   # list of np.ndarray (N, N_NODE_FEATURES)
        self.temporal_feats = []  # list of np.ndarray (3,)
        self.market_feats  = []   # list of np.ndarray (1,)
        self.budget_feats  = []   # list of np.ndarray (2,)
        self.dow_feats     = []   # list of np.ndarray (7,)
        self.current_nodes = []   # list of int
        self.masks         = []   # list of np.ndarray (N,) int8
        self.actions       = []   # list of int
        self.rewards       = []   # list of float
        self.old_log_probs = []   # list of float
        self.old_values    = []   # list of float
        self.dones         = []   # list of bool
        # (start_idx, end_idx) por episodio
        self.episode_boundaries = []
        # Calculados por compute_gae_per_episode()
        self.returns       = None
        self.advantages    = None
        # Self-critical — poblados solo si USE_SELF_CRITICAL
        self.sc_advantage  = None   # (n_episodes,) ventaja normalizada por episodio
        self.episode_idx   = None   # (n_transitions,) índice de episodio por transición

    def add(
        self,
        node_feats:    np.ndarray,
        temporal:      np.ndarray,
        current_node:  int,
        mask:          np.ndarray,
        action:        int,
        reward:        float,
        log_prob:      float,
        value:         float,
        done:          bool,
        market_feats:  np.ndarray = None,   # (1,)
        budget_feats:  np.ndarray = None,   # (2,)
        dow_feats:     np.ndarray = None,   # (7,)
    ):
        self.node_feats.append(node_feats)
        self.temporal_feats.append(temporal)
        self.market_feats.append(market_feats if market_feats is not None
                                 else np.zeros(1, dtype=np.float32))
        self.budget_feats.append(budget_feats if budget_feats is not None
                                 else np.array([1.0, 0.0], dtype=np.float32))
        self.dow_feats.append(dow_feats if dow_feats is not None
                              else np.zeros(N_DOW_FEATURES, dtype=np.float32))
        self.current_nodes.append(current_node)
        self.masks.append(mask)
        self.actions.append(action)
        self.rewards.append(reward)
        self.old_log_probs.append(log_prob)
        self.old_values.append(value)
        self.dones.append(done)

    def __len__(self):
        return len(self.actions)

    def compute_gae_per_episode(
        self,
        gamma:      float = 0.99,
        gae_lambda: float = 0.95,
    ):
        """
        Calcula retornos y ventajas con GAE sobre cada episodio registrado en
        episode_boundaries.

        Cada episodio se procesa en forma independiente. Todos terminan
        naturalmente (retorno al depot, violación de tiempo o máscara vacía),
        así que el valor de bootstrap tras el último paso es 0.
        No se usa la máscara dones dentro del segmento porque el loop de
        recolección rompe en done=True, garantizando que ningún paso
        intermedio tenga done=True.
        """
        n = len(self.rewards)
        returns    = np.zeros(n, dtype=np.float32)
        advantages = np.zeros(n, dtype=np.float32)

        for start_idx, end_idx in self.episode_boundaries:
            seg_len   = end_idx - start_idx
            seg_values = np.array(
                self.old_values[start_idx:end_idx] + [0.0],
                dtype=np.float32,
            )

            gae = 0.0
            for t in reversed(range(seg_len)):
                delta = (
                    self.rewards[start_idx + t]
                    + gamma * seg_values[t + 1]
                    - seg_values[t]
                )
                gae                        = delta + gamma * gae_lambda * gae
                advantages[start_idx + t]  = gae
                returns[start_idx + t]     = gae + seg_values[t]

        self.returns    = returns
        self.advantages = advantages

    def get_batches(self, batch_size: int):
        """
        Genera índices de mini-batches aleatorios sobre el buffer completo.
        Si el buffer es menor que batch_size devuelve un solo batch con todo.
        """
        n = len(self)
        indices = np.random.permutation(n)
        for start in range(0, n, batch_size):
            yield indices[start: start + batch_size]


# ─────────────────────────────────────────────────────────────────────────────
# Forward pass compartido (recolección y actualización)
# ─────────────────────────────────────────────────────────────────────────────

def _forward(
    agent:          AMRoutingAgent,
    critic:         CriticHead,
    node_feats_t:   torch.Tensor,    # (B, N, N_NODE_FEATURES)
    temporal_t:     torch.Tensor,    # (B, 3)
    current_nodes:  torch.Tensor,    # (B,) int64
    mask_t:         torch.Tensor,    # (B, N) int8
    actions_t:      torch.Tensor = None,   # (B,) int64 — None en recolección
    market_feats_t: torch.Tensor = None,   # (B, 1)
    budget_feats_t: torch.Tensor = None,   # (B, 2)
    dow_feats_t:    torch.Tensor = None,   # (B, 7)
):
    """
    Ejecuta encoder → context → decoder → critic para un batch.

    Si actions_t es None (recolección): samplea acciones nuevas.
    Si actions_t no es None (actualización PPO): evalúa acciones almacenadas.

    Retorna
    -------
    actions   : Tensor (B,) int64
    log_probs : Tensor (B,)
    entropies : Tensor (B,)
    values    : Tensor (B,)
    """
    B = node_feats_t.shape[0]

    check_tensor("node_feats_t", node_feats_t)
    check_tensor("temporal_t", temporal_t)

    embeddings, graph_emb = agent.encoder(node_feats_t)           # (B,N,d_h), (B,d_h)

    check_tensor("embeddings", embeddings)
    check_tensor("graph_emb", graph_emb)

    current_emb = embeddings[torch.arange(B), current_nodes, :]   # (B, d_h)
    check_tensor("current_emb", current_emb)

    h_t = agent.context_net(graph_emb, current_emb, temporal_t, market_feats_t, budget_feats_t, dow_feats_t)   # (B, d_h)
    check_tensor("context h_t", h_t)

    values = critic(h_t)                                           # (B,)
    check_tensor("values", values)

    # Validate mask: every sample must have at least one valid action
    valid_actions = mask_t.sum(dim=-1)
    if (valid_actions == 0).any():
        raise RuntimeError(
            f"Invalid mask: sample(s) with zero valid actions detected "
            f"(valid counts per sample: {valid_actions.tolist()})"
        )

    if actions_t is None:
        actions, log_probs, entropies = agent.decoder.act(h_t, embeddings, mask_t)
    else:
        log_probs, entropies = agent.decoder.evaluate_action(
            h_t, embeddings, mask_t, actions_t
        )
        actions = actions_t

    check_tensor("log_probs", log_probs)

    return actions, log_probs, entropies, values


# ─────────────────────────────────────────────────────────────────────────────
# Evaluación greedy periódica (observacional, no afecta gradientes)
# ─────────────────────────────────────────────────────────────────────────────

class _Fases:
    """Cronómetro por fase del update. `sync=True` espera a la GPU antes de leer el reloj
    (los kernels son asíncronos); las fases que ya terminan en .cpu()/.item() no lo necesitan."""

    def __init__(self):
        self.t = defaultdict(float)

    @contextmanager
    def __call__(self, nombre: str, sync: bool = False):
        t0 = time.perf_counter()
        try:
            yield
        finally:
            if sync and DEVICE.type == "cuda":
                torch.cuda.synchronize()
            self.t[nombre] += time.perf_counter() - t0


_FASES_ORDEN = (
    "rollout_features", "rollout_gpu", "rollout_env", "rollout_buffer", "greedy",
    "aplanar_gae", "ppo_armado", "ppo_calculo", "validacion",
)


def _evaluate_greedy(
    agent,
    eval_env:         VectorRoutingEnv,
    eval_start_nodes: np.ndarray,
    eval_start_days:  np.ndarray,
    time_matrix_arr:  np.ndarray,
    distance_arr:     np.ndarray,
    ltr_stack:        np.ndarray,
    reward_global_p95: float,
    num_nodes:        int,
    eval_k_levels:    list = None,   # niveles de K por episodio (None ⇒ todos libres)
    avail_expected_arr: np.ndarray = None,
    ev_esperado:      np.ndarray = None,   # EvEsperado (calcular_ev_pronostico) para las features de continuación
) -> tuple:
    """
    Recompensa promedio en modo greedy (argmax en cada nodo, sin búsqueda) sobre
    un set fijo de episodios (mismos nodos de inicio, días y niveles de K entre
    llamadas), para monitorear el desempeño en el modo que de verdad se usa en
    producción — distinto del avg_reward de los rollouts de entrenamiento, que
    samplean de la política.

    eval_env es un VectorRoutingEnv de replay (market_process=None) sobre los días
    de entrenamiento: el mismo entorno y la misma recompensa que el rollout
    muestreado, sin ruido de mundo simulado.

    Los episodios que no cierran el ciclo en el depot dentro de max_duration se
    excluyen del promedio de reward (para no contaminarlo) y se cuentan aparte
    en valid_rate. El reward está en unidades originales (suma de lanes).

    Retorna
    -------
    avg_reward : float — promedio solo sobre episodios con ruta válida
                 (0.0 si ninguno fue válido).
    valid_rate : float — fracción de episodios con ruta válida.
    """
    res = greedy_rollout(
        agent, eval_env,
        np.asarray(eval_start_nodes, dtype=np.int64),
        np.asarray(eval_start_days,  dtype=np.int64),
        time_matrix_arr, distance_arr, ltr_stack,
        reward_global_p95, num_nodes,
        k_max_arr=eval_k_levels,
        avail_expected_arr=avail_expected_arr,
        ev_esperado=ev_esperado,
    )
    avg_reward = float(res.raw_reward[res.valid].mean()) if res.valid.any() else 0.0
    valid_rate = float(res.valid.mean())
    return avg_reward, valid_rate


class ValidacionSimulada:
    """Validación del greedy en episodios fijos, contra B3 y UB-DP.

    Un conjunto fijo de episodios (nodo, día de la semana y semilla balanceados, con la semilla
    propia MARKET_SEED_VAL) se evalúa por cada nivel de K de K_TRAINING_LEVELS en una sola
    llamada batcheada de greedy_rollout. Reporta el margen en USD (suma de lane_reward, la
    métrica de la evaluación final) y su gap contra UB-DP; B3 y UB-DP se calculan una sola vez.
    Es observacional: no afecta los gradientes ni el RNG global.
    """

    def __init__(self, market_process, time_matrix, rate_stack, loads_stack, distance_arr, diesel_arr,
                 num_nodes, time_matrix_arr, ltr_stack, reward_global_p95, avail_expected_arr,
                 ev_esperado=None):
        M = get_val_worlds(num_nodes)
        dis = disenar_mundos(num_nodes, M, market_process, num_dias=rate_stack.shape[0],
                             semilla=MARKET_SEED_VAL)
        self.niveles = list(K_TRAINING_LEVELS)
        self.M = M
        self.nodos = np.tile(dis["nodo"], len(self.niveles))
        self.dias = np.tile(dis["start_day"], len(self.niveles))
        self.semillas = np.tile(dis["world_seed"], len(self.niveles))
        self.k_arr = [K for K in self.niveles for _ in range(M)]
        self.args = (time_matrix_arr, distance_arr, ltr_stack, reward_global_p95, num_nodes)
        self.avail = avail_expected_arr
        self.ev = ev_esperado
        self.env = VectorRoutingEnv(
            time_matrix=time_matrix, rate_stack=rate_stack, loads_stack=loads_stack, rm_pen_stack=None,
            num_nodes=num_nodes, max_duration=MAX_DURATION, market_process=market_process,
            distance_arr=distance_arr, diesel_arr=diesel_arr,
        )
        # Referencias: B3 (política de la tabla V) y UB-DP, promedio sobre los mismos episodios.
        tabla = cargar_o_calcular_tabla(market_process, time_matrix, distance_arr, diesel_arr, verbose=False)
        b3 = PoliticaB3(tabla, distance_arr, diesel_arr)
        self.ref = {}
        self.b3_mundos = {}   # K → (M,) márgenes de B3 episodio a episodio (para la diferencia pareada)
        for K in self.niveles:
            margenes, ubs = [], []
            for m in range(M):
                h, dow = int(dis["nodo"][m]), int(dis["dow"][m])
                r = ejecutar_politica(b3, self.env, h, int(dis["start_day"][m]), K, int(dis["world_seed"][m]),
                                      distance_arr, diesel_arr, dow)
                margenes.append(r["margen"])
                ubs.append(tabla.ub_dp(h, dow, K))
            self.b3_mundos[K] = np.asarray(margenes, dtype=float)
            self.ref[K] = {"b3": float(np.mean(margenes)), "ub_dp": float(np.mean(ubs))}

    def evaluar(self, agent) -> dict:
        """Margen medio del greedy por nivel de K y global, validez y gap contra UB-DP."""
        res = greedy_rollout(
            agent, self.env, self.nodos, self.dias, *self.args,
            k_max_arr=self.k_arr, avail_expected_arr=self.avail, world_seeds=self.semillas,
            ev_esperado=self.ev,
        )
        margen = np.where(res.valid, res.raw_reward, 0.0)
        out = {"por_k": {}}
        for i, K in enumerate(self.niveles):
            sl = slice(i * self.M, (i + 1) * self.M)
            v = res.valid[sl]
            m = float(res.raw_reward[sl][v].mean()) if v.any() else 0.0
            ub = self.ref[K]["ub_dp"]
            dif_b3 = float(np.mean(res.raw_reward[sl] - self.b3_mundos[K]))   # media de (DRL − B3) episodio a episodio
            out["por_k"]["free" if K is None else str(K)] = {
                "margen": m, "valid_rate": float(v.mean()), "dif_b3": dif_b3,
                "gap_ub_dp_pct": float((ub - m) / abs(ub) * 100.0) if ub else float("nan"),
                "b3": self.ref[K]["b3"], "ub_dp": ub,
            }
        out["margen"] = float(res.raw_reward[res.valid].mean()) if res.valid.any() else 0.0
        out["valid_rate"] = float(res.valid.mean())
        out["score"] = float(margen.mean())    # inválidos cuentan 0: elige el mejor update
        return out


# ─────────────────────────────────────────────────────────────────────────────
# Función principal de entrenamiento
# ─────────────────────────────────────────────────────────────────────────────

def run_am_training(
    time_matrix,
    rate_stack:     np.ndarray,
    loads_stack:    np.ndarray,
    distance_arr:   np.ndarray,
    diesel_arr:     np.ndarray,
    num_nodes:      int,
    pretrained_agent:  AMRoutingAgent = None,
    pretrained_critic: CriticHead     = None,
    ltr_stack:         np.ndarray = None,   # [num_nodes, num_days]
    reward_global_p95: float      = 1.0,
    avail_expected_arr: np.ndarray = None,  # [num_nodes, num_nodes]
    market_process = None,   # MarketProcess opcional (proceso exógeno de cargas y tarifas). None = días
                             # reales del histórico (comportamiento de siempre).
    best_checkpoint_path: str = None,  # con market_process: ruta donde se guarda el update con mejor
                                       # margen de validación (agent + critic); None = no se guarda.
    trial_callback: callable = None,  # HPO opcional: fn(step, n_updates, log_entry) -> bool
                                       # (True = detener el entrenamiento ya, p.ej. Optuna pruning).
                                       # Solo se invoca en los updates de log_freq (donde greedy_reward
                                       # está disponible). No acopla este módulo a Optuna.
    ev_esperado: np.ndarray = None,   # EvEsperado, calcular_ev_pronostico. Para las features de continuación;
                                       # si es None se calcula una vez desde market_process (obligatorio
                                       # en ese caso con USE_CONTINUATION_FEATURES).
) -> tuple:
    """
    Entrena AMRoutingAgent + CriticHead con PPO.

    Parámetros
    ----------
    time_matrix    : np.ndarray — tiempos entre nodos.
    rate_stack     : np.ndarray [num_days, N, N]
    loads_stack    : np.ndarray [num_days, N, N]
    distance_arr   : np.ndarray [N, N]
    diesel_arr     : np.ndarray [N, N]
    num_nodes      : int
    ltr_stack      : np.ndarray [num_nodes, num_days] — LTR por hub y día
    avail_expected_arr : np.ndarray [num_nodes, num_nodes] — E[loads] histórico
                     (train), usado solo como feature de entrada (avail_prior)
    market_process : MarketProcess | None — si se pasa, en cada llegada a un nodo las cargas y
                     tarifas se sortean del proceso exógeno (cada episodio es un mundo con semilla
                     propia, generada desde MARKET_SEED_TRAIN) en vez de leerse del histórico; el
                     día de inicio solo aporta el día de la semana. No se usa rm_pen_stack.
    best_checkpoint_path : con market_process, guarda {"agent", "critic"} del update con mejor
                     margen en los episodios fijos de validación (ValidacionSimulada). El log trae el margen
                     de validación en `val`, y `greedy_reward` / `greedy_valid_rate` pasan a ser el
                     margen (USD) y la validez en esos episodios.
    trial_callback : callable opcional fn(step, n_updates, log_entry) -> bool, invocado en
                     cada update de log_freq; si devuelve True, el entrenamiento se detiene
                     ahí mismo (usado por hpo_optuna.py para pruning).

    Retorna
    -------
    agent          : AMRoutingAgent entrenado.
    critic         : CriticHead entrenado.
    episode_rewards: list[float]
    episode_losses : list[float]
    """
    d_h                   = AM_D_H
    n_heads               = AM_N_HEADS
    n_layers              = AM_N_LAYERS
    d_ff                  = AM_D_FF
    # hp(): lo definido en config.HP_POR_NODOS para este tamaño manda; si no cambia el valor,
    # se usa el global del módulo (que hpo_optuna.py sobreescribe por trial).
    n_episodes_per_update = hp("PPO_N_EPISODES_PER_UPDATE", PPO_N_EPISODES_PER_UPDATE)
    n_ppo_epochs          = PPO_N_EPOCHS
    ppo_batch_size        = PPO_BATCH_SIZE
    lr                    = hp("PPO_LR", PPO_LR)
    entropy_start         = hp("PPO_ENTROPY_COEF_START", PPO_ENTROPY_COEF_START)
    entropy_end           = hp("PPO_ENTROPY_COEF_END",   PPO_ENTROPY_COEF_END)
    lr_final_frac         = hp("LR_FINAL_FRAC", LR_FINAL_FRAC)
    gamma                 = PPO_GAMMA
    gae_lambda            = PPO_GAE_LAMBDA
    clip_eps              = PPO_CLIP_EPS
    grad_clip             = PPO_GRAD_CLIP

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    episodes_per_node = get_episodes_per_node(num_nodes)
    total_episodes    = episodes_per_node * num_nodes
    n_updates         = max(1, total_episodes // n_episodes_per_update)
    num_days          = rate_stack.shape[0]
    num_train_days    = min(TRAIN_DAYS, num_days)

    # ── Modelos ───────────────────────────────────────────────────────────────
    agent  = pretrained_agent  if pretrained_agent  is not None \
             else AMRoutingAgent(
                 num_nodes, d_h, n_heads, n_layers, d_ff, device=DEVICE,
                 pre_norm=AM_PRE_NORM, dropout=AM_DROPOUT, use_post_head=False,
             )
    critic = pretrained_critic if pretrained_critic is not None \
             else CriticHead(d_h).to(DEVICE)

    # Optimizers separados: el critic necesita converger más rápido que el actor
    # para dar señales de ventaja de calidad. Con un optimizer compartido y lr=3e-5
    # el critic aprende demasiado lento y el EV se estanca por debajo de 0.7.
    # Ratio 10x entre critic y actor es estándar en PPO para problemas combinatorios.
    # Actor con AdamW: weight_decay es el regularizador PPO-safe primario (no
    # corrompe el ratio de importancia como lo haría dropout en el path de logits).
    # Critic se deja en Adam puro (sin weight decay): su objetivo es MSE de
    # regresión, no política, y no comparte el problema de corrupción del ratio.
    # Las cabezas post-decisión (PRICING_PLAN.md), si el agente las trae, NO son de la política y el PPO no las
    # toca: se entrenan después (post_training.py). Quedan fuera del actor para que el PPO sea el mismo con o sin ellas.
    post_ids     = {id(p) for h in (getattr(agent, "post_value_a", None), getattr(agent, "post_value_b", None))
                    if h is not None for p in h.parameters()}
    actor_params = [p for p in agent.parameters() if id(p) not in post_ids]
    actor_optimizer  = torch.optim.AdamW(actor_params, lr=lr, weight_decay=PPO_WEIGHT_DECAY)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=lr * 10)

    # LR con decaimiento coseno: cada LR baja desde su valor inicial hasta LR_FINAL_FRAC de ese valor
    # al terminar n_updates (un scheduler.step() al final de cada update).
    schedulers = []
    if USE_LR_COSINE:
        def _cos(u, n=n_updates, f=lr_final_frac):
            return f + (1.0 - f) * 0.5 * (1.0 + math.cos(math.pi * min(u, n) / max(n, 1)))
        schedulers = [torch.optim.lr_scheduler.LambdaLR(o, _cos) for o in (actor_optimizer, critic_optimizer)]

    # ── Precomputar stack de reward matrices (una sola vez, fuera del loop) ──────
    time_matrix_arr = (
        time_matrix.to_numpy(dtype=float)
        if hasattr(time_matrix, "to_numpy")
        else np.asarray(time_matrix, dtype=float)
    )
    if market_process is None:
        rm_pen_stack = build_rm_pen_stack(
            rate_stack[:num_train_days], loads_stack[:num_train_days],
            distance_arr, diesel_arr,
        )   # (num_train_days, N, N) float32
    else:
        rm_pen_stack = None   # con proceso exógeno la recompensa sale de las filas sorteadas

    # ── Valor esperado por lane (features de continuación): una sola vez, fuera de los loops ──
    if ev_esperado is None and USE_CONTINUATION_FEATURES:
        if market_process is None:
            raise ValueError("USE_CONTINUATION_FEATURES requiere ev_esperado o market_process")
        ev_esperado = calcular_ev_pronostico(market_process, distance_arr, diesel_arr)

    # ── Entorno vectorizado (una sola instancia, reutilizada en cada update) ──
    # Índice de mercado observado de los días de entrenamiento (194 nodos, recortado a N): lo ven los entornos
    # de replay (el monitor greedy y el entrenamiento con días reales). Con proceso exógeno el entrenamiento
    # usa el índice del mundo de cada episodio.
    if market_process is not None:
        mercado = market_process
    else:
        asegurar_parametros()
        mercado = MarketProcess(num_nodes)
    indice_train = mercado.indice_historico(0, num_train_days)
    vec_env = VectorRoutingEnv(
        time_matrix=time_matrix,
        rate_stack=rate_stack,
        loads_stack=loads_stack,
        rm_pen_stack=rm_pen_stack,
        num_nodes=num_nodes,
        max_duration=MAX_DURATION,
        market_process=market_process,
        distance_arr=distance_arr,
        diesel_arr=diesel_arr,
        indice_stack=indice_train,
        indice_std=mercado.indice_std,
    )
    # Entorno del monitor greedy: replay de los días de entrenamiento (market_process=None),
    # aunque el entrenamiento use mundos simulados. Si ya hay rm_pen_stack se reutiliza.
    eval_env = VectorRoutingEnv(
        time_matrix=time_matrix,
        rate_stack=rate_stack,
        loads_stack=loads_stack,
        rm_pen_stack=rm_pen_stack if rm_pen_stack is not None else build_rm_pen_stack(
            rate_stack[:num_train_days], loads_stack[:num_train_days],
            distance_arr, diesel_arr,
        ),
        num_nodes=num_nodes,
        max_duration=MAX_DURATION,
        indice_stack=indice_train,
        indice_std=mercado.indice_std,
    )
    # Generador PROPIO de semillas de mundo (nunca el estado global de numpy): una
    # semilla por episodio y update, independiente de las decisiones del agente.
    world_rng = np.random.default_rng(MARKET_SEED_TRAIN)

    print(
        f"\n--- Entrenamiento AM-PPO: {total_episodes} episodios "
        f"({n_updates} updates × {n_episodes_per_update} ep/update) ---"
    )
    if market_process is not None:
        print("Mercado: realizaciones diarias — cada episodio sortea una vez hub inicial, día real "
              "inicial y corrimiento de nivel δ.")

    # ── Set fijo de evaluación greedy (mismos episodios en cada chequeo) ──────
    # RandomState propio, separado del RNG global de numpy: así no perturba la
    # secuencia de muestreo de start_nodes/start_day_idxs del loop de entrenamiento.
    n_eval_episodes = min(N_EVAL_EPISODES, num_nodes, 20)
    eval_rng         = np.random.RandomState(SEED)
    eval_start_nodes = eval_rng.choice(num_nodes, size=n_eval_episodes, replace=False)
    eval_start_days  = eval_rng.randint(0, num_train_days, size=n_eval_episodes)
    # Niveles de K balanceados sobre el set fijo de evaluación — mismo eval_rng
    # dedicado (nunca np.random global), calculado una sola vez.
    eval_k_levels    = _balanced_k_assignment(n_eval_episodes, eval_rng)

    # ── Validación con episodios fijos (solo con proceso exógeno) ─────────────
    val = None
    if market_process is not None:
        val = ValidacionSimulada(
            market_process, time_matrix, rate_stack, loads_stack, distance_arr, diesel_arr, num_nodes,
            time_matrix_arr, ltr_stack, reward_global_p95, avail_expected_arr,
            ev_esperado=ev_esperado,
        )
        refs = " | ".join(
            f"K={'libre' if K is None else K}: B3 {r['b3']:.0f}, UB-DP {r['ub_dp']:.0f}"
            for K, r in val.ref.items()
        )
        print(f"Validación: {val.M} episodios fijos (hub, día inicial y δ sorteados una vez, semilla "
              f"{MARKET_SEED_VAL}) "
              f"× K∈{{{', '.join('libre' if K is None else str(K) for K in val.niveles)}}} "
              f"| referencias (USD): {refs}")
    best_score = -np.inf
    # Selección por media móvil: pesos (CPU) y score de las últimas CKPT_WINDOW validaciones.
    ventana = deque(maxlen=CKPT_WINDOW)
    best_ma = -np.inf
    ckpt_update = None        # update cuyos pesos están hoy en best_checkpoint_path
    ckpt_score = None         # score puntual de ese update

    episode_rewards = []
    episode_losses  = []
    training_log    = []

    # Tiempos por fase acumulados desde el último log (se imprimen y reinician en log_freq).
    fases_acum  = defaultdict(float)
    updates_acum = 0

    # ── Loop principal ────────────────────────────────────────────────────────
    for update in range(n_updates):
        t_update0    = time.perf_counter()
        fases        = _Fases()
        progress     = update / max(1, n_updates - 1)
        entropy_coef = entropy_start + (entropy_end - entropy_start) * progress

        buffer = RolloutBuffer()
        update_ep_rewards = []

        # ── Fase 1: Recolección vectorizada (B episodios en paralelo) ─────────
        agent.eval()
        critic.eval()

        B = n_episodes_per_update

        # Balanced start-node sampling: each node appears exactly floor(B/N) times,
        # with remainder filled by random choice — same balance as the sequential
        # greedy node_ep_counts approach but per-update instead of globally.
        episodes_per_node = B // num_nodes
        remainder         = B % num_nodes
        start_nodes_arr   = np.repeat(np.arange(num_nodes), episodes_per_node)
        if remainder:
            extra = np.random.choice(num_nodes, size=remainder, replace=False)
            start_nodes_arr = np.concatenate([start_nodes_arr, extra])
        start_nodes_arr = np.random.permutation(start_nodes_arr).astype(np.int64)
        start_day_idxs  = np.random.randint(0, num_train_days, size=B).astype(np.int64)

        # Balanced K-level sampling: los niveles de K_TRAINING_LEVELS (restringido
        # + libre) se reparten balanceados sobre los B episodios, con su propia permutación
        # (np.random.permutation independiente de la de start_nodes_arr —
        # cruce aproximadamente balanceado).
        k_max_arr = _balanced_k_assignment(B, np.random)

        # Per-episode accumulators — each is a list of transition dicts.
        # Accumulated separately to keep episodes contiguous in the flat buffer,
        # since episodes of different lengths interleave in the step loop.
        ep_bufs        = [[] for _ in range(B)]

        # Un mundo (semilla) por episodio; el rollout greedy del self-critical reusa
        # las mismas semillas para ver la misma realización.
        world_seeds = world_rng.integers(0, 2**31 - 1, size=B) if market_process is not None else None

        masks  = vec_env.reset(start_nodes_arr, start_day_idxs, k_max_arr=k_max_arr,
                               world_seeds=world_seeds)   # (B, N)
        active = np.ones(B, dtype=bool)

        with torch.no_grad():
            for step_i in range(max_pasos(time_matrix_arr, MAX_DURATION)):   # defensive ceiling
                if not active.any():
                    break

                cur_nodes = vec_env.current_node     # (B,) int64
                time_el   = vec_env.time_elapsed     # (B,) float32
                step_cnt  = vec_env.step_count       # (B,) int64
                day_idx   = vec_env.current_day_idx  # (B,) int64

                # Feature construction — no Python loops over N or B
                with fases("rollout_features"):
                    node_feats = build_node_features_batch(
                        cur_nodes, start_nodes_arr,
                        rm_pen_stack, time_matrix_arr, distance_arr,
                        day_idx, num_nodes, MAX_DURATION,
                        reward_global_p95=reward_global_p95,
                        avail_expected_arr=avail_expected_arr,
                        reward_rows=vec_env.reward_rows,
                        ev=ev_esperado, time_elapsed=time_el, dow0=vec_env.start_dow,
                        market_index=vec_env.market_index, index_std=vec_env.index_std,
                    )   # (B, N, N_NODE_FEATURES)
                    temporal = build_temporal_features_batch(time_el, step_cnt, MAX_DURATION, num_nodes)   # (B, 3)
                    market   = build_market_features_batch(cur_nodes, day_idx, ltr_stack) if ltr_stack is not None \
                               else np.zeros((B, 1), dtype=np.float32)                                     # (B, 1)
                    budget   = build_budget_features_batch(vec_env.k_remaining, vec_env.k_restricted)      # (B, 2)
                    dow      = build_dow_features_batch(vec_env.current_dow)                               # (B, 7)

                # Single forward pass over the full batch
                with fases("rollout_gpu"):
                    node_feats_t = torch.from_numpy(node_feats).to(DEVICE)
                    temporal_t   = torch.from_numpy(temporal).to(DEVICE)
                    market_t     = torch.from_numpy(market).to(DEVICE)
                    budget_t     = torch.from_numpy(budget).to(DEVICE)
                    dow_t        = torch.from_numpy(dow).to(DEVICE)
                    cur_nodes_t  = torch.tensor(cur_nodes, dtype=torch.long, device=DEVICE)
                    mask_t       = torch.from_numpy(masks).to(DEVICE)

                    actions_t, log_probs_t, _, values_t = _forward(
                        agent, critic, node_feats_t, temporal_t, cur_nodes_t, mask_t,
                        market_feats_t=market_t, budget_feats_t=budget_t, dow_feats_t=dow_t,
                    )
                    actions   = actions_t.cpu().numpy()    # (B,) int64
                    log_probs = log_probs_t.cpu().numpy()  # (B,)
                    values    = values_t.cpu().numpy()     # (B,)

                # Vectorized environment step
                with fases("rollout_env"):
                    rewards, terminated, next_masks = vec_env.step(actions)
                    time_violation = vec_env.last_time_violation   # (B,) bool — instrumentación de diagnósticos de K

                # Write transitions — only for rows active BEFORE this step
                with fases("rollout_buffer"):
                    for b in np.where(active)[0]:
                        ep_bufs[b].append({
                            "node_feats":   node_feats[b],        # (N, N_NODE_FEATURES)
                            "temporal":     temporal[b],           # (3,)
                            "market_feats": market[b],             # (1,)
                            "budget_feats": budget[b],              # (2,)
                            "dow_feats":    dow[b],                 # (7,)
                            "current_node": int(cur_nodes[b]),
                            "mask":         masks[b],              # (N,) int8
                            "action":       int(actions[b]),
                            "reward":       float(rewards[b]),
                            "log_prob":     float(log_probs[b]),
                            "value":        float(values[b]),
                            "done":         bool(terminated[b]),
                            "time_violation": bool(time_violation[b]),
                        })

                active = active & ~terminated
                masks  = next_masks

        # max_pasos bastan para cerrar cualquier ciclo (cada arco consume al menos el menor tiempo de
        # viaje y el que excede MAX_DURATION termina el episodio): un episodio activo aquí sería un bug.
        if active.any():
            raise RuntimeError(
                f"{int(active.sum())} episodios no terminaron dentro de max_pasos="
                f"{max_pasos(time_matrix_arr, MAX_DURATION)} pasos"
            )

        # ── Self-critical baseline: rollout greedy vectorizado ──────
        # Sobre los MISMOS start_nodes_arr/start_day_idxs del rollout muestreado,
        # para comparar en las mismas unidades y la misma realización del mundo.
        # Resetea vec_env (su estado post-rollout muestreado ya no se necesita).
        if USE_SELF_CRITICAL:
            with fases("greedy"):
                greedy_ep_rewards = greedy_rollout(
                    agent, vec_env, start_nodes_arr, start_day_idxs,
                    time_matrix_arr, distance_arr, ltr_stack,
                    reward_global_p95, num_nodes,
                    k_max_arr=k_max_arr,
                    avail_expected_arr=avail_expected_arr,
                    world_seeds=world_seeds,
                    ev_esperado=ev_esperado,
                ).env_reward
        else:
            greedy_ep_rewards = None

        with fases("aplanar_gae"):
            # Flatten per-episode accumulators into the main RolloutBuffer
            # Episodes are written sequentially so compute_gae_per_episode sees
            # contiguous segments — its logic is unchanged.
            update_r_greedy = []
            # Diagnósticos de K: longitud realizada y violación temporal en
            # retorno forzado, agrupadas por nivel de K asignado al episodio.
            k_diag = {lvl: {"lengths": [], "time_violations": 0} for lvl in K_TRAINING_LEVELS}
            for b in range(B):
                if not ep_bufs[b]:
                    continue
                ep_start = len(buffer)
                for trans in ep_bufs[b]:
                    buffer.add(
                        node_feats=trans["node_feats"],
                        temporal=trans["temporal"],
                        current_node=trans["current_node"],
                        mask=trans["mask"],
                        action=trans["action"],
                        reward=trans["reward"],
                        log_prob=trans["log_prob"],
                        value=trans["value"],
                        done=trans["done"],
                        market_feats=trans["market_feats"],
                        budget_feats=trans["budget_feats"],
                        dow_feats=trans["dow_feats"],
                    )
                ep_end = len(buffer)
                buffer.episode_boundaries.append(
                    (ep_start, ep_end)
                )
                ep_reward = sum(t["reward"] for t in ep_bufs[b])
                update_ep_rewards.append(ep_reward)
                episode_rewards.append(ep_reward)
                if USE_SELF_CRITICAL:
                    update_r_greedy.append(float(greedy_ep_rewards[b]))

                k_lvl = k_max_arr[b]
                k_diag[k_lvl]["lengths"].append(len(ep_bufs[b]))
                if ep_bufs[b][-1]["time_violation"]:
                    k_diag[k_lvl]["time_violations"] += 1

            buffer.compute_gae_per_episode(gamma=gamma, gae_lambda=gae_lambda)

            # ── Ventaja self-critical por episodio ──────────────────────
            sc_adv_raw_mean = 0.0
            r_greedy_avg    = 0.0
            if USE_SELF_CRITICAL:
                sc_advantage_raw = np.array(update_ep_rewards, dtype=np.float32) \
                                  - np.array(update_r_greedy,  dtype=np.float32)
                sc_adv_raw_mean = float(sc_advantage_raw.mean())
                r_greedy_avg    = float(np.mean(update_r_greedy))

                sc_advantage_norm = (
                    (sc_advantage_raw - sc_advantage_raw.mean())
                    / (sc_advantage_raw.std() + 1e-8)
                )

                episode_idx_arr = np.zeros(len(buffer), dtype=np.int64)
                for k, (s, e) in enumerate(buffer.episode_boundaries):
                    episode_idx_arr[s:e] = k

                buffer.sc_advantage = sc_advantage_norm
                buffer.episode_idx  = episode_idx_arr

        # ── Fase 2: Actualización PPO ─────────────────────────────────────────
        agent.train()
        critic.train()

        batch_total_losses  = []
        batch_actor_losses  = []
        batch_value_losses  = []
        batch_entropies     = []
        batch_kl_divs       = []
        batch_clip_fracs    = []
        batch_sc_losses     = []
        apply_self_critical = USE_SELF_CRITICAL and update >= SELF_CRITICAL_WARMUP_UPDATES

        for epoch_i in range(n_ppo_epochs):

            kl_too_high = False

            for _batch_idx, idx_batch in enumerate(buffer.get_batches(ppo_batch_size)):
                B = len(idx_batch)

                with fases("ppo_armado", sync=True):
                    # Reconstruir tensores del batch
                    nf_b   = torch.from_numpy(
                        np.stack([buffer.node_feats[i] for i in idx_batch])
                    ).to(DEVICE)                                           # (B, N, N_NODE_FEATURES)
                    tf_b   = torch.from_numpy(
                        np.stack([buffer.temporal_feats[i] for i in idx_batch])
                    ).to(DEVICE)                                           # (B, 3)
                    mf_b   = torch.from_numpy(
                        np.stack([buffer.market_feats[i] for i in idx_batch])
                    ).to(DEVICE)                                           # (B, 1)
                    bf_b   = torch.from_numpy(
                        np.stack([buffer.budget_feats[i] for i in idx_batch])
                    ).to(DEVICE)                                           # (B, 2)
                    dw_b   = torch.from_numpy(
                        np.stack([buffer.dow_feats[i] for i in idx_batch])
                    ).to(DEVICE)                                           # (B, 7)
                    cn_b   = torch.tensor(
                        [buffer.current_nodes[i] for i in idx_batch],
                        dtype=torch.long
                    ).to(DEVICE)                                           # (B,)
                    mask_b = torch.from_numpy(
                        np.stack([buffer.masks[i] for i in idx_batch])
                    ).to(DEVICE)                                           # (B, N)
                    act_b  = torch.tensor(
                        [buffer.actions[i] for i in idx_batch],
                        dtype=torch.long
                    ).to(DEVICE)                                           # (B,)
                    old_lp = torch.tensor(
                        [buffer.old_log_probs[i] for i in idx_batch],
                        dtype=torch.float32
                    ).to(DEVICE)                                           # (B,)
                    ret_b  = torch.tensor(
                        buffer.returns[idx_batch], dtype=torch.float32
                    ).to(DEVICE)                                           # (B,)
                    adv_b  = torch.tensor(
                        buffer.advantages[idx_batch], dtype=torch.float32
                    ).to(DEVICE)                                           # (B,)

                    if apply_self_critical:
                        sc_adv_b = torch.tensor(
                            buffer.sc_advantage[buffer.episode_idx[idx_batch]],
                            dtype=torch.float32,
                        ).to(DEVICE)                                       # (B,)

                with fases("ppo_calculo", sync=True):
                    # Normalización por mini-batch: cada gradiente ve ventajas
                    # con media=0, std=1, independiente del resto del buffer.
                    # Más estable que normalizar el buffer completo una sola vez
                    # porque evita que outliers de otros mini-batches contaminen
                    # la escala de la actualización actual.
                    adv_b = (adv_b - adv_b.mean()) / (adv_b.std(unbiased=False) + 1e-8)

                    # Clipping de ventajas: limita casos extremos sin afectar
                    # la dirección del gradiente (solo la magnitud).
                    adv_b = torch.clamp(adv_b, -5.0, 5.0)

                    check_tensor("advantages adv_b", adv_b)

                    # Sentinelas para el bloque except (pueden no existir si el
                    # error ocurre antes de que se calculen)
                    new_lp = ratio = None

                    try:
                        # Forward con gradientes
                        _, new_lp, entropy, new_val = _forward(
                            agent, critic, nf_b, tf_b, cn_b, mask_b,
                            actions_t=act_b, market_feats_t=mf_b, budget_feats_t=bf_b, dow_feats_t=dw_b,
                        )

                        # ── Loss del actor (PPO-Clip) ────────────────────────────
                        ratio       = torch.exp(new_lp - old_lp)
                        check_tensor("PPO ratio", ratio)

                        surr1       = ratio * adv_b
                        surr2       = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * adv_b
                        actor_loss  = -torch.min(surr1, surr2).mean()

                        # ── Loss del critic (MSE) ────────────────────────────────
                        value_loss  = nn.functional.mse_loss(new_val, ret_b)

                        # ── Bonus de entropía (exploración) ─────────────────────
                        entropy_loss = -entropy.mean()

                        actor_loss_total = actor_loss + entropy_coef * entropy_loss

                        # ── Self-critical: término REINFORCE adicional ──
                        # No toca el critic — sc_loss depende solo de new_lp (actor).
                        if apply_self_critical:
                            sc_loss = -(SELF_CRITICAL_COEF * sc_adv_b * new_lp).mean()
                            actor_loss_total = actor_loss_total + sc_loss

                    except RuntimeError as e:
                        sep = "=" * 60
                        print(f"\n{sep}")
                        print(f"NaN/Inf ABORT  update={update + 1}  batch={_batch_idx}")
                        print(f"  error        : {e}")
                        print(f"  avg reward   : {float(np.mean(update_ep_rewards)):.4f}")
                        print(f"  mask valid   : {mask_b.sum(dim=-1).tolist()}")
                        print(f"  action mask  :\n{mask_b.cpu().numpy()}")
                        print(
                            f"  adv_b        : min={adv_b.min():.4f}  max={adv_b.max():.4f}"
                            f"  mean={adv_b.mean():.4f}  std={adv_b.std():.4f}"
                        )
                        print(
                            f"  old_lp       : min={old_lp.min():.4f}  max={old_lp.max():.4f}"
                            f"  mean={old_lp.mean():.4f}"
                        )
                        if new_lp is not None:
                            print(
                                f"  new_lp       : min={new_lp.min():.4f}  max={new_lp.max():.4f}"
                                f"  mean={new_lp.mean():.4f}"
                            )
                        if ratio is not None:
                            print(
                                f"  ratio        : min={ratio.min():.4f}  max={ratio.max():.4f}"
                                f"  mean={ratio.mean():.4f}"
                            )
                        print(sep)
                        raise

                    # ── Update actor & critic ─────────────────────────────────
                    # value_loss también backpropaga hacia el encoder/context_net
                    # del actor (h_t no se detacha antes del critic), así que un
                    # solo backward sobre la suma evita recorrer ese tramo
                    # compartido dos veces — los gradientes resultantes son
                    # idénticos a hacer backward por separado (gradiente de la
                    # suma = suma de gradientes), pero ~1.3x más rápido.
                    actor_optimizer.zero_grad()
                    critic_optimizer.zero_grad()
                    total_loss = actor_loss_total + value_loss
                    total_loss.backward()
                    nn.utils.clip_grad_norm_(actor_params, grad_clip)
                    actor_optimizer.step()
                    nn.utils.clip_grad_norm_(critic.parameters(), grad_clip)
                    critic_optimizer.step()

                    with torch.no_grad():
                        # KL aproximada: E[log π_old - log π_new]
                        approx_kl  = (old_lp - new_lp).mean().item()
                        # Fracción de ratios que el clip PPO recortó
                        clip_frac  = ((ratio - 1).abs() > clip_eps).float().mean().item()

                if approx_kl > 0.05:
                    kl_too_high = True
                    batch_kl_divs.append(approx_kl)
                    batch_clip_fracs.append(clip_frac)
                    break


                batch_total_losses.append(actor_loss_total.item())
                batch_actor_losses.append(actor_loss.item())
                batch_value_losses.append(value_loss.item())
                batch_entropies.append(entropy.mean().item())
                batch_kl_divs.append(approx_kl)
                batch_clip_fracs.append(clip_frac)
                if apply_self_critical:
                    batch_sc_losses.append(sc_loss.item())

            if kl_too_high:  # sale también de n_ppo_epochs
                break

        # Explained variance: cuánto explica el critic los retornos reales
        ret_all = buffer.returns
        val_all = np.array(buffer.old_values, dtype=np.float32)
        var_ret = np.var(ret_all)
        explained_var = (
            float(1.0 - np.var(ret_all - val_all) / (var_ret + 1e-8))
            if var_ret > 1e-8 else 0.0
        )

        avg_reward    = float(np.mean(update_ep_rewards))
        avg_loss      = float(np.mean(batch_total_losses))  if batch_total_losses  else 0.0
        avg_pol_loss  = float(np.mean(batch_actor_losses))  if batch_actor_losses  else 0.0
        avg_val_loss  = float(np.mean(batch_value_losses))  if batch_value_losses  else 0.0
        avg_entropy   = float(np.mean(batch_entropies))     if batch_entropies     else 0.0
        avg_kl        = float(np.mean(batch_kl_divs))       if batch_kl_divs       else 0.0
        avg_clip_frac = float(np.mean(batch_clip_fracs))    if batch_clip_fracs    else 0.0
        avg_sc_loss   = float(np.mean(batch_sc_losses))     if batch_sc_losses     else 0.0

        episode_losses.append(avg_loss)

        # ── Diagnósticos de K: resumen por nivel — tasa de conformidad
        # (debe ser 100% en niveles restringidos, garantizado por máscara),
        # distribución de longitud realizada y frecuencia de violación
        # temporal en el retorno forzado. Se calcula cada update (barato,
        # agregación numpy sobre datos ya recolectados) pero solo se imprime
        # un resumen en log_freq, igual que el resto de las métricas.
        k_diagnostics = {}
        for lvl, d in k_diag.items():
            lengths = np.array(d["lengths"], dtype=np.int64)
            key = "free" if lvl is None else str(lvl)
            if len(lengths) == 0:
                k_diagnostics[key] = None
                continue
            entry = {
                "n_episodes":        int(len(lengths)),
                "mean_length":       float(lengths.mean()),
                "p50_length":        float(np.percentile(lengths, 50)),
                "p90_length":        float(np.percentile(lengths, 90)),
                "max_length":        int(lengths.max()),
                "time_violation_rate": d["time_violations"] / len(lengths),
            }
            if lvl is not None:
                entry["conformity_rate"] = float((lengths <= lvl).mean())
            k_diagnostics[key] = entry

        log_entry = {
            "update":           update + 1,
            "reward":           avg_reward,
            "total_loss":       avg_loss,
            "policy_loss":      avg_pol_loss,
            "value_loss":       avg_val_loss,
            "entropy":          avg_entropy,
            "kl_divergence":    avg_kl,
            "clip_fraction":    avg_clip_frac,
            "explained_var":    explained_var,
            "entropy_coef":     entropy_coef,
            "k_diagnostics":    k_diagnostics,
        }
        if USE_SELF_CRITICAL:
            log_entry["r_greedy_avg"]      = r_greedy_avg
            log_entry["sc_advantage_mean"] = sc_adv_raw_mean
            log_entry["sc_loss_value"]     = avg_sc_loss

        stop_now = False
        log_freq = max(1, n_updates // 20)
        val_freq = max(1, n_updates // VAL_CHECKS)
        es_log = (update + 1) % log_freq == 0

        with fases("validacion"):
            # Validación con episodios fijos (observacional); guarda el mejor update.
            v = None
            if val is not None and (es_log or (update + 1) % val_freq == 0 or update == n_updates - 1):
                v = val.evaluar(agent)
                log_entry["val"]               = v["por_k"]
                log_entry["greedy_reward"]     = v["margen"]
                log_entry["greedy_valid_rate"] = v["valid_rate"]
                log_entry["val_score"] = v["score"]
                if CKPT_SELECTION == "moving_avg":
                    ventana.append({
                        "update": update + 1, "score": v["score"],
                        "agent":  {k: t.detach().cpu().clone() for k, t in agent.state_dict().items()},
                        "critic": {k: t.detach().cpu().clone() for k, t in critic.state_dict().items()},
                    })
                    ma = float(np.mean([w["score"] for w in ventana])) if len(ventana) == CKPT_WINDOW else None
                    if ma is not None and ma > best_ma:
                        best_ma = ma
                        centro = ventana[len(ventana) // 2]         # update CENTRAL de la ventana, no el último
                        ckpt_update, ckpt_score = centro["update"], centro["score"]
                        log_entry["val_best"] = True
                        if best_checkpoint_path:
                            torch.save({"agent": centro["agent"], "critic": centro["critic"]}, best_checkpoint_path)
                    log_entry["val_ma"] = ma
                    log_entry["ckpt_update"] = ckpt_update
                    log_entry["ckpt_score"] = ckpt_score
                elif v["score"] > best_score:
                    best_score = v["score"]
                    log_entry["val_best"] = True
                    ckpt_update, ckpt_score = update + 1, v["score"]
                    if best_checkpoint_path:
                        torch.save({"agent": agent.state_dict(), "critic": critic.state_dict()}, best_checkpoint_path)
                if CKPT_SELECTION != "moving_avg":
                    log_entry["ckpt_update"] = ckpt_update
                    log_entry["ckpt_score"] = ckpt_score

        if es_log:
            if val is None:
                with fases("validacion"):
                    greedy_reward, greedy_valid_rate = _evaluate_greedy(
                        agent, eval_env, eval_start_nodes, eval_start_days,
                        time_matrix_arr, distance_arr, ltr_stack,
                        reward_global_p95, num_nodes,
                        eval_k_levels=eval_k_levels,
                        avail_expected_arr=avail_expected_arr,
                        ev_esperado=ev_esperado,
                    )
                    log_entry["greedy_reward"]     = greedy_reward
                    log_entry["greedy_valid_rate"] = greedy_valid_rate
            else:
                greedy_reward, greedy_valid_rate = v["margen"], v["valid_rate"]

            episodes_done = (update + 1) * n_episodes_per_update
            # Detalle completo (loss/pol/val/clip/r_greedy/sc_adv/sc_loss) sigue
            # en training_log y en las gráficas de plot_ppo_diagnostics; esto es
            # solo el resumen de consola.
            print(
                f"  update {update+1:>5}/{n_updates} "
                f"| episodios {episodes_done:>6}/{total_episodes} "
                f"| reward_train {avg_reward:6.1f}"
            )
            if val is None:
                print(f"    greedy={greedy_reward:.1f}usd")
            else:
                val_linea = " ; ".join(
                    f"val[K={'libre' if k == 'free' else k}] margen={d['margen']:.0f}usd "
                    f"(vs_B3 {d['dif_b3']:+.1f}usd, gap_UBDP {d['gap_ub_dp_pct']:.1f}%, ref_B3={d['b3']:.0f}usd)"
                    for k, d in v["por_k"].items()
                )
                print(f"    {val_linea}")
                if log_entry.get("val_ma") is not None:
                    ckpt = (f" *checkpoint guardado (mejor update={log_entry['ckpt_update']})"
                            if log_entry.get("val_best") else "")
                    print(f"    MA({CKPT_WINDOW} val)={log_entry['val_ma']:.0f}usd{ckpt}")
            print(
                f"    validez={greedy_valid_rate:.0%} "
                f"| entropia={avg_entropy:.3f} "
                f"| KL={avg_kl:.4f} "
                f"| var_explicada={explained_var:.3f}"
            )

            # Resumen compacto de diagnósticos de K — conformidad debe ser
            # 1.00 siempre (garantizada por máscara); si no lo es, hay un bug.
            k_summary_parts = []
            for lvl in K_TRAINING_LEVELS:
                key = "free" if lvl is None else str(lvl)
                d = k_diagnostics.get(key)
                if d is None:
                    continue
                if lvl is None:
                    k_summary_parts.append(f"free:len={d['mean_length']:.1f}")
                else:
                    k_summary_parts.append(
                        f"K={lvl}:conf={d['conformity_rate']:.2f},"
                        f"len={d['mean_length']:.1f}/{d['p90_length']:.0f},"
                        f"tviol={d['time_violation_rate']:.2f}"
                    )
            print("    K-diag: " + " | ".join(k_summary_parts))

            if trial_callback is not None:
                stop_now = bool(trial_callback(update + 1, n_updates, log_entry))

        log_entry["lr_actor"] = actor_optimizer.param_groups[0]["lr"]
        for sch in schedulers:
            sch.step()

        # ── Tiempos por fase (s). "otro" = lo que no cae en ninguna fase medida.
        t_total = time.perf_counter() - t_update0
        for nombre in _FASES_ORDEN:
            log_entry[f"t_{nombre}"] = fases.t.get(nombre, 0.0)
            fases_acum[nombre] += fases.t.get(nombre, 0.0)
        log_entry["t_otro"]  = t_total - sum(fases.t.values())
        log_entry["t_total"] = t_total
        fases_acum["otro"]  += log_entry["t_otro"]
        fases_acum["total"] += t_total
        updates_acum += 1
        if es_log:
            partes = " | ".join(
                f"{n} {fases_acum[n] / updates_acum:.2f}s ({100 * fases_acum[n] / fases_acum['total']:.0f}%)"
                for n in (*_FASES_ORDEN, "otro") if fases_acum[n] > 0
            )
            print(f"    tiempos/update ({updates_acum} upd): total {fases_acum['total'] / updates_acum:.2f}s | {partes}")
            fases_acum.clear()
            updates_acum = 0

        training_log.append(log_entry)
        if stop_now:
            break

    # Entrenamientos tan cortos que nunca llenaron la ventana: guarda el centro de lo que haya.
    if CKPT_SELECTION == "moving_avg" and ckpt_update is None and ventana and best_checkpoint_path:
        centro = ventana[len(ventana) // 2]
        torch.save({"agent": centro["agent"], "critic": centro["critic"]}, best_checkpoint_path)
        ckpt_update = centro["update"]
        for e in training_log:
            if e["update"] == ckpt_update:
                e["val_best"] = True
        if training_log:
            training_log[-1]["ckpt_update"] = ckpt_update

    print("Entrenamiento AM-PPO completo.")
    return agent, critic, episode_rewards, episode_losses, training_log
