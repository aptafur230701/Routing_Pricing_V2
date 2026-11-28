"""
am_training.py
==============
Loop de entrenamiento PPO para AMRoutingAgent + CriticHead, diseñado para la GPU.

Algoritmo: Proximal Policy Optimization (PPO-Clip)
  1. Recolectar B episodios en paralelo con la política actual (sin gradientes).
  2. Calcular ventajas con GAE (Generalized Advantage Estimation).
  3. Actualizar actor + critic durante K épocas con mini-batches.
  4. Repetir hasta completar el número de updates.

Todo vive en el dispositivo (config.DEVICE): los mundos se sortean en la GPU al empezar cada update
(market_process.sortear_mundos), el entorno y las features son tensores, las transiciones se guardan como
tensores y GAE se calcula vectorizado sobre los B episodios. A la CPU solo vuelven los escalares del log.

· La acción es (destino, nivel de precio) en el entorno con cotización (routing_env.VectorRoutingEnv;
  especificacion_fase0_ruteo_pricing.md §4-5).
"""

import math
import os
import random
import time
from collections import defaultdict, deque
from contextlib import contextmanager

import numpy as np
import torch
import torch.nn as nn

from config import (
    MARKET_SEED_VAL, VAL_CHECKS, get_val_worlds,
    DEVICE,
    SEED,
    TRAIN_DAYS,
    MARKET_DIAS_EPISODIO,
    get_n_updates,
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
    PPO_ENTROPY_COEF_PRECIO_START, PPO_ENTROPY_COEF_PRECIO_END,
    RESUME_CADA_UPDATES, RESUME_CADA_MIN,
)
from routing_env import VectorRoutingEnv
from greedy_rollout import EntradasAgente, greedy_rollout, construir_features
from model.agent import AMRoutingAgent
from model.encoder import COT_COLS
from model.critic_head import CriticHead
from baselines.dp_relajada import PENALIZACION_USD
from model.debug_utils import check_tensor
from evaluation import dias_de_salida, disenar_mundos

_FEATS = ("node_feats", "temporal", "market", "budget", "dow", "cot")


# ─────────────────────────────────────────────────────────────────────────────
# Muestreo balanceado de niveles de K
# ─────────────────────────────────────────────────────────────────────────────

def _balanced_k_assignment(n: int, rng) -> list:
    """
    Reparte K_TRAINING_LEVELS (K_MAX restringido + caso libre None) de
    forma balanceada sobre n episodios y permuta. Mismo patrón que el
    balanceo de start_nodes_arr, pero independiente (cruce aproximadamente balanceado).

    rng : np.random.Generator | np.random.RandomState — debe exponer
          .permutation(n) y (si hace falta relleno) .integers/.randint.
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


def _k_codigo(k_list) -> np.ndarray:
    """Lista de (int | None) → array con 0 = libre (para agrupar diagnósticos en el dispositivo)."""
    return np.array([0 if k is None else int(k) for k in k_list], dtype=np.int64)


# ─────────────────────────────────────────────────────────────────────────────
# Forward pass compartido (recolección y actualización)
# ─────────────────────────────────────────────────────────────────────────────

def _forward(agent, critic, feats_t: dict, mask_t, actions_t=None, opcion_t=None):
    """encoder → context → decoder → critic para un batch, con la acción (destino, opción de precio).

    feats_t : tensores node_feats, temporal, market, budget, dow, cot, opciones y current_node. Si actions_t es None
    samplea (recolección); si no, evalúa las acciones almacenadas (actualización PPO).
    Retorna (destino, opción, log π(j, o), H_destino, H_precio, V(s)).
    """
    nf = feats_t["node_feats"]
    B = nf.shape[0]
    embeddings, graph_emb = agent.encoder(nf)
    check_tensor("embeddings", embeddings)
    current_emb = embeddings[torch.arange(B, device=nf.device), feats_t["current_node"], :]
    h_t = agent.context_net(graph_emb, current_emb, feats_t["temporal"], feats_t["market"], feats_t["budget"],
                            feats_t["dow"], feats_t["cot"])
    check_tensor("context h_t", h_t)
    values = critic(h_t)
    cot = nf[..., COT_COLS]                                                    # entra también a la cabeza de precio
    if actions_t is None:
        actions, opcion, log_probs, h_dest, h_precio = agent.decoder.act_cotizacion(
            h_t, embeddings, mask_t, feats_t["opciones"], cot)
    else:
        log_probs, h_dest, h_precio = agent.decoder.evaluate_cotizacion(
            h_t, embeddings, mask_t, feats_t["opciones"], cot, actions_t, opcion_t)
        actions, opcion = actions_t, opcion_t
    check_tensor("log_probs", log_probs)
    return actions, opcion, log_probs, h_dest, h_precio, values


# ─────────────────────────────────────────────────────────────────────────────
# Cronómetro por fase
# ─────────────────────────────────────────────────────────────────────────────

class _Fases:
    """Cronómetro por fase del update. `sync=True` espera a la GPU antes de leer el reloj
    (los kernels son asíncronos)."""

    def __init__(self):
        self.t = defaultdict(float)

    @contextmanager
    def __call__(self, nombre: str, sync: bool = True):
        t0 = time.perf_counter()
        try:
            yield
        finally:
            if sync and DEVICE.type == "cuda":
                torch.cuda.synchronize()
            self.t[nombre] += time.perf_counter() - t0


_FASES_ORDEN = ("mundos", "rollout", "greedy", "gae", "ppo", "validacion")


# ─────────────────────────────────────────────────────────────────────────────
# Validación en mundos fijos
# ─────────────────────────────────────────────────────────────────────────────

class ValidacionSimulada:
    """Validación del greedy en episodios fijos de mundos simulados.

    Un conjunto fijo de episodios (nodo, día de la semana y día de salida balanceados, con la semilla
    propia MARKET_SEED_VAL) se sortea UNA vez en el dispositivo y se evalúa por cada nivel de K de
    K_TRAINING_LEVELS en una sola llamada batcheada de greedy_rollout. Reporta el margen en USD (suma del reward
    de cada lane con el precio cotizado), la validez, la tasa de aceptación y el nivel medio cotizado.
    Es observacional: no afecta los gradientes ni el RNG global.

    No calcula B3 ni UB-DP: el entrenamiento no puede depender de la tabla de la DP (a 75-100 nodos
    puede no ser tratable). Las comparaciones contra los baselines son de la evaluación final.
    """

    def __init__(self, market_process, env: VectorRoutingEnv, entradas: EntradasAgente, num_dias: int, num_nodes: int):
        M = get_val_worlds(num_nodes)
        dis = disenar_mundos(num_nodes, M, market_process, num_dias=num_dias, semilla=MARKET_SEED_VAL)
        self.niveles = list(K_TRAINING_LEVELS)
        self.M = M
        g = torch.Generator(device=env.device).manual_seed(MARKET_SEED_VAL)
        base = market_process.sortear_mundos(dis["start_day"], MARKET_DIAS_EPISODIO, num_dias - 1, g)
        self.mundos = base.tomar(np.tile(np.arange(M), len(self.niveles)))
        self.nodos = np.tile(dis["nodo"], len(self.niveles))
        self.k_arr = [K for K in self.niveles for _ in range(M)]
        self.env, self.entradas = env, entradas

    def evaluar(self, agent) -> dict:
        """Margen medio del greedy por nivel de K y global, validez, aceptación y nivel medio."""
        res = greedy_rollout(agent, self.env, self.mundos, self.nodos, self.entradas, k_max=self.k_arr)
        margen = torch.where(res.valid, res.raw_reward, res.raw_reward + PENALIZACION_USD)   # como en la evaluación
        margen, valid = margen.cpu().numpy(), res.valid.cpu().numpy()
        n_cot, n_gan, s_niv = (x.cpu().numpy() for x in (res.n_cotizaciones, res.n_ganadas, res.suma_niveles))
        out = {"por_k": {}}
        for i, K in enumerate(self.niveles):
            sl = slice(i * self.M, (i + 1) * self.M)
            cot = int(n_cot[sl].sum())
            out["por_k"]["free" if K is None else str(K)] = {
                "margen": float(margen[sl].mean()),
                "valid_rate": float(valid[sl].mean()),
                "tasa_aceptacion": float(n_gan[sl].sum() / cot) if cot else float("nan"),
                "nivel_medio": float(s_niv[sl].sum() / cot) if cot else float("nan"),
            }
        out["margen"] = float(margen.mean())
        out["valid_rate"] = float(valid.mean())
        out["score"] = out["margen"]           # con las fallas penalizadas: elige el mejor update
        return out


# ─────────────────────────────────────────────────────────────────────────────
# Rollout muestreado y GAE (en el dispositivo)
# ─────────────────────────────────────────────────────────────────────────────

def _rollout_muestreado(agent, critic, env, mundos, start_nodes, k_list, entradas) -> dict:
    """B episodios en paralelo con la política muestreada. Devuelve las transiciones de las filas activas
    concatenadas (tensores (n, ...)), con el episodio `ep` y el paso `t` de cada una."""
    masks = env.reset(mundos, start_nodes, k_list)
    active = env.active.clone()                   # una fila sin ninguna jugada desde casa no sale (sin transiciones)
    trans = defaultdict(list)
    for t in range(env.max_steps):
        idx = active.nonzero().squeeze(1)
        if idx.numel() == 0:
            break
        feats = construir_features(env, entradas)
        a, o, lp, _, _, v = _forward(agent, critic, feats, masks)
        rewards, terminated, next_masks = env.step(a, o)
        for k in _FEATS + ("opciones", "current_node"):
            trans[k].append(feats[k][idx])
        for k, x in (("mask", masks), ("action", a), ("opcion", o), ("log_prob", lp), ("value", v),
                     ("reward", rewards), ("done", terminated), ("time_violation", env.last_time_violation),
                     ("viajo", ~(env.last_cotizo & ~env.last_gano) & ~env.last_espero),
                     ("cotizo", env.last_cotizo), ("gano", env.last_gano)):
            trans[k].append(x[idx])
        trans["ep"].append(idx)
        trans["t"].append(torch.full_like(idx, t))
        masks = next_masks
        active = active & ~terminated
    if bool(active.any()):
        # max_pasos_cotizacion bastan para cerrar cualquier ciclo: un episodio activo aquí sería un bug.
        raise RuntimeError(f"{int(active.sum())} episodios no terminaron dentro del techo de {env.max_steps} pasos")
    return {k: torch.cat(v) for k, v in trans.items()}


def _gae(tr: dict, B: int, gamma: float, lam: float) -> tuple:
    """Ventajas y retornos con GAE, vectorizado sobre los B episodios. Las transiciones de un episodio ocupan los
    pasos 0 … len−1 sin huecos (la fila está activa hasta que termina) y el valor después del último es 0."""
    dev = tr["reward"].device
    P = int(tr["t"].max()) + 1
    R = torch.zeros(B, P + 1, device=dev)
    V = torch.zeros(B, P + 1, device=dev)
    hay = torch.zeros(B, P + 1, dtype=torch.bool, device=dev)
    R[tr["ep"], tr["t"]] = tr["reward"].float()
    V[tr["ep"], tr["t"]] = tr["value"].float()
    hay[tr["ep"], tr["t"]] = True
    adv = torch.zeros(B, P + 1, device=dev)
    for t in range(P - 1, -1, -1):
        sig = hay[:, t + 1].float()
        delta = R[:, t] + gamma * V[:, t + 1] * sig - V[:, t]
        adv[:, t] = delta + gamma * lam * adv[:, t + 1] * sig
    ventajas = adv[tr["ep"], tr["t"]]
    return ventajas, ventajas + tr["value"].float()


# ─────────────────────────────────────────────────────────────────────────────
# Reanudación: estado completo del entrenamiento en un archivo
# ─────────────────────────────────────────────────────────────────────────────

def _guardar_atomico(obj, path: str) -> None:
    """torch.save a un temporal y después os.replace: un corte a mitad de la escritura no deja el archivo corrupto."""
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def _cargar_reanudacion(path: str) -> dict:
    # Archivo propio (no de terceros): weights_only=False para los estados de numpy y random y el historial.
    return torch.load(path, map_location="cpu", weights_only=False)


def tiempo_previo_reanudacion(path: str) -> float:
    """Segundos de entrenamiento ya acumulados en el archivo de reanudación (0.0 si no existe)."""
    if not path or not os.path.exists(path):
        return 0.0
    return float(_cargar_reanudacion(path).get("t_acumulado", 0.0))


def _diferencias_huella(guardada: dict, actual: dict) -> list:
    return [f"{k}: archivo={guardada.get(k)!r} ahora={actual.get(k)!r}"
            for k in sorted(set(guardada) | set(actual)) if guardada.get(k) != actual.get(k)]


# ─────────────────────────────────────────────────────────────────────────────
# Función principal de entrenamiento
# ─────────────────────────────────────────────────────────────────────────────

def run_am_training(
    time_matrix,
    distance_arr:   np.ndarray,
    diesel_arr:     np.ndarray,
    num_nodes:      int,
    market_process,                           # MarketProcess: cada episodio es un mundo simulado del proceso exógeno
    avail_expected_arr: np.ndarray = None,    # [num_nodes, num_nodes] E[loads] histórico (feature avail_prior)
    reward_global_p95: float      = 1.0,
    num_dias:       int = TRAIN_DAYS,         # días de entrenamiento (los mundos solo usan esos días)
    pretrained_agent:  AMRoutingAgent = None,
    pretrained_critic: CriticHead     = None,
    best_checkpoint_path: str = None,  # ruta donde se guarda el update con mejor margen de validación; None = no se guarda
    trial_callback: callable = None,   # HPO opcional: fn(step, n_updates, log_entry) -> bool (True = detener ya)
    est_lane: object = None,           # EstadisticasLane (MarketProcess.estadisticas_dow); None ⇒ se calcula aquí
    resume_path: str = None,           # archivo con el estado completo para reanudar; None = no se guarda
    reanudar: bool = False,            # True: si resume_path existe, sigue desde ahí; False: empieza de cero
) -> tuple:
    """
    Entrena AMRoutingAgent + CriticHead con PPO, todo en config.DEVICE.

    En cada update se sortean B = PPO_N_EPISODES_PER_UPDATE mundos (hub inicial, día real inicial, δ, cargas y
    competidores) y se recolecta un episodio en cada uno, en paralelo. `best_checkpoint_path` guarda
    {"agent", "critic"} del update con mejor margen en los episodios fijos de validación (ValidacionSimulada).

    Reanudación: con `resume_path`, cada RESUME_CADA_UPDATES updates o RESUME_CADA_MIN minutos (y al terminar) se
    guarda el estado completo (pesos, optimizadores, schedulers, update, selección del mejor checkpoint, historial y
    estados aleatorios de random, numpy, torch, CUDA y el generador de mundos). Con `reanudar=True` y el archivo
    presente, el entrenamiento sigue en el update siguiente al guardado; si ya había terminado, no entrena más. En CPU
    el resultado es idéntico al de una corrida sin cortes; en GPU, equivalente (algunos kernels de CUDA no son
    deterministas). Si la configuración del archivo no coincide con la actual, se detiene con un error.

    Retorna (agent, critic, episode_rewards (list), episode_losses (list), training_log (list de dicts)).
    """
    d_h, n_heads, n_layers, d_ff = AM_D_H, AM_N_HEADS, AM_N_LAYERS, AM_D_FF
    # hp(): lo definido en config.HP_POR_NODOS para este tamaño manda; si no cambia el valor,
    # se usa el global del módulo (que hpo_optuna.py sobreescribe por trial).
    n_episodes_per_update = hp("PPO_N_EPISODES_PER_UPDATE", PPO_N_EPISODES_PER_UPDATE)
    n_ppo_epochs          = PPO_N_EPOCHS
    ppo_batch_size        = PPO_BATCH_SIZE
    lr                    = hp("PPO_LR", PPO_LR)
    entropy_start         = hp("PPO_ENTROPY_COEF_START", PPO_ENTROPY_COEF_START)
    entropy_end           = hp("PPO_ENTROPY_COEF_END",   PPO_ENTROPY_COEF_END)
    lr_final_frac         = hp("LR_FINAL_FRAC", LR_FINAL_FRAC)
    gamma, gae_lambda, clip_eps, grad_clip = PPO_GAMMA, PPO_GAE_LAMBDA, PPO_CLIP_EPS, PPO_GRAD_CLIP
    entropy_precio_start  = hp("PPO_ENTROPY_COEF_PRECIO_START", PPO_ENTROPY_COEF_PRECIO_START)
    entropy_precio_end    = hp("PPO_ENTROPY_COEF_PRECIO_END",   PPO_ENTROPY_COEF_PRECIO_END)

    t_sesion0 = time.perf_counter()
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    n_updates      = get_n_updates(num_nodes)
    total_episodes = n_updates * n_episodes_per_update
    num_train_days = min(int(num_dias), market_process.num_dias_total)

    # ── Modelos ───────────────────────────────────────────────────────────────
    agent  = pretrained_agent  if pretrained_agent  is not None \
             else AMRoutingAgent(num_nodes, d_h, n_heads, n_layers, d_ff, device=DEVICE,
                                 pre_norm=AM_PRE_NORM, dropout=AM_DROPOUT)
    critic = pretrained_critic if pretrained_critic is not None else CriticHead(d_h).to(DEVICE)
    agent.to(DEVICE)
    critic.to(DEVICE)

    # Optimizers separados: el critic converge más rápido (lr 10x). Actor con AdamW (weight_decay es el regularizador
    # PPO-safe primario); critic con Adam puro.
    actor_params = list(agent.parameters())
    actor_optimizer  = torch.optim.AdamW(actor_params, lr=lr, weight_decay=PPO_WEIGHT_DECAY)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=lr * 10)

    # LR con decaimiento coseno: cada LR baja desde su valor inicial hasta LR_FINAL_FRAC de ese valor
    # al terminar n_updates (un scheduler.step() al final de cada update).
    schedulers = []
    if USE_LR_COSINE:
        def _cos(u, n=n_updates, f=lr_final_frac):
            return f + (1.0 - f) * 0.5 * (1.0 + math.cos(math.pi * min(u, n) / max(n, 1)))
        schedulers = [torch.optim.lr_scheduler.LambdaLR(o, _cos) for o in (actor_optimizer, critic_optimizer)]

    # ── Entorno, entradas fijas del agente y validación (una sola vez) ────────
    if est_lane is None and USE_CONTINUATION_FEATURES:
        est_lane = market_process.estadisticas_dow(distance_arr, diesel_arr)
    env = VectorRoutingEnv(time_matrix, distance_arr, diesel_arr, market_process, device=DEVICE)
    entradas = EntradasAgente.crear(env, avail_expected_arr, est_lane, reward_global_p95)
    val = ValidacionSimulada(market_process, env, entradas, num_train_days, num_nodes)
    gen_mundos = torch.Generator(device=DEVICE).manual_seed(MARKET_SEED_TRAIN)   # mundos de entrenamiento
    n_salida = dias_de_salida(num_train_days)

    print(f"\n--- Entrenamiento AM-PPO en {DEVICE}: {total_episodes} episodios "
          f"({n_updates} updates × {n_episodes_per_update} ep/update) ---")
    print("Mercado: realizaciones diarias — cada episodio sortea una vez hub inicial, día real inicial y "
          "corrimiento de nivel δ.")
    print(f"Validación: {val.M} episodios fijos (semilla {MARKET_SEED_VAL}) × K∈"
          f"{{{', '.join('libre' if K is None else str(K) for K in val.niveles)}}}")

    best_score = -np.inf
    ventana = deque(maxlen=CKPT_WINDOW)     # selección por media móvil: pesos (CPU) y score de las últimas validaciones
    best_ma = -np.inf
    ckpt_update = ckpt_score = None
    best_pesos = None                       # {"agent", "critic"} (CPU) del mejor checkpoint: va en la reanudación
    episode_rewards, episode_losses, training_log = [], [], []
    fases_acum, updates_acum = defaultdict(float), 0

    # ── Reanudación ───────────────────────────────────────────────────────────
    # Huella: lo que tiene que coincidir para que seguir un entrenamiento guardado sea válido.
    huella = {
        "num_nodes": int(num_nodes), "n_updates": int(n_updates), "episodios_por_update": int(n_episodes_per_update),
        "lr": float(lr), "entropy": (float(entropy_start), float(entropy_end)),
        "entropy_precio": (float(entropy_precio_start), float(entropy_precio_end)),
        "lr_final_frac": float(lr_final_frac), "lr_coseno": bool(USE_LR_COSINE),
        "self_critical": bool(USE_SELF_CRITICAL), "ckpt": (CKPT_SELECTION, int(CKPT_WINDOW)),
        "k_niveles": ["libre" if k is None else int(k) for k in K_TRAINING_LEVELS],
        "seed": int(SEED), "market_seed_train": int(MARKET_SEED_TRAIN), "num_train_days": int(num_train_days),
        "device": DEVICE.type, "arquitectura": {k: tuple(v.shape) for k, v in agent.state_dict().items()},
    }
    t_previo = 0.0                # segundos de sesiones anteriores
    start_update = 0
    if resume_path and reanudar and os.path.exists(resume_path):
        st = _cargar_reanudacion(resume_path)
        dif = _diferencias_huella(st["huella"], huella)
        if dif:
            raise ValueError(f"{resume_path} es de un entrenamiento con otra configuración; no se puede reanudar.\n  "
                             + "\n  ".join(dif) + "\nBórralo o corre sin RESUME=1 para empezar de cero.")
        agent.load_state_dict(st["agent"])
        critic.load_state_dict(st["critic"])
        actor_optimizer.load_state_dict(st["actor_optimizer"])
        critic_optimizer.load_state_dict(st["critic_optimizer"])
        for sch, s in zip(schedulers, st["schedulers"]):
            sch.load_state_dict(s)
        start_update = int(st["update"])
        best_score, best_ma = st["best_score"], st["best_ma"]
        ckpt_update, ckpt_score = st["ckpt_update"], st["ckpt_score"]
        ventana.extend(st["ventana"])
        best_pesos = st["best_pesos"]
        episode_rewards = np.asarray(st["episode_rewards"], dtype=np.float64).tolist()
        episode_losses, training_log = list(st["episode_losses"]), list(st["training_log"])
        t_previo = float(st["t_acumulado"])
        random.setstate(st["rng"]["python"])
        np.random.set_state(st["rng"]["numpy"])
        torch.set_rng_state(st["rng"]["torch"].cpu())
        if st["rng"]["cuda"] is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state(st["rng"]["cuda"].cpu())
        gen_mundos.set_state(st["rng"]["gen_mundos"].cpu())
        if best_pesos is not None and best_checkpoint_path:      # por si el _best.pt no viajó con la reanudación
            torch.save(best_pesos, best_checkpoint_path)
        print(f"Reanudando desde {resume_path}: {start_update}/{n_updates} updates hechos "
              f"({t_previo / 3600:.2f} h de entrenamiento previo).")
        del st
    elif resume_path and os.path.exists(resume_path):
        print(f"Aviso: {resume_path} existe pero no se reanuda (RESUME=1 no está activo): se entrena desde cero "
              f"y se sobrescribirá.")

    def _guardar_reanudacion(siguiente_update: int) -> None:
        cpu = lambda sd: {k: t.detach().cpu().clone() for k, t in sd.items()}
        _guardar_atomico({
            "huella": huella, "update": siguiente_update, "completo": siguiente_update >= n_updates,
            "agent": cpu(agent.state_dict()), "critic": cpu(critic.state_dict()),
            "actor_optimizer": actor_optimizer.state_dict(), "critic_optimizer": critic_optimizer.state_dict(),
            "schedulers": [s.state_dict() for s in schedulers],
            "best_score": best_score, "best_ma": best_ma, "ckpt_update": ckpt_update, "ckpt_score": ckpt_score,
            "ventana": list(ventana), "best_pesos": best_pesos,
            "episode_rewards": np.asarray(episode_rewards, dtype=np.float32),
            "episode_losses": episode_losses, "training_log": training_log,
            "t_acumulado": t_previo + (time.perf_counter() - t_sesion0),
            "rng": {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
                    "cuda": torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
                    "gen_mundos": gen_mundos.get_state()},
        }, resume_path)

    t_ultimo_guardado = time.perf_counter()

    # ── Loop principal ────────────────────────────────────────────────────────
    for update in range(start_update, n_updates):
        t_update0 = time.perf_counter()
        fases     = _Fases()
        progress  = update / max(1, n_updates - 1)
        entropy_coef = entropy_start + (entropy_end - entropy_start) * progress
        entropy_coef_precio = entropy_precio_start + (entropy_precio_end - entropy_precio_start) * progress
        B = n_episodes_per_update

        # ── Fase 1: mundos y recolección (B episodios en paralelo) ────────────
        with fases("mundos"):
            # Nodos de inicio balanceados: cada nodo aparece ⌊B/N⌋ veces y el resto al azar.
            start_nodes = np.repeat(np.arange(num_nodes), B // num_nodes)
            if B % num_nodes:
                start_nodes = np.concatenate([start_nodes, np.random.choice(num_nodes, size=B % num_nodes, replace=False)])
            start_nodes = np.random.permutation(start_nodes).astype(np.int64)
            start_days = np.random.randint(0, n_salida, size=B).astype(np.int64)
            k_list = _balanced_k_assignment(B, np.random)
            mundos = market_process.sortear_mundos(start_days, MARKET_DIAS_EPISODIO, num_train_days - 1, gen_mundos)

        agent.eval()
        critic.eval()
        with fases("rollout"), torch.no_grad():
            tr = _rollout_muestreado(agent, critic, env, mundos, start_nodes, k_list, entradas)

        # ── Self-critical: rollout greedy sobre los MISMOS mundos, nodos y K ───
        r_greedy = None
        if USE_SELF_CRITICAL:
            with fases("greedy"):
                r_greedy = greedy_rollout(agent, env, mundos, start_nodes, entradas, k_max=k_list).env_reward

        with fases("gae"):
            ventajas, retornos = _gae(tr, B, gamma, gae_lambda)
            ep, dev = tr["ep"], tr["reward"].device
            n_tr = ep.numel()
            r_ep = torch.zeros(B, device=dev).index_add_(0, ep, tr["reward"].float())
            n_pasos_ep = torch.zeros(B, dtype=torch.long, device=dev).index_add_(0, ep, torch.ones_like(ep))
            con_tr = n_pasos_ep > 0                        # episodios con al menos una transición
            sc_ep = None
            if USE_SELF_CRITICAL:
                sc_raw = (r_ep - r_greedy)[con_tr]
                sc_norm = torch.zeros(B, device=dev)
                sc_norm[con_tr] = (sc_raw - sc_raw.mean()) / (sc_raw.std(unbiased=False) + 1e-8)
                sc_ep = sc_norm[ep]                        # ventaja self-critical de cada transición

        # ── Fase 2: actualización PPO ─────────────────────────────────────────
        agent.train()
        critic.train()
        stats = defaultdict(list)
        apply_self_critical = USE_SELF_CRITICAL and update >= SELF_CRITICAL_WARMUP_UPDATES
        with fases("ppo"):
            kl_too_high = False
            for _ in range(n_ppo_epochs):
                perm = torch.randperm(n_tr, device=dev)
                for ini in range(0, n_tr, ppo_batch_size):
                    idx = perm[ini:ini + ppo_batch_size]
                    feats_b = {k: tr[k][idx] for k in _FEATS + ("opciones", "current_node")}
                    old_lp, ret_b = tr["log_prob"][idx], retornos[idx]
                    adv_b = ventajas[idx]
                    # Normalización por mini-batch (media 0, std 1) y recorte de ventajas extremas.
                    adv_b = ((adv_b - adv_b.mean()) / (adv_b.std(unbiased=False) + 1e-8)).clamp(-5.0, 5.0)

                    _, _, new_lp, entropy, entropy_precio, new_val = _forward(
                        agent, critic, feats_b, tr["mask"][idx], actions_t=tr["action"][idx], opcion_t=tr["opcion"][idx])
                    ratio = torch.exp(new_lp - old_lp)
                    actor_loss = -torch.min(ratio * adv_b, ratio.clamp(1 - clip_eps, 1 + clip_eps) * adv_b).mean()
                    value_loss = nn.functional.mse_loss(new_val, ret_b)
                    # Entropía del destino y de la cabeza de precio, cada una con su coeficiente.
                    actor_loss_total = (actor_loss - entropy_coef * entropy.mean()
                                        - entropy_coef_precio * entropy_precio.mean())
                    if apply_self_critical:          # término REINFORCE adicional (solo actor)
                        sc_loss = -(SELF_CRITICAL_COEF * sc_ep[idx] * new_lp).mean()
                        actor_loss_total = actor_loss_total + sc_loss

                    # value_loss también backpropaga hacia el encoder/context_net (h_t no se detacha): un solo
                    # backward sobre la suma da los mismos gradientes que dos por separado.
                    actor_optimizer.zero_grad(set_to_none=True)
                    critic_optimizer.zero_grad(set_to_none=True)
                    (actor_loss_total + value_loss).backward()
                    nn.utils.clip_grad_norm_(actor_params, grad_clip)
                    actor_optimizer.step()
                    nn.utils.clip_grad_norm_(critic.parameters(), grad_clip)
                    critic_optimizer.step()

                    with torch.no_grad():
                        fila = torch.stack([(old_lp - new_lp).mean(), ((ratio - 1).abs() > clip_eps).float().mean(),
                                            actor_loss_total.detach(), actor_loss.detach(), value_loss.detach(),
                                            entropy.mean(), entropy_precio.mean(),
                                            sc_loss.detach() if apply_self_critical else torch.zeros((), device=dev)])
                    kl, clip_frac, tot, pol, vl, ent, ent_p, scl = fila.tolist()   # una sincronización por minibatch
                    stats["kl"].append(kl)
                    stats["clip"].append(clip_frac)
                    if kl > 0.05:
                        kl_too_high = True
                        break
                    for k, x in (("total", tot), ("pol", pol), ("val", vl), ("ent", ent), ("ent_p", ent_p), ("sc", scl)):
                        stats[k].append(x)
                if kl_too_high:   # sale también de n_ppo_epochs
                    break

        # ── Diagnósticos del update (en el dispositivo; a la CPU solo los escalares) ──
        with torch.no_grad():
            ret_all, val_all = retornos, tr["value"].float()
            var_ret = ret_all.var(unbiased=False)
            explained_var = float(1.0 - (ret_all - val_all).var(unbiased=False) / (var_ret + 1e-8)) if var_ret > 1e-8 else 0.0
            r_ep_c = r_ep[con_tr]
            n_ep = int(con_tr.sum())
            update_ep_rewards = r_ep_c.cpu().tolist()
            largo = torch.zeros(B, dtype=torch.long, device=dev).index_add_(0, ep, tr["viajo"].long())
            ult = torch.zeros(B, dtype=torch.long, device=dev).scatter_reduce_(0, ep, torch.arange(n_tr, device=dev),
                                                                                "amax", include_self=False)
            viol_ep = tr["time_violation"][ult] & con_tr
            k_cod = torch.as_tensor(_k_codigo(k_list), device=dev)
            n_cot = int(tr["cotizo"].sum())
            n_gan = int(tr["gano"].sum())
            nivel_medio = float(tr["opcion"][tr["cotizo"]].float().mean()) if n_cot else float("nan")
            k_diagnostics = {}
            for lvl in K_TRAINING_LEVELS:
                sel = con_tr & (k_cod == (0 if lvl is None else int(lvl)))
                key = "free" if lvl is None else str(lvl)
                if not bool(sel.any()):
                    k_diagnostics[key] = None
                    continue
                lens = largo[sel].cpu().numpy()
                entry = {"n_episodes": int(len(lens)), "mean_length": float(lens.mean()),
                         "p50_length": float(np.percentile(lens, 50)), "p90_length": float(np.percentile(lens, 90)),
                         "max_length": int(lens.max()), "time_violation_rate": float(viol_ep[sel].float().mean())}
                if lvl is not None:
                    entry["conformity_rate"] = float((lens <= lvl).mean())
                k_diagnostics[key] = entry

        avg_reward = float(np.mean(update_ep_rewards)) if update_ep_rewards else 0.0
        episode_rewards.extend(update_ep_rewards)
        media = lambda k: float(np.mean(stats[k])) if stats[k] else 0.0
        episode_losses.append(media("total"))

        log_entry = {
            "update": update + 1, "reward": avg_reward, "total_loss": media("total"), "policy_loss": media("pol"),
            "value_loss": media("val"), "entropy": media("ent"), "kl_divergence": media("kl"),
            "clip_fraction": media("clip"), "explained_var": explained_var, "entropy_coef": entropy_coef,
            "k_diagnostics": k_diagnostics,
        }
        if USE_SELF_CRITICAL:
            sc_raw_mean = float((r_ep - r_greedy)[con_tr].mean()) if n_ep else 0.0
            log_entry["r_greedy_avg"] = float(r_greedy[con_tr].mean()) if n_ep else 0.0
            log_entry["sc_advantage_mean"] = sc_raw_mean
            log_entry["sc_loss_value"] = media("sc")
        # Del rollout muestreado: cuántas cotizaciones se ganan, en qué nivel se cotiza y cuántas por episodio.
        log_entry["entropy_precio"]      = media("ent_p")
        log_entry["entropy_coef_precio"] = entropy_coef_precio
        log_entry["tasa_aceptacion"]     = n_gan / n_cot if n_cot else float("nan")
        log_entry["nivel_medio"]         = nivel_medio
        log_entry["cotizaciones_por_ep"] = n_cot / max(n_ep, 1)
        log_entry["rechazos_por_ep"]     = (n_cot - n_gan) / max(n_ep, 1)

        stop_now = False
        log_freq = max(1, n_updates // 20)
        val_freq = max(1, n_updates // VAL_CHECKS)
        es_log = (update + 1) % log_freq == 0

        with fases("validacion"):
            # Validación con episodios fijos (observacional); guarda el mejor update.
            v = None
            if es_log or (update + 1) % val_freq == 0 or update == n_updates - 1:
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
                        best_pesos = {"agent": centro["agent"], "critic": centro["critic"]}
                        if best_checkpoint_path:
                            torch.save(best_pesos, best_checkpoint_path)
                    log_entry["val_ma"] = ma
                    log_entry["ckpt_update"] = ckpt_update
                    log_entry["ckpt_score"] = ckpt_score
                elif v["score"] > best_score:
                    best_score = v["score"]
                    log_entry["val_best"] = True
                    ckpt_update, ckpt_score = update + 1, v["score"]
                    best_pesos = {"agent":  {k: t.detach().cpu().clone() for k, t in agent.state_dict().items()},
                                  "critic": {k: t.detach().cpu().clone() for k, t in critic.state_dict().items()}}
                    if best_checkpoint_path:
                        torch.save(best_pesos, best_checkpoint_path)
                if CKPT_SELECTION != "moving_avg":
                    log_entry["ckpt_update"] = ckpt_update
                    log_entry["ckpt_score"] = ckpt_score

        if es_log:
            greedy_valid_rate = v["valid_rate"]
            episodes_done = (update + 1) * n_episodes_per_update
            print(f"  update {update+1:>5}/{n_updates} | episodios {episodes_done:>8}/{total_episodes} "
                  f"| reward_train {avg_reward:6.1f}")
            val_linea = " ; ".join(
                f"val[K={'libre' if k == 'free' else k}] margen={d['margen']:.0f}usd "
                f"acept={d['tasa_aceptacion']:.0%} nivel={d['nivel_medio']:.1f}" for k, d in v["por_k"].items())
            print(f"    {val_linea}")
            if log_entry.get("val_ma") is not None:
                ckpt = (f" *checkpoint guardado (mejor update={log_entry['ckpt_update']})"
                        if log_entry.get("val_best") else "")
                print(f"    MA({CKPT_WINDOW} val)={log_entry['val_ma']:.0f}usd{ckpt}")
            print(f"    validez={greedy_valid_rate:.0%} | entropía destino={log_entry['entropy']:.3f} "
                  f"| KL={log_entry['kl_divergence']:.4f} | var_explicada={explained_var:.3f}")
            print(f"    cotización (rollout): aceptación={log_entry['tasa_aceptacion']:.0%} "
                  f"| nivel medio={log_entry['nivel_medio']:.1f} "
                  f"| cotizaciones/ep={log_entry['cotizaciones_por_ep']:.2f} "
                  f"| rechazos/ep={log_entry['rechazos_por_ep']:.2f} "
                  f"| entropía precio={log_entry['entropy_precio']:.3f}")
            partes = []
            for lvl in K_TRAINING_LEVELS:
                d = k_diagnostics.get("free" if lvl is None else str(lvl))
                if d is None:
                    continue
                partes.append(f"free:len={d['mean_length']:.1f}" if lvl is None else
                              f"K={lvl}:conf={d['conformity_rate']:.2f},len={d['mean_length']:.1f}/"
                              f"{d['p90_length']:.0f},tviol={d['time_violation_rate']:.2f}")
            print("    K-diag: " + " | ".join(partes))
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
        log_entry["t_otro"] = t_total - sum(fases.t.values())
        log_entry["t_total"] = t_total
        fases_acum["otro"] += log_entry["t_otro"]
        fases_acum["total"] += t_total
        updates_acum += 1
        if es_log:
            partes = " | ".join(
                f"{n} {fases_acum[n] / updates_acum:.2f}s ({100 * fases_acum[n] / fases_acum['total']:.0f}%)"
                for n in (*_FASES_ORDEN, "otro") if fases_acum[n] > 0)
            print(f"    tiempos/update ({updates_acum} upd): total {fases_acum['total'] / updates_acum:.2f}s | {partes}")
            fases_acum.clear()
            updates_acum = 0

        training_log.append(log_entry)
        if resume_path and ((update + 1) % RESUME_CADA_UPDATES == 0 or update == n_updates - 1 or stop_now
                            or time.perf_counter() - t_ultimo_guardado >= 60.0 * RESUME_CADA_MIN):
            _guardar_reanudacion(update + 1)
            t_ultimo_guardado = time.perf_counter()
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
