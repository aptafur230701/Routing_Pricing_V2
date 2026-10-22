"""
config.py
=========
All global constants, toggles, and hyperparameter defaults for the AM-PPO pipeline.
This file is imported by all modules, so it should not contain any heavy dependencies.

"""

import os

import torch

# ── Problem identity ──────────────────────────────────────────
MPG             = 6.5
MARGINAL_COST_SIN_DIESEL = 1.73

# ── Environment constraints ───────────────────────────────────
DURATION_LIMIT      = 70.0
DURATION_TOLERANCE  = 0.10
MAX_DURATION        = DURATION_LIMIT * (1 + DURATION_TOLERANCE)

# ── Reward shaping ────────────────────────────────────────────
REWARD_SCALE_FACTOR    = 100.0
RETURN_SUCCESS_BONUS   =  500.0 / REWARD_SCALE_FACTOR
TIME_VIOLATION_PENALTY =  -3000.0 / REWARD_SCALE_FACTOR
BIG_M_PENALTY          = -1e9
USE_REWARD_SHAPING     = False   # True: el retorno exitoso suma RETURN_SUCCESS_BONUS + bonos por arcos antes del
                                 # retorno (step_count) y time_util (shaping que no es margen). False: terminal_reward = 0 en el
                                 # retorno exitoso, así que el reward del episodio es margen / REWARD_SCALE_FACTOR.
                                 # TIME_VIOLATION_PENALTY se aplica en ambos casos. Ablacionable.

# ── Evaluation ────────────────────────────────────────────────
N_EVAL_EPISODES = 50

# ── AM Model architecture ─────────────────────────────────────
AM_D_H      = 192  # Dimensión de embeddings del Transformer. Subido de 128: la política greedy
                    # saturaba en una meseta ~update 300-400 con crítico sano, indicando techo
                    # de capacidad de representación a 35 nodos, no inestabilidad de PPO.
AM_N_HEADS  = 8    # Número de cabezas de atención en el Transformer (192/8=24, exacto)
AM_N_LAYERS = 4    # Número de capas del encoder Transformer. Subido de 3 junto con AM_D_H.
AM_D_FF     = 768  # Dimensión de la capa feed-forward. Mantiene la proporción 4×AM_D_H.
AM_PRE_NORM = True  # Pre-norm (True) vs post-norm (False) en TransformerEncoderLayer.
                    # Ablacionable: False reproduce el comportamiento post-norm anterior.
AM_DROPOUT  = 0.0   # Dropout en las sublayers del encoder (nunca en el path de logits del
                    # decoder). Ablacionable: probar 0.05 / 0.1 con cuidado — en PPO el
                    # dropout puede corromper el ratio de importancia si old/new log_probs
                    # se calculan con distinto modo train/eval.

# ── PPO hyperparameters ───────────────────────────────────────
PPO_N_EPISODES_PER_UPDATE = 700    # Episodios recolectados antes de cada update PPO
PPO_N_EPOCHS              = 4      # Épocas de entrenamiento PPO por rollout
PPO_BATCH_SIZE            = 256    # Tamaño de batch para entrenamiento PPO. Subido de 64: la GPU
                                    # está dominada por overhead de lanzamiento de kernels a batch
                                    # pequeño (tiempo/batch ~constante de 64 a 700), así que un batch
                                    # mayor da ~3.4x más rápido por update con caída mínima de EV.
PPO_LR                    = 7.574781400016766e-05   # Learning rate para el actor. Subido de 1e-5: la KL media
                                    # iba muy por debajo del target 0.02 con clip < 0.10,
                                    # indicando under-stepping. knob: subir hacia 5e-5 si KL se
                                    # mantiene < ~0.015 y el reward sigue subiendo pasado update 150.
PPO_GAMMA                 = 1.0    # Factor de descuento. 1.0: horizonte finito de 2–5 pasos, sin descuento
                                    # (el margen del episodio es la suma de las lanes). Era 0.99.
PPO_GAE_LAMBDA            = 0.9738083627173082   # Factor de GAE
PPO_CLIP_EPS              = 0.22021368918814568   # Clip PPO para limitar cambios de política
PPO_ENTROPY_COEF_START    = 0.07842497176858756   # coeficiente de entropía al inicio del entrenamiento
PPO_ENTROPY_COEF_END      = 0.022646509161679096   # coeficiente de entropía al final del entrenamiento (decay lineal)
PPO_GRAD_CLIP             = 1.7476269465868015    # Clipping de gradiente para PPO
PPO_WEIGHT_DECAY          = 6.093123878168306e-05  # Weight decay (AdamW) del actor. Regularizador PRIMARIO
                                    # PPO-safe: no corrompe el ratio de importancia como el
                                    # dropout en el path de logits.

# ── Precio de reserva (PRICING_PLAN.md) ───────────────────────
# Cabeza de valor post-decisión Ṽ(s, j) sobre el estado congelado (h_t y embeddings con detach):
# no cambia la política ni el PPO. Con ella se calcula p*, el precio mínimo por milla al que
# conviene aceptar una lane (pricing.py).
USE_POST_HEAD = True      # activa la cabeza de valor post-decisión
POST_HIDDEN   = AM_D_H    # tamaño oculto del MLP (igual al encoder)
POST_COEF     = 1.0       # peso de loss_post. Con detach() no afecta a la política, así que 1.0 es seguro.
                          # Bajar solo si post_mse explota.
PRICING_VAL_SEED = 20261022   # semilla del muestreo de validate_pricing.py (estados de Monte Carlo, ofertas, azar de "acepto siempre")

# ── Self-critical baseline (híbrido sobre actor-critic) ────────────
USE_SELF_CRITICAL            = True   # término self-critical sobre el argmax greedy
SELF_CRITICAL_COEF           = 0.5     # peso del término self-critical en el loss total
SELF_CRITICAL_WARMUP_UPDATES = 50      # no aplicar self-critical hasta este update
                                        # (deja que el crítico se estabilice primero)

# ── Selección del mejor checkpoint (am_training.py) ───────────
# "max": guarda el update con el mayor score puntual de validación (comportamiento anterior).
# "moving_avg": guarda los pesos del update CENTRAL de la ventana de CKPT_WINDOW validaciones
# consecutivas cuya media de score es la mejor hasta ahora (menos ruido, menos selección por azar).
CKPT_SELECTION = "moving_avg"   # "moving_avg" | "max"
CKPT_WINDOW    = 3

# ── Learning rate con decaimiento coseno ──────────────────────
USE_LR_COSINE  = True   # True: el LR de actor y critic baja con coseno desde su valor inicial hasta
                        # LR_FINAL_FRAC de ese valor al terminar n_updates (un scheduler.step() por update).
LR_FINAL_FRAC  = 0.10

# ── Cardinalidad de ruta (K) ───────────────────────────────────
# K_MAX es el único techo restringido que se entrena — la restricción real de
# la operación. Formulación ≤K: con un solo nivel, el caso libre
# (K_TRAINING_LEVELS incluye None) sigue siendo necesario — representa la
# generalidad del método (el modelo resuelve la familia general del problema,
# sin presupuesto de cardinalidad), mientras que K_MAX es la instancia puntual
# que sirve a la operación real.
K_MAX  = 5    # techo de cardinalidad real de negocio (nº de cargas por tour)
K_NORM = K_MAX  # normalizador GLOBAL de k_remaining_norm: va de 1.0 (presupuesto
                # completo) a 1/K_NORM (último arco antes del retorno forzado).
K_LEVELS          = [K_MAX]              # un solo nivel restringido — la constraint real
K_TRAINING_LEVELS = K_LEVELS + [None]    # + caso libre (None) — 2 niveles en total


def configurar_k(restringido: bool) -> None:
    """Fija los niveles de K del entrenamiento, la validación y la evaluación.

    restringido=True : K_TRAINING_LEVELS = [K_MAX, None] (restringido + libre).
    restringido=False: K_TRAINING_LEVELS = [None] (solo caso libre; K_LEVELS queda vacío).

    Muta las listas en el sitio, así que los módulos que ya hicieron
    `from config import K_TRAINING_LEVELS` ven el cambio. Llamar desde main() antes de
    entrenar o evaluar. K_NORM no cambia (el modelo conserva la misma arquitectura/features).
    """
    K_LEVELS[:] = [K_MAX] if restringido else []
    K_TRAINING_LEVELS[:] = K_LEVELS + [None]

# ── Features de continuación por nodo ────────────────────────
USE_CONTINUATION_FEATURES = True   # True: 4 features extra por nodo j (N_NODE_FEATURES = 11) con la
                                   # información de qué pasa después de j (tiempo y holgura de regreso a
                                   # casa, valor esperado de volver a casa y de seguir un salto más; ver
                                   # attention_encoder.py). False: las 7 features base. Ablacionable.

# ── Train / eval split ───────────────────────────────────────
TRAIN_DAYS = 90   # días usados para entrenamiento; el resto (días 90-119) queda para evaluación

# ── Market signal normalisation clamps ───────────────────────
LTR_CLIP           = 200.0   # p95 del LTR es 124; 200 es techo seguro
TRUCKS_CLIP        =  50.0   # p95 de trucks delta=1 es 45
LOADS_EXPECTED_CLIP = 100.0  # p95 de E[loads] por lane (train, excl. diagonal) es ~66; 100 es techo seguro

# ── Reproducibility ───────────────────────────────────────────
SEED = int(os.environ.get("SEED", 42))   # torch/numpy del entrenamiento; variable de entorno SEED para variar por corrida

# ── Proceso exógeno de cargas y tarifas (market_process) ─────
# Ver proceso_exogeno_cargas_tarifas.md. Ningún índice de día va escrito a
# mano: el número de días sale de la forma de loads_stack / rate_stack y las
# fechas salen de FECHA_INICIO + índice del día.
FECHA_INICIO         = "2026-01-01"   # fecha del día 0 del histórico (jueves)
DIAS_EXCLUIDOS       = []             # fechas ("AAAA-MM-DD") a dejar fuera del ajuste, solo si
                                      # la fuente confirma que son erróneas. Vacía = usa todos los días de ajuste.
MARKET_PHI_MAX       = 1000.0         # tope de la dispersión phi (phi → ∞ equivale a Poisson)
MARKET_N_GRUPOS      = 4              # grupos de volumen h (cuartiles de mu_ij) para estimar phi_h
MARKET_PARAMS_PATH   = "datos/market_params.npz"   # relativo a la raíz del proyecto
MARKET_SEED_TRAIN    = int(os.environ.get("MARKET_SEED_TRAIN", 20260))   # semilla de los mundos de entrenamiento
                                                                          # (variable de entorno para variarla por corrida)
MARKET_SEED_EVAL     = 777001  # semilla base de los mundos de evaluación (distinta de la de entrenamiento)
MARKET_SEED_EVAL_FINAL = 777003  # semilla de un segundo conjunto de mundos de evaluación (conjunto B, test final):
                                  # distinta de entrenamiento, validación y MARKET_SEED_EVAL. MARKET_SEED_EVAL es el
                                  # conjunto de desarrollo (con él se decidieron cosas, p. ej. el checkpoint); el B se
                                  # usa UNA vez, con la configuración congelada. main.py: EVAL_SET=final.
MARKET_M_EVAL_WORLDS = 1000    # M: nº de mundos de evaluación simulados por nivel de K (eran 100). Con
                                # MARKET_SEED_EVAL fija, todas las corridas y baselines ven los MISMOS mundos.

# ── Validación durante el entrenamiento (mundos simulados) ───
# Mundos fijos con semilla propia (distinta de entrenamiento y de MARKET_SEED_EVAL, para que el
# reporte final no use mundos con los que se eligió el checkpoint). Se evalúa el greedy del
# agente en ellos y se compara con B3 / UB-DP; el mejor update por margen se guarda aparte.
MARKET_SEED_VAL = 777002
VAL_CHECKS      = 60      # nº aproximado de chequeos de validación por entrenamiento


def get_val_worlds(num_nodes: int) -> int:
    """Mundos de validación: 200 hasta 35 nodos, 300 desde 50 (≥ 3 mundos por nodo de inicio)."""
    return 200 if num_nodes < 50 else 300


# ── Baselines (baselines_drl_ruteo_estocastico.md) ───────────
# UB-DP / B3: programación dinámica sobre (hogar, día de salida, nodo, franja, k). El camión puede
# revisitar nodos, así que el estado no lleva visitados. La tabla V se guarda en disco y se recarga
# si ya existe y coincide.
UBDP_DT_HORAS   = 0.25    # paso de tiempo de la DP (franjas de 15 min; tiempos de viaje redondeados hacia abajo);
                          # también lo usa B2
UBPI_DT_HORAS   = 1.0 / 60.0   # paso de tiempo de la DP hacia atrás de UB-PI (franjas de 1 min; redondeo hacia abajo)
UBDP_S          = 500     # sorteos de filas por par (nodo, día de la semana)
UBDP_SEED       = 20261018   # semilla del Generator propio de la DP
UBDP_TABLA_PATH = "datos/ubdp_tabla_{n}nodos.npz"   # relativo a la raíz del proyecto; {n} = nº de nodos
REPLAY_MAX_CASOS = 300   # tope de casos (nodo, día) del replay por nivel de K; si hay más, se toma un subconjunto balanceado
BOOTSTRAP_B     = 10000   # remuestreos del intervalo bootstrap de la diferencia pareada
BOOTSTRAP_SEED  = 20261019

# ── Etiqueta de corrida ──────────────────────────────────────
# Vacía por defecto (results/<N>nodes y checkpoints/ de siempre). Con RUN_TAG="abl_a" main.py usa
# results/<N>nodes_abl_a y checkpoints/abl_a, para correr ablaciones sin pisar otras corridas.
RUN_TAG = os.environ.get("RUN_TAG", "")

# ── Hiperparámetros por tamaño de grafo ──────────────────────
# Lo que se escriba aquí para un tamaño pisa el valor global de este archivo; {} = los de siempre.
# Claves admitidas: PPO_LR, PPO_N_EPISODES_PER_UPDATE, PPO_ENTROPY_COEF_START, PPO_ENTROPY_COEF_END,
# LR_FINAL_FRAC y EPISODES_PER_NODE (reemplaza a get_episodes_per_node).
HP_POR_NODOS = {
    10: {},
    20: {"PPO_ENTROPY_COEF_START": 0.10, "PPO_ENTROPY_COEF_END": 0.04},   # más exploración al final
    35: {"EPISODES_PER_NODE": 60000, "LR_FINAL_FRAC": 0.30},              # el doble de episodios, piso de LR más alto
}
HP_OVERRIDES = {}   # se llena en configurar_hp()


def configurar_hp(num_nodes: int) -> None:
    """Carga HP_POR_NODOS[num_nodes] en HP_OVERRIDES (en el sitio). Llamar desde main()."""
    HP_OVERRIDES.clear()
    HP_OVERRIDES.update(HP_POR_NODOS.get(num_nodes, {}))


def hp(nombre: str, default):
    """Valor de la variante activa para `nombre`, o `default` si la variante no lo cambia."""
    return HP_OVERRIDES.get(nombre, default)

# ── Device ───────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_episodes_per_node(num_nodes: int) -> int:
    """Episodes per start-node for the full training run."""
    if "EPISODES_PER_NODE" in HP_OVERRIDES:      # hiperparámetros propios del tamaño (HP_POR_NODOS)
        return HP_OVERRIDES["EPISODES_PER_NODE"]
    if num_nodes <= 10:  return 20000  # subido de 5000: a 5000 (71 updates) reward/entropy
                                        # no se aplanaban; 20000 da ~285 updates de margen
    if num_nodes <= 20:  return 25000
    if num_nodes <= 35:  return 30000   # bajado de 25000: la política saturaba ~update 400;
                                        # con ~700 ep/update da ~500-600 updates, holgura sobre
                                        # el punto de saturación observado
    if num_nodes <= 50:  return 60000
    if num_nodes <= 75:  return 38000
    return 60000          # 100 nodos

