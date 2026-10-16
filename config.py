"""
config.py
=========
All global constants, toggles, and hyperparameter defaults for the AM-PPO pipeline.
This file is imported by all modules, so it should not contain any heavy dependencies.

"""

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
PPO_GAMMA                 = 0.99   # Factor de descuento para las recompensas futuras
PPO_GAE_LAMBDA            = 0.9738083627173082   # Factor de GAE
PPO_CLIP_EPS              = 0.22021368918814568   # Clip PPO para limitar cambios de política
PPO_ENTROPY_COEF_START    = 0.07842497176858756   # coeficiente de entropía al inicio del entrenamiento
PPO_ENTROPY_COEF_END      = 0.022646509161679096   # coeficiente de entropía al final del entrenamiento (decay lineal)
PPO_GRAD_CLIP             = 1.7476269465868015    # Clipping de gradiente para PPO
PPO_WEIGHT_DECAY          = 6.093123878168306e-05  # Weight decay (AdamW) del actor. Regularizador PRIMARIO
                                    # PPO-safe: no corrompe el ratio de importancia como el
                                    # dropout en el path de logits.

# ── Self-critical baseline (híbrido sobre actor-critic) ────────────
USE_SELF_CRITICAL            = True   # término self-critical sobre el argmax greedy
SELF_CRITICAL_COEF           = 0.5     # peso del término self-critical en el loss total
SELF_CRITICAL_WARMUP_UPDATES = 50      # no aplicar self-critical hasta este update
                                        # (deja que el crítico se estabilice primero)

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

# ── Train / eval split ───────────────────────────────────────
TRAIN_DAYS = 90   # días usados para entrenamiento; el resto (días 90-119) queda para evaluación

# ── Market signal normalisation clamps ───────────────────────
LTR_CLIP           = 200.0   # p95 del LTR es 124; 200 es techo seguro
TRUCKS_CLIP        =  50.0   # p95 de trucks delta=1 es 45
LOADS_EXPECTED_CLIP = 100.0  # p95 de E[loads] por lane (train, excl. diagonal) es ~66; 100 es techo seguro

# ── Reproducibility ───────────────────────────────────────────
SEED = 42

# ── Device ───────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_episodes_per_node(num_nodes: int) -> int:
    """Episodes per start-node for the full training run."""
    if num_nodes <= 10:  return 20000  # subido de 5000: a 5000 (71 updates) reward/entropy
                                        # no se aplanaban; 20000 da ~285 updates de margen
    if num_nodes <= 20:  return 25000
    if num_nodes <= 35:  return 30000   # bajado de 25000: la política saturaba ~update 400;
                                        # con ~700 ep/update da ~500-600 updates, holgura sobre
                                        # el punto de saturación observado
    if num_nodes <= 50:  return 32000
    if num_nodes <= 75:  return 38000
    return 43000          # 100 nodos


def get_beam_width_det(num_nodes: int) -> int:
    """Beam width for inference, scaled by problem size.
    Modelos pequeños necesitan más exploración en inferencia.
    Modelos grandes tienen políticas más robustas y beam=1 es suficiente."""
    if num_nodes <= 10: return 5    # modelo pequeño, necesita más exploración
    if num_nodes <= 20: return 7   # validado empíricamente: gap bajó de ~20-29% a ~14-16% vs beam=5
    if num_nodes <= 35: return 7   # balance costo/calidad
    if num_nodes <= 50: return 8
    if num_nodes <= 75: return 4
    return 3                       # modelo grande, confiar en la política


def get_beam_width_real(num_nodes: int) -> int:
    """Beam width para evaluación estocástica (DRL Real).
    Validado empíricamente sobre nodos con gap >20%: beam=20 satura
    la mejora (beam=30 no produce cambios adicionales); beam=10-14
    deja gap residual de exploración sin explotar."""
    if num_nodes <= 10: return 10
    if num_nodes <= 20: return 7
    if num_nodes <= 35: return 7
    if num_nodes <= 50: return 10
    if num_nodes <= 75: return 6
    return 5


def get_beam_width(num_nodes: int) -> int:
    return get_beam_width_det(num_nodes)

