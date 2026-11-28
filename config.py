"""
config.py
=========
All global constants, toggles, and hyperparameter defaults for the AM-PPO pipeline.
This file is imported by all modules, so it should not contain any heavy dependencies.

"""

import os
import random

import numpy as np
import torch

# ── Device ───────────────────────────────────────────────────
GPU_PREFERIDA = ""   # Subcadena del nombre de la GPU a preferir (p.ej. "3070 Ti");
                      # vacío = usar la GPU por defecto (cuda:0) de la máquina.


def _elegir_device() -> torch.device:
    """Si GPU_PREFERIDA está vacío, usa la GPU por defecto (cuda:0). Si se especifica,
    busca una GPU cuyo nombre la contenga (por nombre, no por índice: el orden de CUDA
    puede diferir del de nvidia-smi). Se fija además como dispositivo por defecto, para que
    torch.cuda.manual_seed y current_device() apunten a la misma GPU."""
    if not torch.cuda.is_available():
        return torch.device("cpu")
    if not GPU_PREFERIDA:
        torch.cuda.set_device(0)
        return torch.device("cuda:0")
    for i in range(torch.cuda.device_count()):
        if GPU_PREFERIDA in torch.cuda.get_device_name(i):
            torch.cuda.set_device(i)
            return torch.device(f"cuda:{i}")
    import warnings
    warnings.warn(f"GPU '{GPU_PREFERIDA}' no visible (CUDA_VISIBLE_DEVICES="
                  f"{os.environ.get('CUDA_VISIBLE_DEVICES')!r}); usando {torch.cuda.get_device_name(0)}.")
    return torch.device("cuda:0")


DEVICE = _elegir_device()
EN_GPU = DEVICE.type == "cuda"   # los tamaños de corrida salen de aquí (ver "Tamaños de corrida" más abajo)

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
# Tamaños de corrida: se eligen solos según el dispositivo. En GPU (el servidor, A100) son los de la tesis; en CPU (el
# computador personal) son pruebas chicas del mismo código, idealmente a 10 nodos, que no sirven como resultado.
# Episodios por update y minibatch en GPU: los del HPO (700 y 512; results/20nodes/best_ppo_hyperparams.json), con los
# que están calibrados el LR, las entropías y el clip. Todos los episodios corren en paralelo en la GPU (un solo rollout
# batcheado). El número de updates por tamaño (get_n_updates) no depende del dispositivo.
PPO_N_EPISODES_PER_UPDATE = 700 if EN_GPU else 256
PPO_N_EPOCHS              = 4      # Épocas de entrenamiento PPO por rollout
PPO_BATCH_SIZE            = 512 if EN_GPU else 256    # transiciones por minibatch de PPO
PPO_LR                    = 0.0001027303916436541   # Learning rate para el actor. Subido de 1e-5: la KL media
                                    # iba muy por debajo del target 0.02 con clip < 0.10,
                                    # indicando under-stepping. knob: subir hacia 5e-5 si KL se
                                    # mantiene < ~0.015 y el reward sigue subiendo pasado update 150.
PPO_GAMMA                 = 1.0    # Factor de descuento. 1.0: horizonte finito de 2–5 pasos, sin descuento
                                    # (el margen del episodio es la suma de las lanes). Era 0.99.
PPO_GAE_LAMBDA            = 0.9700773994877414   # Factor de GAE
PPO_CLIP_EPS              = 0.1855572933544571   # Clip PPO para limitar cambios de política
PPO_ENTROPY_COEF_START    = 0.03839069309492253   # coeficiente de entropía al inicio del entrenamiento
PPO_ENTROPY_COEF_END      = 0.009450213956724884   # coeficiente de entropía al final del entrenamiento (decay lineal)
PPO_GRAD_CLIP             = 0.8164399100201447    # Clipping de gradiente para PPO
PPO_WEIGHT_DECAY          = 1.8010135852882087e-05  # Weight decay (AdamW) del actor. Regularizador PRIMARIO
                                    # PPO-safe: no corrompe el ratio de importancia como el
                                    # dropout en el path de logits.

# ── Pricing ───────────────────────────────────────────────────
# (El precio de reserva con las cabezas Ṽ A/B y post_training.py se eliminó en la Fase 3 de
# especificacion_fase0_ruteo_pricing.md: el precio ahora es una decisión de la política.)
PRICING_VAL_SEED = 20261022   # semilla del muestreo de la validación del pricing (Fase 6)
BE_MARGEN       = 0.30        # USD/milla sobre el break-even de la regla de precio fija "be_margen" (pricing.nivel_regla)

# ── Ruteo y pricing conjunto (especificacion_fase0_ruteo_pricing.md) ──
# En cada nodo la política elige (destino, nivel de precio). Los niveles son percentiles fijos, comunes a todas
# las lanes, que se convierten en USD/milla con los cuantiles de cada lane y día (pricing.precios_niveles).
U_NIVELES = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)   # p10 … p90: ningún nivel gana con seguridad (no hay
                                                            # "cotizar al mínimo del mercado")
COMPETENCIA_KAPPA = 1.0    # m = clip(1 + κ · LTR_REF / LTR, 1, M_MAX); sensibilidad κ ∈ {0.5, 1, 2} (entrenado con 1)
COMPETENCIA_M_MAX = 20.0   # tope de competidores efectivos (con LTR ≈ 0.8, p5 del train, m sin tope sería ≈ 33)
# LTR_REF no es constante: es la mediana del LTR por hub y día en los días de train (los 194 hubs, así no depende
# de num_nodes) y la calcula MarketProcess al iniciar (≈ 25.5 con los datos jun-sep 2026).
COTIZACION_HORAS = 0.25          # horas que se pierden por cada cotización rechazada (= una franja de la DP)
COTIZACIONES_MAX_POR_NODO = 4    # rechazos por visita a un nodo (o por día de espera); después solo quedan esperar
BANDA_PRECIO_MASA = 0.8   # rango de precio por arco (evaluación): niveles contiguos alrededor del elegido que juntan esta probabilidad de la cabeza de precio
                                 # o moverse vacío. Se pueden gastar varios en la misma lane: cada carga de la lane
                                 # es una cotización distinta, con su propio competidor
# Movimientos sin carga (routing_env.py): en cualquier nodo, casa incluida antes de salir, el camión puede
#   · ESPERAR al siguiente día de mercado (múltiplo de HORAS_DIA_MERCADO) y cotizar con un tablero nuevo;
#   · moverse VACÍO a un nodo a ≤ REPOSICION_MAX_MILLAS (reposicionamiento, paga el costo completo por milla);
#   · y desde un nodo intermedio, volver a casa VACÍO a cualquier distancia.
REPOSICION_MAX_MILLAS = 300.0
REGLA_PRECIO_BASELINES = "miope" # regla fija de precio de B1-B3 ("mediana" | "be_margen" | "miope")
# Entropía de la cabeza de precio (aparte de la del destino). Máximo = log(10) ≈ 2.3 con 9 niveles + VACÍO.
# Valores fijos: los del HPO de 20 nodos (hpo_optuna.py, results/20nodes/best_ppo_hyperparams.json).
PPO_ENTROPY_COEF_PRECIO_START = 0.03688801102074321
PPO_ENTROPY_COEF_PRECIO_END   = 0.019523395818174943
UBDP_TIEMPO_MAX_HORAS = 24.0     # si la tabla de UB-DP tarda más, ese tamaño se reporta "no tratable"

# ── Self-critical baseline (híbrido sobre actor-critic) ────────────
USE_SELF_CRITICAL            = True   # término self-critical sobre el argmax greedy
SELF_CRITICAL_COEF           = 0.41233283242927726     # peso del término self-critical en el loss total
SELF_CRITICAL_WARMUP_UPDATES = 50      # no aplicar self-critical hasta este update
                                        # (deja que el crítico se estabilice primero)

# ── Selección del mejor checkpoint (am_training.py) ───────────
# "max": guarda el update con el mayor score puntual de validación (comportamiento anterior).
# "moving_avg": guarda los pesos del update CENTRAL de la ventana de CKPT_WINDOW validaciones
# consecutivas cuya media de score es la mejor hasta ahora (menos ruido, menos selección por azar).
CKPT_SELECTION = "moving_avg"   # "moving_avg" | "max"
CKPT_WINDOW    = 3

# ── Reanudación del entrenamiento (am_training.py) ────────────
# Cada tanto se guarda el estado COMPLETO del entrenamiento en checkpoints/am_resume_<N>nodes.pt: pesos, optimizadores,
# scheduler, update, selección del mejor checkpoint, historial y estados aleatorios. Con RESUME=1 (variable de entorno),
# main.py sigue desde ahí si el archivo existe; sin RESUME=1 se entrena desde cero y el archivo se sobrescribe.
RESUME_CADA_UPDATES = 20    # guarda cada tantos updates…
RESUME_CADA_MIN     = 15.0  # …o cada tantos minutos desde el último guardado, lo que pase primero

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
USE_CONTINUATION_FEATURES = True   # True: 8 features extra por nodo j (N_NODE_FEATURES = 38 con el índice de mercado) con la
                                   # información de qué pasa después de j: tiempo y holgura de regreso a casa y
                                   # estadísticas por día de la semana de las lanes que salen de j (market_process.
                                   # EstadisticasLane; ver model/encoder.py). False: las 6 features base.

# ── Train / eval split ───────────────────────────────────────
# jun 1 - sep 30 2026 (122 días) entrenamiento; oct 1-31 2026 (31 días) prueba, fuera de muestra.
TRAIN_DAYS = 122

# ── Market signal normalisation clamps ───────────────────────
LTR_CLIP           = 200.0   # techo seguro para el load_to_truck_ratio diario del hub
LOADS_EXPECTED_CLIP = 100.0  # p95 de E[loads] por lane (train, excl. diagonal) es ~66; 100 es techo seguro

# ── Reproducibility ───────────────────────────────────────────
SEED = 42   # torch/numpy del entrenamiento (una sola semilla: no se corren réplicas)


def set_seeds(seed: int) -> None:
    """Semilla de random/numpy/torch (CPU y CUDA si hay GPU). Único lugar: antes estaba
    duplicada en main.py, hpo_optuna.py y run_k_comparison.py."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

# ── Generador de realizaciones diarias (market_process) ──────
# Ver metodologia_realizaciones_mercado_reefer.md (generador); el índice de mercado por nodo se
# documenta en fit_market_process.py. Ningún índice de día va escrito a mano: el número de
# días sale de los CSV diarios de datos/datos_finales_jun_oct_2026 y las fechas salen de sus nombres
# de archivo (FECHA_INICIO = la primera fecha de esa carpeta, jun 1 2026).
FECHA_INICIO         = "2026-06-01"   # primer día de datos/datos_finales_jun_oct_2026 (lunes)
DIAS_EXCLUIDOS       = []             # fechas ("AAAA-MM-DD") a dejar fuera del ajuste, solo si
                                      # la fuente confirma que son erróneas. Vacía = usa todos los días de ajuste.
MARKET_R_MAX         = 1000.0         # tope de la dispersión r de la binomial negativa (r → ∞ equivale a Poisson)
MARKET_N_GRUPOS      = 10             # grupos de volumen h (deciles de la m promedio de cada lane) para estimar r_h
MARKET_VENTANA_VECINOS = 3            # días antes/después (sin el propio día) para la media móvil de m (Paso 4b)
MARKET_PARAMS_PATH   = "datos/market_params.npz"   # relativo a la raíz del proyecto
VENTANA_LENTO_DIAS   = 21             # ventana (impar, centrada) del promedio que separa el estado lento del rápido del
                                      # índice de mercado; los días con la ventana completa son los "días de escenario"
HORAS_DIA_MERCADO    = 14.0           # un día de mercado = 14 h (igual que el entorno)
MARKET_DIAS_EPISODIO = int(MAX_DURATION // HORAS_DIA_MERCADO) + 1   # días de mercado que puede tocar un episodio
                                      # (el de salida y los que caben en MAX_DURATION: 6 con 77 h)
MARKET_SEED_TRAIN    = int(os.environ.get("MARKET_SEED_TRAIN", 20260))   # semilla de los mundos de entrenamiento
                                                                          # (variable de entorno para variarla por corrida)
MARKET_SEED_EVAL     = 777001  # semilla base de los mundos de evaluación (distinta de la de entrenamiento)
MARKET_SEED_EVAL_FINAL = 777003  # semilla de un segundo conjunto de mundos de evaluación (conjunto B, test final):
                                  # distinta de entrenamiento, validación y MARKET_SEED_EVAL. MARKET_SEED_EVAL es el
                                  # conjunto de desarrollo (con él se decidieron cosas, p. ej. el checkpoint); el B se
                                  # usa UNA vez, con la configuración congelada. main.py: EVAL_SET=final.
MARKET_M_EVAL_WORLDS = 1000 if EN_GPU else 100   # M: nº de mundos de evaluación simulados por nivel de K. Con
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
UBDP_S          = 5000    # sorteos de filas por par (nodo, día de la semana)
UBDP_MAX_NODOS  = 75      # por encima, UB-DP (y B3) no se calcula: se reporta "no tratable" sin gastar horas
                          # (con los dos intentos de cotización el costo crece con N³ · S)
UBDP_SEED       = 20261018   # semilla del Generator propio de la DP
UBDP_TABLA_PATH = "datos/ubdp_tabla_{n}nodos.npz"   # relativo a la raíz del proyecto; {n} = nº de nodos
REPLAY_MAX_CASOS = 300   # tope de casos (nodo, día) del replay por nivel de K; si hay más, se toma un subconjunto balanceado
BOOTSTRAP_B     = 10000   # remuestreos del intervalo bootstrap de la diferencia pareada
BOOTSTRAP_SEED  = 20261019

# ── Hiperparámetros por tamaño de grafo ──────────────────────
# Lo que se escriba aquí para un tamaño pisa el valor global de este archivo; {} = los de siempre.
# Claves admitidas: PPO_LR, PPO_N_EPISODES_PER_UPDATE, PPO_ENTROPY_COEF_START, PPO_ENTROPY_COEF_END,
# LR_FINAL_FRAC.
HP_POR_NODOS = {
    10: {},
    20: {},   # sin overrides: los hiperparámetros del HPO (hpo_optuna.py) ya están fijos en este archivo
    35: {"LR_FINAL_FRAC": 0.30},
    50: {},
    75: {},
    100: {},
}
HP_OVERRIDES = {}   # se llena en configurar_hp()


def configurar_hp(num_nodes: int) -> None:
    """Carga HP_POR_NODOS[num_nodes] en HP_OVERRIDES (en el sitio). Llamar desde main()."""
    HP_OVERRIDES.clear()
    HP_OVERRIDES.update(HP_POR_NODOS.get(num_nodes, {}))


def hp(nombre: str, default):
    """Valor de la variante activa para `nombre`, o `default` si la variante no lo cambia."""
    return HP_OVERRIDES.get(nombre, default)

# ── Debug ──────────────────────────────────────────────────────
CHECK_TENSORS = os.environ.get("CHECK_TENSORS", "0") == "1"   # True: valida NaN/Inf en cada tensor
                                  # de _forward (model/debug_utils.check_tensor). Cada check fuerza
                                  # una sincronización CPU-GPU; con d_h=192/N chico (overhead de
                                  # kernels domina, ver comentario de PPO_BATCH_SIZE) representan
                                  # ~200 syncs/update dentro de ppo_calculo (el 77% del tiempo/update
                                  # del log). Activar con CHECK_TENSORS=1 solo para diagnosticar NaN/Inf.



def get_n_updates(num_nodes: int) -> int:
    """Updates PPO del entrenamiento completo por tamaño (los mismos que daba el presupuesto de episodios anterior
    con 700 episodios por update). Los episodios totales son n_updates × PPO_N_EPISODES_PER_UPDATE."""
    if num_nodes <= 10:  return 285
    if num_nodes <= 20:  return 714
    if num_nodes <= 35:  return 1300
    if num_nodes <= 50:  return 1000
    if num_nodes <= 75:  return 1070
    return 1290                         # 100 nodos
