# AM-PPO Routing & Pricing Model

Agente de ruteo basado en un **Attention Model (Transformer encoder-decoder)**
entrenado con **PPO (Proximal Policy Optimization)**, para un problema de
ruteo estocástico con revelación post-decisión de disponibilidad de carga
(Stochastic DVRP). El agente decide, en cada nodo, a qué nodo moverse a
continuación y a qué precio cotizar la carga de esa lane (la carga se gana o se
pierde frente a la competencia), para maximizar el margen de una ruta completa
dentro de una ventana de duración máxima.


## Cómo correrlo

```bash
pip install -r requirements.txt
python main.py          # entrena → evalúa cada tamaño; resultados en results/<N>nodes/
python inference.py     # inferencia interactiva con un checkpoint ya entrenado
```

En el servidor con GPU, instalar primero el `torch` con CUDA que corresponda al driver (ver pytorch.org) y después
el resto de `requirements.txt`; el `pip install` directo puede traer la versión de solo CPU.

Todo lo que hay que tocar para un run distinto está al principio de
`main()` en [main.py](main.py): `NODE_SIZES` (tamaños a correr en orden; ahora `[20]`, la cadena completa es
`list(NODE_SEQUENCE)` = `[20, 35, 50, 75]`), `EVAL_ONLY`, `USE_TRANSFER`, `K_RESTRINGIDO`. Los hiperparámetros del modelo y del PPO están
todos en [config.py](config.py).

**GPU y CPU.** El código está escrito para la GPU: los mundos (cargas, competidores y δ) se sortean en la GPU, el
entorno, las features, el rollout, el buffer de PPO, GAE y la evaluación del DRL son tensores de torch en el
dispositivo, y a la CPU solo vuelven los escalares del log y las filas de resultados. El mismo código corre en CPU.
El dispositivo se detecta solo (`config.DEVICE`) y de él salen los tamaños de corrida (`config.EN_GPU`):

| | GPU (servidor, A100) | CPU (computador personal) |
|---|---|---|
| Episodios por update (`PPO_N_EPISODES_PER_UPDATE`) | 700 | 256 |
| Transiciones por minibatch (`PPO_BATCH_SIZE`) | 512 | 256 |
| Mundos de evaluación por nivel de K (`MARKET_M_EVAL_WORLDS`) | 1000 | 100 |

El número de updates por tamaño (`config.get_n_updates`) es el mismo en los dos. En CPU son pruebas chicas del mismo
código (idealmente 10 nodos), para revisar que todo corra de punta a punta; los resultados de la tesis salen de la GPU.
Los baselines B1, B2, B2P y B3 y la cota UB-PI también deciden para todo el lote a la vez en el dispositivo (se
calculan una vez y se cachean en `datos/cache_eval/`).

**Tests.** `python -m pytest tests` (en CPU, a 10 nodos o menos; unos 12 min, la mayor parte la tabla de la DP de B3 de
10 nodos, que se calcula una sola vez). Un archivo por módulo:
[test_pricing.py](tests/test_pricing.py) (modelo de mercado de la cotización y sus versiones en torch),
[test_market_process.py](tests/test_market_process.py) (mundos y ajuste), [test_env.py](tests/test_env.py) (entorno),
[test_modelo.py](tests/test_modelo.py) (features, decoder, rollout greedy y transfer),
[test_baselines.py](tests/test_baselines.py) (tabla de B3, UB-PI, B1–B3 y estadística),
[test_evaluacion.py](tests/test_evaluacion.py) (arnés) y [test_entrenamiento.py](tests/test_entrenamiento.py) (PPO y
validación). [tests/conftest.py](tests/conftest.py) tiene los datos de 10 nodos y un mercado sintético.

**Hiperparámetros de PPO.** Son fijos en [config.py](config.py) y salen de un HPO con Optuna que se corrió una
sola vez en 20 nodos ([hpo_optuna.py](hpo_optuna.py); los resultados están en `results/20nodes/`). `main.py` ya no
corre el HPO; los mismos valores se usan en todos los tamaños. Para rehacerlo: `python hpo_optuna.py`.

Si `DRL_Routing_Summary_AM.xlsx` o `mundos_AM.csv` están abiertos en otro programa (Excel en Windows),
`main.py` se detiene antes de evaluar, así no se pierde la evaluación al final.

## Arquitectura del modelo

```
problem_data.py ──► matrices (tiempo, tarifa, carga, distancia, diesel, LTR)
                              │
                              ▼
                    routing_env.py (VectorRoutingEnv, con cotización)
                              ▼
        ┌───────────────────────────────────────────────────┐
        │                  model/agent.py                    │
        │  AttentionEncoder  (model/encoder.py)              │
        │      → embeddings por nodo + embedding global      │
        │  ContextNetwork    (model/encoder.py)              │
        │      → vector de contexto h_t                      │
        │  AttentionDecoder + PriceHead (model/decoder.py)   │
        │      → distribución π(a|s) sobre nodos válidos     │
        │  CriticHead        (model/critic_head.py)          │
        │      → V(s_t) con encoder propio (no compartido)   │
        └───────────────────────────────────────────────────┘
                              │
                              ▼
                    am_training.py  (loop PPO-Clip + GAE)
                              │
                              ▼
                checkpoints/am_checkpoint_<N>nodes.pt
                              │
                              ▼
                    evaluation.py (DRL vs B1/B2/B3 y cota UB-PI, en octubre)
                              │
                              ▼
                results/<N>nodes/  (Excel + gráficos + training log)
```

**Reglas del entorno**: el objetivo es el margen total del ciclo. El camión puede volver a pasar por un nodo intermedio (no hay registro de visitados). En cada nodo, casa incluida antes de salir, puede cotizar una lane con carga, **esperar** al siguiente día de mercado (el self-loop) o **reposicionarse vacío** a un nodo a ≤ `REPOSICION_MAX_MILLAS` (300 mi); desde un intermedio, además, volver a casa vacío a cualquier distancia. La máscara excluye los destinos desde los que no se alcanza a volver a casa dentro de `MAX_DURATION`; volver a casa termina el episodio y con K activo el último arco es el retorno. Si desde casa no hay ninguna jugada al inicio, el camión no sale (margen 0, válido); si se queda sin jugadas después, el ciclo falla. Esperar abre una visita nueva (tablero nuevo, rechazos en cero). Durante una visita todo lo que describe "hoy" (tablero, índice, LTR, día de la semana, tablero de la red) es del día de llegada, aunque las cotizaciones perdidas (15 min cada una) lleven el reloj al día siguiente; el reloj sí cuenta para el tiempo restante.

**Features por nodo** (`model/encoder.py`): 6 base + 12 de continuación + 1 de índice de mercado + 23 de
cotización = 42. Las 8 primeras de continuación son el tiempo de j a casa, la holgura después de j y 6 estadísticas de las
lanes que SALEN de j por día de la semana de llegada (`MarketProcess.estadisticas_dow`, solo días de entrenamiento):
margen y cargas esperadas de la lane de regreso a casa, valor de la mejor lane saliente factible, cargas medias y
fracción de lanes salientes factibles, y los competidores m medios del hub. Son la media del generador del mercado
(la misma que promedia la DP de B3), no la salida de ningún baseline: el DRL no importa nada de `baselines`
para sus entradas. Las otras 4 de continuación son el **tablero de hoy** de toda la red (lo que muestra un load
board público): cargas medias publicadas hoy en las lanes salientes factibles de j, el mejor margen de hoy entre las que
tienen carga, y las cargas y el margen de hoy de la lane de regreso j→casa (`env.cargas_hoy`, `env.margen_hoy`). Las
estadísticas por día de la semana son lo mismo que sabe la DP de B3; el tablero de hoy describe el régimen del día, que
es persistente de un día al siguiente, y es información que la DP no puede meter en su estado. Las de cotización de la lane i→j son dispersión del precio, break-even relativo a la mediana,
intentos que quedan (/ `COTIZACIONES_MAX_POR_NODO`), y por nivel el valor esperado `v_k = P_k·(p_k·d − c)/p95`
y `P_k` (2 × 9), más el valor de la lane con todos sus intentos (`pricing.valor_reintentos`) si perderlos vale 0
o si es volver vacío. La fila j = i es la espera: su tiempo son las horas hasta el siguiente día de mercado
(`routing_env.fin_espera`) y la continuación es la que se tiene desde i mañana.

**Por qué el modelo escala sin reentrenar desde cero**: ninguna capa de `AMRoutingAgent` ni de `CriticHead` (`model/`) depende de `num_nodes` — el Transformer opera sobre conjuntos de tamaño variable. Esto es lo que permite el transfer learning en `transfer_learning.py` (cadena 20 → 35 → 50 → 75 nodos, ver `NODE_SEQUENCE`). Un checkpoint anterior a la cabeza de precio
actual (sin `decoder.precio.log_tau`, o con otro número de opciones de precio, es decir, los de antes del
2026-11-25) se rechaza en vez de cargarse a medias; lo mismo pasa con los de antes del 2026-11-27 (34 features de nodo, con otra
continuación: están archivados en `checkpoints/antes_2026-11-27/`) y con los de antes del 2026-11-28 (38 features, sin
el tablero de hoy, y crítico sin encoder propio; archivados en `checkpoints/antes_2026-11-28/`), así que la cadena parte
de cero en su primer tamaño.

## Estructura de archivos

### Pipeline y modelo (raíz)

| Archivo | Rol |
|---|---|
| [main.py](main.py) | Punto de entrada: carga datos → entrena → evalúa → valida el precio → guarda resultados, para cada tamaño de `NODE_SIZES` (no corre el HPO). |
| [config.py](config.py) | Todas las constantes globales e hiperparámetros (arquitectura, PPO, reward shaping). |
| [hpo_optuna.py](hpo_optuna.py) | HPO de PPO con Optuna (TPE + MedianPruner) en 20 nodos: LR, clip, λ de GAE, entropías del destino y del precio, grad clip, batch, coeficiente self-critical y weight decay. Cada trial entrena en mundos simulados de entrenamiento (jun–ago) y se mide con el `val_score` de validación (salidas en septiembre; nunca con los mundos de prueba de octubre). Se puede correr solo: `python hpo_optuna.py [--n-trials N] [--sampler random]`. |
| [problem_data.py](problem_data.py) | Carga de `datos/` y construcción de matrices de recompensa por día. |
| [routing_env.py](routing_env.py) | Entorno vectorizado (`VectorRoutingEnv`) del DVRP estocástico con cotización, en tensores de torch en el dispositivo: en cada nodo la acción es (destino, nivel de precio) y la carga se gana o se pierde frente al mejor competidor. |
| [model/encoder.py](model/encoder.py) | `AttentionEncoder` (Transformer) + `ContextNetwork`. |
| [model/decoder.py](model/decoder.py) | `AttentionDecoder` (pointer network) con glimpse + pointer attention para el destino, y `PriceHead` para el nivel de precio del destino elegido (`act/greedy/evaluate_cotizacion`). |
| [model/critic_head.py](model/critic_head.py) | `CriticHead`: estima V(s_t) para la ventaja PPO. |
| [pricing.py](pricing.py) | Niveles de precio por percentil, competidores efectivos m(LTR), aceptación `(1 − u)^m` y reglas fijas de precio (mediana, break-even + margen, miope). |
| [validate_pricing.py](validate_pricing.py) | Validación de la decisión de precio del DRL: nivel según el estado y sensibilidad a κ (main.py la corre al final de cada tamaño). |
| [model/agent.py](model/agent.py) | `AMRoutingAgent`: combina encoder + contexto + decoder; el rollout se ejecuta paso a paso sobre el entorno. |
| [am_training.py](am_training.py) | Loop de entrenamiento PPO-Clip + GAE, todo en el dispositivo (mundos, rollout, transiciones, GAE y minibatches). |
| [greedy_rollout.py](greedy_rollout.py) | Features del agente desde el estado del entorno (`construir_features`, tensores) y rollout greedy (argmax por nodo) sobre `VectorRoutingEnv`: baseline self-critical y validación del entrenamiento. |
| [model/debug_utils.py](model/debug_utils.py) | `check_tensor`: verificación de NaN/Inf en el forward pass (falla rápido en vez de silenciar con `nan_to_num`). Usado en producción por el encoder/decoder/training, no es un script de debug puntual. |
| [transfer_learning.py](transfer_learning.py) | Warm-start de pesos entre tamaños de grafo (`NODE_SEQUENCE`). |
| [baselines/ub_pi.py](baselines/ub_pi.py) | DP hacia atrás de UB-PI (`solve_ub_pi_lote`, muchos mundos a la vez en el dispositivo): valor de la mejor ruta con el mundo completo conocido, estado (nodo, franja de 1 min, k). |
| [inference.py](inference.py) | Inferencia standalone: carga el checkpoint y genera la ruta del DRL desde un nodo dado, con el nivel y el precio cotizados, la mediana, P(ganar) y si se ganó en cada paso. |

### Generador de realizaciones diarias

Cada episodio recorre días REALES consecutivos de jun–oct 2026 (nunca un día inventado), y cada uno se convierte en una realización sintética de cargas y tarifas por lane con (a) binomial negativa centrada en la media de 6 días vecinos, (b) interpolación inversa sobre los cuantiles reales de precio de ese día/lane, y (c) un corrimiento de nivel nacional δ, fijo por episodio. El entrenamiento usa siempre los mundos simulados de los días de entrenamiento; la validación y la evaluación, mundos simulados de sus propios períodos (ver el split temporal abajo).

**Split temporal** (`config.TRAIN_DAYS`, `VAL_DAYS`, `TEST_START`, `config.periodos`):

| Período | Días | Para qué |
|---|---|---|
| Entrenamiento | jun 1 – ago 31 (92) | Ajuste del proceso (`fit_market_process.py`), entrenamiento del DRL, estadísticas del agente y tablas de B2, B2P y B3. Todo lo que se aprende sale de aquí. |
| Validación | sep 1 – sep 30 (30) | Solo para elegir: checkpoint (`ValidacionSimulada`, salidas sep 1–25) e hiperparámetros (`hpo_optuna.py`). |
| Prueba | oct 1 – oct 31 (31) | Solo para reportar: 1000 mundos simulados con salidas oct 1–26 (`disenar_mundos(periodo="test")`), sobre los que corren el DRL, B1–B3 y UB-PI. |

Ninguna cantidad de un período usa días de otro: la media de cargas `m` (ventana de ±3 días) y el día más cercano con cargas de una lane (precios y LTR faltantes) se calculan dentro de cada período, y un episodio nunca pasa del último día de su período. `market_params.npz` guarda con cuántos días se ajustó (`train_days`) y se reajusta solo si no coincide con `TRAIN_DAYS`.

El índice de mercado por nodo (nivel lento + rápido AR(1), feature del agente y usado por B2P para pronosticar) no se simula: al recorrer siempre días reales, es un dato observado (lookup) para cualquier día, no una trayectoria generada.

| Archivo | Rol |
|---|---|
| [fit_market_process.py](fit_market_process.py) | Estima f_w (factor de día de semana) y r por grupo de volumen (cargas, Paso 4b), el índice de mercado (media por lane, escenarios del nivel lento con `VENTANA_LENTO_DIAS`, ρ y σ_i del rápido) y s (dispersión de δ, Paso 3), con los primeros `TRAIN_DAYS` días (jun–ago) de `datos/datos_finales_jun_oct_2026` y los 194 nodos, sin los de validación ni prueba, y guarda `datos/market_params.npz`. `main.py` lo corre solo si el archivo no existe o se ajustó con otro `TRAIN_DAYS`. |
| [market_process.py](market_process.py) | `MarketProcess`: `sortear_mundos` (B mundos sorteados de una vez en la GPU: cargas por lane y día, competidores de cada carga y δ; `Mundos`), `mundos_replay` (cargas reales de un día, solo se sortea el competidor; lo usa `inference.py`), `tensores` (tablas por día en el dispositivo: precios por nivel, cuantiles, m, índice, LTR), `sortear_cargas` (la política que ignora el mercado de la DP y B2) e `indice_historico` (índice observado del día real). |
| [validate_market_process.py](validate_market_process.py) | Paso 6 de la metodología: cuantiles de precios sorteados vs. archivo, dispersión y fracción de ceros de las cargas por grupo de volumen, δ vs. el factor nacional real, y persistencia del índice (pendiente empírica vs. la que implica el ajuste AR(1)). Tablas en `results/market_validation/<N>nodos/`. |
| [tests/test_market_process.py](tests/test_market_process.py) | Mundos en el dispositivo (formas, reproducibles con la semilla), cargas vs. histórico por grupo de volumen, replay con cargas reales (el mecanismo de `inference.py`), tablas por día, estadísticas del agente, firma, índice, ajuste y split temporal (m y día cercano sin cruzar períodos). |

Constantes (`FECHA_INICIO`, `DIAS_EXCLUIDOS`, `MARKET_R_MAX`, `MARKET_N_GRUPOS`, semillas, M) en
[config.py](config.py). La fórmula de recompensa del paso es `lane_reward` en
[problem_data.py](problem_data.py), compartida por el modo real y el simulado.

### Baselines y evaluación

El DRL se compara contra cuatro políticas no anticipativas y una cota superior, todas sobre
los mismos mundos simulados de prueba (salidas en octubre) (el mismo `Mundos`), todos los episodios a la vez en `VectorRoutingEnv`:

| Id | Baseline | Archivo |
|---|---|---|
| B1 | Greedy miope: jugada válida (cotizar, esperar o moverse vacío) de mayor valor inmediato esperado `max_k P_k (p_k d − c)` (PFA). | [baselines/politicas.py](baselines/politicas.py) |
| B2 | Reoptimización determinista con valores esperados de los arcos futuros; ejecuta solo el primer arco. En la tesis se reporta con K libre (CFA); también admite K = 5. Tabla hacia atrás sobre (casa, día de salida, nodo, franja de 15 min, k restante), calculada una vez en el dispositivo; el camión puede revisitar nodos. | [baselines/politicas.py](baselines/politicas.py) |
| B2P | B2 con pronósticos del índice de mercado: los arcos futuros j→l a h días de hoy usan `tarifa_media_empírica(j→l) + b_h · índice(j, hoy)`, con `b_h = f + (1 − f) · ρ^h` (f = fracción de varianza lenta, ρ = persistencia del rápido, ambas del ajuste del índice). Un arco futuro vale lo que rinde cotizarlo con la regla miope sobre los cuantiles medios del día de la semana, `prob · P̄_k*(p̄_k* d − c) − (1 − prob) c` (`calcular_ev_esperado`/`_valor_cotizado_por_dow`, promedio directo sobre los días de entrenamiento reales). Está en el arnés de evaluación como un baseline más (fila en el Resumen y comparación pareada del DRL), y [evaluar_b2p.py](evaluar_b2p.py) la compara con B2 sin entrenar (salida en `results/b2p/`). El DRL no usa esta tabla: sus features de continuación salen de `MarketProcess.estadisticas_dow`. | [baselines/politicas.py](baselines/politicas.py) |
| B3 | Política de un paso sobre la tabla V de una DP con los cuatro intentos de cotización (VFA de tabla); cotiza con el óptimo de su tabla. La tabla es el valor aproximado de la política que ignora el estado del mercado, construida solo con días de entrenamiento: DP en (hogar, día de salida, nodo, franja de 15 min, k), con revisitas permitidas, espera al día siguiente (también en casa antes de salir), reposicionamiento vacío y `COTIZACIONES_MAX_POR_NODO` = 4 intentos de cotización sobre el mismo tablero (salvo la discretización y el muestreo con S = 5000). Tabla en `baselines/datos/ubdp_tabla_<N>nodos.npz`, se recarga si coincide con los datos; se puede calcular aparte con `python -m baselines.dp_relajada --num-nodes N`. Por encima de `UBDP_MAX_NODOS` = 75 nodos no se intenta; y si a 75 o menos el cálculo supera `UBDP_TIEMPO_MAX_HORAS` (24 h) o la memoria de la GPU, deja `ubdp_tabla_<N>nodos.no_tratable.json` (con las horas, la fracción completada y el total estimado). En ambos casos B3 se reporta "no tratable". El valor de la tabla ya no se reporta como referencia (UB-DP): corresponde al mercado de jun–ago, no al de los mundos de prueba. | [baselines/politicas.py](baselines/politicas.py), [baselines/dp_relajada.py](baselines/dp_relajada.py) |
| UB-PI | Cota con información perfecta: mejor ruta con el mundo completo conocido, incluido el competidor (cotiza en cada arco el nivel más alto que gana; `evaluation.mundos_ub_pi` + DP hacia atrás con franjas de 1 min). Se resuelve mundo a mundo con el mundo evaluado; es la única cota y debe quedar por encima de todos los métodos. | [evaluation.py](evaluation.py), [baselines/ub_pi.py](baselines/ub_pi.py) |

| Archivo | Rol |
|---|---|
| [evaluation.py](evaluation.py) | Arnés único en lote (`ejecutar_lote`: una política sobre todos los episodios a la vez), `evaluar_mundos_simulados` (M mundos con salida en el período de prueba, K libre — `K_RESTRINGIDO = False` en main.py; con True también K = 5 —, nodo y día de la semana balanceados), `resumen_metodos`, diagnósticos. |
| [excel_report.py](excel_report.py) | Excel de resultados (hojas Resumen y Rutas y precios; stats_analysis agrega Comparación) y `mundos_AM.csv` con los datos por mundo. |
| [stats_analysis.py](stats_analysis.py) | Diferencia pareada DRL − método con IC 95 % por bootstrap, Wilcoxon pareado de una cola con Holm en dos familias (ruteo: B1–B3; precio: comparación B), y gap contra UB-PI. |
| [tests/test_baselines.py](tests/test_baselines.py) | Tabla V de la DP de B3 vs enumeración ingenua, marcador "no tratable", UB-PI vs fuerza bruta (revisitas, esperas, retrasos), B2 vs enumeración ingenua de rutas, B2P con índice cero = B2, máscara, determinismo, UB-PI ≥ políticas, B3 ≈ su tabla en mundos de entrenamiento, comparación pareada y limpieza de nombres borrados (todo a 10 nodos o menos). |
| [tests/test_evaluacion.py](tests/test_evaluacion.py) | Rango de precio por arco del DRL, métricas por episodio, caché de baselines y mundos de prueba en octubre. |

Las políticas solo leen la máscara, el tablero observado (`current_loads_row`, la mediana `rate_row_obs`, los
precios por nivel `precios_row`, m y las opciones válidas), el nodo, la hora, `k_remaining` y los rechazos de la
visita, más los parámetros del proceso; ninguna ve la semilla del mundo ni el competidor. El margen es la suma de
`lane_reward` con el precio cotizado en las cargas ganadas (USD), no la recompensa escalada del entorno. El tiempo por
decisión (`ms_decision`) es el de cada paso del lote dividido por las filas activas: es el tiempo
amortizado del lote en el dispositivo (la GPU en el servidor), para el DRL y para los baselines.

### Scripts aparte (no los corre main.py)

| Archivo | Rol |
|---|---|
| [run_k_comparison.py](run_k_comparison.py) | Evalúa en los mundos de prueba uno o varios niveles de K con un checkpoint ya entrenado. Salida: `results/<N>nodes/K_Comparison_AM.xlsx`. |
| [evaluar_b2p.py](evaluar_b2p.py) | B2 vs B2P sin entrenar nada, en los mundos de prueba. Salida: `results/b2p/`. |
| [diagnostico_rejilla_precio.py](diagnostico_rejilla_precio.py) | Cuánto margen se pierde por cotizar solo en los 9 niveles `U_NIVELES` frente a una rejilla fina (sin modelo). Salida: `results/<N>nodes/pricing_val/tables/rejilla_precio.csv`. |
| [validate_market_process.py](validate_market_process.py) | Paso 6 de la metodología del generador (ver abajo). Salida: `results/market_validation/<N>nodos/`. |
| [validar_generador_datos.py](validar_generador_datos.py) | Validación estadística del generador con los 194 hubs (KS de cargas, media del rate, normalidad de δ); produce lo que resume [validacion_generador_datos.md](validacion_generador_datos.md). Salida: `results/market_validation/validacion_metodologia/`. |
| [hpo_optuna.py](hpo_optuna.py) | HPO de PPO (ver arriba). |
| [inference.py](inference.py) | Inferencia puntual con un checkpoint (ver arriba). |

Documentos de método: [metodologia_realizaciones_mercado_reefer.md](metodologia_realizaciones_mercado_reefer.md)
(generador de realizaciones diarias) y [validacion_generador_datos.md](validacion_generador_datos.md) (su validación).


### Datos y artefactos

| Carpeta | Contenido |
|---|---|
| `datos/` | `distance.csv`/`duration.csv` (matrices fijas por par de hubs), `fuel_cost_hub_jun_oct_2026.csv` (diésel por hub), `datos_finales_jun_oct_2026/` (153 CSV, uno por día jun–oct 2026, con cargas, cuantiles de tarifa y LTR por lane REEFER), `market_params.npz` (ajuste de `fit_market_process.py`, generado). |
| `checkpoints/` | Pesos entrenados (`am_checkpoint_<N>nodes.pt` y `_best.pt`), compartidos entre tamaños para transfer learning. Git ignora solo el estado de reanudación (`am_resume_*.pt`). `antes_2026-11-27/` guarda checkpoints viejos (34 features), que el código actual ya no carga. |
| `results/<N>nodes/` | Salida de cada run: `DRL_Routing_Summary_AM.xlsx` (Resumen, Comparación, Rutas y precios), `mundos_AM.csv`, diagnósticos PPO/training (`.png`), `training_log_<N>nodes.pkl`/`.xlsx`, `pricing_val/`. En `results/20nodes/` además el HPO: `best_ppo_hyperparams.json`, `optuna_study.db` (resumible) y `optuna_trials.csv`. |
| `results/market_validation/<N>nodos/` | Tablas y figuras de `validate_market_process.py`. |
| `results/market_validation/validacion_metodologia/` | Tablas y figuras de `validar_generador_datos.py`. |
| `datos/cache_eval/` | Caché de baselines y cotas de la evaluación; se regenera en minutos (ignorado por git). |
| `baselines/datos/` | Tabla V de la DP de B3 (`ubdp_tabla_<N>nodos.npz`) y marcadores `.no_tratable.json`. Se versionan: recalcularlas cuesta horas. Si una tabla pasa de 100 MB, GitHub no la acepta y hay que moverla a mano (p. ej. como dataset de Kaggle). |

## Pricing: cotización conjunta con el ruteo

El precio no es una etapa posterior al ruteo: en cada nodo el DRL elige a la vez el destino `j` y una opción de
precio `o`.

- **Niveles.** 9 niveles comunes a todas las lanes en escala de percentiles, `U_NIVELES = 0.1, …, 0.9`
  (p10 a p90: ninguno gana con seguridad); el precio en dólares es el cuantil `u_k` de la distribución de
  tarifas de esa lane y ese día (`pricing.py`). La última opción es VACÍO: reposicionarse (≤ 300 mi), volver a
  casa desde un intermedio, o esperar si el destino es el propio nodo.
- **Aceptación.** `P(ganar | u_k) = (1 − u_k)^m`, con `m = clip(1 + κ · LTR_REF / LTR, 1, M_MAX)` competidores
  efectivos según el LTR del nodo de origen (κ = `COMPETENCIA_KAPPA` = 1). Se implementa con un competidor
  `u_comp ~ Beta(1, m)` sorteado con la semilla del mundo, así que todos los métodos ven el mismo competidor.
- **Rechazo.** Si se pierde la carga, el camión se queda en el nodo, pierde `COTIZACION_HORAS` (15 min), no
  consume K y pierde solo esa carga: puede cotizar otra de la misma lane (otro competidor independiente), de las que
  haya. Hay a lo sumo `COTIZACIONES_MAX_POR_NODO` = 4 cotizaciones por visita; esperar al día siguiente abre una
  visita nueva (tablero nuevo y rechazos en cero).
- **Reward.** `p · dist − costo` si se gana; `−costo` en un VACÍO; 0 al perder o al esperar.
- **Lo que cuesta perder.** En la vuelta a casa, perder la carga es volver vacío: las reglas fijas lo cuentan
  (`baselines/politicas.costo_de_perder`). El margen reportado resta 3000 USD por ciclo fallido.
- **Última oportunidad.** Si perder deja al camión sin jugadas en casa, las reglas fijas cotizan la miope que
  cuenta el costo de fallar el ciclo (`baselines/politicas.ultima_oportunidad`).
- **Modelo.** `PriceHead` (model/decoder.py) da el nivel condicionado al destino elegido a partir del
  valor esperado de cada nivel (`τ·(v_k + P_k·Δ) + MLP`); la entropía del precio
  tiene su propio coeficiente (`PPO_ENTROPY_COEF_PRECIO_*`). Δ = W − L (valor de seguir después de ganar menos el de
  después de perder) sale de una cabeza propia que se ajusta por **regresión** sobre lo realizado en la rama que ocurrió
  (pérdida auxiliar en `am_training.py`), no por gradiente de política. El crítico es `CriticHead` V(s), con su propio
  encoder (Andrychowicz et al., 2021).

**Comparación B.** Para separar lo que aporta la cabeza de precio, la evaluación corre el DRL completo y sus
mismos destinos con una regla fija de precio: `DRL-dest+mediana`, `DRL-dest+be_margen` (break-even +
`BE_MARGEN`) y `DRL-dest+miope` (el nivel que maximiza `P_k (p_k d − c)`). B1–B2P cotizan con la regla miope
(`REGLA_PRECIO_BASELINES`) y B3 con el óptimo de su tabla. `stats_analysis.py` las compara en una familia de Holm aparte ("precio").

**Validación del precio.**

```bash
python validate_pricing.py --num-nodes 10 [--checkpoint RUTA] [--quick]
```

`main.py` la corre al final de cada tamaño (o aparte con el comando de arriba sobre un checkpoint). Escribe
`results/<N>nodes/pricing_val/{tables,figures}`,
en mundos de prueba (salidas en octubre) con semilla propia (`PRICING_VAL_SEED`): (a) nivel cotizado por el DRL por quintil de LTR, m,
tiempo restante e índice de mercado de origen y destino, con Spearman; (b) sensibilidad a κ ∈ {0.5, 1, 2},
con el DRL entrenado en κ = 1 y la regla miope como referencia. La comparación B la hace la evaluación.

**Limitaciones.** (1) La aceptación es un modelo (no hay datos de cargas ganadas y perdidas): κ fija qué tanto
pesa el LTR, y por eso se reporta la sensibilidad. (2) La tabla de B3 cuesta unas 20 veces más que sin precio y su costo crece como N³ · S: de 35 a 50 nodos
es ≈ 2.9× y de 35 a 75 nodos ≈ 9.8×. Se intenta hasta `UBDP_MAX_NODOS` = 75 con un tope de
`UBDP_TIEMPO_MAX_HORAS` = 24 h; si un tamaño lo excede o se queda sin memoria (a 75 nodos es lo más probable),
B3 se reporta "no tratable" bajo ese presupuesto, no como una imposibilidad teórica.

## Flujo de trabajo típico

1. **Entrenar uno o varios tamaños**: fijar `NODE_SIZES` en `main.py` (p. ej. `[20]` o
   `list(NODE_SEQUENCE)`), dejar `USE_TRANSFER = True` para partir del checkpoint del tamaño anterior en
   `NODE_SEQUENCE` (o `False` para pesos aleatorios). `python main.py`. Si un tamaño falla, se imprime el error y se sigue con el siguiente.
2. **Solo evaluar** un checkpoint existente sin reentrenar: `EVAL_ONLY=1 python main.py`.
3. **Comparar contra baselines**: `main.py` calcula (o recarga) la tabla V de B3,
   evalúa en los mundos de prueba (salidas en octubre) e imprime el resumen.
   `results/<N>nodes/DRL_Routing_Summary_AM.xlsx` tiene cuatro hojas: Resumen
   (margen medio con IC95, Δ vs DRL, gap contra UB-PI), Comparación (bootstrap y Wilcoxon con Holm), Rutas
   (cada ciclo del DRL: la ruta, p. ej. 1-5-8-7-1, y por arco el rango de precio y la referencia) y Precios por
   arco (el rango típico de cada lane frente a la mediana publicada y el break-even). La referencia es el precio
   al que el DRL ganó la carga (el del margen de la ruta); el rango, los niveles contiguos alrededor del elegido
   que juntan el 80% (`BANDA_PRECIO_MASA`) de la probabilidad de la cabeza de precio en esa decisión.
4. **Generador de realizaciones diarias**: si falta `datos/market_params.npz`, `main.py` corre
   `fit_market_process.py`. Validar el generador con `python validate_market_process.py`.
5. **Inferencia puntual**: `python inference.py` con un checkpoint `am_checkpoint_<N>nodes.pt` ya
   guardado en `checkpoints/`.
6. **Validar el precio**: `python validate_pricing.py --num-nodes N` sobre ese checkpoint.

## Requisitos

Ver [requirements.txt](requirements.txt). **Python 3.10 – 3.12** (el piso lo marca `requires_python >= 3.10` de los
pins; probado en 3.10.16 y 3.12.2). Paquetes: PyTorch, NumPy/Pandas/SciPy, Optuna, Matplotlib, openpyxl y pytest.
El entorno es propio (`VectorRoutingEnv` en torch), no usa Gymnasium.
