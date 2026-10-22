# AM-PPO Routing & Pricing Model

Agente de ruteo basado en un **Attention Model (Transformer encoder-decoder)**
entrenado con **PPO (Proximal Policy Optimization)**, para un problema de
ruteo estocástico con revelación post-decisión de disponibilidad de carga
(Stochastic DVRP). El agente decide, en cada nodo, a qué nodo moverse a
continuación para maximizar la recompensa acumulada (margen de flete) dentro
de una ventana de duración máxima.


## Cómo correrlo

```bash
pip install -r requirements.txt
python main.py          # entrena (o evalúa) y guarda resultados en results/<N>nodes/
python inference.py     # inferencia interactiva con un checkpoint ya entrenado
```

Todo lo que hay que tocar para un run distinto está al principio de
`main()` en [main.py](main.py): `NUM_NODES`, `EVAL_ONLY`, `USE_TRANSFER`,
`USE_MARKET_PROCESS`. Los hiperparámetros del modelo y del PPO están todos en
[config.py](config.py).

## Arquitectura del modelo

```
problem_data.py ──► matrices (tiempo, tarifa, carga, distancia, diesel, LTR, camiones)
                              │
                              ▼
                    routing_env.py (RoutingEnv / VectorRoutingEnv)
                              ▼
        ┌─────────────────────────────────────────────┐
        │                am_agent.py                   │
        │  AttentionEncoder  (attention_encoder.py)     │
        │      → embeddings por nodo + embedding global │
        │  ContextNetwork    (attention_encoder.py)     │
        │      → vector de contexto h_t                 │
        │  AttentionDecoder  (attention_decoder.py)      │
        │      → distribución π(a|s) sobre nodos válidos │
        │  CriticHead        (critic_head.py)             │
        │      → V(s_t) para la ventaja PPO                │
        │  PostDecisionValueHead (critic_head.py)         │
        │      → Ṽ(s, j) para el precio de reserva p*     │
        └─────────────────────────────────────────────┘
                              │
                              ▼
                    am_training.py  (loop PPO-Clip + GAE)
                              │
                              ▼
                checkpoints/am_checkpoint_<N>nodes.pt
                              │
                              ▼
                    evaluation.py (DRL vs B1/B2/B3 y cotas UB-PI/UB-DP)
                              │
                              ▼
                results/<N>nodes/  (Excel + gráficos + training log)
```

**Por qué el modelo escala sin reentrenar desde cero**: ninguna capa de
`AMRoutingAgent` ni de `CriticHead` depende de `num_nodes` — el Transformer
opera sobre conjuntos de tamaño variable. Esto es lo que permite el transfer
learning en `transfer_learning.py` (cadena 10 → 20 → 35 → 50 → 75 → 100
nodos, ver `NODE_SEQUENCE`).

## Estructura de archivos

### Pipeline y modelo (raíz)

| Archivo | Rol |
|---|---|
| [main.py](main.py) | Punto de entrada: carga datos → entrena → evalúa → guarda resultados. |
| [config.py](config.py) | Todas las constantes globales e hiperparámetros (arquitectura, PPO, reward shaping). |
| [problem_data.py](problem_data.py) | Carga de `datos/` y construcción de matrices de recompensa por día. |
| [routing_env.py](routing_env.py) | Entorno Gymnasium (`RoutingEnv`, `VectorRoutingEnv`) para el DVRP estocástico. |
| [attention_encoder.py](attention_encoder.py) | `AttentionEncoder` (Transformer) + `ContextNetwork`. |
| [attention_decoder.py](attention_decoder.py) | `AttentionDecoder` (pointer network) con glimpse + pointer attention. |
| [critic_head.py](critic_head.py) | `CriticHead`: estima V(s_t) para la ventaja PPO. `PostDecisionValueHead`: Ṽ(s, j), valor de llegar a j antes de ver su tablero (precio de reserva). |
| [pricing.py](pricing.py) | `reservation_price`: precio de reserva p* por lane. |
| [validate_pricing.py](validate_pricing.py) | Validación de p*: Monte Carlo de Ṽ, posición histórica, política de aceptación. |
| [am_agent.py](am_agent.py) | `AMRoutingAgent`: combina encoder + contexto + decoder; el rollout se ejecuta paso a paso sobre el entorno. |
| [am_training.py](am_training.py) | Loop de entrenamiento PPO-Clip + GAE. |
| [greedy_rollout.py](greedy_rollout.py) | Rollout greedy (argmax por nodo) sobre `VectorRoutingEnv`: evaluación del DRL en replay de días reales o en mundos simulados, baseline self-critical y monitor de entrenamiento. |
| [debug_utils.py](debug_utils.py) | `check_tensor`: verificación de NaN/Inf en el forward pass (falla rápido en vez de silenciar con `nan_to_num`). Usado en producción por el encoder/decoder/training, no es un script de debug puntual. |
| [transfer_learning.py](transfer_learning.py) | Warm-start de pesos entre tamaños de grafo (`NODE_SEQUENCE`). |
| [Solvers.py](Solvers.py) | Label-setting oráculo (`solve_label_setting_oracle`), único solver que se conserva: resuelve UB-PI. |
| [inference.py](inference.py) | Inferencia standalone: carga un checkpoint y genera una ruta greedy desde un nodo dado. |

### Proceso exógeno de cargas y tarifas (opcional)

Diseño completo en [proceso_exogeno_cargas_tarifas.md](proceso_exogeno_cargas_tarifas.md).
Con `USE_MARKET_PROCESS = True` el entrenamiento ya no reproduce días del
histórico: al llegar a cada nodo se sortean las cargas (binomial negativa con
factor de día de la semana) y las tarifas (remuestreo del histórico de la lane).
Con `False` el entrenamiento reproduce días reales como antes. La evaluación siempre usa el proceso.

| Archivo | Rol |
|---|---|
| [fit_market_process.py](fit_market_process.py) | Estima s_d, mu_ij y phi_h con los primeros `TRAIN_DAYS` días (0-89) del histórico, sin los de evaluación, y guarda `datos/market_params.npz`. `main.py` lo corre solo si el archivo no existe. |
| [market_process.py](market_process.py) | `MarketProcess`: `sample_arrivals` (lotes con un Generator), `sample_arrivals_por_mundo` (usado por el entorno; determinista por semilla de mundo, nodo y día) y `sample_world` (mundo completo [D, N, N] para baselines). |
| [validate_market_process.py](validate_market_process.py) | Simula 1.000 réplicas del periodo de ajuste (dentro de muestra) y compara cargas por lane y tarifas con el histórico (tablas y gráficos en `results/market_validation/`). |
| [tests/test_market_process.py](tests/test_market_process.py) | Reproducibilidad, independencia de las acciones, formas, equivalencia con `market_process=None`, recompensa compartida, ajuste y smoke de entrenamiento. |

Constantes (`FECHA_INICIO`, `DIAS_EXCLUIDOS`, `MARKET_PHI_MAX`, semillas, M) en
[config.py](config.py). La fórmula de recompensa del paso es `lane_reward` en
[problem_data.py](problem_data.py), compartida por el modo real y el simulado.

### Baselines y evaluación

Diseño completo en [baselines_drl_ruteo_estocastico.md](baselines_drl_ruteo_estocastico.md).
El DRL se compara contra tres políticas no anticipativas y dos cotas superiores, todas sobre
los mismos mundos simulados (mismo `world_seed`), paso a paso en `VectorRoutingEnv`:

| Id | Baseline | Archivo |
|---|---|---|
| B1 | Greedy miope: arco válido de mayor `lane_reward` de la fila observada (PFA). | [baselines.py](baselines.py) |
| B2 | Reoptimización determinista con valores esperados de los arcos futuros; ejecuta solo el primer arco. Con K = 5 y con K libre (CFA); con K libre la enumeración exacta solo es viable hasta ~50 nodos. | [baselines.py](baselines.py) |
| B3 | Política de un paso sobre la tabla V de la DP relajada (VFA de tabla). | [dp_relajada.py](dp_relajada.py) |
| UB-DP | Cota ajustada: DP en (hogar, día de salida, nodo, franja de 15 min, k) permitiendo revisitar. Tabla en `datos/ubdp_tabla_<N>nodos.npz`, se recarga si coincide con los datos. | [dp_relajada.py](dp_relajada.py) |
| UB-PI | Cota laxa: mejor ruta con el mundo completo conocido (`sample_world` + label-setting oráculo sin poda por dominancia). | [evaluation.py](evaluation.py), [Solvers.py](Solvers.py) |

| Archivo | Rol |
|---|---|
| [evaluation.py](evaluation.py) | Arnés único (`ejecutar_politica`), `evaluar_mundos_simulados` (M mundos por K ∈ {5, libre}, nodo y día de la semana balanceados), `evaluar_replay` (días reales 90-119, fuera de muestra: "replay, fuera de muestra (días 90-119)"), `resumen_metodos`, diagnósticos. |
| [excel_report.py](excel_report.py) | Excel de resultados: hojas Resumen, Mundos simulados y Replay. |
| [stats_analysis.py](stats_analysis.py) | Diferencia pareada DRL − baseline con IC 95 % por bootstrap, Wilcoxon pareado de una cola con Holm, y gaps contra UB-DP y UB-PI. |
| [tests/test_baselines.py](tests/test_baselines.py) | Tabla V vs recursión ingenua, máscara, determinismo, UB-PI ≥ políticas, B3 ≤ UB-DP, margen = Σ `lane_reward`, oráculo vs enumeración, y limpieza de nombres borrados. |

Las políticas solo leen la máscara, la fila observada (`current_loads_row`, `current_rate_row`),
el nodo, la hora, `k_remaining` y los visitados, más los parámetros del proceso; ninguna ve la
semilla del mundo. El margen es la suma de `lane_reward` de los arcos recorridos (USD), no la
recompensa del entorno. Los tiempos por decisión se miden en CPU con un solo hilo, también para
el DRL.

### Validación final del modelo

| Archivo | Rol |
|---|---|
| [run_k_comparison.py](run_k_comparison.py) | Script aparte: evalúa en mundos simulados uno o varios niveles de K con un checkpoint ya entrenado. |


### Datos y artefactos

| Carpeta | Contenido |
|---|---|
| `datos/` | Matrices fuente (`.csv`/`.npy`): distancia, duración, tarifa, diesel, disponibilidad de carga, LTR, camiones. |
| `checkpoints/` | Pesos entrenados (`am_checkpoint_<N>nodes.pt`), compartidos entre tamaños para transfer learning. Ignorado por git (`.gitignore`). |
| `results/<N>nodes/` | Salida de cada run: `DRL_Routing_Summary_AM.xlsx`, diagnósticos PPO/training (`.png`), `training_log_<N>nodes.pkl`/`.xlsx`. |
| `docs/architecture.html` / `.png` | Diagrama explicativo del pipeline AM-PPO. |

## Pricing: precio de reserva

Además de la ruta, el modelo entrega una segunda salida: el **precio mínimo por milla** al que al camión le
conviene aceptar una lane `i→j`, dada su posición y el tiempo restante. Es una recomendación del lado del
transportista; **no modela aceptación** (no hay datos de aceptación y rechazo).

**Modelo.** En `i` el agente ya ve `rate(i→j)` y `loads(i→j)`. Aceptar `i→j` rinde
`r_obs = rate·dist − costo` y llega a `j`, cuyo tablero aún no conoce; su valor esperado es `Ṽ(s, j)` (valor
post-decisión, cabeza `agent.post_value`, con `Ṽ(s, depot) = 0`). Con `S = REWARD_SCALE_FACTOR`:

```
Q(s, a)  = r_obs(i→a)/S + Ṽ(s, a)
Q_alt(s) = max_{a ≠ j válida} Q(s, a)
p*(i→j)  = ( S · (Q_alt − Ṽ(s, j)) + costo(i→j) ) / dist(i→j)      [USD/milla]
```

Es el costo de oportunidad por milla de ir a `j`: puede quedar por debajo del break-even `costo/dist` cuando
`j` es una posición valiosa. Si `j` es la única acción factible se devuelve el piso `costo/dist` con
`only_option = True`.

**Entrenamiento.** `Ṽ` se regresa con el mismo rollout de PPO: `loss_post = MSE(Ṽ(s_t, a_t), G_t − r_t)` solo
sobre la acción tomada (el retorno al depot se excluye: es el ancla `Ṽ = 0`). La cabeza usa `h_t.detach()` y
`embeddings.detach()`, y sus parámetros van al optimizer del crítico, así que **no cambia la política**: con la
misma semilla el reward del PPO es idéntico con o sin ella. Config: `USE_POST_HEAD`, `POST_HIDDEN`, `POST_COEF`.
`post_mse` y `post_ev` quedan en el `training_log`.

**Limitaciones.** (1) `p*` es un costo de oportunidad, no una probabilidad de aceptación. (2) `Ṽ` se entrena solo
en acciones muestreadas: para lanes casi nunca tomadas la estimación (y `p*`) es más ruidosa. (3) Con tours de
2-4 arcos `Ṽ` tiene varianza alta. (4) Los checkpoints anteriores no traen `post_value`: se cargan, pero la
cabeza arranca desde cero; hay que reentrenar. La validación lo reporta (ver abajo).

**Cómo ejecutarlo.**

```bash
python main.py                      # entrena con la cabeza y, al final (RUN_PRICING_VALIDATION), valida p*
python validate_pricing.py --num-nodes 10 [--best] [--quick]   # validación sola sobre un checkpoint
python inference.py [--show-board]  # ruta + p* por arco (y tablero por nodo); escribe results/<N>nodes/pricing_<run_tag>.csv
```

La validación escribe `results/<N>nodes/pricing_val/{figures,tables}` y la pestaña "Pricing" del Excel:
(a) calibración de `Ṽ` contra Monte Carlo (sesgo y RMSE en USD, por tiempo restante), (b) percentil de `p*` en la
serie histórica de la lane (por día de la semana y tiempo restante) y (c) "acepto si tarifa ≥ p*" contra "acepto
la mejor tarifa" y "acepto siempre", bajo el supuesto explícito de que toda carga ofrecida está disponible a su
tarifa observada.

## Flujo de trabajo típico

1. **Entrenar un tamaño nuevo**: fijar `NUM_NODES` en `main.py`, dejar
   `USE_TRANSFER = True` para partir del checkpoint del tamaño anterior en
   `NODE_SEQUENCE` (o `False` para pesos aleatorios). `python main.py`.
2. **Solo evaluar** un checkpoint existente sin reentrenar: `EVAL_ONLY = True`.
3. **Comparar contra baselines**: `main.py` calcula (o recarga) la tabla V de UB-DP,
   evalúa en mundos simulados y en replay, e imprime el resumen.
   `results/<N>nodes/DRL_Routing_Summary_AM.xlsx` trae márgenes, gaps contra UB-DP y
   UB-PI, y las hojas de `stats_analysis.py` (bootstrap y Wilcoxon con Holm).
4. **Entrenar con el proceso exógeno**: `USE_MARKET_PROCESS = True` en
   `main.py` (si falta `datos/market_params.npz` corre `fit_market_process.py`).
   Validar el simulador con `python validate_market_process.py`.
5. **Inferencia puntual**: `python inference.py` con un checkpoint ya
   guardado en `checkpoints/`.

## Requisitos

Ver [requirements.txt](requirements.txt). Entorno principal: PyTorch,
Gymnasium, NumPy/Pandas/SciPy, Optuna, Matplotlib, openpyxl.
