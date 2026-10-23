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

**Reglas del entorno**: el camión puede volver a pasar por un nodo intermedio dentro del tour (no hay
registro de visitados). La máscara solo excluye el self-loop (no hay acción de esperar), los nodos desde
los que no se alcanza a volver a casa dentro de `MAX_DURATION` y las lanes sin carga (salvo en el primer
movimiento desde casa); volver a casa termina el episodio y con K activo el último arco es el retorno
forzado. Las features por nodo son 7 base + 4 de continuación + 1 de índice de mercado (última columna) = 12.

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
| [critic_head.py](critic_head.py) | `CriticHead`: estima V(s_t) para la ventaja PPO. `PostDecisionValueHead`: Ṽ(s, j), valor de llegar a j antes de ver su tablero (precio de reserva); el agente trae dos (A y B). |
| [pricing.py](pricing.py) | `reservation_price`: precio de reserva p* por lane (parejas AB/BA, rango); `crear_selector_q`: ruta DRL-Q (argmax de Q). |
| [post_training.py](post_training.py) | Fase posterior al PPO: entrena las cabezas Ṽ A/B con el actor congelado y guarda `am_checkpoint_<N>nodes_pricing.pt`. |
| [validate_pricing.py](validate_pricing.py) | Validación de p* y de DRL-Q: contrafactual, utilidad (márgenes pareados, regla por umbral), plausibilidad. |
| [am_agent.py](am_agent.py) | `AMRoutingAgent`: combina encoder + contexto + decoder; el rollout se ejecuta paso a paso sobre el entorno. |
| [am_training.py](am_training.py) | Loop de entrenamiento PPO-Clip + GAE. |
| [greedy_rollout.py](greedy_rollout.py) | Rollout greedy (argmax por nodo) sobre `VectorRoutingEnv`: evaluación del DRL en replay de días reales o en mundos simulados, baseline self-critical y monitor de entrenamiento. |
| [debug_utils.py](debug_utils.py) | `check_tensor`: verificación de NaN/Inf en el forward pass (falla rápido en vez de silenciar con `nan_to_num`). Usado en producción por el encoder/decoder/training, no es un script de debug puntual. |
| [transfer_learning.py](transfer_learning.py) | Warm-start de pesos entre tamaños de grafo (`NODE_SEQUENCE`). |
| [Solvers.py](Solvers.py) | DP hacia atrás de UB-PI (`solve_ub_pi`), único solver que se conserva: mejor ruta con el mundo completo conocido, estado (nodo, franja de 1 min, k). |
| [inference.py](inference.py) | Inferencia standalone: carga el checkpoint con las cabezas Ṽ y genera la ruta de DRL-Q desde un nodo dado, con el precio de reserva p* por arco. |

### Proceso exógeno de cargas y tarifas (opcional)

Diseño completo en [proceso_exogeno_cargas_tarifas.md](proceso_exogeno_cargas_tarifas.md).
Con `USE_MARKET_PROCESS = True` el entrenamiento ya no reproduce días del
histórico: al llegar a cada nodo se sortean las cargas (binomial negativa con
factor de día de la semana) y las tarifas (media de la lane + índice de mercado del nodo + residuo; el índice es un nivel lento fijo en el mundo más una variación diaria AR(1), y el agente lo observa como feature).
Con `False` el entrenamiento reproduce días reales como antes. La evaluación siempre usa el proceso.

| Archivo | Rol |
|---|---|
| [fit_market_process.py](fit_market_process.py) | Estima s_d, mu_ij y phi_h, y el modelo de tarifas (media por lane, escenarios del nivel lento con `VENTANA_LENTO_DIAS`, ρ y σ_i del rápido, residuos), con los primeros `TRAIN_DAYS` días del histórico y los 194 nodos, sin los de evaluación, y guarda `datos/market_params.npz`. `main.py` lo corre solo si el archivo no existe. |
| [market_process.py](market_process.py) | `MarketProcess`: `sample_arrivals` (lotes con un Generator, sin estado de mercado: la política que ignora el mercado de la DP y B2), `sample_arrivals_por_mundo` (usado por el entorno; determinista por semilla de mundo, nodo y día, con lento fijo y rápido AR(1) del mundo), `sample_world` (mundo completo [D, N, N] para baselines), `indice_mundo` e `indice_historico` (índice de mercado simulado y observado en días reales). |
| [validate_market_process.py](validate_market_process.py) | Simula 200 mundos del periodo de ajuste (dentro de muestra) y compara cargas por lane, cuantiles de tarifa por lane y persistencia del índice (h = 1, 3, 5, 7) con el histórico (tablas y gráficos en `results/market_validation/`). |
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
| B2 | Reoptimización determinista con valores esperados de los arcos futuros; ejecuta solo el primer arco. Con K = 5 y con K libre (CFA). Recursión memoizada sobre (nodo, franja de 15 min, k restante) por (casa, día de salida); el camión puede revisitar nodos. | [baselines.py](baselines.py) |
| B2P | B2 con pronósticos del índice de mercado: los arcos futuros j→l a h días de hoy usan `media_lane + b_h · índice(j, hoy)`, con `b_h = f + (1 − f) · ρ^h` (f = fracción de varianza lenta, ρ = persistencia del rápido, ambas del ajuste). Está en el arnés de evaluación como un baseline más (columna en el Excel y comparación pareada del DRL), y [evaluar_b2p.py](evaluar_b2p.py) la compara con B2 sin entrenar (salida en `results/b2p/`). La tabla de valores esperados (`calcular_ev_pronostico`) también alimenta las features de continuación del DRL ([resumen_semillas.py](resumen_semillas.py) resume varias semillas). | [baselines.py](baselines.py) |
| B3 | Política de un paso sobre la tabla V de la DP de UB-DP (VFA de tabla). | [dp_relajada.py](dp_relajada.py) |
| UB-DP | Valor aproximado de la política que ignora el estado del mercado (no es cota; el DRL, que observa el índice, puede superarlo): DP en (hogar, día de salida, nodo, franja de 15 min, k), con revisitas permitidas (salvo la discretización y el muestreo con S = 5000). Tabla en `datos/ubdp_tabla_<N>nodos.npz`, se recarga si coincide con los datos. | [dp_relajada.py](dp_relajada.py) |
| UB-PI | Cota con información perfecta: mejor ruta con el mundo completo conocido (`sample_world` + DP hacia atrás con franjas de 1 min). Debe quedar por encima de UB-DP y del DRL. | [evaluation.py](evaluation.py), [Solvers.py](Solvers.py) |

| Archivo | Rol |
|---|---|
| [evaluation.py](evaluation.py) | Arnés único (`ejecutar_politica`), `evaluar_mundos_simulados` (M mundos por K ∈ {5, libre}, nodo y día de la semana balanceados), `evaluar_replay` (días reales 90-119, fuera de muestra: "replay, fuera de muestra (días 90-119)"), `resumen_metodos`, diagnósticos. |
| [excel_report.py](excel_report.py) | Excel de resultados: hojas Resumen, Mundos simulados y Replay. |
| [stats_analysis.py](stats_analysis.py) | Diferencia pareada DRL − baseline con IC 95 % por bootstrap, Wilcoxon pareado de una cola con Holm, y gaps contra UB-DP y UB-PI. |
| [tests/test_baselines.py](tests/test_baselines.py) | Tabla V vs recursión ingenua, máscara, determinismo, UB-PI ≥ políticas, B3 ≤ UB-DP ≤ UB-PI, margen = Σ `lane_reward`, UB-PI alineado con el entorno, y limpieza de nombres borrados. |
| [tests/test_revisit.py](tests/test_revisit.py) | Revisitas: la máscara de `RoutingEnv` y `VectorRoutingEnv` permite un intermedio ya recorrido, `max_pasos`, UB-PI vs fuerza bruta y B2 memoizado vs enumeración ingenua. |

Las políticas solo leen la máscara, la fila observada (`current_loads_row`, `current_rate_row`),
el nodo, la hora y `k_remaining`, más los parámetros del proceso; ninguna ve la
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

## Pricing: precio de reserva y ruta DRL-Q

Descripción completa en [PRICING_PLAN.md](PRICING_PLAN.md). El modelo entrega dos salidas: la **ruta** y, para cada
arco con carga, el **precio mínimo por milla** (p\*) al que al camión le conviene aceptar la lane frente a su mejor
alternativa. Es una recomendación del lado del transportista; **no modela aceptación** (no hay datos de aceptación
y rechazo) y p\* es un costo de oportunidad, no una probabilidad.

**Modelo.** Con `S = REWARD_SCALE_FACTOR` y `r_obs = tarifa·dist·1[carga] − costo`:

```
Q(s, a)  = r_obs(i→a)/S + Ṽ(s, a)                     Ṽ(s, casa) = 0
p*(i→j)  = ( S · (Q_alt − Ṽ(s, j)) + costo(i→j) ) / dist(i→j)      [USD/milla],   Q_alt = mejor Q entre a ≠ j válidas
```

`Ṽ(s, a)` es el margen esperado desde la llegada a `a` hasta volver a casa siguiendo el actor en modo greedy.
Hay dos cabezas, A y B, entrenadas con mitades distintas de los mundos: p\*_AB elige la alternativa con A y evalúa
con B (p\*_BA, al revés); p\* es el promedio y el **rango** [mín, máx] es la variabilidad del estimador (partición
de datos y ruido de etiquetas), **no un intervalo de confianza** (comparten encoder, contexto y actor).

**Ruta.** *DRL-Q* elige en cada nodo el argmax de Q con `Ṽ = (Ṽ_A + Ṽ_B)/2`; es la ruta que se entrega y está en el
arnés de evaluación junto al DRL de siempre (el actor) y los baselines.

**Entrenamiento.** El PPO no entrena Ṽ. Después del PPO (`post_training.py`), con encoder, contexto y actor
congelados: se generan ~200 000 ejemplos (estado de una ruta del actor, acción al azar entre las válidas salvo volver
a casa, y margen real posterior con el actor greedy en el mismo mundo) y se ajustan A y B. Config: `USE_POST_HEAD`,
`POST_*`, `MARKET_SEED_POST_*` en `config.py`. Solo se calcula p\* para lanes con carga (no para el retorno ni para
arcos de reposicionamiento).

**Cómo ejecutarlo.**

```bash
python main.py                                   # PPO (o checkpoint) → Ṽ A/B → evaluación con DRL-Q → validación
python post_training.py --num-nodes 10           # solo la fase de Ṽ (checkpoint del PPO → *_pricing.pt)
python validate_pricing.py --num-nodes 10        # solo la validación sobre el checkpoint *_pricing.pt
python inference.py [--show-board]               # ruta DRL-Q + p* por arco; escribe results/<N>nodes/pricing_<run_tag>.csv
```

La validación escribe `results/<N>nodes/pricing_val/{figures,tables}` y la pestaña "Pricing" del Excel:
(b) prueba contrafactual del sesgo de p\* (IC95 por bootstrap de mundos), (c) margen de DRL-Q contra DRL, B1, B2, B2P y B3
(pareado) y regla de aceptación por umbral con tres pisos (break-even, break-even + 0,30 USD/mi y p\*), y
(d) percentil de p\* en las tarifas históricas de la lane y correlación con el índice de mercado del destino.

**Limitaciones.** (1) p\* es un costo de oportunidad, no una probabilidad de aceptación. (2) Ṽ se entrena con una
sola realización por ejemplo, así que el error es alto por construcción. (3) Con K libre la ruta típica es una lane
con carga más el retorno. (4) Ṽ sigue al actor, no a DRL-Q.

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
5. **Inferencia puntual**: `python inference.py` con un checkpoint `*_pricing.pt` ya
   guardado en `checkpoints/`.

## Requisitos

Ver [requirements.txt](requirements.txt). Entorno principal: PyTorch,
Gymnasium, NumPy/Pandas/SciPy, Optuna, Matplotlib, openpyxl.
