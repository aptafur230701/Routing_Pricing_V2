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
python main.py          # entrena (o evalúa) y guarda resultados en results/<N>nodes/
python inference.py     # inferencia interactiva con un checkpoint ya entrenado
```

Todo lo que hay que tocar para un run distinto está al principio de
`main()` en [main.py](main.py): `NUM_NODES`, `EVAL_ONLY`, `USE_TRANSFER`,
`USE_MARKET_PROCESS`. Los hiperparámetros del modelo y del PPO están todos en
[config.py](config.py).

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
        │      → V(s_t) para la ventaja PPO                  │
        └───────────────────────────────────────────────────┘
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

**Reglas del entorno**: el camión puede volver a pasar por un nodo intermedio dentro del tour (no hay registro de visitados). La máscara solo excluye el self-loop (no hay acción de esperar), los nodos desde los que no se alcanza a volver a casa dentro de `MAX_DURATION` y las lanes sin carga (salvo en el primer movimiento desde casa); volver a casa termina el episodio y con K activo el último arco es el retorno forzado. Las features por nodo son 6 base + 4 de continuación + 1 de índice de mercado (última columna) = 11.

**Por qué el modelo escala sin reentrenar desde cero**: ninguna capa de `AMRoutingAgent` ni de `CriticHead` (`model/`) depende de `num_nodes` — el Transformer opera sobre conjuntos de tamaño variable. Esto es lo que permite el transfer learning en `transfer_learning.py` (cadena 10 → 20 → 50 → 75 → 100 nodos, ver `NODE_SEQUENCE`).

## Estructura de archivos

### Pipeline y modelo (raíz)

| Archivo | Rol |
|---|---|
| [main.py](main.py) | Punto de entrada: carga datos → entrena → evalúa → guarda resultados. |
| [config.py](config.py) | Todas las constantes globales e hiperparámetros (arquitectura, PPO, reward shaping). |
| [problem_data.py](problem_data.py) | Carga de `datos/` y construcción de matrices de recompensa por día. |
| [routing_env.py](routing_env.py) | Entorno vectorizado (`VectorRoutingEnv`) del DVRP estocástico con cotización: en cada nodo la acción es (destino, nivel de precio) y la carga se gana o se pierde frente al mejor competidor. |
| [model/encoder.py](model/encoder.py) | `AttentionEncoder` (Transformer) + `ContextNetwork`. |
| [model/decoder.py](model/decoder.py) | `AttentionDecoder` (pointer network) con glimpse + pointer attention para el destino, y `PriceHead` para el nivel de precio del destino elegido (`act/greedy/evaluate_cotizacion`). |
| [model/critic_head.py](model/critic_head.py) | `CriticHead`: estima V(s_t) para la ventaja PPO. |
| [pricing.py](pricing.py) | Niveles de precio por percentil, competidores efectivos m(LTR), aceptación `(1 − u)^m` y reglas fijas de precio (mediana, break-even + margen, miope). |
| [validate_pricing.py](validate_pricing.py) | Validación de la decisión de precio del DRL: nivel según el estado y sensibilidad a κ (main.py la corre al final de cada tamaño). |
| [model/agent.py](model/agent.py) | `AMRoutingAgent`: combina encoder + contexto + decoder; el rollout se ejecuta paso a paso sobre el entorno. |
| [am_training.py](am_training.py) | Loop de entrenamiento PPO-Clip + GAE. |
| [greedy_rollout.py](greedy_rollout.py) | Rollout greedy (argmax por nodo) sobre `VectorRoutingEnv`: evaluación del DRL en replay de días reales o en mundos simulados, baseline self-critical y monitor de entrenamiento. |
| [model/debug_utils.py](model/debug_utils.py) | `check_tensor`: verificación de NaN/Inf en el forward pass (falla rápido en vez de silenciar con `nan_to_num`). Usado en producción por el encoder/decoder/training, no es un script de debug puntual. |
| [transfer_learning.py](transfer_learning.py) | Warm-start de pesos entre tamaños de grafo (`NODE_SEQUENCE`). |
| [baselines/ub_pi.py](baselines/ub_pi.py) | DP hacia atrás de UB-PI (`solve_ub_pi`), único solver que se conserva: mejor ruta con el mundo completo conocido, estado (nodo, franja de 1 min, k). |
| [inference.py](inference.py) | Inferencia standalone: carga el checkpoint y genera la ruta del DRL desde un nodo dado, con el nivel y el precio cotizados, la mediana, P(ganar) y si se ganó en cada paso. |

### Generador de realizaciones diarias

Cada episodio recorre días REALES consecutivos de jun–oct 2026 (nunca un día inventado), y cada uno se convierte en una realización sintética de cargas y tarifas por lane con (a) binomial negativa centrada en la media de 6 días vecinos, (b) interpolación inversa sobre los cuantiles reales de precio de ese día/lane, y (c) un corrimiento de nivel nacional δ, fijo por episodio. Con `USE_MARKET_PROCESS = False` se usa el replay literal de los días reales. La evaluación usa ambos esquemas: mundos simulados (con δ) y replay de los días de prueba (oct 2026, fuera de muestra).

El índice de mercado por nodo (nivel lento + rápido AR(1), feature del agente y usado por B2P para pronosticar) no se simula: al recorrer siempre días reales, es un dato observado (lookup) para cualquier día, no una trayectoria generada.

| Archivo | Rol |
|---|---|
| [fit_market_process.py](fit_market_process.py) | Estima f_w (factor de día de semana) y r por grupo de volumen (cargas, Paso 4b), el índice de mercado (media por lane, escenarios del nivel lento con `VENTANA_LENTO_DIAS`, ρ y σ_i del rápido) y s (dispersión de δ, Paso 3), con los primeros `TRAIN_DAYS` días de `datos/datos_finales_jun_oct_2026` y los 194 nodos, sin los de prueba, y guarda `datos/market_params.npz`. `main.py` lo corre solo si el archivo no existe. |
| [market_process.py](market_process.py) | `MarketProcess`: `sample_arrivals` (lotes con un Generator, sin estado de mercado ni memoria del día real: la política que ignora el mercado de la DP y B2), `sample_arrivals_por_mundo` (usado por el entorno; determinista por semilla de episodio, nodo y día real), `sample_world` (mundo completo [D, N, N] para baselines), `indice_mundo` e `indice_historico` (ambos son el índice observado del día real; `indice_mundo` conserva el parámetro de semilla por compatibilidad pero lo ignora). |
| [validate_market_process.py](validate_market_process.py) | Paso 6 de la metodología: cuantiles de precios sorteados vs. archivo, dispersión y fracción de ceros de las cargas por grupo de volumen, δ vs. el factor nacional real, y persistencia del índice (pendiente empírica vs. la que implica el ajuste AR(1)). Tablas en `results/market_validation/`. |
| [tests/test_market_process.py](tests/test_market_process.py) | Reproducibilidad, independencia de las acciones, formas, equivalencia con `market_process=None`, recompensa compartida, ajuste y smoke de entrenamiento. |

Constantes (`FECHA_INICIO`, `DIAS_EXCLUIDOS`, `MARKET_R_MAX`, `MARKET_N_GRUPOS`, semillas, M) en
[config.py](config.py). La fórmula de recompensa del paso es `lane_reward` en
[problem_data.py](problem_data.py), compartida por el modo real y el simulado.

### Baselines y evaluación

El DRL se compara contra tres políticas no anticipativas y dos cotas superiores, todas sobre
los mismos mundos simulados (mismo `world_seed`), paso a paso en `VectorRoutingEnv`:

| Id | Baseline | Archivo |
|---|---|---|
| B1 | Greedy miope: arco válido de mayor valor inmediato esperado `max_k P_k (p_k d − c)` (PFA). | [baselines/politicas.py](baselines/politicas.py) |
| B2 | Reoptimización determinista con valores esperados de los arcos futuros; ejecuta solo el primer arco. En la tesis se reporta con K libre (CFA); también admite K = 5. Recursión memoizada sobre (nodo, franja de 15 min, k restante) por (casa, día de salida); el camión puede revisitar nodos. | [baselines/politicas.py](baselines/politicas.py) |
| B2P | B2 con pronósticos del índice de mercado: los arcos futuros j→l a h días de hoy usan `tarifa_media_empírica(j→l) + b_h · índice(j, hoy)`, con `b_h = f + (1 − f) · ρ^h` (f = fracción de varianza lenta, ρ = persistencia del rápido, ambas del ajuste del índice). Un arco futuro vale lo que rinde cotizarlo con la regla miope sobre los cuantiles medios del día de la semana, `prob · P̄_k*(p̄_k* d − c) − (1 − prob) c` (`calcular_ev_esperado`/`_valor_cotizado_por_dow`, promedio directo sobre los días de entrenamiento reales). Está en el arnés de evaluación como un baseline más (columna en el Excel y comparación pareada del DRL), y [evaluar_b2p.py](evaluar_b2p.py) la compara con B2 sin entrenar (salida en `results/b2p/`). La tabla de valores esperados (`calcular_ev_pronostico`) también alimenta las features de continuación del DRL ([resumen_semillas.py](resumen_semillas.py) resume varias semillas). | [baselines/politicas.py](baselines/politicas.py) |
| B3 | Política de un paso sobre la tabla V de la DP de UB-DP con los dos intentos de cotización (VFA de tabla); cotiza con la regla miope. | [baselines/politicas.py](baselines/politicas.py) |
| UB-DP | Valor aproximado de la política que ignora el estado del mercado (no es cota; el DRL, que observa el índice, puede superarlo): DP en (hogar, día de salida, nodo, franja de 15 min, k), con revisitas permitidas y dos intentos de cotización sobre el mismo tablero (salvo la discretización y el muestreo con S = 5000). Tabla en `baselines/datos/ubdp_tabla_<N>nodos.npz`, se recarga si coincide con los datos; se puede calcular aparte con `python -m baselines.dp_relajada --num-nodes N`. Por encima de `UBDP_MAX_NODOS` = 35 nodos no se calcula; y si a 35 o menos supera `UBDP_TIEMPO_MAX_HORAS` o la memoria, deja `ubdp_tabla_<N>nodos.no_tratable.json`. En ambos casos B3 / UB-DP se reportan "no tratable". | [baselines/dp_relajada.py](baselines/dp_relajada.py) |
| UB-PI | Cota con información perfecta: mejor ruta con el mundo completo conocido, incluido el competidor (cotiza en cada arco el nivel más alto que gana; `sample_world` + DP hacia atrás con franjas de 1 min). Debe quedar por encima de UB-DP y del DRL. | [evaluation.py](evaluation.py), [baselines/ub_pi.py](baselines/ub_pi.py) |

| Archivo | Rol |
|---|---|
| [evaluation.py](evaluation.py) | Arnés único (`ejecutar_politica`), `evaluar_mundos_simulados` (M mundos con K libre — `K_RESTRINGIDO = False` en main.py; con True también K = 5 —, nodo y día de la semana balanceados), `evaluar_replay` (días reales de prueba, oct 2026, fuera de muestra), `resumen_metodos`, diagnósticos. |
| [excel_report.py](excel_report.py) | Excel de resultados: hojas Resumen, Mundos simulados y Replay. |
| [stats_analysis.py](stats_analysis.py) | Diferencia pareada DRL − método con IC 95 % por bootstrap, Wilcoxon pareado de una cola con Holm en dos familias (ruteo: B1–B3; precio: comparación B), y gaps contra UB-DP y UB-PI. |
| [tests/test_baselines.py](tests/test_baselines.py) | Tabla V vs enumeración ingenua de los dos intentos, marcador "no tratable", máscara, determinismo, UB-PI ≥ políticas, B3 ≤ UB-DP ≤ UB-PI, UB-PI alineado con el entorno, comparación pareada, y limpieza de nombres borrados (todo a 10 nodos o menos). |
| [tests/test_revisit.py](tests/test_revisit.py) | Revisitas: la máscara de `VectorRoutingEnv` permite un intermedio ya recorrido, `max_pasos`, UB-PI vs fuerza bruta y B2 memoizado vs enumeración ingenua. |

Las políticas solo leen la máscara, el tablero observado (`current_loads_row`, la mediana `rate_row_obs`, los
precios por nivel `precios_row`, m y las opciones válidas), el nodo, la hora, `k_remaining` y los rechazos de la
visita, más los parámetros del proceso; ninguna ve la semilla del mundo ni el competidor. El margen es la suma de
`lane_reward` con el precio cotizado en las cargas ganadas (USD), no la recompensa escalada del entorno. Los tiempos por decisión se miden en CPU con un solo hilo, también para
el DRL.

### Validación final del modelo

| Archivo | Rol |
|---|---|
| [run_k_comparison.py](run_k_comparison.py) | Script aparte: evalúa en mundos simulados uno o varios niveles de K con un checkpoint ya entrenado. |


### Datos y artefactos

| Carpeta | Contenido |
|---|---|
| `datos/` | `distance.csv`/`duration.csv` (matrices fijas por par de hubs), `fuel_cost_hub_jun_oct_2026.csv` (diésel por hub), `datos_finales_jun_oct_2026/` (153 CSV, uno por día jun–oct 2026, con cargas, cuantiles de tarifa y LTR por lane REEFER), `market_params.npz` (ajuste de `fit_market_process.py`, generado). |
| `checkpoints/` | Pesos entrenados (`am_checkpoint_<N>nodes.pt`), compartidos entre tamaños para transfer learning. Ignorado por git (`.gitignore`). |
| `results/<N>nodes/` | Salida de cada run: `DRL_Routing_Summary_AM.xlsx`, diagnósticos PPO/training (`.png`), `training_log_<N>nodes.pkl`/`.xlsx`. |
| `docs/architecture.html` / `.png` | Diagrama explicativo del pipeline AM-PPO. |

## Pricing: cotización conjunta con el ruteo

El precio no es una etapa posterior al ruteo: en cada nodo el DRL elige a la vez el destino `j` y una opción de
precio `o` (especificacion_fase0_ruteo_pricing.md).

- **Niveles.** 10 niveles comunes a todas las lanes en escala de percentiles, `U_NIVELES = 0, 0.1, …, 0.9`; el
  precio en dólares es el cuantil `u_k` de la distribución de tarifas de esa lane y ese día (`pricing.py`). La
  opción 10 es VACÍO (reposicionar sin carga), válida solo hacia casa o en el primer movimiento si la lane no
  es cotizable.
- **Aceptación.** `P(ganar | u_k) = (1 − u_k)^m`, con `m = clip(1 + κ · LTR_REF / LTR, 1, M_MAX)` competidores
  efectivos según el LTR del nodo de origen (κ = `COMPETENCIA_KAPPA` = 1). Se implementa con un competidor
  `u_comp ~ Beta(1, m)` sorteado con la semilla del mundo, así que todos los métodos ven el mismo competidor.
- **Rechazo.** Si se pierde la carga, el camión se queda en el nodo, pierde `COTIZACION_HORAS` (15 min), no
  consume K, ve el mismo tablero sin esa lane y vuelve a decidir. Hay a lo sumo
  `COTIZACIONES_MAX_POR_NODO` = 2 cotizaciones por visita.
- **Reward.** `p · dist − costo` si se gana; `−costo` en un VACÍO.
- **Modelo.** `PriceHead` (model/decoder.py) da el nivel condicionado al destino elegido; la entropía del precio
  tiene su propio coeficiente (`PPO_ENTROPY_COEF_PRECIO_*`). El crítico es `CriticHead` V(s).

**Comparación B.** Para separar lo que aporta la cabeza de precio, la evaluación corre el DRL completo y sus
mismos destinos con una regla fija de precio: `DRL-dest+mediana`, `DRL-dest+be_margen` (break-even +
`BE_MARGEN`) y `DRL-dest+miope` (el nivel que maximiza `P_k (p_k d − c)`). B1–B3 cotizan con la regla miope
(`REGLA_PRECIO_BASELINES`). `stats_analysis.py` las compara en una familia de Holm aparte ("precio").

**Validación del precio.**

```bash
python validate_pricing.py --num-nodes 10 [--checkpoint RUTA] [--quick]
```

`main.py` la corre al final de cada tamaño (o aparte con el comando de arriba sobre un checkpoint). Escribe
`results/<N>nodes/pricing_val/{tables,figures}` y las hojas "Precio vs estado" y "Sensibilidad κ" del Excel,
en mundos con semilla propia (`PRICING_VAL_SEED`): (a) nivel cotizado por el DRL por quintil de LTR, m,
tiempo restante e índice de mercado de origen y destino, con Spearman; (b) sensibilidad a κ ∈ {0.5, 1, 2},
con el DRL entrenado en κ = 1 y la regla miope como referencia. La comparación B la hace la evaluación.

**Limitaciones.** (1) La aceptación es un modelo (no hay datos de cargas ganadas y perdidas): κ fija qué tanto
pesa el LTR, y por eso se reporta la sensibilidad. (2) B3 elige el destino con la DP de dos intentos pero cotiza
con la regla miope, no con el nivel óptimo de la DP. (3) UB-DP cuesta unas 20 veces más que sin precio: a 50, 75 y 100 nodos
(más de `UBDP_MAX_NODOS`) no se calcula, y B3 y UB-DP se reportan "no tratable".

## Flujo de trabajo típico

1. **Entrenar un tamaño nuevo**: fijar `NUM_NODES` en `main.py`, dejar
   `USE_TRANSFER = True` para partir del checkpoint del tamaño anterior en
   `NODE_SEQUENCE` (o `False` para pesos aleatorios). `python main.py`.
2. **Solo evaluar** un checkpoint existente sin reentrenar: `EVAL_ONLY = True`.
3. **Comparar contra baselines**: `main.py` calcula (o recarga) la tabla V de UB-DP,
   evalúa en mundos simulados y en replay, e imprime el resumen.
   `results/<N>nodes/DRL_Routing_Summary_AM.xlsx` trae márgenes, gaps contra UB-DP y
   UB-PI, y las hojas de `stats_analysis.py` (bootstrap y Wilcoxon con Holm).
4. **Entrenar con el generador de realizaciones diarias**: `USE_MARKET_PROCESS = True` en
   `main.py` (si falta `datos/market_params.npz` corre `fit_market_process.py`).
   Validar el generador con `python validate_market_process.py`.
5. **Inferencia puntual**: `python inference.py` con un checkpoint `am_checkpoint_<N>nodes.pt` ya
   guardado en `checkpoints/`.
6. **Validar el precio**: `python validate_pricing.py --num-nodes N` sobre ese checkpoint.

## Requisitos

Ver [requirements.txt](requirements.txt). **Python 3.10 – 3.12** (el piso lo marca `requires_python >= 3.10` de los
pins; probado en 3.10.16 y 3.12.2). Entorno principal: PyTorch,
Gymnasium, NumPy/Pandas/SciPy, Optuna, Matplotlib, openpyxl.
