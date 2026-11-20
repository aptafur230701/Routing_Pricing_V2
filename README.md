# AM-PPO Routing & Pricing Model

Agente de ruteo basado en un **Attention Model (Transformer encoder-decoder)**
entrenado con **PPO (Proximal Policy Optimization)**, para un problema de
ruteo estocástico con revelación post-decisión de disponibilidad de carga
(Stochastic DVRP). El agente decide, en cada nodo, a qué nodo moverse a
continuación para maximizar la recompensa acumulada de una ruta completa dentro
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
problem_data.py ──► matrices (tiempo, tarifa, carga, distancia, diesel, LTR)
                              │
                              ▼
                    routing_env.py (RoutingEnv / VectorRoutingEnv)
                              ▼
        ┌───────────────────────────────────────────────────┐
        │                  model/agent.py                    │
        │  AttentionEncoder  (model/encoder.py)              │
        │      → embeddings por nodo + embedding global      │
        │  ContextNetwork    (model/encoder.py)              │
        │      → vector de contexto h_t                      │
        │  AttentionDecoder  (model/decoder.py)              │
        │      → distribución π(a|s) sobre nodos válidos     │
        │  CriticHead        (model/critic_head.py)          │
        │      → V(s_t) para la ventaja PPO                  │
        │  PostDecisionValueHead (model/critic_head.py)      │
        │      → Ṽ(s, j) para el precio de reserva p*        │
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

**Por qué el modelo escala sin reentrenar desde cero**: ninguna capa de `AMRoutingAgent` ni de `CriticHead` (`model/`) depende de `num_nodes` — el Transformer opera sobre conjuntos de tamaño variable. Esto es lo que permite el transfer learning en `transfer_learning.py` (cadena 10 → 20 → 35 → 50 → 75 → 100 nodos, ver `NODE_SEQUENCE`).

## Estructura de archivos

### Pipeline y modelo (raíz)

| Archivo | Rol |
|---|---|
| [main.py](main.py) | Punto de entrada: carga datos → entrena → evalúa → guarda resultados. |
| [config.py](config.py) | Todas las constantes globales e hiperparámetros (arquitectura, PPO, reward shaping). |
| [problem_data.py](problem_data.py) | Carga de `datos/` y construcción de matrices de recompensa por día. |
| [routing_env.py](routing_env.py) | Entorno Gymnasium (`RoutingEnv`, `VectorRoutingEnv`) para el DVRP estocástico. |
| [model/encoder.py](model/encoder.py) | `AttentionEncoder` (Transformer) + `ContextNetwork`. |
| [model/decoder.py](model/decoder.py) | `AttentionDecoder` (pointer network) con glimpse + pointer attention. |
| [model/critic_head.py](model/critic_head.py) | `CriticHead`: estima V(s_t) para la ventaja PPO. `PostDecisionValueHead`: Ṽ(s, j), valor de llegar a j antes de ver su tablero (precio de reserva); el agente trae dos (A y B). |
| [pricing.py](pricing.py) | `reservation_price`: precio de reserva p* por lane (parejas AB/BA, rango) sobre el arco que el actor ya elige. |
| [post_training.py](post_training.py) | Fase posterior al PPO: entrena las cabezas Ṽ A/B con el actor congelado y guarda `am_checkpoint_<N>nodes_pricing.pt`. |
| [validate_pricing.py](validate_pricing.py) | Validación de p* sobre la ruta del DRL: contrafactual, utilidad (regla por umbral), plausibilidad. |
| [model/agent.py](model/agent.py) | `AMRoutingAgent`: combina encoder + contexto + decoder; el rollout se ejecuta paso a paso sobre el entorno. |
| [am_training.py](am_training.py) | Loop de entrenamiento PPO-Clip + GAE. |
| [greedy_rollout.py](greedy_rollout.py) | Rollout greedy (argmax por nodo) sobre `VectorRoutingEnv`: evaluación del DRL en replay de días reales o en mundos simulados, baseline self-critical y monitor de entrenamiento. |
| [model/debug_utils.py](model/debug_utils.py) | `check_tensor`: verificación de NaN/Inf en el forward pass (falla rápido en vez de silenciar con `nan_to_num`). Usado en producción por el encoder/decoder/training, no es un script de debug puntual. |
| [transfer_learning.py](transfer_learning.py) | Warm-start de pesos entre tamaños de grafo (`NODE_SEQUENCE`). |
| [baselines/ub_pi.py](baselines/ub_pi.py) | DP hacia atrás de UB-PI (`solve_ub_pi`), único solver que se conserva: mejor ruta con el mundo completo conocido, estado (nodo, franja de 1 min, k). |
| [inference.py](inference.py) | Inferencia standalone: carga el checkpoint con las cabezas Ṽ y genera la ruta del DRL desde un nodo dado, con el precio de reserva p* por arco. |

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
| B1 | Greedy miope: arco válido de mayor `lane_reward` de la fila observada (PFA). | [baselines/politicas.py](baselines/politicas.py) |
| B2 | Reoptimización determinista con valores esperados de los arcos futuros; ejecuta solo el primer arco. Con K = 5 y con K libre (CFA). Recursión memoizada sobre (nodo, franja de 15 min, k restante) por (casa, día de salida); el camión puede revisitar nodos. | [baselines/politicas.py](baselines/politicas.py) |
| B2P | B2 con pronósticos del índice de mercado: los arcos futuros j→l a h días de hoy usan `tarifa_media_empírica(j→l) + b_h · índice(j, hoy)`, con `b_h = f + (1 − f) · ρ^h` (f = fracción de varianza lenta, ρ = persistencia del rápido, ambas del ajuste del índice). La tarifa media empírica y la probabilidad de carga salen de un promedio directo sobre los días de entrenamiento reales (`calcular_ev_esperado`/`_promedios_empiricos_por_dow`), ya no de un modelo paramétrico. Está en el arnés de evaluación como un baseline más (columna en el Excel y comparación pareada del DRL), y [evaluar_b2p.py](evaluar_b2p.py) la compara con B2 sin entrenar (salida en `results/b2p/`). La tabla de valores esperados (`calcular_ev_pronostico`) también alimenta las features de continuación del DRL ([resumen_semillas.py](resumen_semillas.py) resume varias semillas). | [baselines/politicas.py](baselines/politicas.py) |
| B3 | Política de un paso sobre la tabla V de la DP de UB-DP (VFA de tabla). | [baselines/politicas.py](baselines/politicas.py) |
| UB-DP | Valor aproximado de la política que ignora el estado del mercado (no es cota; el DRL, que observa el índice, puede superarlo): DP en (hogar, día de salida, nodo, franja de 15 min, k), con revisitas permitidas (salvo la discretización y el muestreo con S = 5000). Tabla en `datos/ubdp_tabla_<N>nodos.npz`, se recarga si coincide con los datos. | [baselines/dp_relajada.py](baselines/dp_relajada.py) |
| UB-PI | Cota con información perfecta: mejor ruta con el mundo completo conocido (`sample_world` + DP hacia atrás con franjas de 1 min). Debe quedar por encima de UB-DP y del DRL. | [evaluation.py](evaluation.py), [baselines/ub_pi.py](baselines/ub_pi.py) |

| Archivo | Rol |
|---|---|
| [evaluation.py](evaluation.py) | Arnés único (`ejecutar_politica`), `evaluar_mundos_simulados` (M mundos por K ∈ {5, libre}, nodo y día de la semana balanceados), `evaluar_replay` (días reales de prueba, oct 2026, fuera de muestra), `resumen_metodos`, diagnósticos. |
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
| `datos/` | `distance.csv`/`duration.csv` (matrices fijas por par de hubs), `fuel_cost_hub_jun_oct_2026.csv` (diésel por hub), `datos_finales_jun_oct_2026/` (153 CSV, uno por día jun–oct 2026, con cargas, cuantiles de tarifa y LTR por lane REEFER), `market_params.npz` (ajuste de `fit_market_process.py`, generado). |
| `checkpoints/` | Pesos entrenados (`am_checkpoint_<N>nodes.pt`), compartidos entre tamaños para transfer learning. Ignorado por git (`.gitignore`). |
| `results/<N>nodes/` | Salida de cada run: `DRL_Routing_Summary_AM.xlsx`, diagnósticos PPO/training (`.png`), `training_log_<N>nodes.pkl`/`.xlsx`. |
| `docs/architecture.html` / `.png` | Diagrama explicativo del pipeline AM-PPO. |

## Pricing: precio de reserva por arco

El DRL (el actor del PPO) es el único método de ruteo; el pricing es estrictamente downstream de
esa ruta ya decidida: para cada arco con carga calcula el **precio mínimo
por milla** (p\*) al que al camión le conviene aceptar esa lane frente a su mejor alternativa. p\* nunca re-decide
qué lane tomar — es una recomendación del lado del transportista sobre la decisión que el actor ya hizo. **No
modela aceptación** (no hay datos de aceptación y rechazo) y p\* es un costo de oportunidad, no una probabilidad.

**Modelo.** Con `S = REWARD_SCALE_FACTOR` y `r_obs = tarifa·dist·1[carga] − costo`:

```
Q(s, a)  = r_obs(i→a)/S + Ṽ(s, a)                     Ṽ(s, casa) = 0
p*(i→j)  = ( S · (Q_alt − Ṽ(s, j)) + costo(i→j) ) / dist(i→j)      [USD/milla],   Q_alt = mejor Q entre a ≠ j válidas
```

`j` es la acción que el actor ya tomó; Q solo se evalúa para valorar `Q_alt`, la mejor alternativa a `j` (el costo
de oportunidad), nunca para elegir la acción real. `Ṽ(s, a)` es el margen esperado desde la llegada a `a` hasta
volver a casa siguiendo el actor en modo greedy. Hay dos cabezas, A y B, entrenadas con mitades distintas de los
mundos: p\*_AB elige la alternativa con A y evalúa con B (p\*_BA, al revés); p\* es el promedio y el **rango**
[mín, máx] es la variabilidad del estimador (partición de datos y ruido de etiquetas), **no un intervalo de
confianza** (comparten encoder, contexto y actor).

El LTR (load-to-truck ratio) del nodo **actual** `i` (de donde sale el arco, no del destino) no se agrega como
feature a Ṽ: alimentar un valor que es el mismo para todas las alternativas desde un mismo estado se cancelaría
casi exactamente en `Q_alt − Ṽ(s, j)`, que por construcción solo es sensible a diferencias ENTRE alternativas del
mismo estado. Ya está implícito en `r_obs` (si el mercado en `i` está tensionado, debería subir la tarifa
observada de todas las lanes que salen de `i`, no solo de una). `validate_pricing.py` correlaciona (p\* −
break-even) con el LTR de `i` en la prueba de plausibilidad (d) — esperada positiva: mercado tensionado en `i` ⇒
más poder de negociación sobre las lanes que salen de ahí ⇒ p\* más alto — sin tocar el modelo ni reentrenar nada.

**Entrenamiento.** El PPO no entrena Ṽ. Después del PPO (`post_training.py`), con encoder, contexto y actor
congelados: se generan ~200 000 ejemplos (estado de una ruta del actor, acción al azar entre las válidas salvo volver
a casa, y margen real posterior con el actor greedy en el mismo mundo) y se ajustan A y B. Config: `USE_POST_HEAD`,
`POST_*`, `MARKET_SEED_POST_*` en `config.py`. Solo se calcula p\* para lanes con carga (no para el retorno ni para
arcos de reposicionamiento).

**Cómo ejecutarlo.**

```bash
python main.py                                   # PPO (o checkpoint) → Ṽ A/B → evaluación → validación del pricing
python post_training.py --num-nodes 10           # solo la fase de Ṽ (checkpoint del PPO → *_pricing.pt)
python validate_pricing.py --num-nodes 10        # solo el pricing sobre el checkpoint *_pricing.pt (no toca el ruteo)
python inference.py [--show-board]               # ruta del DRL + p* por arco; escribe results/<N>nodes/pricing_<run_tag>.csv
```

`validate_pricing.py` (solo o dentro de `main.py`) nunca entrena ni evalúa el ruteo contra B1-B3/UB-PI/UB-DP — eso
es trabajo de `evaluation.py`, que ya corre esa comparación para el DRL.

La validación escribe `results/<N>nodes/pricing_val/{figures,tables}` y la pestaña "Pricing" del Excel:
(b) prueba contrafactual del sesgo de p\* (IC95 por bootstrap de mundos), (c) % de arcos con p\* bajo el break-even
y regla de aceptación por umbral con tres pisos (break-even, break-even + 0,30 USD/mi y p\*), y
(d) percentil de p\* en las tarifas históricas de la lane y correlación con el índice de mercado del destino.

**Limitaciones.** (1) p\* es un costo de oportunidad, no una probabilidad de aceptación. (2) Ṽ se entrena con una
sola realización por ejemplo, así que el error es alto por construcción. (3) Con K libre la ruta típica es una lane
con carga más el retorno.

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
5. **Inferencia puntual**: `python inference.py` con un checkpoint `*_pricing.pt` ya
   guardado en `checkpoints/`.

## Requisitos

Ver [requirements.txt](requirements.txt). **Python 3.10 – 3.12** (el piso lo marca `requires_python >= 3.10` de los
pins; probado en 3.10.16 y 3.12.2). Entorno principal: PyTorch,
Gymnasium, NumPy/Pandas/SciPy, Optuna, Matplotlib, openpyxl.
