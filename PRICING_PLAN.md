# Precio de reserva y ruta DRL-Q (lo implementado)

El modelo entrega **dos decisiones por episodio**: la ruta que el camión debe seguir y, para cada arco con carga de esa
ruta, el **precio de reserva p\*** (USD/milla), el mínimo al que le conviene aceptar la lane frente a su mejor
alternativa. Es una recomendación del lado del transportista: **no modela aceptación** (no hay datos de aceptación y
rechazo) y p\* es un costo de oportunidad, no una probabilidad.

Ambas salidas salen de la misma función de valor Q, construida sobre el actor ya entrenado con PPO (que no cambia).

---

## 1. Definiciones

Con `S = REWARD_SCALE_FACTOR`, en el nodo `i` con la fila observada de cargas y tarifas:

```
r_obs(i→a) = tarifa(i→a) · dist(i→a) · 1[hay carga] − costo(i→a)          (sin redondear)
Q(s, a)    = r_obs(i→a)/S + Ṽ(s, a)                                          Ṽ(s, casa) = 0
Q_alt      = mejor Q entre las acciones válidas distintas de j
p*(i→j)    = ( S · (Q_alt − Ṽ(s, j)) + costo(i→j) ) / dist(i→j)              [USD/milla]
```

* **Ṽ(s, a)**: margen esperado desde la llegada a `a` (antes de ver su tablero) hasta volver a casa, **siguiendo
  después la política base**: el actor en modo greedy (determinista).
* p\* es la tarifa a la que `Q(s, j) = Q_alt`: puede quedar bajo el break-even `costo/dist` cuando `j` es una
  posición valiosa. Es invariante en `S` (Ṽ está en USD/S).
* **Solo se calcula para lanes con carga.** No lleva p\*: el retorno a casa (su línea dice
  `cost/dist (retorno al depot, Ṽ = 0)`), un arco sin carga (reposicionamiento: el entorno no filtra lanes sin carga
  en el paso 0, desde el depot) ni el caso `only_option` (j es la única acción factible, no hay alternativa).
* `r_obs` se lee de la fila observada del entorno (sorteada al llegar al nodo en simulado; del histórico en
  replay: `VectorRoutingEnv.rate_row_obs / loads_row_obs`).

## 2. Dos cabezas Ṽ (A y B)

`agent.post_value_a` y `agent.post_value_b` (`PostDecisionValueHead`, mismo MLP `[h_t, e_a] → Ṽ`, inicializaciones
distintas). Cada una se entrena con **mitades distintas de los mundos** de entrenamiento (la partición es por mundo,
no por ejemplo: los ejemplos de un mismo mundo comparten el ruido de la realización).

```
p*_AB: A elige la mejor alternativa; B evalúa Q_alt y Ṽ(s, j).     p*_BA: al revés.
p* = (p*_AB + p*_BA) / 2          rango = [min, max] de los dos
```

El **rango mide la variabilidad del estimador** (partición de datos y ruido de las etiquetas); **no es un intervalo de
confianza**: las dos cabezas comparten encoder, contexto y actor.

## 3. Ruta DRL-Q

Política nueva que, en cada nodo, elige el **argmax de Q** sobre la máscara del entorno, con `Ṽ = (Ṽ_A + Ṽ_B)/2`
(`pricing.crear_selector_q`, usado por `greedy_rollout(selector=...)`). Es **la ruta que se entrega**. Está en el arnés
de `evaluation.py` como método `DRL-Q`, junto al DRL de siempre (el actor) y B1, B2, B2P y B3.

## 4. Entrenamiento de Ṽ: fase posterior al PPO (`post_training.py`)

El PPO ya **no** entrena Ṽ (se quitó `loss_post`; el PPO queda bit a bit igual y el agente del PPO no trae cabezas).
Con encoder, contexto y actor congelados:

1. **Estados**: rutas del actor (muestreado) en mundos nuevos de entrenamiento (`MARKET_SEED_POST_TRAIN`), K libre.
2. En cada estado se elige una **acción al azar entre las válidas excepto volver a casa** y, desde la llegada, se
   sigue con el **actor greedy hasta casa en el mismo mundo** (`Escenario.reproducir`: el mundo es función pura de
   (semilla, nodo, día), así que se reanuda reproduciendo el prefijo de acciones).
3. **Objetivo** = margen real posterior a llegar a la acción (suma de `last_raw_reward` de los pasos siguientes) / S.
4. `POST_N_EJEMPLOS` = 200 000 ejemplos (se aumenta si el error de validación sigue bajando entre 50 %, 75 % y 100 %
   de los datos: `post_training_curva.csv`), validados en `POST_N_VAL` ejemplos de mundos retenidos
   (`MARKET_SEED_POST_VAL`).
5. Adam, parada temprana con el MSE de validación. Se guarda un **checkpoint nuevo**
   `am_checkpoint_<N>nodes_pricing.pt`; el del PPO queda intacto.

## 5. Mapa de archivos

| Archivo | Papel |
|---|---|
| `model/critic_head.py` | `PostDecisionValueHead(seed)` (+ `forward_pairs`); detach interno de `h_t` y embeddings |
| `model/agent.py` | `post_value_a/b`, `post_values(h_t, emb)`, `post_entrenadas` |
| `am_training.py` | PPO sin pérdida de Ṽ (las cabezas, si existen, quedan fuera de los optimizers) |
| `post_training.py` | generación de ejemplos, ajuste de A/B, curva de error, checkpoint `_pricing` |
| `pricing.py` | `q_values`, `reservation_price` (A/B, devuelve `PrecioReserva`), `crear_selector_q`, `percentil_historico` |
| `greedy_rollout.py` | parámetro `selector` (por defecto el argmax del actor) |
| `routing_env.py` | `rate_row_obs` / `loads_row_obs` |
| `evaluation.py`, `stats_analysis.py` | método `DRL-Q`; comparaciones pareadas de DRL-Q; bootstrap por conglomerados |
| `validate_pricing.py` | `Escenario` (rollouts por lotes) y la validación (§6) |
| `inference.py` | ruta DRL-Q + p\* por arco (§8) y CSV |
| `main.py` | PPO → fase de Ṽ → evaluación con DRL-Q → validación (`RUN_PRICING`, `RETRAIN_POST`) |
| `tests/` | pruebas de §7 |

## 6. Validación (`validate_pricing.py`)

* **(a) Pruebas internas** (`tests/test_pricing.py`): con la tarifa de `j` igual a p\*, `Q(s, j) = Q_alt` (para AB y
  BA); p\* sube con el costo de `j`; el resultado no depende de `S`.
* **(b) Contrafactual**: `N_EST_CONTRAFACTUAL` = 2000 estados de rutas de DRL-Q en mundos de validación
  (`MARKET_SEED_VAL`, ≠ entrenamiento, Ṽ y evaluación). Desde el estado, en el mismo mundo, se sigue la lane `j` de
  DRL-Q y, por separado, la mejor alternativa según el modelo (la de A para p\*_AB, la de B para p\*_BA); después, el
  actor greedy hasta casa. `p*_realizado = (G_alt − C_j + costo_j)/dist_j` (G_alt: margen total desde el estado
  tomando la alternativa, con su carga; C_j: margen posterior a llegar a `j`). Reporta el sesgo medio
  (p\* predicho − realizado) en USD/milla y USD/lane con **IC95 por bootstrap que remuestrea mundos** (los estados de
  un mismo mundo no son independientes), la correlación y el sesgo por tramo (< 300, 300-800, > 800 mi, con n; a
  N = 10 no hay lanes < 300 mi). Se expresa también como fracción de la dispersión semanal del índice de mercado
  (0,10-0,12 USD/mi).
* **(c) Utilidad**, en los mundos simulados de evaluación y en replay (días de prueba, oct 2026, fuera de muestra): margen de DRL-Q contra DRL,
  B1, B2, B2P y B3 (diferencias pareadas, IC95 por mundo); diferencia relativa al margen del actor con su IC95;
  % de arcos de DRL-Q con p\* bajo el break-even y sobre break-even + 0,30 USD/mi; y la **regla de aceptación por
  umbral** con tres pisos (break-even, break-even + 0,30, p\*): entre las lanes con carga válidas con tarifa
  observada ≥ umbral de la lane toma la de mayor (tarifa − umbral)·dist; si ninguna califica vuelve a casa; en el
  paso 0 usa la acción de DRL-Q. Diferencias pareadas (p\* − piso) con IC95 por mundo.
* **(d) Plausibilidad**: percentil de p\* en las tarifas históricas de la lane (días de entrenamiento **con carga**:
  el histórico trae tarifa también en días sin carga) y correlación (Pearson y Spearman) de (p\* − break-even) con el
  índice de mercado del destino. Se espera correlación **negativa**: mayor índice ⇒ mayor Ṽ(j) ⇒ p\* menor.

**Criterios de parada** (se reporta antes de cambiar el diseño): el límite superior del IC95 de
(DRL-Q − DRL)/margen del actor es < −2 %, o |sesgo medio de p\*| > 0,05 USD/milla.
Riesgo conocido: *winner's curse* — DRL-Q elige `j` donde Ṽ_j tiende a estar sobreestimada, lo que sesga p\* a la baja.

## 6b. Excel de resultados

`main.py` agrega al `DRL_Routing_Summary_*.xlsx` dos hojas (`validate_pricing.append_pricing_sheet`):

* **Pricing**: error de Ṽ en validación por nº de ejemplos, contrafactual (sesgo e IC95, por tramo), resumen de arcos
  de DRL-Q (% bajo el break-even, percentil, correlación con el índice), regla por umbral, y DRL-Q frente al actor y
  los baselines.
* **Pricing arcos**: una fila por arco de las rutas de DRL-Q (mundos simulados de evaluación y replay): ruta, desde,
  hacia, distancia, `rate_hoy`, `p_estrella`, rango (`p_estrella_min/max`), `break_even`, `percentil_historico`,
  `posicion_valiosa`, índice de mercado del destino, margen del tour y una nota en los arcos sin p\* (retorno, sin
  carga, única acción factible). Con filtro y encabezado fijo.

Las cabezas Ṽ de `*_pricing.pt` se reutilizan solo si su huella coincide con el actor cargado (`pricing_vigente`):
tras reentrenar el PPO, `main.py` las reentrena solo.

## 7. Pruebas (`tests/`)

`test_post_value_head.py` (dos cabezas, init distinto, detach, RNG), `test_ppo_sin_cabezas.py` (el PPO no entrena
Ṽ ni toca las cabezas), `test_pricing.py` (§6a, rango, lanes sin carga, `only_option`, percentil con carga),
`test_escenario.py` (reproducción del prefijo, selector Q = argmax a mano, DRL-Q vectorizado = `greedy_rollout`,
regla por umbral, contrafactual y su álgebra, generación/ajuste de ejemplos), `test_drlq_arnes.py` (DRL-Q en el arnés y
en las comparaciones), `test_pricing_integration.py` (validación reducida, Excel, formato de `inference.py`).

## 8. Formato de salida (`inference.py`)

Por cada ruta, un bloque con una línea por arco y un CSV `results/<N>nodes/pricing_<run_tag>.csv` con una fila por arco:

```
Ruta: 0 → 7 → 5 → 9 → 0
Precio de reserva por arco:
  0→7: p* = 2.58 /mi  [rango 2.50–2.66]  (rate_hoy = 2.65, break-even = 2.28, percentil histórico 42 %)
  7→5: p* = 2.74 /mi  [rango 2.70–2.79]  (rate_hoy = 2.81, break-even = 2.28, percentil histórico 55 %)
  5→9: p* = 2.19 /mi  [rango 2.05–2.31]  (rate_hoy = 2.44, break-even = 2.28, percentil histórico 18 %)  ← posición valiosa
  9→0: p* = cost/dist (retorno al depot, Ṽ = 0)
```

* `p*` y `rate_hoy` en USD/milla con dos decimales; `break-even = cost/dist`; el **rango** es [mín, máx] de p\*_AB y
  p\*_BA (variabilidad del estimador, no un IC); `percentil histórico` = percentil de p\* en `rate_stack[:, i, j]` de
  los días de entrenamiento con carga, redondeado al entero.
* `← posición valiosa` cuando `p* < break-even`.
* Retorno a casa: `p* = cost/dist (retorno al depot, Ṽ = 0)`. Arco sin carga: `sin carga en esta lane
  (reposicionamiento): sin p*`. Si `only_option`: `p* = cost/dist (única acción factible)`.

Columnas del CSV: `episode_id, step, from, to, rate_hoy, p_estrella, p_min, p_max, break_even, percentil_historico,
only_option, is_return, has_load` (p\* en NaN donde no está definido).

## 9. Hiperparámetros (`config.py`)

`USE_POST_HEAD`, `POST_HIDDEN`, `POST_SEED_A/B`, `POST_N_EJEMPLOS`, `POST_N_VAL`, `POST_LR`, `POST_BATCH`,
`POST_EPOCHS`, `POST_PACIENCIA`, `MARKET_SEED_POST_TRAIN/VAL`, `N_EST_CONTRAFACTUAL`, `N_MUNDOS_CONTRAFACTUAL`
(mundos por bloque), `BE_MARGEN` (0,30), `INDICE_DISPERSION_SEMANAL`.

## 10. Uso

```bash
python main.py                                  # PPO (o checkpoint) → Ṽ A/B → evaluación con DRL-Q → validación
python post_training.py --num-nodes 10          # solo la fase de Ṽ sobre checkpoints/am_checkpoint_10nodes.pt
python validate_pricing.py --num-nodes 10       # solo la validación (usa el checkpoint *_pricing.pt)
python inference.py --start-node 0 [--show-board]
```

## 11. Limitaciones

* p\* es un costo de oportunidad; no se modela la aceptación del cliente.
* Ṽ se entrena con una sola realización por ejemplo (el objetivo es ruidoso: el piso de error es alto); las acciones
  se muestrean al azar, así que cubren también lanes que el actor casi nunca toma.
* Con K libre las rutas típicas son una lane cargada más el retorno: casi todos los p\* son del primer arco, donde
  el depot no es alternativa válida.
* Ṽ sigue al **actor** hasta casa, no a DRL-Q: DRL-Q es una mejora de política de un paso sobre el actor. Una segunda
  ronda de entrenamiento con estados de DRL-Q no está implementada.
