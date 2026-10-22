# Plan de implementación del precio de reserva

Documento para Claude Code, para ejecutarse en el repo del DRL. Añade al modelo
una segunda salida: el precio mínimo por milla al que al camión le conviene aceptar
una lane, dada su posición y el tiempo restante. **No modela aceptación** (no hay
datos): es una recomendación del lado del transportista.

Una fase por sesión; cada una termina con sus pruebas en verde y un commit.

---

## 1. Lo que ya está (no romper)

- **Entorno:** `RoutingEnv` y `VectorRoutingEnv` ya revelan `loads_row`/`rate_row` al
  llegar a un nodo (modo `market_process`). La recompensa pasa por `lane_reward`
  (`problem_data.py`).
- **Agente:** `AMRoutingAgent` = `AttentionEncoder` + `ContextNetwork` + `AttentionDecoder`.
  `ContextNetwork` ya devuelve un vector de contexto `h_t` **compartido** por el
  decoder, el crítico y, en el futuro, el pricing (lo dice su propio docstring).
- **Crítico:** `CriticHead(h_t) → V(s)` (MLP d_h → d_h → d_h/2 → 1).
- **Entrenamiento:** PPO con GAE, self-critical opcional, validación en mundos
  simulados, selección de checkpoint por media móvil.
- **Baselines:** B3 (política sobre la tabla V de la DP) y UB-DP ya existen. No se usan para el pricing
  del DRL, solo como referencia para el ruteo (ya cubiertos en el entrenamiento).

El pricing **no cambia** nada de eso. Reutiliza el mismo rollout de PPO, el mismo
contexto `h_t` y los mismos embeddings.

---

## 2. Qué es el precio de reserva

En el instante de decisión el agente está en `i`, con tiempo transcurrido `t`.
Ya ve `rate(i→j)` y `loads(i→j)` del tablero de hoy. Si acepta la carga en la lane
`i→j`, obtiene:

```
r_obs(i→j) = rate(i→j) · dist(i→j) − costo(i→j)
```

y llega a `j` en `t + t_ij`. Lo que encontrará en `j` (cargas y tarifas salientes) es
incierto; su valor esperado es `Ṽ(s, j)`: **valor post-decisión** de llegar a `j`
antes de ver su tablero.

El valor de aceptar la lane es `Q(s, j) = r_obs(i→j)/S + Ṽ(s, j)` (con
`S = REWARD_SCALE_FACTOR`). El de la mejor alternativa es
`Q_alt(s) = max_{a ≠ j válida} Q(s, a)` (incluye volver al depot y, cuando aplique,
otras lanes). Entonces el agente queda indiferente cuando `Q(s, j) = Q_alt(s)`,
despejando `rate(i→j)`:

```
p*(i→j, s) = ( S · (Q_alt(s) − Ṽ(s, j)) + costo(i→j) ) / dist(i→j)       [USD/milla]
```

Es el **costo de oportunidad** por milla de ir a `j`: el precio al que es igual de
bueno aceptar esa lane que su mejor alternativa. Puede ser menor que el break-even
`costo/dist` cuando `j` es una posición muy valiosa.

Dos casos límite:
- Si `j = depot` (cerrar el tour), `Ṽ(s, depot) = 0` por definición.
- Si `j` es la única acción factible (por ejemplo, el retorno al depot cuando la
  máscara descarta todas las demás por tiempo), `p*` no está definido: se devuelve
  el piso `costo/dist` y se marca `only_option = True`.

---

## 3. Decisiones de diseño

### D1. `Ṽ(s, j)` como cabeza nueva
Un MLP `PostDecisionValueHead(h_t, e_j) → Ṽ_j` por candidato `j`. Devuelve `(B, N)`.
Se entrena con el mismo rollout de PPO.

### D2. Objetivo de entrenamiento
Para la transición `(s_t, a_t, r_t)` con retorno GAE `G_t = ret_t`:

```
target_post[t] = G_t − r_t / S
```

(Nota: `rewards[t]` del buffer ya está dividido por `S`.) Pérdida: MSE **solo sobre
la acción tomada**, con coeficiente `POST_COEF` configurable.

```
loss_post = MSE( Ṽ(s_t, a_t),  G_t − r_t )
```

Para el **retorno al depot** `Ṽ(s_t, depot) = 0` siempre: ese paso **no** entra en la
loss (es el ancla de la fórmula).

### D3. `pricing.py` tiene una sola función
`reservation_price(...)` calcula `p*` para una lane `(i, j)`. Para imprimir el
tablero completo o para la validación, se llama en loop sobre las lanes salientes
válidas (son pocas: la mediana es ~2 a N=10, ~18 a N=35). No hace falta una versión
batch sobre todas las lanes.

### D4. Validación (tres cosas)
1. **Monte Carlo:** rollouts de la política desde la llegada a `j` → estimación
   empírica de `Ṽ(s, j)` → error contra la cabeza. Reportar bias y RMSE por bucket
   de tiempo restante.
2. **Posición:** percentil de `p*` dentro de la serie histórica de `rate` de la lane
   (`rate_stack` en los días de train) y su sensibilidad al día de la semana y al
   tiempo restante.
3. **Política de aceptación:** comparar "acepto si tarifa ≥ p*" contra "acepto la
   mejor tarifa" y "acepto siempre", bajo el supuesto explícito de que toda carga
   ofrecida está disponible a su tarifa histórica.

### D5. Qué NO se hace
- No modelar aceptación (no hay datos de aceptación y rechazo).
- No cambiar el objetivo de PPO ni el formato de `rate_stack`.
- No añadir features nuevas al encoder.
- No introducir una red separada; se comparte el encoder y `h_t`.

---

## 4. Mapa de archivos

| Archivo | Acción |
|---|---|
| `critic_head.py` | Añadir `PostDecisionValueHead` (clase nueva, al lado de `CriticHead`) |
| `am_agent.py` | Instanciar la cabeza nueva dentro de `AMRoutingAgent` (expuesta como `agent.post_value`) |
| `am_training.py` | `_forward` devuelve también `post_values` (B, N); nueva loss `loss_post` (comparte optimizer con el crítico); logging de MSE de `Ṽ` |
| `pricing.py` | **Nuevo:** `reservation_price(...)` por lane |
| `validate_pricing.py` | **Nuevo:** las validaciones de D4 → tablas y figuras |
| `config.py` | Añadir `USE_POST_HEAD`, `POST_HIDDEN`, `POST_COEF` |
| `main.py` | Flag `RUN_PRICING_VALIDATION` tras entrenar |
| `inference.py` | Al cargar un checkpoint, ofrecer `p*` por lane además de la acción greedy |

---

## 5. Fases

### Fase P0 — `PostDecisionValueHead`

```text
Añade a critic_head.py la clase PostDecisionValueHead(d_h, hidden=POST_HIDDEN):
  - MLP sobre la concatenación [h_t, e_j]: 2·d_h → hidden → hidden/2 → 1 (lineal).
  - Forward: dado h_t (B, d_h) y embeddings (B, N, d_h), devuelve (B, N) sin
    activación final (permite negativos).
  - IMPORTANTE: dentro del forward, usa `h_t.detach()` y `embeddings.detach()` antes
    de pasarlos al MLP. Hacen falta los dos: `h_t.detach()` corta el gradiente hacia
    context_net y decoder; `embeddings.detach()` lo corta hacia el encoder. Sin
    cualquiera de los dos, loss_post seguiría contaminando los parámetros de la
    política. Documenta esto en el docstring: "la cabeza aprende una regresión sobre
    el estado congelado; no perturba al PPO".
  - Inicialización: xavier_uniform. Sin dropout.
En am_agent.py, instancia la cabeza como `self.post_value = PostDecisionValueHead(d_h)`
dentro de AMRoutingAgent (nota: AMRoutingAgent es un contenedor de módulos sin
método forward propio; el forward real está en `_forward` de am_training.py). Tests:
  1. Shapes: (B=4, N=10) y (B=4, N=100), mismos pesos.
  2. Rango de salida: valores finitos, sin NaN.
  3. Equivarianza: si se permutan los embeddings de entrada por una permutación π,
     la salida sale permutada por la misma π (post_value[:, j] = post_value_perm[:, π(j)]).
  4. El gradiente de `agent.post_value(h_t, emb).sum()` respecto a los parámetros
     del encoder es cero (prueba del detach).
Commit al cerrar.
```

**Aceptación:** tests en verde; `AMRoutingAgent` tiene `post_value` cuando
`USE_POST_HEAD=True`.

### Fase P1 — Loss de `Ṽ` en el loop PPO

```text
En am_training.py:
1. Extiende _forward para devolver también post_values (B, N) cuando USE_POST_HEAD.
   En recolección NO se guarda nada nuevo (post_values se recomputa en la fase de
   actualización; es barato y evita un campo más en el buffer).
2. compute_gae_per_episode ya calcula `returns`. Define
   target_post[t] = returns[t] - rewards[t]  (ambos ya escalados por 1/S).
3. En el loop de epochs PPO: con el nuevo post_values (B, N) del forward, haz gather
   sobre las acciones (B,) → new_post_val (B,). MSE contra target_post → loss_post.
4. Para transiciones donde action == start_node (retorno al depot): se EXCLUYEN del
   cómputo de loss_post (el ancla Ṽ(s, depot) = 0 no se aprende, se impone).
5. Los parámetros de agent.post_value se registran en el critic_optimizer
   existente (al construirlo, pasa list(critic.parameters()) + list(
   agent.post_value.parameters())). El backward conjunto sigue siendo:
   actor_optimizer.zero_grad(); critic_optimizer.zero_grad()
   (actor_loss_total + value_loss + POST_COEF * loss_post).backward()
   actor_optimizer.step(); critic_optimizer.step()
   El detach() de la Fase P0 asegura que loss_post NO toque los parámetros del
   actor/encoder/context_net. Solo el crítico y el MLP de post_value aprenden de
   ella. No se agrega un optimizer nuevo.
6. Logging: por update agregar `post_mse` y `post_ev` (explained variance de
   target_post) a log_entry; imprimirlas en es_log junto con `ev` y `kl`.
Tests:
  - target_post del retorno al depot vale 0.
  - Un smoke test de 2 updates con N=10 corre sin NaN; post_mse decrece.
  - El reward medio de un update sin USE_POST_HEAD y con USE_POST_HEAD (con la
    misma semilla) son IDÉNTICOS en los primeros 3 updates (prueba de que la
    política no cambió por el detach).
Commit al cerrar.
```

**Aceptación:** 2 updates smoke en N=10 corren limpios; `post_mse` cae; `post_ev`
sube con el tiempo. El reward es idéntico a la corrida sin pricing (gracias al
detach).

### Fase P2 — `pricing.py`

```text
Crea pricing.py con UNA sola función:

reservation_price(agent, h_t, embeddings, action_mask, current_node, j,
                  rate_row_obs, distance_row, cost_row, start_node, S)
  → (p_estrella: float, only_option: bool)

Procedimiento (sin gradiente, @torch.no_grad()):
  1. Ṽ = agent.post_value(h_t, embeddings).squeeze(0)   # (N,) para B=1
     Ṽ[start_node] = 0.0 (ancla del retorno al depot).
  2. r_obs[a] = rate_row_obs[a] * distance_row[a] - cost_row[a]  para toda a válida
     con loads_row[a] > 0; para el retorno al depot, r_obs[start_node] se calcula
     igual (sin exigir carga).
     r_obs_scaled = r_obs / S.
  3. Q[a] = r_obs_scaled[a] + Ṽ[a].
  4. Alternativas válidas = {a : action_mask[a] == 1 y a != j}.
     Si no hay ninguna → only_option = True, p_estrella = cost_row[j] / distance_row[j].
     Si hay → Q_alt = max_a Q[a], p_estrella = (S * (Q_alt - Ṽ[j]) + cost_row[j]) / distance_row[j].
  5. Devuelve (float(p_estrella), only_option).

Para imprimir el tablero completo o para la validación se llama en loop sobre las j
válidas (son pocas: mediana ~2 a N=10, ~18 a N=35; no hace falta versión batch).

Tests:
  - Para j = start_node con Ṽ[start_node] = 0, Q_alt = 0 → p* = cost/dist (break-even).
  - Si Ṽ(s, j) = Q_alt(s), entonces p* = cost_row[j] / dist[j] (break-even).
  - Si Ṽ(s, j) > Q_alt(s), p* < break-even (j es posición valiosa).
Commit al cerrar.
```

**Aceptación:** tests en verde; `inference.py` puede imprimir una tabla con `p*`,
`rate_hoy`, break-even y percentil por lane al llegar a un nodo.

### Fase P3 — `validate_pricing.py`

```text
Crea validate_pricing.py que, cargando un checkpoint, hace tres validaciones:
(a) Calibración de Ṽ por Monte Carlo:
    - 300 (s, j) muestreados del rollout de la política en val.
    - Para cada uno, 50 rollouts desde la llegada a j → mean estimado.
    - Reporta scatter de Ṽ (predicho) vs mean MC; bias y RMSE globales y por
      bucket de tiempo restante.
(b) Posición de p* dentro del histórico:
    - Para 500 (s, j) muestreados, reporta percentil de p* dentro de la serie
      histórica `rate_stack[:, i, j]` de la lane; sensibilidad al día de la semana
      y al tiempo restante (tabla pivot).
(c) Política de aceptación:
    - Simula 500 mundos de val. Compara "acepto si rate_obs >= p*" vs "acepto la
      mejor tarifa disponible" vs "acepto siempre que haya carga". Reporta margen
      medio, % de ofertas aceptadas, longitud del tour y validez.
Salidas: results/<N>nodes/pricing_val/figures/*.png y results/<N>nodes/pricing_val/tables/*.csv.
Tiempo estimado: 10-20 minutos con N=35.
```

### Fase P4 — Integración en `main.py` e `inference.py`

```text
En main.py:
  - Si USE_POST_HEAD está activo, al terminar el entreno corre validate_pricing con
    el checkpoint seleccionado.
En inference.py:
  - Al cargar un checkpoint con post_value, por cada arco del tour evaluado
    imprime el bloque descrito en la §8 y escribe el CSV descrito ahí.
  - Al llegar a un nodo, además de la decisión greedy, puede imprimir una tabla
    opcional con el p* de cada lane saliente válida (bandera --show-board).
```

### Fase P5 — Documentación y cierre

```text
- Añade a README.md una sección "Pricing: precio de reserva" con el modelo y la
  fórmula del §2, limitaciones (D5), y cómo ejecutarlo.
- Añade al Excel de resultados una pestaña "Pricing" con las métricas de P3.
```

---

## 6. Hiperparámetros nuevos (`config.py`)

```
USE_POST_HEAD = True       # activa la cabeza de valor post-decisión
POST_HIDDEN   = AM_D_H     # tamaño oculto del MLP (igual al encoder)
POST_COEF     = 1.0        # peso de loss_post. Con detach() no afecta a la política,
                            # así que 1.0 es seguro. Bajar solo si post_mse explota.
```

El MLP de post_value usa el optimizer del crítico (no se agrega un lr nuevo).

---

## 7. Riesgos y mitigaciones

- **`Ṽ` se entrena solo en acciones muestreadas.** Para lanes que el agente casi
  nunca toma, la estimación es ruidosa y `p*` también. La validación (a) de P3 lo
  reporta; si el error es alto en algún subconjunto, se puede añadir después una
  calibración off-policy sin cambiar el diseño actual.
- **Horizonte corto.** Con 77 h de tope y lanes de mediana 38.7 h, los tours tienen
  pocos arcos (típicamente 2-4), así que `Ṽ(s, j)` tiene varianza alta: pocos arcos
  restantes implican alta sensibilidad al tablero revelado. La validación por MC
  debe reportar el error en función del tiempo restante.
- **Interpretación.** `p*` es costo de oportunidad, no probabilidad de aceptación.
  La tesis debe decirlo explícitamente (D5).
- **Checkpoints anteriores.** No contienen `post_value`. `load_state_dict(...,
  strict=False)` permite cargarlos y la cabeza nueva arranca desde cero. Para la
  tesis se recomienda reentrenar desde cero en N=10 y hacer transfer a tamaños
  mayores (así todos los checkpoints son comparables y la cabeza tiene tiempo de
  converger).

---

## 8. Formato de salida esperado por la autora (uso en la tesis)

El entregable del modelo son **dos decisiones por episodio**: la secuencia de ruta
que el camión debe seguir y el precio mínimo de reserva por milla de cada arco
recorrido. `p*` es el precio sugerido **desde el lado del transportista**: por
debajo de él, aceptar la lane deja de convenir frente a la mejor alternativa. No es
una probabilidad de aceptación por parte del cliente (no se modela porque no hay
datos de aceptación y rechazo).

La Fase P4 debe hacer que `inference.py` imprima, para cada tour evaluado, un
bloque con este formato exacto:

```
Ruta: 0 → 7 → 5 → 9 → 0
Precio de reserva por arco:
  0→7: p* = 2.58 /mi  (rate_hoy = 2.65, break-even = 2.28, percentil histórico 42 %)
  7→5: p* = 2.74 /mi  (rate_hoy = 2.81, break-even = 2.28, percentil histórico 55 %)
  5→9: p* = 2.19 /mi  (rate_hoy = 2.44, break-even = 2.28, percentil histórico 18 %)  ← posición valiosa
  9→0: p* = cost/dist (retorno al depot, Ṽ = 0)
```

Reglas de formato:
- `p*` y `rate_hoy` en USD/milla con dos decimales.
- `break-even = cost/dist` para esa lane.
- `percentil histórico` = percentil de `p*` dentro de la serie `rate_stack[:, i, j]`
  en los días de train, redondeado al entero (p. ej. 42 %).
- Marcar `← posición valiosa` cuando `p* < break-even` (el destino `j` compensa
  parte del costo porque es una buena posición futura).
- Para el arco de retorno (`j = start_node`): `p* = cost/dist (retorno al depot,
  Ṽ = 0)`. Nada de percentil (el retorno no se oferta como carga).
- Si en algún arco intermedio `only_option = True`, imprimir
  `p* = cost/dist (única acción factible)` en lugar del percentil.

Además, en la Fase P4 escribir un CSV por corrida en `results/<N>nodes/pricing_<run_tag>.csv`
con las columnas:

```
episode_id, step, from, to, rate_hoy, p_estrella, break_even,
percentil_historico, only_option, is_return
```

una fila por arco del tour. Esto facilita construir tablas de resultados para la
tesis sin tener que parsear el stdout.

---

## 9. Reglas de trabajo con Claude Code

1. Una sesión por fase, en modo plan. Aprobar el plan antes de que edite.
2. Al iniciar cada sesión: que lea este documento.
3. Commit al cerrar cada fase, con los tests en verde.
4. No tocar hiperparámetros del PPO ni del encoder.
5. Mantener los checkpoints anteriores intactos; `post_value` se añade en nuevos.
6. Si un test o una métrica contradicen el diseño (por ejemplo, `post_mse` no cae o
   el reward del PPO empeora), detenerse y reportar antes de "ajustar" el diseño.
