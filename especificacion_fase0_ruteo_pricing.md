# Fase 0: especificación del ruteo y pricing conjunto

Estado: **aprobada** (2026-11-23). Todas las decisiones de §12 están cerradas.

**La cotización es el único modo** (decidido el 2026-11-23): el entorno, el entrenamiento, la validación,
la evaluación y la inferencia ya no tienen el modo anterior (sin precio) ni una bandera para elegirlo.

## 1. Objetivo

Hoy el precio de cada lane es un dato del entorno (la tarifa sorteada, o `mean` en replay), la
carga se gana siempre y p\* se calcula después, sobre una ruta ya decidida. En el modelo nuevo:

- En cada nodo, la política decide **a dónde ir (j)** y **a qué nivel de precio cotizar**.
- La carga se gana o se pierde según la competencia en el nodo de origen.
- El reward del tramo es el **precio cotizado** × distancia − costo, y el reward de la ruta es la
  suma de los tramos. La misma regla vale para el DRL, los baselines y las cotas.

Por qué no se cobra el precio de reserva exacto: si cada tramo cobra justo el costo de oportunidad,
`r(i→j) = V(i) − V(j)`, la suma a lo largo de la ruta se cancela término a término y todas las
rutas dan `V(casa, inicio)`. El precio tiene que ser una decisión con riesgo de perder la carga.

## 2. Decisiones ya tomadas

| Tema | Decisión |
|---|---|
| Quién decide el precio | El DRL, con una cabeza de precio en el decoder (acción híbrida destino + nivel). |
| Espacio de precio | 10 niveles discretos en escala de percentiles, comunes a todas las lanes. |
| Aceptación | `P(ganar \| u_k) = (1 − u_k)^m(LTR_i)`, con números aleatorios comunes. |
| Rechazo | La lane se enmascara y el camión vuelve a decidir en el mismo nodo, con penalización de tiempo. |
| Ṽ(s,j) / post_training | Se eliminan. Solo queda la CriticHead V(s). |
| Comparaciones | Solo la **comparación B**: destino del DRL + reglas de precio fijas vs. el DRL completo. Además, la tabla de baselines (B1–B3, UB-DP, UB-PI) con una regla fija. |
| Tamaños | Cadena completa 10 → 20 → 50 → 75 → 100 (35 se quitó el 2026-11-24), todo reentrenado desde cero. UB-DP y B3 se calculan a 10 y 20 nodos (`UBDP_MAX_NODOS` = 35); a 50, 75 y 100 son "no tratable". |

## 3. Modelo de mercado

### 3.1 Niveles de precio

```
U_NIVELES = [0.00, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]     (config.py)
p_k(i→j, día) = _interp_cuantiles(u_k, cuantiles_{i→j, día_p}) · e^δ          [USD/milla]
```

- `cuantiles` son los 6 puntos (min, q1, q2, q3, p90, max) de la lane ese día. `día_p` es el mismo
  día con cuantiles que usa hoy `_dia_precio`: el propio día o el más cercano con cargas.
- `δ` es el corrimiento nacional del episodio (`MarketProcess._delta`); en replay, δ = 0.
- El nivel 0 es el mínimo del mercado y siempre gana la carga. El nivel 9 es el p90.
- Puede quedar por debajo del break-even. Así la política puede aceptar cargas con pérdida que la
  dejan bien ubicada.
- Solo se calculan las N lanes salientes del nodo actual, al llegar. No se guarda ninguna tabla por lane.

### 3.2 Competidor y aceptación

Al llegar al nodo `i`, por cada lane `i→j` se sortea el percentil del mejor competidor:

```
U_ij      ~ Uniforme(0,1)                      el mismo sorteo `u` que ya usa _sortear (misma semilla)
u_comp_ij = 1 − (1 − U_ij)^(1/m_i)             mínimo de m_i competidores uniformes ⇒ u_comp ~ Beta(1, m_i)
ganar     ⇔ u_k < u_comp_ij                    comparación en percentiles (evita empates)
P(ganar | u_k) = (1 − u_k)^m_i
```

- **Reutilizar `U_ij`** no cambia el flujo del generador. Con `m = 1`, el precio del competidor
  `F⁻¹(u_comp)·e^δ` es exactamente la tarifa que se sortea hoy, así que el mundo actual es el caso
  particular "un solo competidor".
- Es determinista dada (semilla del mundo, nodo, día). Todos los métodos que cotizan el mismo nivel
  en el mismo mundo obtienen el mismo resultado.
- El agente **no observa** `u_comp`. Solo las cotas con información perfecta (UB-PI) lo ven.

### 3.3 Número efectivo de competidores m(LTR)

```
m_i = clip( 1 + κ · LTR_REF / LTR_i ,  1 ,  M_MAX )
```

| Parámetro | Valor propuesto | Justificación |
|---|---|---|
| `LTR_REF` | mediana de LTR por hub y día en train (≈ 25.5) | Normaliza: en el LTR mediano, m = 1 + κ. |
| `κ` | 1.0, con sensibilidad {0.5, 1, 2} | Sin datos de aceptación es un supuesto. La sensibilidad muestra que las conclusiones no dependen de él. |
| `M_MAX` | 20 | Con LTR ≈ 0.8 (p5) quedaría m ≈ 33. El tope evita que la aceptación sea prácticamente nula. |

Lectura: con LTR alto (más cargas que camiones), m → 1 y cotizar en la mediana gana la mitad de
las veces. Con LTR bajo, m es alto y hay que cotizar cerca del mínimo. El LTR es el del **nodo de
origen** en el día real de llegada: es un dato observado, igual que el índice de mercado.

### 3.4 Replay (octubre, fuera de muestra)

En replay se usan los cuantiles reales del día y δ = 0. El competidor se sortea con la semilla del
caso. La columna `mean` deja de ser el ingreso.

## 4. El MDP

### 4.1 Acción

Acción = (j, o), donde `o ∈ {nivel 0, …, nivel 9, VACÍO}` (11 opciones).

| Opción | Cuándo es válida |
|---|---|
| Nivel k | La lane i→j tiene carga (`loads > 0`) y no fue rechazada en esta visita al nodo. |
| VACÍO (moverse sin carga, sin ingreso) | j = casa (el retorno siempre es válido), o i = casa (primer movimiento, reposicionamiento), **y solo si la lane i→j no se puede cotizar** (sin carga, carga rechazada o tope de cotizaciones). Si hay carga cotizable se cobra siempre: VACÍO estaría dominado, porque el nivel 0 gana siempre y cobra p₀·d − c > −c. |

j es válido si tiene al menos una opción válida y pasa las reglas de hoy: sin self-loop, regreso a
casa factible dentro de `MAX_DURATION` y presupuesto K.

### 4.2 Transición

- **Se gana** (nivel k con `u_k < u_comp`) o **VACÍO**: el camión va a j como hoy. Consume un arco
  de K, avanza el tiempo `T[i,j]` y al llegar se revela el tablero de j.
- **Se pierde**: el camión se queda en i, el tiempo avanza `COTIZACION_HORAS`, la lane i→j queda sin
  la opción de cotizar en esta visita, **no se consume K** y no se revela un tablero nuevo, porque es
  el mismo tablero.
- **Tope de cotizaciones** `COTIZACIONES_MAX_POR_NODO`: después de ese número de rechazos en la
  misma visita, solo quedan opciones VACÍO (en la práctica, volver a casa, salvo en el primer
  movimiento). Así los episodios quedan acotados y el entorno coincide con la DP de §7.3.

Decidido: `COTIZACION_HORAS = 0.25` (una franja de la DP, 15 min) y `COTIZACIONES_MAX_POR_NODO = 2`.

### 4.3 Reward

```
r = round( p_k · dist(i→j) − costo(i→j) )     si se gana con el nivel k
r = round( − costo(i→j) )                      si VACÍO
r = 0                                          si se pierde (solo pasa el tiempo)
```

`costo` es la fórmula actual (`lane_cost_row`). `lane_reward` sigue siendo la única fórmula, pero
recibe el **precio cotizado** en lugar de la tarifa. El escalado, `RETURN_SUCCESS_BONUS` y
`TIME_VIOLATION_PENALTY` no cambian.

### 4.4 Observación (features)

| Feature por nodo j | Hoy | Nuevo |
|---|---|---|
| [0] reward_norm (ranking en la fila) | margen con la tarifa sorteada | margen **esperado a la mediana**: `q2·e^δ·d − c` |
| [5] reward_abs | ídem | ídem, escala absoluta |
| nueva | — | dispersión de la lane: `(p90 − q1)/q2` |
| nueva | — | break-even normalizado: `c / (d · q2·e^δ)` |
| nueva | — | opción de cotizar disponible en j (1 si tiene carga y no fue rechazada) |
| [7–10] continuación | EV con tarifa media | EV con la regla miope óptima (§6) sobre cuantiles medios |

| Feature de contexto | Nuevo |
|---|---|
| `log(m_i)` | competencia en el nodo actual |
| rechazos en esta visita / `COTIZACIONES_MAX_POR_NODO` | — |

La tarifa sorteada deja de ser observable, porque ahora es el competidor. N_NODE_FEATURES pasa de 11 a 14.

**Implementado (Fase 4).** Las 3 features nuevas van en las **últimas** columnas (11–13), después del
índice de mercado (columna 10, `INDEX_COL`), y el bloque de contexto de cotización va al **final** de la
entrada de ContextNetwork. Así los checkpoints anteriores cargan con esas columnas en cero. La dispersión
se calcula con los niveles, `(p_0.8 − p_0.2)/p_0.5`, porque q1 y p90 no son niveles. Las features de
continuación [6–9] usan, desde la Fase 5, la misma tabla que B2 y B2P: el valor de cotizar con la regla
miope sobre los cuantiles medios del día de la semana (ver §7).

## 5. Arquitectura DRL y PPO

```
Encoder Transformer → embeddings e_j, contexto h_t
  ├─ Pointer decoder       → π(j | s)                (sin cambios, máscara de §4.1)
  ├─ Cabeza de precio      → π(o | s, j)             MLP([h_t ; e_j ; glimpse_t]) → 11 logits con máscara
  └─ CriticHead            → V(s)                    (sin cambios)
```

- **Orden autorregresivo**: primero el destino, después la opción condicionada en j. Así π(j|s) no
  depende de la regla de precio, y la comparación B puede fijar el destino del DRL y cambiar solo el precio.
- **Log-prob**: `log π(a|s) = log π(j|s) + log π(o|s,j)`. Cuando solo VACÍO es válido, el segundo
  término es 0 automáticamente.
- **Entropía separada**: `PPO_ENTROPY_COEF_PRECIO_{START,END}`, independiente de la entropía del destino.
- **Paso PPO = una cotización**: un rechazo es un paso con r = 0 y el tiempo avanzado. GAE no cambia
  (γ = 1).
- **Greedy y self-critical**: argmax de j y después argmax de o. El rollout greedy usa los mismos
  competidores (misma semilla) que el rollout muestreado.
- **Buffer**: `max_steps` pasa a `max_pasos · (COTIZACIONES_MAX_POR_NODO + 1)`, porque en cada visita
  caben hasta 2 rechazos más el viaje (`routing_env.max_pasos_cotizacion`).

### Se elimina (hecho en la Fase 3)

`PostDecisionValueHead`, `post_value_a/b`, `has_post_heads`, `post_values`, `use_post_head`
(model/critic_head.py, model/agent.py); post_training.py; validate_pricing.py (se reescribe desde cero en
la Fase 6); `reservation_price` / `q_values` / `r_obs` / `PrecioReserva` (pricing.py);
`checkpoint_tiene_post_head` y `post_entrenadas` (transfer_learning.py); las etapas de Ṽ y de validación
de p\* en main.py; `USE_POST_HEAD`, `POST_*`, `MARKET_SEED_POST_*`, `N_EST_CONTRAFACTUAL`,
`N_MUNDOS_CONTRAFACTUAL` e `INDICE_DISPERSION_SEMANAL` (config.py); y los tests test_post_value_head.py,
test_escenario.py y test_pricing_integration.py. inference.py muestra la ruta sin p\* hasta la Fase 6.

Los checkpoints viejos que traen Ṽ siguen cargando: `_reconcile_state_dict` omite las claves que el
modelo ya no tiene (lo prueba test_ppo_sin_cabezas.py). Los archivos `checkpoints/*_pricing.pt` se
borraron (2026-11-23; el borrado entra en el commit final). La cabeza Ṽ tenía su propio generador aleatorio, así que
quitarla no cambia la inicialización del resto del agente. El entorno escalar `RoutingEnv` se eliminó junto
con el modo sin precio.

## 6. Reglas de precio fijas ([pricing.py](pricing.py))

Todas devuelven un nivel `k` dada la lane, su distancia y costo, y `m_i`:

| Regla | Definición |
|---|---|
| `R_MEDIANA` | k = 5 (u = 0.5). |
| `R_BE_MARGEN` | El nivel más bajo con `p_k ≥ break_even + BE_MARGEN` (0.30 USD/mi); si no hay ninguno, k = 9. |
| `R_MIOPE` | `argmax_k (1 − u_k)^m · (p_k·d − c)`: optimiza contra el mercado, pero solo el tramo actual. |

Decidido: los baselines B1–B3 de la tabla principal cotizan con `R_MIOPE`, porque así son más
fuertes y la comparación es más creíble. `R_MEDIANA` queda como sensibilidad.

## 7. Baselines y cotas

Todos se ejecutan en el entorno nuevo: rechazos, re-decisión y tope de cotizaciones.

| Método | Destino | Precio |
|---|---|---|
| B1 | arco con mayor valor inmediato esperado `max_k P_k(p_k d − c)` | regla fija |
| B2 / B2P | recursión actual, pero el valor esperado de un arco futuro es `P(carga)·max_k P_k(p_k d − c)` sobre cuantiles medios (B2P: más el pronóstico del índice) | regla fija |
| B3 | un paso sobre la tabla V de UB-DP: `argmax_j max_k [P_k G_jk + (1 − P_k) W_j]` (§7.3) | regla fija (`R_MIOPE`, igual que los demás) |
| UB-DP | DP de §7.3 | óptimo dentro de la DP |
| UB-PI | información perfecta (§7.4) | el nivel más alto que gana |

### 7.1 Comparación B

El destino lo da la cabeza de destino del DRL, y si hay un rechazo vuelve a decidir con la lane
enmascarada. El precio se calcula con cada regla:

| Fila | Destino | Precio |
|---|---|---|
| DRL | DRL | cabeza de precio |
| DRL-dest + R_MEDIANA | DRL | mediana |
| DRL-dest + R_BE_MARGEN | DRL | break-even + margen |
| DRL-dest + R_MIOPE | DRL | miope óptima |

Lectura: si el DRL le gana a `R_MIOPE`, el precio aprovecha el momento del ciclo y la expectativa
futura, no solo el mercado del tramo actual.

### 7.2 Firma de la tabla

`_firma` en [baselines/dp_relajada.py](baselines/dp_relajada.py) incluye `U_NIVELES`, κ, `M_MAX`,
`LTR_REF`, `COTIZACION_HORAS` y `COTIZACIONES_MAX_POR_NODO`. Las tablas `ubdp_tabla_*.npz` actuales
quedan inválidas automáticamente.

### 7.3 UB-DP con decisión de precio (dos intentos, mismo tablero)

Para un tablero sorteado `b` en (casa h, nodo i, franja s, k):

```
G_jk(s)  = p_jk·d_ij − c_ij + V_{k−1}(j, s + f_ij)          V(casa) = 0
P_k      = (1 − u_k)^m_i
E(s)     = mejor opción VACÍO (retorno a casa; si i = casa, cualquier j)
A_j      = max_k [ P_k G_jk(s+τ) + (1 − P_k) E(s+τ) ]        segundo intento, con VACÍO como respaldo
W_j      = max( E(s+τ), max_{j' ≠ j} A_j' )                  top-2 sobre j (torch.topk)
V_k(i,s) = media_b  max( E(s), max_j max_k [ P_k G_jk(s) + (1 − P_k) W_j ] )
```

- Sigue siendo una inducción hacia atrás. `s+τ` y `s+f` son franjas posteriores, ya calculadas,
  porque τ = 1 franja.
- **No hay sesgo por volver a sortear**: después de un rechazo se reusa el **mismo** tablero (W_j),
  no un tablero nuevo. Si se usara uno nuevo, cotizar alto a propósito sería una forma gratis de
  "esperar a ver otro tablero".
- **Memoria**: el máximo sobre k se hace en un bucle de 10 iteraciones con un máximo acumulado,
  dentro de los mismos bloques de `_h_chunk`. El pico de memoria es el actual más un tensor
  `(h_chunk, I, S)`. El cómputo crece unas 20 veces (10 niveles × 2 intentos).
- **Tratabilidad a 75 y 100 nodos (decidido)**: la tabla se calcula en un proceso aparte, antes de
  entrenar, con `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`. No se baja `UBDP_S`. Si el
  cálculo supera `UBDP_TIEMPO_MAX_HORAS` o se queda sin memoria, se detiene y se guarda un marcador
  `ubdp_tabla_<N>nodos.no_tratable.json` con N, el tiempo transcurrido, la franja alcanzada y una
  estimación del tiempo total. En ese tamaño:
  - UB-DP y B3 se reportan como **"no tratable"** (no como un error) en el Excel y en el resumen.
    Esto es un resultado válido: la DP no escala y el DRL sí.
  - El entrenamiento **no puede depender de la tabla**. Hoy `ValidacionSimulada`
    ([am_training.py:400](am_training.py#L400)) la calcula al iniciar el PPO, y ahí falló la corrida
    a 75 y 100 nodos. Esa validación tiene que funcionar sin la tabla, con el gap contra UB-DP solo
    cuando la tabla existe.
  - UB-PI, B1, B2 y B2P se siguen reportando.
  - Decidido el 2026-11-24: por encima de `UBDP_MAX_NODOS` = 35 nodos (50, 75 y 100) la tabla **no se intenta**
    (`cargar_o_calcular_tabla` devuelve None de inmediato): con los dos intentos de cotización el costo crece
    con N³ · S y esperar el tope de 24 h por tamaño no es viable. B3 y UB-DP se reportan "no tratable".

### 7.4 UB-PI

Con el mundo completo se conoce `u_comp`, así que en cada arco se cotiza el nivel más alto con
`u_k < u_comp` (el nivel 0 siempre gana). El reward del arco queda determinado y no hay rechazos.
`solve_ub_pi` mantiene su forma actual: solo cambia la matriz R. Sigue siendo cota de todos los métodos.

## 8. Evaluación y reportes

Métricas nuevas en [evaluation.py](evaluation.py), [stats_analysis.py](stats_analysis.py) y
[excel_report.py](excel_report.py), por método:

- margen total (Σ reward en USD; definición nueva);
- tasa de aceptación = cargas ganadas / cotizaciones;
- cotizaciones por arco recorrido y horas perdidas por rechazos;
- nivel cotizado (histograma) y USD/mi cobrado frente a la mediana de la lane;
- dinero dejado en la mesa, solo en mundos simulados: `Σ (p_comp − p_cobrado)·d` en las cargas ganadas.

[validate_pricing.py](validate_pricing.py) se reescribe. Pasa a reportar el nivel cotizado por el DRL
según el LTR, el momento del ciclo (tiempo y k restantes) y el índice de mercado; la comparación B; y
la sensibilidad a κ. Desaparecen la prueba contrafactual y el rango A/B.

## 9. Configuración ([config.py](config.py))

**Nuevas**: `U_NIVELES`, `COMPETENCIA_KAPPA`, `COMPETENCIA_M_MAX`, `COTIZACION_HORAS`, `COTIZACIONES_MAX_POR_NODO`,
`PPO_ENTROPY_COEF_PRECIO_START/END`, `REGLA_PRECIO_BASELINES`, `UBDP_TIEMPO_MAX_HORAS`.
`LTR_REF` no es una constante: `MarketProcess` la calcula al iniciar (mediana del LTR por hub y día
en train, sobre los 194 hubs) y entra en `firma_params`. No hace falta reajustar `market_params.npz`.

**Se eliminan**: `USE_POST_HEAD`, `POST_*`, `MARKET_SEED_POST_TRAIN/VAL`, `N_EST_CONTRAFACTUAL`,
`N_MUNDOS_CONTRAFACTUAL` y, si ya no se usa, `INDICE_DISPERSION_SEMANAL`.

**Se conservan**: `BE_MARGEN` (regla `R_BE_MARGEN`) y `PRICING_VAL_SEED`.

## 10. Archivos

| Archivo | Cambio |
|---|---|
| [pricing.py](pricing.py) | Se reescribe: niveles, m(LTR), aceptación, reglas fijas; conserva `lane_cost_row` y `percentil_historico`. |
| [market_process.py](market_process.py) | Expone cuantiles por lane, LTR y `u_comp`; `sample_world` también devuelve cuantiles y `u_comp`. |
| [problem_data.py](problem_data.py) | `lane_reward` recibe el precio cotizado. |
| [routing_env.py](routing_env.py) | **Hecho.** `VectorRoutingEnv` solo con cotización: acción (j, o), rechazo y re-decisión, tope de cotizaciones, máscara de opciones. El tablero sale de `market_process` (mundos simulados) o de `mercado_replay` (días reales, `tablero_real`). Se eliminaron el modo sin precio, `rm_pen_stack` en el entorno y el entorno escalar `RoutingEnv`. |
| [model/encoder.py](model/encoder.py), [model/decoder.py](model/decoder.py), [model/agent.py](model/agent.py), [model/critic_head.py](model/critic_head.py) | **Hecho (Fases 3 y 4):** features nuevas, `PriceHead` (con su propio generador aleatorio, no consume el RNG global) y `act/greedy/evaluate_cotizacion`; Ṽ eliminada. |
| [am_training.py](am_training.py), [greedy_rollout.py](greedy_rollout.py) | **Hecho:** acción híbrida (destino, nivel), entropía separada (`PPO_ENTROPY_COEF_PRECIO_*`), buffer con opciones, diagnósticos de aceptación y nivel, conformidad de K sobre arcos recorridos. Entrena en mundos simulados o, sin `market_process`, en replay de los días de entrenamiento (HPO). `ValidacionSimulada` no usa la tabla de UB-DP ni B3. `construir_features` es compartida con la evaluación y la inferencia. |
| [baselines/politicas.py](baselines/politicas.py) | **Hecho (Fase 5):** `Observacion` con precios por nivel, m, opciones y rechazos de la visita; `actuar` da el destino y `cotizar` aplica `REGLA_PRECIO_BASELINES`. B1 usa el valor inmediato esperado `max_k P_k(p_k d − c)`. B2/B2P (y las features de continuación) valoran un arco futuro como `prob · P̄_k*(p̄_k* d − c) − (1 − prob) c` (`_valor_cotizado_por_dow`): con P̄ = 1 y un solo precio es el valor de antes, `prob · q̄ d − c`; la parte `−(1 − prob) c` (recorrer la lane vacía) la conserva de la fórmula anterior. B3 elige el destino con los dos intentos sobre la tabla (§7.3) y cotiza con la regla miope. |
| [baselines/dp_relajada.py](baselines/dp_relajada.py), [baselines/ub_pi.py](baselines/ub_pi.py) | **Hecho (Fase 5).** UB-PI (§7.4): `evaluation.mundo_ub_pi` cotiza el nivel más alto que gana; `solve_ub_pi` sin cambios. UB-DP (§7.3): dos intentos sobre el mismo tablero, τ = `COTIZACION_HORAS` en franjas (redondeo hacia abajo). Cada sorteo guarda solo el día de train y las cargas; los rewards por nivel y `P_k` salen de una tabla por día real (7, días, N, N, K), así que la memoria no crece con K × S. Tope `UBDP_TIEMPO_MAX_HORAS` y falta de memoria ⇒ marcador `.no_tratable.json`, `cargar_o_calcular_tabla` devuelve None y la evaluación reporta B3 / UB-DP "no tratable". Cálculo aparte: `python -m baselines.dp_relajada --num-nodes N`. |
| [evaluation.py](evaluation.py), [stats_analysis.py](stats_analysis.py), [excel_report.py](excel_report.py), validate_pricing.py (nuevo), [inference.py](inference.py) | **Hecho:** arnés único con cotización (`actuar` + `cotizar`), comparación B (`DRL`, `DRL-dest+mediana`, `DRL-dest+be_margen`, `DRL-dest+miope`), métricas de §8 en el resumen y el Excel, replay con competidor por caso, inferencia con la cotización por paso. **Fase 6 (hecho):** `stats_analysis.comparar_con_baselines` tiene dos familias de Holm, "ruteo" (DRL − B1…B3) y "precio" (comparación B: DRL − DRL-dest+regla). validate_pricing.py (nuevo, mundos con `PRICING_VAL_SEED`; main.py la corre al final de cada tamaño): nivel cotizado por el DRL por quintil de LTR, m, tiempo restante e índice de origen/destino (con Spearman) y la sensibilidad a κ ∈ {0.5, 1, 2} (`MarketProcess.kappa`, que entra en la firma). La comparación B queda solo en la evaluación. |
| [main.py](main.py), [transfer_learning.py](transfer_learning.py) | **Hecho (Fase 3):** sin la fase de Ṽ. `NODE_SEQUENCE` sin cambios. |
| post_training.py, tests/test_post_value_head.py, tests/test_escenario.py, tests/test_pricing_integration.py | **Eliminados (Fase 3).** validate_pricing.py se reescribió en la Fase 6. |
| README, [metodologia_realizaciones_mercado_reefer.md](metodologia_realizaciones_mercado_reefer.md) | **Hecho (Fase 7).** README sin Ṽ ni p*, con la cotización, la comparación B y validate_pricing. Metodología: Paso 4b punto 4 (mejor competidor, m_c por L/T), pseudocódigo y Paso 5 con la cotización. |

## 11. Tests de aceptación

1. Con `m = 1`, el precio del competidor `F⁻¹(u_comp)·e^δ` coincide con la tarifa sorteada actual.
2. La frecuencia empírica de ganar en el nivel k converge a `(1 − u_k)^m` (Monte Carlo).
3. El nivel 0 siempre gana; los niveles son monótonos en precio por lane.
4. Determinismo: misma semilla, nodo, día y nivel ⇒ mismo resultado, sin importar el lote ni el orden.
5. Un rechazo no consume K, avanza `COTIZACION_HORAS`, enmascara la cotización en esa lane y no
   cambia el tablero.
6. Después de `COTIZACIONES_MAX_POR_NODO` rechazos solo quedan opciones VACÍO.
7. Margen reportado = Σ `lane_reward` con el precio cotizado, para todos los métodos.
8. UB-PI ≥ todos los métodos en cada mundo; B3 ≤ UB-DP ≤ UB-PI en promedio (las pruebas actuales,
   adaptadas).
9. UB-DP con tablas chicas = enumeración ingenua de los dos intentos (como el test actual de
   tabla V vs. recursión ingenua).
10. La log-prob de la acción híbrida es la suma de los dos términos; con solo VACÍO válido, el término
    de precio es 0.
11. Smoke de entrenamiento a 10 nodos: el loss es finito y la tasa de aceptación está en (0, 1).

## 12. Decisiones (cerradas el 2026-11-23)

| # | Decisión | Valor |
|---|---|---|
| 1 | Forma de m(LTR), κ, M_MAX | `1 + κ·LTR_REF/LTR`, **κ = 1**, tope 20; sensibilidad κ ∈ {0.5, 2} |
| 2 | Tiempo por cotización rechazada | **0.25 h** |
| 3 | Tope de cotizaciones por visita | **2** |
| 4 | Regla de precio de B1–B3 | **`R_MIOPE`** (sensibilidad: `R_MEDIANA`) |
| 5 | Precio de B3 | **regla fija**, igual que los demás |
| 6 | UB-DP a 75/100 nodos | proceso aparte, sin bajar `UBDP_S`; si supera `UBDP_TIEMPO_MAX_HORAS` (24 h por defecto) o la memoria, se reporta **"no tratable"** (§7.3) |

**Entornos de ejecución.** Las corridas completas (10 a 100 nodos) son en el servidor con GPU.
En Windows se corre hasta 10 nodos en CPU, así que todos los tests tienen que ser rápidos a 10 nodos.

## 13. Orden de implementación

1. [pricing.py](pricing.py) y [market_process.py](market_process.py), con los tests 1–4.
2. Entorno, con los tests 5–7.
3. Limpieza de Ṽ.
4. Modelo y PPO, con los tests 10–11.
5. Baselines, UB-PI y UB-DP, con los tests 8–9.
6. Evaluación y reportes.
7. Entrenamiento de 10 a 100 nodos.
