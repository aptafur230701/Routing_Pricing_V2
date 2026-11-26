# Fase 0: especificación del ruteo y pricing conjunto

Estado: **aprobada** (2026-11-23). Todas las decisiones de §12 están cerradas.

**La cotización es el único modo** (decidido el 2026-11-23): el entorno, el entrenamiento, la validación,
la evaluación y la inferencia ya no tienen el modo anterior (sin precio) ni una bandera para elegirlo.

**Esperar y reposicionarse** (decidido el 2026-11-24, §4): el objetivo es el margen total del ciclo. En cada
nodo, casa incluida antes de salir, el camión puede cotizar (p10 a p90, ningún nivel gana con seguridad),
esperar al siguiente día de mercado o reposicionarse vacío hasta 300 mi; desde un intermedio, volver a casa
vacío a cualquier distancia. Reemplaza la regla anterior, en la que VACÍO solo valía hacia casa o en la
primera salida (a cualquier distancia) y perder la cotización a casa obligaba a volver vacío.

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
| Espacio de precio | 9 niveles discretos en escala de percentiles (p10 … p90), comunes a todas las lanes. Desde el 2026-11-24 no hay nivel que gane siempre. |
| Movimientos sin carga | Esperar al siguiente día de mercado; reposicionarse vacío a ≤ `REPOSICION_MAX_MILLAS` = 300 mi; volver a casa vacío desde un intermedio. Iguales en casa y en los intermedios (2026-11-24). |
| Aceptación | `P(ganar \| u_k) = (1 − u_k)^m(LTR_i)`, con números aleatorios comunes. |
| Rechazo | La lane se enmascara y el camión vuelve a decidir en el mismo nodo, con penalización de tiempo. |
| Ṽ(s,j) / post_training | Se eliminan. Solo queda la CriticHead V(s). |
| Comparaciones | Solo la **comparación B**: destino del DRL + reglas de precio fijas vs. el DRL completo. Además, la tabla de baselines (B1–B3, UB-DP, UB-PI) con una regla fija. |
| Tamaños | Cadena completa 10 → 20 → 50 → 75 → 100 (35 se quitó el 2026-11-24), todo reentrenado desde cero. UB-DP y B3 se calculan a 10 y 20 nodos (`UBDP_MAX_NODOS` = 35); a 50, 75 y 100 son "no tratable". |

## 3. Modelo de mercado

### 3.1 Niveles de precio

```
U_NIVELES = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]     (config.py)
p_k(i→j, día) = _interp_cuantiles(u_k, cuantiles_{i→j, día_p}) · e^δ          [USD/milla]
```

- `cuantiles` son los 6 puntos (min, q1, q2, q3, p90, max) de la lane ese día. `día_p` es el mismo
  día con cuantiles que usa hoy `_dia_precio`: el propio día o el más cercano con cargas.
- `δ` es el corrimiento nacional del episodio (`MarketProcess._delta`); en replay, δ = 0.
- El nivel más bajo es el p10 y el más alto el p90. Ningún nivel gana con seguridad: P(ganar | p10) =
  0.9^m. (Hasta el 2026-11-24 existía el nivel 0, el mínimo del mercado, que ganaba siempre: el DRL
  aprendió a cotizar ahí, cobrando ~1 USD/mi bajo la mediana, porque perder significaba volver vacío.)
- Puede quedar por debajo del break-even. Así la política puede aceptar cargas con pérdida que la
  dejan bien ubicada (la evaluación mide cuántas: `cargas_bajo_be_pct`).
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

Acción = (j, o), donde `o ∈ {nivel 0 (p10), …, nivel 8 (p90), VACÍO}` (10 opciones). Las reglas son las
mismas en casa (antes de salir) y en los nodos intermedios.

| Opción | Cuándo es válida |
|---|---|
| Nivel k, j ≠ i | A la lane i→j le queda alguna carga sin perder en esta visita (`loads > perdidas`) y quedan cotizaciones. |
| VACÍO, j ≠ i (reposicionarse) | `dist(i, j) ≤ REPOSICION_MAX_MILLAS` (300 mi). Vale aunque la lane tenga carga. |
| VACÍO, j = casa desde un intermedio | Siempre, a cualquier distancia. |
| VACÍO, j = i (**esperar**) | El camión se queda en i hasta el siguiente día de mercado (`fin_espera`: el próximo múltiplo de 14 h). Válido si después todavía puede volver a casa (en casa antes de salir: hacer la ida y vuelta más corta). |

Factibilidad de tiempo: un destino intermedio exige poder volver a casa dentro de `MAX_DURATION`; volver a
casa desde un intermedio vale siempre (si ya no alcanza, el paso termina con la penalización). Con K, cuando
queda un solo arco solo valen casa y esperar.

El tope de 300 mi es fijo en todos los tamaños. A 10 nodos ningún par queda a ≤ 300 mi (el más cercano está
a 347 mi), así que ahí solo hay cotizar, esperar y volver a casa; a 20 nodos 10 de los 20 nodos tienen un
vecino a ≤ 300 mi y desde 50 nodos casi todos.

### 4.2 Transición

- **Se gana** (nivel k con `u_k < u_comp`) o **VACÍO a j ≠ i**: el camión va a j. Consume un arco
  de K, avanza el tiempo `T[i,j]` y al llegar se revela el tablero de j.
- **Se pierde**: el camión se queda en i, el tiempo avanza `COTIZACION_HORAS` y se pierde **esa carga** de la lane
  i→j. La lane se puede volver a cotizar con otra carga (otro broker), hasta agotar las suyas o el tope de la
  visita. **No se consume K** y el resto del tablero no cambia.
- **Varias cargas por lane (2026-11-25).** Cada carga tiene su propio competidor `u_comp`, independiente y
  sorteado con la semilla del mundo (los primeros C = `COTIZACIONES_MAX_POR_NODO` de cada lane). Antes, perder una
  vez cerraba la lane por todo el día, aunque la mediana en los datos es de 14 cargas por lane y día: de ahí
  venían los regresos vacíos y las cotizaciones cerca del piso.
- **Espera**: el tiempo salta a `fin_espera(t)`, reward 0, no consume K ni cuenta como arco. Se revela el
  tablero de i del día nuevo (cargas, precios y competidor nuevos, con la misma semilla del mundo) y se
  reinician los rechazos: es una visita nueva.
- **Tope de cotizaciones** `COTIZACIONES_MAX_POR_NODO`: después de ese número de rechazos en la
  misma visita, solo quedan esperar, reposicionarse o volver a casa. Así los episodios quedan acotados y
  el entorno coincide con la DP de §7.3.
- **Sin jugadas en casa.** Si ya en `reset()` no hay ninguna jugada, el camión **no sale**: margen 0, no es
  violación (`env.sin_salida`); es exógeno y le pasa igual a todos los métodos (a 10 nodos, la casa 9, cuya
  ida y vuelta más corta es de 72.7 h, en ~91 % de sus mundos). Si la política tenía jugadas y se quedó sin
  ellas (perdió las cotizaciones o esperó hasta que no quedó nada), el ciclo **falla**: violación con
  `TIME_VIOLATION_PENALTY`. El ciclo hay que hacerlo; cotizar alto a propósito no sirve para saltárselo.

Decidido: `COTIZACION_HORAS = 0.25` (una franja de la DP, 15 min) y `COTIZACIONES_MAX_POR_NODO = 4`
(2026-11-25; antes 2): hasta 1 h de llamadas si se pierden todas.

### 4.3 Reward

```
r = round( p_k · dist(i→j) − costo(i→j) )     si se gana con el nivel k
r = round( − costo(i→j) )                      si VACÍO a j ≠ i
r = 0                                          si se pierde o si espera (solo pasa el tiempo)
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
| nueva | — | cargas que se pueden cotizar todavía en j / `COTIZACIONES_MAX_POR_NODO` (0 = ninguna) |
| [7–10] continuación | EV con tarifa media | EV con la regla miope óptima (§6) sobre cuantiles medios |

| Feature de contexto | Nuevo |
|---|---|
| `log(m_i)` | competencia en el nodo actual |
| rechazos en esta visita / `COTIZACIONES_MAX_POR_NODO` | — |

La tarifa sorteada deja de ser observable, porque ahora es el competidor. N_NODE_FEATURES pasa de 11 a 14.

**Fila del propio nodo = la espera (2026-11-24).** La fila j = i describe esperar: reward [0] = 0, tiempo [1] =
horas hasta el siguiente día de mercado y continuación [6–9] desde i mañana (en casa, [7] es la holgura
después de esperar y [6], [8], [9] valen 0). No hay columnas nuevas.

**Implementado (Fase 4).** Las 3 features nuevas van en las **últimas** columnas (11–13), después del
índice de mercado (columna 10, `INDEX_COL`), y el bloque de contexto de cotización va al **final** de la
entrada de ContextNetwork. Así los checkpoints anteriores cargan con esas columnas en cero. La dispersión
se calcula con los niveles, `(p_0.8 − p_0.2)/p_0.5`, porque q1 y p90 no son niveles. Las features de
continuación [6–9] usan, desde la Fase 5, la misma tabla que B2 y B2P: el valor de cotizar con la regla
miope sobre los cuantiles medios del día de la semana (ver §7).

**Valor por nivel (2026-11-25).** La cabeza de precio no veía los precios: solo la dispersión y el break-even. Por
lane se agregan `v_k = P_k·(p_k·d − c)/p95` y `P_k` de cada nivel, el valor de la lane con todos sus intentos (`pricing.valor_reintentos`) si
perderlos vale 0 y si vale −c (0 si no se puede cotizar); N_NODE_FEATURES pasa a 34. Esas columnas entran además directo a `PriceHead`, cuyos logits son
`τ·(v_k + P_k·Δ) + MLP_k`, con Δ aprendido = lo que vale ganar la carga más allá del tramo. Con Δ = 0 arranca en
la cotización miope. Las features [8–9] usan `EvEsperado.ev_casa` (en el arco que cierra el ciclo, perder la carga
es volver vacío). Los checkpoints anteriores no cargan.

## 5. Arquitectura DRL y PPO

```
Encoder Transformer → embeddings e_j, contexto h_t
  ├─ Pointer decoder       → π(j | s)                (sin cambios, máscara de §4.1)
  ├─ Cabeza de precio      → π(o | s, j)             MLP([h_t ; e_j ; glimpse_t]) → 10 logits con máscara
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
- **Buffer**: `max_steps` = `(max_pasos + max_esperas) · (COTIZACIONES_MAX_POR_NODO + 1)`, porque en cada
  visita (una llegada o una espera) caben hasta 4 rechazos más el paso que viaja o espera
  (`routing_env.max_pasos_cotizacion`). Un episodio sin ninguna jugada desde casa arranca inactivo y no
  aporta transiciones.
- **Checkpoints**: la cabeza de precio pasó de 11 a 10 salidas, así que los checkpoints anteriores al
  2026-11-24 no sirven; se entrena de nuevo desde 10 nodos.

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
| `R_MIOPE` | `argmax_k P_k·(p_k·d − c) + (1 − P_k)·V_{a−1}` con a cargas por cotizar en la lane (`valor_reintentos`, V_0 = 0): optimiza contra el mercado, pero solo el tramo actual. |

Decidido: los baselines B1–B3 de la tabla principal cotizan con `R_MIOPE`, porque así son más
fuertes y la comparación es más creíble. `R_MEDIANA` queda como sensibilidad.

**Última oportunidad (2026-11-24).** Si perder la cotización deja al camión sin ninguna jugada (en casa, sin
poder esperar ni reposicionarse y sin otra cotización), el ciclo fallaría. Ahí toda regla fija (también las de
la comparación B) cotiza `argmax_k P_k·(p_k·d − c) + (1 − P_k)·PENALIZACION_USD` (−3000 USD): baja el precio
solo lo que pide la competencia del hub, en vez de fallar con su nivel de siempre (`ultima_oportunidad` en
[baselines/politicas.py](baselines/politicas.py)). Sin esta regla, a 10 nodos los baselines fallaban 38–46 %
de los ciclos; con ella, ~10 %, el riesgo que queda porque ningún nivel gana con seguridad.

**Lo que cuesta perder (2026-11-25).** En la vuelta a casa perder la cotización no vale 0: el camión vuelve vacío
(−c). Ahí toda regla fija cotiza `argmax_k P_k·(p_k·d − c) + (1 − P_k)·F` con F = −c (`costo_de_perder`); en la
última oportunidad, F = −3000; en los demás tramos F = 0. Antes las reglas cotizaban la vuelta en p70–p90, la
perdían y volvían vacías en casi la mitad de los ciclos.

**Margen con las fallas (2026-11-25).** El margen que se reporta (y el de la validación del entrenamiento) resta
3000 USD por ciclo fallido, igual que el entrenamiento y UB-DP. Antes contaba 0 y no era comparable con UB-DP.

## 7. Baselines y cotas

Todos se ejecutan en el entorno nuevo: rechazos, re-decisión y tope de cotizaciones.

| Método | Destino | Precio |
|---|---|---|
| B1 | jugada con mayor valor inmediato esperado `max_k P_k(p_k d − c)` (vacío: −c; esperar: 0) | regla fija |
| B2 / B2P | recursión actual, pero el valor esperado de un arco futuro es `P(carga)·max_k P_k(p_k d − c)` sobre cuantiles medios (B2P: más el pronóstico del índice); en cada nodo futuro también puede esperar al día siguiente | regla fija |
| B3 | un paso sobre la tabla V de UB-DP: el (j, k) que alcanza B_0 (§7.3) | el k de ese mismo máximo |
| UB-DP | DP de §7.3 | óptimo dentro de la DP |
| UB-PI | información perfecta (§7.4) | el nivel más alto que gana |

### 7.1 Comparación B

El destino lo da la cabeza de destino del DRL, y si hay un rechazo vuelve a decidir (puede
insistir en la misma lane con otra carga). El precio se calcula con cada regla:

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

### 7.3 UB-DP con decisión de precio (cuatro intentos, mismo tablero)

Para un tablero sorteado `b` en (casa h, nodo i, franja s, k):

```
G_jk(s)  = p_jk·d_ij − c_ij + V_{k−1}(j, s + f_ij)          V(casa) = 0
P_k      = (1 − u_k)^m_i
E(s)     = mejor jugada sin carga: max( volver a casa vacío −c_ih,
                                       reposicionarse R0_ij + V_{k−1}(j, s + f_ij)  (d_ij ≤ 300 mi),
                                       esperar V_k(i, s_w) )                         (s_w: siguiente día)
B_r(s)   = max( E(s+rτ), max_j max_k [ P_k G_jk(s+rτ) + (1 − P_k) B_{r+1} ] )    tras r rechazos, lanes con más de r cargas
B_R(s)   = E(s+Rτ)                                                              R = COTIZACIONES_MAX_POR_NODO
V_k(i,s) = media_b B_0(s)
H_k(h,s) = lo mismo en casa sin haber salido (sin "volver"; esperar lleva a H_k(h, s_w)); UB-DP = H_K(h, 0)
```

- La espera lee la misma tabla en una franja posterior (`s_w > s`), así que la inducción hacia atrás sigue
  bien fundada. Si en casa no hay ninguna jugada en s = 0, el valor es 0 (no sale); más tarde, o si después
  de perder no queda respaldo, es `PENALIZACION_USD` (el ciclo falla), igual que el entorno. B3 usa
  `valor_salida` (H) para esperar en casa.

- Sigue siendo una inducción hacia atrás. `s+τ` y `s+f` son franjas posteriores, ya calculadas,
  porque τ = 1 franja (hasta R·τ franjas después).
- **No hay sesgo por volver a sortear**: después de un rechazo se reusa el **mismo** tablero,
  no un tablero nuevo. Un tablero nuevo solo se ve esperando al día siguiente, que cuesta tiempo.
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
    (`cargar_o_calcular_tabla` devuelve None de inmediato): con los intentos de cotización el costo crece
    con N³ · S y esperar el tope de 24 h por tamaño no es viable. B3 y UB-DP se reportan "no tratable".

### 7.4 UB-PI

Con el mundo completo se conoce `u_comp`, así que en cada arco se cotiza el nivel más alto con
`u_k < u_comp`; si ningún nivel gana, la lane queda sin carga (solo se puede recorrer vacía, si las reglas lo
permiten). El reward del arco queda determinado y no hay rechazos. `solve_ub_pi` es una DP hacia atrás sobre
(nodo, franja, k) con las mismas jugadas del entorno: cargada, vacía (≤ 300 mi o a casa) y esperar; la salida
de casa es una tabla aparte (puede esperar en casa). Sigue siendo cota de todos los métodos (lo verifica la
evaluación mundo a mundo).

## 8. Evaluación y reportes

Métricas nuevas en [evaluation.py](evaluation.py), [stats_analysis.py](stats_analysis.py) y
[excel_report.py](excel_report.py), por método:

- margen total (Σ reward en USD; definición nueva);
- tasa de aceptación = cargas ganadas / cotizaciones;
- cotizaciones por arco recorrido y horas perdidas por rechazos;
- nivel cotizado (histograma) y USD/mi cobrado frente a la mediana de la lane;
- dinero dejado en la mesa, solo en mundos simulados: `Σ (p_comp − p_cobrado)·d` en las cargas ganadas;
- (2026-11-24) esperas y reposicionamientos por ruta, y % de cargas ganadas por debajo del costo de la lane
  (`cargas_bajo_be_pct`) con lo perdido en ellas: si el DRL baja de precio para llegar a mejores mercados,
  aparece aquí y en el margen de la ruta.

[validate_pricing.py](validate_pricing.py) se reescribe. Pasa a reportar el nivel cotizado por el DRL
según el LTR, el momento del ciclo (tiempo y k restantes) y el índice de mercado; la comparación B; y
la sensibilidad a κ. Desaparecen la prueba contrafactual y el rango A/B.

## 9. Configuración ([config.py](config.py))

**Nuevas**: `U_NIVELES`, `COMPETENCIA_KAPPA`, `COMPETENCIA_M_MAX`, `COTIZACION_HORAS`, `COTIZACIONES_MAX_POR_NODO`,
`PPO_ENTROPY_COEF_PRECIO_START/END`, `REGLA_PRECIO_BASELINES`, `UBDP_TIEMPO_MAX_HORAS`, `REPOSICION_MAX_MILLAS`
(2026-11-24).
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
3. Ningún nivel gana siempre (el más bajo es p10); los niveles son monótonos en precio por lane.
4. Determinismo: misma semilla, nodo, día y nivel ⇒ mismo resultado, sin importar el lote ni el orden.
5. Un rechazo no consume K, avanza `COTIZACION_HORAS`, gasta una carga de esa lane y no
   cambia el tablero.
6. Después de `COTIZACIONES_MAX_POR_NODO` rechazos solo quedan opciones VACÍO (casa, reposicionarse a
   ≤ 300 mi, esperar). Esperar salta al siguiente día de mercado, no consume K, revela un tablero nuevo y
   reinicia los rechazos. Sin jugadas en reset() ⇒ no sale (válido); sin jugadas después ⇒ falla.
7. Margen reportado = Σ `lane_reward` con el precio cotizado, para todos los métodos.
8. UB-PI ≥ todos los métodos en cada mundo; B3 ≤ UB-DP ≤ UB-PI en promedio (las pruebas actuales,
   adaptadas).
9. UB-DP con tablas chicas = enumeración ingenua de los cuatro intentos (como el test actual de
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
| 5 | Precio de B3 | ~~regla fija~~ → **óptimo de su tabla** (cambiado el 2026-11-25) |
| 6 | UB-DP a 75/100 nodos | proceso aparte, sin bajar `UBDP_S`; si supera `UBDP_TIEMPO_MAX_HORAS` (24 h por defecto) o la memoria, se reporta **"no tratable"** (§7.3) |
| 7 | Niveles de precio (2026-11-24) | **p10 … p90**: se quita el nivel que ganaba siempre |
| 8 | Movimientos sin carga (2026-11-24) | **esperar** al siguiente día de mercado; **reposicionarse** vacío a ≤ **300 mi** (fijo en todos los tamaños); volver a casa vacío desde un intermedio. Mismas reglas en casa |
| 9 | Sin jugadas en casa (2026-11-24) | desde el inicio ⇒ **no sale** (0, válido); después de tener jugadas ⇒ **el ciclo falla** (violación) |
| 10 | Última oportunidad de las reglas fijas (2026-11-24) | **miope con el costo de fallar** (`PENALIZACION_USD`), no p10 |
| 11 | Costo de perder en las reglas fijas (2026-11-25) | en la vuelta a casa, perder = **volver vacío (−c)** |
| 12 | Margen reportado (2026-11-25) | **−3000 USD por ciclo fallido** |
| 13 | Cabeza de precio (2026-11-25) | ve el **valor esperado de cada nivel**; `τ·(v_k + P_k·Δ) + MLP` |
| 14 | Cargas por lane (2026-11-25) | **varias**: perder gasta una carga, no la lane; tope **4** cotizaciones por visita |
| 15 | Tarifas del archivo (2026-11-25) | son **publicadas** en el load board, no pagadas; κ sigue siendo un supuesto (sensibilidad {0.5, 1, 2}) |

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
