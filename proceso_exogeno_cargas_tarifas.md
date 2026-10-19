# Implementación: proceso exógeno de cargas y tarifas

Carlos Vergara · octubre de 2026

Cada vez que el agente llega a un nodo, el proceso sortea cuántas cargas hay en cada lane que sale de él y a qué tarifa por milla. Se estima del histórico con dos modelos, uno de conteos y otro de tarifas, y no depende de las decisiones del agente.

## Contexto y motivación

Esta sección explica por qué existe este documento. Léela antes de implementar: define qué problema se resuelve, qué se decidió y por qué, y qué principios no deben romperse.

### El problema que resuelve

Hoy el entorno del DRL es determinista una vez elegido el día de inicio de cada episodio. Según el código actual, cuando el camión llega a un nodo el entorno lee lo que había ese día en el histórico (`loads_stack[día, nodo]` y `rate_stack[día, nodo]`, a través de `draw_lane_availability` y `rm_pen_stack`). La única fuente de aleatoriedad es qué día del histórico se sortea al empezar. Esto tiene dos consecuencias:

- **Hay pocos mundos posibles y la red puede memorizarlos.** Con los días del histórico como únicos mundos, las cargas y tarifas de cada día funcionan como una huella casi única de ese día. La red puede aprender a reconocerlo y a anticipar lo que vendrá, en lugar de decidir bajo incertidumbre. Es el sobreajuste a un conjunto finito de "niveles" que describen Cobbe et al. (2019) en aprendizaje por refuerzo.
- **No se puede defender que el problema sea estocástico.** Es un problema determinista con un sorteo inicial.

Verifica estas afirmaciones contra el código antes de empezar (ver "Antes de empezar" en las instrucciones).

### El problema real

Un camión que se mueve a un nodo no sabe, antes de llegar, si habrá cargas allí ni a qué tarifa. Las ve al llegar. Esa incertidumbre es el centro del problema que se quiere resolver.

En el marco de Powell (2019) es **información exógena**: llega al sistema después de que se toma la decisión y no depende de ella. Una política es válida si decide solo con lo que sabe en ese momento (no anticipatividad). Este documento modela esa información con un proceso estimado del histórico, siguiendo el enfoque de Wang, Martins do Nascimento y Powell (Sección 5.2), donde las llegadas de cargas se estiman directamente de datos históricos, con estacionalidad por día de la semana, y se muestrean en un simulador.

### Qué logra este proceso

1. **Hace estocástico el problema.** El mismo nodo puede traer cargas y tarifas distintas en cada episodio, y el agente decide sin saber qué encontrará.
2. **Da mundos ilimitados.** En entrenamiento cada episodio es distinto, así que no hay un conjunto finito que memorizar.
3. **Permite evaluar bien.** Se generan mundos con semillas fijas y se evalúan el DRL y los baselines sobre los mismos mundos, con comparaciones pareadas e intervalos de confianza.
4. **Prepara el pricing.** El precio de reserva de una lane se compara contra la distribución de tarifas que ofrece el mercado. Esa distribución sale de este proceso, y el agente aprende el valor de aceptar o rechazar contra ella.

### Por qué este diseño

| Decisión | Por qué |
| --- | --- |
| Conteos con binomial negativa, no Poisson | Dentro de cada lane y día de la semana, la varianza es entre 1,8 y 10,2 veces la media (una Poisson daría 1). La binomial negativa absorbe esa sobredispersión |
| Se simula el número de cargas, no solo si hay o no | El histórico ya trae conteos. La lane existe si hay al menos una carga, y el volumen del nodo es la suma de sus lanes |
| Factor de día de la semana | El volumen del mercado varía con el día de la semana, y los factores se estiman del histórico |
| Dispersión por grupo de volumen, no por lane | Estimar una dispersión por lane no es confiable cuando muchas lanes tienen conteos bajos o en cero |
| Sin shrinkage en la tasa base | La lane mediana tiene cientos de cargas en el periodo, y las lanes con menos de una carga al día son una cuarta parte de las lanes pero menos del 1 % del volumen |
| Tarifa remuestreada del histórico de cada lane | La tarifa no depende del día de la semana ni, de forma relevante, del número de cargas, y el remuestreo no supone ninguna forma funcional |
| Lanes independientes entre sí | Es una simplificación asumida: la correlación media entre lanes del mismo origen es 0,14, frente a 0,045 entre orígenes distintos. Se declara como limitación |
| Parámetros estimados con todos los días | Decisión tomada: aprovechar todo el histórico. La consecuencia es que la validación es dentro de muestra y el replay de días reales no es una prueba independiente del simulador (se reporta como referencia descriptiva) |
| Parámetros tratados como conocidos | Se simplifica: no se modela la incertidumbre sobre los parámetros |

La regla general que guía el diseño es la parsimonia: con 120 días por lane no hay datos para modelos más complejos, y cada pieza se justifica con evidencia de los datos.

### Cómo encaja con el resto del proyecto

- **No cambia la arquitectura del modelo** (encoder, decoder, critic) ni los hiperparámetros de PPO. Cambia lo que el entorno le muestra al agente.
- **El modo actual sigue funcionando y es el valor por defecto.** El proceso nuevo se activa pasando un objeto `market_process` al entorno. Con `market_process=None` los resultados deben ser idénticos a los actuales.
- **El reloj es el del entorno actual.** El tiempo está en horas de servicio: 14 horas equivalen a un día de mercado, y la duración máxima de la ruta es de 77 horas. El proceso solo necesita el día de la semana del día de llegada a cada nodo.
- **Alcance actual.** Se implementa el proceso exógeno, su uso en el entrenamiento y su validación. La evaluación con mundos simulados y los baselines (Paso 6) se aplazan hasta que se reformulen los baselines. `sample_world` se implementa igual, para que queden listos.
- **Camino al pricing.** Hoy la tarifa se remuestrea sin condicionar. Las extensiones naturales, que **no** se hacen ahora, son la tendencia del nivel de tarifa, su persistencia de un día al siguiente y su relación con el mercado (LTR, supply, llegadas de camiones).

### Principios que no deben romperse

1. **El proceso no depende de las acciones del agente.** Con la misma semilla y los mismos parámetros, la salida es idéntica sin importar qué decida el agente.
2. **El agente observa `loads_row` y `rate_row` solo al llegar al nodo,** nunca antes ni de nodos que no ha visitado.
3. **Toda la aleatoriedad sale de un generador numpy propio,** nunca del estado global.
4. **Con `market_process=None`, el entorno reproduce los resultados actuales.**
5. **La fórmula de recompensa vive en una sola función compartida** entre el modo real y el simulado, para que no diverjan.

### Glosario

| Término | Significado |
| --- | --- |
| Nodo | Un hub de la red (hay 194 en los datos; se usan los primeros `num_nodes`) |
| Lane (i, j) | Par origen-destino. Las lanes que salen de un nodo son las posibles cargas desde él |
| `loads_row`, `rate_row` | Vectores de largo N con las cargas y la tarifa por milla de las lanes que salen del nodo en el momento de llegar. Son los objetos que el entorno hoy lee de `loads_stack[día, nodo]` y `rate_stack[día, nodo]` |
| Mundo (o escenario) | Un conjunto completo de cargas y tarifas, de forma [D, N, N], generado con una semilla. El DRL y los baselines se evalúan sobre los mismos mundos |
| Día de mercado | 14 horas de servicio |

### Marco conceptual

- Powell WB (2019). [A unified framework for stochastic optimization](https://collaborate.princeton.edu/en/publications/a-unified-framework-for-stochastic-optimization/). European Journal of Operational Research. Estado, decisión, información exógena, transición y objetivo.
- Ulmer MW, Goodson JC, Mattfeld DC, Thomas BW (2020). [On modeling stochastic dynamic vehicle routing problems](https://iro.uiowa.edu/esploro/outputs/journalArticle/On-modeling-stochastic-dynamic-vehicle-routing/9984380425102771). EURO Journal on Transportation and Logistics 9(2).
- Cobbe K, Klimov O, Hesse C, Kim T, Schulman J (2019). [Quantifying generalization in reinforcement learning](https://icml.cc/virtual/2019/oral/5146). ICML.

La tabla de referencias del proceso exógeno está al final de este documento.

## Datos y supuestos

El histórico de 120 días contiene todas las cargas publicadas en el spot market, es decir, en las loadboards. Por eso representa la oferta del mercado y no solo lo que alguien terminó moviendo. Llega como dos matrices de forma [día, origen, destino]: `loads_stack`, con el número de cargas de la lane en el día, y `rate_stack`, con la tarifa por milla de la lane en el día. Tanto la disponibilidad como la tarifa se observan por lane-día. El día 0 es el 1 de enero de 2026, un jueves.

**Supuestos del modelo**

- Las cargas publicadas en un día son las disponibles cuando el agente llega al nodo ese día.
- Las lanes se sortean de forma independiente entre sí, dado el día de la semana. Es una simplificación: en los datos la correlación media entre las series diarias de lanes del mismo origen es 0,14 (0,09 tras quitar el factor diario de todo el mercado), frente a 0,045 entre orígenes distintos. Se declara como limitación.
- Las tasas de cada lane son estables en el periodo, salvo el efecto del día de la semana. Para las cargas es una aproximación. Para la tarifa hay una tendencia al alza y persistencia día a día que el remuestreo no reproduce (fase 3); se declaran como limitaciones.
- La tarifa por milla de una lane no depende del día de la semana ni, de forma relevante, del número de cargas (evidencia en la fase 3).
- Una lane sin cargas en todo el histórico no genera cargas.
- Los parámetros se estiman con todos los días del histórico, sin excluir ninguno, y se tratan como conocidos.
- El agente no altera el proceso: sus decisiones no cambian las cargas ni las tarifas que encuentra.

## Proceso exógeno

Cuando el agente llega al nodo i el día d, el entorno sortea dos cosas para cada lane (i, j) que sale del nodo:

1. Cuántas cargas hay en la lane ese día, con una binomial negativa. La lane existe si hay al menos una carga.
2. La tarifa por milla de la lane ese día, remuestreada del histórico de la lane. Se sortea en todas las lanes y solo cuenta donde hay carga.

El resultado son dos vectores de largo N (número de nodos): `loads_row`, con las cargas de cada lane saliente, y `rate_row`, con su tarifa por milla (definida en todas las lanes, aunque solo cuenta donde hay carga). Son los mismos objetos que el entorno lee hoy de `loads_stack[día, nodo]` y `rate_stack[día, nodo]`. El agente los observa solo al llegar. El volumen del nodo es la suma de las cargas de sus lanes y no un sorteo aparte.

## Fase 1. Datos de ajuste

El modelo de cargas se estima sobre `loads_stack`, que ya incluye los días en que la lane no tuvo cargas, y el de tarifas sobre `rate_stack`.

**Días atípicos.** El volumen total del mercado tiene 12 días atípicos: el 13 de marzo y del 27 de marzo al 6 de abril. El 13 de marzo y del 27 al 30 de marzo el total es entre 3,4 y 4,5 veces el promedio de ese día de la semana; del 31 de marzo al 6 de abril cae a entre 0,38 y 0,53 veces. Se está revisando con el equipo de datos si son un problema de captura o demanda real. Mientras tanto no se excluyen: el ajuste usa todos los días. Incluirlos mueve los factores de día de la semana (lunes 1,43 frente a 1,31 sin ellos; viernes 1,14 frente a 0,96) y baja el phi del grupo de mayor volumen de 4,4 a 2,6, es decir, más sobredispersión: los picos y caídas del mercado se absorben como variabilidad extra de cada lane.

**Días de ajuste.** Se usan solo los primeros `TRAIN_DAYS` días (90) de `loads_stack` y `rate_stack`; los días 90-119 quedan fuera del ajuste para que el replay de evaluación sea fuera de muestra. El código no lleva índices de día escritos a mano: el número de días sale de la forma de las matrices y las fechas salen de `FECHA_INICIO` (1 de enero de 2026) más el índice del día. `DIAS_EXCLUIDOS`, una lista de fechas vacía por defecto, permite dejar fuera días que la fuente confirme como erróneos.

**Construcción**

- Universo de lanes: las lanes (origen, destino) con al menos una carga en el histórico.
- Conteos: `loads_stack` tal cual, con los ceros.
- Tarifas: `rate_stack` de cada lane en todos los días del histórico, con o sin cargas. En los datos la tarifa es positiva en todos los lane-días y casi no cambia cuando la lane no tiene cargas (0,7 % menos).
- Día de la semana: se calcula desde `FECHA_INICIO` (jueves 1 de enero de 2026). Con esa fecha equivale a `dow = (día + 3) % 7`, con lunes igual a 0.
- Se estiman los parámetros con las matrices completas de 194 nodos y se recortan a `num_nodes` al cargarlos, igual que `load_matrices`.

**Entregable:** las matrices de ajuste con todos los días y la fecha de cada día.

## Fase 2. Cuántas cargas hay en cada lane

El número de cargas de la lane (i, j) en el día d se sortea de una binomial negativa, de forma independiente para cada lane que sale del nodo i:

```latex
N_{ij,t} \sim \text{NegBin}\left(\mu_{ij} \cdot s_d,\ \phi_{h(ij)}\right)
```

La media es m = mu_ij × s_d y la varianza es m + m² / phi. Se usa binomial negativa y no Poisson porque los conteos tienen sobredispersión: dentro de cada lane y día de la semana, la varianza es entre 1,8 y 10,2 veces la media (mediana por grupo de volumen, con los 120 días), cuando una Poisson daría 1. Parte de esa sobredispersión viene de la variación diaria común a todo el mercado, que la binomial negativa absorbe como variabilidad extra.

### Parámetros

| Parámetro | Qué es | Cómo se estima |
| --- | --- | --- |
| s_d | Factor del día de la semana, con promedio 1 | Cargas totales por fecha, promediadas por día de la semana |
| mu_ij | Tasa diaria base de la lane (i, j) | Cargas de la lane divididas entre la exposición E |
| phi_h | Dispersión del grupo h de lanes de volumen similar | Máxima verosimilitud conjunta por grupo |

### Estimación, en orden

1. **Factores de día de la semana.** Sumar las cargas de todas las lanes por fecha. Para cada día de la semana d, promediar esa suma entre sus fechas (unas 17 por día) y dividir por el promedio de todas las fechas. Reescalar para que los siete factores promedien 1.
2. **Tasa por lane.** Para cada lane, dividir su total de cargas n_ij entre la exposición E, que es la suma de s_d sobre las fechas de ajuste (los `TRAIN_DAYS` días de entrenamiento):

   ```latex
   \mu_{ij} = \frac{n_{ij}}{E}
   ```

   No se aplica shrinkage: la lane mediana tiene cientos de cargas en el periodo, y las lanes con menos de 1 carga por día son cerca de una cuarta parte de las lanes pero menos del 1 % del volumen.
3. **Dispersión por grupo de lanes.** Estimar un phi por lane no es confiable cuando muchas tienen conteos bajos o en cero. Se forman grupos h por cuartiles de mu_ij y, en cada grupo, se estima phi_h por máxima verosimilitud conjunta sobre todos los conteos lane-día del grupo, con media mu_ij × s_d. Si el estimador tiende a infinito, que equivale a Poisson, se fija un tope phi_max = 1000.

### Sorteo en cada llegada

Para cada lane j que sale del nodo i se calcula m = mu_ij × s_d y se sortea N con `rng.negative_binomial(phi_h, phi_h / (phi_h + m))`. Es equivalente a sortear una tasa gamma con media m y luego una Poisson con esa tasa. Un resultado N = 0 significa que la lane está vacía en ese momento. El valor N es el que se entrega en `loads_row`.

### Prueba de implementación

Con 1.000 simulaciones del panel, la media de cargas por lane y día y la proporción de días con cero cargas se parecen a las del histórico en cada grupo h.

**Entregable:** la tabla de parámetros (s_d, mu_ij, phi_h) y una función que recibe el nodo, el día de la semana, los parámetros y el generador aleatorio, y devuelve los conteos N de cada lane saliente.

## Fase 3. Tarifa por milla de la lane

Cada día, la tarifa por milla de la lane se remuestrea, con reemplazo, de las tarifas históricas de esa misma lane en todos los días del histórico, con o sin cargas. Es una distribución empírica por lane, sin forma funcional supuesta, y produce una sola tarifa por lane y día, igual que `rate_stack`.

**Evidencia y límites.** Con el archivo de tarifas, el día de la semana no mueve la tarifa (diferencia entre días menor de 1 %, frente a una variación normal de cerca de 10 % dentro de la lane; ANOVA p = 0,98) y el número de cargas casi tampoco (correlación mediana dentro de la lane entre 0,03 y 0,06; los días sin cargas tienen una tarifa 0,7 % menor). Por eso remuestrear sin condicionar es razonable. Hay dos límites. Primero, la tendencia: frente al promedio de enero a marzo, la tarifa media es -2,6 % en enero, +0,9 % en febrero, +2,6 % en marzo y +6,3 % en abril. Remuestrear sobre todos los días mezcla esos niveles, así que el simulador reproduce el nivel promedio del periodo. Segundo, la persistencia: la tarifa de una lane se parece a la del día anterior (autocorrelación mediana 0,61), mientras el remuestreo sortea cada día por separado. Ambas se declaran como limitaciones y no se corrigen.

### Prueba de implementación

- Los cuantiles 10, 50 y 90 de las tarifas simuladas coinciden con los históricos en cada lane.

**Entregable:** la tabla de tarifas históricas por lane, y una función que recibe la lane y el generador aleatorio y devuelve la tarifa por milla.

## Fase 4. Ensamblar la llegada a un nodo

La llegada a un nodo es una sola función que combina los dos pasos. Recibe el nodo, el día de la semana, los parámetros estimados y un generador aleatorio, y devuelve dos vectores de largo N, uno de cargas y otro de tarifas por milla.

```python
def llegada_a_nodo(i, d, params, rng):
    """Devuelve (loads_row, rate_row) para las lanes que salen del nodo i el día d."""
    # Paso 1: cuántas cargas hay en cada lane
    m = params.mu[i] * params.s[d]                           # (N,) media por lane
    phi = params.phi[params.grupo_volumen[i]]                 # (N,) dispersión por lane
    loads_row = rng.negative_binomial(phi, phi / (phi + m))   # (N,)

    # Paso 2: tarifa por milla remuestreada del histórico de la lane
    # (cualquiera de sus días del histórico)
    rate_row = remuestrear(params.tarifas, i, rng)            # (N,)
    # la tarifa se sortea en todas las lanes; el entorno la usa solo donde loads_row > 0
    return loads_row, rate_row
```

La misma clase entrega dos modos de uso:

- **Entrenamiento:** una versión por lotes, `sample_arrivals(nodos, días, rng)`, que devuelve matrices de forma (B, N) para los B episodios en paralelo. Reemplaza el ciclo de `draw_lane_availability` y la tabla `rm_pen_stack` del entorno vectorizado.
- **Evaluación y baselines:** `sample_world(semilla)`, que genera un mundo completo con forma [D, N, N], sustituto directo de `loads_stack` y `rate_stack`. Con la misma semilla para el DRL y los baselines, cada escenario es el mismo para todos los métodos.

Un arreglo de ceros en `loads_row` significa que no hay carga en ninguna lane del nodo en ese momento. El generador `rng` es independiente de las acciones del agente.

**Entregable:** el módulo con ambos modos y una prueba que verifique que, con la misma semilla y los mismos parámetros, la salida es idéntica sin importar qué decisiones tome el agente.

## Fase 5. Aleatoriedad en entrenamiento y evaluación

Los conteos y las tarifas se sortean cada vez que el agente llega a un nodo. El agente nunca observa la realización antes de llegar, y por eso la información es realmente incierta tanto en el entrenamiento como en la evaluación.

- Usar un generador de números aleatorios independiente de las decisiones del agente.
- En entrenamiento, sortear en cada llegada con `sample_arrivals`. El día de inicio de cada episodio solo aporta el día de la semana.
- En evaluación, generar M mundos completos con `sample_world` y semillas fijas, y usar los mismos mundos para el DRL y todos los baselines.
- Usar semillas distintas en entrenamiento y en evaluación, y fijar la semilla de cada episodio para poder reproducir los resultados.
- Mantener también la evaluación por replay de los días reales. Como los parámetros se estiman con todos los días, no es una prueba independiente del simulador: se reporta como referencia descriptiva.

**Alcance actual.** Los puntos de evaluación de esta fase se implementan con el Paso 6, que está aplazado. Por ahora solo se implementa la parte de entrenamiento y `sample_world`, para que los baselines puedan usarlo después sin cambios.

**Entregable:** la configuración de semillas, el valor de M y la prueba de reproducibilidad de la Fase 4.

## Fase 6. Validación del simulador

Como los parámetros se estiman con todos los días, la validación es dentro de muestra: se simulan 1.000 réplicas del periodo completo y se comparan con el histórico en tres comparaciones, de las cuales la de política queda aplazada hasta reformular los baselines:

| Comparación | Qué se compara | Cómo se lee |
| --- | --- | --- |
| Cargas por lane | Media diaria y proporción de días en cero de cada lane, agrupadas por grupo de volumen h | El valor real cae dentro del intervalo del 90 % de las réplicas simuladas |
| Tarifas | Cuantiles 10, 50 y 90 de la tarifa por milla por grupo de distancia (cuartiles), con QQ-plot | Los cuantiles reales caen dentro del intervalo del 90 % simulado en el periodo completo. Se reporta también la tarifa media real y la simulada por mes, que muestra la tendencia |
| Política (aplazada) | Rentabilidad por ruta de la política entrenada, en mundos simulados y en el replay de los días reales | Se reporta la diferencia; si la política rinde parecido en ambos, el simulador es suficiente para evaluarla |

**Entregable:** un script que produce la tabla resumen y los gráficos de las tres comparaciones.

## Instrucciones para Claude Code

Este bloque está escrito para Claude Code, en el repositorio AM-PPO. Implementa las fases 1 a 6 de este documento en el orden indicado, excepto lo marcado como **aplazado** (Paso 6 y la comparación de política de la fase 6), que no debes implementar todavía.

### Antes de empezar

Lee `routing_env.py` (RoutingEnv y VectorRoutingEnv), `problem_data.py` (`load_matrices`, `draw_lane_availability`, `build_day_matrices`, `build_rm_pen_stack`) y `config.py` (`TRAIN_DAYS`, `SEED`, `MPG`, `MARGINAL_COST_SIN_DIESEL`), y los puntos de `Solvers.py`, `evaluation.py` y `stats_analysis.py` que usan `loads_stack` y `rate_stack`. Confirma que `rate_stack` es tarifa por milla (el ingreso es tarifa por `distance_arr`) y revisa si trae cero o NaN en algún lane-día. En los datos entregados la tarifa es positiva en todos los lane-días, incluso sin cargas. Si encuentras ceros o NaN, reporta cuántos y dónde antes de seguir.

### Reglas generales

- El modo actual, que reproduce días reales, debe seguir funcionando y ser el valor por defecto. El proceso nuevo se activa pasando un objeto `market_process` al entorno.
- No cambies la arquitectura del modelo (encoder, decoder, critic) ni los hiperparámetros de PPO.
- La fórmula de recompensa del paso debe vivir en una sola función compartida con `build_day_matrices`, para que el modo real y el simulado no diverjan.
- Todo el muestreo usa un numpy Generator propio, nunca el estado global ni nada que dependa de las acciones del agente.
- No agregues dependencias nuevas. Escribe comentarios y docstrings en español, como el resto del repositorio.

### Pasos

1. **Configuración (`config.py`).** Agrega `FECHA_INICIO` (2026-01-01), `DIAS_EXCLUIDOS` (lista de fechas, vacía por defecto), el tope `phi_max = 1000`, la ruta de `datos/market_params.npz` y el número M de mundos de evaluación. No escribas índices de día a mano: el número de días sale de la forma de `loads_stack` y `rate_stack`, de modo que al actualizar el archivo el ajuste incluya los días nuevos.
2. **Estimación (`fit_market_process.py`, fases 1 a 3).** Con `loads_stack` y `rate_stack` de los 194 nodos y todos los días (menos `DIAS_EXCLUIDOS`, vacía por defecto), estima s_d, mu_ij y phi_h como describe la fase 2, y reúne las tarifas históricas de cada lane como describe la fase 3. Guarda todo en `datos/market_params.npz`, indexable por [origen, destino]. Imprime s_d y phi_h.
3. **Muestreo (`market_process.py`, fase 4).** Crea la clase `MarketProcess`. Carga `market_params.npz` y recorta a `num_nodes`, igual que `load_matrices`. Implementa `sample_arrivals(nodos, días, rng)`, que devuelve `loads` (B, N) y `rates` (B, N) sin ciclos sobre B, y `sample_world(semilla, num_días, día_inicial)`, que devuelve `loads_sim` y `rate_sim` con forma [D, N, N] y es determinista dada la semilla. El día de la semana sale de `FECHA_INICIO` más el índice del día (para el archivo actual equivale a `dow = (día + 3) % 7`). `rate` se sortea en todas las lanes y el entorno la usa solo donde `loads` es mayor que cero.
4. **Entorno (`routing_env.py`).** En RoutingEnv y VectorRoutingEnv agrega el parámetro opcional `market_process`, con valor por defecto `None`. Si se pasa, al hacer reset y al llegar a cada nodo obtén `loads_row` y `rate_row` con el proceso en lugar de `draw_lane_availability`. La máscara usa `loads_row` mayor que 0 con las mismas reglas de hoy, incluida la excepción del nodo de inicio. La recompensa del paso desde el nodo actual usa las filas sorteadas al llegar a ese nodo, con la fórmula compartida: ingreso igual a tarifa por distancia solo si hay carga, menos costo, redondeado. En este modo no uses `rm_pen_stack`. Con `market_process=None` los resultados deben ser idénticos a los actuales.

### Pasos de integración y pruebas

- **Paso 5. Entrenamiento (`main.py`, `am_training.py`).** Al iniciar, si `datos/market_params.npz` no existe, corre el ajuste. Crea el `MarketProcess` con `num_nodes` y pásaselo a los entornos de entrenamiento. El día de inicio de cada episodio se muestrea como hoy, pero solo aporta el día de la semana. Agrega un interruptor `USE_MARKET_PROCESS` en `main.py`. No cambies nada más del loop de PPO.
- **Paso 6. Evaluación (`evaluation.py`, `Solvers.py`, `stats_analysis.py`). APLAZADO: no lo implementes ahora.** Se hará cuando se reformulen los baselines, y no debes tocar esos tres archivos. Queda descrito para referencia. Agrega un modo de escenarios: genera M mundos con `sample_world` y semillas fijas, y evalúa el DRL y todos los solvers sobre los mismos mundos, pasando `loads_sim` y `rate_sim` donde hoy van `loads_stack` y `rate_stack`. Mantén la evaluación por replay con la partición `TRAIN_DAYS` actual, e indica en el reporte que no es independiente del simulador. Reporta los dos resultados por separado y haz que `stats_analysis.py` compare de forma pareada por escenario.
- **Paso 7. Validación (`validate_market_process.py`, fase 6).** Simula 1.000 réplicas del periodo completo con los parámetros estimados y produce la tabla y los gráficos de las tres comparaciones. La comparación de política (tercera fila de la tabla de la fase 6) queda aplazada hasta que exista el modo de escenarios del Paso 6: por ahora el script produce solo las comparaciones de cargas por lane y de tarifas.
- **Paso 8. Pruebas (carpeta `tests/`).** Misma semilla y mismos parámetros dan la misma salida. La salida no cambia con las acciones del agente. Formas y tipos son los esperados, y rate es positiva en todas las lanes. Con `market_process=None`, una ruta fija da las mismas recompensas que antes. La función compartida de recompensa, aplicada a una fila real, coincide con `rm_pen_stack`. La media y la proporción de ceros simuladas se parecen a las del histórico en cada grupo h. Otra prueba verifica que, con `DIAS_EXCLUIDOS` vacía, el ajuste usa todos los días de la matriz.
- **Paso 9. Documentación.** Actualiza `README.md` con los archivos nuevos y el flujo de trabajo.

### Criterios de aceptación

- Todas las pruebas pasan.
- Con `USE_MARKET_PROCESS = False`, `main.py` reproduce los resultados actuales.
- Con `USE_MARKET_PROCESS = True`, entrena sin errores al menos con 10 y con 35 nodos (la evaluación queda aplazada).
- Reporta el tiempo por paso de VectorRoutingEnv antes y después, para confirmar que el muestreo no es el cuello de botella.
- `validate_market_process.py` corre completo y genera las comparaciones de cargas por lane y de tarifas.

### Al terminar

Reporta los archivos creados y modificados, el resultado de las pruebas, los valores estimados de s_d y phi_h, y cualquier discrepancia que hayas encontrado entre este documento y el código.

## Referencias

| Pieza | Respaldo |
| --- | --- |
| Proceso de llegada de cargas estimado directamente del histórico, con estacionalidad por día de la semana | Wang, Martins do Nascimento y Powell (Sección 5.2) |
| Conteos con sobredispersión por binomial negativa | Cameron y Trivedi (2013); Hilbe (2011) |
| Tarifas estimadas del histórico | Wang, Martins do Nascimento y Powell |
| Información nueva que llega al sistema sin que el agente la influya: las demandas se revelan en t+1 | Cheung y Powell (1996); Godfrey y Powell (2002); Topaloglu y Powell (2006); Simao et al. (2009); Yang, Jaillet y Mahmassani (2004) |

- [Wang Y, Martins do Nascimento J, Powell WB. Reinforcement Learning for Dynamic Bidding in Truckload Markets: an Application to Large-Scale Fleet Management with Advance Commitments. arXiv:1802.08976v2, 2019.](https://ar5iv.arxiv.org/html/1802.08976)
- [Cameron AC, Trivedi PK (2013). Regression Analysis of Count Data, 2nd ed. Cambridge University Press.](https://cameron.econ.ucdavis.edu/racd2/)
- [Hilbe JM (2011). Negative Binomial Regression, 2nd ed. Cambridge University Press.](https://resolve.cambridge.org/core/books/abs/negative-binomial-regression/introduction/2765C67FDF4904A16F7AAC6368AE541C)
- Cheung RK, Powell WB (1996). An algorithm for multistage dynamic networks with random arc capacities, with an application to dynamic fleet management. Operations Research 44(6):951–963.
- Godfrey GA, Powell WB (2002). An adaptive dynamic programming algorithm for dynamic fleet management, I: Single period travel times. Transportation Science 36(1):21–39.
- Topaloglu H, Powell WB (2006). Dynamic-programming approximations for stochastic time-staged integer multicommodity-flow problems. INFORMS Journal on Computing 18(1):31–42.
- Simao HP, Day J, George AP, Gifford T, Nienow J, Powell WB (2009). An approximate dynamic programming algorithm for large-scale fleet management: A case application. Transportation Science 43(2):178–197.
- [Yang J, Jaillet P, Mahmassani H (2004). Real-time multivehicle truckload pickup and delivery problems. Transportation Science 38(2):135–148.](https://pubsonline.informs.org/doi/references/10.1287/trsc.1030.0068)

Los datos bibliográficos de las referencias sin enlace salen de la lista bibliográfica de Wang et al.; verifícalos en tu gestor bibliográfico antes de citarlos.
