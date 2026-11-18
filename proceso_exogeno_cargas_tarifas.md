# Implementación: proceso exógeno de cargas y tarifas

Carlos Vergara · octubre de 2026

> **Nota (nov 2026).** El generador de cargas y tarifas que describe este documento (binomial
> negativa con tasa global `mu_ij·s_d` + tarifa = `media_lane + índice + residuo`, ajustado sobre
> 120 días ene–abr 2026) fue **reemplazado** por el de
> [metodologia_realizaciones_mercado_reefer.md](metodologia_realizaciones_mercado_reefer.md), que
> usa días reales de jun–oct 2026 y cuantiles reales en vez de un modelo paramétrico de tarifas.
> Lo que SÍ sigue vigente de este documento es el **índice de mercado por nodo** (lento/rápido
> AR(1), ρ, σ_i, frac_var_lento) — se conserva tal cual, solo repuntado a los datos nuevos y ya no
> simulado por "mundo" (el índice de cualquier día es ahora un dato observado directo, porque los
> episodios siempre recorren días reales). Lo demás de este documento (fases 1-2 y la fase 3 vieja
> del modelo de tarifas) queda como referencia histórica del diseño anterior.

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
| Tarifa = media de la lane + índice de mercado del nodo + residuo de la lane | La tarifa no depende del día de la semana ni, de forma relevante, del número de cargas, pero sí tiene un componente común a las lanes que salen del mismo nodo y persistente en el tiempo (el estado del mercado del nodo). El índice lo captura con un nivel lento fijo en el mundo y una variación diaria AR(1) de poca memoria; el residuo se remuestrea del histórico de la lane, sin forma funcional |
| Lanes independientes entre sí | Es una simplificación asumida: la correlación media entre lanes del mismo origen es 0,14, frente a 0,045 entre orígenes distintos. Se declara como limitación |
| Parámetros estimados solo con los días de entrenamiento (0-89) | Decisión tomada: dejar los días 90-119 fuera del ajuste. La consecuencia es que el replay de esos días es una prueba fuera de muestra, mientras que la validación del simulador sobre los días de ajuste sigue siendo dentro de muestra |
| Parámetros tratados como conocidos | Se simplifica: no se modela la incertidumbre sobre los parámetros |

La regla general que guía el diseño es la parsimonia: con 90 días de ajuste por lane no hay datos para modelos más complejos, y cada pieza se justifica con evidencia de los datos.

### Cómo encaja con el resto del proyecto

- **No cambia la arquitectura del modelo** (encoder, decoder, critic) ni los hiperparámetros de PPO. Cambia lo que el entorno le muestra al agente.
- **El modo actual sigue funcionando y es el valor por defecto.** El proceso nuevo se activa pasando un objeto `market_process` al entorno. Con `market_process=None` los resultados deben ser idénticos a los actuales.
- **El reloj es el del entorno actual.** El tiempo está en horas de servicio: 14 horas equivalen a un día de mercado, y la duración máxima de la ruta es de 77 horas. El proceso solo necesita el día de la semana del día de llegada a cada nodo.
- **Alcance actual.** Se implementó el proceso exógeno, su uso en el entrenamiento, su validación y la evaluación con mundos simulados y baselines (Paso 6, ver `baselines_drl_ruteo_estocastico.md`).
- **Camino al pricing.** La tarifa tiene ahora un estado de mercado por nodo (índice = lento + rápido) que el agente observa. Las extensiones naturales, que **no** se hacen ahora, son la tendencia del nivel de tarifa, la correlación entre nodos del rápido y su relación con otras señales del mercado (LTR, supply, llegadas de camiones).

### Principios que no deben romperse

1. **El proceso no depende de las acciones del agente.** Con la misma semilla y los mismos parámetros, la salida es idéntica sin importar qué decida el agente.
2. **El agente observa `loads_row` y `rate_row` solo al llegar al nodo,** nunca antes ni de nodos que no ha visitado. La única señal de otros nodos es el índice de mercado del día actual (una feature por nodo, `market_index`), que no revela cargas ni tarifas de lanes concretas.
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
- Las tasas de cada lane son estables en el periodo, salvo el efecto del día de la semana. Para las cargas es una aproximación. Para la tarifa hay una tendencia al alza del nivel medio que el modelo no reproduce (fase 3); se declara como limitación. La persistencia día a día sí se modela, a nivel de nodo (índice).
- La tarifa por milla de una lane no depende del día de la semana ni, de forma relevante, del número de cargas (evidencia en la fase 3).
- Una lane sin cargas en todo el histórico no genera cargas.
- Los parámetros se estiman con los primeros `TRAIN_DAYS` días (0-89), sin excluir ninguno de ellos, y se tratan como conocidos. Los días 90-119 no participan.
- El agente no altera el proceso: sus decisiones no cambian las cargas ni las tarifas que encuentra.

## Proceso exógeno

Cuando el agente llega al nodo i el día d, el entorno sortea dos cosas para cada lane (i, j) que sale del nodo:

1. Cuántas cargas hay en la lane ese día, con una binomial negativa. La lane existe si hay al menos una carga.
2. La tarifa por milla de la lane ese día: media de la lane + índice de mercado del nodo ese día + residuo remuestreado del histórico de la lane (modelo de la fase 3). Se sortea en todas las lanes y solo cuenta donde hay carga.

El resultado son dos vectores de largo N (número de nodos): `loads_row`, con las cargas de cada lane saliente, y `rate_row`, con su tarifa por milla (definida en todas las lanes, aunque solo cuenta donde hay carga). Son los mismos objetos que el entorno lee hoy de `loads_stack[día, nodo]` y `rate_stack[día, nodo]`. El agente los observa solo al llegar. El volumen del nodo es la suma de las cargas de sus lanes y no un sorteo aparte.

## Fase 1. Datos de ajuste

El modelo de cargas se estima sobre `loads_stack`, que ya incluye los días en que la lane no tuvo cargas, y el de tarifas sobre `rate_stack`.

**Días atípicos.** El volumen total del mercado tiene 12 días atípicos: el 13 de marzo y del 27 de marzo al 6 de abril. El 13 de marzo y del 27 al 30 de marzo el total es entre 3,4 y 4,5 veces el promedio de ese día de la semana; del 31 de marzo al 6 de abril cae a entre 0,38 y 0,53 veces. Se está revisando con el equipo de datos si son un problema de captura o demanda real. Mientras tanto no se excluyen: el ajuste usa todos sus días. Del 13 de marzo y el 27-30 de marzo (picos) y el 31 de marzo (caída), los 6 días quedan dentro del ajuste (días 85-89 y el 13 de marzo); del 1 al 6 de abril (caída, entre 0,28 y 0,44 veces el volumen de ajuste de ese día de la semana) quedan en el replay de evaluación, que por lo tanto incluye un régimen que el proceso no vio. Los porcentajes de este párrafo sobre el nivel de los atípicos (3,4 a 4,5 veces y 0,38 a 0,53 veces) se calcularon con los 120 días. Con los 90 días de ajuste, incluir los atípicos mueve los factores de día de la semana (lunes 1,45 frente a 1,29 sin ellos; viernes 1,19 frente a 0,96) y baja el phi del grupo de mayor volumen de 4,5 a 2,65, es decir, más sobredispersión: los picos y caídas del mercado se absorben como variabilidad extra de cada lane.

**Días de ajuste.** Se usan solo los primeros `TRAIN_DAYS` días (90) de `loads_stack` y `rate_stack`; los días 90-119 quedan fuera del ajuste para que el replay de evaluación sea fuera de muestra (verificado: `datos/market_params.npz` trae `dias_usados` = 0-89, del 2026-01-01 al 2026-03-31, y `market_process.py` lo reajusta si no coincide con `TRAIN_DAYS`). El código no lleva índices de día escritos a mano: el número de días sale de la forma de las matrices y las fechas salen de `FECHA_INICIO` (1 de enero de 2026) más el índice del día. `DIAS_EXCLUIDOS`, una lista de fechas vacía por defecto, permite dejar fuera días que la fuente confirme como erróneos.

**Construcción**

- Universo de lanes: las lanes (origen, destino) con al menos una carga en el histórico.
- Conteos: `loads_stack` tal cual, con los ceros.
- Tarifas: `rate_stack` de cada lane en todos los días de ajuste, con o sin cargas. En los datos la tarifa es positiva en todos los lane-días y casi no cambia cuando la lane no tiene cargas (0,7 % menos). El índice de mercado se calcula sobre las lanes del universo, sin la diagonal (las lanes i→i no son jugables).
- Día de la semana: se calcula desde `FECHA_INICIO` (jueves 1 de enero de 2026). Con esa fecha equivale a `dow = (día + 3) % 7`, con lunes igual a 0.
- Se estiman los parámetros con las matrices completas de 194 nodos y se recortan a `num_nodes` al cargarlos, igual que `load_matrices`.

**Entregable:** las matrices de ajuste (días 0-89) y la fecha de cada día.

## Fase 2. Cuántas cargas hay en cada lane

El número de cargas de la lane (i, j) en el día d se sortea de una binomial negativa, de forma independiente para cada lane que sale del nodo i:

```latex
N_{ij,t} \sim \text{NegBin}\left(\mu_{ij} \cdot s_d,\ \phi_{h(ij)}\right)
```

La media es m = mu_ij × s_d y la varianza es m + m² / phi. Se usa binomial negativa y no Poisson porque los conteos tienen sobredispersión: dentro de cada lane y día de la semana, la varianza es entre 1,8 y 10,2 veces la media (mediana por grupo de volumen, calculada con los 120 días; no recalculada con los 90 de ajuste), cuando una Poisson daría 1. Parte de esa sobredispersión viene de la variación diaria común a todo el mercado, que la binomial negativa absorbe como variabilidad extra.

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

La tarifa por milla de la lane (i→j) el día d es

```
tarifa(i→j, d) = media_lane(ij) + índice(i, d) + residuo(ij)
índice(i, d)   = lento(i) + rápido(i, d)
rápido(i, d)   = ρ · rápido(i, d−1) + ε(i, d),   ε ~ Normal(0, σ_i²)
```

- **media_lane**: promedio de la tarifa de la lane en los días de ajuste, fija.
- **índice**: estado del mercado de salida del nodo i, neto del nivel nacional. Se estima como el promedio, sobre las lanes salientes de i del universo (sin la diagonal), de tarifa − media_lane; a eso se le resta el nivel nacional (el promedio sobre los 194 nodos).
- **lento**: estado de fondo de cada nodo, el promedio centrado del índice en `VENTANA_LENTO_DIAS` (21) días. Solo se usan los días con la ventana completa (los "días de escenario", 70 con `TRAIN_DAYS` = 90). En cada mundo se toma el vector de los 194 nodos de un día de escenario al azar y queda fijo durante todo el episodio.
- **rápido**: índice − lento. AR(1) con un solo ρ (regresión por mínimos cuadrados de rápido(d) sobre rápido(d−1), todos los nodos juntos) y una σ_i por nodo (desviación del error de esa regresión). Los nodos evolucionan de forma independiente. En un mundo arranca el día 0 en su distribución estacionaria Normal(0, σ_i² / (1 − ρ²)) y se genera hasta `MARKET_DIA_MAX` (último día de salida posible más los días que caben en `MAX_DURATION`).
- **residuo**: `tarifa − media_lane − índice`, por lane y día de ajuste. En el simulador se sortea un día al azar del histórico de residuos de esa lane (independiente por lane); incluye la variación nacional y la propia de la lane.
- Las tarifas por debajo del mínimo histórico de la lane se recortan a ese mínimo (≈ 3 % de las tarifas sorteadas); el contador está en `MarketProcess.estadisticas_recorte`.

Con los 90 días de ajuste: ρ ≈ 0,45, σ_i entre 0,03 y 0,10 $/milla, y el lento explica ≈ 56 % de la varianza del índice. La desviación del índice histórico es ≈ 0,106 $/milla, frente a ≈ 0,35 del residuo: el estado de mercado es una señal de baja amplitud.

**Qué ve el agente.** Una feature más por nodo j: `índice(j, día actual) / desviación del índice histórico` (una sola desviación, calculada en el ajuste). En mundos simulados es el índice latente del mundo; en el replay de días reales es el índice observado del día, calculado con la definición del ajuste (194 nodos, `media_lane` de entrenamiento, tarifas reales de ese día) y recortado a los N nodos.

**Dos formas de sortear.** (1) Los mundos (`sample_arrivals_por_mundo`, `sample_world`, entorno) usan el lento fijo y el rápido AR(1) de la semilla del mundo. (2) `sample_arrivals`, que usan la DP de UB-DP y B2, sortea en cada llegada un escenario lento al azar y un rápido de su distribución estacionaria, sin memoria: representa una política que conoce la distribución de las tarifas pero ignora el estado del mercado. Por eso UB-DP es el valor de la política que ignora el mercado y no es una cota; UB-PI sí lo es.

**Evidencia y límites.** El día de la semana no mueve la tarifa (diferencia entre días menor de 1 %, frente a una variación normal de cerca de 10 % dentro de la lane; ANOVA p = 0,98) y el número de cargas casi tampoco (correlación mediana dentro de la lane entre 0,03 y 0,06; los días sin cargas tienen una tarifa 0,7 % menor). Límites: (1) la tendencia del nivel: frente al promedio de enero a marzo, la tarifa media es -2,6 % en enero, +0,9 % en febrero, +2,6 % en marzo y +6,3 % en abril; el simulador reproduce el nivel promedio del periodo de ajuste y el replay encuentra tarifas en promedio más altas, lo que afecta a todos los métodos; (2) en un mundo el lento queda fijo, mientras que en el histórico deriva durante los 90 días, así que a horizontes largos la persistencia simulada del índice es algo mayor; (3) los nodos evolucionan de forma independiente (no hay correlación entre nodos del rápido).

### Prueba de implementación

- Los cuantiles 10, 50 y 90 de las tarifas simuladas coinciden con los históricos en cada lane.
- La pendiente de índice(d + h) sobre índice(d), juntando nodos, para h = 1, 3, 5 y 7, simulada frente a histórica.

**Entregable:** `fit_market_process.py` (parámetros en `market_params.npz`) y `market_process.py` (índice y tarifa del mundo).

## Fase 4. Ensamblar la llegada a un nodo

La llegada a un nodo es una sola función que combina los dos pasos. Recibe el nodo, el día de la semana, los parámetros estimados y un generador aleatorio, y devuelve dos vectores de largo N, uno de cargas y otro de tarifas por milla.

```python
def llegada_a_nodo(i, d, params, rng):
    """Devuelve (loads_row, rate_row) para las lanes que salen del nodo i el día d."""
    # Paso 1: cuántas cargas hay en cada lane
    m = params.mu[i] * params.s[d]                           # (N,) media por lane
    phi = params.phi[params.grupo_volumen[i]]                 # (N,) dispersión por lane
    loads_row = rng.negative_binomial(phi, phi / (phi + m))   # (N,)

    # Paso 2: tarifa por milla = media_lane + índice del mundo en (i, d) + residuo de la lane
    # (un día al azar del histórico de residuos, independiente por lane); recorte al mínimo histórico
    rate_row = armar_tarifa(params, i, indice_mundo(i, d), rng)   # (N,)
    # la tarifa se sortea en todas las lanes; el entorno la usa solo donde loads_row > 0
    return loads_row, rate_row
```

La misma clase entrega dos modos de uso:

- **Entrenamiento:** `sample_arrivals_por_mundo(nodos, días, semillas)`, que devuelve matrices de forma (B, N) para los B episodios en paralelo, cada fila función pura de (semilla del mundo, nodo, día). Reemplaza el ciclo de `draw_lane_availability` y la tabla `rm_pen_stack` del entorno vectorizado. `sample_arrivals(nodos, días, rng)` es la versión sin estado de mercado (un escenario lento y un rápido estacionario nuevos en cada llegada) que usan la DP de UB-DP y B2.
- **Evaluación y baselines:** `sample_world(semilla)`, que genera un mundo completo con forma [D, N, N], sustituto directo de `loads_stack` y `rate_stack`. Con la misma semilla para el DRL y los baselines, cada escenario es el mismo para todos los métodos.

Un arreglo de ceros en `loads_row` significa que no hay carga en ninguna lane del nodo en ese momento. El generador `rng` es independiente de las acciones del agente.

**Entregable:** el módulo con ambos modos y una prueba que verifique que, con la misma semilla y los mismos parámetros, la salida es idéntica sin importar qué decisiones tome el agente.

## Fase 5. Aleatoriedad en entrenamiento y evaluación

Los conteos y las tarifas se sortean cada vez que el agente llega a un nodo. El agente nunca observa la realización antes de llegar, y por eso la información es realmente incierta tanto en el entrenamiento como en la evaluación.

- Usar un generador de números aleatorios independiente de las decisiones del agente.
- En entrenamiento, sortear en cada llegada con `sample_arrivals_por_mundo` (un mundo por episodio). El día de inicio de cada episodio aporta el día de la semana y el día de partida del mundo.
- En evaluación, generar M mundos completos con `sample_world` y semillas fijas, y usar los mismos mundos para el DRL y todos los baselines.
- Usar semillas distintas en entrenamiento y en evaluación, y fijar la semilla de cada episodio para poder reproducir los resultados.
- Mantener también la evaluación por replay de los días reales 90-119. Como los parámetros se estiman solo con los días 0-89, es una prueba fuera de muestra del simulador y de la política, con las salvedades de régimen de abril descritas en la fase 1 (es una sola ventana de 30 días).

**Alcance actual.** Los puntos de evaluación de esta fase están implementados con el Paso 6 (`evaluation.py`, `stats_analysis.py`, `excel_report.py`).

**Entregable:** la configuración de semillas, el valor de M y la prueba de reproducibilidad de la Fase 4.

## Fase 6. Validación del simulador

Como los parámetros se estiman con los días de ajuste (0-89), esta validación es dentro de muestra: se simulan 200 mundos del periodo de ajuste y se comparan con ese histórico en tres comparaciones. La de política se hace en el arnés de evaluación (no en `validate_market_process.py`) y es la prueba fuera de muestra:

| Comparación | Qué se compara | Cómo se lee |
| --- | --- | --- |
| Cargas por lane | Media diaria y proporción de días en cero de cada lane, agrupadas por grupo de volumen h | El valor real cae dentro del intervalo del 90 % de las réplicas simuladas |
| Tarifas | Cuantiles 10, 50 y 90 de la tarifa por milla por lane (sesgo, error absoluto, cobertura del IC90 de las réplicas) y por grupo de distancia (cuartiles), con QQ-plot; % de tarifas recortadas al mínimo de la lane | Los cuantiles reales caen dentro del intervalo del 90 % simulado en el periodo de ajuste. Se reporta también la tarifa media real y la simulada por mes, que muestra la tendencia |
| Persistencia del índice | Pendiente de la regresión de índice(i, d + h) sobre índice(i, d), juntando nodos, h = 1, 3, 5 y 7 | La pendiente histórica cae dentro del IC90 simulado (a h largo se espera una simulada algo mayor: en un mundo el lento no deriva) |
| Política (en `evaluation.py`) | Rentabilidad por ruta de la política entrenada, en mundos simulados y en el replay fuera de muestra de los días 90-119 | Se reporta la diferencia; si la política rinde parecido en ambos, el simulador es suficiente para evaluarla |

**Entregable:** un script que produce la tabla resumen y los gráficos de las comparaciones.

## Instrucciones para Claude Code

> **Nota (modelo de tarifas actual).** Estas instrucciones se escribieron para el remuestreo de tarifas por lane, que ya no existe: la fase 3 y el ajuste de tarifas (`fit_market_process.ajustar_tarifas`, `market_process._mundo`) se reemplazaron por el modelo con estado de mercado por nodo descrito arriba. Lo que dice el bloque sobre `tarifas` históricas y `sample_arrivals` queda como referencia histórica.

Este bloque está escrito para Claude Code, en el repositorio AM-PPO. Implementa las fases 1 a 6 de este documento en el orden indicado, (el Paso 6 y la comparación de política de la fase 6 ya están implementados; el texto de esos pasos queda como referencia histórica).

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
2. **Estimación (`fit_market_process.py`, fases 1 a 3).** Con `loads_stack` y `rate_stack` de los 194 nodos y los primeros `TRAIN_DAYS` días (menos `DIAS_EXCLUIDOS`, vacía por defecto), estima s_d, mu_ij y phi_h como describe la fase 2, y reúne las tarifas históricas de cada lane como describe la fase 3. Guarda todo en `datos/market_params.npz`, indexable por [origen, destino]. Imprime s_d y phi_h.
3. **Muestreo (`market_process.py`, fase 4).** Crea la clase `MarketProcess`. Carga `market_params.npz` y recorta a `num_nodes`, igual que `load_matrices`. Implementa `sample_arrivals(nodos, días, rng)`, que devuelve `loads` (B, N) y `rates` (B, N) sin ciclos sobre B, y `sample_world(semilla, num_días, día_inicial)`, que devuelve `loads_sim` y `rate_sim` con forma [D, N, N] y es determinista dada la semilla. El día de la semana sale de `FECHA_INICIO` más el índice del día (para el archivo actual equivale a `dow = (día + 3) % 7`). `rate` se sortea en todas las lanes y el entorno la usa solo donde `loads` es mayor que cero.
4. **Entorno (`routing_env.py`).** En RoutingEnv y VectorRoutingEnv agrega el parámetro opcional `market_process`, con valor por defecto `None`. Si se pasa, al hacer reset y al llegar a cada nodo obtén `loads_row` y `rate_row` con el proceso en lugar de `draw_lane_availability`. La máscara usa `loads_row` mayor que 0 con las mismas reglas de hoy, incluida la excepción del nodo de inicio. La recompensa del paso desde el nodo actual usa las filas sorteadas al llegar a ese nodo, con la fórmula compartida: ingreso igual a tarifa por distancia solo si hay carga, menos costo, redondeado. En este modo no uses `rm_pen_stack`. Con `market_process=None` los resultados deben ser idénticos a los actuales.

### Pasos de integración y pruebas

- **Paso 5. Entrenamiento (`main.py`, `am_training.py`).** Al iniciar, si `datos/market_params.npz` no existe, corre el ajuste. Crea el `MarketProcess` con `num_nodes` y pásaselo a los entornos de entrenamiento. El día de inicio de cada episodio se muestrea como hoy, pero solo aporta el día de la semana. Agrega un interruptor `USE_MARKET_PROCESS` en `main.py`. No cambies nada más del loop de PPO.
- **Paso 6. Evaluación (`evaluation.py`, `Solvers.py`, `stats_analysis.py`). IMPLEMENTADO** con la reformulación de baselines; el texto queda como referencia histórica. Agrega un modo de escenarios: genera M mundos con `sample_world` y semillas fijas, y evalúa el DRL y todos los solvers sobre los mismos mundos, pasando `loads_sim` y `rate_sim` donde hoy van `loads_stack` y `rate_stack`. Mantén la evaluación por replay con la partición `TRAIN_DAYS` actual, y etiquétala en el reporte como "replay, fuera de muestra (días 90-119)". Reporta los dos resultados por separado y haz que `stats_analysis.py` compare de forma pareada por escenario.
- **Paso 7. Validación (`validate_market_process.py`, fase 6).** Simula 1.000 réplicas del periodo de ajuste con los parámetros estimados y produce la tabla y los gráficos de cargas por lane y de tarifas. La comparación de política (tercera fila de la tabla de la fase 6) se hace en `evaluation.py`, no en este script.
- **Paso 8. Pruebas (carpeta `tests/`).** Misma semilla y mismos parámetros dan la misma salida. La salida no cambia con las acciones del agente. Formas y tipos son los esperados, y rate es positiva en todas las lanes. Con `market_process=None`, una ruta fija da las mismas recompensas que antes. La función compartida de recompensa, aplicada a una fila real, coincide con `rm_pen_stack`. La media y la proporción de ceros simuladas se parecen a las del histórico en cada grupo h. Otra prueba verifica que, con `DIAS_EXCLUIDOS` vacía, el ajuste usa todos los días de la matriz.
- **Paso 9. Documentación.** Actualiza `README.md` con los archivos nuevos y el flujo de trabajo.

### Criterios de aceptación

- Todas las pruebas pasan.
- Con `USE_MARKET_PROCESS = False`, `main.py` reproduce los resultados actuales.
- Con `USE_MARKET_PROCESS = True`, entrena sin errores al menos con 10 y con 35 nodos (la evaluación se implementó después, en el Paso 6).
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
