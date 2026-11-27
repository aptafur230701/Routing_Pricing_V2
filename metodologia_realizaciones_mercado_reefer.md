# Metodología: realizaciones diarias del mercado reefer para entrenar el agente DRL

## Objetivo y resumen

El método genera muchas realizaciones aleatorias de cada día real del mercado reefer (junio–octubre de 2026), **sin alterar el orden cronológico de los días**. Así el agente DRL entrena con decenas de miles de episodios distintos y no puede memorizar los valores de un día concreto.

Cada episodio recorre días reales consecutivos (por ejemplo 15 → 16 → 17). Lo que cambia entre episodios es cómo se ve cada día: cuántas cargas hay, cuánto paga cada una y el nivel general de precios. La tabla resume qué se mantiene fijo y qué se sortea.

| Elemento del mercado | Cómo se obtiene en cada realización | De dónde sale la variación |
| --- | --- | --- |
| Orden de los días | Real y cronológico | No varía |
| Número de cargas (load availability) | Binomial negativa con media igual a las cargas esperadas ese día (media de los días vecinos), en vez de `count_loads` del día real | Dispersión observada en los datos |
| Precios de la lane | Cuantiles reales de ese día y esa lane (los niveles que puede cotizar el agente) | Distribución real de precios del día |
| Mejor competidor | Percentil sorteado con la competencia del hub de origen (Paso 4b, punto 4) | L/T del hub: menos camiones por carga, menos competencia |
| Load to truck ratio | Recalculado por hub de origen con las cargas sorteadas y los camiones del día real | Sigue a las cargas |
| Nivel nacional de precios | Corrimiento δ, fijo durante el episodio | Variación diaria real del mercado |

Toda la aleatoriedad sale de los propios datos. Por eso cualquier número que vea el agente se puede rastrear a un día y una lane reales.

## Paso 1. Alcance de los datos y partición temporal

Se usan solo cargas **REEFER** de hub a hub: **194 hubs y 37,442 lanes**. Son los 194 × 194 = 37,636 pares posibles menos los 194 que salen y llegan al mismo hub.

La ventana es **junio–octubre de 2026**, partida en el tiempo:

| Conjunto | Fechas | Días | Uso |
| --- | --- | --- | --- |
| Entrenamiento | 1 jun – 30 sep 2026 | 122 | Generar los episodios con los que aprende el agente |
| Prueba | 1 oct – 31 oct 2026 | 31 | Evaluar al agente y a los baselines en días que nunca vio |

**Por qué esta ventana.** Es el tramo más estable de la serie nacional. Los meses anteriores muestran una subida sostenida de ~2.3 a ~3.6 $/milla, que corresponde a otro régimen de mercado. Incluirlos enseñaría al agente condiciones que ya no existen.

**Por qué partir por fechas y no al azar.** La política se usará en días futuros, con lo aprendido del pasado. Probarla en un mes posterior al entrenamiento reproduce esa situación y evita que la evaluación use, sin querer, información del mismo período con el que se entrenó.

## Paso 2. Leer los archivos diarios de la carpeta

Los datos se quedan como están: un CSV por día en la carpeta `datos_finales_jul_oct_2026`, sin unirlos en un solo archivo. Al iniciar el entrenamiento, el generador lee cada CSV una sola vez, se queda con las columnas que necesita y las guarda en memoria como un arreglo por día. Después, la pregunta del entorno "¿cómo está la lane X el día d?" se responde desde memoria, sin volver a abrir archivos.

- **Por qué no unir los archivos.** No hace falta: cada CSV ya trae todas las lanes de su día, y el entorno consulta un día a la vez.
- **Por qué cargar todo al inicio y no leer en cada paso.** Abrir un CSV de ~5 MB en cada paso del entorno haría el entrenamiento muy lento. En memoria, cada día ocupa ~1.5 MB (37,442 lanes × 10 columnas numéricas en float32), así que los ~150 días caben en ~250 MB.
- **Mismo orden de lanes todos los días.** Con el primer archivo se arma una sola vez el orden de las lanes (origen, destino), y cada CSV se reordena con ese orden. Así la lane 20→47 queda en la misma posición todos los días.
- **Controles al cargar.** Se verifica que no falte ningún día entre el primero y el último, y que ningún día tenga lanes faltantes (que saldrían como valores vacíos al reordenar).

```python
from pathlib import Path
import numpy as np, pandas as pd

CARPETA = Path('datos_finales_jul_oct_2026')
COLS = ['count_loads', 'min', 'q1', 'q2', 'q3', 'p90', 'max',
        'mean', 'load_to_truck_ratio', 'distance']

def leer(f):
    df = pd.read_csv(f, usecols=['start_date', 'origin', 'destination'] + COLS,
                     dtype={'origin': str, 'destination': str})
    dia = pd.Timestamp(df['start_date'].iat[0]).date()
    df = df[df.origin != df.destination].set_index(['origin', 'destination'])
    return dia, df

archivos = sorted(CARPETA.glob('*.csv'))       # los nombres ordenan por fecha
_, primero = leer(archivos[0])
LANES = primero.index.sort_values()             # 37,442 lanes, orden fijo

datos = {}
for f in archivos:
    dia, df = leer(f)
    arr = df.reindex(LANES)[COLS].to_numpy(np.float32)
    assert not np.isnan(arr[:, 0]).any(), f'lanes faltantes en {dia}'
    datos[dia] = arr                            # datos[dia][posicion_lane, columna]
```

Cada CSV trae estas columnas:

| Columna del archivo | Nivel | Contenido | Uso en el generador |
| --- | --- | --- | --- |
| `count_loads` | Lane | Cargas publicadas; es la misma variable que load availability | Estimación de la media esperada m y de r |
| `min`, `q1`, `q2`, `q3`, `p90`, `max` | Lane | Cuantiles del rate ($/milla) de las cargas del día | Sorteo del precio de cada oferta |
| `mean` | Lane | Rate promedio del día | Cálculo del factor nacional (Paso 3) |
| `load_to_truck_ratio` | Hub de origen | Cargas por camión en el hub de origen; es el mismo valor en todas las lanes que salen de ese hub | Recálculo del L/T de cada realización |
| `trucks` | Lane | `count_loads` / `load_to_truck_ratio`: reparte los camiones del hub entre sus lanes; no es un dato independiente | No se usa directamente |
| `distance` | Lane | Millas de la lane | Recompensa del MDP |

Antes de usar los archivos se aplican tres reglas de limpieza:

1. **Días sin cargas.** Algunas lanes tienen `count_loads = 0` pero igual traen valores de rate. Esos valores no vienen de transacciones de ese día; parecen imputados por el proveedor de datos. Por eso **los cuantiles de un lane-día con 0 cargas no se usan para sortear precios**. Si en una realización esa lane recibe ofertas (Paso 4b), los precios salen de los cuantiles del día más cercano en que la lane sí tuvo cargas. La fracción de lanes sin cargas cambia mucho según el día: 8% el lunes 1 de junio y 36% el sábado 31 de octubre.
2. **L/T igual a 0 en un hub con cargas.** El 1 de junio, un hub de origen tiene `load_to_truck_ratio = 0` aunque sus lanes suman 746 cargas; por eso su columna `trucks` queda vacía. Un L/T de 0 con cargas no es posible, así que se trata como dato faltante y se reemplaza por el L/T de ese mismo hub en el día válido más cercano. Hay que contar cuántos casos así hay en todos los archivos y reportarlo.
3. **Lanes de un hub a sí mismo.** Cada archivo trae 37,636 filas, incluidas las 194 lanes de un hub a sí mismo. Estas lanes sí tienen cargas reales (son cortas, con mediana de 43 millas y 98 cargas el 1 de junio), pero según el Paso 1 no se le ofrecen al agente. También quedan fuera del factor nacional (Paso 3), de la estimación de r y del recálculo del L/T (Paso 4), para que su volumen alto no domine esos cálculos.

## Paso 3. Factor nacional del mercado y su variación

Se calcula un índice diario del nivel general de precios, F_t, y cuánto se mueve de un día a otro. Esa variación define el tamaño del corrimiento del Paso 4.

**Cálculo.** F_t es el promedio del logaritmo del rate de las 37,442 lanes (sin las de un hub a sí mismo), ponderado por el número de cargas de cada lane ese día:

$$
F_t = \frac{\sum_l \text{count\_loads}_{t,l} \cdot \log(\text{mean}_{t,l})}{\sum_l \text{count\_loads}_{t,l}}
$$

Con los días de entrenamiento se calcula el cambio diario y su desviación estándar:

$$
\Delta F_t = F_t - F_{t-1}, \qquad s = \operatorname{desv.est.}(\Delta F_t)
$$

**Por qué un factor común.** Las 37,442 lanes no se mueven cada una por su lado: casi todas suben y bajan juntas. La justificación tiene tres partes:

1. **Que existe un factor común (Stock y Watson, 2016).** La premisa de los modelos de factores dinámicos es que la dinámica común de muchas series de tiempo proviene de unos pocos factores no observados. Esto justifica que exista una "marea" nacional; ellos la estiman con componentes principales.
2. **Que un promedio de corte transversal lo estima bien (Pesaran, 2006).** Pesaran propone *"filter the individual-specific regressors by means of cross-section averages"*: cuando el número de series crece, el promedio de corte transversal captura el efecto de los factores comunes no observados. Con 37,442 lanes, el número de series es muy grande, que es justo el caso que cubre su teoría.
3. **Que funciona aunque el factor tenga cambios de nivel (Kapetanios, Pesaran y Yamagata, 2011).** Extienden el resultado al caso en que los factores comunes siguen procesos de raíz unitaria, es decir, no estacionarios. En sus simulaciones, *"the cross-sectional-average-based method is robust to a wide variety of data generation processes"*.

**Por qué en logaritmo.** Los cambios quedan en términos proporcionales. Un +1% significa lo mismo en una lane de 2 $/milla que en una de 5 $/milla.

**Por qué ponderado por cargas.** Es una decisión de diseño: las lanes con más actividad reflejan mejor el mercado, y las lanes con 0 cargas, cuyo rate es imputado, quedan con peso cero. El marco de Pesaran (2006) admite promedios ponderados siempre que ningún peso individual domine; este punto hay que verificarlo en la definición de pesos del artículo. Por eso se comprueba que ninguna lane concentre una fracción grande de las cargas de un día.

## Paso 4. Generador de realizaciones diarias

El generador hace dos cosas: al inicio de cada episodio sortea dónde y cuándo empieza y el nivel general de precios; después, cada día real del episodio, sortea las ofertas de carga.

### 4a. Al iniciar cada episodio

1. **Hub inicial:** uno de los 194 hubs, al azar.
2. **Día inicial:** un día de entrenamiento al azar, dejando espacio para que quepan los H días del episodio antes del 30 de septiembre. H es el horizonte definido en el MDP.
3. **Corrimiento de nivel:** δ ~ Normal(0, s²), con s del Paso 3. Se mantiene **fijo durante todo el episodio**.

**Por qué un corrimiento.** Todos los episodios recorren la misma historia de fondo. Sin corrimiento, el agente vería siempre el mismo nivel de precios para cada día y podría memorizarlo. Esto es aleatorización de dominio, introducida por Tobin et al. (2017): con suficiente variación en el simulador, el mundo real aparece como una variación más. Vuong et al. (2019) la describen como *"training a policy on a distribution of environmental conditions"*, de modo que la política generaliza si esos entornos son lo bastante diversos. Peng et al. (2018) muestran que *"dynamics randomization using low fidelity simulations can also be an effective approach"*, lo que respalda usar un simulador simple como este. Chen et al. (2022) dan el marco teórico: modelan el simulador como *"a set of MDPs with tunable parameters"*, que es exactamente lo que hace δ con el MDP de aceptación de cargas.

**Por qué fijo durante el episodio.** Si δ se sorteara de nuevo cada día, entre el día 15 y el 16 podría aparecer un salto de nivel que nunca ocurrió. Fijarlo conserva la consecutividad natural de los días reales.

**Por qué de tamaño s.** El corrimiento tiene la magnitud de los movimientos diarios reales del mercado. Así agrega variedad sin inventar niveles de precio fuera de lo observado.

### 4b. Cada día del episodio, en orden real

Para cada lane que sale del hub donde está el camión, el día real d se convierte en una realización con tres cálculos.

**1. Número de ofertas: binomial negativa.**

$$
n^*_{d,l} \sim \text{BinomialNegativa}(\mu = m_{d,l},\ r), \qquad \operatorname{Var}(n^*) = \mu + \frac{\mu^2}{r}
$$

La media m es el número de cargas que se esperaba en esa lane ese día. Se calcula con los 3 días anteriores y los 3 posteriores, sin el propio día, ajustados por día de la semana:

$$
m_{d,l} = f_{w(d)} \cdot \frac{1}{6} \sum_{k=-3,\,k \neq 0}^{3} \frac{\text{count\_loads}_{d+k,\,l}}{f_{w(d+k)}}
$$

f_w es el factor nacional del día de la semana w: el promedio de cargas de ese día de la semana dividido por el promedio general, calculado con los días de entrenamiento. Dividir por f quita el efecto del día de la semana a los vecinos, y multiplicar por f_w(d) lo vuelve a poner para el día d. En los 3 primeros y los 3 últimos días de la ventana se usan los vecinos disponibles.

- **Por qué binomial negativa.** En validaciones previas con estos datos, los conteos de cargas mostraron más variabilidad que la que permite Poisson, que obliga a que la varianza sea igual a la media. La binomial negativa agrega el parámetro r, que controla esa dispersión extra. Es la elección estándar en pronóstico de conteos: DeepAR (Salinas et al., 2020) incorpora *"a negative Binomial likelihood for count data"*.
- **Por qué tiene que haber una m.** La binomial negativa necesita un centro alrededor del cual sortear. Usar el conteo observado sumaría ruido sobre ruido, porque ese conteo ya trae el azar de ese día. Usar el promedio de la lane en todo el período mezclaría el lunes con el sábado y julio con agosto, y esas diferencias sistemáticas se contarían como ruido. m es la mejor estimación de cuántas cargas se esperaban en esa lane ese día. Con m, las realizaciones representan el mercado alrededor de ese día, y el conteo observado es uno de los valores posibles, no el centro. Además, r se estima con esta misma m, así que el sorteo reproduce exactamente la variabilidad que se midió.
- **Por qué 3 días antes y 3 después.** Es la media móvil de orden 7, la estándar para datos diarios con patrón semanal. En la descomposición clásica de series de tiempo, la tendencia se estima con una media móvil cuyo orden es el período estacional; Hyndman y Athanasopoulos (*Forecasting: Principles and Practice*, sección 6.3) dan como ejemplo m = 7 para datos diarios con patrón semanal, y una media móvil de orden 2k + 1 promedia k observaciones anteriores y k posteriores (sección 6.2). Con orden 7 queda k = 3. Una ventana de 7 días cubre cada día de la semana una vez, así que el patrón semanal se compensa dentro del promedio.
- **Por qué m no incluye al propio día.** Es la única diferencia con la media móvil clásica. Si m incluyera al día, se movería con el conteo de ese día y la variabilidad medida saldría más pequeña que la real. Al quitar el día ya no están los 7 días de la semana, y el ajuste por día de la semana (los factores f) corrige ese desbalance.
- **Lanes con m < 1.** Se sortea con Poisson(m). Con medias tan bajas casi todo es ruido de conteos pequeños, r no se estima bien, y el resultado son sobre todo ceros.
- **Estimación de r: máxima verosimilitud.** Con m fija como media de cada lane-día, se busca el r que hace más probables los conteos observados en los días de entrenamiento. Se usan los lane-días con m ≥ 1 y sin las lanes de un hub a sí mismo. Máxima verosimilitud usa toda la forma de la distribución (cuántos ceros, cuántos valores altos), no solo la media y la varianza. Saha y Paul (2005) comparan estimadores del parámetro de dispersión de la binomial negativa, y su versión de máxima verosimilitud con corrección de sesgo muestra *"superior bias and efficiency properties in most instances"*. Esa corrección importa en muestras pequeñas; con millones de lane-días basta la máxima verosimilitud estándar.
- **r por grupo de volumen.** Los lane-días se ordenan según m y se parten en 10 grupos con el mismo número de lane-días, y se estima un r en cada uno. Si los 10 valores se parecen, se usa un r global estimado con todos los datos. Si cambian con el volumen, cada lane usa el r del grupo de su volumen promedio. No se estima por lane: ~122 días no alcanzan para un r estable.
- **Limitación que se declara.** m es una estimación con 6 días, no la media verdadera del mercado. Su error agrega algo de variabilidad a la medida, así que r puede salir algo más pequeño (con más ruido) que el real. La prueba de dispersión del Paso 6 verifica que ese efecto sea pequeño.

```python
import numpy as np
from scipy.special import gammaln
from scipy.optimize import minimize_scalar

def menos_logverosimilitud(log_r, y, m):
    r = np.exp(log_r)
    return -(gammaln(y + r) - gammaln(r) - gammaln(y + 1)
             + r * np.log(r / (r + m)) + y * np.log(m / (r + m))).sum()

# y = count_loads observado, m = media de los 6 vecinos ajustada; solo m >= 1
# se corre una vez por grupo de volumen (o una vez con todo, si r es global)
res = minimize_scalar(menos_logverosimilitud, bounds=(-5, 10),
                      args=(y, m), method='bounded')
r = np.exp(res.x)
```

**2. Precio de cada oferta: sorteo dentro de los cuantiles del día.**

Cada oferta recibe un número u al azar entre 0 y 1. Ese número se convierte en precio con la distribución de ese día y esa lane, interpolando en línea recta entre estos puntos:

| Percentil | 0 | 25 | 50 | 75 | 90 | 100 |
| --- | --- | --- | --- | --- | --- | --- |
| Valor del día | min | q1 | q2 | q3 | p90 | max |

El precio final es ese valor multiplicado por e^δ.

- **Por qué la inversa.** Es el método de la inversa (Devroye, 1986, cap. II, sección 2): si U es uniforme en [0, 1] y F es una función de distribución, F⁻¹(U) produce valores con distribución F. Aquí F es la distribución de precios de ese día y esa lane, así que el 25% de las ofertas queda por debajo de q1, la mitad por debajo de q2, y así sucesivamente.
- **Por qué interpolación lineal entre puntos.** Es el tratamiento estándar de distribuciones empíricas en simulación. Law y Kelton (*Simulation Modeling and Analysis*, cap. 6) construyen una distribución continua lineal por tramos que interpola entre los datos, con el mínimo y el máximo como límites. Kaczynski et al. (2012) lo confirman como el enfoque estándar: *"construct a piecewise-linear cdf that fills in the gaps between the data values"*.
- **Adaptación que se declara.** Law y Kelton y Kaczynski et al. interpolan entre todas las observaciones de una muestra. Aquí se interpola entre 6 cuantiles resumen, porque los archivos no traen el precio de cada carga individual. Es la misma técnica con menos puntos de apoyo, y su parte más débil son las colas (entre p90 y max).
- **Por qué no ruido inventado.** Los precios salen de la distribución real del día, así que cualquier precio sorteado se puede rastrear a un día y una lane reales.

**3. Load to truck ratio: recalculado por hub de origen con las cargas sorteadas.**

En los datos, el L/T se mide por hub de origen: es el mismo valor para todas las lanes que salen de un hub. Por eso se recalcula a ese nivel. Los camiones del hub en el día real se mantienen, y el ratio se ajusta en la misma proporción en que cambian las cargas del hub. Las sumas van sobre las lanes que salen del hub de origen o (sin contar la lane del hub a sí mismo):

$$
LT^*_{d,o} = LT_{d,o} \times \frac{\sum_{l \in o} n^*_{d,l}}{\sum_{l \in o} \text{count\_loads}_{d,l}}
$$

Ejemplo: un hub tuvo 1,200 cargas en sus lanes de salida y L/T = 30, es decir, 40 camiones. Si la realización sortea 1,320 cargas en esas lanes, L/T* = 30 × 1,320 / 1,200 = 33, que equivale a 1,320 cargas para los mismos 40 camiones. Si el hub no tuvo cargas ese día, se deja el L/T real.

- **Por qué así.** Si el L/T quedara fijo mientras las cargas cambian, el agente vería señales contradictorias, por ejemplo muchas cargas con un mercado igual de holgado. Recalcularlo mantiene la relación real entre cargas y camiones sin agregar parámetros: toda la aleatoriedad del lado de la oferta sigue viniendo de la binomial negativa.
- **Por qué no se usa la columna `trucks`.** Esa columna es `count_loads` / `load_to_truck_ratio` lane por lane, es decir, un reparto del L/T del hub, no un conteo independiente. La fórmula de proporción da el mismo resultado sin depender de ella y funciona aunque `trucks` esté vacía.

**4. Competencia: el mejor competidor de cada lane.**

El agente no acepta un precio dado: cotiza uno de 9 niveles comunes a todas las lanes, en escala de percentiles (u = 0.1, 0.2, …, 0.9, es decir p10 … p90). El precio de cada nivel sale de la misma interpolación del punto 2, F⁻¹(u) · e^δ, con los cuantiles de ese día y esa lane. El agente gana la carga si su percentil queda por debajo del de su mejor competidor, u_comp:

$$
u_{comp} = 1 - (1 - U)^{1/m_c}, \qquad P(\text{ganar con el nivel } u) = (1 - u)^{m_c}
$$

Cada carga de la lane tiene su propio U y, por tanto, su propio u_comp. El U de la primera carga es el mismo número uniforme del punto 2; las demás cargas (hasta C = 4, el máximo de cotizaciones por visita) usan uniformes independientes. m_c es el número de competidores efectivos del hub de origen:

$$
m_c = \text{clip}\left(1 + \kappa \, \frac{LTR_{ref}}{LTR_{d,o}}, \; 1, \; M_{max}\right)
$$

LTR_ref es la mediana del L/T por hub y día en los días de entrenamiento, y κ = 1.

- **Por qué así.** El mejor de m_c competidores que cotizan al azar dentro de la distribución del día tiene percentil Beta(1, m_c), y la inversa de esa distribución es la fórmula de u_comp. Un hub con L/T bajo (pocas cargas por camión) tiene más competencia, y un hub tensionado (L/T alto) tiene menos.
- **Por qué con el mismo U (primera carga).** Con m_c = 1 el competidor cotiza exactamente la tarifa sorteada del punto 2, así que el mercado sin competencia adicional es el de siempre.
- **Por qué un competidor independiente por carga.** Cada carga publicada es una oportunidad distinta, con su propio mejor competidor. Si el agente pierde una carga y vuelve a cotizar la misma lane, enfrenta otra carga y otro competidor, no el mismo rival con un precio nuevo; así un rechazo no revela nada sobre la siguiente carga. El número de cargas cotizables de una lane es min(cargas sorteadas, C).
- **Por qué ningún nivel gana con seguridad.** Se quitó el nivel u = 0 (el mínimo del mercado). Cotizar al mínimo no garantiza ganar: un competidor puede quedar por debajo, y los datos no permiten asegurar lo contrario. Con el nivel más bajo en p10, `gana` es simplemente u_nivel < u_comp, sin caso especial.
- **Supuesto que se declara.** No hay datos de cargas ganadas y perdidas, así que la aceptación es un modelo: κ fija cuánto pesa el L/T. El agente se entrena con κ = 1, y la tesis reporta la sensibilidad a κ ∈ {0.5, 1, 2} evaluando ese mismo modelo, sin reentrenarlo.
- **Mismo competidor para todos los métodos.** u_comp se sortea con la semilla del episodio, así que el DRL y los baselines enfrentan el mismo competidor en cada lane y cada día (números aleatorios comunes).

### Pseudocódigo del generador

```python
def reset():
    hub   = random.choice(hubs)
    dia   = random.choice(dias_entrenamiento[:-H])
    delta = np.random.normal(0, s)                  # fijo todo el episodio
    return hub, dia, delta

def realizacion(dia, hub, delta):
    ofertas, n_hub, c_hub = [], 0, 0
    for lane in lanes_desde(hub):                   # sin la lane hub -> mismo hub
        mu = m[dia, lane]                           # media esperada (6 vecinos)
        if mu < 1:
            n = np.random.poisson(mu)
        else:
            k = r_de(lane)                          # r global o del grupo de volumen
            n = np.random.negative_binomial(k, k / (k + mu))
        d_p = dia_con_cargas_mas_cercano(dia, lane) # cuantiles reales, no imputados
        U = np.random.rand(C)                       # C = 4: un uniforme por carga cotizable
        tarifa = interp_cuantiles(cuantiles[d_p, lane], U[0]) * np.exp(delta)
        u_comp = 1 - (1 - U) ** (1 / m_comp(LT[dia, hub]))      # mejor competidor de cada carga (punto 4)
        precios_niveles = interp_cuantiles(cuantiles[d_p, lane], U_NIVELES) * np.exp(delta)   # 9 niveles, p10 ... p90
        ofertas += [(lane, n, tarifa, precios_niveles, u_comp)]
        n_hub, c_hub = n_hub + n, c_hub + count_loads[dia, lane]
    lt = LT[dia, hub] * n_hub / c_hub if c_hub > 0 else LT[dia, hub]
    return ofertas, lt
```

## Paso 5. Integración con el MDP

El generador es la parte exógena del MDP: describe cómo está el mercado cada día, y el agente no lo controla. Ubicación del camión, decisiones y recompensas siguen las reglas del MDP de ruteo con cotización.

Cada episodio sigue este ciclo, que se repite cada día real hasta completar H días:

1. **Inicio del episodio:** hub al azar, día real al azar y δ fijo.
2. **Realización del día d:** cargas con binomial negativa, precios por nivel desde los cuantiles, mejor competidor, L/T recalculado.
3. **Decisión del agente:** un destino y un nivel de precio para la carga de esa lane, o reposicionar vacío.
4. **Resultado:** si el nivel queda por debajo del competidor de esa carga, gana la carga, cobra ese precio y viaja. Si pierde, esa carga se pierde: el agente se queda en el hub, pierde 15 minutos, no ve un tablero nuevo y vuelve a decidir. Puede volver a cotizar la misma lane con otra de sus cargas (otro competidor independiente) hasta agotarlas, con a lo sumo 4 cotizaciones rechazadas por visita (`COTIZACIONES_MAX_POR_NODO`).
5. **Avance:** al llegar al nuevo hub se toma la realización del día real en que llega y se vuelve al punto 3.

El orden de los días siempre es el real; lo único que se sortea en cada vuelta es la realización del día.

- **`reset()`** ejecuta el Paso 4a: hub inicial, día inicial y δ.
- **`step(destino, nivel)`** resuelve la cotización contra u_comp, calcula la recompensa (precio · distancia − costo si gana, −costo si viaja vacío) y avanza el tiempo y, si viaja, la ubicación. Al llegar a un hub genera la realización del día real de llegada (Paso 4b).
- **Observación del agente:** las cargas sorteadas en las lanes que salen de su hub, los precios de cada nivel, la mediana de la lane, el número de competidores efectivos m_c y el L/T* de ese hub, más información de los hubs destino. El agente nunca ve u_comp.
- **La fecha no forma parte de la observación.** Todos los episodios recorren la misma historia de fondo. Si el agente viera "día 15", podría aprender el calendario ("en julio suben los precios") en vez de aprender a leer el mercado. Sin la fecha, decide solo con lo que vería un despachador. El día de la semana sí se puede incluir: tiene solo 7 valores, se repite igual en octubre y refleja un patrón real del mercado (menos cargas el fin de semana).

**Ejemplo.** El episodio 1,050 sortea hub 20, día real 15 y δ = +1.5%.

1. **Día real 15, hub 20.** En la lane 20→47 salen 11 cargas; los niveles de precio salen de los cuantiles del día 15, multiplicados por e^0.015. En la lane 20→63 no hay cargas en esta realización. El agente cotiza 20→47 en el nivel u = 0.6. El competidor de la primera carga está en u_comp = 0.45, así que pierde: se queda en el hub 20, pierde 15 minutos y cotiza de nuevo, en la misma lane (con otra de sus 11 cargas y otro competidor) o en otra. Gana la segunda cotización y viaja.
2. **Día real 16, hub de llegada.** Los precios salen de los cuantiles del día 16, con el mismo δ = +1.5%. El agente decide otra vez.
3. **Días reales 17, 18, …** continúan igual hasta completar el horizonte H.

El episodio 31,077 también arranca en el hub 20 y el día real 15, pero sortea δ = −0.8%. En la lane 20→47 salen ahora 14 cargas, con otros precios y otro competidor. La secuencia de días es la misma; los números que ve el agente, no.

## Paso 6. Validar el generador antes de entrenar

Antes de entrenar se comprueba que las realizaciones reproducen el mercado real. Si el generador está mal calibrado, el agente aprende un mercado falso. Estas pruebas toman minutos y evitan entrenar 50,000 episodios sobre una base equivocada.

Para una muestra de días y lanes de entrenamiento (con lanes de volumen alto, medio y bajo), se generan miles de realizaciones con δ = 0 y se revisa:

| Prueba | Qué se compara | Se aprueba si |
| --- | --- | --- |
| Precios | Cuantiles de los precios sorteados frente a min, q1, q2, q3, p90 y max del archivo | Coinciden en cada punto |
| Dispersión de cargas | Razón varianza / media de n* frente a la observada, en cada grupo de volumen | Son similares. Si la simulada es mucho mayor, r es demasiado pequeño; si es mucho menor, r es demasiado grande |
| Días sin cargas | Fracción de lane-días con 0 ofertas frente a la fracción real con `count_loads = 0` | Coinciden |
| Coherencia del L/T | Camiones implícitos del hub de origen, Σn* / L/T*, frente a los del día real | Son iguales en todas las realizaciones |
| Nivel de precios | Distribución de δ frente a la de ΔF_t real | Tienen la misma escala |

La tolerancia de cada prueba se fija antes de correrla y se reporta en la tesis junto con los resultados.

**Dónde está hecha.** `validar_generador_datos.py` corre estas pruebas con las 37,442 lanes y deja el informe en `validacion_generador_datos.md` (tablas y figuras en `results/market_validation/validacion_metodologia/`). Cubre las cargas (Kolmogorov-Smirnov por lane, con jun–sep y octubre por separado, más errores de media, desviación y fracción de días sin cargas), los precios (media sorteada frente a media real) y la normalidad de δ. **No cubre** la coherencia del L/T ni la competencia: u_comp y κ no tienen datos reales con los que compararse, y su efecto se evalúa con la sensibilidad a κ de `validate_pricing.py`.

## Referencias

- Chen, X., Hu, J., Jin, C., Li, L. y Wang, L. (2022). *Understanding Domain Randomization for Sim-to-real Transfer*. ICLR. [arXiv:2110.03239](https://arxiv.org/pdf/2110.03239)
- Devroye, L. (1986). *Non-Uniform Random Variate Generation*. Springer-Verlag. Cap. II, sección 2: "The inversion method", p. 27. [Edición web](https://www.numerical.recipes/devroye/)
- Hyndman, R. J. y Athanasopoulos, G. *Forecasting: Principles and Practice*. OTexts. Secciones 6.2 (Moving averages) y 6.3 (Classical decomposition). [6.2](https://f0nzie.github.io/hyndman-bookdown-rsuite/moving-averages.html), [6.3](https://f0nzie.github.io/hyndman-bookdown-rsuite/classical-decomposition.html)
- Kaczynski, W., Leemis, L., Loehr, N. y McQueston, J. (2012). *Nonparametric Random Variate Generation Using a Piecewise-Linear Cumulative Distribution Function*. Communications in Statistics – Simulation and Computation. [PDF](https://www.math.wm.edu/~leemis/2012commst1.pdf)
- Kapetanios, G., Pesaran, M. H. y Yamagata, T. (2011). *Panels with non-stationary multifactor error structures*. Journal of Econometrics, 160(2), 326–348. [Resumen](https://www.econ.cam.ac.uk/research/pub-abstracts?pub=380)
- Law, A. M. y Kelton, W. D. *Simulation Modeling and Analysis*. McGraw-Hill, cap. 6 (distribuciones empíricas). [Descripción del método](https://sim-tools.github.io/sim-tools/01_sampling/02_empirical_distributions.html). Pendiente: confirmar la edición y la sección exacta en el libro.
- Peng, X. B., Andrychowicz, M., Zaremba, W. y Abbeel, P. (2018). *Sim-to-Real Transfer of Robotic Control with Dynamics Randomization*. ICRA. [arXiv:1710.06537](https://huggingface.co/papers/1710.06537.md)
- Pesaran, M. H. (2006). *Estimation and Inference in Large Heterogeneous Panels with a Multifactor Error Structure*. Econometrica, 74(4), 967–1012. [doi:10.1111/j.1468-0262.2006.00692.x](https://www.econometricsociety.org/publications/econometrica/browse/2006/07/01/estimation-and-inference-large-heterogeneous-panels-multifactor)
- Saha, K. y Paul, S. (2005). *Bias-Corrected Maximum Likelihood Estimator of the Negative Binomial Dispersion Parameter*. Biometrics, 61(1), 179–185. [doi:10.1111/j.0006-341X.2005.030833.x](https://peeref.com/works/70090559)
- Salinas, D., Flunkert, V., Gasthaus, J. y Januschowski, T. (2020). *DeepAR: Probabilistic forecasting with autoregressive recurrent networks*. International Journal of Forecasting, 36(3), 1181–1191. [arXiv:1704.04110](https://ar5iv.labs.arxiv.org/html/1704.04110)
- Stock, J. H. y Watson, M. W. (2016). *Dynamic Factor Models, Factor-Augmented Vector Autoregressions, and Structural Vector Autoregressions in Macroeconomics*. Handbook of Macroeconomics, Vol. 2, cap. 8. [PDF](https://moodle.novasbe.pt/pluginfile.php/671971/mod_resource/content/0/Stock_Watson_HOM_Vol2.pdf)
- Tobin, J., Fong, R., Ray, A., Schneider, J., Zaremba, W. y Abbeel, P. (2017). *Domain Randomization for Transferring Deep Neural Networks from Simulation to the Real World*. IROS. [doi:10.1109/IROS.2017.8202133](https://synapsesocial.com/papers/W2605102758)
- Vuong, Q., Vikram, S., Su, H., Gao, S. y Christensen, H. I. (2019). *How to pick the domain randomization parameters for sim-to-real transfer of reinforcement learning policies?* arXiv. [Resumen](https://aitopics.org/doc/arxivorg:F57C1580)
