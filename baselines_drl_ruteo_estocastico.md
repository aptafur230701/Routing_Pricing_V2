# Baselines para el DRL de ruteo estocástico

## Resumen

Compararía el DRL contra dos cotas superiores y tres políticas, todas evaluadas sobre los mismos mundos simulados. La pieza central es UB-DP, una programación dinámica sobre (nodo, hora, cargas restantes) que relaja la regla de no revisitar. Con el proceso exógeno nuevo es barata de calcular, da un gap de optimalidad casi certificado y su política, B3, es el rival más fuerte.

| Id | Baseline | Clase (Powell) | Usa información futura | Rol en la tesis |
| --- | --- | --- | --- | --- |
| UB-PI | Ruta óptima con el mundo completo conocido | Relajación de información | Sí | Techo laxo; mide el valor de la información perfecta |
| UB-DP | DP en (nodo, hora, K) permitiendo revisitar | Relajación de restricción | No | Techo ajustado; gap de optimalidad |
| B1 | Greedy miope | PFA | No | Benchmark miope estándar |
| B2 | Reoptimización determinista con valores esperados (solo K = 5) | CFA / lookahead determinista | No | Lo que haría un planificador con pronóstico puntual |
| B3 | Política de la DP relajada | VFA (tabla) | No | ADP clásica; el rival más fuerte esperado |

B1, B2 y B3 son no anticipativas: deciden con lo que ve el camión en el nodo actual y con los parámetros del proceso, nunca con la realización del mundo. Las dos cotas no son políticas implementables y se reportan como techos.

## Qué problema es y por qué eso fija los baselines

Tu problema es un ruteo estocástico de un solo vehículo donde la información de cada nodo se revela al llegar. En la literatura cae en la intersección de tres familias, y cada una aporta un tipo de baseline.

- **Canadian Traveller Problem estocástico.** El estado de los arcos que salen de un nodo se conoce solo al llegar a él. Es tu máscara de lanes. La versión con arcos independientes es #P-difícil ([Canadian traveller problem](https://en.wikipedia.org/wiki/Canadian_traveller_problem); Papadimitriou y Yannakakis, 1989). Esto justifica no buscar el óptimo exacto con el conjunto de visitados en el estado y comparar contra cotas.
- **Ruteo dinámico y estocástico (SDVRP).** Ulmer et al. (2020) y Hildebrandt, Thomas y Ulmer clasifican las soluciones en enfoques miopes de horizonte rodante, PFA, CFA, lookahead con escenarios y VFA ([survey](https://ar5iv.labs.arxiv.org/html/2103.00507)). Los trabajos de DRL en esta familia se comparan contra miope, aproximación de valor y escenarios; Joe y Lau (2020) usan exactamente AVI, MSA y miope ([ICAPS](https://ojs.aaai.org/index.php/ICAPS/article/view/6685)).
- **Gestión de flota truckload.** La línea de Powell modela las cargas como información exógena que llega después de decidir y resuelve con ADP sobre tablas de valor (Simao et al., 2009). Es el origen de B3.

Powell sostiene que toda política cae en una de cuatro clases (PFA, CFA, VFA, DLA) o en un híbrido, y que hay que considerarlas antes de elegir ([RLSO, 2022](https://castle.princeton.edu/RLSO/)). Los baselines cubren las tres que importan aquí: una regla directa (PFA), un lookahead determinista con pronóstico puntual (CFA/DLA determinista) y una función de valor (VFA). El lookahead estocástico, como un rollout, queda fuera porque B3 ya anticipa el futuro con la distribución completa del proceso y a un costo mucho menor por decisión.

Un detalle de tu entorno define qué es justo comparar. Al llegar al nodo i el camión ve `loads_row` y `rate_row`, así que la recompensa del arco que elige ahora es conocida. La incertidumbre está solo en lo que encontrará en los nodos siguientes. Un baseline es válido si usa la fila actual y los parámetros del proceso (mu, s_d, phi, tarifas históricas), y no lo es si mira filas de nodos o días futuros del mundo evaluado.

Hay una asimetría que conviene declarar. B1 a B3 usan los parámetros verdaderos del proceso (mu, s_d, phi y las tarifas históricas), es decir, conocen el modelo que genera los mundos. El DRL no los ve: recibe como prior `avail_expected_arr`, la media de cargas de los primeros 90 días, y aprende la distribución por experiencia. Es la comparación estándar entre métodos basados en modelo y sin modelo, y juega a favor de los baselines.

## Las dos cotas superiores

Reportaría el DRL contra un techo laxo y uno ajustado. El laxo dice cuánto vale saber el futuro; el ajustado dice cuánto le falta al DRL para el óptimo alcanzable sin saberlo.

### UB-PI: información perfecta

Para cada mundo de `sample_world`, resuelve la mejor ruta con todo el mundo conocido: qué cargas habrá en cada nodo y día. Es la relajación de no anticipatividad sin penalidad de Brown, Smith y Sun (2010), que acotan el valor óptimo relajando la regla de decidir solo con lo ya observado ([Operations Research](https://pubsonline.informs.org/doi/10.1287/opre.1090.0796)). Genera el mundo con `sample_world(semilla, 6, día de inicio)`: 77 horas son 5,5 días de mercado, así que 6 días cubren cualquier ruta. Ese mundo es exactamente el que ve `RoutingEnv` con la misma semilla, porque ambos sortean cada fila en función de (semilla, nodo, día). El label-setting oráculo que ya tenías sirve para resolverlo.

Es un techo laxo. En un problema donde la información llega al llegar, el clarividente evita siempre los nodos vacíos, y ninguna política real puede hacerlo. Un gap grande contra UB-PI no prueba que el DRL sea malo. La diferencia UB-PI menos UB-DP es el valor esperado de la información perfecta (EVPI) y responde una pregunta de negocio: cuánto valdría saber las cargas antes de moverse.

### UB-DP: programación dinámica sin la regla de no revisitar

El estado exacto incluye el conjunto de nodos visitados, y eso lo vuelve intratable. Si se permite revisitar nodos intermedios, el estado se reduce a (nodo i, hora t, arcos restantes k) y el día de la semana de salida. Con el proceso nuevo, las filas de cada llegada son independientes y dependen solo del nodo y del día de la semana, así que la ecuación de Bellman es exacta:

```latex
V(i,t,k) \;=\; \mathbb{E}_{\omega \sim P(\cdot \mid i,\, d(t))}\Big[\max_{j \in A(i,t,k,\omega)} \big\{ r_{ij}(\omega) + V(j,\, t+\tau_{ij},\, k-1) \big\}\Big]
```

Aquí ω es la pareja (loads_row, rate_row) que sortea `MarketProcess` en el nodo i, d(t) es el día de la semana de llegada, A son los destinos factibles (lane con carga, retorno a casa posible dentro de 77 horas, k mayor que cero) y r es la recompensa de la fórmula compartida. Al volver a casa el valor terminal es cero. La esperanza se estima con S sorteos por estado (S = 500).

Como el conjunto de políticas que nunca revisitan está contenido en el de las que sí pueden, V(casa, 0, K) es una cota superior del valor de la mejor política no anticipativa. Dos cuidados para que siga siendo cota:

- Discretiza el tiempo en franjas de 15 minutos redondeando los tiempos de viaje hacia abajo. Redondear hacia arriba eliminaría rutas factibles. Redondear hacia abajo puede adelantar una llegada al día anterior y cambiar su día de la semana; con franjas de 15 minutos el efecto es mínimo, pero en rigor la cota es aproximada y así debe declararse.
- Usa S = 500 sorteos por estado. El error de muestreo dentro del máximo sesga V hacia arriba, lo cual es conservador para una cota.

En tu código cada fila es fija dentro de un mundo para cada par (nodo, día), mientras que la DP sortea de nuevo en cada llegada. Eso no rompe la cota: una ruta que no revisita nunca pasa dos veces por el mismo nodo intermedio, así que para esas rutas ambos modelos dan exactamente la misma distribución, y la DP solo agrega rutas que el problema real no permite.

El tamaño es manejable. Con 35 nodos, unas 310 franjas de 15 minutos, 6 valores de k y 7 días de salida hay cerca de 460.000 estados. Los sorteos dependen solo del nodo y del día de la semana, así que se generan una vez por par (nodo, día) y se reutilizan en todos los estados. Con numpy vectorizado son minutos. Para el caso libre sin K, el límite de 77 horas acota la profundidad y basta quitar k del estado.

El gap DRL contra UB-DP es el número central de la evaluación: si el DRL queda a 3 % de UB-DP, está a lo sumo a cerca de 3 % del óptimo. Esa afirmación es mucho más fuerte que cualquier comparación contra heurísticas.

## Las políticas baseline

Cada política recibe lo mismo que el DRL en cada nodo: la máscara, la fila actual de cargas y tarifas, la hora, k restante, los visitados y los parámetros del proceso. Todas respetan las mismas reglas de máscara, incluida la excepción del primer movimiento desde casa.

### B1. Greedy miope (PFA)

Elige el arco con mayor recompensa inmediata r_ij entre las acciones válidas. Es el benchmark miope estándar en ruteo dinámico y el que usan Joe y Lau como referencia. No tiene parámetros y cuesta unas pocas líneas.

### B2. Reoptimización determinista con valores esperados (CFA)

En cada nodo, resuelve de forma exacta la mejor ruta restante reemplazando el futuro por su valor esperado y ejecuta solo el primer arco. Para un arco futuro (j, l):

```latex
\mathbb{E}[r_{jl}] \;=\; P(N_{jl} \ge 1)\,\bar{q}_{jl}\,\text{dist}_{jl} \;-\; c_{jl}, \qquad P(N_{jl} \ge 1) = 1 - \Big(\frac{\phi}{\phi + m_{jl}}\Big)^{\phi}
```

con m = mu_jl × s_d y q la tarifa media histórica de la lane. El arco actual usa la fila real observada. Con K = 5 la búsqueda exacta es pequeña y reutiliza tu label-setting; en el caso libre las rutas son más largas y reoptimizar en cada paso se vuelve caro, así que B2 se evalúa solo con K = 5. Es el horizonte rodante que la literatura de ruteo dinámico llama miope en información: replanifica con un pronóstico puntual e ignora la varianza. Su debilidad conocida es planear a través de nodos que en promedio lucen bien pero a menudo están vacíos, y el DRL debería ganarle justo ahí.

### B3. Política de la DP relajada (VFA de tabla)

Usa la tabla V de UB-DP como función de valor y decide en línea:

```latex
a = \arg\max_{j \in A,\; j \notin \text{visitados}} \; r_{ij} + V(j,\, t+\tau_{ij},\, k-1)
```

Es ADP de tabla en el sentido de Simao et al. y Powell: una función de valor calculada fuera de línea y una decisión de un paso en línea. La regla de no revisitar se aplica en la ejecución, así que V es optimista sobre nodos ya visitados; aun así espero que sea muy fuerte, porque la estructura del proceso es exactamente la que asume la DP. Si B3 le gana al DRL, es un hallazgo que la tesis debe reportar, no esconder: el argumento del DRL pasaría a ser la escala y la transferencia entre tamaños de grafo, no la calidad por instancia.

### El DRL

El DRL se evalúa con su política greedy: en cada nodo elige el argmax de la red, ejecutado paso a paso en `RoutingEnv`. No se usa beam search; con un ancho mayor que 1 compararía rutas usando cargas de nodos aún no visitados y dejaría de ser una política válida para este problema.

## Protocolo experimental

La comparación es pareada por mundo: todas las políticas corren sobre el mismo mundo, identificado por su `world_seed`, y la unidad de análisis es la diferencia de margen en ese mundo.

### Diseño

1. **Instancias.** Para cada tamaño N y cada nivel de K (5 y libre), genera M mundos con semillas derivadas de `MARKET_SEED_EVAL`, distintas de las de entrenamiento. Cada mundo trae asignados un nodo de inicio y un día de la semana en rotación, de modo que todos los nodos y los siete días quedan cubiertos sin cruzarlos todos contra todos.
2. **Tamaño de M.** Hoy `MARKET_M_EVAL_WORLDS = 100`. Corre con ese valor, mide la desviación estándar de las diferencias pareadas DRL menos baseline y sube M solo si la semiamplitud del intervalo del 95 % es mayor que la mitad de la diferencia que quieres detectar.
3. **Arnés único.** Evalúa todas las políticas paso a paso en `RoutingEnv(market_process=...)` con `reset(options={"start_node": s, "start_day_idx": d, "k_max": K, "world_seed": w})`. En cada nodo la política lee solo la máscara, `current_loads_row` y `current_rate_row`. El mismo `world_seed` da números aleatorios comunes y la no anticipatividad queda garantizada por construcción. `sample_world` se usa solo para UB-PI, y los sorteos de UB-DP usan `sample_arrivals` con un Generator propio.
4. **Semillas de entrenamiento.** Entrena el DRL con 3 semillas en el tamaño principal de la tesis y reporta la dispersión entre ellas; los demás tamaños pueden ir con una. Henderson et al. (2018) mostraron que una sola semilla puede invertir conclusiones en DRL.
5. **Replay real.** Mantén los días 90 a 119 como evaluación descriptiva, con la advertencia de que el proceso se ajustó con los 120 días y no es independiente del simulador. La curva de `_evaluate_greedy` que imprime el entrenamiento también es replay, así que no se reporta como resultado.

### Métricas por mundo

| Métrica | Unidad | Por qué |
| --- | --- | --- |
| Margen de la ruta | USD | Objetivo de negocio: suma de `lane_reward` de los arcos, no la recompensa del entorno (que trae bono, penalidad y escala) |
| Gap contra UB-DP | % | Distancia certificada al óptimo |
| Gap contra UB-PI | % | Valor de la información perfecta |
| Rutas con violación de tiempo | % de mundos | Factibilidad |
| Millas cargadas sobre millas totales | % | Eficiencia operativa; deadhead |
| Cargas por ruta y duración | conteo, horas | Forma de la solución |
| Tiempo por decisión | ms, CPU un hilo | Viabilidad en línea |
| Tiempo fuera de línea | horas | Entrenamiento del DRL, cálculo de la tabla V |

### Pruebas estadísticas

- Diferencia media pareada DRL menos cada baseline, con intervalo del 95 % por bootstrap sobre mundos (Agarwal et al., 2021).
- Prueba de Wilcoxon de rangos con signo, pareada y de una cola, con corrección de Holm por los tres baselines. Es lo que recomiendan Accorsi, Lodi y Vigo para ML en ruteo ([guías](https://ar5iv.arxiv.org/html/2109.13983)), y tu `stats_analysis.py` ya la usa.
- Tiempos medidos en las mismas condiciones para todos: CPU, un hilo, máquina sin otra carga.

## Lo que no recomiendo

- **Metaheurísticas sobre el mundo completo como rivales.** Un HGA-LNS o un label-setting que ven todo el mundo resuelven el problema clarividente. Su resultado es una estimación de UB-PI, no una política, y presentarlo como competidor confunde al lector.
- **Comparar con la recompensa de entrenamiento.** El bono de retorno y la penalidad de tiempo son shaping. La tabla de resultados va en margen en dólares, y la factibilidad se reporta aparte.

Una limitación del modelo que conviene declarar en la tesis. En modo `market_process`, las cargas y tarifas se sortean, pero la feature de camiones (f5) y el LTR del contexto se siguen leyendo del día histórico de inicio (`start_day_idx`, entre 0 y 89). Como el proceso condiciona solo en el día de la semana, esas señales no tienen relación con lo que el agente encuentra en los mundos simulados.

## Instrucciones para Claude Code

Este bloque está escrito para Claude Code, en el repositorio AM-PPO. Implementa el conjunto de baselines de este documento (UB-PI, UB-DP, B1, B2, B3) y el arnés de evaluación en mundos simulados, y borra los baselines anteriores. No implementes nada que este documento no pida.

### Antes de empezar

1. Lee este documento completo y `proceso_exogeno_cargas_tarifas.md`.
2. Lee `routing_env.py`, `market_process.py`, `problem_data.py` (`lane_reward`), `config.py`, `evaluation.py`, `Solvers.py`, `stats_analysis.py`, `excel_report.py`, `main.py`, `run_k_comparison.py`, `inference.py` y `am_agent.py`.
3. Confirma cómo se evalúa hoy el DRL. El beam search ya fue eliminado y el DRL se evalúa con un rollout greedy en el entorno; ubica esa función y úsala tal cual. Si encuentras restos de beam search, repórtalos antes de seguir.
4. Confirma que `solve_label_setting_oracle` calcula el día de llegada igual que `RoutingEnv` (día de inicio más la parte entera de horas / 14) y aplica las mismas reglas de máscara, incluida la excepción del primer movimiento desde el nodo de inicio. Si no coincide, repórtalo antes de seguir.

### Reglas generales

- No cambies el encoder, el decoder, el crítico, los hiperparámetros de PPO, el loop de entrenamiento, `MarketProcess` ni la dinámica de `RoutingEnv`.
- Todas las políticas (DRL, B1, B2, B3) se ejecutan paso a paso en `RoutingEnv` con la misma interfaz. Una política solo puede leer la máscara, `current_loads_row`, `current_rate_row`, el nodo actual, la hora, `k_remaining`, los visitados y los parámetros del proceso. Ninguna política puede llamar a `sample_world` ni usar la semilla del mundo evaluado.
- El margen de una ruta es la suma de `lane_reward` de cada arco recorrido, calculada con la fila observada en el nodo de salida. No uses la recompensa que devuelve el entorno, que trae bono, penalidad y escala.
- Todo el muestreo propio (UB-DP) usa un `numpy.random.Generator` con semilla fija, nunca el estado global.
- No agregues dependencias. Comentarios y docstrings en español, como el resto del repositorio.

### Pasos

1. **Borrar los baselines viejos.** En `Solvers.py` elimina `solve_HGA_LNS_metaheuristic`, `solve_label_setting_exact`, las cuatro variantes `solve_heuristic_rolling_horizon*`, `solve_mc_rollout_stochastic` y los auxiliares que solo usaban ellas. Conserva `solve_label_setting_oracle` y lo que necesite. Elimina también el modo DRL Det. Busca cada nombre borrado en todo el repositorio (`evaluation.py`, `stats_analysis.py`, `excel_report.py`, `main.py`, `run_k_comparison.py`, `inference.py`, `hpo_optuna.py`) y quita o actualiza cada referencia.
2. **Configuración (`config.py`).** Agrega el paso de tiempo de UB-DP (`UBDP_DT_HORAS = 0.25`), el número de sorteos (`UBDP_S = 500`), su semilla y la ruta donde se guarda la tabla V. Mantén `MARKET_M_EVAL_WORLDS = 100` y `MARKET_SEED_EVAL`.
3. **UB-DP y B3 (`dp_relajada.py`).** Calcula V(día de la semana de salida, nodo, franja de tiempo, k) con la ecuación de Bellman de la sección UB-DP, sin la regla de no revisitar.
    - Los tiempos de viaje se redondean hacia abajo a la franja; el día de llegada es el día de salida más la parte entera de horas / 14.
    - Las acciones factibles siguen las reglas de la máscara del entorno, salvo los visitados: lane con carga en la fila sorteada, retorno a casa posible dentro de `MAX_DURATION`, retorno forzado cuando queda un arco, y la excepción del primer movimiento desde casa.
    - Volver a casa da la recompensa del arco y termina con valor cero.
    - Las filas se sortean con `MarketProcess.sample_arrivals` y un Generator propio: S filas por cada par (nodo, día de la semana), generadas una vez y reutilizadas en todos los estados.
    - La recompensa es `lane_reward`, sin shaping.
    - Calcula una tabla para K = 5 y otra para el caso libre (sin la dimensión k). Guárdalas en disco y recárgalas si ya existen.
    - Expón `ub_dp(nodo_inicio, dia_semana, K)` y la política B3: entre las acciones válidas de la máscara del entorno, elige el argmax de r_ij + V(j, t + τ_ij, k − 1); para el retorno a casa usa solo r_ij.
4. **B1 (greedy).** Entre las acciones válidas, elige el arco con mayor `lane_reward` según la fila observada. Desempata por el índice menor.
5. **B2 (reoptimización con valores esperados), solo K = 5.** En cada nodo:
    - El arco actual usa la recompensa real de la fila observada.
    - Los arcos futuros usan la recompensa esperada de la sección B2, con P(N ≥ 1) de la binomial negativa con el s_d del día de llegada estimado y la tarifa media histórica de la lane.
    - Resuelve de forma exacta la mejor ruta restante (sin revisitar, retorno dentro de `MAX_DURATION`, respetando k) y ejecuta solo el primer arco. Reutiliza la lógica de búsqueda de `solve_label_setting_oracle` si se adapta sin cambiarla.
6. **UB-PI.** Para cada mundo evaluado, genera `sample_world(semilla, 6, dia_inicio)` y resuélvelo con `solve_label_setting_oracle`, pasando ese mundo con día de inicio 0.
7. **Arnés de evaluación (`evaluation.py`).** Reemplaza `run_solver_comparison` y `run_solver_comparison_by_k`.
    - Para cada K en {5, libre}, genera M mundos con semillas derivadas de `MARKET_SEED_EVAL`.
    - Asigna a cada mundo un nodo de inicio y un día de la semana de forma balanceada: cada nodo aparece ⌊M/N⌋ o ⌈M/N⌉ veces y cada día de la semana ⌊M/7⌋ o ⌈M/7⌉ veces, sin que el día quede atado al nodo. Usa como `start_day_idx` un día del histórico con ese día de la semana.
    - En cada mundo corre DRL, B1, B3 y, si K = 5, B2, todos con el mismo `world_seed`. Agrega UB-PI y el valor UB-DP de su (nodo, día, K).
    - Por mundo y método guarda: margen en USD, si hubo violación de tiempo, fracción de millas cargadas, número de cargas, duración y tiempo medio por decisión.
    - Mide los tiempos en CPU con un solo hilo, también para el DRL.
8. **Replay.** Corre el mismo arnés con `market_process=None` sobre los días 90 a 119, con DRL, B1, B2, B3 y UB-PI resuelto sobre el histórico. Repórtalo aparte con la etiqueta "replay, no independiente del simulador".
9. **Estadística (`stats_analysis.py`).** Reemplaza los métodos por DRL, B1, B2 y B3. Para cada comparación DRL contra baseline: diferencia media pareada con intervalo del 95 % por bootstrap sobre mundos (10.000 remuestreos, semilla fija) y Wilcoxon pareado de una cola con corrección de Holm por las tres comparaciones. B2 solo sobre los mundos con K = 5. Agrega el gap de cada método contra UB-DP y contra UB-PI.
10. **Reporte y `main.py`.** Actualiza `excel_report.py` y el resumen de `main.py` a las columnas nuevas. `main.py` corre la evaluación en mundos simulados y el replay. Quita `N_DAYS_PER_NODE` y todo lo que dependía de los solvers borrados.
11. **Pruebas (`tests/`).**
    - B1, B2 y B3 nunca eligen una acción fuera de la máscara.
    - Con el mismo `world_seed`, cada método da exactamente el mismo resultado dos veces.
    - En cada mundo, UB-PI es mayor o igual que el margen de cada política.
    - El promedio de B3 no supera UB-DP más su error de Monte Carlo.
    - El margen calculado coincide con la suma de `lane_reward` de la ruta.
    - Ningún nombre borrado sigue referenciado en el repositorio.
12. **Documentación.** Actualiza `README.md` con los archivos nuevos, los baselines y el flujo de evaluación.

### Criterios de aceptación

- Todas las pruebas pasan.
- `main.py` corre de punta a punta con 10 y con 35 nodos y genera el Excel con los resultados simulados y de replay.
- El repositorio ya no contiene los baselines viejos ni referencias a ellos.

### Al terminar

Reporta los archivos creados, modificados y borrados; el resultado de las pruebas; el tiempo de cálculo de la tabla V para 10 y 35 nodos; el tiempo por decisión de cada método; la tabla resumen de márgenes y gaps; y cualquier discrepancia entre este documento y el código.

## Referencias

| Referencia | Qué respalda |
| --- | --- |
| [Powell WB (2022). Reinforcement Learning and Stochastic Optimization. Wiley](https://castle.princeton.edu/RLSO/) | Clases de política que cubren los baselines |
| [Powell WB (2019). From Reinforcement Learning to Optimal Control. arXiv:1912.03513](https://ar5iv.labs.arxiv.org/html/1912.03513) | Toda política es una de las cuatro clases o un híbrido |
| [Brown DB, Smith JE, Sun P (2010). Information relaxations and duality in stochastic dynamic programs. Operations Research 58(4)](https://pubsonline.informs.org/doi/10.1287/opre.1090.0796) | UB-PI como relajación de información |
| [Hildebrandt FD, Thomas BW, Ulmer MW. Opportunities for reinforcement learning in stochastic dynamic vehicle routing](https://ar5iv.labs.arxiv.org/html/2103.00507) | Taxonomía de políticas en ruteo dinámico |
| [Joe W, Lau HC (2020). Deep RL approach to solve dynamic VRP with stochastic customers. ICAPS 30](https://ojs.aaai.org/index.php/ICAPS/article/view/6685) | Precedente de DRL comparado contra miope, AVI y MSA |
| [Accorsi L, Lodi A, Vigo D. Guidelines for the computational testing of machine learning approaches to vehicle routing problems](https://ar5iv.arxiv.org/html/2109.13983) | Wilcoxon pareado con corrección por comparaciones múltiples, tiempos en CPU |
| [Wang Y, Martins do Nascimento J, Powell WB (2019). RL for dynamic bidding in truckload markets. arXiv:1802.08976](https://arxiv.org/html/1802.08976) | Proceso de cargas estimado del histórico |
| [Canadian traveller problem](https://en.wikipedia.org/wiki/Canadian_traveller_problem) (Papadimitriou y Yannakakis, 1989) | Revelación de arcos al llegar; dificultad del óptimo exacto |
| [Kool W, van Hoof H, Welling M (2019). Attention, learn to solve routing problems! ICLR](https://arxiv.org/abs/1803.08475v3) | Arquitectura base del DRL |
| [Henderson P et al. (2018). Deep reinforcement learning that matters. AAAI](https://mlanthology.org/aaai/2018/henderson2018aaai-deep) | Varias semillas de entrenamiento |
| [Agarwal R et al. (2021). Deep RL at the edge of the statistical precipice. NeurIPS](https://arxiv.org/abs/2108.13264v4) | Intervalos por bootstrap |
| Simao HP et al. (2009). An ADP algorithm for large-scale fleet management. Transportation Science 43(2) | VFA de tabla en truckload (B3); ya está en tu documento |

Verifica el año y la revista de Hildebrandt et al. y de Accorsi et al. en tu gestor bibliográfico; las páginas consultadas no los mostraban.
