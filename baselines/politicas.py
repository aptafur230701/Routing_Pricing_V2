"""
baselines/politicas.py
========================
Políticas baseline no anticipativas: B1, B2, B2P (B2 con pronósticos del índice de mercado) y B3
(política de un paso sobre la tabla V de UB-DP, baselines_drl_ruteo_estocastico.md).

Todas se ejecutan paso a paso en el entorno con cotización, con la misma interfaz que el DRL: en
cada nodo reciben una `Observacion` (máscara, tablero observado: cargas, precios por nivel y
competencia m, nodo actual, hora y k restante) y `actuar` devuelve el DESTINO. El precio lo pone
una regla fija (`cotizar`, por defecto REGLA_PRECIO_BASELINES = R_MIOPE;
especificacion_fase0_ruteo_pricing.md §6-7). Si la carga se pierde, la política vuelve a decidir
en el mismo nodo con la lane enmascarada. El destino puede ser el propio nodo: esperar al siguiente día de
mercado (routing_env.py), o un nodo a ≤ REPOSICION_MAX_MILLAS para reposicionarse vacío.

Cada lane tiene varias cargas: si se pierde una, se puede cotizar otra (pricing.valor_reintentos). El precio cuenta
lo que cuesta perderlas todas (`costo_de_perder`): en la vuelta a casa es volver vacío (−c), no 0. B3 cotiza con el óptimo de su propia tabla (`actuar` decide destino y precio a la vez).

El valor inmediato de un arco es el esperado con la cotización miope, max_k P_k·(p_k·d − c) con
P_k = (1 − u_k)^m (`valor_inmediato`); en la vuelta a casa, max_k P_k·p_k·d − c (perder es volver vacío); −c si la
lane solo se puede recorrer VACÍA, 0 si es la espera. Los parámetros
del proceso (mu, s_d, phi y la tarifa media por lane) se pasan al construir la política. Ninguna
recibe la semilla del mundo evaluado ni ve el competidor.
"""

from dataclasses import dataclass

import numpy as np

from config import (
    MAX_DURATION, UBDP_DT_HORAS, MARKET_DIAS_EPISODIO, TRAIN_DAYS,
    REGLA_PRECIO_BASELINES, COTIZACION_HORAS, COTIZACIONES_MAX_POR_NODO,
)
from pricing import (
    N_NIVELES, U_NIVELES_ARR, nivel_regla, prob_ganar, lane_cost_row, precios_niveles, valor_reintentos,
)
from problem_data import lane_reward
from routing_env import fin_espera
from baselines.dp_relajada import TablaDP, PENALIZACION_USD

_DIA_HORAS = 14.0   # un día de mercado = 14 h (igual que el entorno)
_VACIO = N_NIVELES  # índice de la opción VACÍO (= routing_env.OPCION_VACIO)


@dataclass
class Observacion:
    """Lo único que una política puede leer en un nodo.

    nodo, casa        : nodo actual y nodo de inicio (donde se cierra el ciclo).
    tiempo            : horas transcurridas desde la salida.
    k_restante        : arcos restantes incluido el retorno (None = caso libre).
    mascara           : (N,) int8, destinos válidos del entorno.
    loads_row         : (N,) cargas del tablero del nodo.
    rate_row          : (N,) precio en la mediana del mercado de cada lane (USD/milla).
    precios           : (N, K) USD/milla de cada nivel de precio.
    m                 : competidores efectivos del nodo.
    opciones          : (N, K+1) bool, opciones válidas por destino (niveles y VACÍO).
    dia_semana_salida : día de la semana (lunes = 0) del día de salida.
    indice            : (N,) índice de mercado de hoy (solo B2P).
    n_rechazos        : cotizaciones ya perdidas en esta visita al nodo.
    restantes         : (N,) cargas de cada lane que no se han perdido en esta visita (None = loads_row).
    """
    nodo: int
    casa: int
    tiempo: float
    k_restante: object
    mascara: np.ndarray
    loads_row: np.ndarray
    rate_row: np.ndarray
    precios: np.ndarray
    m: float
    opciones: np.ndarray
    dia_semana_salida: int
    indice: np.ndarray = None
    n_rechazos: int = 0
    restantes: np.ndarray = None

    @property
    def intentos(self) -> np.ndarray:
        """(N,) cargas de cada lane que se pueden cotizar todavía en esta visita (como env.intentos)."""
        cargas = self.loads_row if self.restantes is None else self.restantes
        return np.clip(np.minimum(cargas, COTIZACIONES_MAX_POR_NODO - self.n_rechazos), 0, None).astype(np.int64)


def valor_inmediato(obs: Observacion, distance_arr, diesel_arr, cierre: bool = False) -> np.ndarray:
    """(N,) valor esperado de salir por cada lane: max_k P_k·(p_k·d − c) sobre los niveles válidos, y −c si
    la lane se puede recorrer VACÍA; con cierre=True (B2 / B2P, que suman el valor de toda la ruta) la vuelta a
    casa vale max_k P_k·p_k·d − c: si se pierde la carga, se vuelve vacío. B1 compara contra no moverse (0) y usa
    el valor sin cierre; su PRECIO sí cuenta volver vacío (`cotizar_regla`). 0 en la espera (j = i: la diagonal de distancias trae una distancia
    intra-hub, pero esperar no recorre nada); −inf donde no hay ninguna opción válida."""
    i = obs.nodo
    d, c = distance_arr[i].astype(float).copy(), lane_cost_row(distance_arr[i], diesel_arr[i]).astype(float)
    d[i], c[i] = 0.0, 0.0
    base = np.zeros(len(d))                                                              # perder todas las cargas
    if cierre and i != obs.casa:
        base[obs.casa] = -c[obs.casa]
    V, _ = valor_reintentos(prob_ganar(U_NIVELES_ARR, obs.m), obs.precios * d[:, None] - c[:, None], base,
                            COTIZACIONES_MAX_POR_NODO)                                   # (A+1, N)
    v = np.where(obs.opciones[:, :N_NIVELES].any(axis=1), V[obs.intentos, np.arange(len(d))], -np.inf)
    return np.where(obs.opciones[:, _VACIO], np.maximum(v, -c), v)


def costo_de_perder(obs: Observacion, j: int, costo: float) -> float:
    """F: lo que vale (USD) perder todas las cargas cotizables de la lane j, sin mirar más allá del tramo. La falla
    del ciclo si el camión está en casa y no puede ni esperar ni reposicionarse; volver vacío (−costo) si j es
    casa; 0 en los demás tramos (perder no mueve el camión)."""
    if obs.nodo == obs.casa and not obs.opciones[:, _VACIO].any():
        return PENALIZACION_USD
    return -float(costo) if (j == obs.casa and obs.nodo != obs.casa) else 0.0


def _intentos_hasta_perder(obs: Observacion, j: int) -> int:
    """Cargas que se pueden cotizar antes de quedarse sin nada: las de la lane j o, si hay otras lanes
    cotizables, todas las cotizaciones que quedan en la visita."""
    otras = obs.opciones[:, :N_NIVELES].any(axis=1)
    otras[j] = False
    if obs.nodo == obs.casa and otras.any():
        return max(1, COTIZACIONES_MAX_POR_NODO - obs.n_rechazos)
    return max(1, int(obs.intentos[j]))


def cotizar_regla(obs: Observacion, j: int, distance_arr, diesel_arr, regla: str = REGLA_PRECIO_BASELINES) -> int:
    """Opción para el destino j: el nivel de la regla fija si la lane se puede cotizar; si no, VACÍO.

    La regla miope planea los intentos que le quedan en la lane (pricing.valor_reintentos): empieza más arriba si
    después puede cotizar otra carga, y baja a medida que se agotan, con F = `costo_de_perder` si las pierde
    todas. Cuando F < 0 (la vuelta a casa, o en casa sin esperar ni reposicionarse) toda regla fija cotiza así, en
    vez de volver vacío o fallar el ciclo con su nivel de siempre; en los demás tramos cada una usa su nivel."""
    if not obs.opciones[j, :N_NIVELES].any():
        if not obs.opciones[j, _VACIO]:
            raise ValueError(f"el destino {j} no tiene ninguna opción válida")
        return _VACIO
    i = obs.nodo
    costo = lane_cost_row(distance_arr[i, j], diesel_arr[i, j])
    F = costo_de_perder(obs, j, costo)
    if F < 0 or regla == "miope":
        _, ks = valor_reintentos(prob_ganar(U_NIVELES_ARR, obs.m), obs.precios[j] * distance_arr[i, j] - costo, F,
                                 _intentos_hasta_perder(obs, j))
        return int(ks[-1])
    return int(nivel_regla(regla, obs.precios[j], distance_arr[i, j], costo, obs.m))


class _Cotiza:
    """Mezcla: precio con una regla fija (REGLA_PRECIO_BASELINES salvo que se pase otra)."""
    regla = REGLA_PRECIO_BASELINES

    def cotizar(self, obs: Observacion, j: int) -> int:
        return cotizar_regla(obs, j, self._dist, self._diesel, self.regla)


def _valor_cotizado_por_dow(market, distance_arr, diesel_arr, a_casa: bool = False) -> tuple:
    """(ev, pendiente), cada uno (7, N, N): valor esperado de un arco futuro cotizando con la regla miope sobre
    los cuantiles medios del día de la semana (especificacion_fase0_ruteo_pricing.md §7, B2 / B2P).

    Con los días de ENTRENAMIENTO reales de ese día de la semana:
        f_a[dow, j, l]   = fracción de días en que la lane tuvo a = min(cargas, COTIZACIONES_MAX_POR_NODO) cotizables
        p̄_k[dow, j, l]  = precio del nivel k con los cuantiles medios de la lane (media de los días)
        P̄_k[dow, j]     = media de los días de (1 − u_k)^m_j
        V_a              = valor de la lane con a intentos (pricing.valor_reintentos; perderlos todos vale 0)
        ev               = Σ_{a ≥ 1} f_a · V_a − f_0 · c            (sin carga el arco se recorre vacío)
    a_casa=True: el arco cierra el ciclo, así que perder todas las cargas es volver vacío (base −c).
    pendiente = d · Σ_a f_a · P(ganar alguna de las a) es la derivada de ev respecto de un corrimiento del precio
    (la usa B2P para el pronóstico del índice).
    """
    N = market.num_nodes
    dist = np.asarray(distance_arr, dtype=float)
    costo = lane_cost_row(dist, np.asarray(diesel_arr, dtype=float))
    dow_train = market.dia_semana(np.arange(TRAIN_DAYS))
    ev = np.zeros((7, N, N))
    pendiente = np.zeros((7, N, N))
    for k in range(7):
        dias = np.flatnonzero(dow_train == k)
        n = np.minimum(market.count_loads[dias], COTIZACIONES_MAX_POR_NODO)              # (días, N, N)
        q_medio = np.mean([market.cuantiles_dia(int(d)) for d in dias], axis=0)         # (N, N, 6)
        p = precios_niveles(q_medio)                                                     # (N, N, K)
        P = np.mean([prob_ganar(U_NIVELES_ARR[None, :], market.competidores(np.arange(N), np.full(N, int(d)))[:, None])
                     for d in dias], axis=0)                                             # (N, K)
        base = -costo if a_casa else np.zeros((N, N))
        V, ks = valor_reintentos(P[:, None, :], p * dist[..., None] - costo[..., None], base, COTIZACIONES_MAX_POR_NODO)
        ev[k] = -(n <= 0).mean(axis=0) * costo
        gana_alguna = np.zeros((N, N))                                                   # P(ganar alguna de a cargas)
        for a in range(1, COTIZACIONES_MAX_POR_NODO + 1):
            p_a = np.take_along_axis(P, ks[a - 1], axis=1)                               # (N, N) P̄ del nivel del intento
            gana_alguna = p_a + (1.0 - p_a) * gana_alguna
            f_a = (n == a).mean(axis=0)
            ev[k] += f_a * V[a]
            pendiente[k] += f_a * gana_alguna * dist
    return ev, pendiente


def calcular_ev_esperado(market, distance_arr, diesel_arr) -> np.ndarray:
    """Valor esperado por lane y día de la semana, (7, N, N), con la cotización miope (ver
    `_valor_cotizado_por_dow`). Es el valor que usa B2 para los arcos futuros; se calcula una vez por número de nodos."""
    return _valor_cotizado_por_dow(market, distance_arr, diesel_arr)[0]


@dataclass(frozen=True)
class EvEsperado:
    """Valor esperado por lane con pronóstico del índice de mercado (tabla de B2P).

    Para un arco j→l que sale de j un día de mercado `dow` de la semana, h días de mercado después de hoy:

        ev_ajustado(h, dow)[j, l] = ev[dow, j, l] + índice(j, hoy) · q[h, dow, j, l]

    ev (7, N, N)      : calcular_ev_esperado (cotización miope sobre los cuantiles medios; índice cero)
    q  (H+1, 7, N, N) : b_h · P(N_jl ≥ 1) · P̄_k* · dist_jl, con b_h = f + (1 − f) · ρ^h
    ev_casa, q_casa   : lo mismo para el arco que cierra el ciclo (perder la carga = volver vacío)
    b  (H+1,)         : los coeficientes b_h (f = fracción de varianza lenta del índice, ρ = persistencia del rápido)
    """
    ev: np.ndarray
    q:  np.ndarray
    b:  np.ndarray
    ev_casa: np.ndarray
    q_casa:  np.ndarray

    @property
    def h_max(self) -> int:
        return self.q.shape[0] - 1


def calcular_ev_pronostico(market, distance_arr, diesel_arr) -> EvEsperado:
    """Tabla de valores esperados por lane con pronóstico del índice (ver EvEsperado); una vez por nº de nodos."""
    ev, pendiente = _valor_cotizado_por_dow(market, distance_arr, diesel_arr)   # (7,N,N) cada uno
    b = market.frac_var_lento + (1.0 - market.frac_var_lento) * market.rho ** np.arange(MARKET_DIAS_EPISODIO)  # h = 0 … H
    q = b[:, None, None, None] * pendiente[None]                          # (H+1, 7, N, N)
    ev_c, pend_c = _valor_cotizado_por_dow(market, distance_arr, diesel_arr, a_casa=True)
    return EvEsperado(ev=ev, q=q, b=b, ev_casa=ev_c, q_casa=b[:, None, None, None] * pend_c[None])


class PoliticaB1(_Cotiza):
    """B1, greedy miope (PFA): jugada válida de mayor valor inmediato esperado (`valor_inmediato`). La espera
    vale 0, así que B1 espera mientras ninguna lane tenga valor esperado positivo y todavía pueda esperar.

    Empates por el índice menor.
    """
    nombre = "B1"

    def __init__(self, distance_arr, diesel_arr):
        self._dist, self._diesel = distance_arr, diesel_arr

    def actuar(self, obs: Observacion) -> int:
        i = obs.nodo
        acts = np.flatnonzero(obs.mascara)
        r = valor_inmediato(obs, self._dist, self._diesel)[acts]
        return int(acts[int(np.argmax(r))])      # argmax devuelve el primero ⇒ índice menor


class PoliticaB2(_Cotiza):
    """B2, reoptimización determinista con valores esperados (CFA); con K = 5 y con K libre.

    En cada nodo resuelve de forma exacta la mejor ruta restante y ejecuta solo el primer
    arco. El arco actual usa la recompensa real de la fila observada; los arcos futuros
    (j, l) usan la esperada

        E[r_jl] = P(N_jl ≥ 1) · q̄_jl · dist_jl − c_jl,   P(N ≥ 1) = 1 − (φ / (φ + m_jl))^φ

    con m = mu_jl · s_d del día de llegada estimado y q̄ = media_lane, la tarifa media de la lane.

    El camión puede volver a pasar por un nodo, así que el estado del futuro es (j, franja de
    tiempo, k restante) y la mejor continuación es una recursión memoizada por (casa, día de
    salida). Las franjas son de UBDP_DT_HORAS y los tiempos de viaje se redondean hacia abajo, como
    en baselines/dp_relajada.py; la regla de retorno factible es la de la máscara (volver a casa dentro de
    MAX_DURATION). Con k_restante None (caso libre) no hay dimensión k y la profundidad solo la
    acota MAX_DURATION. Como ev no depende del mundo, el memo se reutiliza entre decisiones y mundos.
    En cada nodo futuro también puede esperar al siguiente día de mercado (mismo k). Esperar en casa antes de
    salir se valora con una sola salida al día siguiente (`_salida`, sin volver a esperar).
    """
    nombre = "B2"

    def __init__(self, market, time_matrix, distance_arr, diesel_arr, max_duration=MAX_DURATION,
                 dt=UBDP_DT_HORAS):
        self._T = np.asarray(time_matrix, dtype=float)
        self._dist, self._diesel = distance_arr, diesel_arr
        self._max = float(max_duration)
        self._dt = float(dt)
        self.N = self._T.shape[0]
        self._ev = calcular_ev_esperado(market, distance_arr, diesel_arr)   # (7, N, N)
        self._ev_casa = _valor_cotizado_por_dow(market, distance_arr, diesel_arr, a_casa=True)[0]   # arco de cierre
        self._f = np.floor(self._T / self._dt + 1e-9).astype(np.int64)      # franjas de viaje, redondeo hacia abajo
        if self._f[~np.eye(self.N, dtype=bool)].min() < 1:
            raise ValueError("Hay tiempos de viaje menores a una franja: la recursión del caso libre "
                             "no estaría bien fundada. Reduce dt.")
        self._smax = int(np.floor(self._max / self._dt + 1e-9))
        self._por_dia = int(round(_DIA_HORAS / self._dt))                  # franjas de un día de mercado
        self._nodos = np.arange(self.N)
        self._memo = {}                                                     # (casa, dow0) → {(j, s, k): valor}

    def _franja(self, t) -> int:
        return min(int(np.floor(t / self._dt + 1e-9)), self._smax)

    def _salida(self, h, dow0, s, k, memo) -> float:
        """Valor de salir de casa en la franja s (después de esperar en casa): la mejor primera lane."""
        ev = self._ev[(dow0 + int(s * self._dt / _DIA_HORAS + 1e-9)) % 7]
        k_sig = None if k is None else k - 1
        mejor = -np.inf
        for j in np.flatnonzero((s + self._f[h] + self._f[:, h] <= self._smax) & (self._nodos != h)):
            mejor = max(mejor, ev[h, j] + self._valor(h, dow0, int(j), s + int(self._f[h, j]), k_sig, memo))
        return float(mejor)

    def actuar(self, obs: Observacion) -> int:
        i, h, dow0 = obs.nodo, obs.casa, obs.dia_semana_salida
        acts = np.flatnonzero(obs.mascara)
        r = valor_inmediato(obs, self._dist, self._diesel, cierre=True)[acts]
        memo = self._memo.setdefault((h, dow0), {})
        k_sig = None if obs.k_restante is None else obs.k_restante - 1
        mejor, mejor_a = -np.inf, -1
        for a, ra in zip(acts, r):
            if a == i:                                                    # esperar al día siguiente (mismo k)
                s_w = self._franja(fin_espera(obs.tiempo))
                v = (self._salida(h, dow0, s_w, obs.k_restante, memo) if i == h
                     else self._valor(h, dow0, int(i), s_w, obs.k_restante, memo))
            elif a == h:
                v = float(ra)                                             # cerrar el ciclo ahora
            else:
                s_a = min(int(np.floor((obs.tiempo + self._T[i, a]) / self._dt + 1e-9)), self._smax)
                v = float(ra) + self._valor(h, dow0, int(a), s_a, k_sig, memo)
            if v > mejor:
                mejor, mejor_a = v, int(a)
        return mejor_a

    def _valor(self, h, dow0, j, s, k, memo) -> float:
        """Mejor valor esperado de seguir desde j (llegada en la franja s, k arcos restantes; None = libre)."""
        clave = (j, s, k)
        if clave in memo:
            return memo[clave]
        dow = (dow0 + int(s * self._dt / _DIA_HORAS + 1e-9)) % 7
        ev = self._ev[dow]
        mejor = self._ev_casa[dow, j, h]                                  # cerrar el ciclo desde j
        s_w = (s // self._por_dia + 1) * self._por_dia                     # esperar al día siguiente en j
        if s_w + self._f[j, h] <= self._smax:
            mejor = max(mejor, self._valor(h, dow0, j, s_w, k, memo))
        if k is None or k >= 2:
            f = self._f
            cand = np.flatnonzero(
                (s + f[j] + f[:, h] <= self._smax) & (self._nodos != j) & (self._nodos != h)
            )
            for l in cand:
                v = ev[j, l] + self._valor(h, dow0, int(l), s + int(f[j, l]),
                                           None if k is None else k - 1, memo)
                if v > mejor:
                    mejor = v
        memo[clave] = float(mejor)
        return memo[clave]


class PoliticaB2P(PoliticaB2):
    """B2P, B2 con pronósticos del índice de mercado.

    Igual que PoliticaB2 salvo en los valores esperados de los arcos futuros. Para un arco futuro j→l al
    que se llega h días de mercado después de hoy, la tarifa esperada es

        media_lane(jl) + b_h · índice(j, hoy),     b_h = f + (1 − f) · ρ^h

    con f la fracción de varianza lenta del índice y ρ la persistencia del rápido (ambas del ajuste): la
    mejor predicción lineal del índice de j h días adelante, dado el de hoy (el lento no decae y el rápido
    decae como ρ^h). El índice de hoy es el que observa el camión (`Observacion.indice`). El arco actual
    usa la fila observada, como en B2.

        E[r_jl | h] = P(N_jl ≥ 1) · (q̄_jl + b_h · índice_j) · dist_jl − c_jl

    es lineal en el índice, así que se precalculan las tablas por h y día de la semana
    (`calcular_ev_pronostico`):
    E0[dow] = ev de B2 (b = 0) y Q[h, dow] = b_h · P(N ≥ 1) · dist; por decisión, ev(h, dow) =
    E0[dow] + índice[:, None] · Q[h, dow] (N×N). Con b_h = 0 (índice cero) B2P coincide con B2.

    Como los valores esperados dependen del índice de hoy, la mejor continuación no se puede memorizar
    entre decisiones: se recalcula en cada nodo con una recursión hacia atrás sobre las franjas, vectorizada
    sobre los nodos (misma regla de factibilidad, franjas, retorno y espera que B2).
    """
    nombre = "B2P"
    usa_indice = True            # el arnés le pasa Observacion.indice

    def __init__(self, market, time_matrix, distance_arr, diesel_arr, max_duration=MAX_DURATION,
                 dt=UBDP_DT_HORAS):
        super().__init__(market, time_matrix, distance_arr, diesel_arr, max_duration, dt)
        self.f, self.rho = float(market.frac_var_lento), float(market.rho)
        tabla = calcular_ev_pronostico(market, distance_arr, diesel_arr)                   # tabla de B2P
        self._h_max = tabla.h_max
        self.b = tabla.b                                                                   # (H+1,)
        self._E0 = tabla.ev                                                                # b = 0
        self._Q = tabla.q                                                                  # (H+1, 7, N, N)
        self._E0_casa, self._Q_casa = tabla.ev_casa, tabla.q_casa                          # arco de cierre
        self._casa = {}                                                                    # h → (f_jl + f_lh, l ∉ {j, h})

    def _tabla_casa(self, h):
        if h not in self._casa:
            base = self._f + self._f[:, h][None, :]                                        # (j, l): f_jl + f_lh
            ok = (self._nodos[None, :] != self._nodos[:, None]) & (self._nodos[None, :] != h)
            self._casa[h] = (base, ok)
        return self._casa[h]

    def _ev_dia(self, cache, dow0, dia_hoy, dia, indice):
        """(N, N) valor esperado de los arcos que salen de un nodo al que se llega el día `dia` (h = dia − hoy)."""
        if dia not in cache:
            dow = (dow0 + dia) % 7
            hh = min(max(dia - dia_hoy, 0), self._h_max)
            cache[dia] = self._E0[dow] + indice[:, None] * self._Q[hh, dow]
        return cache[dia]

    def _ev_cierre(self, dow0, dia_hoy, dia, indice, h):
        """(N,) valor esperado del arco j→casa que cierra el ciclo, saliendo de j el día `dia`."""
        dow = (dow0 + dia) % 7
        hh = min(max(dia - dia_hoy, 0), self._h_max)
        return self._E0_casa[dow, :, h] + indice * self._Q_casa[hh, dow, :, h]

    def _valores(self, h, dow0, dia_hoy, indice, s_lo, k) -> np.ndarray:
        """V[j, s] para s ≥ s_lo: mejor valor esperado de seguir desde j (llegada en la franja s, k arcos restantes
        incluido el retorno; None = libre), con los pronósticos del índice de hoy."""
        N, smax, dt = self.N, self._smax, self._dt
        F, nodos = self._f, self._nodos
        base, ok = self._tabla_casa(h)
        cache = {}
        dia_de = lambda s: int(s * dt / _DIA_HORAS + 1e-9)

        por_dia = self._por_dia
        vuelta = F[:, h]                                                                   # (N,) f[j, h]

        def esperar(s, V, mejor):
            """max(mejor, V[:, s_w]): esperar en j al siguiente día de mercado (misma tabla: no consume k)."""
            s_w = (s // por_dia + 1) * por_dia
            if s_w > smax:
                return mejor
            return np.where(s_w + vuelta <= smax, np.maximum(mejor, V[:, s_w]), mejor)

        cierre = {}

        def ev_h(s):
            d = dia_de(s)
            if d not in cierre:
                cierre[d] = self._ev_cierre(dow0, dia_hoy, d, indice, h)
            return cierre[d]

        def paso(s, V_sig, ev):
            """V[:, s] sin la espera, a partir de V_sig[l, s'] (V_sig es la tabla del nivel k−1, o la propia en
            el caso libre)."""
            mejor = ev_h(s).copy()                                                         # cerrar el ciclo desde j
            factible = ok & (s + base <= smax)
            if factible.any():
                s_sig = np.minimum(s + F, smax)
                cand = np.where(factible, ev + V_sig[nodos[None, :], s_sig], -np.inf)
                mejor = np.maximum(mejor, cand.max(axis=1))
            return mejor

        # Todas las tablas se llenan hacia atrás en s: la espera lee franjas posteriores del mismo nivel.
        if k is None:                                                                      # caso libre
            V = np.zeros((N, smax + 1))
            for s in range(smax, s_lo - 1, -1):
                V[:, s] = esperar(s, V, paso(s, V, self._ev_dia(cache, dow0, dia_hoy, dia_de(s), indice)))
            return V
        V = np.zeros((N, smax + 1))
        for s in range(smax, s_lo - 1, -1):                                                # k = 1: casa o esperar
            V[:, s] = esperar(s, V, ev_h(s))
        for _ in range(2, int(k) + 1):                                                     # k ≥ 2: un nivel por arco
            V_ant, V = V, np.zeros((N, smax + 1))
            for s in range(smax, s_lo - 1, -1):
                V[:, s] = esperar(s, V, paso(s, V_ant, self._ev_dia(cache, dow0, dia_hoy, dia_de(s), indice)))
        return V

    def actuar(self, obs: Observacion) -> int:
        if obs.indice is None:
            raise ValueError("B2P necesita Observacion.indice (el índice de mercado de hoy)")
        i, h, dow0 = obs.nodo, obs.casa, obs.dia_semana_salida
        acts = np.flatnonzero(obs.mascara)
        r = valor_inmediato(obs, self._dist, self._diesel, cierre=True)[acts]
        k_sig = None if obs.k_restante is None else obs.k_restante - 1
        s_a = np.minimum(np.floor((obs.tiempo + self._T[i, acts]) / self._dt + 1e-9).astype(np.int64), self._smax)
        espera = acts == i
        s_w = self._franja(fin_espera(obs.tiempo))
        s_a[espera] = s_w
        sigue = acts != h
        indice = np.asarray(obs.indice, dtype=float)
        dia_hoy = int(np.floor(obs.tiempo / self._dt + 1e-9) * self._dt / _DIA_HORAS + 1e-9)
        s_lo = [int(s_a[sigue].min())] if sigue.any() else []
        if espera.any() and i == h:                        # la salida de casa al día siguiente
            s_lo.append(int(min(s_w + self._f[h][self._nodos != h].min(), self._smax)))
        if s_lo:
            V = self._valores(h, dow0, dia_hoy, indice, min(s_lo), k_sig)
        valor_espera = 0.0
        if espera.any():
            if i == h:                                     # esperar en casa: la mejor salida al día siguiente
                ev_w = self._ev_dia({}, dow0, dia_hoy, int(s_w * self._dt / _DIA_HORAS + 1e-9), indice)
                ok = (s_w + self._f[h] + self._f[:, h] <= self._smax) & (self._nodos != h)
                s_j = np.minimum(s_w + self._f[h], self._smax)
                valor_espera = float(np.max(np.where(ok, ev_w[h] + V[self._nodos, s_j], -np.inf))) if ok.any() else -np.inf
            else:                                          # mismo k: la espera no consume arcos
                V_k = V if obs.k_restante is None else self._valores(h, dow0, dia_hoy, indice, s_w, obs.k_restante)
                valor_espera = float(V_k[i, s_w])
        mejor, mejor_a = -np.inf, -1
        for a, ra, sa, cont, esp in zip(acts, r, s_a, sigue, espera):
            v = valor_espera if esp else float(ra) + (float(V[a, sa]) if cont else 0.0)
            if v > mejor:
                mejor, mejor_a = v, int(a)
        return mejor_a


class PoliticaB3:
    """B3: un paso sobre la tabla V de UB-DP con los intentos de cotización (especificacion_fase0_ruteo_pricing.md §7.3).

    Con el tablero observado del nodo i (precios por nivel, m, cargas) y G_jk(t) = r_jk + V(j, t + τ_ij, k − 1)
    (V(casa) = 0), P_k = (1 − u_k)^m y t_r = t + r · COTIZACION_HORAS:

        B_r = max( E(t_r), max_j max_k [ P_k G_jk(t_r) + (1 − P_k) B_{r+1} ] )    (lanes con más de r cargas)
        E(t) = max_j e_j(t): la mejor jugada sin carga
        e_j(t) = R0_ij + V(j, t + τ_ij, k − 1)  (vacío a j: casa o reposicionamiento);
                 V(i, fin_espera(t), k) para j = i (esperar; en casa antes de salir, H(casa, fin_espera(t), k))

    r cuenta los rechazos futuros, hasta los que quedan en la visita. Elige la jugada (j, k) que alcanza B_0: destino
    y precio salen del mismo máximo (VACÍO si gana e_j). Empates por el índice menor.
    """
    nombre = "B3"

    def __init__(self, tabla: TablaDP, distance_arr, diesel_arr):
        self.tabla = tabla
        self._opcion = _VACIO
        self._dist, self._diesel = distance_arr, diesel_arr

    def _cont(self, obs, acts, t, k_sig):
        """V(j, t + τ_ij, k − 1) de los destinos `acts` (0 para casa)."""
        if k_sig == 0:
            return np.zeros(len(acts))
        cont = self.tabla.valor_continuacion(obs.dia_semana_salida, obs.casa, acts,
                                             t + self.tabla.T[obs.nodo, acts], k_sig)
        return np.where(acts == obs.casa, 0.0, cont)

    def _espera(self, obs, t) -> float:
        """Valor de esperar en el nodo actual hasta el día siguiente a partir de la hora t (mismo k)."""
        t_w = float(fin_espera(t))
        if obs.nodo == obs.casa:
            return self.tabla.valor_salida(obs.dia_semana_salida, obs.casa, t_w, obs.k_restante)
        return float(self.tabla.valor_continuacion(obs.dia_semana_salida, obs.casa, np.array([obs.nodo]),
                                                   np.array([t_w]), obs.k_restante)[0])

    def _resolver(self, obs: Observacion) -> tuple:
        """(acts, v_cot, e1, Q) de la decisión AHORA (r = 0): para cada destino factible `acts[x]`, el valor de la mejor
        cotización v_cot[x] (la opción de precio que la alcanza es argmax Q[x]) y el de la mejor jugada sin carga e1[x]."""
        i, h = obs.nodo, obs.casa
        acts = np.flatnonzero(obs.mascara)
        k_sig = None if obs.k_restante is None else obs.k_restante - 1
        r_k = lane_reward(obs.precios[acts], 1, self._dist[i, acts, None], self._diesel[i, acts, None])   # (A, K)
        r0 = lane_reward(0.0, 0, self._dist[i, acts], self._diesel[i, acts])                             # (A,)
        P = prob_ganar(U_NIVELES_ARR, obs.m)                                                              # (K,)
        cotiza = obs.opciones[acts, :N_NIVELES].any(axis=1)
        vacia = obs.opciones[acts, _VACIO]
        espera = acts == i
        intentos = obs.intentos[acts]

        def sin_carga(t_, c):
            """(A,) e_j(t_): vacío a j (casa o reposicionamiento) o esperar si j = i; −inf si no vale."""
            e = np.where(vacia, r0 + c, -np.inf)
            if espera.any():
                e[espera] = self._espera(obs, t_)
            return e

        # Hacia atrás sobre los intentos que quedan: B = lo que vale el nodo después de r rechazos más. Después de r
        # rechazos se cotizan las lanes con más de r cargas (siempre factible, sea cual sea la lane perdida).
        quedan = COTIZACIONES_MAX_POR_NODO - obs.n_rechazos
        B = -np.inf
        for r in range(quedan, -1, -1):
            t_r = obs.tiempo + r * COTIZACION_HORAS
            c_r = self._cont(obs, acts, t_r, k_sig)
            e1 = sin_carga(t_r, c_r)
            # Como la máscara del entorno: lejos de casa se cotiza solo si, al perder, todavía se puede volver.
            a_tiempo = i == h or t_r + COTIZACION_HORAS + self.tabla.T[i, h] <= MAX_DURATION
            Q = np.where((cotiza & a_tiempo & (intentos > r))[:, None], P * (r_k + c_r[:, None]) + (1 - P) * B, -np.inf) \
                if r < quedan else np.full((len(acts), len(P)), -np.inf)                                      # (A, K)
            v_cot = Q.max(axis=1)
            B = float(np.maximum(v_cot, e1).max(initial=-np.inf))
            if not np.isfinite(B):
                B = PENALIZACION_USD                   # sin respaldo (solo en casa): el ciclo falla
        return acts, v_cot, e1, Q

    def actuar(self, obs: Observacion) -> int:
        acts, v_cot, e1, Q = self._resolver(obs)
        x = int(np.argmax(np.maximum(v_cot, e1)))      # r = 0: ahora; argmax devuelve el primero ⇒ índice menor
        self._opcion = int(np.argmax(Q[x])) if v_cot[x] >= e1[x] else _VACIO
        return int(acts[x])

    def precio_en(self, obs: Observacion, j: int) -> int:
        """Opción de B3 para un destino FIJADO desde afuera (comparación "rutas del DRL + precio de la tabla"): con el mismo
        máximo de `actuar` pero restringido a j. El valor de lo que pasa después de un rechazo (B_1) sigue siendo libre: se
        puede cambiar de destino. Devuelve el nivel de precio que maximiza el valor, o VACÍO si gana la jugada sin carga."""
        acts, v_cot, e1, Q = self._resolver(obs)
        x = int(np.flatnonzero(acts == j)[0])
        if not np.isfinite(max(v_cot[x], e1[x])):      # la tabla no ve jugada en j (no debería pasar con j de la máscara)
            return cotizar_regla(obs, j, self._dist, self._diesel, "miope")
        return int(np.argmax(Q[x])) if v_cot[x] >= e1[x] else _VACIO

    def cotizar(self, obs: Observacion, j: int) -> int:
        return self._opcion                            # la decidió `actuar` con el mismo máximo
