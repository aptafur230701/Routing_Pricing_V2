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
en el mismo nodo con la lane enmascarada.

El valor inmediato de un arco es el esperado con la cotización miope, max_k P_k·(p_k·d − c) con
P_k = (1 − u_k)^m (`valor_inmediato`); −c si la lane solo se puede recorrer VACÍA. Los parámetros
del proceso (mu, s_d, phi y la tarifa media por lane) se pasan al construir la política. Ninguna
recibe la semilla del mundo evaluado ni ve el competidor.
"""

from dataclasses import dataclass

import numpy as np

from config import (
    MAX_DURATION, UBDP_DT_HORAS, MARKET_DIAS_EPISODIO, TRAIN_DAYS,
    REGLA_PRECIO_BASELINES, COTIZACION_HORAS, COTIZACIONES_MAX_POR_NODO,
)
from pricing import N_NIVELES, U_NIVELES_ARR, nivel_regla, prob_ganar, lane_cost_row, precios_niveles
from problem_data import lane_reward
from baselines.dp_relajada import TablaDP

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
    n_rechazos        : cotizaciones ya perdidas en esta visita al nodo (B3).
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


def valor_inmediato(obs: Observacion, distance_arr, diesel_arr) -> np.ndarray:
    """(N,) valor esperado de salir por cada lane: max_k P_k·(p_k·d − c) sobre los niveles válidos, o −c si
    la lane solo se puede recorrer VACÍA; −inf donde no hay ninguna opción válida."""
    i = obs.nodo
    d, c = distance_arr[i], lane_cost_row(distance_arr[i], diesel_arr[i])
    v = prob_ganar(U_NIVELES_ARR, obs.m) * (obs.precios * d[:, None] - c[:, None])        # (N, K)
    v = np.where(obs.opciones[:, :N_NIVELES], v, -np.inf).max(axis=1)
    return np.where(obs.opciones[:, _VACIO], np.maximum(v, -c), v)


def cotizar_regla(obs: Observacion, j: int, distance_arr, diesel_arr, regla: str = REGLA_PRECIO_BASELINES) -> int:
    """Opción para el destino j: el nivel de la regla fija si la lane se puede cotizar; si no, VACÍO."""
    if not obs.opciones[j, :N_NIVELES].any():
        if not obs.opciones[j, _VACIO]:
            raise ValueError(f"el destino {j} no tiene ninguna opción válida")
        return _VACIO
    i = obs.nodo
    costo = lane_cost_row(distance_arr[i, j], diesel_arr[i, j])
    return int(nivel_regla(regla, obs.precios[j], distance_arr[i, j], costo, obs.m))


class _Cotiza:
    """Mezcla: precio con una regla fija (REGLA_PRECIO_BASELINES salvo que se pase otra)."""
    regla = REGLA_PRECIO_BASELINES

    def cotizar(self, obs: Observacion, j: int) -> int:
        return cotizar_regla(obs, j, self._dist, self._diesel, self.regla)


def _valor_cotizado_por_dow(market, distance_arr, diesel_arr) -> tuple:
    """(ev, pendiente), cada uno (7, N, N): valor esperado de un arco futuro cotizando con la regla miope sobre
    los cuantiles medios del día de la semana (especificacion_fase0_ruteo_pricing.md §7, B2 / B2P).

    Con los días de ENTRENAMIENTO reales de ese día de la semana:
        prob[dow, j, l]  = fracción de días en que la lane tuvo carga
        p̄_k[dow, j, l]  = precio del nivel k con los cuantiles medios de la lane (media de los días)
        P̄_k[dow, j]     = media de los días de (1 − u_k)^m_j
        k*              = argmax_k P̄_k · (p̄_k · d − c)            (la regla miope con esos valores)
        ev              = prob · P̄_k* · (p̄_k* · d − c) − (1 − prob) · c
    Con P̄ = 1 y un solo precio (la tarifa media) se recupera el valor de antes, prob · q̄ · d − c: sin carga
    el arco se recorre vacío (−c); con carga se gana con P̄_k* y, si se pierde, el tramo no se cobra.
    pendiente = prob · P̄_k* · d es la derivada de ev respecto de un corrimiento del precio (la usa B2P para
    el pronóstico del índice).
    """
    N = market.num_nodes
    dist = np.asarray(distance_arr, dtype=float)
    costo = lane_cost_row(dist, np.asarray(diesel_arr, dtype=float))
    dow_train = market.dia_semana(np.arange(TRAIN_DAYS))
    con_carga = market.count_loads[:TRAIN_DAYS] > 0
    ev = np.zeros((7, N, N))
    pendiente = np.zeros((7, N, N))
    for k in range(7):
        dias = np.flatnonzero(dow_train == k)
        prob = con_carga[dias].mean(axis=0)                                             # (N, N)
        q_medio = np.mean([market.cuantiles_dia(int(d)) for d in dias], axis=0)         # (N, N, 6)
        p = precios_niveles(q_medio)                                                     # (N, N, K)
        P = np.mean([prob_ganar(U_NIVELES_ARR[None, :], market.competidores(np.arange(N), np.full(N, int(d)))[:, None])
                     for d in dias], axis=0)                                             # (N, K)
        v = P[:, None, :] * (p * dist[..., None] - costo[..., None])                    # (N, N, K)
        k_star = v.argmax(axis=-1)[..., None]
        ev[k] = prob * np.take_along_axis(v, k_star, axis=-1)[..., 0] - (1.0 - prob) * costo
        pendiente[k] = prob * np.take_along_axis(np.broadcast_to(P[:, None, :], v.shape), k_star, axis=-1)[..., 0] * dist
    return ev, pendiente


def calcular_ev_esperado(market, distance_arr, diesel_arr) -> np.ndarray:
    """Valor esperado por lane y día de la semana, (7, N, N), con la cotización miope (ver
    `_valor_cotizado_por_dow`). Es el valor que usa B2 para los arcos futuros y el que alimenta las
    features de continuación del DRL; se calcula una vez por número de nodos."""
    return _valor_cotizado_por_dow(market, distance_arr, diesel_arr)[0]


@dataclass(frozen=True)
class EvEsperado:
    """Valor esperado por lane con pronóstico del índice de mercado (tabla compartida por B2P y el encoder).

    Para un arco j→l que sale de j un día de mercado `dow` de la semana, h días de mercado después de hoy:

        ev_ajustado(h, dow)[j, l] = ev[dow, j, l] + índice(j, hoy) · q[h, dow, j, l]

    ev (7, N, N)      : calcular_ev_esperado (cotización miope sobre los cuantiles medios; índice cero)
    q  (H+1, 7, N, N) : b_h · P(N_jl ≥ 1) · P̄_k* · dist_jl, con b_h = f + (1 − f) · ρ^h
    b  (H+1,)         : los coeficientes b_h (f = fracción de varianza lenta del índice, ρ = persistencia del rápido)
    """
    ev: np.ndarray
    q:  np.ndarray
    b:  np.ndarray

    @property
    def h_max(self) -> int:
        return self.q.shape[0] - 1


def calcular_ev_pronostico(market, distance_arr, diesel_arr) -> EvEsperado:
    """Tabla de valores esperados por lane con pronóstico del índice (ver EvEsperado); una vez por nº de nodos."""
    ev, pendiente = _valor_cotizado_por_dow(market, distance_arr, diesel_arr)   # (7,N,N) cada uno
    b = market.frac_var_lento + (1.0 - market.frac_var_lento) * market.rho ** np.arange(MARKET_DIAS_EPISODIO)  # h = 0 … H
    q = b[:, None, None, None] * pendiente[None]                          # (H+1, 7, N, N)
    return EvEsperado(ev=ev, q=q, b=b)


class PoliticaB1(_Cotiza):
    """B1, greedy miope (PFA): arco válido de mayor valor inmediato esperado (`valor_inmediato`).

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
        self._f = np.floor(self._T / self._dt + 1e-9).astype(np.int64)      # franjas de viaje, redondeo hacia abajo
        if self._f[~np.eye(self.N, dtype=bool)].min() < 1:
            raise ValueError("Hay tiempos de viaje menores a una franja: la recursión del caso libre "
                             "no estaría bien fundada. Reduce dt.")
        self._smax = int(np.floor(self._max / self._dt + 1e-9))
        self._nodos = np.arange(self.N)
        self._memo = {}                                                     # (casa, dow0) → {(j, s, k): valor}

    def actuar(self, obs: Observacion) -> int:
        i, h, dow0 = obs.nodo, obs.casa, obs.dia_semana_salida
        acts = np.flatnonzero(obs.mascara)
        r = valor_inmediato(obs, self._dist, self._diesel)[acts]
        memo = self._memo.setdefault((h, dow0), {})
        k_sig = None if obs.k_restante is None else obs.k_restante - 1
        mejor, mejor_a = -np.inf, -1
        for a, ra in zip(acts, r):
            if a == h:
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
        ev = self._ev[(dow0 + int(s * self._dt / _DIA_HORAS + 1e-9)) % 7]
        mejor = ev[j, h]                                                  # cerrar el ciclo desde j
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
    (`calcular_ev_pronostico`, compartida con las features de continuación del DRL):
    E0[dow] = ev de B2 (b = 0) y Q[h, dow] = b_h · P(N ≥ 1) · dist; por decisión, ev(h, dow) =
    E0[dow] + índice[:, None] · Q[h, dow] (N×N). Con b_h = 0 (índice cero) B2P coincide con B2.

    Como los valores esperados dependen del índice de hoy, la mejor continuación no se puede memorizar
    entre decisiones: se recalcula en cada nodo con una recursión hacia atrás sobre las franjas, vectorizada
    sobre los nodos (misma regla de factibilidad, franjas y retorno que B2).
    """
    nombre = "B2P"
    usa_indice = True            # el arnés le pasa Observacion.indice

    def __init__(self, market, time_matrix, distance_arr, diesel_arr, max_duration=MAX_DURATION,
                 dt=UBDP_DT_HORAS):
        super().__init__(market, time_matrix, distance_arr, diesel_arr, max_duration, dt)
        self.f, self.rho = float(market.frac_var_lento), float(market.rho)
        tabla = calcular_ev_pronostico(market, distance_arr, diesel_arr)                   # la misma que usa el encoder
        self._h_max = tabla.h_max
        self.b = tabla.b                                                                   # (H+1,)
        self._E0 = tabla.ev                                                                # b = 0
        self._Q = tabla.q                                                                  # (H+1, 7, N, N)
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

    def _valores(self, h, dow0, dia_hoy, indice, s_lo, k) -> np.ndarray:
        """V[j, s] para s ≥ s_lo: mejor valor esperado de seguir desde j (llegada en la franja s, k arcos restantes
        incluido el retorno; None = libre), con los pronósticos del índice de hoy."""
        N, smax, dt = self.N, self._smax, self._dt
        F, nodos = self._f, self._nodos
        base, ok = self._tabla_casa(h)
        cache = {}
        dia_de = lambda s: int(s * dt / _DIA_HORAS + 1e-9)

        def paso(s, V_sig, ev):
            """V[:, s] a partir de V_sig[l, s'] (V_sig es la tabla del nivel k−1, o la propia en el caso libre)."""
            mejor = ev[:, h].copy()                                                        # cerrar el ciclo desde j
            factible = ok & (s + base <= smax)
            if factible.any():
                s_sig = np.minimum(s + F, smax)
                cand = np.where(factible, ev + V_sig[nodos[None, :], s_sig], -np.inf)
                mejor = np.maximum(mejor, cand.max(axis=1))
            return mejor

        if k is None:                                                                      # caso libre: s hacia atrás
            V = np.zeros((N, smax + 1))
            for s in range(smax, s_lo - 1, -1):
                V[:, s] = paso(s, V, self._ev_dia(cache, dow0, dia_hoy, dia_de(s), indice))
            return V
        V = np.zeros((N, smax + 1))
        for s in range(s_lo, smax + 1):                                                    # k = 1: solo el retorno
            V[:, s] = self._ev_dia(cache, dow0, dia_hoy, dia_de(s), indice)[:, h]
        for _ in range(2, int(k) + 1):                                                     # k ≥ 2: un nivel por arco
            V_ant, V = V, np.zeros((N, smax + 1))
            for s in range(s_lo, smax + 1):
                V[:, s] = paso(s, V_ant, self._ev_dia(cache, dow0, dia_hoy, dia_de(s), indice))
        return V

    def actuar(self, obs: Observacion) -> int:
        if obs.indice is None:
            raise ValueError("B2P necesita Observacion.indice (el índice de mercado de hoy)")
        i, h, dow0 = obs.nodo, obs.casa, obs.dia_semana_salida
        acts = np.flatnonzero(obs.mascara)
        r = valor_inmediato(obs, self._dist, self._diesel)[acts]
        k_sig = None if obs.k_restante is None else obs.k_restante - 1
        s_a = np.minimum(np.floor((obs.tiempo + self._T[i, acts]) / self._dt + 1e-9).astype(np.int64), self._smax)
        sigue = acts != h
        if sigue.any():
            dia_hoy = int(np.floor(obs.tiempo / self._dt + 1e-9) * self._dt / _DIA_HORAS + 1e-9)
            V = self._valores(h, dow0, dia_hoy, np.asarray(obs.indice, dtype=float), int(s_a[sigue].min()), k_sig)
        mejor, mejor_a = -np.inf, -1
        for a, ra, sa, cont in zip(acts, r, s_a, sigue):
            v = float(ra) + (float(V[a, sa]) if cont else 0.0)
            if v > mejor:
                mejor, mejor_a = v, int(a)
        return mejor_a


class PoliticaB3(_Cotiza):
    """B3: un paso sobre la tabla V de UB-DP con los dos intentos (especificacion_fase0_ruteo_pricing.md §7.3).

    Con el tablero observado del nodo i (precios por nivel, m, opciones) y G_jk = r_jk + V(j, t + τ_ij, k − 1)
    (V(casa) = 0), P_k = (1 − u_k)^m:

        destino = argmax_j  max_k [ P_k G_jk(t) + (1 − P_k) W_j ]       (lanes cotizables)
                            R0_j + V(j, t + τ_ij, k − 1)                  (lanes que solo se recorren VACÍAS)
        W_j     = max( E(t'), max_{j' ≠ j} max_k [ P_k G_j'k(t') + (1 − P_k) E(t') ] ),  t' = t + COTIZACION_HORAS
        E(t)    = volver a casa vacío (en la primera salida desde casa: el mejor VACÍO a cualquier destino)

    Si ya se perdió una cotización en esta visita, queda un solo intento: W_j = E(t'). El precio lo pone la
    regla fija (REGLA_PRECIO_BASELINES = R_MIOPE, igual que los demás baselines). Empates por el índice menor.
    """
    nombre = "B3"

    def __init__(self, tabla: TablaDP, distance_arr, diesel_arr):
        self.tabla = tabla
        self._dist, self._diesel = distance_arr, diesel_arr

    def _cont(self, obs, acts, t, k_sig):
        """V(j, t + τ_ij, k − 1) de los destinos `acts` (0 para casa)."""
        if k_sig == 0:
            return np.zeros(len(acts))
        cont = self.tabla.valor_continuacion(obs.dia_semana_salida, obs.casa, acts,
                                             t + self.tabla.T[obs.nodo, acts], k_sig)
        return np.where(acts == obs.casa, 0.0, cont)

    def actuar(self, obs: Observacion) -> int:
        i, h = obs.nodo, obs.casa
        acts = np.flatnonzero(obs.mascara)
        k_sig = None if obs.k_restante is None else obs.k_restante - 1
        r_k = lane_reward(obs.precios[acts], 1, self._dist[i, acts, None], self._diesel[i, acts, None])   # (A, K)
        r0 = lane_reward(0.0, 0, self._dist[i, acts], self._diesel[i, acts])                             # (A,)
        P = prob_ganar(U_NIVELES_ARR, obs.m)                                                              # (K,)
        cotiza = obs.opciones[acts, :N_NIVELES].any(axis=1)
        t, t2 = obs.tiempo, obs.tiempo + COTIZACION_HORAS
        c1, c2 = self._cont(obs, acts, t, k_sig), self._cont(obs, acts, t2, k_sig)

        def vacio(c):
            if i == h:                                     # primera salida: VACÍO a cualquier destino factible
                return float(np.max(r0 + c))
            casa = np.flatnonzero(acts == h)
            return float(r0[casa[0]] + c[casa[0]]) if len(casa) else float(lane_reward(0.0, 0, self._dist[i, h],
                                                                                         self._diesel[i, h]))
        E2 = vacio(c2)
        if obs.n_rechazos + 1 >= COTIZACIONES_MAX_POR_NODO:
            W = np.full(len(acts), E2)                     # este es el último intento
        else:
            A2 = np.where(cotiza, (P * (r_k + c2[:, None]) + (1 - P) * E2).max(axis=1), -np.inf)
            orden = np.argsort(-A2, kind="stable")
            mejor, segundo = A2[orden[0]], (A2[orden[1]] if len(acts) > 1 else -np.inf)
            W = np.maximum(np.where(np.arange(len(acts)) == orden[0], segundo, mejor), E2)
        valor = np.where(cotiza, (P * (r_k + c1[:, None]) + (1 - P) * W[:, None]).max(axis=1), r0 + c1)
        return int(acts[int(np.argmax(valor))])        # argmax devuelve el primero ⇒ índice menor
