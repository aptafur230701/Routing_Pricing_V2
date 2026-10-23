"""
baselines.py
============
Políticas baseline no anticipativas B1 y B2 (baselines_drl_ruteo_estocastico.md) y B2P (B2 con
pronósticos del índice de mercado).
B3 (política sobre la tabla V de la DP) vive en dp_relajada.py.

Todas se ejecutan paso a paso en el entorno con la misma interfaz que el DRL: en cada
nodo reciben una `Observacion` con la máscara, la fila observada de cargas y tarifas, el
nodo actual, la hora y k restante. Los parámetros del proceso (mu, s_d,
phi y la tarifa media por lane) se pasan al construir la política. Ninguna recibe la
semilla del mundo evaluado ni llama a `sample_world`.
"""

from dataclasses import dataclass

import numpy as np

from config import (
    MAX_DURATION, MPG, MARGINAL_COST_SIN_DIESEL, UBDP_DT_HORAS, MARKET_DIAS_EPISODIO,
)
from problem_data import lane_reward

_DIA_HORAS = 14.0   # un día de mercado = 14 h (igual que RoutingEnv)


@dataclass
class Observacion:
    """Lo único que una política puede leer en un nodo.

    nodo, casa        : nodo actual y nodo de inicio (donde se cierra el ciclo).
    tiempo            : horas transcurridas desde la salida.
    k_restante        : arcos restantes incluido el retorno (None = caso libre).
    mascara           : (N,) int8, acciones válidas del entorno.
    loads_row, rate_row : (N,) cargas y tarifa por milla observadas al llegar al nodo.
    dia_semana_salida : día de la semana (lunes = 0) del día de salida.
    """
    nodo: int
    casa: int
    tiempo: float
    k_restante: object
    mascara: np.ndarray
    loads_row: np.ndarray
    rate_row: np.ndarray
    dia_semana_salida: int
    indice: np.ndarray = None


def calcular_ev_esperado(market, distance_arr, diesel_arr) -> np.ndarray:
    """Recompensa esperada por lane y día de la semana, (7, N, N).

        E[r_jl] = P(N_jl ≥ 1) · q̄_jl · dist_jl − c_jl,   P(N ≥ 1) = 1 − (φ / (φ + m_jl))^φ

    con m = mu_jl · s_d (d = día de la semana de llegada) y q̄ = media_lane, la tarifa media de la
    lane. Es el valor que usa B2 para los arcos futuros y el que alimenta las features de
    continuación del DRL; se calcula una vez por número de nodos.
    """
    p = market.params
    phi = p.phi[p.grupo]                                              # (N, N)
    q_medio = p.media_lane                                            # (N, N) tarifa media de la lane
    costo = distance_arr * (diesel_arr / MPG) + distance_arr * MARGINAL_COST_SIN_DIESEL
    m = p.mu[None, :, :] * p.s[:, None, None]                         # (7, N, N)
    prob = 1.0 - (phi[None] / (phi[None] + m)) ** phi[None]           # P(N ≥ 1); m = 0 ⇒ 0
    return prob * q_medio[None] * distance_arr[None] - costo[None]    # (7, N, N)


@dataclass(frozen=True)
class EvEsperado:
    """Valor esperado por lane con pronóstico del índice de mercado (tabla compartida por B2P y el encoder).

    Para un arco j→l que sale de j un día de mercado `dow` de la semana, h días de mercado después de hoy:

        ev_ajustado(h, dow)[j, l] = ev[dow, j, l] + índice(j, hoy) · q[h, dow, j, l]

    ev (7, N, N)      : calcular_ev_esperado (tarifa media de la lane; el valor con índice cero)
    q  (H+1, 7, N, N) : b_h · P(N_jl ≥ 1) · dist_jl, con b_h = f + (1 − f) · ρ^h
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
    p = market.params
    ev = calcular_ev_esperado(market, distance_arr, diesel_arr)
    b = p.frac_var_lento + (1.0 - p.frac_var_lento) * p.rho ** np.arange(MARKET_DIAS_EPISODIO)   # h = 0 … H
    phi = p.phi[p.grupo]
    m = p.mu[None, :, :] * p.s[:, None, None]
    prob = 1.0 - (phi[None] / (phi[None] + m)) ** phi[None]                                        # P(N ≥ 1) (7, N, N)
    q = b[:, None, None, None] * (prob * distance_arr[None])[None]                                 # (H+1, 7, N, N)
    return EvEsperado(ev=ev, q=q, b=b)


class PoliticaB1:
    """B1, greedy miope (PFA): arco válido de mayor `lane_reward` según la fila observada.

    Empates por el índice menor.
    """
    nombre = "B1"

    def __init__(self, distance_arr, diesel_arr):
        self._dist, self._diesel = distance_arr, diesel_arr

    def actuar(self, obs: Observacion) -> int:
        i = obs.nodo
        acts = np.flatnonzero(obs.mascara)
        r = lane_reward(obs.rate_row[acts], obs.loads_row[acts], self._dist[i, acts], self._diesel[i, acts])
        return int(acts[int(np.argmax(r))])      # argmax devuelve el primero ⇒ índice menor


class PoliticaB2:
    """B2, reoptimización determinista con valores esperados (CFA); con K = 5 y con K libre.

    En cada nodo resuelve de forma exacta la mejor ruta restante y ejecuta solo el primer
    arco. El arco actual usa la recompensa real de la fila observada; los arcos futuros
    (j, l) usan la esperada

        E[r_jl] = P(N_jl ≥ 1) · q̄_jl · dist_jl − c_jl,   P(N ≥ 1) = 1 − (φ / (φ + m_jl))^φ

    con m = mu_jl · s_d del día de llegada estimado y q̄ = media_lane, la tarifa media de la lane.

    El camión puede volver a pasar por un nodo, así que el estado del futuro es (j, franja de
    tiempo, k restante) y la mejor continuación es una recursión memoizada por (casa, día de
    salida). Las franjas son de UBDP_DT_HORAS y los tiempos de viaje se redondean hacia abajo, como
    en dp_relajada.py; la regla de retorno factible es la de la máscara (volver a casa dentro de
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
        r = lane_reward(obs.rate_row[acts], obs.loads_row[acts], self._dist[i, acts], self._diesel[i, acts])
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
        p = market.params
        self.f, self.rho = float(p.frac_var_lento), float(p.rho)
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
        r = lane_reward(obs.rate_row[acts], obs.loads_row[acts], self._dist[i, acts], self._diesel[i, acts])
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
