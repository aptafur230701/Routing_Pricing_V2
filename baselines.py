"""
baselines.py
============
Políticas baseline no anticipativas B1 y B2 (baselines_drl_ruteo_estocastico.md).
B3 (política sobre la tabla V de la DP) vive en dp_relajada.py.

Todas se ejecutan paso a paso en el entorno con la misma interfaz que el DRL: en cada
nodo reciben una `Observacion` con la máscara, la fila observada de cargas y tarifas, el
nodo actual, la hora y k restante. Los parámetros del proceso (mu, s_d,
phi y la tarifa media por lane) se pasan al construir la política. Ninguna recibe la
semilla del mundo evaluado ni llama a `sample_world`.
"""

from dataclasses import dataclass

import numpy as np

from config import MAX_DURATION, MPG, MARGINAL_COST_SIN_DIESEL, UBDP_DT_HORAS
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
