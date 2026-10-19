"""
baselines.py
============
Políticas baseline no anticipativas B1 y B2 (baselines_drl_ruteo_estocastico.md).
B3 (política de la DP relajada) vive en dp_relajada.py.

Todas se ejecutan paso a paso en el entorno con la misma interfaz que el DRL: en cada
nodo reciben una `Observacion` con la máscara, la fila observada de cargas y tarifas, el
nodo actual, la hora, k restante y los visitados. Los parámetros del proceso (mu, s_d,
phi y las tarifas históricas) se pasan al construir la política. Ninguna recibe la
semilla del mundo evaluado ni llama a `sample_world`.
"""

from dataclasses import dataclass

import numpy as np

from config import MAX_DURATION, MPG, MARGINAL_COST_SIN_DIESEL
from problem_data import lane_reward

_DIA_HORAS = 14.0   # un día de mercado = 14 h (igual que RoutingEnv)


@dataclass
class Observacion:
    """Lo único que una política puede leer en un nodo.

    nodo, casa        : nodo actual y nodo de inicio (donde se cierra el ciclo).
    tiempo            : horas transcurridas desde la salida.
    k_restante        : arcos restantes incluido el retorno (None = caso libre).
    visitados         : (N,) bool, incluye el nodo de casa.
    mascara           : (N,) int8, acciones válidas del entorno.
    loads_row, rate_row : (N,) cargas y tarifa por milla observadas al llegar al nodo.
    dia_semana_salida : día de la semana (lunes = 0) del día de salida.
    """
    nodo: int
    casa: int
    tiempo: float
    k_restante: object
    visitados: np.ndarray
    mascara: np.ndarray
    loads_row: np.ndarray
    rate_row: np.ndarray
    dia_semana_salida: int


def calcular_ev_esperado(market, distance_arr, diesel_arr) -> np.ndarray:
    """Recompensa esperada por lane y día de la semana, (7, N, N).

        E[r_jl] = P(N_jl ≥ 1) · q̄_jl · dist_jl − c_jl,   P(N ≥ 1) = 1 − (φ / (φ + m_jl))^φ

    con m = mu_jl · s_d (d = día de la semana de llegada) y q̄ la tarifa media histórica de la
    lane. Es el valor que usa B2 para los arcos futuros y el que alimenta las features de
    continuación del DRL; se calcula una vez por número de nodos.
    """
    p = market.params
    phi = p.phi[p.grupo]                                              # (N, N)
    q_medio = p.tarifas.mean(axis=0).astype(float)                    # (N, N)
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
    """B2, reoptimización determinista con valores esperados (CFA); se evalúa solo con K = 5.

    En cada nodo resuelve de forma exacta la mejor ruta restante y ejecuta solo el primer
    arco. El arco actual usa la recompensa real de la fila observada; los arcos futuros
    (j, l) usan la esperada

        E[r_jl] = P(N_jl ≥ 1) · q̄_jl · dist_jl − c_jl,   P(N ≥ 1) = 1 − (φ / (φ + m_jl))^φ

    con m = mu_jl · s_d del día de llegada estimado y q̄ la tarifa media histórica de la lane.
    La búsqueda es una enumeración exacta de rutas sin revisitar, con retorno dentro de
    MAX_DURATION y respetando k (con los tiempos de viaje reales de los datos las rutas
    factibles son pocas, así que no hace falta podar por dominancia).
    """
    nombre = "B2"

    def __init__(self, market, time_matrix, distance_arr, diesel_arr, max_duration=MAX_DURATION):
        self._T = np.asarray(time_matrix, dtype=float)
        self._dist, self._diesel = distance_arr, diesel_arr
        self._max = float(max_duration)
        self.N = self._T.shape[0]
        self._ev = calcular_ev_esperado(market, distance_arr, diesel_arr)   # (7, N, N)

    def actuar(self, obs: Observacion) -> int:
        i, h = obs.nodo, obs.casa
        acts = np.flatnonzero(obs.mascara)
        r = lane_reward(obs.rate_row[acts], obs.loads_row[acts], self._dist[i, acts], self._diesel[i, acts])
        vis = np.asarray(obs.visitados, dtype=bool).copy()
        mejor, mejor_a = -np.inf, -1
        for a, ra in zip(acts, r):
            if a == h:
                v = float(ra)                                             # cerrar el ciclo ahora
            else:
                vis[a] = True
                v = float(ra) + self._continuacion(
                    int(a), obs.tiempo + self._T[i, a], vis,
                    obs.k_restante - 1, h, obs.dia_semana_salida,
                )
                vis[a] = False
            if v > mejor:
                mejor, mejor_a = v, int(a)
        return mejor_a

    def _continuacion(self, j, t, vis, k_rest, h, dow0) -> float:
        """Mejor valor esperado de seguir desde j (llegada en t, k_rest arcos restantes).

        `vis` (N,) bool se modifica y se restaura durante la recursión.
        """
        ev = self._ev[(dow0 + int(t // _DIA_HORAS)) % 7]
        mejor = ev[j, h]                                                  # cerrar el ciclo desde j
        if k_rest >= 2:
            cand = np.flatnonzero((self._T[j] + t + self._T[:, h] <= self._max) & ~vis)
            for l in cand:
                vis[l] = True
                v = ev[j, l] + self._continuacion(int(l), t + self._T[j, l], vis, k_rest - 1, h, dow0)
                vis[l] = False
                if v > mejor:
                    mejor = v
        return float(mejor)
