"""
market_process.py
=================
Proceso exógeno de cargas y tarifas (fase 4 de proceso_exogeno_cargas_tarifas.md).

Cuando el agente llega al nodo i el día d, el proceso sortea para cada lane
(i, j) que sale del nodo:
  1. cuántas cargas hay: NegBin(mu_ij · s_d, phi_h)   → loads_row (N,)
  2. la tarifa por milla: remuestreo con reemplazo del histórico de la lane
                                                      → rate_row  (N,)
La tarifa se sortea en todas las lanes; el entorno solo la usa donde loads_row > 0.

El proceso NO depende de las acciones del agente. Dos modos de uso:

  · Entrenamiento : `sample_arrivals_por_mundo(nodos, días, semillas)` — un
    "mundo" por episodio. Cada sorteo es función pura de (semilla del mundo,
    nodo, día), así que (a) la salida no depende de qué decida el agente ni de
    cuántos episodios sigan activos, y (b) el rollout muestreado y el greedy del
    baseline self-critical, con la misma semilla de mundo, ven la misma
    realización. `sample_arrivals(nodos, días, rng)` es la versión por lotes con
    un único Generator (más rápida, para uso fuera de los entornos).
  · Evaluación / baselines : `sample_world(semilla, num_días, día_inicial)`
    genera un mundo completo [D, N, N]; es el mismo mundo que ve el entorno con
    esa semilla de mundo.

Toda la aleatoriedad sale de numpy Generators propios (nunca del estado global).
"""

import os
from dataclasses import dataclass

import numpy as np

from fit_market_process import dia_de_semana, params_path, main as _ajustar


@dataclass
class MarketParams:
    """Parámetros del proceso, ya recortados a num_nodes."""
    s:        np.ndarray   # (7,)        factor del día de la semana (lunes = 0), promedio 1
    mu:       np.ndarray   # (N, N)      tasa diaria base de la lane
    phi:      np.ndarray   # (G,)        dispersión por grupo de volumen
    grupo:    np.ndarray   # (N, N) int  grupo de volumen h de cada lane
    tarifas:  np.ndarray   # (D, N, N)   tarifa por milla histórica de cada lane
    fecha_inicio: str


class MarketProcess:
    """Proceso exógeno de cargas y tarifas estimado del histórico.

    Parámetros
    ----------
    num_nodes : se usan los primeros `num_nodes` nodos (igual que load_matrices).
                Los parámetros se estiman con los 194 y se recortan aquí.
    path      : ruta del .npz (por defecto datos/market_params.npz).
    """

    def __init__(self, num_nodes: int, path: str = None):
        path = path or params_path()
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"{path} no existe. Corre `python fit_market_process.py` primero."
            )
        with np.load(path) as z:
            n_total = z["mu"].shape[0]
            if num_nodes > n_total:
                raise ValueError(f"num_nodes={num_nodes} excede los {n_total} nodos ajustados")
            n = num_nodes
            self.params = MarketParams(
                s=z["s"].astype(np.float64),
                mu=z["mu"][:n, :n].astype(np.float64),
                phi=z["phi"].astype(np.float64),
                grupo=z["grupo"][:n, :n].astype(np.int64),
                tarifas=np.ascontiguousarray(z["tarifas"][:, :n, :n]),
                fecha_inicio=str(z["fecha_inicio"]),
            )
        self.num_nodes = num_nodes
        self.num_dias_historico = self.params.tarifas.shape[0]
        # phi de cada lane según su grupo de volumen (N, N).
        self._phi_lane = self.params.phi[self.params.grupo]
        self._cols = np.arange(num_nodes)

    # ── utilidades ───────────────────────────────────────────────────────────

    def dia_semana(self, dias) -> np.ndarray:
        """Día de la semana (lunes = 0) de los índices de día `dias`."""
        return dia_de_semana(dias, self.params.fecha_inicio)

    # ── Llegada a un nodo (una fila) ─────────────────────────────────────────

    def _sortear_fila(self, i: int, dow: int, rng: np.random.Generator) -> tuple:
        """llegada_a_nodo del documento: (loads_row, rate_row) del nodo i en el día de semana dow."""
        N = self.num_nodes
        m   = self.params.mu[i] * self.params.s[dow]                  # (N,) media por lane
        phi = self._phi_lane[i]                                       # (N,) dispersión por lane
        # m = 0 (lane sin cargas en el histórico) ⇒ p = 1 ⇒ siempre 0 cargas.
        loads_row = rng.negative_binomial(phi, phi / (phi + m))       # (N,)
        # Tarifa: un día del histórico elegido al azar para cada lane.
        dia_hist = rng.integers(0, self.num_dias_historico, size=N)
        rate_row = self.params.tarifas[dia_hist, i, self._cols]       # (N,)
        return loads_row, rate_row

    def llegada_a_nodo(self, i: int, dia: int, rng: np.random.Generator) -> tuple:
        """Devuelve (loads_row, rate_row) para las lanes que salen del nodo i el día `dia`."""
        return self._sortear_fila(int(i), int(self.dia_semana(dia)), rng)

    # ── Modo entrenamiento ───────────────────────────────────────────────────

    def sample_arrivals(self, nodos, dias, rng: np.random.Generator) -> tuple:
        """Llegadas por lotes con un único Generator (sin ciclos sobre B).

        nodos, dias : (B,) nodo de llegada y día (índice) de llegada.
        Devuelve loads (B, N) int64 y rates (B, N) float32.
        El resultado depende del orden y tamaño del lote (consume `rng`); para
        sorteos independientes de las decisiones del agente usa
        `sample_arrivals_por_mundo`.
        """
        nodos = np.asarray(nodos, dtype=np.int64)
        dow   = self.dia_semana(dias)                                       # (B,)
        m     = self.params.mu[nodos] * self.params.s[dow][:, None]         # (B, N)
        phi   = self._phi_lane[nodos]                                       # (B, N)
        loads = rng.negative_binomial(phi, phi / (phi + m))                 # (B, N)
        dia_hist = rng.integers(0, self.num_dias_historico, size=m.shape)   # (B, N)
        rates = self.params.tarifas[dia_hist, nodos[:, None], self._cols[None, :]]
        return loads.astype(np.int64), rates

    def sample_arrivals_por_mundo(self, nodos, dias, semillas) -> tuple:
        """Llegadas por lotes, cada fila determinista dada (semilla de mundo, nodo, día).

        nodos, dias, semillas : (B,). `semillas` es la semilla del mundo de cada
        episodio (entero ≥ 0). Dos filas con la misma terna ven exactamente el
        mismo sorteo, sin importar el resto del lote, el orden de las llamadas ni
        las acciones del agente.
        Devuelve loads (B, N) int64 y rates (B, N) float32.
        """
        nodos    = np.asarray(nodos,    dtype=np.int64)
        dias     = np.asarray(dias,     dtype=np.int64)
        semillas = np.asarray(semillas, dtype=np.int64)
        dow = self.dia_semana(dias)
        B, N = len(nodos), self.num_nodes
        loads = np.empty((B, N), dtype=np.int64)
        rates = np.empty((B, N), dtype=self.params.tarifas.dtype)
        for b in range(B):
            rng = np.random.Generator(np.random.PCG64(
                [int(semillas[b]), int(nodos[b]), int(dias[b])]
            ))
            loads[b], rates[b] = self._sortear_fila(int(nodos[b]), int(dow[b]), rng)
        return loads, rates

    # ── Modo evaluación / baselines ──────────────────────────────────────────

    def sample_world(self, semilla: int, num_dias: int, dia_inicial: int = 0) -> tuple:
        """Mundo completo: loads_sim y rate_sim de forma [D, N, N], determinista dada la semilla.

        loads_sim[d, i] y rate_sim[d, i] son la fila que ve el agente al llegar
        al nodo i el día `dia_inicial + d`: el mismo sorteo que entrega
        `sample_arrivals_por_mundo` con esa semilla de mundo. Sustituto directo
        de loads_stack / rate_stack (el día 0 del mundo es `dia_inicial`).
        """
        N = self.num_nodes
        nodos = np.arange(N)
        loads_sim = np.empty((num_dias, N, N), dtype=np.float64)
        rate_sim  = np.empty((num_dias, N, N), dtype=np.float64)
        for d in range(num_dias):
            l, r = self.sample_arrivals_por_mundo(
                nodos, np.full(N, dia_inicial + d), np.full(N, semilla),
            )
            loads_sim[d], rate_sim[d] = l, r
        return loads_sim, rate_sim


def asegurar_parametros() -> None:
    """Corre el ajuste (fit_market_process) si datos/market_params.npz no existe."""
    if not os.path.exists(params_path()):
        print("market_params.npz no existe: corriendo el ajuste del proceso exógeno...")
        _ajustar()
