"""
mercado_sintetico.py
====================
Mercado mínimo para tests sintéticos del entorno con cotización: la misma interfaz que usa
VectorRoutingEnv de MarketProcess (`tablero_por_mundo`, `tablero_real`, `indice_mundo_lote`,
`indice_historico`, `indice_std`), sin datos reales.

Cada lane i→j tiene `loads[i, j]` cargas y precios por nivel `precio_base[i, j] · (1 + U_NIVELES)`; el
competidor u_comp es función pura de (semilla, nodo, día). Con `m` competidores efectivos fijos.
"""

import numpy as np

from market_process import Tablero
from pricing import U_NIVELES_ARR, percentil_competidor


class MercadoSintetico:
    def __init__(self, loads, precio_base, m: float = 1.0, indice_std: float = 0.1, num_dias: int = 30):
        self.loads = np.asarray(loads, dtype=np.int64)              # (N, N)
        self.precio_base = np.asarray(precio_base, dtype=float)     # (N, N) USD/milla
        self.num_nodes = self.loads.shape[0]
        self.m = float(m)
        self.indice_std = float(indice_std)
        self._indice = np.random.RandomState(1).normal(0.0, indice_std, size=(num_dias, self.num_nodes))

    def _u(self, semilla, nodo, dia) -> np.ndarray:
        return np.random.Generator(np.random.PCG64([int(semilla), int(nodo), int(dia), 7])).random(self.num_nodes)

    def tablero_por_mundo(self, nodos, dias, semillas) -> Tablero:
        nodos = np.asarray(nodos, dtype=np.int64)
        loads = self.loads[nodos]                                                       # (B, N)
        precios = self.precio_base[nodos][..., None] * (1.0 + U_NIVELES_ARR)            # (B, N, K)
        U = np.stack([self._u(s, n, d) for s, n, d in zip(semillas, nodos, dias)])
        u_comp = percentil_competidor(U, self.m)
        p_comp = self.precio_base[nodos] * (1.0 + u_comp)
        m = np.full(len(nodos), self.m)
        return Tablero(loads, precios, u_comp, p_comp, m, p_comp.astype(np.float32))

    tablero_real = tablero_por_mundo

    def indice_mundo_lote(self, semillas, dias) -> np.ndarray:
        return self._indice[np.minimum(np.asarray(dias), len(self._indice) - 1)]

    def indice_historico(self, dia_ini: int = 0, dia_fin: int = None) -> np.ndarray:
        return self._indice[dia_ini:dia_fin]
