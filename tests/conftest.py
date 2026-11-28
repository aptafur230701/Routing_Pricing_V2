"""
conftest.py
===========
Piezas comunes de los tests (todo en tensores, en config.DEVICE: CPU en el computador, GPU en el servidor).

  · `d10`: datos reales de 10 nodos (matrices, MarketProcess, estadísticas por lane, agente chico al azar).
  · `mundos_sim`: mundos simulados del proceso sorteados con una semilla.
  · `MercadoSintetico`: mercado mínimo con la interfaz de MarketProcess que usa el entorno (`tensores`, `indice_std`),
    para escenarios controlados (tiempos de 1 h, una carga por lane, competidor conocido).
  · `accion_al_azar`, `primera_opcion`: (destino, opción) entre las opciones válidas de cada fila.
"""

import os
import sys

import torch  # noqa: F401  (antes que pandas: en Windows, pandas primero rompe la carga de las DLL de torch)
import numpy as np
import pytest

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if RAIZ not in sys.path:
    sys.path.insert(0, RAIZ)

from config import DEVICE, TRAIN_DAYS, MARKET_DIAS_EPISODIO, COTIZACIONES_MAX_POR_NODO  # noqa: E402
from market_process import MarketProcess, Mundos, asegurar_parametros  # noqa: E402
from pricing import N_NIVELES, U_NIVELES_ARR  # noqa: E402
from problem_data import load_matrices  # noqa: E402

N10 = 10


def agente_chico(n, semilla=0):
    from model.agent import AMRoutingAgent
    torch.manual_seed(semilla)
    return AMRoutingAgent(num_nodes=n, d_h=32, n_heads=4, n_layers=1, d_ff=64, device=DEVICE).eval()


class Datos:
    """Datos reales de n nodos."""

    def __init__(self, n):
        asegurar_parametros()
        self.n = n
        (self.tm, self.rate, self.loads, self.dist, self.diesel, self.ltr, self.avail, self.p95) = load_matrices(n)
        self.T = np.asarray(self.tm, dtype=float)
        self.market = MarketProcess(n)
        self.est = self.market.estadisticas_dow(self.dist, self.diesel)

    def env(self, max_duration=None):
        from routing_env import VectorRoutingEnv
        kw = {} if max_duration is None else dict(max_duration=max_duration)
        return VectorRoutingEnv(self.tm, self.dist, self.diesel, self.market, device=DEVICE, **kw)

    def entradas(self, env):
        from greedy_rollout import EntradasAgente
        return EntradasAgente.crear(env, self.avail, self.est, self.p95)


@pytest.fixture(scope="session")
def d10():
    return Datos(N10)


@pytest.fixture(scope="session")
def tabla10(d10):
    """Tabla V de UB-DP de 10 nodos con S = 50 (la parte más lenta de los tests en CPU: se calcula una vez)."""
    from baselines.dp_relajada import calcular_tabla
    from config import K_MAX
    return calcular_tabla(d10.market, d10.tm, d10.dist, d10.diesel, K=K_MAX, S=50, device=DEVICE)


def generador(semilla):
    return torch.Generator(device=DEVICE).manual_seed(int(semilla))


def mundos_sim(market, B, semilla=0, dia0=None, replay=False):
    """B mundos (simulados, o replay de los días de prueba) con días de salida al azar (o `dia0`)."""
    rng = np.random.default_rng(semilla)
    if replay:
        dia_max = market.num_dias_total - 1
        dia0 = TRAIN_DAYS + rng.integers(0, dia_max - TRAIN_DAYS - 2, size=B) if dia0 is None else np.asarray(dia0)
        return market.mundos_replay(dia0, MARKET_DIAS_EPISODIO, dia_max, generador(semilla))
    dia0 = rng.integers(0, TRAIN_DAYS - MARKET_DIAS_EPISODIO, size=B) if dia0 is None else np.asarray(dia0)
    return market.sortear_mundos(dia0, MARKET_DIAS_EPISODIO, TRAIN_DAYS - 1, generador(semilla))


def accion_al_azar(env, gen=None):
    """(destino, opción) uniforme entre las opciones válidas de cada fila (filas sin opciones: (0, 0))."""
    op = env.opciones_validas.reshape(env.B, -1).float()
    vacias = op.sum(dim=1) == 0
    op[vacias, 0] = 1.0
    idx = torch.multinomial(op, 1, generator=gen).squeeze(1)
    return idx // (N_NIVELES + 1), idx % (N_NIVELES + 1)


def primera_opcion(env):
    """(destino, opción) de la primera opción válida de cada fila."""
    idx = env.opciones_validas.reshape(env.B, -1).float().argmax(dim=1)
    return idx // (N_NIVELES + 1), idx % (N_NIVELES + 1)


def correr_al_azar(env, mundos, nodos, k=None, semilla=0, al_paso=None):
    """Episodios completos con acciones válidas al azar. al_paso(env, a, o, activo) se llama antes de cada step()."""
    env.reset(mundos, nodos, k)
    gen = torch.Generator(device=env.device).manual_seed(semilla)
    activo = env.active.clone()
    for _ in range(env.max_steps):
        if not bool(activo.any()):
            break
        a, o = accion_al_azar(env, gen)
        if al_paso is not None:
            al_paso(env, a, o, activo)
        _, term, _ = env.step(a, o)
        activo &= ~term
    return activo


# ── Mercado sintético ────────────────────────────────────────────────────────

class MercadoSintetico:
    """Mercado mínimo para el entorno: `num_dias` días iguales; la lane i→j tiene precios por nivel
    precio_base[i, j] · (1 + U_NIVELES) y m competidores efectivos. El mundo (cargas y competidores) se arma con
    `mundos`: con u_min el competidor nunca cotiza por debajo de ese percentil (u_min > 0.1: p10 gana siempre)."""

    def __init__(self, precio_base, m: float = 1.0, num_dias: int = 30, indice_std: float = 0.1):
        self.precio_base = np.asarray(precio_base, dtype=float)
        self.num_nodes = N = self.precio_base.shape[0]
        self.num_dias_total = num_dias
        self.m, self.indice_std = float(m), float(indice_std)
        self._indice = np.random.RandomState(1).normal(0.0, indice_std, size=(num_dias, N))

    def tensores(self, device):
        D, N = self.num_dias_total, self.num_nodes
        dev = torch.device(device)
        precios = self.precio_base[None, :, :, None] * (1.0 + U_NIVELES_ARR)                   # (1, N, N, K)
        cuant = self.precio_base[..., None] * np.array([1.0, 1.25, 1.5, 1.75, 1.9, 2.0])       # (N, N, 6)
        f64 = dict(dtype=torch.float64, device=dev)
        return dict(
            precios=torch.as_tensor(np.broadcast_to(precios, (D, N, N, N_NIVELES)).copy(), **f64),
            cuant=torch.as_tensor(np.broadcast_to(cuant, (D, N, N, 6)).copy(), **f64),
            m=torch.full((D, N), self.m, **f64),
            indice=torch.as_tensor(self._indice, **f64),
            ltr=torch.full((D, N), 10.0, dtype=torch.float32, device=dev),
            dow=torch.arange(D, device=dev) % 7,
        )

    def mundos(self, loads, B, semilla=0, u_min=0.0):
        """B mundos con las cargas (N, N) de todos los días y competidores u ∈ [u_min, 1)."""
        N, C, D = self.num_nodes, COTIZACIONES_MAX_POR_NODO, self.num_dias_total
        g = generador(semilla)
        L = torch.as_tensor(np.asarray(loads), dtype=torch.int32, device=DEVICE).expand(B, D, N, N).contiguous()
        u = u_min + (1.0 - u_min) * torch.rand((B, D, N, N, C), generator=g, device=DEVICE)
        z = torch.zeros(B, dtype=torch.long, device=DEVICE)
        return Mundos(loads=L, u=u, delta=torch.zeros(B, dtype=torch.float64, device=DEVICE), dia0=z, dia_max=D - 1)


def escenario_sintetico(N, precio=3.0, dist=100.0, m=1.0):
    """Tiempos de 1 h, distancias y diésel constantes, 5 cargas por lane (sin self-loop)."""
    T = np.ones((N, N))
    np.fill_diagonal(T, 0.0)
    D = np.full((N, N), dist)
    np.fill_diagonal(D, 0.0)
    loads = np.full((N, N), 5)
    np.fill_diagonal(loads, 0)
    return dict(T=T, dist=D, diesel=np.full((N, N), 3.0), loads=loads,
                mercado=MercadoSintetico(np.full((N, N), precio), m=m))
