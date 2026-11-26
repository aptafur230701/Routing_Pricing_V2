"""
test_k_cardinality_env.py
==========================
Criterio de aceptación obligatorio del condicionamiento en cardinalidad de ruta (K):

    Con una política ALEATORIA (destino y opción de precio al azar entre las válidas), el 100% de los
    episodios en todos los niveles de K cumplen longitud (nº de arcos recorridos) <= K. Debe ser cierto
    por construcción de la máscara, nunca por penalización de reward. Un rechazo no consume K.

Cubre el caso libre (k_max=None): k_remaining no se usa y no hay retorno forzado.
"""

import numpy as np
import pytest

from config import MAX_DURATION, K_LEVELS, K_TRAINING_LEVELS, TRAIN_DAYS
from market_process import MarketProcess, asegurar_parametros
from problem_data import load_matrices
from routing_env import VectorRoutingEnv

NUM_NODES = 10


@pytest.fixture(scope="module")
def data():
    asegurar_parametros()
    time_matrix, _, _, distance_arr, diesel_arr, *_ = load_matrices(NUM_NODES)
    return dict(time_matrix=np.asarray(time_matrix, dtype=float), distance_arr=distance_arr,
                diesel_arr=diesel_arr, mp=MarketProcess(NUM_NODES))


def _accion_al_azar(env, active, rng):
    """(destino, opción) uniforme entre las opciones válidas de cada fila activa."""
    B = len(active)
    acts, opc = np.zeros(B, dtype=np.int64), np.zeros(B, dtype=np.int64)
    for b in np.flatnonzero(active):
        pares = np.argwhere(env.opciones_validas[b])
        assert len(pares), "fila activa sin opciones válidas"
        acts[b], opc[b] = pares[rng.integers(len(pares))]
    return acts, opc


def test_vector_env_random_policy_respects_k(data):
    env = VectorRoutingEnv(data["time_matrix"], NUM_NODES, data["distance_arr"], data["diesel_arr"], TRAIN_DAYS,
                           max_duration=MAX_DURATION, market_process=data["mp"])
    rng = np.random.default_rng(1)
    B = 600
    k_max_arr = [K_TRAINING_LEVELS[i % len(K_TRAINING_LEVELS)] for i in rng.permutation(B)]
    start_nodes = rng.integers(0, NUM_NODES, size=B).astype(np.int64)
    start_days = rng.integers(0, TRAIN_DAYS, size=B).astype(np.int64)

    env.reset(start_nodes, start_days, k_max_arr=k_max_arr, world_seeds=np.arange(B) + 7)
    active = env.active.copy()                                   # sin ninguna jugada desde casa: no sale
    rechazos = 0
    for _ in range(env.max_steps):
        if not active.any():
            break
        acts, opc = _accion_al_azar(env, active, rng)
        _, terminated, _ = env.step(acts, opc)
        rechazos += int((active & env.last_cotizo & ~env.last_gano).sum())
        active = active & ~terminated

    assert not active.any(), "todos los episodios deben terminar dentro del techo defensivo"
    assert rechazos > 0, "la política aleatoria debería perder alguna cotización"
    for b in range(B):
        k_max = k_max_arr[b]
        if k_max is not None:
            assert env.step_count[b] <= k_max, f"fila {b}: K={k_max} pero {env.step_count[b]} arcos"
            assert env.k_remaining[b] == k_max - env.step_count[b]
    for lvl in K_LEVELS:
        assert any(k == lvl for k in k_max_arr)


def test_caso_libre_sin_presupuesto(data):
    env = VectorRoutingEnv(data["time_matrix"], NUM_NODES, data["distance_arr"], data["diesel_arr"], TRAIN_DAYS,
                           market_process=data["mp"])
    env.reset(np.arange(4), np.zeros(4, dtype=np.int64), world_seeds=np.arange(4))
    assert not env.k_restricted.any()
