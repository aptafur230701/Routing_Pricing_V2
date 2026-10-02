"""
test_beam_and_sampling_k_conformity.py
========================================
Fase 4 — criterio de aceptación (§8 paso 4): beam_search_dynamic y
generate_route_sampling honran el presupuesto de cardinalidad K (≤ K arcos,
incluido el retorno), tanto en el caso restringido como en el caso libre y
en la ablación k_condition_neutral.

Usa un escenario sintético (no los datos reales de 10 nodos) con tiempos de
viaje pequeños y max_duration holgado, para que K sea la restricción
BINDING — con los datos reales a 10 nodos, MAX_DURATION ya limita las rutas
a 2-3 arcos independientemente de K, lo que no ejercitaría el camino de
retorno forzado por K. Un agente de pesos aleatorios es suficiente: solo se
verifica conformidad estructural (impuesta por máscara), no calidad de ruta.
"""

import numpy as np
import pytest
import torch

from config import DEVICE, K_LEVELS
from am_agent import AMRoutingAgent

N = 9   # > max(K_LEVELS): un ciclo puede visitar hasta N-1 intermedios + retorno = N arcos,
        # así que todos los niveles de K_LEVELS son físicamente alcanzables.


@pytest.fixture(scope="module")
def synthetic_data():
    time_matrix  = np.ones((N, N))
    np.fill_diagonal(time_matrix, 0.0)
    distance_arr = np.ones((N, N)) * 10.0
    np.fill_diagonal(distance_arr, 0.0)
    diesel_arr   = np.ones((N, N)) * 3.0
    rate_stack   = np.ones((3, N, N)) * 5.0
    loads_stack  = np.ones((3, N, N)) * 5.0
    avail_prob_arr = np.ones((N, N), dtype=np.float32)   # lanes siempre existen
    trucks_stack = np.ones((N, 3, 3), dtype=np.float32) * 2.0
    ltr_stack    = np.ones((N, 3), dtype=np.float32) * 10.0
    max_duration = 30.0   # con time_matrix=1.0/arco, permite hasta 30 arcos — K es el binding
    return dict(
        time_matrix=time_matrix, rate_stack=rate_stack, loads_stack=loads_stack,
        distance_arr=distance_arr, diesel_arr=diesel_arr,
        avail_prob_arr=avail_prob_arr, trucks_stack=trucks_stack, ltr_stack=ltr_stack,
        max_duration=max_duration,
    )


@pytest.fixture(scope="module")
def agent():
    torch.manual_seed(0)
    return AMRoutingAgent(num_nodes=N, d_h=32, n_heads=4, n_layers=1, d_ff=64, device=DEVICE)


def _route_len(route):
    return None if route is None else len(route) - 1


# ─────────────────────────────────────────────────────────────────────────────
# beam_search_dynamic
# ─────────────────────────────────────────────────────────────────────────────

def test_beam_search_dynamic_respects_k(agent, synthetic_data):
    for k in K_LEVELS:
        lens = []
        for start in range(N):
            route, reward, t = agent.beam_search_dynamic(
                start, 0, beam_width=N, k_max=k, **synthetic_data,
            )
            n = _route_len(route)
            if n is not None:
                lens.append(n)
                assert n <= k, f"K={k} start={start}: ruta con {n} arcos > K"
        # K debe ser efectivamente binding en este escenario sintético.
        assert max(lens) == k, f"K={k}: ninguna ruta alcanzó el presupuesto (max={max(lens)})"


def test_beam_search_dynamic_free_case_unaffected(agent, synthetic_data):
    """k_max=None (default) debe seguir sin límite de K explícito."""
    route, reward, t = agent.beam_search_dynamic(0, 0, beam_width=N, **synthetic_data)
    assert route is not None


def test_beam_search_dynamic_k_condition_neutral_ablation(agent, synthetic_data):
    """Ablación: máscara real de K activa, features neutras — debe seguir
    respetando K por máscara (la ablación no relaja la máscara)."""
    for k in K_LEVELS:
        route, reward, t = agent.beam_search_dynamic(
            0, 0, beam_width=N, k_max=k, k_condition_neutral=True, **synthetic_data,
        )
        n = _route_len(route)
        if n is not None:
            assert n <= k


# ─────────────────────────────────────────────────────────────────────────────
# generate_route_sampling
# ─────────────────────────────────────────────────────────────────────────────

def test_generate_route_sampling_respects_k(agent, synthetic_data):
    for k in K_LEVELS:
        lens = []
        for trial in range(30):
            route, reward, t = agent.generate_route_sampling(
                trial % N, 0, n_samples=1, temperature=5.0, k_max=k, **synthetic_data,
            )
            n = _route_len(route)
            if n is not None:
                lens.append(n)
                assert n <= k, f"K={k} trial={trial}: ruta con {n} arcos > K"
        assert max(lens) == k, f"K={k}: ninguna muestra alcanzó el presupuesto (max={max(lens)})"


def test_generate_route_sampling_free_case_unaffected(agent, synthetic_data):
    route, reward, t = agent.generate_route_sampling(0, 0, n_samples=4, **synthetic_data)
    assert route is not None


def test_generate_route_sampling_k_condition_neutral_ablation(agent, synthetic_data):
    for k in K_LEVELS:
        route, reward, t = agent.generate_route_sampling(
            0, 0, n_samples=4, temperature=5.0, k_max=k, k_condition_neutral=True,
            **synthetic_data,
        )
        n = _route_len(route)
        if n is not None:
            assert n <= k
