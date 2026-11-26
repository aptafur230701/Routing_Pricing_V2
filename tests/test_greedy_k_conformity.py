"""
test_greedy_k_conformity.py
===========================
El rollout greedy paso a paso sobre el entorno con cotización (greedy_rollout) honra el
presupuesto de cardinalidad K (≤ K arcos, incluido el retorno), tanto en el caso restringido
como en el libre, y reporta el margen como la suma del reward de cada lane con el precio cotizado.

Usa un escenario sintético (tests/mercado_sintetico.py) con tiempos de viaje de 1 h y max_duration
holgado, para que K sea la restricción BINDING — con los datos reales a 10 nodos, MAX_DURATION ya
limita las rutas a 2-3 arcos independientemente de K. Un agente de pesos aleatorios es suficiente:
solo se verifica conformidad estructural (impuesta por máscara), no calidad de ruta.
"""

import numpy as np
import pytest
import torch

from config import DEVICE, K_LEVELS, REWARD_SCALE_FACTOR, TIME_VIOLATION_PENALTY
from model.agent import AMRoutingAgent
from greedy_rollout import greedy_rollout
from problem_data import lane_reward
from baselines.politicas import EvEsperado
from routing_env import VectorRoutingEnv
from pricing import U_NIVELES_ARR
from mercado_sintetico import MercadoSintetico

N = 9   # el camión puede revisitar nodos, así que con MAX_DURATION_SYN holgado cualquier K de K_LEVELS
        # es físicamente alcanzable.
MAX_DURATION_SYN = 30.0   # con time_matrix=1.0/arco caben ~30 arcos más los rechazos — K es el binding


@pytest.fixture(scope="module")
def synthetic():
    time_matrix  = np.ones((N, N))
    np.fill_diagonal(time_matrix, 0.0)
    distance_arr = np.ones((N, N)) * 10.0
    np.fill_diagonal(distance_arr, 0.0)
    diesel_arr   = np.ones((N, N)) * 3.0
    loads        = np.full((N, N), 5)
    np.fill_diagonal(loads, 0)
    precio_base  = np.full((N, N), 5.0)
    return dict(
        time_matrix=time_matrix, distance_arr=distance_arr, diesel_arr=diesel_arr,
        precio_base=precio_base,
        mercado=MercadoSintetico(loads, precio_base, m=2.0, u_min=0.15),   # p10 gana siempre aquí
        ltr_stack=np.ones((N, 3), dtype=np.float32) * 10.0,
        ev_esperado=EvEsperado(                                                    # features de continuación
            ev=np.random.RandomState(0).uniform(-50, 200, size=(7, N, N)),
            q=np.random.RandomState(2).uniform(0, 50, size=(6, 7, N, N)), b=np.linspace(1.0, 0.5, 6),
            ev_casa=np.random.RandomState(3).uniform(-50, 200, size=(7, N, N)),
            q_casa=np.random.RandomState(4).uniform(0, 50, size=(6, 7, N, N))),
    )


@pytest.fixture
def env(synthetic):
    return VectorRoutingEnv(
        synthetic["time_matrix"], N, synthetic["distance_arr"], synthetic["diesel_arr"], 3,
        max_duration=MAX_DURATION_SYN, market_process=synthetic["mercado"],
    )


@pytest.fixture(scope="module")
def agent():
    torch.manual_seed(0)
    return AMRoutingAgent(num_nodes=N, d_h=32, n_heads=4, n_layers=1, d_ff=64, device=DEVICE)


def _run(agent, env, synthetic, k_max):
    """Un episodio por nodo de inicio, todos con el mismo presupuesto k_max."""
    starts = np.arange(N, dtype=np.int64)
    return greedy_rollout(
        agent, env, starts, np.zeros(N, dtype=np.int64),
        synthetic["time_matrix"], synthetic["distance_arr"],
        synthetic["ltr_stack"],
        reward_global_p95=1.0, num_nodes=N,
        k_max_arr=[k_max] * N, max_duration=MAX_DURATION_SYN,
        world_seeds=np.arange(N) + 100, ev_esperado=synthetic["ev_esperado"],
    )


def _never_return_voluntarily(start_nodes):
    """Greedy que cotiza el nivel 0 (p10; con u_min = 0.15 gana siempre) en una lane que no vaya a casa mientras
    haya una, y si no, vuelve a casa: obliga a que sea la máscara de K la que cierre el ciclo."""
    starts = torch.as_tensor(start_nodes, device=DEVICE)

    def pick(h_t, embeddings, mask_t, opciones_t, cot_t):
        allowed = mask_t.bool().clone()
        rows = torch.arange(allowed.shape[0], device=allowed.device)
        cotizable = opciones_t[:, :, :-1].any(dim=-1) & allowed
        cotizable[rows, starts] = False
        casa = torch.zeros_like(allowed)
        casa[rows, starts] = True
        use = cotizable.any(dim=1, keepdim=True)
        j = torch.where(use, cotizable, casa & allowed).float().argmax(dim=1)
        o = opciones_t[rows, j].float().argmax(dim=1)                 # nivel 0 si se puede cotizar; si no, VACÍO
        return j, o

    return pick


def test_greedy_rollout_respects_k(agent, env, synthetic):
    for k in K_LEVELS:
        res = _run(agent, env, synthetic, k)
        for start, route, ok in zip(range(N), res.routes, res.valid):
            if ok:
                assert len(route) - 1 <= k, f"K={k} start={start}: ruta con {len(route) - 1} arcos > K"


def test_greedy_rollout_k_is_binding_when_policy_never_returns_early(agent, env, synthetic, monkeypatch):
    monkeypatch.setattr(agent.decoder, "greedy_cotizacion", _never_return_voluntarily(np.arange(N)))
    for k in K_LEVELS:
        res = _run(agent, env, synthetic, k)
        assert res.valid.all(), f"K={k}: rutas inválidas {np.where(~res.valid)[0]}"
        lens = [len(r) - 1 for r in res.routes]
        assert lens == [k] * N, f"K={k}: el retorno forzado debe cerrar en exactamente K arcos, got {lens}"
        assert (res.n_ganadas == res.n_cotizaciones).all()          # p10 nunca se pierde con u_min = 0.15


def test_greedy_rollout_free_case_closes_cycle(agent, env, synthetic):
    """Con pesos al azar el agente puede esperar en casa hasta quedarse sin jugadas: ese ciclo falla (la
    penalización); los demás cierran en casa."""
    res = _run(agent, env, synthetic, None)
    assert res.valid.any()
    for start, route, ok, r, raw in zip(range(N), res.routes, res.valid, res.env_reward, res.raw_reward):
        if ok:
            assert route[0] == start and route[-1] == start
        else:                                                       # margen recorrido + la penalización
            assert r == pytest.approx(raw / REWARD_SCALE_FACTOR + TIME_VIOLATION_PENALTY, abs=1e-4)


def test_greedy_rollout_raw_reward_is_sum_of_lane_rewards(agent, env, synthetic, monkeypatch):
    """Con el nivel 0 (p10: precio = precio_base · 1.1, gana siempre aquí), raw_reward es la suma de lane_reward
    de las lanes recorridas a ese precio, sin escala ni bonos terminales."""
    monkeypatch.setattr(agent.decoder, "greedy_cotizacion", _never_return_voluntarily(np.arange(N)))
    res = _run(agent, env, synthetic, 4)
    d, c, p = synthetic["distance_arr"], synthetic["diesel_arr"], synthetic["precio_base"]
    for route, raw in zip(res.routes, res.raw_reward):
        expected = sum(float(lane_reward(p[a, b] * (1.0 + U_NIVELES_ARR[0]), 1, d[a, b], c[a, b]))
                       for a, b in zip(route[:-1], route[1:]))
        assert raw == pytest.approx(expected)


def test_greedy_env_reward_es_margen_sin_bonos(agent, env, synthetic, monkeypatch):
    """Sin shaping, env_reward (el que alimenta el self-critical) es margen / REWARD_SCALE_FACTOR
    en las rutas válidas: sin bono de retorno ni términos de arcos previos / time_util."""
    import routing_env
    monkeypatch.setattr(routing_env, "USE_REWARD_SHAPING", False)
    for k in (*K_LEVELS, None):
        res = _run(agent, env, synthetic, k)
        assert res.valid.any()
        v = res.valid
        assert res.env_reward[v] == pytest.approx(res.raw_reward[v] / REWARD_SCALE_FACTOR, abs=1e-5)
