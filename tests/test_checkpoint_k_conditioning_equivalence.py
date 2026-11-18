"""
test_checkpoint_k_conditioning_equivalence.py
==============================================
Red de seguridad sobre el checkpoint entrenado de 10 nodos
(checkpoints/am_checkpoint_10nodes.pt).

Historia: este archivo pinneaba rutas concretas de un rollout greedy
capturadas con un checkpoint anterior. Como checkpoints/ está en
.gitignore y se reentrena, esos valores caducaban en cada reentrenamiento y el
test dejaba de decir algo útil. Además el docstring afirmaba que las rutas
pinneadas coincidían con el óptimo exacto, lo cual no era cierto en todos los
casos (p. ej. start_node=3, day=0: 690 pinneado vs 851 óptimo).

Ahora el test mide lo que realmente importa y no caduca con un reentrenamiento:

1. El checkpoint carga con load_checkpoint_compatible (compatibilidad de
   arquitectura/state_dict, incluidas las columnas de presupuesto K).
2. En el caso libre (k_max=None), con el rollout greedy paso a paso en el
   entorno (argmax por nodo, sin beam search), el DRL se compara contra el óptimo con el
   día conocido (DP hacia atrás de UB-PI sobre el replay) en los días de evaluación:
   nunca lo supera (validez del evaluador) y lo alcanza con una frecuencia y un reward
   medio mínimos.

La equivalencia arquitectónica estricta del caso libre (columnas de presupuesto
neutras) se verifica con un agente sintético y semilla fija en
test_greedy_k_conformity.py, que no depende de ningún checkpoint.

Se omite (skip) si el checkpoint no existe.
"""

import os
import pytest

import numpy as np

from config import TRAIN_DAYS
from baselines.politicas import calcular_ev_pronostico
from evaluation import make_replay_env, rollout_drl_env, ub_pi
from market_process import MarketProcess, asegurar_parametros
from transfer_learning import load_checkpoint_compatible
from problem_data import load_matrices

NUM_NODES = 10
CHECKPOINT_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "checkpoints", f"am_checkpoint_{NUM_NODES}nodes.pt",
)

# Umbrales con margen sobre lo medido el 2026-10-18 con el rollout greedy (sin
# beam search) y el checkpoint de 10 nodos, 300 casos de evaluación: 56% óptimos,
# reward medio 66% del óptimo (63.7% / 75% con el P95 con que se entrenó ese
# checkpoint). Con beam=5 el mismo checkpoint daba 96% / 98%: el greedy es una
# política de un solo paso, sin búsqueda. Si un reentrenamiento cae por debajo,
# es una señal real.
MIN_OPTIMAL_HIT_RATE = 0.40
MIN_MEAN_REWARD_RATIO = 0.55
EPS = 1e-6


@pytest.fixture(scope="module")
def loaded():
    if not os.path.exists(CHECKPOINT_PATH):
        pytest.skip(f"Checkpoint no encontrado: {CHECKPOINT_PATH}")
    agent, critic = load_checkpoint_compatible(
        CHECKPOINT_PATH, num_nodes=NUM_NODES, d_h=192, n_heads=8, n_layers=4, d_ff=768,
        verbose=False,
    )
    agent.eval()
    data = load_matrices(NUM_NODES)
    return agent, data


def test_free_case_rollout_quality_vs_exact_optimum(loaded):
    agent, data = loaded
    (time_matrix, rate_stack, loads_stack, distance_arr, diesel_arr,
     ltr_stack, avail_expected_arr, reward_global_p95) = data
    num_days = loads_stack.shape[0]
    time_matrix_np = np.asarray(time_matrix, dtype=float)
    asegurar_parametros()
    market = MarketProcess(NUM_NODES)
    env = make_replay_env(time_matrix, rate_stack, loads_stack, distance_arr, diesel_arr, NUM_NODES, market=market)
    ev_esperado = calcular_ev_pronostico(market, distance_arr, diesel_arr)

    drl_rewards, opt_rewards = [], []
    for day in range(TRAIN_DAYS, num_days):
        for start_node in range(NUM_NODES):
            route, reward, _ = rollout_drl_env(
                agent, env, start_node, day,
                time_matrix_np, distance_arr, ltr_stack, NUM_NODES,
                reward_global_p95=reward_global_p95,
                avail_expected_arr=avail_expected_arr,
                k_max=None,   # caso libre — k_restricted=0
                ev_esperado=ev_esperado,
            )
            assert route is not None, f"start_node={start_node} day={day}: ruta inválida"
            assert route[0] == start_node and route[-1] == start_node, (
                f"start_node={start_node} day={day}: ruta {route} no cierra el ciclo"
            )
            opt_reward, _ = ub_pi(
                time_matrix_np, distance_arr, diesel_arr, start_node, None,
                (loads_stack, rate_stack), start_day_idx=day,
            )
            # El óptimo con el día conocido es una cota superior: si el DRL la supera, el
            # evaluador del agente y el del solver están desalineados.
            assert reward <= opt_reward + EPS, (
                f"start_node={start_node} day={day}: DRL {reward} > óptimo {opt_reward} "
                f"(ruta {route}) — evaluador desalineado con UB-PI"
            )
            drl_rewards.append(reward)
            opt_rewards.append(opt_reward)

    n = len(opt_rewards)
    hit_rate = sum(abs(o - d) < EPS for d, o in zip(drl_rewards, opt_rewards)) / n
    ratio = sum(drl_rewards) / sum(opt_rewards)

    assert hit_rate >= MIN_OPTIMAL_HIT_RATE, (
        f"Solo {hit_rate:.1%} de {n} casos alcanzan el óptimo "
        f"(mínimo {MIN_OPTIMAL_HIT_RATE:.0%}) — ¿checkpoint degradado o reentrenamiento fallido?"
    )
    assert ratio >= MIN_MEAN_REWARD_RATIO, (
        f"Reward medio DRL = {ratio:.1%} del óptimo (mínimo {MIN_MEAN_REWARD_RATIO:.0%})"
    )
