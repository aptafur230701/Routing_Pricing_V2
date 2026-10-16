"""
test_checkpoint_k_conditioning_equivalence.py
==============================================
Red de seguridad sobre el checkpoint entrenado de 10 nodos
(checkpoints/am_checkpoint_10nodes.pt).

Historia: este archivo pinneaba rutas concretas de un rollout greedy
(beam_width=1) capturadas con un checkpoint anterior. Como checkpoints/ está en
.gitignore y se reentrena, esos valores caducaban en cada reentrenamiento y el
test dejaba de decir algo útil. Además el docstring afirmaba que las rutas
pinneadas coincidían con el óptimo exacto, lo cual no era cierto en todos los
casos (p. ej. start_node=3, day=0: 690 pinneado vs 851 óptimo).

Ahora el test mide lo que realmente importa y no caduca con un reentrenamiento:

1. El checkpoint carga con load_checkpoint_compatible (compatibilidad de
   arquitectura/state_dict, incluidas las columnas de presupuesto K).
2. En el caso libre (k_max=None), con el beam de producción
   (get_beam_width_det), el DRL se compara contra el óptimo exacto
   (label-setting) en los días de evaluación: nunca lo supera (validez del
   evaluador) y lo alcanza con una frecuencia y un reward medio mínimos.

La equivalencia arquitectónica estricta del caso libre (columnas de presupuesto
neutras) se verifica con un agente sintético y semilla fija en
test_beam_k_conformity.py, que no depende de ningún checkpoint.

Se omite (skip) si el checkpoint no existe.
"""

import os
import pytest

from config import MAX_DURATION, TRAIN_DAYS, get_beam_width_det
from transfer_learning import load_checkpoint_compatible
from problem_data import load_matrices
from Solvers import solve_label_setting_exact

NUM_NODES = 10
CHECKPOINT_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "checkpoints", f"am_checkpoint_{NUM_NODES}nodes.pt",
)

# Umbrales con margen sobre lo medido el 2026-10-15 con el checkpoint del
# 2 de octubre (beam=5, 300 casos de evaluación): 96.0% óptimos, reward medio
# 98.4% del óptimo. Si un reentrenamiento cae por debajo, es una señal real.
MIN_OPTIMAL_HIT_RATE = 0.90
MIN_MEAN_REWARD_RATIO = 0.95
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
     ltr_stack, trucks_stack, avail_expected_arr, reward_global_p95) = data
    beam_width = get_beam_width_det(NUM_NODES)
    num_days = loads_stack.shape[0]

    drl_rewards, opt_rewards = [], []
    for day in range(TRAIN_DAYS, num_days):
        for start_node in range(NUM_NODES):
            route, reward, _ = agent.beam_search_dynamic(
                start_node, day, time_matrix, rate_stack, loads_stack,
                distance_arr, diesel_arr, beam_width=beam_width,
                ltr_stack=ltr_stack, trucks_stack=trucks_stack,
                lane_filter=True, reward_global_p95=reward_global_p95,
                k_max=None,   # caso libre — k_restricted=0
            )
            assert route[0] == start_node and route[-1] == start_node, (
                f"start_node={start_node} day={day}: ruta {route} no cierra el ciclo"
            )
            _, _, opt_reward, _ = solve_label_setting_exact(
                start_node, time_matrix, rate_stack, loads_stack,
                distance_arr, diesel_arr, MAX_DURATION, NUM_NODES,
                start_day_idx=day,
            )
            # El óptimo exacto es una cota superior: si el DRL la supera, el
            # evaluador del agente y el del solver están desalineados.
            assert reward <= opt_reward + EPS, (
                f"start_node={start_node} day={day}: DRL {reward} > óptimo {opt_reward} "
                f"(ruta {route}) — evaluador desalineado con LS-Exact"
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
