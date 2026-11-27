"""
test_checkpoint_k_conditioning_equivalence.py
==============================================
Red de seguridad sobre el checkpoint entrenado de 10 nodos
(checkpoints/am_checkpoint_10nodes.pt):

1. El checkpoint carga con load_checkpoint_compatible.
2. En el caso libre (k_max=None), con el rollout greedy paso a paso en el replay de los días de
   evaluación (destino y precio del DRL, competidor con una semilla fija), el margen de toda ruta que
   cierra el ciclo nunca supera UB-PI sobre los mismos tableros y el mismo competidor (validez del
   evaluador), y al menos FRACCION_VALIDA_MIN de los casos cierra el ciclo (detecta un checkpoint roto).

No se exige que todas las rutas cierren ni una calidad mínima frente a UB-PI (todavía no hay resultados
del modelo con cotización para fijar umbrales).

La equivalencia arquitectónica estricta del caso libre (columnas de presupuesto neutras) se verifica
con un agente sintético en test_greedy_k_conformity.py, que no depende de ningún checkpoint.

Se omite (skip) si el checkpoint no existe o es anterior a la cotización (sin cabeza de precio: cargarlo
es un error, ver transfer_learning._instanciar_y_transferir).
"""

import os
import pytest

import numpy as np

from config import TRAIN_DAYS
from evaluation import make_replay_env, mundo_ub_pi, rollout_drl_env, tablero_replay, ub_pi
from market_process import MarketProcess, asegurar_parametros
from transfer_learning import load_checkpoint_compatible, es_checkpoint_con_cotizacion
from problem_data import load_matrices

NUM_NODES = 10
CHECKPOINT_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "checkpoints", f"am_checkpoint_{NUM_NODES}nodes.pt",
)
SEMILLA = 2026
EPS = 1e-6
FRACCION_VALIDA_MIN = 0.90   # el checkpoint de 10 nodos cierra el ciclo en ~96% de estos casos


@pytest.fixture(scope="module")
def loaded():
    if not es_checkpoint_con_cotizacion(CHECKPOINT_PATH):
        pytest.skip(f"No hay un checkpoint entrenado con la cotización: {CHECKPOINT_PATH}")
    agent, critic = load_checkpoint_compatible(
        CHECKPOINT_PATH, num_nodes=NUM_NODES, d_h=192, n_heads=8, n_layers=4, d_ff=768,
        verbose=False,
    )
    agent.eval()
    data = load_matrices(NUM_NODES)
    return agent, data


def test_free_case_rollout_no_supera_ub_pi(loaded):
    agent, data = loaded
    (time_matrix, rate_stack, loads_stack, distance_arr, diesel_arr,
     ltr_stack, avail_expected_arr, reward_global_p95) = data
    D = loads_stack.shape[0] - TRAIN_DAYS
    time_matrix_np = np.asarray(time_matrix, dtype=float)
    asegurar_parametros()
    market = MarketProcess(NUM_NODES)
    env = make_replay_env(time_matrix, rate_stack[TRAIN_DAYS:], loads_stack[TRAIN_DAYS:], distance_arr, diesel_arr,
                          NUM_NODES, day_offset=TRAIN_DAYS, market=market)
    est_lane = market.estadisticas_dow(distance_arr, diesel_arr)
    mundo = mundo_ub_pi(tablero_replay(market, SEMILLA, TRAIN_DAYS + np.arange(D)))

    n_casos = n_validos = 0
    for day in range(0, D, 3):
        for start_node in range(NUM_NODES):
            route, reward, _ = rollout_drl_env(
                agent, env, start_node, day,
                time_matrix_np, distance_arr, ltr_stack[:, TRAIN_DAYS:], NUM_NODES,
                reward_global_p95=reward_global_p95,
                avail_expected_arr=avail_expected_arr,
                k_max=None, world_seed=SEMILLA, est_lane=est_lane,
            )
            n_casos += 1
            if route is None:
                # Una política greedy entrenada puede no cerrar el ciclo a tiempo en algún caso: eso es
                # calidad del agente, no validez del evaluador. Se cuenta y se exige un mínimo al final.
                continue
            n_validos += 1
            assert route[0] == start_node and route[-1] == start_node
            opt_reward, _ = ub_pi(time_matrix_np, distance_arr, diesel_arr, start_node, None, mundo,
                                  start_day_idx=day)
            assert reward <= opt_reward + EPS, (
                f"start_node={start_node} day={day}: DRL {reward} > UB-PI {opt_reward} "
                f"(ruta {route}) — evaluador desalineado con UB-PI"
            )

    # Red de seguridad contra un checkpoint roto (que no cierre casi ningún ciclo), sin exigir calidad fina.
    assert n_validos >= FRACCION_VALIDA_MIN * n_casos, (
        f"solo {n_validos} de {n_casos} rutas cierran el ciclo (mínimo {FRACCION_VALIDA_MIN:.0%})"
    )
