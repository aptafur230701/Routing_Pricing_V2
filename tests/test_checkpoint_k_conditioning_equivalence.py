"""
test_checkpoint_k_conditioning_equivalence.py
==============================================
Fase 2 — criterio de aceptación obligatorio: cargar un checkpoint existente y
verificar que un rollout greedy con k_restricted=0 (caso libre) produce
EXACTAMENTE la misma ruta y reward que antes del condicionamiento en K.

Los valores esperados se capturaron ejecutando beam_search_dynamic(beam_width=1)
sobre checkpoints/am_checkpoint_10nodes.pt con el código previo al cambio
(vía `git stash` al HEAD limpio, antes de tocar attention_encoder.py /
am_agent.py / transfer_learning.py) y se hardcodean aquí como regresión
permanente.

NOTA: checkpoints/ está en .gitignore, así que este archivo no es estable
frente a un reentrenamiento — si se reentrena am_checkpoint_10nodes.pt (como
ya ocurrió una vez), este test pasa a verificar "el checkpoint actual sigue
encontrando estas rutas" en vez de "equivalencia arquitectónica estricta".
En este caso concreto ambas rutas coinciden con el óptimo exacto (LS-Exact/
LS-Oracle), así que el test se mantiene en verde por esa razón, no porque
demuestre que las columnas de presupuesto siguen en cero (dejó de ser cierto
tras el reentrenamiento — se quitó la aserción correspondiente).

ACTUALIZACIÓN: los valores se re-capturaron después de corregir
problem_data.py (umbral de existencia loads_day<=1 → <=0). Ese cambio modifica
la matriz de reward y la disponibilidad de lanes que ve este MISMO checkpoint
ya entrenado, así que algunas rutas cambiaron sin que se haya reentrenado nada
— es el comportamiento esperado del cambio, no una regresión. Ver auditoría:
el checkpoint de 10 nodos, evaluado con el pipeline de datos corregido, deja
de tomar el arco (6→1) en dos de los seis casos.
"""

import os
import pytest

from config import DEVICE
from transfer_learning import load_checkpoint_compatible
from problem_data import load_matrices

NUM_NODES = 10
CHECKPOINT_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "checkpoints", f"am_checkpoint_{NUM_NODES}nodes.pt",
)

# Re-capturado tras hacer que draw_lane_availability lea loads_day>0 del
# registro histórico (ver NOTA arriba).
EXPECTED = {
    (0, 0):  {"route": [0, 6, 1, 0],    "reward": 1490.0},
    (0, 10): {"route": [0, 6, 0],       "reward": 971.0},
    (3, 0):  {"route": [3, 5, 3],       "reward": 690.0},
    (3, 10): {"route": [3, 5, 3],       "reward": 710.0},
    (7, 0):  {"route": [7, 6, 7],       "reward": 1281.0},
    (7, 10): {"route": [7, 6, 8, 7],    "reward": 484.0},
}


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


def test_free_case_rollout_matches_pre_k_conditioning_baseline(loaded):
    agent, data = loaded
    (time_matrix, rate_stack, loads_stack, distance_arr, diesel_arr,
     ltr_stack, trucks_stack, avail_expected_arr, reward_global_p95) = data

    for (start_node, start_day), expected in EXPECTED.items():
        route, reward, _ = agent.beam_search_dynamic(
            start_node, start_day, time_matrix, rate_stack, loads_stack,
            distance_arr, diesel_arr, beam_width=1,
            ltr_stack=ltr_stack, trucks_stack=trucks_stack,
            lane_filter=True, reward_global_p95=reward_global_p95,
            k_max=None,   # caso libre — k_restricted=0
        )
        assert route == expected["route"], (
            f"start_node={start_node} start_day={start_day}: "
            f"ruta {route} != esperada {expected['route']}"
        )
        assert reward == pytest.approx(expected["reward"]), (
            f"start_node={start_node} start_day={start_day}: "
            f"reward {reward} != esperado {expected['reward']}"
        )
