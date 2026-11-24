"""
test_pricing.py
===============
Utilidades compartidas de pricing.py: costo por lane (misma fórmula que problem_data.lane_reward) y percentil
histórico. La cotización conjunta (niveles, competencia, aceptación y reglas) está en test_cotizacion.py.
"""

import numpy as np
import pytest

from pricing import lane_cost_row, percentil_historico
from problem_data import lane_reward

N = 6


@pytest.fixture
def lane():
    rng = np.random.default_rng(0)
    dist = rng.uniform(200, 900, N)
    diesel = rng.uniform(3.5, 4.5, N)
    return dict(rate=rng.uniform(2.0, 3.0, N), dist=dist, cost=lane_cost_row(dist, diesel), diesel=diesel)


def test_cost_row_matches_lane_reward(lane):
    ingreso_nulo = lane_reward(lane["rate"], np.zeros(N), lane["dist"], lane["diesel"])
    np.testing.assert_allclose(-lane["cost"], ingreso_nulo, atol=0.5)       # round(·, 0) de lane_reward


def test_percentil_historico():
    assert percentil_historico(2.5, np.array([1.0, 2.0, 3.0, 4.0])) == 50.0
    assert np.isnan(percentil_historico(1.0, np.array([])))
    assert percentil_historico(2.5, np.array([1.0, 2.0, 3.0, 4.0]), cargas=np.array([1, 0, 0, 1])) == 50.0
    assert percentil_historico(2.5, np.array([1.0, 2.0, 9.0, 9.0]), cargas=np.array([0, 1, 1, 1])) == pytest.approx(100 / 3)
    assert np.isnan(percentil_historico(2.5, np.array([1.0, 2.0]), cargas=np.zeros(2)))
