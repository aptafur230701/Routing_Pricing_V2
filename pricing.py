"""
pricing.py
==========
Precio de reserva p*(i→j, s) en USD/milla (PRICING_PLAN.md §2): el costo de oportunidad por milla de ir a j,
es decir, la tarifa a la que al camión le da igual aceptar la lane i→j o su mejor alternativa. Es una
recomendación del lado del transportista, NO una probabilidad de aceptación del cliente.

    Q(s, a)   = r_obs(i→a)/S + Ṽ(s, a)                 valor de aceptar la lane i→a (Ṽ(s, casa) = 0)
    Q_alt(s)  = max_{a ≠ j válida} Q(s, a)               mejor alternativa a j
    p*(i→j)   = ( S · (Q_alt − Ṽ(s, j)) + costo(i→j) ) / dist(i→j)

con r_obs(i→a) = tarifa · dist · 1[hay carga] − costo (sin redondear), S = REWARD_SCALE_FACTOR y Ṽ(s, a) el
margen esperado desde la llegada a `a` hasta volver a casa siguiendo el actor greedy.

Ṽ tiene dos cabezas, A y B (agent.post_value_a / post_value_b), entrenadas con mitades distintas de los datos:
  · p*_AB elige la mejor alternativa con A y la evalúa (Q_alt y Ṽ(s, j)) con B; p*_BA al revés.
  · p* = promedio de ambos; el rango es [mínimo, máximo] de los dos. El rango mide la variabilidad del
    estimador (partición de datos y ruido de las etiquetas), NO es un intervalo de confianza: las dos cabezas
    comparten encoder, contexto y actor.
p* se calcula para el arco i→j que la política de ruteo (el actor) ya decidió tomar; j nunca se re-decide con Q
— Q solo sirve aquí para valorar la MEJOR ALTERNATIVA a esa decisión (el costo de oportunidad).

p* solo existe para lanes con carga; si j es el regreso al depot y trae carga, cotiza igual que cualquier otra
lane (con Ṽ(s, depot) = 0, ya impuesto en q_values). Si j es la única acción factible no hay alternativa y p*
tampoco está definido (`only_option`). En esos casos el resultado trae p* = NaN.
"""

from dataclasses import dataclass

import numpy as np

from config import MPG, MARGINAL_COST_SIN_DIESEL


def lane_cost_row(distance_row: np.ndarray, diesel_row: np.ndarray) -> np.ndarray:
    """Costo en USD de recorrer cada lane saliente (misma fórmula que problem_data.lane_reward); sirve con (N,) o (N, N)."""
    return distance_row * (diesel_row / MPG) + distance_row * MARGINAL_COST_SIN_DIESEL


def percentil_historico(valor: float, serie: np.ndarray, cargas: np.ndarray = None) -> float:
    """Percentil (0-100) de `valor` dentro de `serie` (fracción de la serie menor o igual al valor).

    cargas : si se pasa, solo cuentan los días con carga (> 0): el histórico trae tarifa también en días
        sin carga, que el camión nunca habría visto ofertada.
    """
    serie = np.asarray(serie, dtype=float)
    ok = np.isfinite(serie)
    if cargas is not None:
        ok &= np.asarray(cargas) > 0
    serie = serie[ok]
    return float("nan") if serie.size == 0 else 100.0 * float(np.mean(serie <= valor))


def r_obs(rate_row, loads_row, distance_row, cost_row):
    """Margen de recorrer cada lane saliente sin redondear: ingreso solo si hay carga, menos el costo. Admite (N,) o (B, N)."""
    rate, dist = np.asarray(rate_row, dtype=np.float64), np.asarray(distance_row, dtype=np.float64)
    ingreso = rate * dist if loads_row is None else np.where(np.asarray(loads_row) > 0, rate * dist, 0.0)
    return ingreso - np.asarray(cost_row, dtype=np.float64)


def q_values(rate_row, loads_row, distance_row, cost_row, S: float, v, start_node) -> np.ndarray:
    """Q(s, a) = r_obs(a)/S + Ṽ(s, a) con Ṽ(s, casa) = 0. v, rate_row, ...: (N,) con start_node int, o (B, N) con (B,)."""
    v = np.array(v, dtype=np.float64)
    if v.ndim == 1:
        v[start_node] = 0.0
    else:
        v[np.arange(v.shape[0]), np.asarray(start_node)] = 0.0
    return r_obs(rate_row, loads_row, distance_row, cost_row) / S + v


@dataclass
class PrecioReserva:
    """Resultado de reservation_price para la lane current_node → j (USD/milla; NaN donde no está definido)."""
    p: float                 # promedio de p*_AB y p*_BA
    p_ab: float
    p_ba: float
    p_min: float
    p_max: float
    break_even: float        # costo / dist
    has_load: bool           # hay carga en la lane (el retorno a casa cotiza igual si trae carga)
    only_option: bool        # j es la única acción factible (no hay alternativa)
    alt_a: int = -1          # mejor alternativa según A (usada en p*_AB); -1 si no hay
    alt_b: int = -1          # mejor alternativa según B (usada en p*_BA)

    @property
    def definido(self) -> bool:
        return bool(np.isfinite(self.p))


def reservation_price(
    v_a:           np.ndarray,      # (N,) Ṽ de la cabeza A (unidades escaladas por 1/S)
    v_b:           np.ndarray,      # (N,) Ṽ de la cabeza B
    action_mask:   np.ndarray,      # (N,) 1 = acción válida desde el nodo actual
    current_node:  int,
    j:             int,
    rate_row_obs:  np.ndarray,      # (N,) tarifa USD/milla observada al llegar a current_node
    distance_row:  np.ndarray,      # (N,) distancia current_node → a
    cost_row:      np.ndarray,      # (N,) costo de current_node → a (lane_cost_row)
    start_node:    int,
    S:             float,           # REWARD_SCALE_FACTOR
    loads_row:     np.ndarray = None,   # (N,) cargas observadas; ingreso solo donde > 0 (None ⇒ todas con carga)
) -> PrecioReserva:
    """Precio de reserva de la lane current_node→j con las dos cabezas (ver el docstring del módulo).

    El ingreso de cada alternativa sigue el entorno: tarifa × distancia solo si hay carga en esa lane; si no,
    la acción cuesta el costo sin ingreso. El retorno a casa (start_node) es una alternativa más, con Ṽ = 0.
    """
    break_even = float(cost_row[j] / distance_row[j])
    tiene_carga = loads_row is None or loads_row[j] > 0
    nan = float("nan")
    valid = np.asarray(action_mask).astype(bool).copy()
    valid[current_node] = False
    valid[j] = False
    if not tiene_carga or not valid.any():
        return PrecioReserva(nan, nan, nan, nan, nan, break_even, bool(tiene_carga), bool(tiene_carga) and not valid.any())

    qa = q_values(rate_row_obs, loads_row, distance_row, cost_row, S, v_a, start_node)
    qb = q_values(rate_row_obs, loads_row, distance_row, cost_row, S, v_b, start_node)
    va, vb = np.array(v_a, dtype=np.float64), np.array(v_b, dtype=np.float64)
    va[start_node] = vb[start_node] = 0.0
    idx = np.where(valid)[0]
    alt_a = int(idx[np.argmax(qa[idx])])        # A elige la alternativa …
    alt_b = int(idx[np.argmax(qb[idx])])
    p_ab = (S * (qb[alt_a] - vb[j]) + cost_row[j]) / distance_row[j]    # … y B la evalúa (Q_alt y Ṽ(s, j))
    p_ba = (S * (qa[alt_b] - va[j]) + cost_row[j]) / distance_row[j]
    return PrecioReserva(
        p=float((p_ab + p_ba) / 2), p_ab=float(p_ab), p_ba=float(p_ba), p_min=float(min(p_ab, p_ba)),
        p_max=float(max(p_ab, p_ba)), break_even=break_even, has_load=True, only_option=False,
        alt_a=alt_a, alt_b=alt_b,
    )

