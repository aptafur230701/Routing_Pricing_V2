"""
Solvers.py
==========
UB-PI (ver dp_relajada.py y evaluation.py): cota con información perfecta. Resuelve la mejor ruta
con el mundo completo conocido mediante una DP hacia atrás. Los demás baselines viven en
baselines.py y dp_relajada.py.
"""

import numpy as np

from config import UBPI_DT_HORAS
from problem_data import lane_reward
from routing_env import max_pasos

# ── Constantes de dominio ─────────────────────────────────────────────────────
DAYS_PER_PERIOD = 14            # día de mercado:  day_idx = start + int(t // 14)


def solve_ub_pi(
    start_node, time_matrix, rates, loads, distance_arr, diesel_arr, max_d,
    start_day_idx=0, k_max=None, dt=UBPI_DT_HORAS,
):
    """Mejor ruta con el mundo completo conocido (UB-PI): DP hacia atrás con estado (nodo, franja, k).

    El camión puede volver a pasar por un nodo, así que el estado no lleva visitados. Las
    reglas de factibilidad son las del entorno: lane con carga en la fila del día salvo en el
    primer movimiento desde casa, retorno a casa posible dentro de `max_d`, retorno forzado
    con k = 1, y valor terminal 0 al volver a casa. Es información perfecta, no una política:
    ninguna política causal posee la disponibilidad de los nodos aún no visitados.

    rates, loads : (D, N, N) del mundo (`sample_world`, día 0 = día de salida con start_day_idx = 0)
        o del histórico completo con `start_day_idx`; los días posteriores al último se recortan
        al último, como el entorno en el replay.
    k_max : presupuesto de cardinalidad (nº de arcos, incluido el retorno); None = caso libre
        (sin dimensión k; los tiempos de viaje son ≥ 1 franja, así que la recursión es hacia atrás en s).
    dt : franja de tiempo en horas. Los tiempos de viaje se redondean hacia abajo a la franja
        (nunca se pierde una ruta factible, pero una ruta puede caber en la DP y exceder
        `max_d` en unos minutos); el día de una salida en la franja s es
        start_day_idx + floor(s·dt / 14), recortado al último día del mundo.

    La recompensa de salir de i en la franja s es `lane_reward` de la fila de ese día. El
    redondeo acumula hasta una franja por arco, así que el tiempo real de una salida en la franja
    s puede ser hasta max_pasos franjas mayor y caer en el día siguiente. Para que UB-PI siga
    siendo cota con rutas largas, en las franjas que quedan a menos de max_pasos franjas antes de
    una frontera de día la recompensa es el máximo elemento a elemento de las filas de los dos
    días posibles y la lane existe si existe en alguno de los dos.

    Retorna (valor, ruta) con ruta = [casa, ..., casa], o (nan, None) si no hay ruta factible.
    """
    T = np.asarray(time_matrix, dtype=float)
    N = T.shape[0]
    h = int(start_node)
    if k_max is not None and k_max < 2:
        return float("nan"), None

    f = np.floor(T / dt + 1e-9).astype(np.int64)             # franjas de viaje, redondeo hacia abajo
    if f[~np.eye(N, dtype=bool)].min() < 1:
        raise ValueError("Hay tiempos de viaje menores a una franja: la recursión del caso libre no "
                         "estaría bien fundada. Reduce dt.")
    smax = int(np.floor(max_d / dt + 1e-9))
    Ns = smax + 1

    rates = np.asarray(rates)
    loads = np.asarray(loads)
    D = loads.shape[0]
    R = lane_reward(rates, loads, np.asarray(distance_arr, dtype=float)[None],
                    np.asarray(diesel_arr, dtype=float)[None])             # (D, N, N)
    A = loads > 0
    def dia_de(franjas):
        return np.minimum(start_day_idx + np.floor(franjas * dt / DAYS_PER_PERIOD + 1e-9).astype(np.int64), D - 1)

    deriva = max_pasos(T, max_d)                                           # franjas que el redondeo puede acumular
    dia = dia_de(np.arange(Ns))                                            # (Ns,) día de cada franja
    dia_tarde = dia_de(np.arange(Ns) + deriva)                             # día si la llegada real se atrasó al máximo
    por_dias = {}

    def filas_de(s):
        """(R, A) de la salida en la franja s: la fila del día, o la envolvente de los dos días posibles."""
        d, e = int(dia[s]), int(dia_tarde[s])
        if (d, e) not in por_dias:
            por_dias[(d, e)] = (R[d], A[d]) if d == e else (np.maximum(R[d], R[e]), A[d] | A[e])
        return por_dias[(d, e)]

    jj = np.arange(N)[None, :]
    destino = ~np.eye(N, dtype=bool) & (np.arange(N)[None, :] != h)        # j ≠ i y j ≠ h (el retorno va aparte)
    lim = smax - (f + f[:, h][None, :])                                    # (N, N): s + f_ij + f_jh ≤ smax ⇔ s ≤ lim

    # V[k, s, i]: valor de estar en i en la franja s con k arcos restantes (k = 1: solo retorno).
    # Caso libre: una sola tabla W[s, i] sin k.
    niveles = 1 if k_max is None else k_max - 1                            # tablas necesarias: k = 1..K-1
    V = np.zeros((niveles + 1, Ns, N))
    for s in range(smax, -1, -1):
        Rd, Ad = filas_de(s)
        ret = Rd[:, h]                                                     # volver a casa: siempre disponible
        ok = Ad & destino & (s <= lim)
        sn = np.minimum(s + f, smax)
        if k_max is None:
            V[1, s] = ret if not ok.any() else np.maximum(
                ret, np.where(ok, Rd + V[1][sn, jj], -np.inf).max(axis=1))
            continue
        V[1, s] = ret
        if ok.any():
            for k in range(2, niveles + 1):
                V[k, s] = np.maximum(ret, np.where(ok, Rd + V[k - 1][sn, jj], -np.inf).max(axis=1))
        else:
            for k in range(2, niveles + 1):
                V[k, s] = ret

    # Raíz: primer movimiento desde casa en la franja 0, sin máscara de lane.
    k_sig = niveles                                                        # tabla de k − 1 arcos restantes
    R0 = filas_de(0)[0][h]
    ok0 = destino[h] & (f[h] + f[:, h] <= smax)
    if not ok0.any():
        return float("nan"), None
    cand0 = np.where(ok0, R0 + V[k_sig][np.minimum(f[h], smax), np.arange(N)], -np.inf)
    j = int(np.argmax(cand0))
    valor = float(cand0[j])

    # Traza hacia adelante: en cada estado se elige la opción que alcanza V (el retorno si empata).
    ruta, s, k = [h, j], int(f[h, j]), k_sig
    while k_max is None or k > 1:
        Rd, Ad = filas_de(s)
        ok = Ad[j] & destino[j] & (s <= lim[j])
        if not ok.any():
            break
        sn = np.minimum(s + f[j], smax)
        cand = np.where(ok, Rd[j] + V[1 if k_max is None else k - 1][sn, np.arange(N)], -np.inf)
        l = int(np.argmax(cand))
        if cand[l] <= Rd[j, h]:
            break
        ruta.append(l)
        j, s = l, s + int(f[j, l])
        k = k if k_max is None else k - 1
    ruta.append(h)
    return valor, ruta
