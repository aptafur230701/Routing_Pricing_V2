"""
baselines/ub_pi.py
===================
UB-PI (ver baselines/dp_relajada.py y evaluation.py): cota con información perfecta. Resuelve la mejor ruta
con el mundo completo conocido mediante una DP hacia atrás. Los demás baselines viven en
baselines/politicas.py y baselines/dp_relajada.py.
"""

import numpy as np

from config import UBPI_DT_HORAS, HORAS_DIA_MERCADO, COTIZACION_HORAS, COTIZACIONES_MAX_POR_NODO
from problem_data import lane_reward
from routing_env import max_pasos, vacio_permitido

# ── Constantes de dominio ─────────────────────────────────────────────────────
DAYS_PER_PERIOD = 14            # día de mercado:  day_idx = start + int(t // 14)


def franjas_por_dia(dt: float) -> int:
    """Franjas de un día de mercado (HORAS_DIA_MERCADO / dt), que debe ser entero."""
    n = HORAS_DIA_MERCADO / dt
    if abs(n - round(n)) > 1e-6:
        raise ValueError(f"HORAS_DIA_MERCADO / dt = {n} no es entero: la espera no cae en una franja")
    return int(round(n))


def solve_ub_pi(
    start_node, time_matrix, rates, loads, distance_arr, diesel_arr, max_d,
    start_day_idx=0, k_max=None, dt=UBPI_DT_HORAS, retrasos=COTIZACIONES_MAX_POR_NODO,
):
    """Mejor ruta con el mundo completo conocido (UB-PI): DP hacia atrás con estado (nodo, franja, k).

    Las jugadas son las del entorno (routing_env.py), con el competidor conocido (`loads` ya trae en cero las
    lanes donde ningún nivel gana y `rates` la tarifa del nivel más alto que gana, evaluation.mundo_ub_pi):
      · cargada i→j si hay carga (reward lane_reward con carga);
      · VACÍA i→j a ≤ REPOSICION_MAX_MILLAS (reposicionamiento), o a casa desde un intermedio a cualquier
        distancia (reward −c);
      · salir con 0 … `retrasos` retrasos de COTIZACION_HORAS (una política puede perder cotizaciones y llegar más
        tarde, quizás otro día de mercado): mismo tablero, más tiempo;
      · esperar en i hasta el siguiente día de mercado (reward 0, no consume k), si después todavía se
        puede volver a casa (en casa antes de salir: hacer la ida y vuelta más corta).
    Los destinos intermedios exigen poder volver a casa dentro de `max_d`; volver a casa vale siempre y
    termina con valor 0. Si en casa no hay ninguna jugada, el camión no sale: valor 0 (env.sin_salida).
    Es información perfecta, no una política: ninguna política causal conoce los tableros futuros.

    rates, loads : (D, N, N) del mundo (día 0 = día de salida con start_day_idx = 0) o del histórico completo
        con `start_day_idx`; los días posteriores al último se recortan al último, como el entorno en el replay.
    k_max : presupuesto de cardinalidad (nº de arcos, incluido el retorno); None = caso libre.
    dt : franja de tiempo en horas. Los tiempos de viaje se redondean hacia abajo a la franja (nunca se
        pierde una ruta factible); el día de una salida en la franja s es start_day_idx + floor(s·dt / 14),
        recortado al último día del mundo.

    El redondeo acumula hasta una franja por arco, así que el tiempo real de una salida en la franja s puede
    ser hasta max_pasos franjas mayor y caer en el día siguiente. Para que UB-PI siga siendo cota, en las
    franjas que quedan a menos de max_pasos franjas antes de una frontera de día la recompensa es el máximo
    elemento a elemento de las filas de los dos días posibles y la lane existe si existe en alguno de los dos.

    Retorna (valor, ruta) con ruta = [casa, ..., casa]; una espera aparece como el mismo nodo repetido (i, i),
    así que los arcos son los pares consecutivos distintos. (0.0, [casa]) si el camión no sale.
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
    por_dia = franjas_por_dia(dt)

    rates = np.asarray(rates)
    loads = np.asarray(loads)
    D = loads.shape[0]
    dist = np.asarray(distance_arr, dtype=float)
    R = lane_reward(rates, loads, dist[None], np.asarray(diesel_arr, dtype=float)[None])   # (D, N, N)
    A = loads > 0
    R0 = lane_reward(np.zeros_like(dist), 0, dist, np.asarray(diesel_arr, dtype=float))   # (N, N) vacío
    repos = vacio_permitido(dist)                                          # (N, N) vacío ≤ 300 mi, j ≠ i

    def dia_de(franjas):
        return np.minimum(start_day_idx + np.floor(franjas * dt / DAYS_PER_PERIOD + 1e-9).astype(np.int64), D - 1)

    deriva = max_pasos(T, max_d)                                           # franjas que el redondeo puede acumular
    dia = dia_de(np.arange(Ns))                                            # (Ns,) día de cada franja
    dia_tarde = dia_de(np.arange(Ns) + deriva)                             # día si la llegada real se atrasó al máximo
    por_dias = {}
    destino = ~np.eye(N, dtype=bool) & (np.arange(N)[None, :] != h)        # j ≠ i y j ≠ h (el retorno va aparte)

    def arcos_de(s):
        """(ret (N,), cont (N, N)) de la salida en la franja s: volver a casa desde cada i, y el valor inmediato
        de ir a cada destino intermedio j (cargada si hay carga; si no, vacía si es reposicionamiento; −inf si
        ninguna). Usa la fila del día, o la envolvente de los dos días posibles."""
        d, e = int(dia[s]), int(dia_tarde[s])
        if (d, e) not in por_dias:
            Rd, Ad = (R[d], A[d]) if d == e else (np.maximum(R[d], R[e]), A[d] | A[e])
            cont = np.where(Ad, Rd, np.where(repos, R0, -np.inf))
            por_dias[(d, e)] = (np.where(Ad[:, h], Rd[:, h], R0[:, h]), np.where(destino, cont, -np.inf))
        return por_dias[(d, e)]

    qs = np.arange(retrasos + 1) * int(np.floor(COTIZACION_HORAS / dt + 1e-9))           # franjas de cada retraso

    vuelta = f + f.T
    np.fill_diagonal(vuelta, smax + 1)
    holgura_casa = vuelta[h].min()                                         # ida y vuelta más corta desde casa
    def fin_espera(s):
        return (s // por_dia + 1) * por_dia
    vuelta_desde = np.where(np.arange(N) == h, holgura_casa, f[:, h])     # (N,) franjas para volver tras esperar
    def espera_ok(i, s):
        return fin_espera(s) + vuelta_desde[i] <= smax

    # V[k, s, i]: valor de estar en i (intermedio) en la franja s con k arcos restantes (k = 1: casa o esperar).
    # Hc[k, s]: valor de estar en casa SIN haber salido en la franja s con k arcos. Caso libre: k = 1 sin dimensión.
    niveles = 1 if k_max is None else k_max
    V = np.full((niveles + 1, Ns, N), -np.inf)
    Hc = np.full((niveles + 1, Ns), -np.inf)
    # M[k, s, j] = max_q V[k, s + q, j] sobre los retrasos q, solo con la llegada s + q que deja volver a casa desde j
    # (s + q + f_jh ≤ smax). El retraso no cambia el tablero del origen, así que basta con este máximo móvil.
    M = np.full((niveles + 1, Ns, N), -np.inf)
    jj = np.arange(N)[None, :]
    for s in range(smax, -1, -1):
        ret, cont = arcos_de(s)
        sn = np.minimum(s + f, smax)
        sw = min(fin_espera(s), smax)
        ok_w = fin_espera(s) + vuelta_desde <= smax                       # (N,) se puede esperar en i
        for k in range(1, niveles + 1):
            k_sig = k if k_max is None else k - 1                          # caso libre: la misma tabla
            mejor = ret.copy()
            if k_max is None or k >= 2:
                mejor = np.maximum(mejor, (cont + M[k_sig][sn, jj]).max(axis=1))
            mejor = np.where(ok_w, np.maximum(mejor, V[k][sw]), mejor)
            V[k, s] = mejor
            for q in qs:
                if s + q <= smax:
                    M[k, s] = np.maximum(M[k, s], np.where(s + q + f[:, h] <= smax, V[k, s + q], -np.inf))
            if k_max is None or k == niveles:                              # salida de casa: k = K (o libre)
                c = (cont[h] + M[k_sig][sn[h], np.arange(N)]).max()
                if ok_w[h]:
                    c = max(c, Hc[k, sw])
                Hc[k, s] = c
            if k_max is None:
                break

    nivel_raiz = 1 if k_max is None else niveles
    valor = float(Hc[nivel_raiz, 0])
    if not np.isfinite(valor):
        return 0.0, [h]                                                    # no sale: ninguna jugada desde casa

    # Traza hacia adelante: en cada estado, la jugada que alcanza el valor (el retorno si empata).
    ruta, i, s, k, en_casa = [h], h, 0, nivel_raiz, True
    for _ in range(4 * Ns):
        ret, cont = arcos_de(s)
        k_sig = k if k_max is None else k - 1
        objetivo = Hc[k, s] if en_casa else V[k, s, i]
        if not en_casa and ret[i] >= objetivo - 1e-9:
            ruta.append(h)
            break
        if espera_ok(i, s):
            sw = min(fin_espera(s), smax)
            if (Hc[k, sw] if en_casa else V[k, sw, i]) >= objetivo - 1e-9:
                ruta.append(i)                                             # espera: el nodo se repite
                s = sw
                continue
        cand = cont[i] + (M[k_sig][np.minimum(s + f[i], smax), np.arange(N)] if (k_max is None or k >= 2) else -np.inf)
        j = int(np.argmax(cand))
        ruta.append(j)
        s = int(s + f[i, j])                                           # más el retraso que alcanza el máximo
        s += int(next(q for q in qs if s + q + f[j, h] <= smax and V[k_sig, s + q, j] >= M[k_sig, s, j] - 1e-9))
        i, k, en_casa = j, k_sig, False
    return valor, ruta
