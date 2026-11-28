"""
baselines/ub_pi.py
===================
UB-PI (ver baselines/dp_relajada.py y evaluation.py): cota con información perfecta. Resuelve el valor de la mejor
ruta con el mundo completo conocido mediante una DP hacia atrás, para MUCHOS mundos a la vez en el dispositivo
(la GPU en el servidor). Los demás baselines viven en baselines/politicas.py y baselines/dp_relajada.py.
"""

import numpy as np
import torch

from config import DEVICE, UBPI_DT_HORAS, HORAS_DIA_MERCADO, COTIZACION_HORAS, COTIZACIONES_MAX_POR_NODO
from pricing import lane_reward_t
from routing_env import max_pasos, vacio_permitido

# ── Constantes de dominio ─────────────────────────────────────────────────────
DAYS_PER_PERIOD = 14            # día de mercado:  day_idx = start + int(t // 14)
_INF = float("inf")


def franjas_por_dia(dt: float) -> int:
    """Franjas de un día de mercado (HORAS_DIA_MERCADO / dt), que debe ser entero."""
    n = HORAS_DIA_MERCADO / dt
    if abs(n - round(n)) > 1e-6:
        raise ValueError(f"HORAS_DIA_MERCADO / dt = {n} no es entero: la espera no cae en una franja")
    return int(round(n))


def solve_ub_pi_lote(
    start_nodes, time_matrix, rates, loads, distance_arr, diesel_arr, max_d,
    k_max=None, dt=UBPI_DT_HORAS, retrasos=COTIZACIONES_MAX_POR_NODO, device=None, bloque: int = 128,
) -> np.ndarray:
    """Valor de la mejor ruta con el mundo completo conocido (UB-PI) de M mundos: DP hacia atrás con estado
    (nodo, franja, k), vectorizada sobre los mundos (en bloques de `bloque` mundos).

    Las jugadas son las del entorno (routing_env.py), con el competidor conocido (`loads` ya trae en cero las
    lanes donde ningún nivel gana y `rates` la tarifa del nivel más alto que gana, evaluation.mundos_ub_pi):
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

    start_nodes : (M,) casa de cada mundo.
    rates, loads : (M, D, N, N) de cada mundo (día 0 = día de salida); los días posteriores al último se recortan
        al último, como el entorno.
    k_max : presupuesto de cardinalidad (nº de arcos, incluido el retorno); None = caso libre (con k_max < 2: nan).
    dt : franja de tiempo en horas. Los tiempos de viaje se redondean hacia abajo a la franja (nunca se
        pierde una ruta factible); el día de una salida en la franja s es floor(s·dt / 14), recortado al último
        día del mundo.

    El redondeo acumula hasta una franja por arco, así que el tiempo real de una salida en la franja s puede
    ser hasta max_pasos franjas mayor y caer en el día siguiente. Para que UB-PI siga siendo cota, en las
    franjas que quedan a menos de max_pasos franjas antes de una frontera de día la recompensa es el máximo
    elemento a elemento de las filas de los dos días posibles y la lane existe si existe en alguno de los dos.

    Retorna (M,) valores (USD).
    """
    starts = np.asarray(start_nodes, dtype=np.int64)
    M = len(starts)
    if k_max is not None and k_max < 2:
        return np.full(M, np.nan)
    out = np.empty(M)
    for ini in range(0, M, bloque):
        sl = slice(ini, min(ini + bloque, M))
        out[sl] = _lote(starts[sl], time_matrix, np.asarray(rates)[sl], np.asarray(loads)[sl], distance_arr,
                        diesel_arr, max_d, k_max, dt, retrasos, torch.device(device) if device else DEVICE)
    return out


def _lote(starts, time_matrix, rates, loads, distance_arr, diesel_arr, max_d, k_max, dt, retrasos, dev) -> np.ndarray:
    T = np.asarray(time_matrix, dtype=float)
    N = T.shape[0]
    Mw = len(starts)
    h = torch.as_tensor(starts, device=dev)
    w = torch.arange(Mw, device=dev)
    jj = torch.arange(N, device=dev)

    f_np = np.floor(T / dt + 1e-9).astype(np.int64)                    # franjas de viaje, redondeo hacia abajo
    if f_np[~np.eye(N, dtype=bool)].min() < 1:
        raise ValueError("Hay tiempos de viaje menores a una franja: la recursión del caso libre no "
                         "estaría bien fundada. Reduce dt.")
    f = torch.as_tensor(f_np, device=dev)
    smax = int(np.floor(max_d / dt + 1e-9))
    Ns = smax + 1
    por_dia = franjas_por_dia(dt)

    rates_t = torch.as_tensor(rates, dtype=torch.float64, device=dev)
    loads_t = torch.as_tensor(loads, device=dev)
    D = loads_t.shape[1]
    dist = torch.as_tensor(np.asarray(distance_arr, dtype=float), device=dev)
    diesel = torch.as_tensor(np.asarray(diesel_arr, dtype=float), device=dev)
    R = lane_reward_t(rates_t, loads_t, dist[None, None], diesel[None, None])                # (Mw, D, N, N)
    A = loads_t > 0
    R0 = lane_reward_t(torch.zeros_like(dist), torch.zeros_like(dist), dist, diesel)          # (N, N) vacío
    repos = torch.as_tensor(vacio_permitido(np.asarray(distance_arr, dtype=float)), device=dev)

    def dia_de(franjas):
        return np.minimum(np.floor(franjas * dt / DAYS_PER_PERIOD + 1e-9).astype(np.int64), D - 1)

    deriva = max_pasos(T, max_d)                                           # franjas que el redondeo puede acumular
    dia = dia_de(np.arange(Ns))                                            # (Ns,) día de cada franja
    dia_tarde = dia_de(np.arange(Ns) + deriva)                             # día si la llegada real se atrasó al máximo
    destino = ((jj[None, :] != jj[:, None])[None] & (jj[None, None, :] != h[:, None, None]))   # (Mw, i, j): j ∉ {i, h}
    por_dias = {}

    def arcos_de(s):
        """(ret (Mw, N), cont (Mw, N, N)) de la salida en la franja s: volver a casa desde cada i, y el valor
        inmediato de ir a cada destino intermedio j (cargada si hay carga; si no, vacía si es reposicionamiento;
        −inf si ninguna). Usa la fila del día, o la envolvente de los dos días posibles."""
        d, e = int(dia[s]), int(dia_tarde[s])
        if (d, e) not in por_dias:
            Rd, Ad = (R[:, d], A[:, d]) if d == e else (torch.maximum(R[:, d], R[:, e]), A[:, d] | A[:, e])
            cont = torch.where(Ad, Rd, torch.where(repos, R0, torch.full_like(Rd, -_INF)))
            Rh, Ah = Rd[w, :, h], Ad[w, :, h]                                            # (Mw, N) lane i → h
            ret = torch.where(Ah, Rh, R0[:, h].T)
            por_dias[(d, e)] = (ret, torch.where(destino, cont, torch.full_like(cont, -_INF)))
        return por_dias[(d, e)]

    qs = [q * int(np.floor(COTIZACION_HORAS / dt + 1e-9)) for q in range(retrasos + 1)]   # franjas de cada retraso

    vuelta = f + f.T
    vuelta.fill_diagonal_(smax + 1)
    holgura_casa = vuelta[h].amin(dim=1)                                   # (Mw,) ida y vuelta más corta desde casa
    f_h = f[:, h].T                                                        # (Mw, N) f[j, h]
    vuelta_desde = torch.where(jj[None, :] == h[:, None], holgura_casa[:, None], f_h)   # franjas para volver tras esperar

    # V[k][s]: (Mw, N) valor de estar en i (intermedio) en la franja s con k arcos restantes (k = 1: casa o esperar).
    # Hc[k][s]: (Mw,) valor de estar en casa SIN haber salido en la franja s. Caso libre: k = 1 sin dimensión.
    niveles = 1 if k_max is None else int(k_max)
    V = torch.full((niveles + 1, Ns, Mw, N), -_INF, dtype=torch.float64, device=dev)
    Hc = torch.full((niveles + 1, Ns, Mw), -_INF, dtype=torch.float64, device=dev)
    # Mx[k, s, :, j] = max_q V[k, s + q, :, j] sobre los retrasos q, solo con la llegada s + q que deja volver a casa
    # desde j (s + q + f_jh ≤ smax). El retraso no cambia el tablero del origen, así que basta con este máximo móvil.
    Mx = torch.full((niveles + 1, Ns, Mw, N), -_INF, dtype=torch.float64, device=dev)
    jcol = jj[None, :].expand(N, N)
    for s in range(smax, -1, -1):
        ret, cont = arcos_de(s)
        sn = (s + f).clamp(max=smax)                                       # (i, j)
        fw = (s // por_dia + 1) * por_dia
        sw = min(fw, smax)
        ok_w = fw + vuelta_desde <= smax                                    # (Mw, N) se puede esperar en i
        for k in range(1, niveles + 1):
            k_sig = k if k_max is None else k - 1                          # caso libre: la misma tabla
            mejor = ret
            if k_max is None or k >= 2:
                g = Mx[k_sig][sn, :, jcol].permute(2, 0, 1)                # (Mw, i, j) = Mx[k_sig, s + f_ij, :, j]
                mejor = torch.maximum(mejor, (cont + g).amax(dim=2))
            mejor = torch.where(ok_w, torch.maximum(mejor, V[k, sw]), mejor)
            V[k, s] = mejor
            for q in qs:
                if s + q <= smax:
                    Mx[k, s] = torch.maximum(Mx[k, s], torch.where(s + q + f_h <= smax, V[k, s + q],
                                                                   torch.full_like(mejor, -_INF)))
            if k_max is None or k == niveles:                              # salida de casa: k = K (o libre)
                g_h = Mx[k_sig][sn[h], w[:, None], jj[None, :]]            # (Mw, j) = Mx[k_sig, s + f_hj, w, j]
                c = (cont[w, h] + g_h).amax(dim=1)
                c = torch.where(ok_w[w, h], torch.maximum(c, Hc[k, sw]), c)
                Hc[k, s] = c
            if k_max is None:
                break

    valor = Hc[1 if k_max is None else niveles, 0]
    return torch.where(torch.isfinite(valor), valor, torch.zeros_like(valor)).cpu().numpy()   # no sale: valor 0
