"""
baselines/dp_relajada.py
==========================
UB-DP y la tabla de B3 (baselines_drl_ruteo_estocastico.md, especificacion_fase0_ruteo_pricing.md §7.3):
programación dinámica sobre el estado (nodo, hora, k) con la decisión de precio. El camión puede revisitar
nodos intermedios, así que el estado no lleva el conjunto de visitados.

En cada llegada a i se sortea un tablero b: un día de entrenamiento al azar con el día de la semana de la
llegada (`MarketProcess.tablero_sin_memoria`) y sus cargas; los precios por nivel p_jk y m_i son los de ese
día real. Con P_k = (1 − u_k)^m_i:

    G_jk(s)  = r_jk + V_{k−1}(j, s + f_ij)          r_jk = lane_reward(p_jk, carga, d_ij, c_ij); V(casa) = 0
    E(s)     = mejor opción VACÍO: volver a casa sin carga (en la primera salida desde casa, cualquier j)
    A_j      = max_k [ P_k G_jk(s+τ) + (1 − P_k) E(s+τ) ]        segundo intento, con VACÍO de respaldo
    W_j      = max( E(s+τ), max_{j' ≠ j} A_j' )                  después de perder j (top-2 sobre j)
    V(i, s)  = media_b  max( E(s), max_j max_k [ P_k G_jk(s) + (1 − P_k) W_j ] )

τ = COTIZACION_HORAS en franjas (redondeo hacia abajo). Un rechazo no consume k y reusa el MISMO tablero
(W_j): si se sorteara uno nuevo, cotizar alto a propósito sería una forma gratis de "esperar otro tablero".
Volver a casa vacío solo vale si la lane de casa no se puede cotizar, pero con carga el nivel 0 (que gana
siempre y cobra p_0·d − c ≥ −c) domina esa opción, así que usar E como respaldo en todos los casos da el
mismo valor que las reglas del entorno. Solo hay dos intentos por visita (COTIZACIONES_MAX_POR_NODO = 2).

Es una inducción hacia atrás: s + τ y s + f son franjas posteriores, ya calculadas. UB-DP es el valor
aproximado (franjas de 15 min, S sorteos) de la mejor política que IGNORA EL ESTADO DEL MERCADO: conoce la
distribución de tableros por día de la semana pero no el día real, el índice ni el competidor. No es una
cota para una política que observa el índice; UB-PI (información perfecta) sí lo es.

Estado de la tabla: V[k, d0, h, j, s]
  · k  : arcos restantes (incluido el retorno); solo en la tabla de K fijo.
  · d0 : día de la semana del día de salida (lunes = 0). El día de la semana al llegar
         a la franja s es (d0 + s·dt // 14) % 7.
  · h  : nodo de casa.   · j : nodo actual.   · s : franja de tiempo, t = s · dt.

Representación de los tableros: por cada (día de la semana, nodo, sorteo) se guardan solo el día de
entrenamiento sorteado y las cargas; los rewards por nivel y P_k salen de una tabla por día real
(7, días, N, N, K), así que la memoria no crece con K × S. El máximo sobre k se hace en un bucle con un
máximo acumulado, en bloques de casas (`_h_chunk`).

Tratabilidad (decidido): si el cálculo supera UBDP_TIEMPO_MAX_HORAS o se queda sin memoria, se guarda el
marcador `ubdp_tabla_<N>nodos.no_tratable.json` y la tabla se reporta "no tratable": un resultado válido (la
DP no escala). Se puede calcular aparte, antes de entrenar: `python -m baselines.dp_relajada --num-nodes N`.
"""

import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")   # sin efecto si torch ya se cargó

import hashlib
import json
import time

import numpy as np
import torch

from config import (
    DEVICE, MAX_DURATION, K_MAX,
    UBDP_DT_HORAS, UBDP_S, UBDP_SEED, UBDP_TABLA_PATH, UBDP_TIEMPO_MAX_HORAS, UBDP_MAX_NODOS,
    COTIZACION_HORAS, COTIZACIONES_MAX_POR_NODO, U_NIVELES,
)
from pricing import N_NIVELES, U_NIVELES_ARR, precios_niveles, prob_ganar
from problem_data import lane_reward

_CWD = os.path.dirname(os.path.abspath(__file__))
_NEG = -1e9          # "menos infinito" finito: evita NaN en sumas
_DIA_HORAS = 14.0    # un día de mercado = 14 h (igual que el entorno)


def tabla_path(num_nodes: int) -> str:
    """Ruta absoluta del .npz con la tabla V de `num_nodes` nodos."""
    return os.path.join(_CWD, UBDP_TABLA_PATH.format(n=num_nodes))


def marcador_path(num_nodes: int) -> str:
    """Ruta del marcador "no tratable" de `num_nodes` nodos."""
    return tabla_path(num_nodes).replace(".npz", ".no_tratable.json")


class UBDPNoTratable(Exception):
    """El cálculo de la tabla superó el tiempo máximo o la memoria disponible."""

    def __init__(self, motivo: str, horas: float, fraccion: float, franja: int):
        self.motivo, self.horas, self.fraccion, self.franja = motivo, horas, fraccion, franja
        estimado = horas / fraccion if fraccion > 0 else float("inf")
        self.horas_estimadas = estimado
        super().__init__(f"UB-DP no tratable ({motivo}): {horas:.2f} h, {100 * fraccion:.1f} % del cálculo "
                         f"(franja {franja}); estimado total {estimado:.1f} h")


# ═════════════════════════════════════════════════════════════════════════════
# Tabla de UB-DP (la política B3 que se deriva de ella vive en baselines/politicas.py)
# ═════════════════════════════════════════════════════════════════════════════

class TablaDP:
    """Tablas V (K fijo y caso libre) de UB-DP y acceso en línea para B3.

    UB-DP no es una cota: es el valor de la política que ignora el mercado (ver el docstring del módulo).

    Atributos
    ---------
    V_k    : (K+1, 7, H, N, Ns) float32 — tabla con dimensión k (k = 0 sin uso).
    V_libre: (7, H, N, Ns) float32 — tabla sin dimensión k.
    V0_k   : (K+1, 7, H) float32 — valor de la primera decisión desde casa, por k.
    V0_se_k: (K+1, 7, H) float32 — error estándar de muestreo de V0_k.
    V0_libre, V0_se_libre : (7, H) float32.
    tiempo_calculo : segundos que tardó calcular (0.0 si se cargó de disco).
    """

    def __init__(self, time_matrix, V_k, V_libre, V0_k, V0_se_k, V0_libre, V0_se_libre,
                 dt=UBDP_DT_HORAS, S=UBDP_S, tiempo_calculo=0.0):
        self.K = V_k.shape[0] - 1
        self.dt, self.S = dt, S
        self.Ns = V_libre.shape[-1]
        self.V_k, self.V_libre = V_k, V_libre
        self.V0_k, self.V0_se_k = V0_k, V0_se_k
        self.V0_libre, self.V0_se_libre = V0_libre, V0_se_libre
        self.tiempo_calculo = tiempo_calculo
        self.T = np.asarray(time_matrix, dtype=float)

    # ── UB-DP ────────────────────────────────────────────────────────────────

    def ub_dp(self, nodo_inicio: int, dia_semana: int, K=None) -> float:
        """UB-DP: valor aproximado de la mejor política que ignora el estado del mercado (no es cota).

        K = None → caso libre; K = entero → tabla de K arcos (K ≤ K de la tabla).
        """
        if K is None:
            return float(self.V0_libre[dia_semana, nodo_inicio])
        self._chequear_k(K)
        return float(self.V0_k[K, dia_semana, nodo_inicio])

    def ub_dp_se(self, nodo_inicio: int, dia_semana: int, K=None) -> float:
        """Error estándar de muestreo (S sorteos) de la raíz de ub_dp."""
        if K is None:
            return float(self.V0_se_libre[dia_semana, nodo_inicio])
        self._chequear_k(K)
        return float(self.V0_se_k[K, dia_semana, nodo_inicio])

    def _chequear_k(self, K):
        if not (2 <= K <= self.K):
            raise ValueError(f"K={K} fuera de la tabla (calculada para 2 ≤ K ≤ {self.K})")

    # ── B3 ───────────────────────────────────────────────────────────────────

    def valor_continuacion(self, dia_semana: int, casa: int, nodos: np.ndarray,
                           tiempos: np.ndarray, k_restante) -> np.ndarray:
        """V(nodos, tiempos, k_restante) para el hogar `casa`; k_restante=None ⇒ caso libre."""
        s = np.minimum(np.floor(tiempos / self.dt + 1e-9).astype(np.int64), self.Ns - 1)
        if k_restante is None:
            return self.V_libre[dia_semana, casa, nodos, s]
        return self.V_k[k_restante, dia_semana, casa, nodos, s]


# ═════════════════════════════════════════════════════════════════════════════
# Tableros de la DP
# ═════════════════════════════════════════════════════════════════════════════

def franjas_rechazo(dt: float) -> int:
    """τ: franjas que consume un rechazo (COTIZACION_HORAS redondeado hacia abajo, como los viajes)."""
    return int(np.floor(COTIZACION_HORAS / dt + 1e-9))


def _sortear_tableros(market, distance_arr, diesel_arr, S, seed) -> dict:
    """Tableros sin memoria de S sorteos por (día de la semana, nodo) y las tablas por día real.

    Por cada día de la semana: un día de entrenamiento al azar con ese día de la semana y las cargas
    sorteadas de ese día (lo mismo que `MarketProcess.tablero_sin_memoria`, δ = 0). Devuelve
      A   (7, N, S, N) bool    — la lane tiene carga en el sorteo
      D   (7, N, S) int64      — índice local del día sorteado (en dias[dow])
      Rk  (7, Dmax, N, N, K)   — lane_reward cobrando el nivel k con carga, por día real (float32)
      P   (7, Dmax, N, K)      — P_k = (1 − u_k)^m del nodo de origen ese día (float32)
      R0  (N, N)               — lane_reward sin carga (VACÍO)
      dias: lista de 7 arrays con los días de entrenamiento de cada día de la semana
    """
    N = market.num_nodes
    rng = np.random.Generator(np.random.PCG64(seed))
    dias = [market.dias_train_por_dow(d) for d in range(7)]
    Dmax = max(len(d) for d in dias)
    A = np.empty((7, N, S, N), dtype=bool)
    D = np.empty((7, N, S), dtype=np.int64)
    Rk = np.zeros((7, Dmax, N, N, N_NIVELES), dtype=np.float32)
    P = np.zeros((7, Dmax, N, N_NIVELES), dtype=np.float32)
    nodos = np.repeat(np.arange(N), S)
    for dow in range(7):
        idx = rng.integers(0, len(dias[dow]), size=N * S)
        loads = market.sortear_cargas(nodos, dias[dow][idx], rng)
        A[dow] = (loads > 0).reshape(N, S, N)
        D[dow] = idx.reshape(N, S)
        for n_d, d in enumerate(dias[dow]):
            precios = precios_niveles(market.cuantiles_dia(int(d)))                        # (N, N, K)
            Rk[dow, n_d] = lane_reward(precios, 1, distance_arr[..., None], diesel_arr[..., None])
            m = market.competidores(np.arange(N), np.full(N, int(d)))                      # (N,)
            P[dow, n_d] = prob_ganar(U_NIVELES_ARR[None, :], m[:, None])
    R0 = lane_reward(np.zeros_like(distance_arr), 0, distance_arr, diesel_arr).astype(np.float32)
    return dict(A=A, D=D, Rk=Rk, P=P, R0=R0, dias=dias)


def _firma(market, time_matrix, distance_arr, diesel_arr, K, S, dt, seed) -> str:
    """Huella de todo lo que determina la tabla (decide si un .npz guardado sirve)."""
    h = hashlib.sha256()
    h.update(market.firma_params().encode())          # cargas, cuantiles, LTR, κ, M_MAX, LTR_REF, U_NIVELES
    for a in (time_matrix, distance_arr, diesel_arr):
        h.update(np.ascontiguousarray(a).tobytes())
    h.update(repr((K, S, dt, seed, MAX_DURATION, market.num_nodes, "cotizacion-2-intentos",
                   tuple(U_NIVELES), COTIZACION_HORAS, COTIZACIONES_MAX_POR_NODO)).encode())
    return h.hexdigest()


# ═════════════════════════════════════════════════════════════════════════════
# Cálculo de la tabla
# ═════════════════════════════════════════════════════════════════════════════

_BARRIDO_CAND_BYTES = 1.5e9   # presupuesto de memoria de los tensores (H_chunk, I, S, J) float32 del barrido
_TENSORES_POR_BLOQUE = 5      # acumulados de los dos intentos, W, el candidato y el reward del nivel


def _h_chunk(N, S):
    """Tamaño de bloque de H tal que los ~5 tensores (h_chunk, N, S, N) float32 del barrido no pasen de
    `_BARRIDO_CAND_BYTES`. Sin partir por H, a ~75-100 nodos ya no caben en una GPU de 16 GB (OOM confirmado
    en producción). Partir por H da el mismo resultado con memoria pico acotada e independiente de N."""
    bytes_por_h = _TENSORES_POR_BLOQUE * N * S * N * 4
    return max(1, min(N, int(_BARRIDO_CAND_BYTES // bytes_por_h)))


class _Reloj:
    """Avance del cálculo y corte por tiempo (UBDP_TIEMPO_MAX_HORAS)."""

    def __init__(self, total: int, max_horas: float):
        self.t0, self.total, self.hechas, self.max_horas = time.time(), total, 0, max_horas

    def horas(self) -> float:
        return (time.time() - self.t0) / 3600.0

    def avanzar(self, franja: int):
        self.hechas += 1
        if self.max_horas is not None and self.horas() > self.max_horas:
            raise UBDPNoTratable("tiempo", self.horas(), self.hechas / self.total, franja)


def _mejor_por_niveles(Rk_d, P_d, D_dw, filas, C, respaldo, valido, K):
    """max_k [P_k (r_k + C) + (1 − P_k) respaldo] en (bloque, I, S, J), −∞ (_NEG) donde no es válido.

    Rk_d (Dmax, I, J, K) y P_d (Dmax, I, K) del día de la semana; D_dw (I, S) día sorteado; filas (I, 1) el
    nodo de origen de cada fila; C (bloque, I, 1, J) continuación; respaldo broadcastable a (bloque, I, S, J).
    """
    acc = None
    for k in range(K):
        r = Rk_d[D_dw, filas, :, k]                                # (I, S, J)
        p = P_d[D_dw, filas, k][..., None]                         # (I, S, 1)
        v = p * (r[None] + C) + (1.0 - p) * respaldo
        acc = v if acc is None else torch.maximum(acc, v)
    return acc.masked_fill(~valido, _NEG)


def _despues_de_perder(A2, E2):
    """W_j = max(E, max_{j' ≠ j} A_j'): con el top-2 sobre j (el mejor si j no es el mejor; si no, el segundo)."""
    top = A2.topk(2, dim=-1)
    jj = torch.arange(A2.shape[-1], device=A2.device)
    W = torch.where(jj == top.indices[..., :1], top.values[..., 1:2], top.values[..., :1])
    return torch.maximum(W, E2)


def _barrido(V, V_prev, tab, f, ok_est, base, smax, dt, tau, reloj):
    """Rellena V[d0, h, i, s] hacia atrás en s con los dos intentos (ver el docstring del módulo).

    V_prev : tabla del nivel k−1 (o None para k = 1, solo retorno). Si es el mismo tensor que V (caso
             libre) se lee de franjas posteriores, ya calculadas (f ≥ 1).
    """
    N = f.shape[0]
    S = tab["A"].shape[2]
    K = tab["Rk"].shape[-1]
    dev = f.device
    jj = torch.arange(N, device=dev)
    filas = jj[:, None]                                            # (I, 1)
    casa = (jj[None, :] == jj[:, None])                            # (H, J): j == h
    h_chunk = _h_chunk(N, S)
    E = tab["R0"].T                                                # (H, I): volver a casa vacío desde i
    for s in range(smax, -1, -1):
        q = int((s * dt) // _DIA_HORAS)                            # días enteros transcurridos
        s2 = min(s + tau, smax)
        ok1 = ok_est & (s + base <= smax)                          # (H, I, J) destino intermedio factible
        ok2 = ok_est & (s2 + base <= smax)
        for d0 in range(7):
            dw = (d0 + q) % 7
            Ad, Dd = tab["A"][dw], tab["D"][dw]                    # (I, S, J), (I, S)
            Rk_d, P_d = tab["Rk"][dw], tab["P"][dw]
            if V_prev is not None:
                Vn1 = V_prev[d0][:, jj[None, :], torch.clamp(s + f, max=smax)]    # (H, I, J)
                Vn2 = V_prev[d0][:, jj[None, :], torch.clamp(s2 + f, max=smax)]
            for h0 in range(0, N, h_chunk):
                h1 = min(h0 + h_chunk, N)
                cs = casa[h0:h1][:, None, None, :]                 # (hc, 1, 1, J)
                Ec = E[h0:h1][:, :, None, None]                    # (hc, I, 1, 1)
                if V_prev is None:                                 # k = 1: solo el retorno a casa
                    C1 = C2 = torch.zeros(h1 - h0, N, 1, N, device=dev)
                    v1 = v2 = Ad[None] & cs
                else:
                    C1 = torch.where(cs, 0.0, Vn1[h0:h1][:, :, None, :])          # V(casa) = 0
                    C2 = torch.where(cs, 0.0, Vn2[h0:h1][:, :, None, :])
                    v1 = Ad[None] & (ok1[h0:h1][:, :, None, :] | cs)
                    v2 = Ad[None] & (ok2[h0:h1][:, :, None, :] | cs)
                A2 = _mejor_por_niveles(Rk_d, P_d, Dd, filas, C2, Ec, v2, K)        # segundo intento
                W = _despues_de_perder(A2, Ec)
                del A2
                A1 = _mejor_por_niveles(Rk_d, P_d, Dd, filas, C1, W, v1, K)         # primer intento
                mejor = torch.maximum(A1.amax(dim=-1), Ec[..., 0])                  # (hc, I, S)
                V[d0, h0:h1, :, s] = mejor.mean(dim=-1)
        reloj.avanzar(s)


def _raiz(V_prev, tab, f, smax, tau, dev):
    """V0[d0, h] y su error estándar: primera decisión desde casa (s = 0), con los dos intentos.

    VACÍO puede ir a cualquier destino factible (primer movimiento), así que E = max_j R0[h, j] + V(j, f_hj).
    """
    N = f.shape[0]
    K = tab["Rk"].shape[-1]
    h_idx = torch.arange(N, device=dev)
    filas = h_idx[:, None]
    noeye = ~torch.eye(N, dtype=torch.bool, device=dev)
    j_ok1 = ((f + f.T) <= smax) & noeye                             # ida y vuelta posibles en tiempo
    j_ok2 = ((tau + f + f.T) <= smax) & noeye                       # … después de un rechazo
    V0 = torch.empty(7, N, device=dev)
    se = torch.empty(7, N, device=dev)
    R0 = tab["R0"]                                                 # (H, J): fila de casa
    for d0 in range(7):
        Vr1 = V_prev[d0][h_idx[:, None], h_idx[None, :], f.clamp(max=smax)]            # (H, J)
        Vr2 = V_prev[d0][h_idx[:, None], h_idx[None, :], (tau + f).clamp(max=smax)]
        E1 = (R0 + Vr1).masked_fill(~j_ok1, _NEG).amax(dim=-1)                         # (H,)
        E2 = (R0 + Vr2).masked_fill(~j_ok2, _NEG).amax(dim=-1)
        Ad = tab["A"][d0][h_idx]                                    # (H, S, J): tablero sorteado en casa
        Dd = tab["D"][d0][h_idx]                                    # (H, S)
        Rk_d, P_d = tab["Rk"][d0], tab["P"][d0]
        # Mismas funciones que el barrido, con bloque = 1 (I = H es la casa misma).
        v2 = (Ad & j_ok2[:, None, :])[None]
        v1 = (Ad & j_ok1[:, None, :])[None]
        A2 = _mejor_por_niveles(Rk_d, P_d, Dd, filas, Vr2[None, :, None, :], E2[None, :, None, None], v2, K)
        W = _despues_de_perder(A2, E2[None, :, None, None])
        A1 = _mejor_por_niveles(Rk_d, P_d, Dd, filas, Vr1[None, :, None, :], W, v1, K)
        mejor = torch.maximum(A1.amax(dim=-1), E1[None, :, None])[0]                 # (H, S)
        V0[d0] = mejor.mean(dim=-1)
        se[d0] = mejor.std(dim=-1) / np.sqrt(mejor.shape[-1])
    sin_ruta = ~j_ok1.any(dim=1)                                   # casas sin ningún viaje factible
    V0[:, sin_ruta] = float("nan")
    se[:, sin_ruta] = float("nan")
    return V0, se


def calcular_tabla(market, time_matrix, distance_arr, diesel_arr, K=K_MAX,
                   S=UBDP_S, dt=UBDP_DT_HORAS, seed=UBDP_SEED, device=None,
                   max_horas=UBDP_TIEMPO_MAX_HORAS) -> TablaDP:
    """Calcula V (K fijo y caso libre) y V0 con los dos intentos de cotización.

    K = 0 calcula solo el caso libre (lo que usa la evaluación con K_RESTRINGIDO = False): un barrido en vez
    de K + 1. La firma incluye K, así que una tabla con K = 0 no se confunde con una de K = K_MAX.

    Lanza UBDPNoTratable si pasa de `max_horas` (None = sin tope)."""
    if COTIZACIONES_MAX_POR_NODO != 2:
        raise NotImplementedError("la DP de UB-DP está escrita para dos cotizaciones por visita")
    dev = device or DEVICE
    N = market.num_nodes
    T = np.asarray(time_matrix, dtype=float)
    dist = np.asarray(distance_arr, dtype=float)
    diesel = np.asarray(diesel_arr, dtype=float)
    smax = int(np.floor(MAX_DURATION / dt + 1e-9))
    Ns = smax + 1
    tau = franjas_rechazo(dt)
    f_np = np.floor(T / dt + 1e-9).astype(np.int64)          # franjas de viaje, redondeo hacia abajo
    if f_np[~np.eye(N, dtype=bool)].min() < 1:
        raise ValueError("Hay tiempos de viaje menores a un paso de la DP: la recursión del caso "
                         "libre no estaría bien fundada. Reduce UBDP_DT_HORAS.")
    reloj = _Reloj(total=(K + 1) * Ns, max_horas=max_horas)

    tab_np = _sortear_tableros(market, dist, diesel, S, seed)
    tab = {k: torch.from_numpy(tab_np[k]).to(dev) for k in ("A", "D", "Rk", "P", "R0")}
    f = torch.from_numpy(f_np).to(dev)
    eye = torch.eye(N, dtype=torch.bool, device=dev)
    # Destinos que cuentan como "continuar": j ≠ i y j ≠ h (el retorno va aparte).
    ok_est = (~eye)[None, :, :] & (~eye)[:, None, :]         # (H, I, J)
    base = f[None, :, :] + f.T[:, None, :]                   # (H, I, J) = f_ij + f_jh

    # Tabla con dimensión k: V_k[k] se apoya en V_k[k−1]; k = 1 es solo retorno.
    V_k = torch.zeros(K + 1, 7, N, N, Ns, device=dev)
    V0_k = torch.full((K + 1, 7, N), float("nan"), device=dev)
    V0_se_k = torch.full((K + 1, 7, N), float("nan"), device=dev)
    for k in range(1, K + 1):
        _barrido(V_k[k], V_k[k - 1] if k > 1 else None, tab, f, ok_est, base, smax, dt, tau, reloj)
        if k >= 2:
            V0_k[k], V0_se_k[k] = _raiz(V_k[k - 1], tab, f, smax, tau, dev)

    # Caso libre: misma recursión sin k; f ≥ 1 garantiza que se lee de franjas ya calculadas.
    V_libre = torch.zeros(7, N, N, Ns, device=dev)
    _barrido(V_libre, V_libre, tab, f, ok_est, base, smax, dt, tau, reloj)
    V0_libre, V0_se_libre = _raiz(V_libre, tab, f, smax, tau, dev)

    if dev.type == "cuda":
        torch.cuda.synchronize()
    return TablaDP(
        T, V_k.cpu().numpy(), V_libre.cpu().numpy(),
        V0_k.cpu().numpy(), V0_se_k.cpu().numpy(),
        V0_libre.cpu().numpy(), V0_se_libre.cpu().numpy(),
        dt=dt, S=S, tiempo_calculo=reloj.horas() * 3600.0,
    )


def _es_falta_de_memoria(e: BaseException) -> bool:
    return isinstance(e, MemoryError) or isinstance(e, getattr(torch.cuda, "OutOfMemoryError", ())) or \
        (isinstance(e, RuntimeError) and "out of memory" in str(e).lower())


def cargar_o_calcular_tabla(market, time_matrix, distance_arr, diesel_arr, K=K_MAX,
                            S=UBDP_S, dt=UBDP_DT_HORAS, seed=UBDP_SEED, device=None,
                            path=None, verbose=True, max_horas=UBDP_TIEMPO_MAX_HORAS):
    """La tabla guardada si existe y coincide con los datos; si no, la calcula y la guarda.

    Devuelve None si UB-DP es "no tratable" a este tamaño: más de UBDP_MAX_NODOS nodos (no se intenta), o el
    cálculo pasó de `max_horas` o se quedó sin memoria (ahora o en un intento anterior con la misma firma,
    según el marcador). B3 y UB-DP se reportan entonces como "no tratable"."""
    if market.num_nodes > UBDP_MAX_NODOS:
        if verbose:
            print(f"UB-DP: no tratable a {market.num_nodes} nodos (más de UBDP_MAX_NODOS = {UBDP_MAX_NODOS}); "
                  f"no se calcula.")
        return None
    path = path or tabla_path(market.num_nodes)
    marcador = path.replace(".npz", ".no_tratable.json")
    firma = _firma(market, time_matrix, distance_arr, diesel_arr, K, S, dt, seed)
    if os.path.exists(path):
        with np.load(path) as z:
            if str(z["firma"]) == firma:
                if verbose:
                    print(f"UB-DP: tabla V cargada de {path}")
                return TablaDP(
                    time_matrix, z["V_k"], z["V_libre"], z["V0_k"], z["V0_se_k"],
                    z["V0_libre"], z["V0_se_libre"], dt=dt, S=S,
                    tiempo_calculo=float(z["tiempo_calculo"]),
                )
        if verbose:
            print(f"UB-DP: {path} no coincide con los datos actuales; se recalcula.")
    if os.path.exists(marcador):
        with open(marcador, encoding="utf-8") as fh:
            info = json.load(fh)
        if info.get("firma") == firma:
            if verbose:
                print(f"UB-DP: no tratable a {market.num_nodes} nodos ({info['motivo']}, {marcador}).")
            return None
    if verbose:
        print(f"UB-DP: calculando la tabla V ({market.num_nodes} nodos, S={S}, dt={dt} h, "
              f"tope {max_horas} h)...", flush=True)
    t0 = time.time()
    try:
        tabla = calcular_tabla(market, time_matrix, distance_arr, diesel_arr, K, S, dt, seed, device, max_horas)
    except Exception as e:                                                   # noqa: BLE001
        if isinstance(e, UBDPNoTratable):
            info = dict(motivo=e.motivo, horas=e.horas, fraccion=e.fraccion, franja=e.franja,
                        horas_estimadas_total=e.horas_estimadas)
        elif _es_falta_de_memoria(e):
            info = dict(motivo="memoria", horas=(time.time() - t0) / 3600.0, error=str(e)[:300])
        else:
            raise
        info.update(num_nodes=market.num_nodes, firma=firma, S=S, dt=dt, K=K)
        os.makedirs(os.path.dirname(marcador), exist_ok=True)
        with open(marcador, "w", encoding="utf-8") as fh:
            json.dump(info, fh, indent=2)
        if verbose:
            print(f"UB-DP: no tratable a {market.num_nodes} nodos ({info['motivo']}) → {marcador}")
        return None
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(
        path, firma=firma, V_k=tabla.V_k, V_libre=tabla.V_libre,
        V0_k=tabla.V0_k, V0_se_k=tabla.V0_se_k, V0_libre=tabla.V0_libre,
        V0_se_libre=tabla.V0_se_libre, tiempo_calculo=tabla.tiempo_calculo,
    )
    if verbose:
        print(f"UB-DP: tabla V calculada en {tabla.tiempo_calculo:.1f} s → {path}")
    return tabla


def main():
    """Calcula (o confirma) la tabla de UB-DP de N nodos en un proceso aparte, antes de entrenar."""
    import argparse
    from market_process import MarketProcess, asegurar_parametros
    from problem_data import load_matrices
    ap = argparse.ArgumentParser(description="Tabla V de UB-DP (con los dos intentos de cotización).")
    ap.add_argument("--num-nodes", type=int, required=True)
    ap.add_argument("--max-horas", type=float, default=UBDP_TIEMPO_MAX_HORAS)
    args = ap.parse_args()
    asegurar_parametros()
    tm, _, _, dist, diesel, *_ = load_matrices(args.num_nodes)
    tabla = cargar_o_calcular_tabla(MarketProcess(args.num_nodes), tm, dist, diesel, max_horas=args.max_horas)
    print("no tratable" if tabla is None else f"lista ({tabla.tiempo_calculo / 3600:.2f} h)")


if __name__ == "__main__":
    main()
