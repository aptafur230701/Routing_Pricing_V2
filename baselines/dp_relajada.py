"""
baselines/dp_relajada.py
==========================
UB-DP y B3 (baselines_drl_ruteo_estocastico.md): programación dinámica sobre el estado
(nodo, hora, k). El camión puede revisitar nodos intermedios, así que el estado no lleva el
conjunto de visitados y la ecuación de Bellman es la del problema, no una relajación de él.

    V(i, t, k) = E_ω [ max_{j ∈ A(i,t,k,ω)}  r_ij(ω) + V(j, t + τ_ij, k − 1) ]

ω = (loads_row, rate_row) que sortea `MarketProcess.sample_arrivals` al llegar a i;
A son los destinos factibles según las reglas de la máscara del entorno: lane con carga,
retorno a casa posible dentro de MAX_DURATION y retorno forzado cuando queda un arco.
Volver a casa da la recompensa del arco y termina con valor cero. La recompensa es
`lane_reward`, sin shaping.

UB-DP es, por tanto, el valor aproximado (salvo la discretización de 15 min y el muestreo con
S = 5000 sorteos) de la mejor política que IGNORA EL ESTADO DEL MERCADO: conoce la distribución
de cargas y tarifas pero no el índice del nodo ni su persistencia (`sample_arrivals` sortea un
lento y un rápido estacionario nuevos en cada llegada). No es una cota para una política que
observa el índice, que puede superarla. Una diferencia menor con el entorno: allí, volver al
mismo nodo el mismo día ve la MISMA fila del mundo (es función de la semilla del mundo, el
nodo y el día), mientras que la DP sortea una fila nueva en cada llegada. UB-PI
(Solvers.solve_ub_pi) es la cota con información perfecta y debe quedar por encima de UB-DP.

Estado de la tabla: V[k, d0, h, j, s]
  · k  : arcos restantes (incluido el retorno); solo en la tabla de K fijo.
  · d0 : día de la semana del día de salida (lunes = 0). El día de la semana al llegar
         a la franja s es (d0 + s·dt // 14) % 7.
  · h  : nodo de casa. Entra en el estado porque el retorno y su factibilidad dependen
         de h (el documento lo omite al contar ~460.000 estados).
  · j  : nodo actual (tras sortear la fila de llegada).
  · s  : franja de tiempo, t = s · dt.

Discretización y muestreo: los tiempos de viaje se redondean hacia abajo a la franja
(redondear hacia arriba eliminaría rutas factibles); por eso el valor es aproximado: una
llegada puede adelantarse al día anterior. Con S sorteos por estado, el error de muestreo
dentro del máximo sesga V ligeramente hacia arriba.

El primer movimiento desde casa no aplica la máscara de lanes (se puede salir hacia una
lane sin carga), por eso V0[k, d0, h] se calcula aparte, con la fila sorteada en h y todos
los destinos factibles.

El cálculo corre en `DEVICE` (torch): son ~10^11 operaciones elementales, que en CPU con
numpy tardarían horas.
"""

import hashlib
import os
import time

import numpy as np
import torch

from config import (
    DEVICE, MAX_DURATION, K_MAX,
    UBDP_DT_HORAS, UBDP_S, UBDP_SEED, UBDP_TABLA_PATH,
)
from problem_data import lane_reward

_CWD = os.path.dirname(os.path.abspath(__file__))
_NEG = -1e9          # "menos infinito" finito: evita NaN en sumas
_DIA_HORAS = 14.0    # un día de mercado = 14 h (igual que RoutingEnv)


def tabla_path(num_nodes: int) -> str:
    """Ruta absoluta del .npz con la tabla V de `num_nodes` nodos."""
    return os.path.join(_CWD, UBDP_TABLA_PATH.format(n=num_nodes))


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
# Cálculo de la tabla
# ═════════════════════════════════════════════════════════════════════════════

def _sortear_filas(market, distance_arr, diesel_arr, S, seed):
    """Recompensas y disponibilidad de S filas por cada par (nodo, día de la semana).

    Se sortean una vez con un Generator propio y se reutilizan en todos los estados.
    Devuelve R (7, N, S, N) float32 y A (7, N, S, N) bool.
    """
    N = market.num_nodes
    rng = np.random.Generator(np.random.PCG64(seed))
    R = np.empty((7, N, S, N), dtype=np.float32)
    A = np.empty((7, N, S, N), dtype=bool)
    nodos = np.repeat(np.arange(N), S)
    for dow in range(7):
        # Cualquier índice de día cuyo día de la semana sea `dow`.
        dia = int((dow - int(market.dia_semana(0))) % 7)
        loads, rates = market.sample_arrivals(nodos, np.full(N * S, dia), rng)
        rew = lane_reward(rates, loads, distance_arr[nodos], diesel_arr[nodos])
        R[dow] = rew.reshape(N, S, N)
        A[dow] = (loads > 0).reshape(N, S, N)
    return R, A


def _firma(market, time_matrix, distance_arr, diesel_arr, K, S, dt, seed) -> str:
    """Huella de todo lo que determina la tabla (decide si un .npz guardado sirve)."""
    h = hashlib.sha256()
    h.update(market.firma_params().encode())          # cargas y tarifas (r, media_lane, lento, ρ, σ, ...)
    for a in (time_matrix, distance_arr, diesel_arr):
        h.update(np.ascontiguousarray(a).tobytes())
    h.update(repr((K, S, dt, seed, MAX_DURATION, market.num_nodes)).encode())
    return h.hexdigest()


_BARRIDO_CAND_BYTES = 1.5e9   # presupuesto de memoria para `cand` (H_chunk, I, S, J) float32


def _h_chunk(N, S):
    """Tamaño de bloque de H tal que (h_chunk, N, S, N) float32 no pase de `_BARRIDO_CAND_BYTES`.

    `cand` crece como N²·S (ver docstring de `_barrido`); sin partir por H, a partir de ~75-100
    nodos el tensor completo ya no cabe en una GPU de 16 GB (confirmado en producción: OOM a 75 y
    100 nodos, cand de 7.86 GiB y 18.63 GiB respectivamente). Partir por H da el mismo resultado
    (es la misma suma, solo que en bloques) con memoria pico acotada e independiente de N.
    """
    bytes_por_h = N * S * N * 4
    return max(1, min(N, int(_BARRIDO_CAND_BYTES // bytes_por_h)))


def _barrido(V, V_prev, R, A, f, ok_est, base, smax, dt):
    """Rellena V[d0, h, i, s] hacia atrás en s (bucle de Bellman de una tabla).

    V_prev : tabla del nivel k−1 (o None para k = 1, solo retorno). Si es la misma
             tensor que V (caso libre) se lee de franjas posteriores, ya calculadas.

    El candidato (H, I, S, J) se calcula en bloques de H (ver `_h_chunk`) para no materializar
    nunca el tensor completo: su tamaño crece como N²·S y a N ≳ 75 ya no cabe en una GPU de 16 GB.
    """
    N = f.shape[0]
    S = R.shape[2]
    jj = torch.arange(N, device=f.device)[None, :]            # (1, J) → broadcast con (I, J)
    h_chunk = _h_chunk(N, S)
    for s in range(smax, -1, -1):
        q = int((s * dt) // _DIA_HORAS)                      # días enteros transcurridos
        ok = ok_est & (s + base <= smax)                     # (H, I, J) destino factible
        s_sig = torch.clamp(s + f, max=smax)                 # (I, J) franja de llegada a j
        for d0 in range(7):
            dw = (d0 + q) % 7
            Rd = R[dw]                                       # (I, S, J)
            # Retorno a casa: siempre disponible, recompensa del arco i→h (con o sin carga).
            mejor = Rd.permute(2, 0, 1).clone()              # (H, I, S): R[i, :, h]
            if V_prev is not None:
                Vn = V_prev[d0][:, jj, s_sig]                # (H, I, J) = V(j, s + f_ij) del hogar h
                for h0 in range(0, N, h_chunk):
                    h1 = min(h0 + h_chunk, N)
                    cand = Rd[None] + Vn[h0:h1, :, None, :]  # (h_chunk, I, S, J)
                    cand.masked_fill_(~(ok[h0:h1, :, None, :] & A[dw][None]), _NEG)
                    mejor[h0:h1] = torch.maximum(cand.amax(dim=-1), mejor[h0:h1])
            V[d0, :, :, s] = mejor.mean(dim=-1)


def _raiz(V_prev, R, f, smax, dev):
    """V0[d0, h] y su error estándar: primera decisión desde casa, sin máscara de lanes."""
    N = f.shape[0]
    h_idx = torch.arange(N, device=dev)
    j_ok = (f + f.T) <= smax                                 # ida y vuelta posibles en tiempo
    j_ok = j_ok & ~torch.eye(N, dtype=torch.bool, device=dev)
    V0 = torch.empty(7, N, device=dev)
    se = torch.empty(7, N, device=dev)
    for d0 in range(7):
        Rh = R[d0][h_idx, :, :]                              # (H, S, J): fila sorteada en h
        Vn = V_prev[d0][h_idx[:, None], h_idx[None, :], f.clamp(max=smax)]   # (H, J): V(j, f_hj) del hogar h
        cand = Rh + Vn[:, None, :]
        cand = cand.masked_fill(~j_ok[:, None, :], _NEG)
        mejor = cand.amax(dim=-1)                            # (H, S)
        V0[d0] = mejor.mean(dim=-1)
        se[d0] = mejor.std(dim=-1) / np.sqrt(mejor.shape[-1])
    sin_ruta = ~j_ok.any(dim=1)                              # casas sin ningún viaje factible
    V0[:, sin_ruta] = float("nan")
    se[:, sin_ruta] = float("nan")
    return V0, se


def calcular_tabla(market, time_matrix, distance_arr, diesel_arr, K=K_MAX,
                   S=UBDP_S, dt=UBDP_DT_HORAS, seed=UBDP_SEED, device=None) -> TablaDP:
    """Calcula V (K fijo y caso libre) y V0 con la ecuación de Bellman del problema."""
    dev = device or DEVICE
    N = market.num_nodes
    T = np.asarray(time_matrix, dtype=float)
    dist = np.asarray(distance_arr, dtype=float)
    diesel = np.asarray(diesel_arr, dtype=float)
    smax = int(np.floor(MAX_DURATION / dt + 1e-9))
    Ns = smax + 1
    f_np = np.floor(T / dt + 1e-9).astype(np.int64)          # franjas de viaje, redondeo hacia abajo
    if f_np[~np.eye(N, dtype=bool)].min() < 1:
        raise ValueError("Hay tiempos de viaje menores a un paso de la DP: la recursión del caso "
                         "libre no estaría bien fundada. Reduce UBDP_DT_HORAS.")
    t0 = time.time()

    R_np, A_np = _sortear_filas(market, dist, diesel, S, seed)
    R = torch.from_numpy(R_np).to(dev)                       # (7, N, S, N)
    A = torch.from_numpy(A_np).to(dev)
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
        _barrido(V_k[k], V_k[k - 1] if k > 1 else None, R, A, f, ok_est, base, smax, dt)
        if k >= 2:
            V0_k[k], V0_se_k[k] = _raiz(V_k[k - 1], R, f, smax, dev)

    # Caso libre: misma recursión sin k; f ≥ 1 garantiza que se lee de franjas ya calculadas.
    V_libre = torch.zeros(7, N, N, Ns, device=dev)
    _barrido(V_libre, V_libre, R, A, f, ok_est, base, smax, dt)
    V0_libre, V0_se_libre = _raiz(V_libre, R, f, smax, dev)

    if dev.type == "cuda":
        torch.cuda.synchronize()
    return TablaDP(
        T, V_k.cpu().numpy(), V_libre.cpu().numpy(),
        V0_k.cpu().numpy(), V0_se_k.cpu().numpy(),
        V0_libre.cpu().numpy(), V0_se_libre.cpu().numpy(),
        dt=dt, S=S, tiempo_calculo=time.time() - t0,
    )


def cargar_o_calcular_tabla(market, time_matrix, distance_arr, diesel_arr, K=K_MAX,
                            S=UBDP_S, dt=UBDP_DT_HORAS, seed=UBDP_SEED, device=None,
                            path=None, verbose=True) -> TablaDP:
    """Devuelve la tabla guardada si existe y coincide con los datos; si no, la calcula y la guarda."""
    path = path or tabla_path(market.num_nodes)
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
    if verbose:
        print(f"UB-DP: calculando la tabla V ({market.num_nodes} nodos, S={S}, dt={dt} h)...", flush=True)
    tabla = calcular_tabla(market, time_matrix, distance_arr, diesel_arr, K, S, dt, seed, device)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(
        path, firma=firma, V_k=tabla.V_k, V_libre=tabla.V_libre,
        V0_k=tabla.V0_k, V0_se_k=tabla.V0_se_k, V0_libre=tabla.V0_libre,
        V0_se_libre=tabla.V0_se_libre, tiempo_calculo=tabla.tiempo_calculo,
    )
    if verbose:
        print(f"UB-DP: tabla V calculada en {tabla.tiempo_calculo:.1f} s → {path}")
    return tabla
