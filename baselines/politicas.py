"""
baselines/politicas.py
========================
Políticas baseline no anticipativas: B1, B2, B2P (B2 con pronósticos del índice de mercado) y B3
(política de un paso sobre la tabla V de UB-DP, baselines_drl_ruteo_estocastico.md), en tensores de torch.

Todas deciden para un LOTE de episodios a la vez en el dispositivo del entorno (la GPU en el servidor): en cada
paso reciben una `ObsLote` (máscara, tablero observado: cargas, precios por nivel y competencia m, nodo actual, hora
y k restante de cada fila) y `decidir` devuelve el DESTINO y la OPCIÓN de precio de cada fila. B1, B2 y B2P cotizan
con una regla fija (`cotizar_regla`, por defecto REGLA_PRECIO_BASELINES = miope; especificacion_fase0_ruteo_pricing.md
§6-7). Si la carga se pierde, la política vuelve a decidir en el mismo nodo con la lane enmascarada. El destino puede
ser el propio nodo: esperar al siguiente día de mercado (routing_env.py), o un nodo a ≤ REPOSICION_MAX_MILLAS para
reposicionarse vacío.

Cada lane tiene varias cargas: si se pierde una, se puede cotizar otra (pricing.valor_reintentos). El precio cuenta
lo que cuesta perderlas todas (`cotizar_regla`): en la vuelta a casa es volver vacío (−c), no 0. B3 cotiza con el
óptimo de su propia tabla (decide destino y precio a la vez).

El valor inmediato de un arco es el esperado con la cotización miope, max_k P_k·(p_k·d − c) con
P_k = (1 − u_k)^m (`valor_inmediato`); en la vuelta a casa, max_k P_k·p_k·d − c (perder es volver vacío); −c si la
lane solo se puede recorrer VACÍA, 0 si es la espera. Ninguna política recibe el mundo evaluado ni ve el competidor.

B2 resuelve su recursión de una vez como una tabla hacia atrás (no depende del mundo); B2P la recalcula en cada
decisión porque depende del índice de hoy, para todas las filas del lote a la la vez. Los valores son los mismos que
los de la recursión memoizada (y los empates se rompen igual: el destino de índice menor).
"""

from dataclasses import dataclass

import numpy as np
import torch

from config import (
    MAX_DURATION, UBDP_DT_HORAS, MARKET_DIAS_EPISODIO, TRAIN_DAYS, BE_MARGEN,
    REGLA_PRECIO_BASELINES, COTIZACION_HORAS, COTIZACIONES_MAX_POR_NODO,
)
from pricing import (
    N_NIVELES, U_NIVELES_ARR, indice_nivel, prob_ganar, lane_cost_row, lane_cost_t, lane_reward_t, precios_niveles,
    prob_ganar_t, valor_reintentos, valor_reintentos_t,
)
from routing_env import fin_espera
from baselines.dp_relajada import TablaDP, PENALIZACION_USD

_DIA_HORAS = 14.0   # un día de mercado = 14 h (igual que el entorno)
_VACIO = N_NIVELES  # índice de la opción VACÍO (= routing_env.OPCION_VACIO)
_C = COTIZACIONES_MAX_POR_NODO
_INF = float("inf")


# ═════════════════════════════════════════════════════════════════════════════
# Observación del lote y piezas comunes
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class ObsLote:
    """Lo único que una política puede leer, para B filas (tensores en el dispositivo del entorno).

    nodo, casa (B,)      : nodo actual y nodo de inicio (donde se cierra el ciclo).
    tiempo (B,) f64      : horas transcurridas desde la salida.
    k_rest (B,), k_restr : arcos restantes incluido el retorno, y si la fila tiene presupuesto (False = caso libre).
    mascara (B, N) bool  : destinos válidos del entorno.
    precios (B, N, K) f64: USD/milla de cada nivel de precio.
    m (B,) f64           : competidores efectivos del nodo.
    opciones (B, N, K+1) : opciones válidas por destino (niveles y VACÍO).
    dow0 (B,)            : día de la semana (lunes = 0) del día de salida.
    indice (B, N) f64    : índice de mercado de hoy (solo B2P).
    n_rech (B,)          : cotizaciones ya perdidas en esta visita al nodo.
    intentos (B, N)      : cargas de cada lane que se pueden cotizar todavía en esta visita.
    """
    nodo: torch.Tensor
    casa: torch.Tensor
    tiempo: torch.Tensor
    k_rest: torch.Tensor
    k_restr: torch.Tensor
    mascara: torch.Tensor
    precios: torch.Tensor
    m: torch.Tensor
    opciones: torch.Tensor
    dow0: torch.Tensor
    indice: torch.Tensor
    n_rech: torch.Tensor
    intentos: torch.Tensor

    @staticmethod
    def del_entorno(env, masks) -> "ObsLote":
        return ObsLote(
            nodo=env.current_node, casa=env.start_node, tiempo=env.time_elapsed.double(),
            k_rest=env.k_remaining, k_restr=env.k_restricted, mascara=masks.bool(), precios=env.precios_row,
            m=env.competencia, opciones=env.opciones_validas, dow0=env.start_dow, indice=env.market_index.double(),
            n_rech=env.n_rechazos, intentos=env.intentos.clamp_min(0),
        )

    @property
    def B(self) -> int:
        return int(self.nodo.shape[0])

    @property
    def filas(self) -> torch.Tensor:
        return torch.arange(self.B, device=self.nodo.device)


class _Estatico:
    """Matrices fijas en el dispositivo. Dos costos, como en el código numpy de siempre: `costo` (la fila de la matriz,
    con el diésel en su float32: valor inmediato, B3 y el entorno) y `costo64` (el diésel pasado a float64 antes de
    dividir: la regla de precio, que lo calculaba lane por lane con escalares)."""

    def __init__(self, time_matrix, distance_arr, diesel_arr, device, dt=UBDP_DT_HORAS):
        dev = torch.device(device)
        self.device = dev
        self.T = torch.as_tensor(np.asarray(time_matrix, dtype=float), device=dev)
        self.dist = torch.as_tensor(np.asarray(distance_arr), device=dev)
        self.diesel = torch.as_tensor(np.asarray(diesel_arr), device=dev)
        self.costo = lane_cost_t(self.dist, self.diesel).double()
        self.costo64 = lane_cost_t(self.dist.double(), self.diesel.double())
        self.U = torch.as_tensor(U_NIVELES_ARR, dtype=torch.float64, device=dev)
        self.dt = float(dt)
        self.smax = int(np.floor(MAX_DURATION / self.dt + 1e-9))
        self.por_dia = int(round(_DIA_HORAS / self.dt))                       # franjas de un día de mercado
        f = np.floor(np.asarray(time_matrix, dtype=float) / self.dt + 1e-9).astype(np.int64)
        if f[~np.eye(f.shape[0], dtype=bool)].min() < 1:
            raise ValueError("Hay tiempos de viaje menores a una franja: la recursión del caso libre "
                             "no estaría bien fundada. Reduce dt.")
        self.f = torch.as_tensor(f, device=dev)                               # franjas de viaje, redondeo hacia abajo
        self.N = f.shape[0]
        self.jj = torch.arange(self.N, device=dev)

    def franja(self, t: torch.Tensor) -> torch.Tensor:
        return torch.floor(t / self.dt + 1e-9).long().clamp(max=self.smax)

    def dia(self, s: int) -> int:
        return int(s * self.dt / _DIA_HORAS + 1e-9)


class _ConEstatico:
    """Mezcla: crea las matrices fijas en el dispositivo de la primera observación."""

    def _est(self, device) -> _Estatico:
        e = getattr(self, "_estatico", None)
        if e is None or e.device != torch.device(device):
            e = _Estatico(self._T_np, self._dist_np, self._diesel_np, device)
            self._estatico = e
        return e


def _P(obs: ObsLote, est: _Estatico) -> torch.Tensor:
    """(B, K) P(ganar) de cada nivel con el m del nodo actual."""
    return prob_ganar_t(est.U[None, :], obs.m.double()[:, None])


def _argmax_primero(score: torch.Tensor, mascara: torch.Tensor) -> torch.Tensor:
    """(B,) el destino válido de mayor score; empates por el índice menor (como np.argmax sobre los válidos)."""
    s = torch.where(mascara, score, torch.full_like(score, -_INF))
    idx = s.argmax(dim=1)
    sin = torch.isinf(s.amax(dim=1))
    return torch.where(sin, mascara.float().argmax(dim=1), idx)


def valor_inmediato(obs: ObsLote, est: _Estatico, cierre: bool = False) -> torch.Tensor:
    """(B, N) valor esperado de salir por cada lane: max_k P_k·(p_k·d − c) sobre los niveles válidos (con los intentos
    que le quedan a la lane), y −c si la lane se puede recorrer VACÍA; con cierre=True (B2 / B2P, que suman el valor
    de toda la ruta) la vuelta a casa vale max_k P_k·p_k·d − c: si se pierde la carga, se vuelve vacío. 0 en la
    espera (j = i: la diagonal de distancias trae una distancia intra-hub, pero esperar no recorre nada); −inf donde
    no hay ninguna opción válida."""
    b, i, h = obs.filas, obs.nodo, obs.casa
    d, c = est.dist[i].double().clone(), est.costo[i].clone()
    d[b, i], c[b, i] = 0.0, 0.0
    base = torch.zeros_like(c)
    if cierre:
        base[b, h] = torch.where(i != h, -c[b, h], torch.zeros_like(c[b, h]))
    P = _P(obs, est)[:, None, :]                                                         # (B, 1, K)
    V = valor_reintentos_t(P, obs.precios * d[..., None] - c[..., None], base, _C)     # (C+1, B, N)
    v = torch.gather(V, 0, obs.intentos.long().clamp(max=_C)[None]).squeeze(0)
    v = torch.where(obs.opciones[..., :N_NIVELES].any(dim=-1), v, torch.full_like(v, -_INF))
    return torch.where(obs.opciones[..., _VACIO], torch.maximum(v, -c), v)


def cotizar_regla(obs: ObsLote, j: torch.Tensor, est: _Estatico, regla: str = REGLA_PRECIO_BASELINES) -> torch.Tensor:
    """(B,) opción para el destino j de cada fila: el nivel de la regla fija si la lane se puede cotizar; si no, VACÍO.

    La regla miope planea los intentos que le quedan en la lane (pricing.valor_reintentos): empieza más arriba si
    después puede cotizar otra carga, y baja a medida que se agotan, con F = lo que vale perderlas todas: la falla
    del ciclo si el camión está en casa y no puede ni esperar ni reposicionarse, volver vacío (−c) si j es casa, 0
    en los demás tramos. Cuando F < 0 toda regla fija cotiza así, en vez de volver vacío o fallar el ciclo con su
    nivel de siempre; en los demás tramos cada una usa su nivel (mediana: p50; be_margen: el nivel más bajo con
    precio ≥ break-even + BE_MARGEN, o el último)."""
    b, i, h = obs.filas, obs.nodo, obs.casa
    j = j.long()
    niveles_ok = obs.opciones[..., :N_NIVELES].any(dim=-1)                               # (B, N)
    cot_j = niveles_ok[b, j]
    d, c = est.dist[i, j].double(), est.costo64[i, j]
    sin_respaldo = (i == h) & ~obs.opciones[..., _VACIO].any(dim=-1)
    F = torch.where(sin_respaldo, torch.full_like(c, PENALIZACION_USD),
                    torch.where((j == h) & (i != h), -c, torch.zeros_like(c)))
    otras = niveles_ok.clone()
    otras[b, j] = False
    n_hasta_perder = torch.where((i == h) & otras.any(dim=-1), (_C - obs.n_rech).clamp_min(1),
                                 obs.intentos[b, j].long().clamp_min(1)).clamp(max=_C)
    P = _P(obs, est)
    margen = obs.precios[b, j] * d[:, None] - c[:, None]                                 # (B, K)
    V, ks = F, []
    for _ in range(_C):
        q = P * margen + (1.0 - P) * V[:, None]
        ks.append(q.argmax(dim=-1))
        V = q.amax(dim=-1)
    nivel = torch.stack(ks)[n_hasta_perder - 1, b]                                      # miope con los intentos
    if regla == "mediana":
        nivel = torch.where(F < 0, nivel, torch.full_like(nivel, indice_nivel(0.5)))
    elif regla == "be_margen":
        ok = obs.precios[b, j] >= (c / d.clamp_min(1e-12) + BE_MARGEN)[:, None]
        fijo = torch.where(ok.any(dim=-1), ok.long().argmax(dim=-1), torch.full_like(nivel, N_NIVELES - 1))
        nivel = torch.where(F < 0, nivel, fijo)
    elif regla != "miope":
        raise ValueError(f"regla de precio desconocida: {regla!r}")
    return torch.where(cot_j, nivel, torch.full_like(nivel, _VACIO))


# ═════════════════════════════════════════════════════════════════════════════
# Valores esperados por lane (tablas de B2 / B2P, numpy, una vez por nº de nodos)
# ═════════════════════════════════════════════════════════════════════════════

def _valor_cotizado_por_dow(market, distance_arr, diesel_arr, a_casa: bool = False) -> tuple:
    """(ev, pendiente), cada uno (7, N, N): valor esperado de un arco futuro cotizando con la regla miope sobre
    los cuantiles medios del día de la semana (especificacion_fase0_ruteo_pricing.md §7, B2 / B2P).

    Con los días de ENTRENAMIENTO reales de ese día de la semana:
        f_a[dow, j, l]   = fracción de días en que la lane tuvo a = min(cargas, COTIZACIONES_MAX_POR_NODO) cotizables
        p̄_k[dow, j, l]  = precio del nivel k con los cuantiles medios de la lane (media de los días)
        P̄_k[dow, j]     = media de los días de (1 − u_k)^m_j
        V_a              = valor de la lane con a intentos (pricing.valor_reintentos; perderlos todos vale 0)
        ev               = Σ_{a ≥ 1} f_a · V_a − f_0 · c            (sin carga el arco se recorre vacío)
    a_casa=True: el arco cierra el ciclo, así que perder todas las cargas es volver vacío (base −c).
    pendiente = d · Σ_a f_a · P(ganar alguna de las a) es la derivada de ev respecto de un corrimiento del precio
    (la usa B2P para el pronóstico del índice).
    """
    N = market.num_nodes
    dist = np.asarray(distance_arr, dtype=float)
    costo = lane_cost_row(dist, np.asarray(diesel_arr, dtype=float))
    dow_train = market.dia_semana(np.arange(TRAIN_DAYS))
    ev = np.zeros((7, N, N))
    pendiente = np.zeros((7, N, N))
    for k in range(7):
        dias = np.flatnonzero(dow_train == k)
        n = np.minimum(market.count_loads[dias], COTIZACIONES_MAX_POR_NODO)              # (días, N, N)
        q_medio = np.mean([market.cuantiles_dia(int(d)) for d in dias], axis=0)         # (N, N, 6)
        p = precios_niveles(q_medio)                                                     # (N, N, K)
        P = np.mean([prob_ganar(U_NIVELES_ARR[None, :], market.competidores(np.arange(N), np.full(N, int(d)))[:, None])
                     for d in dias], axis=0)                                             # (N, K)
        base = -costo if a_casa else np.zeros((N, N))
        V, ks = valor_reintentos(P[:, None, :], p * dist[..., None] - costo[..., None], base, COTIZACIONES_MAX_POR_NODO)
        ev[k] = -(n <= 0).mean(axis=0) * costo
        gana_alguna = np.zeros((N, N))                                                   # P(ganar alguna de a cargas)
        for a in range(1, COTIZACIONES_MAX_POR_NODO + 1):
            p_a = np.take_along_axis(P, ks[a - 1], axis=1)                               # (N, N) P̄ del nivel del intento
            gana_alguna = p_a + (1.0 - p_a) * gana_alguna
            f_a = (n == a).mean(axis=0)
            ev[k] += f_a * V[a]
            pendiente[k] += f_a * gana_alguna * dist
    return ev, pendiente


def calcular_ev_esperado(market, distance_arr, diesel_arr) -> np.ndarray:
    """Valor esperado por lane y día de la semana, (7, N, N), con la cotización miope (ver
    `_valor_cotizado_por_dow`). Es el valor que usa B2 para los arcos futuros; se calcula una vez por número de nodos."""
    return _valor_cotizado_por_dow(market, distance_arr, diesel_arr)[0]


@dataclass(frozen=True)
class EvEsperado:
    """Valor esperado por lane con pronóstico del índice de mercado (tabla de B2P).

    Para un arco j→l que sale de j un día de mercado `dow` de la semana, h días de mercado después de hoy:

        ev_ajustado(h, dow)[j, l] = ev[dow, j, l] + índice(j, hoy) · q[h, dow, j, l]

    ev (7, N, N)      : calcular_ev_esperado (cotización miope sobre los cuantiles medios; índice cero)
    q  (H+1, 7, N, N) : b_h · P(N_jl ≥ 1) · P̄_k* · dist_jl, con b_h = f + (1 − f) · ρ^h
    ev_casa, q_casa   : lo mismo para el arco que cierra el ciclo (perder la carga = volver vacío)
    b  (H+1,)         : los coeficientes b_h (f = fracción de varianza lenta del índice, ρ = persistencia del rápido)
    """
    ev: np.ndarray
    q:  np.ndarray
    b:  np.ndarray
    ev_casa: np.ndarray
    q_casa:  np.ndarray

    @property
    def h_max(self) -> int:
        return self.q.shape[0] - 1


def calcular_ev_pronostico(market, distance_arr, diesel_arr) -> EvEsperado:
    """Tabla de valores esperados por lane con pronóstico del índice (ver EvEsperado); una vez por nº de nodos."""
    ev, pendiente = _valor_cotizado_por_dow(market, distance_arr, diesel_arr)   # (7,N,N) cada uno
    b = market.frac_var_lento + (1.0 - market.frac_var_lento) * market.rho ** np.arange(MARKET_DIAS_EPISODIO)  # h = 0 … H
    q = b[:, None, None, None] * pendiente[None]                          # (H+1, 7, N, N)
    ev_c, pend_c = _valor_cotizado_por_dow(market, distance_arr, diesel_arr, a_casa=True)
    return EvEsperado(ev=ev, q=q, b=b, ev_casa=ev_c, q_casa=b[:, None, None, None] * pend_c[None])


# ═════════════════════════════════════════════════════════════════════════════
# B1
# ═════════════════════════════════════════════════════════════════════════════

class PoliticaB1(_ConEstatico):
    """B1, greedy miope (PFA): jugada válida de mayor valor inmediato esperado (`valor_inmediato`). La espera
    vale 0, así que B1 espera mientras ninguna lane tenga valor esperado positivo y todavía pueda esperar.
    Precio con la regla fija. Empates por el índice menor."""
    nombre = "B1"
    regla = REGLA_PRECIO_BASELINES

    def __init__(self, time_matrix, distance_arr, diesel_arr):
        self._T_np, self._dist_np, self._diesel_np = time_matrix, distance_arr, diesel_arr

    def decidir(self, obs: ObsLote, activo=None) -> tuple:
        est = self._est(obs.nodo.device)
        a = _argmax_primero(valor_inmediato(obs, est), obs.mascara)
        return a, cotizar_regla(obs, a, est, self.regla)


# ═════════════════════════════════════════════════════════════════════════════
# B2
# ═════════════════════════════════════════════════════════════════════════════

_LIBRE = -1   # clave del caso libre en las tablas por nivel de k


class PoliticaB2(_ConEstatico):
    """B2, reoptimización determinista con valores esperados (CFA); con K = 5 y con K libre.

    En cada nodo elige la jugada que maximiza el valor inmediato real de la fila observada más la mejor continuación
    con los valores esperados de los arcos futuros (ev[dow, j, l], `calcular_ev_esperado`; el que cierra el ciclo,
    ev_casa), y ejecuta solo el primer arco. El camión puede volver a pasar por un nodo, así que el futuro es
    V[casa, día de salida, j, franja, k]: en cada nodo futuro se puede cerrar el ciclo, esperar al siguiente día de
    mercado (mismo k) o seguir a otro nodo l con tiempo de volver a casa. Las franjas son de UBDP_DT_HORAS y los
    tiempos de viaje se redondean hacia abajo, como en baselines/dp_relajada.py. Con k libre no hay dimensión k y la
    profundidad solo la acota MAX_DURATION. Esperar en casa antes de salir se valora con una sola salida al día
    siguiente (tabla S, sin volver a esperar).

    Como ev no depende del mundo, V se calcula UNA vez, hacia atrás en las franjas, en el dispositivo (para todas las
    casas y días de salida), y cada decisión es una lectura de la tabla.
    """
    nombre = "B2"
    regla = REGLA_PRECIO_BASELINES

    def __init__(self, market, time_matrix, distance_arr, diesel_arr, max_duration=MAX_DURATION, dt=UBDP_DT_HORAS):
        if max_duration != MAX_DURATION or dt != UBDP_DT_HORAS:
            raise ValueError("B2 usa MAX_DURATION y UBDP_DT_HORAS")
        self._T_np, self._dist_np, self._diesel_np = time_matrix, distance_arr, diesel_arr
        self._ev_np = calcular_ev_esperado(market, distance_arr, diesel_arr)                           # (7, N, N)
        self._ev_casa_np = _valor_cotizado_por_dow(market, distance_arr, diesel_arr, a_casa=True)[0]   # arco de cierre
        self._tablas = {}                                                       # nivel → (V (H,7,N,S+1), S (H,7,S+1))

    # ── Tablas ───────────────────────────────────────────────────────────────

    def _tabla(self, nivel: int, est: _Estatico) -> tuple:
        """(V, S) del nivel de k (_LIBRE o 0 … K; el 0 es igual al 1: solo cerrar o esperar). V[h, d0, j, s] es el
        mejor valor de seguir desde j (llegada en la franja s) y S[h, d0, s] el de salir de casa en la franja s."""
        if nivel == 0:
            nivel = 1
        if nivel in self._tablas:
            return self._tablas[nivel]
        V_ant = None if nivel in (_LIBRE, 1) else self._tabla(nivel - 1, est)[0]
        dev, N, smax, F = est.device, est.N, est.smax, est.f
        ev = torch.as_tensor(self._ev_np, device=dev)
        ev_casa = torch.as_tensor(self._ev_casa_np, device=dev)
        V = torch.zeros((N, 7, N, smax + 1), dtype=torch.float64, device=dev)
        jj = est.jj
        mueve = nivel == _LIBRE or nivel >= 2
        # factible[h, j, l]: l ∉ {j, h} y desde l todavía se puede volver (l → h) al llegar en s + f[j, l]
        ida_vuelta = F[None, :, :] + F.T[:, None, :]                            # (H, j, l): f[j, l] + f[l, h]
        distinto = (jj[None, None, :] != jj[None, :, None]) & (jj[None, None, :] != jj[:, None, None])
        for s in range(smax, -1, -1):
            dows = (torch.arange(7, device=dev) + est.dia(s)) % 7
            mejor = ev_casa[dows].permute(2, 0, 1).clone()                      # (H, d0, j): cerrar el ciclo desde j
            s_w = (s // est.por_dia + 1) * est.por_dia                          # esperar al día siguiente en j
            if s_w <= smax:
                ok = (s_w + F[:, :] <= smax).T                                  # (H, j): s_w + f[j, h] ≤ smax
                mejor = torch.where(ok[:, None, :], torch.maximum(mejor, V[:, :, :, s_w]), mejor)
            if mueve:
                V_sig = V if nivel == _LIBRE else V_ant
                s_n = (s + F).clamp(max=smax)                                   # (j, l)
                cont = V_sig[:, :, jj[None, :].expand(N, N), s_n]               # (H, d0, j, l) = V_sig[h, d0, l, s + f_jl]
                cand = ev[dows][None] + cont                                    # ev[dow, j, l] + V(l)
                fact = distinto & (s + ida_vuelta <= smax)                      # (H, j, l)
                cand = torch.where(fact[:, None], cand, torch.full_like(cand, -_INF))
                mejor = torch.maximum(mejor, cand.amax(dim=-1))
            V[:, :, :, s] = mejor
        # Salida de casa en la franja s: la mejor primera lane h → j con V del nivel siguiente.
        V_sal = V if nivel == _LIBRE else (V_ant if V_ant is not None else V)
        if nivel == 1:
            V_sal = V                                                           # k = 1 en casa: k_sig = 0 = nivel 1
        S = torch.full((N, 7, smax + 1), -_INF, dtype=torch.float64, device=dev)
        hh = jj
        f_hj = F                                                                # (h, j)
        for s in range(smax + 1):
            dows = (torch.arange(7, device=dev) + est.dia(s)) % 7
            ok = (s + f_hj + F.T <= smax) & (jj[None, :] != hh[:, None])         # (h, j)
            s_j = (s + f_hj).clamp(max=smax)
            cont = V_sal[hh[:, None], :, jj[None, :], s_j]                      # (h, j, d0)
            val = ev[dows][:, hh, :].permute(1, 2, 0) + cont                    # ev[dow, h, j] + V(j)
            val = torch.where(ok[:, :, None], val, torch.full_like(val, -_INF))
            S[:, :, s] = val.amax(dim=1)
        self._tablas[nivel] = (V, S)
        return V, S

    def _leer(self, obs: ObsLote, est: _Estatico, niveles: torch.Tensor, h, d0, j, s, tabla: int) -> torch.Tensor:
        """V (tabla=0) o S (tabla=1) de cada fila, del nivel de k de esa fila (_LIBRE o k)."""
        out = torch.zeros(niveles.shape[0], dtype=torch.float64, device=est.device)
        for nv in torch.unique(niveles).tolist():
            sel = niveles == nv
            T = self._tabla(int(nv), est)[tabla]
            out[sel] = T[h[sel], d0[sel], j[sel], s[sel]] if tabla == 0 else T[h[sel], d0[sel], s[sel]]
        return out

    # ── Decisión ─────────────────────────────────────────────────────────────

    def decidir(self, obs: ObsLote, activo=None) -> tuple:
        est = self._est(obs.nodo.device)
        b, i, h, d0, t = obs.filas, obs.nodo, obs.casa, obs.dow0, obs.tiempo
        r = valor_inmediato(obs, est, cierre=True)                              # (B, N)
        libre = torch.full_like(obs.k_rest, _LIBRE)
        nivel = torch.where(obs.k_restr, obs.k_rest, libre)
        nivel_sig = torch.where(obs.k_restr, (obs.k_rest - 1).clamp_min(0), libre)
        N = est.N
        # Esperar en i al día siguiente (mismo k): en casa, la mejor salida al día siguiente (S); afuera, V(i).
        s_w = est.franja(fin_espera(t))
        v_esp = torch.where(i == h, self._leer(obs, est, nivel, h, d0, i, s_w, 1),
                            self._leer(obs, est, nivel, h, d0, i, s_w, 0))
        # Ir a j ∉ {i, h}: r_j + V(j, franja de llegada, k − 1). Cerrar en casa: r_h.
        s_a = torch.floor((t[:, None] + est.T[i]) / est.dt + 1e-9).long().clamp(max=est.smax)   # (B, N)
        cont = self._leer(obs, est, nivel_sig.repeat_interleave(N), h.repeat_interleave(N), d0.repeat_interleave(N),
                          est.jj.repeat(obs.B), s_a.reshape(-1), 0).view(obs.B, N)
        v = torch.where(est.jj[None, :] == h[:, None], r, r + cont)
        v[b, i] = v_esp
        a = _argmax_primero(v, obs.mascara)
        return a, cotizar_regla(obs, a, est, self.regla)


# ═════════════════════════════════════════════════════════════════════════════
# B2P
# ═════════════════════════════════════════════════════════════════════════════

class PoliticaB2P(_ConEstatico):
    """B2P, B2 con pronósticos del índice de mercado.

    Igual que PoliticaB2 salvo en los valores esperados de los arcos futuros. Para un arco futuro j→l al
    que se llega h días de mercado después de hoy, la tarifa esperada es

        media_lane(jl) + b_h · índice(j, hoy),     b_h = f + (1 − f) · ρ^h

    con f la fracción de varianza lenta del índice y ρ la persistencia del rápido (ambas del ajuste): la
    mejor predicción lineal del índice de j h días adelante, dado el de hoy (el lento no decae y el rápido
    decae como ρ^h). El índice de hoy es el que observa el camión (`ObsLote.indice`). El arco actual
    usa la fila observada, como en B2. Es lineal en el índice, así que se precalculan las tablas por h y día de la
    semana (`calcular_ev_pronostico`): ev(h, dow) = E0[dow] + índice[:, None] · Q[h, dow]. Con índice cero coincide
    con B2.

    Como los valores esperados dependen del índice de hoy, la mejor continuación se recalcula en cada decisión con
    una recursión hacia atrás sobre las franjas, para todas las filas del lote a la vez (en bloques de `bloque` filas).
    """
    nombre = "B2P"
    regla = REGLA_PRECIO_BASELINES

    def __init__(self, market, time_matrix, distance_arr, diesel_arr, max_duration=MAX_DURATION, dt=UBDP_DT_HORAS,
                 bloque: int = 256):
        if max_duration != MAX_DURATION or dt != UBDP_DT_HORAS:
            raise ValueError("B2P usa MAX_DURATION y UBDP_DT_HORAS")
        self._T_np, self._dist_np, self._diesel_np = time_matrix, distance_arr, diesel_arr
        tabla = calcular_ev_pronostico(market, distance_arr, diesel_arr)
        self._h_max = tabla.h_max
        self.b = tabla.b
        self._np = dict(E0=tabla.ev, Q=tabla.q, E0c=tabla.ev_casa, Qc=tabla.q_casa)
        self._t = None
        self.bloque = int(bloque)

    def _tens(self, dev):
        if self._t is None or self._t["E0"].device != dev:
            self._t = {k: torch.as_tensor(v, device=dev) for k, v in self._np.items()}
        return self._t

    def _valores(self, est, h, d0, dia_hoy, indice, s_lo, k_max) -> dict:
        """{nivel: V (b, N, S+1)} para las filas del bloque: V[b, j, s] es el mejor valor esperado de seguir desde j
        (llegada en la franja s), con los pronósticos del índice de hoy de cada fila. k_max None ⇒ caso libre
        ({_LIBRE: V}); si no, los niveles 1 … k_max (el 0 es igual al 1)."""
        t = self._tens(est.device)
        n, N, smax, F, jj = h.shape[0], est.N, est.smax, est.f, est.jj
        ev_dia, cierre_dia = {}, {}

        def ev(s):
            d = est.dia(s)
            if d not in ev_dia:
                dow = (d0 + d) % 7
                hh = (d - dia_hoy).clamp(0, self._h_max)
                ev_dia[d] = t["E0"][dow] + indice[:, :, None] * t["Q"][hh, dow]                       # (b, N, N)
                cierre_dia[d] = t["E0c"][dow, :, h] + indice * t["Qc"][hh, dow, :, h]                # (b, N)
            return ev_dia[d], cierre_dia[d]

        base = F[None, :, :] + F.T[h][:, None, :]                              # (b, j, l): f[j, l] + f[l, h]
        ok = (jj[None, None, :] != jj[None, :, None]) & (jj[None, None, :] != h[:, None, None])
        vuelta = F[:, h].T                                                     # (b, N) f[j, h]
        s_sig = (torch.arange(smax + 1, device=est.device)[:, None, None] + F[None]).clamp(max=smax)   # (s, j, l)
        l_idx = jj[None, :].expand(N, N)

        def esperar(s, V, mejor):
            s_w = (s // est.por_dia + 1) * est.por_dia
            if s_w > smax:
                return mejor
            return torch.where(s_w + vuelta <= smax, torch.maximum(mejor, V[:, :, s_w]), mejor)

        def paso(s, V_sig):
            e, c = ev(s)
            cand = e + V_sig[:, l_idx, s_sig[s]]                                # (b, j, l)
            cand = torch.where(ok & (s + base <= smax), cand, torch.full_like(cand, -_INF))
            return torch.maximum(c, cand.amax(dim=-1))

        if k_max is None:
            V = torch.zeros((n, N, smax + 1), dtype=torch.float64, device=est.device)
            for s in range(smax, s_lo - 1, -1):
                V[:, :, s] = esperar(s, V, paso(s, V))
            return {_LIBRE: V}
        out = {}
        V = torch.zeros((n, N, smax + 1), dtype=torch.float64, device=est.device)
        for s in range(smax, s_lo - 1, -1):                                    # k = 1: casa o esperar
            V[:, :, s] = esperar(s, V, ev(s)[1])
        out[1] = V
        for k in range(2, int(k_max) + 1):                                     # k ≥ 2: un nivel por arco
            V_ant, V = V, torch.zeros((n, N, smax + 1), dtype=torch.float64, device=est.device)
            for s in range(smax, s_lo - 1, -1):
                V[:, :, s] = esperar(s, V, paso(s, V_ant))
            out[k] = V
        out[0] = out[1]
        return out

    def decidir(self, obs: ObsLote, activo=None) -> tuple:
        est = self._est(obs.nodo.device)
        r = valor_inmediato(obs, est, cierre=True)
        v = torch.full_like(r, -_INF)
        activas = (obs.mascara.any(dim=1) if activo is None else activo).nonzero().squeeze(1)
        for ini in range(0, activas.numel(), self.bloque):
            sel = activas[ini:ini + self.bloque]
            for libre in (True, False):
                filas = sel[~obs.k_restr[sel]] if libre else sel[obs.k_restr[sel]]
                if filas.numel():
                    v[filas] = self._valor_jugadas(obs, est, r, filas, libre)
        a = _argmax_primero(v, obs.mascara)
        return a, cotizar_regla(obs, a, est, self.regla)

    def _valor_jugadas(self, obs, est, r, filas, libre) -> torch.Tensor:
        """(n, N) valor de cada jugada de las filas `filas` (todas libres o todas con presupuesto)."""
        n, smax, F, jj = filas.numel(), est.smax, est.f, est.jj
        bb = torch.arange(n, device=est.device)
        i, h, d0, t = obs.nodo[filas], obs.casa[filas], obs.dow0[filas], obs.tiempo[filas]
        indice = obs.indice[filas]
        k = obs.k_rest[filas]
        s_a = torch.floor((t[:, None] + est.T[i]) / est.dt + 1e-9).long().clamp(max=smax)          # (n, N)
        s_w = est.franja(fin_espera(t))
        s_a[bb, i] = s_w
        dia_hoy = torch.floor(torch.floor(t / est.dt + 1e-9) * est.dt / _DIA_HORAS + 1e-9).long()
        s_lo = int(torch.floor(t / est.dt + 1e-9).long().clamp(max=smax).min())
        tabla = self._valores(est, h, d0, dia_hoy, indice, s_lo, None if libre else int(k.max()))
        if libre:
            V_sig = V_k = tabla[_LIBRE]
        else:
            V_sig = torch.stack([tabla[int(x)][q] for q, x in enumerate((k - 1).clamp_min(0).tolist())])
            V_k = torch.stack([tabla[int(x)][q] for q, x in enumerate(k.tolist())])
        cont = torch.gather(V_sig, 2, s_a[:, :, None]).squeeze(2)                                    # V_sig[b, a, s_a]
        v = torch.where(jj[None, :] == h[:, None], r[filas], r[filas] + cont)
        # Esperar: en casa, la mejor salida al día siguiente; afuera, V del mismo k en i.
        t_ = self._tens(est.device)
        d_w = torch.floor(s_w * est.dt / _DIA_HORAS + 1e-9).long()
        dow_w = (d0 + d_w) % 7
        hh_w = (d_w - dia_hoy).clamp(0, self._h_max)
        ev_w = t_["E0"][dow_w, h] + indice[bb, h][:, None] * t_["Q"][hh_w, dow_w, h]                 # (n, N) fila h
        ok = (s_w[:, None] + F[h] + F[:, h].T <= smax) & (jj[None, :] != h[:, None])
        s_j = (s_w[:, None] + F[h]).clamp(max=smax)
        salida = torch.where(ok, ev_w + torch.gather(V_sig, 2, s_j[:, :, None]).squeeze(2),
                             torch.full_like(ev_w, -_INF)).amax(dim=1)
        afuera = V_k[bb, i, s_w]
        v[bb, i] = torch.where(i == h, salida, afuera)
        return v


# ═════════════════════════════════════════════════════════════════════════════
# B3
# ═════════════════════════════════════════════════════════════════════════════

class PoliticaB3(_ConEstatico):
    """B3: un paso sobre la tabla V de UB-DP con los intentos de cotización (especificacion_fase0_ruteo_pricing.md §7.3).

    Con el tablero observado del nodo i (precios por nivel, m, cargas) y G_jk(t) = r_jk + V(j, t + τ_ij, k − 1)
    (V(casa) = 0), P_k = (1 − u_k)^m y t_r = t + r · COTIZACION_HORAS:

        B_r = max( E(t_r), max_j max_k [ P_k G_jk(t_r) + (1 − P_k) B_{r+1} ] )    (lanes con más de r cargas)
        E(t) = max_j e_j(t): la mejor jugada sin carga
        e_j(t) = R0_ij + V(j, t + τ_ij, k − 1)  (vacío a j: casa o reposicionamiento);
                 V(i, fin_espera(t), k) para j = i (esperar; en casa antes de salir, H(casa, fin_espera(t), k))

    r cuenta los rechazos futuros, hasta los que quedan en la visita. Elige la jugada (j, k) que alcanza B_0: destino
    y precio salen del mismo máximo (VACÍO si gana e_j). Empates por el índice menor.
    """
    nombre = "B3"

    def __init__(self, tabla: TablaDP, time_matrix, distance_arr, diesel_arr):
        self.tabla = tabla
        self._T_np, self._dist_np, self._diesel_np = time_matrix, distance_arr, diesel_arr
        self._t = None

    def _tablas(self, dev):
        """(V (K+2, 7, H, N, Ns), H (K+2, 7, H, Ns)) con el nivel K+1 = caso libre (float32, como la tabla)."""
        if self._t is None or self._t[0].device != dev:
            tb = self.tabla
            V = torch.cat([torch.as_tensor(tb.V_k), torch.as_tensor(tb.V_libre)[None]]).to(dev)
            H = torch.cat([torch.as_tensor(tb.H_k), torch.as_tensor(tb.H_libre)[None]]).to(dev)
            self._t = (V, H)
        return self._t

    def _s(self, tiempos):
        return torch.floor(tiempos / self.tabla.dt + 1e-9).long().clamp(max=self.tabla.Ns - 1)

    def _resolver(self, obs: ObsLote, est: _Estatico) -> tuple:
        """(v_cot, e1, Q) de la decisión AHORA (r = 0) para cada destino: el valor de la mejor cotización v_cot (B, N)
        (la opción de precio que la alcanza es argmax Q (B, N, K)) y el de la mejor jugada sin carga e1 (B, N)."""
        Vt, Ht = self._tablas(est.device)
        libre = self.tabla.K + 1
        b, i, h, d0, t = obs.filas, obs.nodo, obs.casa, obs.dow0, obs.tiempo
        nivel = torch.where(obs.k_restr, obs.k_rest, torch.full_like(obs.k_rest, libre))
        nivel_sig = torch.where(obs.k_restr, obs.k_rest - 1, torch.full_like(obs.k_rest, libre))
        r_k = lane_reward_t(obs.precios, torch.ones_like(obs.precios), est.dist[i][..., None], est.diesel[i][..., None])
        r0 = lane_reward_t(torch.zeros_like(obs.precios[..., 0]), torch.zeros_like(obs.precios[..., 0]),
                           est.dist[i], est.diesel[i])
        P = _P(obs, est)[:, None, :]                                           # (B, 1, K)
        cotiza = obs.opciones[..., :N_NIVELES].any(dim=-1)
        vacia = obs.opciones[..., _VACIO]
        es_i = est.jj[None, :] == i[:, None]
        es_h = est.jj[None, :] == h[:, None]
        T_i = est.T[i]                                                         # (B, N)

        def cont(t_r):
            """(B, N) V(j, t_r + τ_ij, k − 1); 0 para casa y con k − 1 = 0."""
            s = self._s(t_r[:, None] + T_i)
            c = Vt[nivel_sig.clamp_min(0)[:, None], d0[:, None], h[:, None], est.jj[None, :], s].double()
            return torch.where(es_h | (nivel_sig == 0)[:, None], torch.zeros_like(c), c)

        def espera(t_r):
            """(B,) valor de esperar en i hasta el día siguiente a partir de t_r (mismo k)."""
            s = self._s(fin_espera(t_r))
            return torch.where(i == h, Ht[nivel, d0, h, s].double(), Vt[nivel, d0, h, i, s].double())

        quedan = _C - obs.n_rech
        B_ = torch.full((obs.B,), -_INF, dtype=torch.float64, device=est.device)
        for r in range(_C, -1, -1):
            t_r = t + r * COTIZACION_HORAS
            c_r = cont(t_r)
            e1 = torch.where(vacia, r0 + c_r, torch.full_like(c_r, -_INF))
            e1[b, i] = espera(t_r)
            # Como la máscara del entorno: lejos de casa se cotiza solo si, al perder, todavía se puede volver.
            a_tiempo = (i == h) | (t_r + COTIZACION_HORAS + est.T[i, h] <= MAX_DURATION)
            permitido = cotiza & a_tiempo[:, None] & (obs.intentos > r) & (r < quedan)[:, None] & ~es_i
            Q = torch.where(permitido[..., None], P * (r_k + c_r[..., None]) + (1.0 - P) * B_[:, None, None],
                            torch.full_like(r_k, -_INF))
            v_cot = Q.amax(dim=-1)
            Bn = torch.where(obs.mascara, torch.maximum(v_cot, e1), torch.full_like(e1, -_INF)).amax(dim=1)
            Bn = torch.where(torch.isfinite(Bn), Bn, torch.full_like(Bn, PENALIZACION_USD))   # sin respaldo: falla
            B_ = torch.where(r <= quedan, Bn, B_)
        return v_cot, e1, Q

    def decidir(self, obs: ObsLote, activo=None) -> tuple:
        est = self._est(obs.nodo.device)
        v_cot, e1, Q = self._resolver(obs, est)
        a = _argmax_primero(torch.maximum(v_cot, e1), obs.mascara)
        b = obs.filas
        o = torch.where(v_cot[b, a] >= e1[b, a], Q[b, a].argmax(dim=-1), torch.full_like(a, _VACIO))
        return a, o

    def precio_en(self, obs: ObsLote, j: torch.Tensor) -> torch.Tensor:
        """Opción de B3 para un destino FIJADO desde afuera (comparación "rutas del DRL + precio de la tabla"): con el
        mismo máximo de `decidir` pero restringido a j. El valor de lo que pasa después de un rechazo (B_1) sigue
        siendo libre: se puede cambiar de destino. El nivel de precio que maximiza el valor, o VACÍO si gana la
        jugada sin carga (la regla miope si la tabla no ve ninguna jugada en j)."""
        est = self._est(obs.nodo.device)
        v_cot, e1, Q = self._resolver(obs, est)
        b, j = obs.filas, j.long()
        vc, ve = v_cot[b, j], e1[b, j]
        o = torch.where(vc >= ve, Q[b, j].argmax(dim=-1), torch.full_like(j, _VACIO))
        sin_jugada = ~torch.isfinite(torch.maximum(vc, ve))
        return torch.where(sin_jugada, cotizar_regla(obs, j, est, "miope"), o)
