"""
routing_env.py
==============
VectorRoutingEnv: entorno del ruteo estocástico con cotización (especificacion_fase0_ruteo_pricing.md §4),
B episodios en paralelo. Todo el estado son tensores de torch en el dispositivo del entorno (la GPU en el
servidor): no hay numpy ni loops sobre B en un paso, y nada vuelve a la CPU.

MDP
---
En cada nodo la política elige (j, o): destino j y opción o ∈ {nivel 0 … nivel K−1, VACÍO}, con
VACÍO = OPCION_VACIO = K. step(actions, opciones). El objetivo es el margen total del ciclo: en cada nodo
(casa incluida, antes de salir) hay tres clases de jugada.
  · Cotizar (nivel k, percentiles p10 … p90): válido si j ≠ i, la lane i→j tiene alguna carga que
    no se haya perdido en esta visita y quedan cotizaciones (COTIZACIONES_MAX_POR_NODO). Se gana si u_k < u_comp
    (pricing.gana); ningún nivel gana con seguridad. Si se gana, el camión viaja a j y cobra:
    reward = round(p_k·d − c).
    Si se pierde: el camión se queda en i, el tiempo avanza COTIZACION_HORAS, reward 0, esa carga
    se pierde (la lane se puede volver a cotizar con OTRA carga, otro competidor independiente, hasta agotar las
    suyas), no se consume K y el resto del tablero no cambia.
  · VACÍO a j ≠ i (reward round(−c)): reposicionarse a un nodo a ≤ REPOSICION_MAX_MILLAS, o volver a casa
    desde un nodo intermedio a cualquier distancia. Vale aunque la lane tenga carga.
  · ESPERAR = (j = i, VACÍO): el camión se queda en i hasta el siguiente día de mercado (fin_espera:
    el próximo múltiplo de HORAS_DIA_MERCADO), reward 0, sin consumir K ni contar como arco. Al despertar
    se revela el tablero de i del día nuevo y se reinician los rechazos de la visita.
Todo movimiento (y la espera, y perder una cotización lejos de casa) exige poder volver a casa dentro de max_duration. Si un paso se pasa de
max_duration, termina con TIME_VIOLATION_PENALTY.
`opciones_validas` (B, N, K+1) es la máscara de opciones y la máscara de destinos es su `any(-1)`.

Tableros
--------
Los mundos vienen sorteados de antemano (`market_process.Mundos`: cargas y uniformes de los competidores por
episodio, día, nodo y lane; `MarketProcess.sortear_mundos` para mundos simulados y `mundos_replay` para los días
reales). Al llegar a i el día absoluto dia = min(dia0 + ⌊t / 14⌋, dia_max) se lee la fila [b, dia − dia0, i] y se
completa con las tablas por día de `MarketProcess.tensores` (precios por nivel · e^δ, cuantiles, m). Como el mundo
no depende de las decisiones, dos políticas sobre el mismo `Mundos` enfrentan exactamente lo mismo.
El día de la VISITA (el de su tablero) manda durante toda la visita: aunque las cotizaciones perdidas (15 min cada
una) lleven el reloj al día siguiente, el tablero no cambia hasta esperar o viajar, y todo lo que describe "hoy"
(`current_day`, `current_dow`, `market_index`, `ltr_actual`, `cargas_hoy`, `margen_hoy`) es de ese mismo día. El
reloj (`time_elapsed`) sí sigue corriendo para el tiempo restante y la factibilidad.
El agente observa los precios por nivel (`precios_row`), las cargas y m (`competencia`), nunca el
competidor (`u_comp_row` existe solo para la cota con información perfecta y los tests).
`reward_rows` y `rate_row_obs` son el margen y el precio en la mediana de cada lane: lo observable.

Terminación:
  A) el camión viaja de vuelta al start_node, o el tiempo pasa de max_duration;
  B) no queda ninguna acción factible. Fuera de casa no pasa (volver a casa siempre vale). En casa:
     · si ya en reset() no hay ninguna jugada (sin carga cotizable, sin reposicionamiento ni tiempo para
       esperar), el camión NO SALE: el ciclo termina con margen 0, no es violación (`sin_salida`) y la fila
       arranca inactiva (`active`). Es exógeno: le pasa igual a todos los métodos.
     · si la política tenía jugadas y se quedó sin ellas (perdió las cotizaciones o esperó hasta que no quedó
       nada), el ciclo FALLA: violación con TIME_VIOLATION_PENALTY, como pasarse del tiempo. El ciclo hay que
       hacerlo; cotizar alto a propósito no sirve para saltarse un ciclo malo.

Presupuesto de cardinalidad K (número de arcos, incluido el retorno)
----------------------------------------------------------------------
reset(k_max=[K o None, …]). None = caso libre. Con K, k_remaining baja en 1 por cada arco
recorrido (un rechazo o una espera no consumen K) y, con k_remaining == 1, solo quedan volver a casa o
esperar. Formulación ≤ K: la política puede cerrar el ciclo antes.
"""

import numpy as np
import torch

from config import (
    MAX_DURATION,
    REWARD_SCALE_FACTOR,
    RETURN_SUCCESS_BONUS,
    TIME_VIOLATION_PENALTY,
    USE_REWARD_SHAPING,
    COTIZACION_HORAS,
    COTIZACIONES_MAX_POR_NODO,
    HORAS_DIA_MERCADO,
    REPOSICION_MAX_MILLAS,
    CHECK_TENSORS,
)
from pricing import (
    N_NIVELES, U_NIVELES_ARR, indice_nivel, interp_cuantiles_t, lane_cost_t, lane_reward_t, percentil_competidor_t,
)

OPCION_VACIO = N_NIVELES                 # índice de la opción "ir sin cotizar" (tras los K niveles)
_NIVEL_MEDIANA = indice_nivel(0.5)


def fin_espera(t):
    """Hora en que termina una espera que empieza en t: el siguiente día de mercado (próximo múltiplo de
    HORAS_DIA_MERCADO, estrictamente mayor que t). Sirve con tensores (en float64) o con escalares/arrays."""
    if isinstance(t, torch.Tensor):
        return (torch.floor(t.double() / HORAS_DIA_MERCADO) + 1.0) * HORAS_DIA_MERCADO
    return (np.floor(np.asarray(t, dtype=np.float64) / HORAS_DIA_MERCADO) + 1.0) * HORAS_DIA_MERCADO


def max_esperas(max_duration: float = MAX_DURATION) -> int:
    """Cota del número de esperas de un episodio: cada una salta a un día de mercado nuevo."""
    return int(max_duration // HORAS_DIA_MERCADO) + 1


def vacio_permitido(distance_arr, max_millas: float = REPOSICION_MAX_MILLAS) -> np.ndarray:
    """(N, N) bool: lanes i→j (i ≠ j) que se pueden recorrer VACÍAS para reposicionarse (≤ max_millas).
    Volver a casa vacío desde un nodo intermedio vale a cualquier distancia y va aparte."""
    ok = np.asarray(distance_arr, dtype=float) <= max_millas
    np.fill_diagonal(ok, False)
    return ok


def max_pasos(time_matrix, max_duration: float = MAX_DURATION) -> int:
    """Cota del número de arcos de un episodio.

    Sin self-loop, cada arco consume al menos el menor tiempo de viaje fuera de la diagonal;
    el arco que excede `max_duration` termina el episodio, así que como mucho caben
    floor(max_duration / T_min) arcos más ese último.
    """
    T = np.asarray(time_matrix, dtype=float)
    t_min = T[~np.eye(T.shape[0], dtype=bool)].min()
    return int(max_duration // t_min) + 1


def max_pasos_cotizacion(time_matrix, max_duration: float = MAX_DURATION) -> int:
    """Cota de pasos con cotización: en cada visita (una llegada o una espera), hasta COTIZACIONES_MAX_POR_NODO
    rechazos más el paso que viaja o espera."""
    return (max_pasos(time_matrix, max_duration) + max_esperas(max_duration)) * (COTIZACIONES_MAX_POR_NODO + 1)


def k_tensor(k_max, B: int, device) -> tuple:
    """(k_restricted (B,) bool, k_max (B,) long) a partir de una lista de (int | None) o None (todos libres)."""
    if k_max is None:
        return torch.zeros(B, dtype=torch.bool, device=device), torch.zeros(B, dtype=torch.long, device=device)
    if isinstance(k_max, torch.Tensor):                       # tensor con 0 = libre
        k = k_max.to(device=device, dtype=torch.long)
        return k > 0, k
    k = torch.as_tensor([0 if x is None else int(x) for x in k_max], dtype=torch.long, device=device)
    return k > 0, k


class VectorRoutingEnv:
    """
    B episodios en paralelo; el estado son tensores (B, ...) en `device`. Interfaz propia (no Gymnasium).

    Parámetros
    ----------
    time_matrix    : (N, N) horas de viaje.
    distance_arr, diesel_arr : (N, N) para el reward y el costo de cada lane.
    market         : MarketProcess (tablas por día: precios por nivel, cuantiles, m, índice, LTR).
    device         : dispositivo del estado (por defecto config.DEVICE).
    """

    def __init__(self, time_matrix, distance_arr, diesel_arr, market, device=None, max_duration: float = MAX_DURATION):
        from config import DEVICE
        self.device = torch.device(device) if device is not None else DEVICE
        dev = self.device
        T = (time_matrix.to_numpy(dtype=float) if hasattr(time_matrix, "to_numpy")
             else np.asarray(time_matrix, dtype=float))
        self.num_nodes = N = T.shape[0]
        self.max_duration = float(max_duration)
        self._market = market
        self._max_steps = max_pasos_cotizacion(T, max_duration)
        self.T = torch.as_tensor(T, dtype=torch.float64, device=dev)              # (N, N) horas
        self.T32 = self.T.float()
        # Se conservan los dtypes de los datos (el diésel viene en float32): así el costo y su redondeo salen
        # idénticos a los de los baselines, UB-PI y la DP, que usan las mismas matrices en numpy.
        self.dist = torch.as_tensor(np.asarray(distance_arr), device=dev)
        self.diesel = torch.as_tensor(np.asarray(diesel_arr), device=dev)
        self.costo = lane_cost_t(self.dist, self.diesel)                            # (N, N) USD
        self._reposicion = torch.as_tensor(vacio_permitido(distance_arr), device=dev)   # (N, N) vacío permitido
        vuelta = self.T + self.T.T
        vuelta.fill_diagonal_(float("inf"))
        self._t_vuelta_min = vuelta.amin(dim=1)        # (N,) la ida y vuelta más corta desde cada nodo (esperar en casa)
        self._jj = torch.arange(N, device=dev)
        self._u_niveles = torch.as_tensor(U_NIVELES_ARR, dtype=torch.float64, device=dev)
        self._indice_std = float(market.indice_std)

    # ─────────────────────────────────────────────────────────────────────────
    # Ciclo de vida
    # ─────────────────────────────────────────────────────────────────────────

    def reset(self, mundos, start_nodes, k_max=None) -> torch.Tensor:
        """Empieza B episodios: mundos (market_process.Mundos), start_nodes (B,), k_max lista de (int | None) o
        None (todos libres). Devuelve la máscara de destinos (B, N) bool."""
        dev = self.device
        self._tab = self._market.tensores(dev)                       # tablas por día (con el κ actual del mercado)
        self._mundos = mundos
        B, N, K, C = mundos.B, self.num_nodes, N_NIVELES, COTIZACIONES_MAX_POR_NODO
        self._B = B
        self._filas = torch.arange(B, device=dev)
        self._start_node = torch.as_tensor(np.asarray(start_nodes) if not isinstance(start_nodes, torch.Tensor)
                                           else start_nodes, dtype=torch.long, device=dev)
        self._current_node = self._start_node.clone()
        self._time_elapsed = torch.zeros(B, dtype=torch.float32, device=dev)
        self._step_count = torch.zeros(B, dtype=torch.long, device=dev)
        self._active = torch.ones(B, dtype=torch.bool, device=dev)
        self._k_restricted, self._k_max = k_tensor(k_max, B, dev)
        self._k_remaining = self._k_max.clone()
        self._escala = torch.exp(mundos.delta.to(device=dev, dtype=torch.float64))   # (B,) e^δ
        z = lambda *s, dtype=torch.float32: torch.zeros(*s, dtype=dtype, device=dev)
        self._last_time_violation = z(B, dtype=torch.bool)
        self._last_raw_reward = z(B)
        self._reward_row = z(B, N)
        self._current_loads_row = z(B, N, dtype=torch.int32)
        self._current_rate_row = z(B, N)
        self._precios = z(B, N, K, dtype=torch.float64)
        self._u_comp = z(B, N, C, dtype=torch.float64)
        self._precio_comp = z(B, N, C, dtype=torch.float64)
        self._m = torch.ones(B, dtype=torch.float64, device=dev)
        self._avance_visita = z(B, dtype=torch.long)     # días desde la salida al revelar el tablero de la visita
        self._dia_visita = mundos.dia0.to(device=dev, dtype=torch.long).clone()   # su día absoluto (recortado)
        self._perdidas = z(B, N, dtype=torch.long)
        self._n_rechazos = z(B, dtype=torch.long)
        self._last_cotizo = z(B, dtype=torch.bool)
        self._last_gano = z(B, dtype=torch.bool)
        self._last_espero = z(B, dtype=torch.bool)
        nan = torch.full((B,), float("nan"), dtype=torch.float64, device=dev)
        self._last_precio, self._last_precio_comp = nan, nan.clone()
        self._revelar_tablero(torch.ones(B, dtype=torch.bool, device=dev))
        masks = self._compute_masks()
        self._sin_salida = ~masks.any(dim=1)                            # no hay ninguna jugada desde casa: no sale
        self._active &= ~self._sin_salida
        masks[:, 0] |= self._sin_salida                                 # filas inactivas: una opción ficticia válida
        self._opciones[:, 0, OPCION_VACIO] |= self._sin_salida
        return masks

    def step(self, actions: torch.Tensor, opciones: torch.Tensor) -> tuple:
        """Un paso de los B episodios: actions (B,) destino, opciones (B,) nivel 0 … K−1 o VACÍO = K.

        Solo se procesan las filas activas; las inactivas reciben reward 0 y terminated False.
        Devuelve (rewards (B,) f32, terminated (B,) bool, next_masks (B, N) bool), con terminated True solo en las
        filas que terminan en ESTE paso.
        """
        dev, B, K = self.device, self._B, N_NIVELES
        actions = actions.to(device=dev, dtype=torch.long)
        opciones = opciones.to(device=dev, dtype=torch.long)
        filas, act = self._filas, self._active
        if CHECK_TENSORS:
            malas = act & ~self._opciones[filas, actions, opciones]
            if malas.any():
                raise ValueError(f"(destino, opción) fuera de la máscara en las filas {malas.nonzero().flatten().tolist()}")

        cotiza = act & (opciones < K)
        nivel = opciones.clamp(max=K - 1)
        intento = self._perdidas[filas, actions].clamp(max=COTIZACIONES_MAX_POR_NODO - 1)     # la carga que se cotiza
        u_comp = self._u_comp[filas, actions, intento]
        gano = cotiza & (self._u_niveles[nivel] < u_comp)                                      # pricing.gana
        pierde = cotiza & ~gano
        espera = act & (actions == self._current_node)          # (i, VACÍO): esperar al día siguiente
        viaja = act & ~pierde & ~espera                          # ganó, o VACÍO a otro nodo

        # ── Tiempo ────────────────────────────────────────────────────────────
        t = self._time_elapsed
        t_viaje = self.T32[self._current_node, actions]
        next_time = torch.where(viaja, t + t_viaje,
                                torch.where(espera, fin_espera(t).float(), t + COTIZACION_HORAS)).float()

        # ── Reward de la lane (unidades originales) ───────────────────────────
        nodos = self._current_node
        precio = torch.where(gano, self._precios[filas, actions, nivel], torch.zeros_like(u_comp))
        raw = lane_reward_t(precio, gano.long(), self.dist[nodos, actions], self.diesel[nodos, actions]).float()
        raw = torch.where(viaja, raw, torch.zeros_like(raw))
        self._last_raw_reward = raw
        self._last_espero, self._last_cotizo, self._last_gano = espera, cotiza, gano
        nan = torch.full_like(precio, float("nan"))
        self._last_precio = torch.where(gano, precio, nan)
        self._last_precio_comp = torch.where(cotiza, self._precio_comp[filas, actions, intento], nan)

        # ── Terminación (condición A) ─────────────────────────────────────────
        vuelve = viaja & (actions == self._start_node)
        excede = act & (next_time > self.max_duration)
        cond_a = vuelve | excede
        success = vuelve & (next_time <= self.max_duration)
        time_fail = act & cond_a & ~success
        self._last_time_violation = time_fail

        terminal = torch.zeros(B, dtype=torch.float32, device=dev)
        if USE_REWARD_SHAPING:
            denom = max(self.num_nodes - 1, 1)
            bono = (RETURN_SUCCESS_BONUS + (RETURN_SUCCESS_BONUS * 0.5) * (self._step_count.float() / denom)
                    + (RETURN_SUCCESS_BONUS * 0.3) * (next_time / self.max_duration))
            terminal = torch.where(success, bono, terminal)
        terminal = torch.where(time_fail, torch.full_like(terminal, TIME_VIOLATION_PENALTY), terminal)
        rewards = torch.where(act, raw / REWARD_SCALE_FACTOR + terminal, torch.zeros_like(raw))

        # ── Estado ────────────────────────────────────────────────────────────
        self._current_node = torch.where(viaja, actions, self._current_node)
        self._time_elapsed = torch.where(act, next_time, self._time_elapsed)
        self._step_count = self._step_count + viaja.long()                       # arcos recorridos
        self._k_remaining = self._k_remaining - viaja.long()                     # rechazo y espera no consumen K
        self._perdidas[filas, actions] += pierde.long()
        self._n_rechazos = self._n_rechazos + pierde.long()

        # ── Tablero del nodo de llegada, o del día nuevo tras esperar (nueva visita) ──
        llega = (viaja | espera) & ~cond_a
        self._perdidas = torch.where(llega[:, None], torch.zeros_like(self._perdidas), self._perdidas)
        self._n_rechazos = torch.where(llega, torch.zeros_like(self._n_rechazos), self._n_rechazos)
        self._revelar_tablero(llega)

        self._active = act & ~cond_a
        next_masks = self._compute_masks()
        cond_b = ~next_masks.any(dim=1) & self._active          # se quedó sin jugadas en casa: el ciclo falla
        rewards = torch.where(cond_b, rewards + TIME_VIOLATION_PENALTY, rewards)
        self._last_time_violation = self._last_time_violation | cond_b
        self._active = self._active & ~cond_b
        terminated = (act & cond_a) | cond_b

        vacias = ~next_masks.any(dim=1)                          # filas inactivas: una opción ficticia válida
        next_masks[:, 0] |= vacias
        self._opciones[:, 0, OPCION_VACIO] |= vacias
        return rewards, terminated, next_masks

    # ─────────────────────────────────────────────────────────────────────────
    # Internos
    # ─────────────────────────────────────────────────────────────────────────

    def _dia(self) -> torch.Tensor:
        """(B,) día absoluto de la visita actual (el de su tablero), recortado al último día con datos del mundo."""
        return self._dia_visita

    def _revelar_tablero(self, filas: torch.Tensor) -> None:
        """Tablero del nodo actual en las filas `filas` (B,) bool: la fila del mundo del día del reloj y ese nodo. Ese
        día queda como el de la visita (`_dia_visita`) hasta el próximo tablero."""
        tab, mun, b, i = self._tab, self._mundos, self._filas, self._current_node
        avance = torch.div(self._time_elapsed, HORAS_DIA_MERCADO, rounding_mode="floor").long()
        self._avance_visita = torch.where(filas, avance, self._avance_visita)
        self._dia_visita = torch.where(filas, (mun.dia0 + avance).clamp(max=mun.dia_max), self._dia_visita)
        dia = self._dia_visita
        slot = dia - mun.dia0
        loads = mun.loads[b, slot, i]                                            # (B, N)
        u = mun.u[b, slot, i].double()                                           # (B, N, C)
        escala = self._escala[:, None, None]
        precios = tab["precios"][dia, i] * escala                                # (B, N, K)
        m = tab["m"][dia, i]                                                     # (B,)
        u_comp = percentil_competidor_t(u, m[:, None, None])
        cuant = tab["cuant"][dia, i]                                             # (B, N, 6)
        precio_comp = interp_cuantiles_t(u_comp, cuant[:, :, None, :].expand(*u_comp.shape, 6)) * escala
        mediana = precios[..., _NIVEL_MEDIANA]
        reward = lane_reward_t(mediana, loads, self.dist[i], self.diesel[i]).float()
        reward[b, i] = 0.0                                                       # self-loop = esperar: margen 0
        f1, f2, f3 = filas[:, None], filas[:, None, None], filas
        self._current_loads_row = torch.where(f1, loads, self._current_loads_row)
        self._precios = torch.where(f2, precios, self._precios)
        self._u_comp = torch.where(f2, u_comp, self._u_comp)
        self._precio_comp = torch.where(f2, precio_comp, self._precio_comp)
        self._m = torch.where(f3, m, self._m)
        self._current_rate_row = torch.where(f1, mediana.float(), self._current_rate_row)
        self._reward_row = torch.where(f1, reward, self._reward_row)

    def _compute_masks(self) -> torch.Tensor:
        """Máscara de destinos (B, N) bool; deja en self._opciones (B, N, K+1) las opciones válidas.

        Por destino j desde el nodo actual i (ver el docstring del módulo):
          · niveles: j ≠ i y a la lane le queda alguna carga cotizable (`intentos` > 0);
          · VACÍO a j ≠ i: reposicionamiento (≤ REPOSICION_MAX_MILLAS) o, desde un intermedio, volver a casa;
          · VACÍO en j = i: esperar al siguiente día de mercado.
        Factibilidad de tiempo: después del movimiento (o de la espera) hay que poder volver a casa dentro de
        max_duration; en casa, antes de salir, la espera deja tiempo para la ida y vuelta más corta.
        Con k_remaining == 1 solo quedan casa y esperar.
        """
        K = N_NIVELES
        i, h, t = self._current_node, self._start_node, self._time_elapsed.double()
        jj = self._jj[None, :]
        es_i, es_h = jj == i[:, None], jj == h[:, None]
        en_casa = i == h                                                           # (B,) aún no sale
        T = self.T
        MAX = self.max_duration

        # Tiempo: llegar a j y volver a casa (casa misma: solo llegar).
        t_vuelta = torch.where(es_h, torch.zeros_like(T[0]), T[:, h].T)            # (B, N) T[j, h]
        # Volver a casa desde un intermedio vale siempre: si ya no alcanza, el paso termina con la penalización.
        a_tiempo = ((t[:, None] + T[i, :] + t_vuelta) <= MAX) | (es_h & ~en_casa[:, None])
        t_desp = fin_espera(t)                                                     # (B,) fin de la espera
        espera_ok = torch.where(en_casa, t_desp + self._t_vuelta_min[h], t_desp + T[i, h]) <= MAX

        # Lejos de casa, cotizar exige que, si se pierde (COTIZACION_HORAS), todavía se pueda volver a casa.
        pierde_ok = en_casa | (t + COTIZACION_HORAS + T[i, h] <= MAX)
        cotizable = (self.intentos > 0) & ~es_i & a_tiempo & pierde_ok[:, None]
        vacio = (self._reposicion[i] | (es_h & ~en_casa[:, None])) & a_tiempo & ~es_i
        solo_casa = (self._k_restricted & (self._k_remaining == 1))[:, None] & ~es_h     # K: con un arco, solo casa
        cotizable = cotizable & ~solo_casa
        vacio = (vacio & ~solo_casa) | (es_i & espera_ok[:, None])

        op = torch.cat([cotizable[:, :, None].expand(-1, -1, K), vacio[:, :, None]], dim=-1)
        op = op & self._active[:, None, None]
        self._opciones = op
        return op.any(dim=-1)

    # ─────────────────────────────────────────────────────────────────────────
    # Propiedades de solo lectura (tensores en el dispositivo)
    # ─────────────────────────────────────────────────────────────────────────

    @property
    def B(self) -> int:
        return self._B

    @property
    def start_node(self) -> torch.Tensor:
        return self._start_node

    @property
    def current_node(self) -> torch.Tensor:
        return self._current_node

    @property
    def time_elapsed(self) -> torch.Tensor:
        return self._time_elapsed

    @property
    def step_count(self) -> torch.Tensor:
        """(B,) arcos recorridos (un rechazo no cuenta)."""
        return self._step_count

    @property
    def max_steps(self) -> int:
        """Cota del número de pasos de un episodio, rechazos incluidos (max_pasos_cotizacion)."""
        return self._max_steps

    @property
    def current_day(self) -> torch.Tensor:
        """(B,) día absoluto de la visita actual (el de su tablero; recortado al último día con datos del mundo)."""
        return self._dia()

    @property
    def dias_desde_salida(self) -> torch.Tensor:
        """(B,) días de mercado entre la salida y la visita actual (sin recortar)."""
        return self._avance_visita

    @property
    def current_dow(self) -> torch.Tensor:
        """(B,) día de la semana (lunes = 0) de la visita actual, sin recortar."""
        return (self._tab["dow"][self._mundos.dia0] + self._avance_visita) % 7

    @property
    def start_dow(self) -> torch.Tensor:
        """(B,) día de la semana (lunes = 0) del día de salida de cada episodio."""
        return self._tab["dow"][self._mundos.dia0]

    @property
    def reward_rows(self) -> torch.Tensor:
        """(B, N) float32 — margen de las lanes salientes si se cobrara la mediana del mercado
        (f0 de las features del agente; el ingreso real depende de la cotización)."""
        return self._reward_row

    @property
    def mundos(self):
        return self._mundos

    @property
    def cargas_hoy(self) -> torch.Tensor:
        """(B, N, N) cargas publicadas HOY (el día de la visita actual) en todas las lanes de la red: el load
        board público. La fila del nodo actual es su tablero; al llegar a j el mismo día, el de j es su fila."""
        return self._mundos.loads[self._filas, self._dia() - self._mundos.dia0]

    @property
    def margen_hoy(self) -> torch.Tensor:
        """(B, N, N) USD: margen de cada lane de la red si se cobrara la mediana de HOY (p50 · e^δ · dist − costo)."""
        mediana = self._tab["precios"][self._dia(), :, :, _NIVEL_MEDIANA] * self._escala[:, None, None]
        return mediana * self.dist[None] - self.costo[None]

    @property
    def current_loads_row(self) -> torch.Tensor:
        """(B, N) cargas del tablero del nodo actual."""
        return self._current_loads_row

    @property
    def rate_row_obs(self) -> torch.Tensor:
        """(B, N) precio en la mediana del mercado de cada lane saliente (USD/milla)."""
        return self._current_rate_row

    @property
    def index_std(self) -> float:
        """Desviación del índice histórico (normaliza la feature de índice del agente)."""
        return self._indice_std

    @property
    def market_index(self) -> torch.Tensor:
        """(B, N) índice de mercado observado de todos los nodos el día de la visita actual."""
        return self._tab["indice"][self._dia()]

    @property
    def ltr_actual(self) -> torch.Tensor:
        """(B,) load-to-truck ratio del hub actual el día de la visita actual."""
        return self._tab["ltr"][self._dia(), self._current_node]

    @property
    def k_remaining(self) -> torch.Tensor:
        return self._k_remaining

    @property
    def k_restricted(self) -> torch.Tensor:
        return self._k_restricted

    @property
    def last_raw_reward(self) -> torch.Tensor:
        """(B,) float32 — reward de la lane del último step() en USD, sin REWARD_SCALE_FACTOR ni bonos ni
        penalizaciones (0 en un rechazo y en filas inactivas). Es la unidad en que se reporta el margen."""
        return self._last_raw_reward

    @property
    def last_time_violation(self) -> torch.Tensor:
        """(B,) bool — el step() actual terminó por pasarse de max_duration o por quedarse sin jugadas."""
        return self._last_time_violation

    @property
    def opciones_validas(self) -> torch.Tensor:
        """(B, N, K+1) bool — opciones válidas por destino: niveles 0 … K−1 y VACÍO (índice OPCION_VACIO)."""
        return self._opciones

    @property
    def precios_row(self) -> torch.Tensor:
        """(B, N, K) USD/milla de cada nivel en las lanes salientes del nodo actual."""
        return self._precios

    @property
    def dist_rows(self) -> torch.Tensor:
        """(B, N) distancia de las lanes salientes del nodo actual."""
        return self.dist[self._current_node]

    @property
    def cost_rows(self) -> torch.Tensor:
        """(B, N) costo USD de las lanes salientes del nodo actual (misma fórmula que lane_reward)."""
        return self.costo[self._current_node]

    @property
    def competencia(self) -> torch.Tensor:
        """(B,) competidores efectivos m del nodo actual (LTR del día de llegada)."""
        return self._m

    @property
    def perdidas(self) -> torch.Tensor:
        """(B, N) cargas perdidas por lane en la visita actual al nodo."""
        return self._perdidas

    @property
    def intentos(self) -> torch.Tensor:
        """(B, N) cuántas cargas de cada lane se pueden cotizar todavía en esta visita:
        min(cargas − perdidas, COTIZACIONES_MAX_POR_NODO − rechazos)."""
        return torch.minimum(self._current_loads_row.long() - self._perdidas,
                             (COTIZACIONES_MAX_POR_NODO - self._n_rechazos)[:, None])

    @property
    def n_rechazos(self) -> torch.Tensor:
        """(B,) rechazos en la visita actual al nodo."""
        return self._n_rechazos

    @property
    def u_comp_row(self) -> torch.Tensor:
        """(B, N) percentil del competidor de la carga que se cotizaría ahora en cada lane. Solo para la cota con
        información perfecta y los tests: ninguna política lo puede leer."""
        c = self._perdidas.clamp(max=COTIZACIONES_MAX_POR_NODO - 1)
        return torch.gather(self._u_comp, -1, c[..., None])[..., 0]

    @property
    def u_comp_cargas(self) -> torch.Tensor:
        """(B, N, C) percentil del competidor de cada carga (idem: solo cota y tests)."""
        return self._u_comp

    @property
    def active(self) -> torch.Tensor:
        """(B,) bool — episodios aún en curso (tras reset(), False en las filas que no tienen ninguna jugada)."""
        return self._active

    @property
    def sin_salida(self) -> torch.Tensor:
        """(B,) bool — el ciclo terminó en casa sin salir (ninguna jugada posible): margen 0, no es violación."""
        return self._sin_salida

    @property
    def last_espero(self) -> torch.Tensor:
        """(B,) bool — el último step() fue una espera (self-loop VACÍO)."""
        return self._last_espero

    @property
    def last_cotizo(self) -> torch.Tensor:
        """(B,) bool — el último step() fue una cotización (nivel, no VACÍO)."""
        return self._last_cotizo

    @property
    def last_gano(self) -> torch.Tensor:
        """(B,) bool — el último step() cotizó y ganó la carga."""
        return self._last_gano

    @property
    def last_precio(self) -> torch.Tensor:
        """(B,) USD/milla cobrado en el último step() (NaN si no se ganó una carga)."""
        return self._last_precio

    @property
    def last_precio_competidor(self) -> torch.Tensor:
        """(B,) USD/milla del mejor competidor en la lane cotizada en el último step() (NaN si no cotizó)."""
        return self._last_precio_comp
