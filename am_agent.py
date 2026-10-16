"""
am_agent.py
===========
Agente AM-Actor-Critic mínimo para ruteo.

Combina AttentionEncoder + ContextNetwork + AttentionDecoder en un único
nn.Module. Expone:

  generate_route()      — rollout determinista (beam search) para evaluación o inferencia.
  beam_search_dynamic() — beam search con día dinámico y disponibilidad de lanes.
  estimate_value()      — estimación V(s) del crítico, usada como score en el beam search.

El techo de pasos en generate_route() y en _beam_search() es num_nodes
(máximo de arcos posibles en un ciclo hamiltoniano), no un conteo arbitrario.
"""

import numpy as np
import torch
import torch.nn as nn

from config import MAX_DURATION, REWARD_SCALE_FACTOR
from attention_encoder import (
    AttentionEncoder,
    ContextNetwork,
    build_node_features,
    build_market_features,
    build_temporal_features,
    build_budget_features,
    N_NODE_FEATURES,
)
from attention_decoder import AttentionDecoder
from problem_data import draw_lane_availability
from debug_utils import check_tensor


class AMRoutingAgent(nn.Module):
    """
    Agente de ruteo basado en Attention Model.

    Parámetros
    ----------
    num_nodes : int   — tamaño del grafo (escalable).
    d_h       : int   — dimensión del embedding (default 128).
    n_heads   : int   — cabezas de atención (default 8, debe dividir d_h).
    n_layers  : int   — capas del encoder Transformer (default 3).
    d_ff      : int   — dimensión interna FFN (default 512).
    clip_C    : float — clipping tanh del pointer (default 10.0).
    device    : torch.device
    pre_norm  : bool  — pre-norm (True) vs post-norm (False) en el encoder (default True).
    dropout   : float — dropout en las sublayers del encoder (default 0.0).
    """

    def __init__(
        self,
        num_nodes: int,
        d_h:       int   = 128,
        n_heads:   int   = 8,
        n_layers:  int   = 3,
        d_ff:      int   = 512,
        clip_C:    float = 10.0,
        device:    torch.device = None,
        pre_norm:  bool  = True,
        dropout:   float = 0.0,
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.d_h       = d_h
        self.device    = device or torch.device("cpu")

        self.encoder     = AttentionEncoder(
            N_NODE_FEATURES, d_h, n_heads, n_layers, d_ff,
            pre_norm=pre_norm, dropout=dropout,
        )
        self.context_net = ContextNetwork(d_h, n_market=1)
        self.decoder     = AttentionDecoder(d_h, n_heads_glimpse=n_heads, clip_C=clip_C)

        self.to(self.device)

    # ── Auxiliares ────────────────────────────────────────────────────────────

    def _encode_step(
        self,
        current_node:           int,
        start_node:             int,
        visited_set:            set,
        time_elapsed:           float,
        step_count:             int,
        reward_matrix_penalized,
        time_matrix,
        distance_arr:           np.ndarray,
        max_duration:           float,
        ltr_stack:              np.ndarray,        # [num_nodes, 120]
        trucks_stack:           np.ndarray,        # [num_nodes, 120, 3]
        day_idx:                int,
        reward_global_p95:      float      = 1.0,
        k_remaining:            int        = None, # presupuesto K restante (None = libre)
        k_restricted:           bool       = False,
        k_condition_neutral:    bool       = False, # ablación: máscara real, features neutras
        avail_expected_arr:     np.ndarray = None, # [num_nodes, num_nodes] — E[loads] histórico (train)
    ):
        """
        Construye tensores de entrada y ejecuta encoder + context_net.

        k_remaining/k_restricted describen el presupuesto de cardinalidad K
        del episodio (ver routing_env.py). Con los defaults se comporta como
        el caso libre (sin presupuesto K).
        k_condition_neutral fuerza las budget features a neutras
        (k_restricted=0, k_remaining_norm=1.0) sin afectar la máscara —
        ablación para cuantificar el aporte del condicionamiento.

        Retorna
        -------
        embeddings : Tensor (1, N, d_h)
        h_t        : Tensor (1, d_h)
        """
        trucks_day = trucks_stack[:, day_idx, :]   # (num_nodes, 3)

        node_feats = build_node_features(
            current_node, start_node, visited_set,
            reward_matrix_penalized, time_matrix, distance_arr,
            self.num_nodes, max_duration,
            trucks_stack=trucks_day,
            reward_global_p95=reward_global_p95,
            avail_expected_arr=avail_expected_arr,
        )
        temporal = build_temporal_features(
            time_elapsed, step_count, max_duration, self.num_nodes
        )
        market = build_market_features(current_node, day_idx, ltr_stack)
        budget = build_budget_features(
            k_remaining, k_restricted and not k_condition_neutral
        )

        feats_t    = torch.from_numpy(node_feats).unsqueeze(0).to(self.device)   # (1,N,N_NODE_FEATURES)
        temporal_t = torch.from_numpy(temporal).unsqueeze(0).to(self.device)     # (1,3)
        market_t   = torch.from_numpy(market).unsqueeze(0).to(self.device)       # (1,1)
        budget_t   = torch.from_numpy(budget).unsqueeze(0).to(self.device)       # (1,2)

        embeddings, graph_emb = self.encoder(feats_t)                            # (1,N,d_h), (1,d_h)
        check_tensor("_encode_step embeddings", embeddings)
        check_tensor("_encode_step graph_emb", graph_emb)
        current_emb = embeddings[:, current_node, :]                             # (1, d_h)
        h_t = self.context_net(graph_emb, current_emb, temporal_t, market_t, budget_t)  # (1, d_h)
        check_tensor("_encode_step h_t", h_t)

        return embeddings, h_t

    @staticmethod
    def _build_mask(
        current_node: int,
        start_node:   int,
        visited_set:  set,
        num_nodes:    int,
    ) -> torch.Tensor:
        """
        Máscara de acciones compatible con RoutingEnv._get_action_mask():
          1 = válido, 0 = inválido.
        Retorna Tensor (1, N) int8.
        """
        mask = np.ones(num_nodes, dtype=np.int8)
        mask[current_node] = 0
        for v in visited_set:
            if v != start_node:
                mask[v] = 0
        return torch.from_numpy(mask).unsqueeze(0)

    def _beam_search(
        self,
        start_node:             int,
        reward_matrix_penalized,
        time_matrix,
        distance_arr:           np.ndarray,
        max_duration:           float,
        beam_width:             int,
        ltr_stack:              np.ndarray,        # [num_nodes, 120]
        trucks_stack:           np.ndarray,        # [num_nodes, 120, 3]
        day_idx:                int,
        reward_global_p95:      float      = 1.0,
        critic:                 nn.Module  = None,
        value_coef:             float      = 1.0,
        avail_expected_arr:     np.ndarray = None, # [num_nodes, num_nodes] — E[loads] histórico (train)
    ):
        """
        Rollout unificado con beam search para cualquier beam_width >= 1.

        Con beam_width=1 produce exactamente el mismo resultado que el greedy
        original (el nodo de mayor logit equivale a greedy_action). Con
        beam_width>1 mantiene los k mejores caminos parciales ordenados por
        total_reward acumulado.

        El techo de pasos es self.num_nodes — techo defensivo contra bucles
        infinitos; la terminación natural ocurre al regresar al depot o cuando
        ningún nodo intermedio es factible temporalmente.

        Retorna
        -------
        route        : list[int] | None
        total_reward : float
        time_elapsed : float
        """
        beams = [{
            "current_node":  start_node,
            "time_elapsed":  0.0,
            "visited_set":   {start_node},
            "visited_inter": set(),
            "route":         [start_node],
            "total_reward":  0.0,
            "returned_home": False,
        }]

        for step in range(self.num_nodes):
            next_beams = []

            for beam in beams:
                if beam["returned_home"]:
                    next_beams.append(beam)
                    continue

                current_node = beam["current_node"]
                time_elapsed = beam["time_elapsed"]
                visited_set  = set(beam["visited_set"])

                embeddings, h_t = self._encode_step(
                    current_node, start_node, visited_set,
                    time_elapsed, step,
                    reward_matrix_penalized, time_matrix, distance_arr,
                    max_duration, ltr_stack, trucks_stack, day_idx,
                    reward_global_p95=reward_global_p95,
                    avail_expected_arr=avail_expected_arr,
                )

                # Máscara base: self-loop + intermedios visitados
                mask_int = self._build_mask(
                    current_node, start_node, visited_set, self.num_nodes
                ).to(self.device)

                # Lookahead temporal (vectorizado): excluir intermedios que impidan el retorno
                mask_np    = mask_int.cpu().numpy()[0]
                non_start  = np.arange(self.num_nodes) != start_node
                t_to_j     = time_matrix[current_node]              # (N,)
                t_j_start  = time_matrix[:, start_node]             # (N,)
                over_budget = (time_elapsed + t_to_j + t_j_start) > max_duration + 1e-6
                mask_np[non_start & over_budget] = 0

                valid_non_start = [j for j in range(self.num_nodes)
                                   if mask_np[j] == 1 and j != start_node]

                if not valid_non_start and current_node != start_node:
                    # Forzar retorno al depot como único candidato
                    candidate_nodes = [start_node]
                else:
                    # Obtener logits del decoder sin samplear
                    bool_mask = (mask_int == 0)   # True = inválido, (1, N)
                    logits = self.decoder._logits(h_t, embeddings, bool_mask)  # (1, N)
                    logits_np = logits[0].cpu().numpy()

                    valid_nodes = [j for j in range(self.num_nodes)
                                   if mask_np[j] == 1]
                    if not valid_nodes:
                        next_beams.append(beam)
                        continue

                    # Top-k nodos válidos por logit
                    valid_nodes.sort(key=lambda j: logits_np[j], reverse=True)
                    candidate_nodes = valid_nodes[:beam_width]

                # Crear beams hijos para cada candidato
                for next_node in candidate_nodes:
                    step_time   = float(time_matrix[current_node, next_node])
                    step_reward = float(reward_matrix_penalized[current_node, next_node])

                    if time_elapsed + step_time > max_duration + 1e-6 and next_node != start_node:
                        continue

                    new_visited_set   = set(visited_set)
                    new_visited_inter = set(beam["visited_inter"])
                    new_visited_set.add(next_node)
                    if next_node != start_node:
                        new_visited_inter.add(next_node)

                    next_beams.append({
                        "current_node":  next_node,
                        "time_elapsed":  time_elapsed + step_time,
                        "visited_set":   new_visited_set,
                        "visited_inter": new_visited_inter,
                        "route":         beam["route"] + [next_node],
                        "total_reward":  beam["total_reward"] + step_reward,
                        "returned_home": next_node == start_node,
                    })

            if not next_beams:
                break

            if critic is not None:
                def _score(b):
                    if b["returned_home"]:
                        v = 0.0
                    else:
                        v = self.estimate_value(
                            critic, b["current_node"], start_node, b["visited_set"],
                            b["time_elapsed"], step + 1,
                            reward_matrix_penalized, time_matrix, distance_arr,
                            max_duration, ltr_stack, trucks_stack, day_idx,
                            reward_global_p95=reward_global_p95,
                            avail_expected_arr=avail_expected_arr,
                        )
                    return b["total_reward"] + value_coef * v * REWARD_SCALE_FACTOR

                next_beams.sort(key=_score, reverse=True)
            else:
                next_beams.sort(key=lambda b: b["total_reward"], reverse=True)
            beams = next_beams[:beam_width]

            if all(b["returned_home"] for b in beams):
                break

        # Intento de retorno forzado para beams que no cerraron el ciclo
        for beam in beams:
            if not beam["returned_home"] and beam["current_node"] != start_node:
                cn    = beam["current_node"]
                t_ret = float(time_matrix[cn, start_node])
                r_ret = float(reward_matrix_penalized[cn, start_node])
                if beam["time_elapsed"] + t_ret <= max_duration + 1e-6:
                    beam["time_elapsed"] += t_ret
                    beam["total_reward"] += r_ret
                    beam["route"].append(start_node)
                    beam["returned_home"] = True

        best_beam   = None
        best_reward = -np.inf
        for beam in beams:
            is_valid = (
                beam["returned_home"]
                and len(beam["route"]) > 1
                and beam["route"][0] == start_node
                and beam["route"][-1] == start_node
            )
            if is_valid and beam["total_reward"] > best_reward:
                best_reward = beam["total_reward"]
                best_beam   = beam

        if best_beam is None:
            return None, -np.inf, np.inf

        return best_beam["route"], best_beam["total_reward"], best_beam["time_elapsed"]

    # ── API pública ───────────────────────────────────────────────────────────

    @torch.no_grad()
    def generate_route(
        self,
        start_node:             int,
        reward_matrix_penalized,
        time_matrix,
        distance_arr:           np.ndarray,
        max_duration:           float      = MAX_DURATION,
        beam_width:             int        = None,
        ltr_stack:              np.ndarray = None,   # [num_nodes, 120]
        trucks_stack:           np.ndarray = None,   # [num_nodes, 120, 3]
        day_idx:                int        = 0,
        reward_global_p95:      float      = 1.0,
        critic:                 nn.Module  = None,
        value_coef:             float      = 1.0,
        avail_expected_arr:     np.ndarray = None,   # [num_nodes, num_nodes] — E[loads] histórico (train)
    ):
        """
        Rollout determinista (sin gradientes) con el modelo actual.

        Delega la lógica de búsqueda a _beam_search(), que funciona para
        cualquier beam_width >= 1 (beam_width=1 es el rollout greedy).

        Cuando beam_width es None, lo resuelve automáticamente vía
        get_beam_width(self.num_nodes) (ver config.get_beam_width_det: el
        ancho decrece con el tamaño del grafo, de 5 en ≤10 nodos a 3 en >75).

        El techo de pasos en _beam_search() es self.num_nodes — techo
        defensivo contra bucles infinitos; la terminación natural ocurre al
        regresar al depot o cuando ningún nodo intermedio es factible
        temporalmente.

        Retorna
        -------
        route        : list[int] | None
        total_reward : float
        time_elapsed : float
        """
        from config import get_beam_width as _get_beam_width
        if beam_width is None:
            beam_width = _get_beam_width(self.num_nodes)

        self.eval()
        try:
            return self._beam_search(
                start_node, reward_matrix_penalized, time_matrix,
                distance_arr, max_duration, beam_width,
                ltr_stack, trucks_stack, day_idx,
                reward_global_p95=reward_global_p95,
                critic=critic, value_coef=value_coef,
                avail_expected_arr=avail_expected_arr,
            )
        finally:
            self.train()

    @torch.no_grad()
    def beam_search_dynamic(
        self,
        start_node:        int,
        start_day_idx:     int,
        time_matrix,
        rate_stack:        np.ndarray,
        loads_stack:       np.ndarray,
        distance_arr:      np.ndarray,
        diesel_arr:        np.ndarray,
        max_duration:      float      = MAX_DURATION,
        beam_width:        int        = None,
        ltr_stack:         np.ndarray = None,   # [num_nodes, 120]
        trucks_stack:      np.ndarray = None,   # [num_nodes, 120, 3]
        lane_filter:       bool       = False,
        reward_global_p95: float      = 1.0,
        critic:            nn.Module  = None,
        value_coef:        float      = 1.0,
        k_max:              int       = None,   # presupuesto de cardinalidad K (None = libre)
        k_condition_neutral: bool     = False,   # ablación: máscara real, features neutras
        avail_expected_arr: np.ndarray = None,   # [num_nodes, num_nodes] — E[loads] histórico (train)
    ):
        """Beam search con días de mercado dinámicos y disponibilidad estocástica.

        Features del agente: matriz del día corriente de cada beam (igual que training).
        Recompensa acumulada: misma matriz del día corriente según time_elapsed de cada beam.
        lane_filter=True: excluye lanes sin carga en el día de llegada
          (draw_lane_availability), el mismo criterio que RoutingEnv, de modo
          que DRL y baselines compiten sobre la misma realización del mundo.
          lane_filter=False: sin filtrado de lanes (mundo determinista).

        k_max acota el número de arcos (incluido el retorno) por beam — cada
        beam arranca con k_remaining=k_max y lo decrementa por arco recorrido;
        con k_remaining==1 la única acción válida es start_node (retorno
        forzado), igual que la máscara de RoutingEnv. k_max=None (default) es
        el caso libre (sin presupuesto K). k_condition_neutral aplica la máscara real de
        K pero fuerza las budget features vistas por la red a neutras (para
        la ablación "máscara sin condicionamiento").
        """
        from config import get_beam_width as _get_beam_width
        from problem_data import build_day_matrices

        if beam_width is None:
            beam_width = _get_beam_width(self.num_nodes)
        max_day    = rate_stack.shape[0] - 1
        day_cache  = {}  # day_idx → rm_penalized; evita reconstruir el mismo día

        def get_rm(day_idx):
            if day_idx not in day_cache:
                _, rm = build_day_matrices(
                    rate_stack[day_idx], loads_stack[day_idx], distance_arr, diesel_arr
                )
                day_cache[day_idx] = rm
            return day_cache[day_idx]

        # Disponibilidad inicial: lanes desde start_node al llegar con time_elapsed=0
        if lane_filter:
            init_arrival_day   = min(start_day_idx, max_day)
            init_lane_exists   = draw_lane_availability(
                start_node, init_arrival_day, loads_stack,
            )
        else:
            init_lane_exists = None

        beams = [{
            "current_node":  start_node,
            "time_elapsed":  0.0,
            "visited_set":   {start_node},
            "visited_inter": set(),
            "route":         [start_node],
            "total_reward":  0.0,
            "returned_home": False,
            "lane_exists":   init_lane_exists,
            "k_remaining":   k_max,
        }]

        for step in range(self.num_nodes):
            next_beams = []

            for beam in beams:
                if beam["returned_home"]:
                    next_beams.append(beam)
                    continue

                current_node = beam["current_node"]
                time_elapsed = beam["time_elapsed"]
                visited_set  = set(beam["visited_set"])
                k_remaining  = beam["k_remaining"]

                day_offset = int(time_elapsed // 14)
                day_idx    = min(start_day_idx + day_offset, max_day)
                rm_day     = get_rm(day_idx)

                embeddings, h_t = self._encode_step(
                    current_node, start_node, visited_set,
                    time_elapsed, step,
                    rm_day, time_matrix, distance_arr,
                    max_duration, ltr_stack, trucks_stack, day_idx,
                    reward_global_p95=reward_global_p95,
                    k_remaining=k_remaining,
                    k_restricted=(k_max is not None),
                    k_condition_neutral=k_condition_neutral,
                    avail_expected_arr=avail_expected_arr,
                )

                mask_int = self._build_mask(
                    current_node, start_node, visited_set, self.num_nodes
                ).to(self.device)

                # Lookahead temporal (vectorizado)
                mask_np    = mask_int.cpu().numpy()[0]
                non_start  = np.arange(self.num_nodes) != start_node
                t_to_j     = time_matrix[current_node]
                t_j_start  = time_matrix[:, start_node]
                over_budget = (time_elapsed + t_to_j + t_j_start) > max_duration + 1e-6
                mask_np[non_start & over_budget] = 0

                # Filtrado de lanes (vectorizado) — mismo criterio que RoutingEnv.
                # No se aplica cuando current_node == start_node (primer paso del beam).
                if beam["lane_exists"] is not None and current_node != start_node:
                    lane_absent = (beam["lane_exists"] == 0)
                    mask_np[non_start & lane_absent] = 0

                # Presupuesto de cardinalidad K — mismo override que RoutingEnv:
                # con un solo arco de margen, la única acción válida es el retorno.
                force_k_return = (k_remaining == 1) and current_node != start_node

                valid_non_start = [j for j in range(self.num_nodes)
                                   if mask_np[j] == 1 and j != start_node]

                if (not valid_non_start or force_k_return) and current_node != start_node:
                    candidate_nodes = [start_node]
                else:
                    bool_mask = (mask_int == 0)
                    logits    = self.decoder._logits(h_t, embeddings, bool_mask)
                    logits_np = logits[0].cpu().numpy()

                    valid_nodes = [j for j in range(self.num_nodes) if mask_np[j] == 1]
                    if not valid_nodes:
                        next_beams.append(beam)
                        continue

                    valid_nodes.sort(key=lambda j: logits_np[j], reverse=True)
                    candidate_nodes = valid_nodes[:beam_width]

                for next_node in candidate_nodes:
                    step_time = float(time_matrix[current_node, next_node])
                    if time_elapsed + step_time > max_duration + 1e-6 and next_node != start_node:
                        continue

                    step_reward = float(rm_day[current_node, next_node])
                    new_time_elapsed = time_elapsed + step_time

                    # Disponibilidad de lanes desde next_node al llegar (mismo criterio que RoutingEnv).
                    if lane_filter:
                        child_arrival_day  = min(
                            start_day_idx + int(new_time_elapsed // 14), max_day
                        )
                        child_lane_exists = draw_lane_availability(
                            next_node, child_arrival_day, loads_stack,
                        )
                    else:
                        child_lane_exists = None

                    new_visited_set   = set(visited_set)
                    new_visited_inter = set(beam["visited_inter"])
                    new_visited_set.add(next_node)
                    if next_node != start_node:
                        new_visited_inter.add(next_node)

                    next_beams.append({
                        "current_node":  next_node,
                        "time_elapsed":  new_time_elapsed,
                        "visited_set":   new_visited_set,
                        "visited_inter": new_visited_inter,
                        "route":         beam["route"] + [next_node],
                        "total_reward":  beam["total_reward"] + step_reward,
                        "returned_home": next_node == start_node,
                        "lane_exists":   child_lane_exists,
                        "k_remaining":   (k_remaining - 1) if k_remaining is not None else None,
                    })

            if not next_beams:
                break

            if critic is not None:
                def _score(b):
                    if b["returned_home"]:
                        v = 0.0
                    else:
                        b_day_offset = int(b["time_elapsed"] // 14)
                        b_day_idx    = min(start_day_idx + b_day_offset, max_day)
                        b_rm_day     = get_rm(b_day_idx)
                        v = self.estimate_value(
                            critic, b["current_node"], start_node, b["visited_set"],
                            b["time_elapsed"], step + 1,
                            b_rm_day, time_matrix, distance_arr,
                            max_duration, ltr_stack, trucks_stack, b_day_idx,
                            reward_global_p95=reward_global_p95,
                            k_remaining=b["k_remaining"],
                            k_restricted=(k_max is not None),
                            k_condition_neutral=k_condition_neutral,
                            avail_expected_arr=avail_expected_arr,
                        )
                    return b["total_reward"] + value_coef * v * REWARD_SCALE_FACTOR

                next_beams.sort(key=_score, reverse=True)
            else:
                next_beams.sort(key=lambda b: b["total_reward"], reverse=True)
            beams = next_beams[:beam_width]

            if all(b["returned_home"] for b in beams):
                break

        for beam in beams:
            if not beam["returned_home"] and beam["current_node"] != start_node:
                cn         = beam["current_node"]
                day_offset = int(beam["time_elapsed"] // 14)
                rm_day     = get_rm(min(start_day_idx + day_offset, max_day))
                t_ret = float(time_matrix[cn, start_node])
                r_ret = float(rm_day[cn, start_node])
                if beam["time_elapsed"] + t_ret <= max_duration + 1e-6:
                    beam["time_elapsed"] += t_ret
                    beam["total_reward"] += r_ret
                    beam["route"].append(start_node)
                    beam["returned_home"] = True

        best_beam   = None
        best_reward = -np.inf
        for beam in beams:
            is_valid = (
                beam["returned_home"]
                and len(beam["route"]) > 1
                and beam["route"][0] == start_node
                and beam["route"][-1] == start_node
            )
            if is_valid and beam["total_reward"] > best_reward:
                best_reward = beam["total_reward"]
                best_beam   = beam

        if best_beam is None:
            return None, -np.inf, np.inf

        return best_beam["route"], best_beam["total_reward"], best_beam["time_elapsed"]

    def estimate_value(
        self,
        critic:                 nn.Module,
        current_node:           int,
        start_node:             int,
        visited_set:            set,
        time_elapsed:           float,
        step_count:             int,
        reward_matrix_penalized,
        time_matrix,
        distance_arr:           np.ndarray,
        max_duration:           float,
        ltr_stack:              np.ndarray = None,
        trucks_stack:           np.ndarray = None,
        day_idx:                int        = 0,
        reward_global_p95:      float      = 1.0,
        k_remaining:            int        = None,
        k_restricted:           bool       = False,
        k_condition_neutral:    bool       = False,
        avail_expected_arr:     np.ndarray = None,
    ) -> float:
        """Estima V(s) con el crítico (score de beams en generate_route/beam_search_dynamic)."""
        _, h_t = self._encode_step(
            current_node, start_node, visited_set,
            time_elapsed, step_count,
            reward_matrix_penalized, time_matrix, distance_arr,
            max_duration, ltr_stack, trucks_stack, day_idx,
            reward_global_p95=reward_global_p95,
            k_remaining=k_remaining,
            k_restricted=k_restricted,
            k_condition_neutral=k_condition_neutral,
            avail_expected_arr=avail_expected_arr,
        )
        return critic(h_t).item()
