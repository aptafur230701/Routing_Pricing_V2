"""
attention_encoder.py
====================
Transformer encoder for the AM-Actor-Critic routing agent.

Componentes:
  NodeFeatureProjection  — proyecta features crudas de nodos a dimensión d_h
  MultiHeadSelfAttention — atención multi-cabeza estándar (scaled dot-product)
  TransformerEncoderLayer — MHA + FFN + LayerNorm con residual
  AttentionEncoder        — apila L capas, produce embeddings por nodo
                            y embedding global del grafo
  ContextNetwork          — combina embedding global + nodo actual + features
                            temporales → vector de contexto h_t

El vector de contexto h_t es compartido por:
  · AttentionDecoder  (decisión de ruteo)
  · CriticHead        (función de valor V(s))

Features por nodo j (desde la posición actual del camión):
  [0] reward_norm   — ganancia neta del arco actual→j (día actual), normalizada
  [1] time_norm     — tiempo de viaje actual→j, normalizado por max_duration
  [2] dist_norm     — distancia actual→j, normalizada por máximo
  [3] is_depot      — 1.0 si j es el nodo de inicio del episodio, 0.0 si no
  [4] avail_prior   — E[loads] histórico (train) del arco actual→j, normalizado por
                      LOADS_EXPECTED_CLIP y clipeado a [0, 1]. Preserva la magnitud
                      real (avail_expected_arr) en vez de colapsarla a una
                      probabilidad de existencia; es puramente descriptivo.
  [5] reward_abs    — reward crudo / REWARD_GLOBAL_P95, clipeado a [0, 1]; 0 si negativo

Features de continuación (solo con USE_CONTINUATION_FEATURES; i = nodo actual, h = casa, t = tiempo transcurrido,
est = EstadisticasLane de market_process.py: la media del generador por día de la semana, de los días de ENTRENAMIENTO).
Con t_j = llegada a j y dow_j = (dow0 + t_j // 14) % 7. Son datos del mercado, no la salida de ningún baseline: la red
combina sola estas estadísticas en una continuación.
  [6]  home_time     — T[j, h] / MAX_DURATION (tiempo de j a casa)
  [7]  slack_after   — clip((MAX − t_j − T[j,h]) / MAX, 0, 1) (holgura después de j)
  [8]  ret_margin    — clip(G[dow_j, j, h] / p95, −1, 1): margen de la lane de regreso j→casa al precio mediano
  [9]  ret_loads     — min(L[dow_j, j, h], CLIP) / CLIP: cargas esperadas de esa lane
  [10] out_best      — clip(max_{l ∈ F_j} P1·G / p95, −1, 1): valor de la mejor lane que sale de j
  [11] out_loads     — media de min(L, CLIP) / CLIP sobre las lanes salientes factibles
  [12] out_factible  — |F_j| / N: qué fracción de las lanes que salen de j se pueden usar con el tiempo que queda
  [13] m_j           — log(M[dow_j, j]) / log(M_MAX): competencia media en j ese día de la semana
  F_j = {l ∉ {i, j, h}: t_j + T[j,l] + T[l,h] ≤ MAX} (hay tiempo de ir de j a l y volver a casa). Para j == h (el ciclo
  termina) [8]-[13] valen 0, salvo la espera en casa antes de salir (j == i == h), que sí lleva las salientes [10]-[13].
  La fila j == i es la ESPERA (routing_env): su "arco" termina en fin_espera(t), el siguiente día de mercado, así que
  [1] son las horas de espera y [6-13] describen i mañana. Su reward [0] es 0.

Índice de mercado (columna INDEX_COL: [14] con continuación, [6] sin ella):
  [14] market_index — índice(j, día actual) / desviación del índice histórico (una sola para todos los
                      nodos, calculada en el ajuste). Estado del mercado de salida del nodo j neto del
                      nivel nacional, en desviaciones estándar (market_process.py).

Cotización conjunta (especificacion_fase0_ruteo_pricing.md §4.4; las ÚLTIMAS N_COT_FEATURES columnas, en cero
fuera del modo cotización). Van al final para que los checkpoints anteriores se carguen con esas columnas en
cero (transfer_learning._reconcile_state_dict copia las columnas viejas en las primeras posiciones):
  [15] dispersion   — (p_0.8 − p_0.2) / p_0.5 de la lane i→j, recortada a [0, 1]
  [16] break_even   — costo / (dist · p_0.5): qué tan cerca está el mínimo rentable de la mediana, en [0, 2] / 2
  [17] intentos     — cargas de la lane i→j que se pueden cotizar todavía en esta visita / COTIZACIONES_MAX_POR_NODO
                      (> 0 si y solo si se puede cotizar)
  [18 … 17+K]       — v_k = P_k·(p_k·d − c) / p95: valor esperado de cotizar cada nivel, recortado a [−1, 1]
  [18+K … 17+2K]    — P_k = (1 − u_k)^m: probabilidad de ganar con cada nivel
  [18+2K]           — valor de la lane con todos sus intentos (pricing.valor_reintentos) si perderlos vale 0, / p95
  [19+2K]           — lo mismo si perderlos es recorrer la lane vacía (la vuelta): base −c
  Las de [18] en adelante son 0 si la lane no se puede cotizar. Las columnas de cotización entran también a la
  cabeza de precio (model/decoder.PriceHead), que así ve el valor de cada nivel y no solo el embedding.
  Las features [0] y [5] ya son, en ese modo, el margen si se cobrara la mediana (env.reward_rows).

Condicionamiento en presupuesto de cardinalidad K:
  ContextNetwork recibe, además del bloque temporal y de mercado, un bloque
  separado de budget features (ver build_budget_features_batch): k_remaining
  normalizado por K_NORM (constante global) y un flag k_restricted. No forma
  parte de N_NODE_FEATURES (no es una feature por nodo).

Compatibilidad:
  · Sin dependencias de am_agent.py, am_training.py, routing_env.py ni evaluation.py.
  · Las matrices (reward, time, distance) aceptan pd.DataFrame o np.ndarray.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import (
    LTR_CLIP, LOADS_EXPECTED_CLIP, K_NORM, USE_CONTINUATION_FEATURES, COMPETENCIA_M_MAX, COTIZACIONES_MAX_POR_NODO,
)
from pricing import N_NIVELES, U_NIVELES_ARR, indice_nivel, prob_ganar, valor_reintentos
from routing_env import fin_espera

# Número de features por nodo — debe coincidir con build_node_features()
N_BASE_NODE_FEATURES  = 6
N_CONT_FEATURES       = 8   # home_time, slack_after + 6 estadísticas de las lanes que salen de j
N_INDEX_FEATURES      = 1   # índice de mercado (columna INDEX_COL)
N_COT_FEATURES        = 3 + 2 * N_NIVELES + 2   # cotización conjunta: las últimas columnas
COT_V = slice(3, 3 + N_NIVELES)                              # dentro de las de cotización: v_k
COT_P = slice(3 + N_NIVELES, 3 + 2 * N_NIVELES)              # … y P_k
INDEX_COL = N_BASE_NODE_FEATURES + (N_CONT_FEATURES if USE_CONTINUATION_FEATURES else 0)
COT_COLS  = slice(INDEX_COL + N_INDEX_FEATURES, INDEX_COL + N_INDEX_FEATURES + N_COT_FEATURES)
N_NODE_FEATURES = INDEX_COL + N_INDEX_FEATURES + N_COT_FEATURES

# Features de contexto de la cotización (último bloque de ContextNetwork): log(m) / log(M_MAX) del nodo actual y
# rechazos en esta visita / COTIZACIONES_MAX_POR_NODO. En cero fuera del modo cotización.
N_COT_CONTEXT = 2
_U_20, _U_50, _U_80 = indice_nivel(0.2), indice_nivel(0.5), indice_nivel(0.8)

# Número de features de presupuesto de cardinalidad K — debe coincidir con
# build_budget_features_batch().
N_BUDGET_FEATURES = 2

# Día de la semana (one-hot, lunes = 0) del día en que está el camión: el proceso de cargas
# depende de él (factor s_d de 0,58 a 1,43), así que el agente lo necesita para estimar el
# valor de los nodos siguientes. Va como ÚLTIMO bloque de la entrada de ContextNetwork para que
# los checkpoints anteriores se carguen con esas columnas en cero (transfer_learning).
N_DOW_FEATURES = 7


# ─────────────────────────────────────────────────────────────────────────────
# Features de continuación
# ─────────────────────────────────────────────────────────────────────────────

def build_continuation_features_batch(
    current_node:      np.ndarray,   # (B,) int64 — i
    start_node:        np.ndarray,   # (B,) int64 — casa h
    time_elapsed:      np.ndarray,   # (B,) horas transcurridas t
    dow0:              np.ndarray,   # (B,) día de la semana (lunes = 0) del día de salida
    time_matrix_arr:   np.ndarray,   # (N, N) T
    est,                             # EstadisticasLane (market_process.MarketProcess.estadisticas_dow)
    max_duration:      float,
    reward_global_p95: float,
) -> np.ndarray:                     # (B, N, N_CONT_FEATURES) float32
    """Features de continuación por nodo j (ver docstring del módulo), sin loops sobre B ni N.

    Las 6 últimas son estadísticas (por día de la semana de llegada a j) de las lanes que SALEN de j: la red no ve
    esas lanes en ninguna otra entrada, porque sus features describen solo las lanes que salen del nodo actual i.
    """
    N    = np.asarray(time_matrix_arr).shape[0]
    T    = np.asarray(time_matrix_arr, dtype=np.float64)
    i    = np.asarray(current_node, dtype=np.int64)
    h    = np.asarray(start_node, dtype=np.int64)
    t    = np.asarray(time_elapsed, dtype=np.float64)
    dow0 = np.asarray(dow0, dtype=np.int64)
    MAX  = max(float(max_duration), 1e-6)
    p95  = max(float(reward_global_p95), 1e-6)
    jj   = np.arange(N)

    is_h = jj[None, :] == h[:, None]                                         # (B, N)
    is_i = jj[None, :] == i[:, None]                                         # (B, N)
    # j = i es la espera: el "arco" i→i termina al siguiente día de mercado (fin_espera).
    t_j  = np.where(is_i, fin_espera(t)[:, None], t[:, None] + T[i, :])     # (B, N) llegada a j
    T_jh = T[:, h].T                                                         # (B, N) tiempo de j a casa
    T_jh = np.where(is_h, 0.0, T_jh)                                         # casa → casa: sin tiempo

    home_time = T_jh / MAX
    slack     = np.clip((MAX - t_j - T_jh) / MAX, 0.0, 1.0)

    dow_j = (dow0[:, None] + (t_j // 14.0).astype(np.int64)) % 7             # (B, N) día de la semana al llegar a j

    # Lane de regreso j → casa.
    ret_margin = np.clip(est.G[dow_j, jj[None, :], h[:, None]] / p95, -1.0, 1.0)
    ret_loads  = np.minimum(est.L[dow_j, jj[None, :], h[:, None]], LOADS_EXPECTED_CLIP) / LOADS_EXPECTED_CLIP

    # Lanes salientes j → l con tiempo de llegar a l y volver a casa: l ∉ {i, j, h}.
    t_l   = t_j[:, :, None] + T[None, :, :]                                  # (B, N, N) llegada a l desde j
    T_lh  = T[:, h].T[:, None, :]                                            # (B, 1, N) T[l, h]
    F     = (
        (jj[None, None, :] != jj[None, :, None])                             # l ≠ j
        & (jj[None, None, :] != i[:, None, None])                            # l ≠ i
        & (jj[None, None, :] != h[:, None, None])                            # l ≠ h
        & (t_l + T_lh <= max_duration)
    )                                                                        # (B, N, N)
    dj, jx, lx = dow_j[:, :, None], jj[None, :, None], jj[None, None, :]
    n_fact    = F.sum(axis=2)                                                # (B, N)
    hay       = n_fact > 0
    valor     = np.where(F, est.P1[dj, jx, lx] * est.G[dj, jx, lx], -np.inf).max(axis=2)
    out_best  = np.where(hay, np.clip(valor / p95, -1.0, 1.0), 0.0)
    cargas    = np.minimum(est.L[dj, jx, lx], LOADS_EXPECTED_CLIP) / LOADS_EXPECTED_CLIP
    out_loads = np.where(hay, np.where(F, cargas, 0.0).sum(axis=2) / np.maximum(n_fact, 1), 0.0)
    out_fact  = n_fact / max(N, 1)
    m_j = np.clip(np.log(np.maximum(est.M[dow_j, jj[None, :]], 1.0)) / np.log(max(COMPETENCIA_M_MAX, 1.0 + 1e-9)),
                  0.0, 1.0)

    # j == h es volver a casa: la lane de regreso no existe y el ciclo termina (sin salientes). La espera en casa antes de
    # salir (j == i == h) sí conserva las salientes de mañana. La holgura [7] se conserva en ambos casos.
    termina = is_h & ~is_i
    ret_margin[is_h] = 0.0
    ret_loads[is_h] = 0.0
    for x in (out_best, out_loads, out_fact, m_j):
        x[termina] = 0.0

    return np.stack([home_time, slack, ret_margin, ret_loads, out_best, out_loads, out_fact, m_j],
                    axis=-1).astype(np.float32)


def _check_node_features(feats: np.ndarray, B: int, N: int) -> None:
    assert feats.shape == (B, N, N_NODE_FEATURES), f"features de nodo con forma {feats.shape}"
    assert np.isfinite(feats).all(), "features de nodo con NaN/Inf"


# ─────────────────────────────────────────────────────────────────────────────
# Utilidad: construcción de la matriz de features de nodos
# ─────────────────────────────────────────────────────────────────────────────

def build_node_features(
    current_node:           int,
    start_node:             int,
    reward_matrix_penalized: np.ndarray,  # (N×N)
    time_matrix:             np.ndarray,  # (N×N)
    distance_arr:            np.ndarray,  # (N×N)
    num_nodes:               int,
    max_duration:            float,
    reward_global_p95:       float      = 1.0,
    avail_expected_arr:      np.ndarray = None,  # (N, N) — E[loads] histórico (train), usado en feature [4]
    est                                 = None,  # EstadisticasLane (MarketProcess.estadisticas_dow) — features [6-13]
    time_elapsed:            float      = 0.0,   # horas transcurridas — features [6-13]
    dow0:                    int        = 0,     # día de la semana (lunes = 0) de salida — features [6-13]
    market_index:            np.ndarray = None,  # (N,) índice de mercado de todos los nodos hoy — última feature
    index_std:               float      = None,  # desviación del índice histórico (normaliza market_index)
) -> np.ndarray:
    """
    Construye la matriz de features de nodos para el encoder (vectorizada).

    Cada fila j contiene las features del nodo j vistas desde current_node:
      [reward_norm, time_norm, dist_norm, is_depot,
       avail_prior, reward_abs] (+ 8 de continuación si USE_CONTINUATION_FEATURES; requieren
       est, time_elapsed y dow0) + market_index / index_std (después de las de continuación)

    La diagonal de reward_matrix_penalized tiene BIG_M_PENALTY (-1e9).
    Se recorta a [-1e4, 1e4] antes de normalizar para evitar desbordamientos.

    Retorna np.ndarray de forma (N, N_NODE_FEATURES), dtype float32.
    """
    feats = np.zeros((num_nodes, N_NODE_FEATURES), dtype=np.float32)
    if USE_CONTINUATION_FEATURES and est is None:
        raise ValueError("USE_CONTINUATION_FEATURES requiere est (MarketProcess.estadisticas_dow)")
    if market_index is None or index_std is None:
        raise ValueError("build_node_features requiere market_index e index_std (índice de mercado)")

    # Columns 0-2: reward / time / distance, all from current_node row
    feats[:, 0] = reward_matrix_penalized[current_node]
    feats[:, 1] = time_matrix[current_node]
    feats[:, 2] = distance_arr[current_node]

    # Column 3: is_depot
    feats[start_node, 3] = 1.0

    # Column 4: avail_prior — E[loads] histórico (train), no P(existe).
    if avail_expected_arr is not None:
        feats[:, 4] = np.minimum(avail_expected_arr[current_node], LOADS_EXPECTED_CLIP) / LOADS_EXPECTED_CLIP

    # Recortar reward para evitar BIG_M_PENALTY (-1e9) en la diagonal
    feats[:, 0] = np.clip(feats[:, 0], -1e4, 1e4)
    # Copia antes de normalizar: preserva la escala absoluta para feature [5]
    raw_rewards = feats[:, 0].copy()

    # reward y distancia: normalización relativa intra-fila para preservar ranking
    # local entre las alternativas disponibles desde current_node.
    for col in (0, 2):
        col_min = feats[:, col].min()
        col_max = feats[:, col].max()
        if col_max > col_min:
            feats[:, col] = (feats[:, col] - col_min) / (col_max - col_min)
        else:
            feats[:, col] = 0.0

    # tiempo: normalización absoluta respecto al presupuesto global, para preservar
    # información de factibilidad temporal (se aplica una sola vez).
    feats[:, 1] = np.clip(feats[:, 1] / max(max_duration, 1e-6), 0.0, 1.0)

    # Feature [5]: reward absoluto normalizado por el P95 global del entrenamiento.
    # Complementa la feature [0] (ranking local) con información de escala absoluta.
    # Valores negativos (arcos con costo neto) quedan en 0.0.
    _denom = max(reward_global_p95, 1e-6)
    feat5 = np.clip(raw_rewards / _denom, 0.0, 1.0)
    feat5[raw_rewards < 0] = 0.0
    feats[:, 5] = feat5

    # Columnas 6-13: continuación (misma función que la versión batcheada, con B = 1)
    if USE_CONTINUATION_FEATURES:
        feats[:, N_BASE_NODE_FEATURES:N_BASE_NODE_FEATURES + N_CONT_FEATURES] = build_continuation_features_batch(
            np.array([current_node]), np.array([start_node]),
            np.array([time_elapsed]), np.array([dow0]), time_matrix, est,
            max_duration, reward_global_p95,
        )[0]

    # Índice de mercado normalizado por la desviación del índice histórico (las de cotización quedan en cero)
    feats[:, INDEX_COL] = np.asarray(market_index, dtype=np.float64) / max(float(index_std), 1e-12)

    if np.isnan(feats).any():
        bad = np.argwhere(np.isnan(feats)).tolist()
        raise RuntimeError(
            f"build_node_features: NaN in feature matrix "
            f"(current_node={current_node}, positions={bad})"
        )
    if np.isinf(feats).any():
        bad = np.argwhere(np.isinf(feats)).tolist()
        raise RuntimeError(
            f"build_node_features: Inf in feature matrix "
            f"(current_node={current_node}, positions={bad})"
        )

    return feats


# ─────────────────────────────────────────────────────────────────────────────
# Bloques del Transformer Encoder
# ─────────────────────────────────────────────────────────────────────────────

class NodeFeatureProjection(nn.Module):
    """
    Proyección lineal de las features crudas de nodos a la dimensión de embedding.

    Entrada : (batch, N, d_input)
    Salida  : (batch, N, d_h)
    """

    def __init__(self, d_input: int, d_h: int):
        super().__init__()
        self.proj = nn.Linear(d_input, d_h)
        self.norm = nn.LayerNorm(d_h)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(F.relu(self.proj(x)))


class MultiHeadSelfAttention(nn.Module):
    """
    Atención multi-cabeza estándar (scaled dot-product).

    Entrada : (batch, N, d_h)
    Salida  : (batch, N, d_h)
    """

    def __init__(self, d_h: int, n_heads: int):
        super().__init__()
        assert d_h % n_heads == 0, "d_h debe ser divisible por n_heads"
        self.n_heads = n_heads
        self.d_k = d_h // n_heads

        self.W_q = nn.Linear(d_h, d_h, bias=False)
        self.W_k = nn.Linear(d_h, d_h, bias=False)
        self.W_v = nn.Linear(d_h, d_h, bias=False)
        self.W_o = nn.Linear(d_h, d_h, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, _ = x.shape

        def split_heads(t):
            return t.view(B, N, self.n_heads, self.d_k).transpose(1, 2)

        Q = split_heads(self.W_q(x))   # (B, H, N, d_k)
        K = split_heads(self.W_k(x))
        V = split_heads(self.W_v(x))

        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_k ** 0.5)
        attn   = F.softmax(scores, dim=-1)

        out = torch.matmul(attn, V)                         # (B, H, N, d_k)
        out = out.transpose(1, 2).contiguous().view(B, N, -1)  # (B, N, d_h)
        return self.W_o(out)


class TransformerEncoderLayer(nn.Module):
    """
    Una capa del encoder Transformer: MHA + FFN con residual y LayerNorm.

    Entrada : (batch, N, d_h)
    Salida  : (batch, N, d_h)
    """

    def __init__(self, d_h: int, n_heads: int, d_ff: int, pre_norm: bool = True, dropout: float = 0.0):
        super().__init__()
        self.pre_norm = pre_norm
        self.mha   = MultiHeadSelfAttention(d_h, n_heads)
        self.norm1 = nn.LayerNorm(d_h)
        self.ffn   = nn.Sequential(
            nn.Linear(d_h, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_h),
        )
        self.norm2 = nn.LayerNorm(d_h)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.pre_norm:
            x = x + self.dropout(self.mha(self.norm1(x)))
            x = x + self.dropout(self.ffn(self.norm2(x)))
        else:
            x = self.norm1(x + self.mha(x))
            x = self.norm2(x + self.ffn(x))
        return x


class AttentionEncoder(nn.Module):
    """
    Encoder Transformer: mapea N vectores de features a N embeddings ricos.

    Parámetros
    ----------
    d_input  : int — features por nodo (N_NODE_FEATURES: 8, o 12 con USE_CONTINUATION_FEATURES).
    d_h      : int — dimensión del embedding (ej. 128).
    n_heads  : int — cabezas de atención (ej. 8, debe dividir d_h).
    n_layers : int — capas del encoder (ej. 3).
    d_ff     : int — dimensión interna del FFN (ej. 512 = 4 × d_h).

    Forward
    -------
    node_features : Tensor (batch, N, d_input)

    Retorna
    -------
    embeddings      : Tensor (batch, N, d_h)  — embedding por nodo.
    graph_embedding : Tensor (batch, d_h)     — promedio de todos los embeddings.
    """

    def __init__(
        self,
        d_input:  int = N_NODE_FEATURES,
        d_h:      int = 128,
        n_heads:  int = 8,
        n_layers: int = 3,
        d_ff:     int = 512,
        pre_norm: bool = True,
        dropout:  float = 0.0,
    ):
        super().__init__()
        self.pre_norm = pre_norm
        self.input_proj = NodeFeatureProjection(d_input, d_h)
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_h, n_heads, d_ff, pre_norm=pre_norm, dropout=dropout)
            for _ in range(n_layers)
        ])
        self.norm_out = nn.LayerNorm(d_h)

    def forward(self, node_features: torch.Tensor):
        x = self.input_proj(node_features)   # (batch, N, d_h)
        for layer in self.layers:
            x = layer(x)                     # (batch, N, d_h)
        if self.pre_norm:
            x = self.norm_out(x)
        graph_emb = x.mean(dim=1)            # (batch, d_h)
        return x, graph_emb


# ─────────────────────────────────────────────────────────────────────────────
# Vector de contexto h_t
# ─────────────────────────────────────────────────────────────────────────────

class ContextNetwork(nn.Module):
    """
    Construye el vector de contexto h_t usado por el decoder (destino y precio) y el critic.

    Combina:
      · graph_embedding   (d_h)       — estado global del grafo
      · current_node_emb  (d_h)       — posición actual del camión
      · temporal_features (3)         — tiempo_norm, pasos_restantes_norm,
                                        progreso_norm
      · market_features   (n_market)  — MTI, dispersión, forecast (opcional,
                                        por defecto 0 = no se usa)
      · dow_features      (n_dow)     — día de la semana actual (one-hot), último bloque
      · budget_features   (n_budget)  — presupuesto de cardinalidad K:
                                        k_remaining_norm, k_restricted.
                                        Bloque SEPARADO de temporal_features
                                        (conceptualmente distinto: K acota el
                                        número de arcos, no el tiempo) — no se
                                        amplía n_temporal para incorporarlo.

      · cot_features      (n_cot)     — cotización conjunta: competencia del nodo actual y rechazos en esta
                                        visita (build_cotizacion_context_batch). Último bloque, en cero fuera
                                        de ese modo, para que los checkpoints anteriores carguen.

    Entrada total: 2·d_h + 3 + n_market + n_budget + n_dow + n_cot  →  MLP  →  d_h

    Parámetros
    ----------
    d_h      : int — dimensión del embedding (debe coincidir con AttentionEncoder).
    n_market : int — número de features de mercado adicionales (default 1).
    n_budget : int — número de features de presupuesto K (default 2, ver
               build_budget_features_batch()).
    n_dow    : int — one-hot del día de la semana (default N_DOW_FEATURES = 7).
    n_cot    : int — features de contexto de la cotización (default N_COT_CONTEXT = 2).
    """

    def __init__(self, d_h: int, n_market: int = 1, n_budget: int = 2, n_dow: int = N_DOW_FEATURES,
                 n_cot: int = N_COT_CONTEXT):
        super().__init__()
        self.n_dow = n_dow
        self.n_cot = n_cot
        n_temporal = 3          # time_norm, remaining_steps_norm, progress_norm
        d_in = 2 * d_h + n_temporal + n_market + n_budget + n_dow + n_cot
        self.proj = nn.Sequential(
            nn.Linear(d_in, d_h),
            nn.ReLU(),
            nn.Linear(d_h, d_h),
            nn.LayerNorm(d_h),
        )

    def forward(
        self,
        graph_emb:        torch.Tensor,   # (batch, d_h)
        current_node_emb: torch.Tensor,   # (batch, d_h)
        temporal_feats:   torch.Tensor,   # (batch, 3)
        market_feats:     torch.Tensor = None,  # (batch, n_market) | None
        budget_feats:     torch.Tensor = None,  # (batch, n_budget) | None
        dow_feats:        torch.Tensor = None,  # (batch, n_dow) | None (None ⇒ ceros)
        cot_feats:        torch.Tensor = None,  # (batch, n_cot) | None (None ⇒ ceros)
    ) -> torch.Tensor:
        """
        Retorna el vector de contexto h_t de forma (batch, d_h).
        """
        parts = [graph_emb, current_node_emb, temporal_feats]
        if market_feats is not None:
            parts.append(market_feats)
        if budget_feats is not None:
            parts.append(budget_feats)
        if self.n_dow:
            parts.append(dow_feats if dow_feats is not None
                         else torch.zeros(graph_emb.shape[0], self.n_dow, device=graph_emb.device))
        if self.n_cot:
            parts.append(cot_feats if cot_feats is not None
                         else torch.zeros(graph_emb.shape[0], self.n_cot, device=graph_emb.device))
        h = torch.cat(parts, dim=-1)
        return self.proj(h)


# ─────────────────────────────────────────────────────────────────────────────
# Utilidad: features de presupuesto de cardinalidad K para ContextNetwork
# ─────────────────────────────────────────────────────────────────────────────

def build_budget_features_batch(
    k_remaining:  np.ndarray,   # (B,) int64 — indefinido donde k_restricted es False
    k_restricted: np.ndarray,   # (B,) bool
    k_norm:       float = K_NORM,
) -> np.ndarray:                # (B, 2) float32
    """k_remaining_norm (1.0 si caso libre) y flag k_restricted, para ContextNetwork."""
    restricted = np.asarray(k_restricted, dtype=bool)
    B = len(restricted)
    out = np.empty((B, 2), dtype=np.float32)
    out[:, 0] = np.where(
        restricted, np.asarray(k_remaining, dtype=np.float32) / float(k_norm), 1.0
    )
    out[:, 1] = restricted.astype(np.float32)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Utilidad: features temporales desde el estado del entorno
# ─────────────────────────────────────────────────────────────────────────────

def build_node_features_batch(
    current_node:      np.ndarray,   # (B,) int64
    start_node:        np.ndarray,   # (B,) int64
    time_matrix_arr:   np.ndarray,   # (N, N)
    distance_arr:      np.ndarray,   # (N, N)
    day_idx:           np.ndarray,   # (B,) int64
    num_nodes:         int,
    max_duration:      float,
    reward_global_p95: float      = 1.0,
    avail_expected_arr: np.ndarray = None,  # (N, N) — E[loads] histórico (train), usado en feature [4]
    reward_rows:       np.ndarray = None,   # (B, N) — env.reward_rows: margen de las lanes salientes a la
                                            # mediana del mercado (obligatorio)
    est                        = None,      # EstadisticasLane (MarketProcess.estadisticas_dow) — features de continuación [6-13]
    time_elapsed:      np.ndarray = None,   # (B,) horas transcurridas — idem
    dow0:              np.ndarray = None,   # (B,) día de la semana (lunes = 0) de salida — idem
    market_index:      np.ndarray = None,   # (B, N) índice de mercado de todos los nodos en el día actual
                                            # (env.market_index) — última feature
    index_std:         float      = None,   # desviación del índice histórico (env.index_std)
    cotizacion_feats:  np.ndarray = None,   # (B, N, N_COT_FEATURES) — build_cotizacion_node_features_batch;
                                            # None ⇒ esas columnas en cero (solo tests de features)
) -> np.ndarray:                            # (B, N, N_NODE_FEATURES) float32
    """
    Batched version of build_node_features.

    Computes the (N, N_NODE_FEATURES) feature matrix for each of B episodes simultaneously
    using numpy fancy indexing — no Python loops over N or B.
    """
    B = len(current_node)
    N = num_nodes
    if USE_CONTINUATION_FEATURES and (est is None or time_elapsed is None or dow0 is None):
        raise ValueError("USE_CONTINUATION_FEATURES requiere est, time_elapsed y dow0")
    if market_index is None or index_std is None:
        raise ValueError("build_node_features_batch requiere market_index e index_std (índice de mercado)")
    if reward_rows is None:
        raise ValueError("build_node_features_batch requiere reward_rows (env.reward_rows)")
    feats = np.zeros((B, N, N_NODE_FEATURES), dtype=np.float32)

    # ── f0: margen de cada lane saliente a la mediana del mercado (env.reward_rows) ──
    feats[:, :, 0] = reward_rows

    # ── f1: travel time from current node to each j ───────────────────────────
    feats[:, :, 1] = time_matrix_arr[current_node, :]          # (B, N)

    # ── f2: distance from current node to each j ─────────────────────────────
    feats[:, :, 2] = distance_arr[current_node, :]             # (B, N)

    # ── f3: is_depot — 1.0 if j == start_node[b] ─────────────────────────────
    feats[:, :, 3] = (np.arange(N)[None, :] == start_node[:, None]).astype(np.float32)

    # ── f4: lane availability prior — E[loads] histórico (train), no P(existe) ─
    if avail_expected_arr is not None:
        feats[:, :, 4] = (
            np.minimum(avail_expected_arr[current_node, :], LOADS_EXPECTED_CLIP)
            / LOADS_EXPECTED_CLIP
        )       # (B, N)

    # ── Clip reward before normalization (avoids BIG_M_PENALTY overflow) ─────
    feats[:, :, 0] = np.clip(feats[:, :, 0], -1e4, 1e4)
    raw_rewards = feats[:, :, 0].copy()   # (B, N) — before normalization for f5

    # ── Row-wise min-max normalization for f0 (reward) and f2 (distance) ─────
    for col in (0, 2):
        col_min = feats[:, :, col].min(axis=1, keepdims=True)   # (B, 1)
        col_max = feats[:, :, col].max(axis=1, keepdims=True)   # (B, 1)
        rng = col_max - col_min
        safe = (rng > 0).squeeze(axis=1)                         # (B,)
        feats[safe, :, col] = (
            (feats[safe, :, col] - col_min[safe]) / rng[safe]
        )
        feats[~safe, :, col] = 0.0

    # ── Fila propia (j = i) = la espera: f1 son las horas hasta el siguiente día de mercado y f2 = 0 (la
    # diagonal de distancias trae una distancia intra-hub, pero esperar no recorre nada) ──
    feats[np.arange(B), current_node, 2] = 0.0
    if time_elapsed is not None:
        t = np.asarray(time_elapsed, dtype=np.float64)
        feats[np.arange(B), current_node, 1] = fin_espera(t) - t

    # ── Absolute normalization for f1 (time) ─────────────────────────────────
    feats[:, :, 1] = np.clip(feats[:, :, 1] / max(max_duration, 1e-6), 0.0, 1.0)

    # ── f5: absolute reward normalized by global P95 ──────────────────────────
    _denom = max(reward_global_p95, 1e-6)
    feat5 = np.clip(raw_rewards / _denom, 0.0, 1.0)
    feat5[raw_rewards < 0] = 0.0
    feats[:, :, 5] = feat5

    # ── f6–f13: continuación (tiempo/holgura de regreso y estadísticas de las lanes que salen de j) ──
    if USE_CONTINUATION_FEATURES:
        feats[:, :, N_BASE_NODE_FEATURES:N_BASE_NODE_FEATURES + N_CONT_FEATURES] = build_continuation_features_batch(
            current_node, start_node, time_elapsed, dow0,
            time_matrix_arr, est, max_duration, reward_global_p95,
        )

    # ── índice de mercado / desviación del índice histórico ──
    feats[:, :, INDEX_COL] = np.asarray(market_index, dtype=np.float64) / max(float(index_std), 1e-12)

    # ── cotización conjunta (últimas columnas; en cero fuera de ese modo) ──
    if cotizacion_feats is not None:
        feats[:, :, COT_COLS] = cotizacion_feats

    _check_node_features(feats, B, N)
    return feats


def build_cotizacion_node_features_batch(
    current_node: np.ndarray,   # (B,) int64
    precios:      np.ndarray,   # (B, N, K) USD/milla de cada nivel (env.precios_row)
    opciones:     np.ndarray,   # (B, N, K+1) bool (env.opciones_validas)
    dist_rows:    np.ndarray,   # (B, N) distancia i→j
    cost_rows:    np.ndarray,   # (B, N) costo i→j (pricing.lane_cost_row)
    m:            np.ndarray,   # (B,) competidores efectivos del nodo actual (env.competencia)
    intentos:     np.ndarray,   # (B, N) cargas cotizables que quedan por lane (env.intentos)
    reward_global_p95: float,
) -> np.ndarray:                # (B, N, N_COT_FEATURES) float32
    """Features de cotización por lane i→j (ver docstring del módulo): dispersión del precio, break-even
    relativo a la mediana, si se puede cotizar y, por nivel, el valor esperado de cotizarlo y P(ganar). El
    self-loop (j = i) queda en cero."""
    precios = np.asarray(precios, dtype=np.float64)
    B, N = precios.shape[:2]
    p_med = precios[..., _U_50]
    ok = p_med > 1e-9
    seguro = np.where(ok, p_med, 1.0)
    dispersion = np.where(ok, (precios[..., _U_80] - precios[..., _U_20]) / seguro, 0.0)
    break_even = np.where(ok, np.asarray(cost_rows, dtype=np.float64)
                          / (np.maximum(np.asarray(dist_rows, dtype=np.float64), 1e-9) * seguro), 2.0)
    cotizable = np.asarray(opciones)[..., :N_NIVELES].any(axis=-1)                         # (B, N)
    d = np.asarray(dist_rows, dtype=np.float64)[..., None]                                # (B, N, 1)
    c = np.asarray(cost_rows, dtype=np.float64)[..., None]
    P = prob_ganar(U_NIVELES_ARR[None, :], np.asarray(m, dtype=np.float64)[:, None])[:, None, :]      # (B, 1, K)
    p95 = max(float(reward_global_p95), 1e-6)
    v = P * (precios * d - c) / p95                                                       # (B, N, K)
    A = COTIZACIONES_MAX_POR_NODO
    n = np.clip(np.asarray(intentos, dtype=np.int64), 0, A)[None]                         # (1, B, N)
    abierto, cierre = (np.take_along_axis(valor_reintentos(P, precios * d - c, base, A)[0], n, axis=0)[0][..., None] / p95
                       for base in (np.zeros_like(c[..., 0]), -c[..., 0]))                # (B, N, 1)
    por_nivel = np.concatenate([
        np.clip(v, -1.0, 1.0), np.broadcast_to(P, v.shape),
        np.clip(abierto, -1.0, 1.0), np.clip(cierre, -1.0, 1.0),
    ], axis=-1)
    out = np.concatenate([
        np.stack([np.clip(dispersion, 0.0, 1.0), np.clip(break_even, 0.0, 2.0) / 2.0,
                  np.where(cotizable, n[0] / A, 0.0)], axis=-1),
        np.where(cotizable[..., None], por_nivel, 0.0),
    ], axis=-1).astype(np.float32)
    out[np.arange(B), np.asarray(current_node, dtype=np.int64)] = 0.0
    return out


def build_cotizacion_context_batch(m: np.ndarray, n_rechazos: np.ndarray) -> np.ndarray:   # (B,), (B,) → (B, 2)
    """log(m) / log(M_MAX) (competencia en el nodo actual, en [0, 1]) y rechazos / COTIZACIONES_MAX_POR_NODO."""
    m = np.asarray(m, dtype=np.float64)
    out = np.empty((len(m), N_COT_CONTEXT), dtype=np.float32)
    out[:, 0] = np.clip(np.log(np.maximum(m, 1.0)) / np.log(max(COMPETENCIA_M_MAX, 1.0 + 1e-9)), 0.0, 1.0)
    out[:, 1] = np.asarray(n_rechazos, dtype=np.float64) / max(COTIZACIONES_MAX_POR_NODO, 1)
    return out


def build_dow_features_batch(dow: np.ndarray) -> np.ndarray:   # (B,) int → (B, 7) float32
    """One-hot del día de la semana (lunes = 0) de cada episodio."""
    dow = np.asarray(dow, dtype=np.int64)
    out = np.zeros((len(dow), N_DOW_FEATURES), dtype=np.float32)
    out[np.arange(len(dow)), dow % N_DOW_FEATURES] = 1.0
    return out


def build_temporal_features_batch(
    time_elapsed: np.ndarray,   # (B,) float32
    step_count:   np.ndarray,   # (B,) int64
    max_duration: float,
    num_nodes:    int,
) -> np.ndarray:                # (B, 3) float32
    """Features temporales (elapsed/remaining/progreso) para ContextNetwork, batcheadas."""
    denom = max(max_duration, 1e-6)
    out = np.empty((len(time_elapsed), 3), dtype=np.float32)
    out[:, 0] = np.minimum(time_elapsed, max_duration) / denom
    out[:, 1] = np.maximum(0.0, max_duration - time_elapsed) / denom
    out[:, 2] = step_count / max(num_nodes, 1)
    return out


def build_market_features_batch(
    current_node: np.ndarray,   # (B,) int64
    day_idx:      np.ndarray,   # (B,) int64
    ltr_stack:    np.ndarray,   # (N, num_ltr_days)
) -> np.ndarray:                # (B, 1) float32
    """LTR del hub actual en el día actual, normalizado a [0,1], para ContextNetwork (batcheado)."""
    max_ltr_day = ltr_stack.shape[1] - 1
    day_idx_safe = np.minimum(day_idx, max_ltr_day)
    ltr_raw = ltr_stack[current_node, day_idx_safe].astype(np.float32)   # (B,)
    ltr_norm = np.clip(ltr_raw, 0.0, LTR_CLIP) / LTR_CLIP
    return ltr_norm[:, None]   # (B, 1)


