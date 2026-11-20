"""
validate_pricing.py
===================
Validación del precio de reserva p* sobre la ruta del DRL ya entrenado (PRICING_PLAN.md §4). Necesita un
checkpoint con las cabezas Ṽ A/B entrenadas (post_training.py). Todo en K libre salvo que K_TRAINING_LEVELS
diga otra cosa. El DRL es el único método de ruteo: aquí no se re-decide ninguna acción con Q, solo se calcula
el precio de reserva del arco que el actor ya elige.

  (b) Prueba contrafactual. Se muestrean N_EST_CONTRAFACTUAL estados de rutas del DRL (actor) en mundos de
      validación (MARKET_SEED_VAL, distinta de entrenamiento, de Ṽ y de evaluación). Desde el estado, en el
      MISMO mundo, se sigue la lane j que eligió el actor y, por separado, la mejor alternativa según el modelo
      (la de A para p*_AB y la de B para p*_BA); después de cada una, el actor greedy hasta casa:
          p*_realizado = ( G_alt − C_j + costo_j ) / dist_j
      con G_alt el margen total desde el estado tomando la alternativa (incluida su carga) y C_j el margen
      posterior a llegar a j. Sesgo = p* predicho − p*_realizado, con IC95 por bootstrap que remuestrea MUNDOS
      (los estados de un mismo mundo no son independientes).
  (c) Utilidad, en los mundos simulados de evaluación y en replay: % de arcos del DRL con p* bajo el break-even
      y sobre break-even + BE_MARGEN; y la regla de aceptación por umbral con tres pisos: break-even,
      break-even + BE_MARGEN y p*. (El margen del DRL frente a los baselines es ruteo, no pricing: se reporta
      en evaluation.py/main.py, que ya corren esa comparación.)
  (d) Plausibilidad: percentil de p* en las tarifas históricas de la lane (días de entrenamiento con carga) y
      correlación de (p* − break-even) con el índice de mercado del destino (esperada negativa: un destino con
      índice alto vale más ⇒ Ṽ(j) mayor ⇒ p* menor) y con el LTR del nodo ACTUAL i, no del destino (esperada
      positiva: mercado tensionado en i ⇒ más poder de negociación del transportista sobre las lanes que salen
      de i ⇒ tarifas observadas r_obs(i→·) más altas en todas las alternativas ⇒ Q_alt sube sin que suba Ṽ(j)
      (que es el valor de la continuación desde j, ajeno a la economía de salida de i) ⇒ p* sube. A diferencia
      del índice del destino, el LTR de i NO se agrega como feature al modelo: ya está implícito en r_obs
      (pricing.py), así que esta correlación valida el p* existente, no requiere tocar Ṽ ni reentrenar nada
      — alimentar el LTR de i a Ṽ(s, j) de forma simétrica entre candidatos se cancelaría casi exactamente en
      Q_alt − Ṽ(j), que por construcción solo es sensible a diferencias ENTRE alternativas del mismo estado.
  (a) Las pruebas internas (Q(s, j) = Q_alt a tarifa p*, monotonía en el costo, invariancia en S) están en tests/.

p* es un costo de oportunidad del transportista, no una probabilidad de aceptación. Su rango [mín, máx] entre
las parejas AB/BA mide la variabilidad del estimador, NO es un intervalo de confianza.

Salidas: <output_dir>/pricing_val/figures/*.png y <output_dir>/pricing_val/tables/*.csv
(output_dir por defecto results/<N>nodes[_<RUN_TAG>]).

Uso
---
    python validate_pricing.py --num-nodes 10 [--checkpoint RUTA] [--quick]
"""

import argparse
import os
import time

import numpy as np
import pandas as pd
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

from config import (
    DEVICE, MAX_DURATION, REWARD_SCALE_FACTOR as S, TRAIN_DAYS, MARKET_SEED_VAL, MARKET_SEED_EVAL,
    PRICING_VAL_SEED, K_TRAINING_LEVELS, configurar_k, RUN_TAG, AM_D_H, AM_N_HEADS, AM_N_LAYERS, AM_D_FF,
    N_EST_CONTRAFACTUAL, N_MUNDOS_CONTRAFACTUAL, BE_MARGEN, INDICE_DISPERSION_SEMANAL, MARKET_M_EVAL_WORLDS,
    BOOTSTRAP_B, BOOTSTRAP_SEED,
)
from model.encoder import (
    build_node_features_batch, build_temporal_features_batch, build_market_features_batch,
    build_budget_features_batch, build_dow_features_batch,
)
from baselines.politicas import calcular_ev_pronostico
from evaluation import disenar_mundos, make_sim_env, make_replay_env, casos_replay, _etiqueta_k
from market_process import MarketProcess, asegurar_parametros
from pricing import (
    lane_cost_row, percentil_historico, reservation_price, PrecioReserva,
)
from problem_data import load_matrices
from stats_analysis import ic_bootstrap_media, ic_bootstrap_cluster

TRAMOS = [("< 300 mi", 0.0, 300.0), ("300-800 mi", 300.0, 800.0), ("> 800 mi", 800.0, np.inf)]


def _tramo(dist) -> np.ndarray:
    d = np.asarray(dist, dtype=float)
    return np.select([d < 300.0, d <= 800.0], [TRAMOS[0][0], TRAMOS[1][0]], TRAMOS[2][0])


# ─────────────────────────────────────────────────────────────────────────────
# Escenario: datos + red, con el forward compartido por todos los rollouts por lotes
# ─────────────────────────────────────────────────────────────────────────────

class Escenario:
    """Datos del problema y rollouts por lotes sobre mundos simulados o sobre el replay de los días de evaluación.

    modo="simulado": rate_stack/loads_stack = días de entrenamiento (fijan el último día con datos del entorno).
    modo="replay"  : rate_stack/loads_stack = días de evaluación (TRAIN_DAYS en adelante) y ltr recortado
                     a los mismos días; world_seed no existe.
    rate_hist/loads_hist son siempre la serie de entrenamiento (para el percentil histórico de p*).
    """

    def __init__(self, agent, num_nodes, time_matrix, rate_stack, loads_stack, distance_arr, diesel_arr,
                 ltr_stack, avail_expected_arr, reward_global_p95, market, ev_esperado=None,
                 modo="simulado", rate_hist=None, loads_hist=None):
        self.agent, self.N, self.market, self.modo = agent.eval(), num_nodes, market, modo
        self.T = np.asarray(time_matrix, dtype=float)
        self.dist, self.diesel = np.asarray(distance_arr, dtype=float), diesel_arr
        self.cost = lane_cost_row(self.dist, np.asarray(diesel_arr, dtype=float))             # (N, N)
        self.rate_stack, self.loads_stack = rate_stack, loads_stack
        self.rate_hist = rate_stack[:TRAIN_DAYS] if rate_hist is None else rate_hist
        self.loads_hist = loads_stack[:TRAIN_DAYS] if loads_hist is None else loads_hist
        self.time_matrix = time_matrix
        self.ltr, self.avail, self.p95 = ltr_stack, avail_expected_arr, reward_global_p95
        self.ev = ev_esperado if ev_esperado is not None else calcular_ev_pronostico(market, distance_arr, diesel_arr)
        self.device = next(agent.parameters()).device
        self.con_cabezas = bool(getattr(agent, "post_entrenadas", False))

    def make_env(self):
        if self.modo == "replay":
            return make_replay_env(self.time_matrix, self.rate_stack, self.loads_stack, self.dist, self.diesel,
                                   self.N, day_offset=TRAIN_DAYS, market=self.market)
        return make_sim_env(self.time_matrix, self.rate_stack, self.loads_stack, self.dist, self.diesel,
                            self.N, self.market)

    def encode(self, env, masks):
        """(h_t, embeddings, mask_t) del estado actual de `env`; mismas features que greedy_rollout."""
        B, cur, dev = len(env.current_node), env.current_node, self.device
        nf = build_node_features_batch(
            cur, env.start_node, None, self.T, self.dist, env.current_day_idx,
            self.N, MAX_DURATION, reward_global_p95=self.p95,
            avail_expected_arr=self.avail, reward_rows=env.reward_rows, ev=self.ev,
            time_elapsed=env.time_elapsed, dow0=env.start_dow,
            market_index=env.market_index, index_std=env.index_std,
        )
        tmp = build_temporal_features_batch(env.time_elapsed, env.step_count, MAX_DURATION, self.N)
        mkt = build_market_features_batch(cur, env.current_day_idx, self.ltr) if self.ltr is not None \
            else np.zeros((B, 1), dtype=np.float32)
        bud = build_budget_features_batch(env.k_remaining, env.k_restricted)
        dow = build_dow_features_batch(env.current_dow)
        t = lambda a: torch.from_numpy(a).to(dev)
        emb, graph = self.agent.encoder(t(nf))
        cur_t = torch.tensor(cur, dtype=torch.long, device=dev)
        h_t = self.agent.context_net(graph, emb[torch.arange(B), cur_t, :], t(tmp), t(mkt), t(bud), t(dow))
        return h_t, emb, t(masks)

    def valores(self, h_t, emb) -> tuple:
        """(Ṽ_A, Ṽ_B), cada una (B, N) float64 en unidades escaladas por 1/S (sin el ancla Ṽ(depot) = 0)."""
        va, vb = self.agent.post_values(h_t, emb)
        return va.detach().cpu().numpy().astype(np.float64), vb.detach().cpu().numpy().astype(np.float64)

    def acciones(self, h_t, emb, mask_t, modo: str) -> np.ndarray:
        """modo: "greedy" (argmax del actor) o "sample" (muestreada del actor)."""
        if modo == "sample":
            return self.agent.decoder.act(h_t, emb, mask_t)[0].cpu().numpy()
        return self.agent.decoder.greedy_action(h_t, emb, mask_t).cpu().numpy()

    def precio(self, rec, j) -> PrecioReserva:
        """Precio de reserva de la lane rec['i'] → j en el estado del registro."""
        i = rec["i"]
        return reservation_price(rec["va"], rec["vb"], rec["mask"], i, j, rec["rate"], self.dist[i], self.cost[i],
                                 rec["start"], S, loads_row=rec["loads"])

    # ── (1) trayectorias de una política con todo lo que p* necesita ─────────
    @torch.no_grad()
    def recolectar(self, dis, k_arr, modo="greedy", con_indice=False, con_valores=None) -> list:
        """Rollout de la política `modo` sobre los mundos `dis`; un registro por (episodio, paso).

        Cada registro guarda lo observado en el estado s (máscara, tablero, Ṽ de las cabezas, índice de mercado
        si con_indice) y la acción tomada. `futuro_usd` es lo que se cobra DESPUÉS de ese paso en el episodio.
        """
        con_valores = self.con_cabezas if con_valores is None else con_valores
        env = self.make_env()
        B = len(dis["nodo"])
        masks = env.reset(dis["nodo"], dis["start_day"], k_max_arr=k_arr, world_seeds=dis.get("world_seed"))
        active = np.ones(B, dtype=bool)
        recs = []
        for _ in range(env.max_steps):
            if not active.any():
                break
            h_t, emb, mask_t = self.encode(env, masks)
            va, vb = self.valores(h_t, emb) if con_valores else (None, None)
            actions = self.acciones(h_t, emb, mask_t, modo)
            rate, loads = env.rate_row_obs, env.loads_row_obs
            idx = env.market_index if con_indice else None
            ltr_i = None                                        # LTR del nodo ACTUAL (no del destino): no entra al
            if con_indice and self.ltr is not None:               # modelo, solo se usa para validar el p* ya calculado
                dia_ltr = np.minimum(env.current_day_idx, self.ltr.shape[1] - 1)
                ltr_i = self.ltr[env.current_node, dia_ltr]
            paso = []
            for b in np.where(active)[0]:
                paso.append(len(recs))
                recs.append(dict(
                    b=int(b), step=int(env.step_count[b]), i=int(env.current_node[b]), j=int(actions[b]),
                    start=int(env.start_node[b]), start_day=int(dis["start_day"][b]), k=k_arr[b],
                    t=float(env.time_elapsed[b]), mask=masks[b].copy(), rate=np.array(rate[b], dtype=np.float64),
                    loads=np.array(loads[b]), k_rem=int(env.k_remaining[b]), dow=int(env.current_dow[b]),
                    world_seed=None if dis.get("world_seed") is None else int(dis["world_seed"][b]),
                    va=None if va is None else va[b].copy(), vb=None if vb is None else vb[b].copy(),
                    idx=None if idx is None else np.array(idx[b], dtype=np.float64),
                    ltr_i=None if ltr_i is None else float(ltr_i[b]),
                    term=False, valido=False,
                ))
            _, term, masks = env.step(actions)
            for r, b in zip(paso, np.where(active)[0]):
                recs[r]["term"] = bool(term[b])
                recs[r]["r_raw"] = float(env.last_raw_reward[b])              # lane cobrada en este paso (USD)
                recs[r]["valido"] = bool(term[b] and actions[b] == env.start_node[b] and not env.last_time_violation[b])
            active &= ~term
        futuro = {}
        for r in reversed(recs):
            r["futuro_usd"] = futuro.get(r["b"], 0.0)
            futuro[r["b"]] = futuro.get(r["b"], 0.0) + r["r_raw"]
        return recs

    # ── (2) reproducción del prefijo: reanudar un episodio en un estado ─────
    @torch.no_grad()
    def reproducir(self, start, start_day, seeds, k_arr, prefijos, forzada, capturar=False, chunk=8192) -> dict:
        """Reanuda episodios en un estado y los sigue con el actor greedy hasta casa.

        El mundo es función pura de (semilla, nodo, día) y no depende de las acciones, así que basta reproducir el
        prefijo de acciones (el de la trayectoria original) hasta el estado, forzar la acción `forzada[r]` en ese
        paso y seguir con el actor greedy. Solo usa la API pública del entorno.

        prefijos : list[list[int]] — acciones de los pasos anteriores al estado (len = paso del estado).
        forzada  : (R,) acción tomada EN el estado (la que se evalúa).
        Devuelve r_forzada (R,) = lane cobrada por la acción forzada (USD), cont (R,) = margen cobrado después de
        llegar a ella (USD), valida (R,) = el episodio cerró en casa sin violar el tiempo, y con capturar los
        (h_t, e_a) del estado (R, d_h) para entrenar las cabezas Ṽ.
        """
        R = len(start)
        if seeds is not None and all(x is None for x in seeds):
            seeds = None
        out = dict(r_forzada=np.zeros(R), cont=np.zeros(R), valida=np.zeros(R, dtype=bool))
        if capturar:
            out["h"] = torch.zeros(R, self.agent.d_h)
            out["e"] = torch.zeros(R, self.agent.d_h)
        for c0 in range(0, R, chunk):
            sl = slice(c0, min(R, c0 + chunk))
            self._reproducir_chunk(
                np.asarray(start)[sl], np.asarray(start_day)[sl], None if seeds is None else np.asarray(seeds)[sl],
                list(k_arr[sl]), prefijos[sl], np.asarray(forzada)[sl], capturar, out, sl,
            )
        return out

    def _reproducir_chunk(self, start, start_day, seeds, k_arr, prefijos, forzada, capturar, out, sl):
        R = len(start)
        env = self.make_env()
        masks = env.reset(start.astype(np.int64), start_day.astype(np.int64), k_max_arr=k_arr, world_seeds=seeds)
        L = np.array([len(p) for p in prefijos])
        P = np.full((R, int(L.max()) + 1), -1, dtype=np.int64)
        for r, p in enumerate(prefijos):
            P[r, :len(p)] = p
        r_f, cont, valida = np.zeros(R), np.zeros(R), np.zeros(R, dtype=bool)
        active = np.ones(R, dtype=bool)
        for t in range(env.max_steps):
            if not active.any():
                break
            h_t, emb, mask_t = self.encode(env, masks)
            greedy = self.acciones(h_t, emb, mask_t, "greedy")
            en_estado = active & (L == t)
            if capturar and en_estado.any():
                ix = np.where(en_estado)[0]
                out["h"][sl.start + ix] = h_t[ix].detach().cpu().float()
                out["e"][sl.start + ix] = emb[ix, torch.as_tensor(forzada[ix], device=emb.device)].detach().cpu().float()
            pref_t = P[:, min(t, P.shape[1] - 1)]
            acc = np.where(L > t, pref_t, np.where(L == t, forzada, greedy)).astype(np.int64)
            _, term, masks = env.step(acc)
            raw = env.last_raw_reward.astype(np.float64)
            r_f[en_estado] = raw[en_estado]
            cont += np.where(active & (L < t), raw, 0.0)
            valida |= active & term & (acc == start) & ~env.last_time_violation & (L <= t)
            active &= ~term
        sl_abs = slice(sl.start, sl.stop)
        out["r_forzada"][sl_abs], out["cont"][sl_abs], out["valida"][sl_abs] = r_f, cont, valida

    # ── (3) regla de aceptación por umbral ───────────────────────────────────
    @torch.no_grad()
    def umbral(self, dis, k_arr, tipo: str) -> pd.DataFrame:
        """Un episodio por mundo con la regla de aceptación por umbral; una fila por mundo.

        Paso a paso sobre las tarifas observadas: entre las lanes con carga válidas cuya tarifa observada ≥ umbral
        de la lane, toma la de mayor (tarifa − umbral)·dist; si ninguna califica, vuelve a casa; donde volver a casa
        no es válido (paso 0), usa la acción del actor (greedy). tipo: "be" (umbral = costo/dist), "be+margen"
        (costo/dist + BE_MARGEN) o "pstar" (el p* en ese estado).
        """
        env = self.make_env()
        B = len(dis["nodo"])
        masks = env.reset(dis["nodo"], dis["start_day"], k_max_arr=k_arr, world_seeds=dis.get("world_seed"))
        active = np.ones(B, dtype=bool)
        ar = np.arange(self.N)
        margen, n_arcos = np.zeros(B), np.zeros(B, dtype=int)
        valido = np.zeros(B, dtype=bool)
        for _ in range(env.max_steps):
            if not active.any():
                break
            h_t, emb, mask_t = self.encode(env, masks)
            a_def = self.acciones(h_t, emb, mask_t, "greedy")
            va, vb = self.valores(h_t, emb) if tipo == "pstar" else (None, None)
            rate_all, loads_all = env.rate_row_obs, env.loads_row_obs
            acc = a_def.copy()
            for b in np.where(active)[0]:
                i, s0 = int(env.current_node[b]), int(env.start_node[b])
                rate, loads = rate_all[b].astype(np.float64), loads_all[b]
                ofertas = np.where((masks[b] == 1) & (loads > 0) & (ar != s0) & (ar != i))[0]
                be = self.cost[i] / self.dist[i]
                ok = []
                for j in ofertas:
                    if tipo == "be":
                        thr = be[j]
                    elif tipo == "be+margen":
                        thr = be[j] + BE_MARGEN
                    else:
                        thr = self.precio(dict(i=i, mask=masks[b], rate=rate, loads=loads, start=s0,
                                               va=va[b], vb=vb[b]), j).p
                    if np.isfinite(thr) and rate[j] >= thr:
                        ok.append((j, (rate[j] - thr) * self.dist[i, j]))
                if ok:
                    acc[b] = max(ok, key=lambda x: x[1])[0]
                elif masks[b][s0] == 1 and i != s0:
                    acc[b] = s0                                                   # ninguna califica: a casa
            _, term, masks = env.step(acc)
            margen += np.where(active, env.last_raw_reward, 0.0)
            n_arcos += active.astype(int)
            valido |= active & term & (acc == env.start_node) & ~env.last_time_violation
            active &= ~term
        return pd.DataFrame(dict(mundo=np.arange(B), margen=margen, valido=valido, n_arcos=n_arcos))


# ─────────────────────────────────────────────────────────────────────────────
# (b) Prueba contrafactual
# ─────────────────────────────────────────────────────────────────────────────

def _corr_cluster_ic(x, y, grupos, B=1000, semilla=BOOTSTRAP_SEED) -> tuple:
    """IC95 de la correlación de Pearson remuestreando mundos enteros."""
    x, y, g = np.asarray(x, float), np.asarray(y, float), np.asarray(grupos)
    ug, inv = np.unique(g, return_inverse=True)
    filas = [np.where(inv == k)[0] for k in range(len(ug))]
    rng = np.random.Generator(np.random.PCG64(semilla))
    cs = []
    for _ in range(B):
        ix = np.concatenate([filas[k] for k in rng.integers(0, len(ug), size=len(ug))])
        if np.std(x[ix]) > 0 and np.std(y[ix]) > 0:
            cs.append(np.corrcoef(x[ix], y[ix])[0, 1])
    return (float(np.percentile(cs, 2.5)), float(np.percentile(cs, 97.5))) if cs else (float("nan"), float("nan"))


def validar_contrafactual(esc: Escenario, n_mundos: int, n_estados: int, rng, k_nivel=None, max_bloques: int = 10) -> tuple:
    """(b) Retorna (estados DataFrame, resumen DataFrame).

    Los mundos se generan por bloques de `n_mundos` (el primero con exactamente MARKET_SEED_VAL, los siguientes con
    MARKET_SEED_VAL + k) hasta reunir n_estados arcos del actor con carga y alternativa: con K libre la ruta típica
    es 1 lane cargada + el retorno, así que hacen falta más mundos que estados.
    """
    N = esc.N
    elegibles, acciones_mundo, bloque = [], {}, 0
    while len(elegibles) < n_estados and bloque < max_bloques:
        dis = disenar_mundos(N, n_mundos, esc.market, num_dias=TRAIN_DAYS, semilla=MARKET_SEED_VAL + bloque)
        recs = esc.recolectar(dis, [k_nivel] * n_mundos, modo="greedy")
        for r in recs:
            r["b"] += bloque * n_mundos                                     # id de mundo único entre bloques
            acciones_mundo.setdefault(r["b"], []).append(r["j"])
        for r in recs:
            if r["j"] == r["start"] or r["loads"][r["j"]] <= 0:
                continue
            pr = esc.precio(r, r["j"])
            if pr.definido:
                elegibles.append((r, pr))
        bloque += 1
    if not elegibles:
        raise RuntimeError("sin estados elegibles para la prueba contrafactual")
    ix = rng.choice(len(elegibles), size=min(n_estados, len(elegibles)), replace=False)
    estados = [elegibles[k] for k in ix]

    # ramas distintas a evaluar: (estado, acción) → fila
    filas, claves = [], {}
    for s, (r, pr) in enumerate(estados):
        for a in {r["j"], pr.alt_a, pr.alt_b}:
            claves[(s, a)] = len(filas)
            filas.append((r, a))
    rama = esc.reproducir(
        [r["start"] for r, _ in filas], [r["start_day"] for r, _ in filas], [r["world_seed"] for r, _ in filas],
        [r["k"] for r, _ in filas], [acciones_mundo[r["b"]][:r["step"]] for r, _ in filas],
        [a for _, a in filas],
    )
    reg = []
    for s, (r, pr) in enumerate(estados):
        i, j = r["i"], r["j"]
        c_j = rama["cont"][claves[(s, j)]]
        g = lambda a: rama["r_forzada"][claves[(s, a)]] + rama["cont"][claves[(s, a)]]
        d, cj = esc.dist[i, j], esc.cost[i, j]
        p_ab_real = (g(pr.alt_a) - c_j + cj) / d
        p_ba_real = (g(pr.alt_b) - c_j + cj) / d
        validas = all(rama["valida"][claves[(s, a)]] for a in {j, pr.alt_a, pr.alt_b})
        reg.append(dict(
            mundo=r["b"], paso=r["step"], i=i, j=j, dist_j=d, tramo=str(_tramo(d)), break_even=pr.break_even,
            rate_j=float(r["rate"][j]), p_pred=pr.p, p_ab=pr.p_ab, p_ba=pr.p_ba, p_min=pr.p_min, p_max=pr.p_max,
            alt_a=pr.alt_a, alt_b=pr.alt_b, p_real=(p_ab_real + p_ba_real) / 2, p_real_ab=p_ab_real,
            p_real_ba=p_ba_real, ramas_validas=validas, c_j=c_j, g_j=g(j), g_alt_ab=g(pr.alt_a), g_alt_ba=g(pr.alt_b),
        ))
    df = pd.DataFrame(reg)
    df["sesgo"] = df["p_pred"] - df["p_real"]
    df["sesgo_usd_lane"] = df["sesgo"] * df["dist_j"]
    df["ancho_rango"] = df["p_max"] - df["p_min"]

    def resumen(g, nombre):
        if len(g) == 0:
            return dict(grupo=nombre, n_estados=0, n_mundos=0)
        lo, hi = ic_bootstrap_cluster(g["sesgo"], g["mundo"])
        lo_l, hi_l = ic_bootstrap_cluster(g["sesgo_usd_lane"], g["mundo"])
        out = dict(
            grupo=nombre, n_estados=len(g), n_mundos=g["mundo"].nunique(),
            sesgo_usd_milla=g["sesgo"].mean(), sesgo_ic95_inf=lo, sesgo_ic95_sup=hi,
            sesgo_usd_lane=g["sesgo_usd_lane"].mean(), sesgo_lane_ic95_inf=lo_l, sesgo_lane_ic95_sup=hi_l,
            p_pred_medio=g["p_pred"].mean(), p_real_medio=g["p_real"].mean(),
            ancho_rango_medio=g["ancho_rango"].mean(),
            corr_pearson=float(np.corrcoef(g["p_pred"], g["p_real"])[0, 1]) if len(g) > 2 else float("nan"),
            corr_spearman=float(spearmanr(g["p_pred"], g["p_real"])[0]) if len(g) > 2 else float("nan"),
            sd_p_real=g["p_real"].std(),
        )
        out["sesgo_frac_disp_indice_inf"] = out["sesgo_usd_milla"] / INDICE_DISPERSION_SEMANAL[1]
        out["sesgo_frac_disp_indice_sup"] = out["sesgo_usd_milla"] / INDICE_DISPERSION_SEMANAL[0]
        return out

    filas_r = [resumen(df, "global")]
    lo, hi = _corr_cluster_ic(df["p_pred"], df["p_real"], df["mundo"])
    filas_r[0]["corr_pearson_ic95_inf"], filas_r[0]["corr_pearson_ic95_sup"] = lo, hi
    for nombre, _, _ in TRAMOS:
        filas_r.append(resumen(df[df["tramo"] == nombre], nombre))
    return df, pd.DataFrame(filas_r)


# ─────────────────────────────────────────────────────────────────────────────
# (c) y (d): arcos de la ruta del actor
# ─────────────────────────────────────────────────────────────────────────────

def arcos_ruta(esc: Escenario, dis, k_arr) -> pd.DataFrame:
    """Una fila por arco de la ruta del actor (también el retorno y los arcos sin carga, con p* = NaN): ruta, tarifa
    de hoy, p*, rango, break-even, percentil histórico, índice de mercado del destino y LTR del nodo ACTUAL (de
    donde sale el arco, no a donde llega). Los arcos con p* definido (con carga, no retorno, con alternativa) son
    los que entran en los resúmenes."""
    recs = esc.recolectar(dis, k_arr, modo="greedy", con_indice=True)
    rutas = {}
    for r in recs:
        rutas.setdefault(r["b"], [r["i"]]).append(r["j"])
    margen = pd.Series({b: 0.0 for b in range(len(dis["nodo"]))})
    filas = []
    for r in recs:
        margen[r["b"]] += r["r_raw"]
        pr = esc.precio(r, r["j"])
        i, j = r["i"], r["j"]
        filas.append(dict(
            mundo=r["b"], paso=r["step"], ruta=" → ".join(str(n) for n in rutas[r["b"]]), i=i, j=j,
            es_retorno=bool(j == r["start"]), tiene_carga=bool(pr.has_load), only_option=bool(pr.only_option),
            dist_j=esc.dist[i, j], rate_j=float(r["rate"][j]), break_even=pr.break_even,
            p=pr.p, p_min=pr.p_min, p_max=pr.p_max, ancho_rango=pr.p_max - pr.p_min,
            percentil=percentil_historico(pr.p, esc.rate_hist[:, i, j], esc.loads_hist[:, i, j]) if pr.definido else float("nan"),
            indice_j=float(r["idx"][j]), ltr_i=r["ltr_i"],
        ))
    df = pd.DataFrame(filas)
    df["margen_mundo"] = df["mundo"].map(margen)                       # margen del tour completo (USD), repetido en cada arco
    return df


def resumen_arcos(df: pd.DataFrame, nombre: str) -> dict:
    """% de arcos con p* bajo el break-even y sobre break-even + BE_MARGEN; percentil y correlación con el índice
    de mercado del destino (esperada negativa) y con el LTR del nodo ACTUAL (esperada positiva: mercado tensionado
    en origen ⇒ más poder de negociación sobre las lanes que salen de ahí ⇒ p* más alto; ver validate_pricing.py
    docstring (d) y README). Solo cuentan los arcos con p* definido."""
    df = df[df["p"].notna()]
    if len(df) == 0:
        return dict(conjunto=nombre, n_arcos=0)
    dif = df["p"] - df["break_even"]
    ok = df["indice_j"].notna() & dif.notna()
    pear = float(np.corrcoef(dif[ok], df["indice_j"][ok])[0, 1]) if ok.sum() > 2 else float("nan")
    spear = float(spearmanr(dif[ok], df["indice_j"][ok])[0]) if ok.sum() > 2 else float("nan")
    ok_ltr = df["ltr_i"].notna() & dif.notna()
    pear_ltr = float(np.corrcoef(dif[ok_ltr], df["ltr_i"][ok_ltr])[0, 1]) if ok_ltr.sum() > 2 else float("nan")
    spear_ltr = float(spearmanr(dif[ok_ltr], df["ltr_i"][ok_ltr])[0]) if ok_ltr.sum() > 2 else float("nan")
    pc = df["percentil"].dropna()
    lo_m, hi_m = ic_bootstrap_cluster(dif, df["mundo"])
    return dict(
        conjunto=nombre, n_arcos=len(df), n_mundos=df["mundo"].nunique(),
        pct_p_bajo_breakeven=100.0 * (df["p"] < df["break_even"]).mean(),
        pct_p_sobre_breakeven_mas_margen=100.0 * (df["p"] > df["break_even"] + BE_MARGEN).mean(),
        p_menos_breakeven_medio=dif.mean(), p_menos_breakeven_ic95_inf=lo_m, p_menos_breakeven_ic95_sup=hi_m,
        ancho_rango_medio=df["ancho_rango"].mean(),
        percentil_medio=pc.mean(), percentil_mediana=pc.median(), n_percentil=len(pc),
        pct_percentil_bajo_5=100.0 * (pc < 5).mean(), pct_percentil_sobre_95=100.0 * (pc > 95).mean(),
        corr_pearson_dif_indice=pear, corr_spearman_dif_indice=spear,
        corr_pearson_dif_ltr=pear_ltr, corr_spearman_dif_ltr=spear_ltr,
    )


def comparar_umbral(esc: Escenario, dis, k_arr, nombre_modo: str) -> tuple:
    """Regla de aceptación por umbral con los tres pisos; resumen y diferencias pareadas (p* − piso) por mundo."""
    res = {t: esc.umbral(dis, k_arr, t) for t in ("be", "be+margen", "pstar")}
    etiquetas = {"be": "break-even", "be+margen": f"break-even + {BE_MARGEN:.2f}", "pstar": "p*"}
    filas = [dict(conjunto=nombre_modo, umbral=etiquetas[t], margen_medio=d["margen"].mean(), validez=d["valido"].mean(),
                  largo_tour_medio=d["n_arcos"].mean(), n_mundos=len(d)) for t, d in res.items()]
    difs = []
    for t in ("be", "be+margen"):
        dif = res["pstar"]["margen"].to_numpy() - res[t]["margen"].to_numpy()
        lo, hi = ic_bootstrap_media(dif)
        difs.append(dict(conjunto=nombre_modo, comparacion=f"p* − {etiquetas[t]}", dif_media_usd=dif.mean(),
                         dif_ic95_inf=lo, dif_ic95_sup=hi, n_mundos=len(dif)))
    pm = pd.concat([d.assign(umbral=etiquetas[t], conjunto=nombre_modo) for t, d in res.items()], ignore_index=True)
    return pd.DataFrame(filas), pd.DataFrame(difs), pm


# ─────────────────────────────────────────────────────────────────────────────
# Figuras
# ─────────────────────────────────────────────────────────────────────────────

def _fig_contrafactual(df, resumen, path):
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.8))
    ax[0].scatter(df["p_real"], df["p_pred"], s=6, alpha=0.35)
    lim = [min(df["p_real"].min(), df["p_pred"].min()), max(df["p_real"].max(), df["p_pred"].max())]
    ax[0].plot(lim, lim, "k--", lw=0.8)
    ax[0].set(xlabel="p* realizado (USD/mi)", ylabel="p* predicho (USD/mi)", title=f"Contrafactual (n={len(df)})")
    ax[0].grid(alpha=0.3)
    ax[1].hist(df["sesgo"].clip(-2, 2), bins=60, color="tab:purple", alpha=0.8)
    ax[1].axvline(0, color="k", lw=0.8); ax[1].axvline(df["sesgo"].mean(), color="r", lw=1.2, label="media")
    ax[1].set(xlabel="sesgo = predicho − realizado (USD/mi, recortado a ±2)", ylabel="Estados"); ax[1].legend()
    t = resumen[resumen["grupo"] != "global"]
    t = t[t["n_estados"] > 0]
    ax[2].bar(t["grupo"], t["sesgo_usd_milla"], yerr=[t["sesgo_usd_milla"] - t["sesgo_ic95_inf"], t["sesgo_ic95_sup"] - t["sesgo_usd_milla"]], capsize=4)
    for k, (g, n) in enumerate(zip(t["grupo"], t["n_estados"])):
        ax[2].text(k, 0, f"n={n}", ha="center", va="bottom", fontsize=8)
    ax[2].axhline(0, color="k", lw=0.6); ax[2].set(ylabel="Sesgo medio (USD/mi)", title="Por tramo de distancia")
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def _fig_plausibilidad(arcos, path):
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.8))
    ax[0].hist(arcos["percentil"].dropna(), bins=20, range=(0, 100), color="tab:green", alpha=0.8)
    ax[0].set(xlabel="Percentil de p* en las tarifas históricas de la lane", ylabel="Arcos", title=f"n={len(arcos)}")
    ax[1].scatter(arcos["break_even"], arcos["p"], s=6, alpha=0.3)
    lim = [arcos[["break_even", "p"]].min().min(), arcos[["break_even", "p"]].max().max()]
    ax[1].plot(lim, lim, "k--", lw=0.8)
    ax[1].set(xlabel="Break-even (USD/mi)", ylabel="p* (USD/mi)", title="p* vs break-even")
    ax[2].scatter(arcos["indice_j"], arcos["p"] - arcos["break_even"], s=6, alpha=0.3)
    ax[2].axhline(0, color="k", lw=0.6)
    ax[2].set(xlabel="Índice de mercado del destino", ylabel="p* − break-even (USD/mi)")
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Orquestación
# ─────────────────────────────────────────────────────────────────────────────

def run_pricing_validation(*args, **kwargs) -> dict:
    """Ver _run_pricing_validation. Siembra torch solo durante la validación y restaura el RNG global al terminar."""
    devs = [torch.cuda.current_device()] if torch.cuda.is_available() else []
    with torch.random.fork_rng(devices=devs):
        torch.manual_seed(PRICING_VAL_SEED)
        return _run_pricing_validation(*args, **kwargs)


def _run_pricing_validation(
    agent, num_nodes, time_matrix, rate_stack, loads_stack, distance_arr, diesel_arr,
    ltr_stack, avail_expected_arr, reward_global_p95, market, ev_esperado=None,
    output_dir=None, n_mundos_cf=N_MUNDOS_CONTRAFACTUAL,
    n_estados_cf=N_EST_CONTRAFACTUAL, m_eval=MARKET_M_EVAL_WORLDS, semilla_eval=MARKET_SEED_EVAL, verbose=True,
) -> dict:
    """Corre (b), (c) y (d) sobre las rutas que la política (ya entrenada y evaluada contra sus baselines en otra
    fase) calcula en mundos frescos; guarda tablas y figuras en <output_dir>/pricing_val y devuelve los DataFrames."""
    if not getattr(agent, "post_entrenadas", False):
        raise ValueError("el agente no trae las cabezas Ṽ A/B entrenadas (correr post_training.py)")
    t0 = time.time()
    esc_sim = Escenario(agent, num_nodes, time_matrix, rate_stack[:TRAIN_DAYS], loads_stack[:TRAIN_DAYS], distance_arr,
                        diesel_arr, ltr_stack, avail_expected_arr, reward_global_p95, market, ev_esperado)
    esc_rep = Escenario(agent, num_nodes, time_matrix, rate_stack[TRAIN_DAYS:], loads_stack[TRAIN_DAYS:], distance_arr,
                        diesel_arr, ltr_stack[:, TRAIN_DAYS:], avail_expected_arr,
                        reward_global_p95, market, ev_esperado, modo="replay",
                        rate_hist=rate_stack[:TRAIN_DAYS], loads_hist=loads_stack[:TRAIN_DAYS])
    out = output_dir or os.path.join(os.path.dirname(os.path.abspath(__file__)), "results",
                                     f"{num_nodes}nodes" + (f"_{RUN_TAG}" if RUN_TAG else ""))
    d_tab, d_fig = (os.path.join(out, "pricing_val", s) for s in ("tables", "figures"))
    os.makedirs(d_tab, exist_ok=True); os.makedirs(d_fig, exist_ok=True)
    rng = np.random.default_rng(PRICING_VAL_SEED)
    tablas = {}

    # ── (b) contrafactual ─────────────────────────────────────────────────
    for K in K_TRAINING_LEVELS:
        cf, cf_res = validar_contrafactual(esc_sim, n_mundos_cf, n_estados_cf, rng, k_nivel=K)
        sfx = "" if len(K_TRAINING_LEVELS) == 1 else f"_K{_etiqueta_k(K)}"
        tablas[f"contrafactual_estados{sfx}"], tablas[f"contrafactual_resumen{sfx}"] = cf, cf_res
        if verbose:
            print(f"\n  (b) contrafactual K={_etiqueta_k(K)} ({len(cf)} estados, {cf['mundo'].nunique()} mundos; "
                  f"{time.time() - t0:.0f} s)")
            with pd.option_context("display.width", 220, "display.max_columns", None):
                print(cf_res[["grupo", "n_estados", "sesgo_usd_milla", "sesgo_ic95_inf", "sesgo_ic95_sup", "sesgo_usd_lane",
                              "corr_pearson", "ancho_rango_medio"]].to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    _fig_contrafactual(tablas[next(k for k in tablas if k.startswith("contrafactual_estados"))],
                       tablas[next(k for k in tablas if k.startswith("contrafactual_resumen"))],
                       os.path.join(d_fig, "contrafactual.png"))

    # ── (c) y (d): arcos de la ruta del actor y regla por umbral, en simulado y en replay ──
    dis_sim = disenar_mundos(num_nodes, m_eval, market, num_dias=TRAIN_DAYS, semilla=semilla_eval)
    nodos_r, dias_r = casos_replay(num_nodes, rate_stack[TRAIN_DAYS:].shape[0])
    dis_rep = dict(nodo=nodos_r.astype(np.int64), start_day=dias_r.astype(np.int64), world_seed=None)
    resumenes, arcos_sim, arcos_rep = [], None, None
    for K in K_TRAINING_LEVELS:
        k_sim, k_rep = [K] * m_eval, [K] * len(nodos_r)
        arcos_sim = arcos_ruta(esc_sim, dis_sim, k_sim)
        arcos_rep = arcos_ruta(esc_rep, dis_rep, k_rep)
        etq = f"simulado K={_etiqueta_k(K)}", f"replay K={_etiqueta_k(K)}"
        resumenes += [resumen_arcos(arcos_sim, etq[0]), resumen_arcos(arcos_rep, etq[1])]
        u_res_s, u_dif_s, u_pm_s = comparar_umbral(esc_sim, dis_sim, k_sim, etq[0])
        u_res_r, u_dif_r, u_pm_r = comparar_umbral(esc_rep, dis_rep, k_rep, etq[1])
        sfx = "" if len(K_TRAINING_LEVELS) == 1 else f"_K{_etiqueta_k(K)}"
        tablas[f"umbral_resumen{sfx}"] = pd.concat([u_res_s, u_res_r], ignore_index=True)
        tablas[f"umbral_diferencias{sfx}"] = pd.concat([u_dif_s, u_dif_r], ignore_index=True)
        tablas[f"umbral_por_mundo{sfx}"] = pd.concat([u_pm_s, u_pm_r], ignore_index=True)
        tablas[f"arcos_simulado{sfx}"], tablas[f"arcos_replay{sfx}"] = arcos_sim, arcos_rep
        if verbose:
            print(f"\n  (c)/(d) arcos de la ruta y regla por umbral, K={_etiqueta_k(K)} ({time.time() - t0:.0f} s)")
            print(tablas[f"umbral_resumen{sfx}"].to_string(index=False, float_format=lambda x: f"{x:.2f}"))
            print(tablas[f"umbral_diferencias{sfx}"].to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    tablas["arcos_resumen"] = pd.DataFrame(resumenes)
    if verbose:
        with pd.option_context("display.width", 250, "display.max_columns", None):
            print(tablas["arcos_resumen"].T.to_string(float_format=lambda x: f"{x:.3f}"))
    _fig_plausibilidad(arcos_sim, os.path.join(d_fig, "plausibilidad.png"))

    curva_csv = os.path.join(out, "post_training_curva.csv")
    if os.path.exists(curva_csv):
        tablas["post_curva"] = pd.read_csv(curva_csv)                      # error de Ṽ en validación (post_training.py)
    for nombre, df in tablas.items():
        df.to_csv(os.path.join(d_tab, f"{nombre}.csv"), index=False)
    if verbose:
        print(f"  tablas y figuras → {os.path.join(out, 'pricing_val')}  ({time.time() - t0:.0f} s)")
    return tablas


def _hoja_arcos(tablas: dict) -> pd.DataFrame:
    """Una fila por arco de la ruta del actor (simulado y replay), lista para la hoja "Pricing arcos"."""
    partes = []
    for k, nombre in (("arcos_simulado", "simulado"), ("arcos_replay", "replay (días 90-119)")):
        for clave in [c for c in tablas if c.startswith(k)]:
            d = tablas[clave].copy()
            sufijo = clave[len(k):].replace("_K", " K=")
            d.insert(0, "conjunto", nombre + sufijo)
            partes.append(d)
    if not partes:
        return pd.DataFrame()
    d = pd.concat(partes, ignore_index=True)
    d = d.sort_values(["conjunto", "mundo", "paso"], kind="stable").reset_index(drop=True)   # los arcos de
    d["posicion_valiosa"] = (d["p"] < d["break_even"]).where(d["p"].notna())                 # una misma ruta quedan juntos
    d["nota"] = np.select([d["es_retorno"], ~d["tiene_carga"], d["only_option"]],
                          ["retorno al depot (sin p*)", "sin carga: reposicionamiento (sin p*)",
                           "única acción factible (sin p*)"], "")
    cols = ["conjunto", "mundo", "paso", "ruta", "i", "j", "dist_j", "rate_j", "p", "p_min", "p_max", "break_even",
            "percentil", "posicion_valiosa", "indice_j", "margen_mundo", "nota"]
    d = d[cols].rename(columns={"i": "desde", "j": "hacia", "dist_j": "distancia_mi", "rate_j": "rate_hoy",
                                "p": "p_estrella", "p_min": "p_estrella_min", "p_max": "p_estrella_max",
                                "percentil": "percentil_historico", "indice_j": "indice_mercado_destino",
                                "margen_mundo": "margen_tour_usd"})
    return d


def append_pricing_sheet(excel_path: str, tablas: dict) -> None:
    """Agrega al Excel de resultados la hoja "Pricing" (resúmenes de la validación) y la hoja "Pricing arcos" (el
    precio de reserva de cada arco de la ruta del actor: p*, rango, tarifa de hoy, break-even y percentil), sin
    tocar las demás hojas."""
    secciones = []
    if "post_curva" in tablas and len(tablas["post_curva"]):
        c = tablas["post_curva"]
        cols = [x for x in ("n_total", "n_val", "sd_objetivo_usd", "rmse_A_usd", "rmse_B_usd", "rmse_promedio_usd",
                            "sesgo_promedio_usd", "r2_promedio", "rmse_dif_AB_usd") if x in c]
        secciones.append(("Ṽ — error en ejemplos de validación (USD) por nº de ejemplos de entrenamiento", c[cols]))
    for k in tablas:
        if k.startswith("contrafactual_resumen"):
            secciones.append(("Contrafactual — sesgo = p* predicho − p* realizado (USD/mi y USD/lane); IC95 por bootstrap de mundos", tablas[k]))
    secciones.append(("Arcos de la ruta: p* vs break-even, percentil histórico, correlación con el índice del destino y con el LTR del nodo actual", tablas["arcos_resumen"]))
    for k in tablas:
        if k.startswith("umbral_resumen"):
            secciones.append(("Regla de aceptación por umbral (pisos: break-even, break-even + margen, p*)", tablas[k]))
        elif k.startswith("umbral_diferencias"):
            secciones.append(("Regla por umbral — diferencias pareadas de margen (p* − piso), IC95 por mundo", tablas[k]))
    nota = ("p* es el costo de oportunidad del transportista (tarifa mínima por milla a la que aceptar la lane "
            "iguala a su mejor alternativa); no es una probabilidad de aceptación del cliente. Su rango [mín, máx] "
            "(parejas AB/BA) mide la variabilidad del estimador, no es un intervalo de confianza. "
            "El detalle por arco de cada ruta está en la hoja \"Pricing arcos\".")
    arcos = _hoja_arcos(tablas)
    from openpyxl.styles import Font
    from openpyxl.utils import get_column_letter

    def valor(v):
        if v is None or (isinstance(v, float) and not np.isfinite(v)) or v is pd.NaT:
            return None
        if isinstance(v, np.generic):
            v = v.item()
        return round(v, 4) if isinstance(v, float) else v

    def escribir(ws, df, fila):
        """Encabezado en negrita + valores desde `fila` (1-indexada); devuelve la siguiente fila libre."""
        for c, col in enumerate(df.columns, 1):
            ws.cell(fila, c).value = str(col)
            ws.cell(fila, c).font = Font(bold=True)
        for r, vals in enumerate(df.itertuples(index=False), fila + 1):
            for c, v in enumerate(vals, 1):
                ws.cell(r, c).value = valor(v)
        return fila + len(df) + 1

    # Se escribe con openpyxl y no con DataFrame.to_excel: con if_sheet_exists="replace" pandas recrea la hoja en cada
    # llamada y solo sobreviviría la última sección.
    with pd.ExcelWriter(excel_path, engine="openpyxl", mode="a", if_sheet_exists="replace") as writer:
        wb = writer.book
        for nombre in ("Pricing", "Pricing arcos"):
            if nombre in wb.sheetnames:
                del wb[nombre]
        ws = wb.create_sheet("Pricing")
        ws.cell(1, 1).value = "Nota"; ws.cell(1, 1).font = Font(bold=True)
        ws.cell(2, 1).value = nota
        fila = 4
        for titulo, df in secciones:
            ws.cell(fila, 1).value = titulo
            ws.cell(fila, 1).font = Font(bold=True, size=12)
            fila = escribir(ws, df, fila + 1) + 2
        ws.column_dimensions["A"].width = 26
        for c in range(2, 24):
            ws.column_dimensions[get_column_letter(c)].width = 16
        if len(arcos):
            wa = wb.create_sheet("Pricing arcos")
            escribir(wa, arcos, 1)
            for c, col in enumerate(arcos.columns, 1):
                wa.column_dimensions[get_column_letter(c)].width = 34 if col in ("ruta", "nota") else 20 if col == "conjunto" else 14
                if col in ("rate_hoy", "p_estrella", "p_estrella_min", "p_estrella_max", "break_even", "indice_mercado_destino"):
                    for r in range(2, len(arcos) + 2):
                        wa.cell(r, c).number_format = "0.00"
                elif col == "percentil_historico":
                    for r in range(2, len(arcos) + 2):
                        wa.cell(r, c).number_format = "0"
            wa.freeze_panes = "A2"
            wa.auto_filter.ref = wa.dimensions
    print(f"Pestañas Pricing y Pricing arcos agregadas → {excel_path}")


def cargar_datos(num_nodes):
    """Datos y proceso de mercado para una corrida independiente (CLI)."""
    (time_matrix, rate_stack, loads_stack, distance_arr, diesel_arr,
     ltr_stack, avail_expected_arr, reward_global_p95) = load_matrices(num_nodes)
    asegurar_parametros()
    market = MarketProcess(num_nodes)
    return dict(
        time_matrix=time_matrix, rate_stack=rate_stack, loads_stack=loads_stack, distance_arr=distance_arr,
        diesel_arr=diesel_arr, ltr_stack=ltr_stack,
        avail_expected_arr=avail_expected_arr, reward_global_p95=reward_global_p95, market=market,
        ev_esperado=calcular_ev_pronostico(market, distance_arr, diesel_arr),
    )


def main():
    """Carga un checkpoint con las cabezas Ṽ A/B ya entrenadas (post_training.py) y corre (b)/(c)/(d) sobre las
    rutas que la política congelada calcula en mundos frescos. No entrena ni evalúa el ruteo contra sus
    baselines (B1-B3/UB-PI/UB-DP) — eso ya se hizo aparte, en main.py/evaluation.py."""
    from transfer_learning import checkpoint_tiene_post_head, load_checkpoint_compatible
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--num-nodes", type=int, default=10)
    ap.add_argument("--checkpoint", help="ruta del .pt con las cabezas Ṽ (default: checkpoints[/RUN_TAG]/am_checkpoint_<N>nodes_pricing.pt)")
    ap.add_argument("--restringido", action="store_true", help="valida con K libre y K = K_MAX (default: solo K libre)")
    ap.add_argument("--output-dir", help="carpeta de salida (default: results/<N>nodes[_<RUN_TAG>])")
    ap.add_argument("--quick", action="store_true", help="tamaños reducidos para una prueba rápida")
    a = ap.parse_args()

    configurar_k(a.restringido)
    cwd = os.path.dirname(os.path.abspath(__file__))
    path = a.checkpoint or os.path.join(cwd, "checkpoints", RUN_TAG, f"am_checkpoint_{a.num_nodes}nodes_pricing.pt")
    if not checkpoint_tiene_post_head(path):
        raise SystemExit(f"{path} no trae las cabezas Ṽ A/B: corre post_training.py antes de validar el pricing.")
    agent, _ = load_checkpoint_compatible(path, a.num_nodes, AM_D_H, AM_N_HEADS, AM_N_LAYERS, AM_D_FF, verbose=False)
    agent.eval()
    d = cargar_datos(a.num_nodes)
    tam = dict(n_mundos_cf=200, n_estados_cf=100, m_eval=60) if a.quick else {}
    run_pricing_validation(agent, a.num_nodes, **d, output_dir=a.output_dir, **tam)


if __name__ == "__main__":
    main()
