"""
validate_pricing.py
===================
Validación del precio de reserva p* (PRICING_PLAN.md, Fase P3). Tres comprobaciones sobre
mundos simulados de validación (MARKET_SEED_VAL), con un checkpoint que trae `post_value`:

  (a) Calibración de Ṽ por Monte Carlo. Se muestrean (s, j) del rollout de la política; desde la
      llegada a j se corren R rollouts de la política con mundos nuevos (el tablero de j y los
      siguientes se vuelven a sortear) y se compara la media con Ṽ(s, j) de la cabeza. Sesgo y
      RMSE en USD, globales y por bucket de tiempo restante.
  (b) Posición de p* dentro de la serie histórica de tarifas de la lane (días de train): percentil
      de p* y su sensibilidad al día de la semana y al tiempo restante (tabla pivot).
  (c) Política de aceptación, bajo el supuesto EXPLÍCITO de que toda carga ofrecida está
      disponible a su tarifa observada: "acepto si tarifa ≥ p*" contra "acepto la mejor tarifa" y
      "acepto siempre que haya carga" (más el greedy del DRL como referencia).

p* es un costo de oportunidad del transportista, no una probabilidad de aceptación (D5).

Salidas: <output_dir>/pricing_val/figures/*.png y <output_dir>/pricing_val/tables/*.csv
(output_dir por defecto results/<N>nodes[_<RUN_TAG>]).

Uso
---
    python validate_pricing.py --num-nodes 10 [--checkpoint RUTA] [--best] [--quick]
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

from config import (
    DEVICE, MAX_DURATION, REWARD_SCALE_FACTOR as S, TRAIN_DAYS, MARKET_SEED_VAL, PRICING_VAL_SEED,
    K_TRAINING_LEVELS, configurar_k, RUN_TAG, AM_D_H, AM_N_HEADS, AM_N_LAYERS, AM_D_FF,
)
from attention_encoder import (
    build_node_features_batch, build_temporal_features_batch, build_market_features_batch,
    build_budget_features_batch, build_dow_features_batch,
)
from baselines import calcular_ev_esperado
from evaluation import disenar_mundos, make_sim_env
from market_process import MarketProcess, asegurar_parametros
from pricing import lane_cost_row, percentil_historico, reservation_price
from problem_data import load_matrices
from stats_analysis import ic_bootstrap_media

DOW = ["Lun", "Mar", "Mié", "Jue", "Vie", "Sáb", "Dom"]
# Buckets de horas restantes. En (b) se mide en el estado de decisión s (el tablero de hoy); en (a) a la
# llegada a j, que es de lo que depende Ṽ(s, j): casi todos los s de un tour de 2-3 arcos tienen ≥ 60 h.
BUCKET_LABELS = ["<20 h", "20-40 h", "40-60 h", "≥60 h"]
BUCKET_LABELS_LLEGADA = ["<15 h", "15-30 h", "30-45 h", "≥45 h"]


def _bucket(rem_h, edges=(0.0, 20.0, 40.0, 60.0, np.inf), labels=None) -> np.ndarray:
    return pd.cut(np.asarray(rem_h, dtype=float), list(edges), right=False, labels=labels or BUCKET_LABELS)


# ─────────────────────────────────────────────────────────────────────────────
# Escenario: datos + red, con el forward compartido por los tres rollouts
# ─────────────────────────────────────────────────────────────────────────────

class Escenario:
    """Datos del problema y utilidades de rollout por lotes sobre mundos simulados."""

    def __init__(self, agent, num_nodes, time_matrix, rate_stack, loads_stack, distance_arr, diesel_arr,
                 ltr_stack, trucks_stack, avail_expected_arr, reward_global_p95, market, ev_esperado=None):
        if getattr(agent, "post_value", None) is None:
            raise ValueError("el agente no tiene post_value (USE_POST_HEAD=False o checkpoint sin la cabeza)")
        self.agent, self.N, self.market = agent.eval(), num_nodes, market
        self.T = np.asarray(time_matrix, dtype=float)
        self.dist, self.diesel = distance_arr, diesel_arr
        self.cost = lane_cost_row(distance_arr, diesel_arr)            # (N, N)
        self.rate_train, self.loads_train = rate_stack[:TRAIN_DAYS], loads_stack[:TRAIN_DAYS]
        self.time_matrix = time_matrix
        self.ltr, self.trucks, self.avail, self.p95 = ltr_stack, trucks_stack, avail_expected_arr, reward_global_p95
        self.ev = ev_esperado if ev_esperado is not None else calcular_ev_esperado(market, distance_arr, diesel_arr)
        self.device = next(agent.parameters()).device

    def make_env(self):
        return make_sim_env(self.time_matrix, self.rate_train, self.loads_train, self.dist, self.diesel,
                            self.N, self.market)

    def encode(self, env, masks):
        """(h_t, embeddings, mask_t) del estado actual de `env`; mismas features que greedy_rollout."""
        B, cur, dev = len(env.current_node), env.current_node, self.device
        nf = build_node_features_batch(
            cur, env.start_node, None, self.T, self.dist, env.current_day_idx,
            self.N, MAX_DURATION, trucks_stack_raw=self.trucks, reward_global_p95=self.p95,
            avail_expected_arr=self.avail, reward_rows=env.reward_rows, ev=self.ev,
            time_elapsed=env.time_elapsed, dow0=env.start_dow,
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

    def post_values(self, h_t, emb) -> np.ndarray:
        """(B, N) Ṽ(s, j) de la cabeza, con el ancla Ṽ(s, depot) = 0 aplicada por el llamador."""
        return self.agent.post_value(h_t, emb).cpu().numpy().astype(np.float64)

    def act(self, h_t, emb, mask_t, sample: bool) -> np.ndarray:
        if sample:
            return self.agent.decoder.act(h_t, emb, mask_t)[0].cpu().numpy()
        return self.agent.decoder.greedy_action(h_t, emb, mask_t).cpu().numpy()

    # ── (1) trayectorias de la política con todo lo que p* necesita ──────────
    @torch.no_grad()
    def recolectar(self, dis, k_arr, sample=True) -> list:
        """Rollout de la política sobre los mundos `dis`; un registro por (episodio, paso).

        Cada registro guarda lo observado en el estado s (máscara, tablero, Ṽ de la cabeza) y la
        acción tomada, con lo necesario para reanudar el episodio en el nodo de llegada.
        """
        env = self.make_env()
        B = len(dis["nodo"])
        masks = env.reset(dis["nodo"], dis["start_day"], k_max_arr=k_arr, world_seeds=dis["world_seed"])
        active = np.ones(B, dtype=bool)
        recs = []
        for _ in range(env.max_steps):
            if not active.any():
                break
            h_t, emb, mask_t = self.encode(env, masks)
            v = self.post_values(h_t, emb)
            actions = self.act(h_t, emb, mask_t, sample)
            paso = []
            for b in np.where(active)[0]:
                paso.append(len(recs))
                recs.append(dict(
                    b=int(b), step=int(env.step_count[b]), i=int(env.current_node[b]), j=int(actions[b]),
                    start=int(env.start_node[b]), start_day=int(dis["start_day"][b]), k=k_arr[b],
                    t=float(env.time_elapsed[b]), mask=masks[b].copy(), v=v[b].copy(),
                    rate=env.current_rate_row[b].astype(np.float64), loads=env.current_loads_row[b].copy(),
                    k_rem=int(env.k_remaining[b]),
                    dow=int(env.current_dow[b]), term=False,
                ))
            _, term, masks = env.step(actions)
            for r, b in zip(paso, np.where(active)[0]):
                recs[r]["term"] = bool(term[b])
                recs[r]["r_raw"] = float(env.last_raw_reward[b])        # lane cobrada en este paso (USD)
            active &= ~term
        # Lo que se cobra DESPUÉS de este paso en el mismo episodio: una muestra del valor que Ṽ y el MC estiman.
        futuro = {}
        for r in reversed(recs):
            r["futuro_usd"] = futuro.get(r["b"], 0.0)
            futuro[r["b"]] = futuro.get(r["b"], 0.0) + r["r_raw"]
        return recs

    def p_estrella(self, rec, j) -> tuple:
        """(p*, only_option) de la lane rec['i']→j en el estado del registro."""
        i = rec["i"]
        return reservation_price(
            self.agent, None, None, rec["mask"], i, j, rec["rate"], self.dist[i], self.cost[i],
            rec["start"], S, loads_row=rec["loads"], post_values=rec["v"],
        )

    # ── (2) valor de continuación por Monte Carlo ───────────────────────────
    @torch.no_grad()
    def mc_valores(self, estados, R, rng, chunk_rows=3000, sample=True) -> tuple:
        """Media y error estándar del retorno escalado (1/S) que sigue a llegar a j.

        estados : list[(rec, j)]. Cada uno se reanuda en el nodo j, en t + t_ij, con el mismo
        presupuesto K, y R mundos nuevos: el tablero de j se vuelve a sortear.
        """
        medias, ses = [], []
        por_chunk = max(1, chunk_rows // R)
        for c0 in range(0, len(estados), por_chunk):
            sub = estados[c0:c0 + por_chunk]
            ret = self._mc_chunk(sub, R, rng, sample)                     # (len(sub), R)
            medias += list(ret.mean(axis=1))
            ses += list(ret.std(axis=1, ddof=1) / np.sqrt(R))
        return np.array(medias), np.array(ses)

    def _mc_chunk(self, sub, R, rng, sample) -> np.ndarray:
        rep = lambda xs: np.repeat(np.asarray(xs), R, axis=0)
        n = len(sub)
        B = n * R
        start = rep([r["start"] for r, _ in sub]).astype(np.int64)
        sday = rep([r["start_day"] for r, _ in sub]).astype(np.int64)
        k_arr = [r["k"] for r, _ in sub for _ in range(R)]
        env = self.make_env()
        env.reset(start, sday, k_max_arr=k_arr, world_seeds=rng.integers(0, 2**31 - 1, size=B))
        # Estado de llegada a j (los atributos privados son los que step() actualiza).
        j = rep([jj for _, jj in sub]).astype(np.int64)
        i = rep([r["i"] for r, _ in sub]).astype(np.int64)
        env._current_node = j
        env._time_elapsed = (rep([r["t"] for r, _ in sub]).astype(np.float32)
                             + self.T[i, j].astype(np.float32))
        env._step_count = rep([r["step"] + 1 for r, _ in sub]).astype(np.int64)
        env._k_remaining = rep([r["k_rem"] - 1 for r, _ in sub]).astype(np.int64)
        env._active = np.ones(B, dtype=bool)
        env._revelar_mercado(np.arange(B), env._start_day_idx + (env._time_elapsed // 14).astype(np.int64))
        masks = env._compute_masks()

        ret = np.zeros(B, dtype=np.float64)
        active = np.ones(B, dtype=bool)
        for _ in range(env.max_steps):
            if not active.any():
                break
            h_t, emb, mask_t = self.encode(env, masks)
            actions = self.act(h_t, emb, mask_t, sample)
            rewards, term, masks = env.step(actions)
            ret += np.where(active, rewards, 0.0)
            active &= ~term
        return ret.reshape(n, R)

    # ── (3) política de aceptación ──────────────────────────────────────────
    @torch.no_grad()
    def aceptacion(self, dis, k_arr, regla: str, rng) -> pd.DataFrame:
        """Un episodio por mundo bajo la regla de aceptación; una fila por mundo.

        Reglas: "drl" (greedy de la política), "pstar" (acepto si tarifa ≥ p*), "mejor_tarifa" (la
        oferta de mayor tarifa/milla) y "siempre" (una oferta al azar). Oferta = lane saliente
        válida con carga. Sin oferta aceptable se vuelve al depot; en el primer paso, donde el depot
        está enmascarado, se usa la acción greedy de la política.
        """
        env = self.make_env()
        B = len(dis["nodo"])
        masks = env.reset(dis["nodo"], dis["start_day"], k_max_arr=k_arr, world_seeds=dis["world_seed"])
        active = np.ones(B, dtype=bool)
        ar = np.arange(self.N)
        margen = np.zeros(B)
        n_ofertas = np.zeros(B)
        n_elegibles = np.zeros(B)
        n_aceptadas = np.zeros(B)
        n_arcos = np.zeros(B, dtype=int)
        valido = np.zeros(B, dtype=bool)
        for _ in range(env.max_steps):
            if not active.any():
                break
            h_t, emb, mask_t = self.encode(env, masks)
            greedy = self.act(h_t, emb, mask_t, False)
            v = self.post_values(h_t, emb) if regla == "pstar" else None
            acc = greedy.copy()
            for b in np.where(active)[0]:
                i, s0 = int(env.current_node[b]), int(env.start_node[b])
                loads, rate = env.current_loads_row[b], env.current_rate_row[b].astype(np.float64)
                ofertas = np.where((masks[b] == 1) & (loads > 0) & (ar != s0) & (ar != i))[0]
                n_ofertas[b] += len(ofertas)
                volver = s0 if (masks[b][s0] == 1 and i != s0) else int(greedy[b])
                if regla == "drl":
                    acc[b] = greedy[b]
                elif regla == "pstar":
                    rec = dict(i=i, mask=masks[b], rate=rate, loads=loads, start=s0, v=v[b])
                    pj = [(j, self.p_estrella(rec, j)[0]) for j in ofertas]
                    ok = [(j, (rate[j] - p) * self.dist[i, j]) for j, p in pj if rate[j] >= p]
                    n_elegibles[b] += len(ok)
                    acc[b] = max(ok, key=lambda x: x[1])[0] if ok else volver
                elif regla == "mejor_tarifa":
                    acc[b] = int(ofertas[np.argmax(rate[ofertas])]) if len(ofertas) else volver
                elif regla == "siempre":
                    acc[b] = int(rng.choice(ofertas)) if len(ofertas) else volver
                else:
                    raise ValueError(regla)
                if loads[acc[b]] > 0 and acc[b] != s0:
                    n_aceptadas[b] += 1
            rewards, term, masks = env.step(acc)
            margen += np.where(active, env.last_raw_reward, 0.0)
            n_arcos += active.astype(int)
            valido |= active & term & (acc == env.start_node) & ~env.last_time_violation
            active &= ~term
        return pd.DataFrame(dict(
            mundo=np.arange(B), margen=margen, valido=valido, n_arcos=n_arcos,
            n_ofertas=n_ofertas, n_aceptadas=n_aceptadas, n_elegibles=n_elegibles,
        ))


# ─────────────────────────────────────────────────────────────────────────────
# Las tres validaciones
# ─────────────────────────────────────────────────────────────────────────────

def validar_calibracion(esc: Escenario, recs, rng, n_estados=300, n_rollouts=50, n_no_tomadas=100) -> tuple:
    """(a) Ṽ de la cabeza contra la media Monte Carlo. Retorna (muestras, resumen) en USD."""
    N = esc.N
    tomadas = [(r, r["j"]) for r in recs if r["j"] != r["start"] and not r["term"]]
    no_tomadas = []
    for r in recs:
        cand = np.where((r["mask"] == 1) & (r["loads"] > 0) & (np.arange(N) != r["start"])
                        & (np.arange(N) != r["i"]) & (np.arange(N) != r["j"]))[0]
        no_tomadas += [(r, int(j)) for j in cand]
    ix = lambda lst, n: [lst[k] for k in rng.choice(len(lst), size=min(n, len(lst)), replace=False)]
    estados = [(e, True) for e in ix(tomadas, n_estados)] + [(e, False) for e in ix(no_tomadas, n_no_tomadas)]
    mc, se = esc.mc_valores([e for e, _ in estados], n_rollouts, rng)
    filas = []
    for ((r, j), tomada), m, s_ in zip(estados, mc, se):
        filas.append(dict(
            i=r["i"], j=j, tomada=tomada, tiempo_restante_h=MAX_DURATION - r["t"],
            tiempo_restante_llegada_h=MAX_DURATION - r["t"] - esc.T[r["i"], j], dow=r["dow"],
            v_pred_usd=float(r["v"][j]) * S, v_mc_usd=m * S, se_mc_usd=s_ * S,
            v_real_usd=r["futuro_usd"] if tomada else np.nan,        # una realización del mismo episodio (chequeo del MC)
        ))
    df = pd.DataFrame(filas)
    df["error_usd"] = df["v_pred_usd"] - df["v_mc_usd"]
    df["bucket"] = _bucket(df["tiempo_restante_llegada_h"], (-np.inf, 15.0, 30.0, 45.0, np.inf), BUCKET_LABELS_LLEGADA)

    def resumen(g, nombre, bucket):
        return dict(grupo=nombre, bucket=bucket, n=len(g), sesgo_usd=g["error_usd"].mean(),
                    sesgo_mc_vs_realizado_usd=(g["v_mc_usd"] - g["v_real_usd"]).mean(),
                    rmse_usd=float(np.sqrt((g["error_usd"] ** 2).mean())),
                    se_mc_medio_usd=g["se_mc_usd"].mean(), v_mc_medio_usd=g["v_mc_usd"].mean())

    filas = []
    for nombre, g0 in (("tomadas", df[df["tomada"]]), ("no tomadas", df[~df["tomada"]]), ("todas", df)):
        if len(g0):
            filas.append(resumen(g0, nombre, "global"))
            for bk in BUCKET_LABELS_LLEGADA:
                g = g0[g0["bucket"] == bk]
                if len(g):
                    filas.append(resumen(g, nombre, bk))
    return df, pd.DataFrame(filas)


def validar_posicion(esc: Escenario, recs, rng, n_muestras=500) -> tuple:
    """(b) Percentil de p* en la serie histórica de la lane. Retorna (muestras, pivot_percentil, pivot_n)."""
    N = esc.N
    ofertas = []
    for r in recs:
        cand = np.where((r["mask"] == 1) & (r["loads"] > 0) & (np.arange(N) != r["start"])
                        & (np.arange(N) != r["i"]))[0]
        ofertas += [(r, int(j)) for j in cand]
    filas = []
    for k in rng.choice(len(ofertas), size=min(n_muestras, len(ofertas)), replace=False):
        r, j = ofertas[k]
        i = r["i"]
        p, only = esc.p_estrella(r, j)
        serie = esc.rate_train[:, i, j]
        filas.append(dict(
            i=i, j=j, dow=r["dow"], tiempo_restante_h=MAX_DURATION - r["t"], p_estrella=p,
            rate_hoy=float(r["rate"][j]), break_even=float(esc.cost[i, j] / esc.dist[i, j]),
            only_option=only, percentil_p_estrella=percentil_historico(p, serie),
            percentil_rate_hoy=percentil_historico(float(r["rate"][j]), serie),
        ))
    df = pd.DataFrame(filas)
    df["posicion_valiosa"] = df["p_estrella"] < df["break_even"]
    df["bucket"] = _bucket(df["tiempo_restante_h"])
    df["dia"] = df["dow"].map(lambda d: DOW[d])
    piv = lambda col, fn: df.pivot_table(index="dia", columns="bucket", values=col, aggfunc=fn,
                                         observed=False).reindex(DOW).reindex(columns=BUCKET_LABELS)
    return df, piv("percentil_p_estrella", "mean"), piv("percentil_p_estrella", "count")


def validar_aceptacion(esc: Escenario, dis, k_arr, rng_seed) -> tuple:
    """(c) Reglas de aceptación sobre los mismos mundos. Retorna (resumen, por_mundo)."""
    reglas = [("drl", "Greedy del DRL (referencia)"), ("pstar", "Acepto si tarifa ≥ p*"),
              ("mejor_tarifa", "Acepto la mejor tarifa"), ("siempre", "Acepto siempre que haya carga")]
    por_mundo = {}
    for clave, _ in reglas:
        por_mundo[clave] = esc.aceptacion(dis, k_arr, clave, np.random.default_rng(rng_seed))
    ref = por_mundo["pstar"]["margen"].to_numpy()
    filas = []
    for clave, nombre in reglas:
        d = por_mundo[clave]
        dif = d["margen"].to_numpy() - ref
        lo, hi = ic_bootstrap_media(dif)
        filas.append(dict(
            regla=nombre, margen_medio_usd=d["margen"].mean(), dif_vs_pstar_usd=dif.mean(),
            dif_ic95_bajo=lo, dif_ic95_alto=hi,
            pct_ofertas_aceptadas=100.0 * d["n_aceptadas"].sum() / max(d["n_ofertas"].sum(), 1),
            pct_ofertas_elegibles=(100.0 * d["n_elegibles"].sum() / max(d["n_ofertas"].sum(), 1)
                                   if clave == "pstar" else float("nan")),
            largo_tour_medio=d["n_arcos"].mean(), validez=d["valido"].mean(), n_mundos=len(d),
        ))
    resumen = pd.DataFrame(filas)
    pm = pd.concat([d.assign(regla=c) for c, d in por_mundo.items()], ignore_index=True)
    return resumen, pm


# ─────────────────────────────────────────────────────────────────────────────
# Figuras
# ─────────────────────────────────────────────────────────────────────────────

def _fig_calibracion(df, resumen, path):
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    for tomada, color, lab in ((True, "tab:blue", "acción tomada"), (False, "tab:orange", "no tomada")):
        g = df[df["tomada"] == tomada]
        ax[0].errorbar(g["v_mc_usd"], g["v_pred_usd"], xerr=1.96 * g["se_mc_usd"], fmt="o", ms=3,
                       alpha=0.5, color=color, elinewidth=0.5, label=f"{lab} (n={len(g)})")
    lim = [min(df["v_mc_usd"].min(), df["v_pred_usd"].min()), max(df["v_mc_usd"].max(), df["v_pred_usd"].max())]
    ax[0].plot(lim, lim, "k--", lw=0.8)
    ax[0].set(xlabel="Ṽ Monte Carlo (USD)", ylabel="Ṽ cabeza (USD)", title="Calibración de Ṽ(s, j)")
    ax[0].grid(alpha=0.3); ax[0].legend(fontsize=8)
    g = resumen[(resumen["grupo"] == "todas") & (resumen["bucket"] != "global")]
    x = np.arange(len(g))
    ax[1].bar(x - 0.2, g["sesgo_usd"], 0.4, label="sesgo")
    ax[1].bar(x + 0.2, g["rmse_usd"], 0.4, label="RMSE")
    ax[1].set_xticks(x); ax[1].set_xticklabels([f"{b}\n(n={n})" for b, n in zip(g["bucket"], g["n"])])
    ax[1].axhline(0, color="k", lw=0.6)
    ax[1].set(xlabel="Tiempo restante a la llegada a j", ylabel="USD", title="Error de Ṽ por tiempo restante")
    ax[1].legend(); ax[1].grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def _fig_posicion(df, piv, path):
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.8))
    ax[0].hist(df["percentil_p_estrella"].dropna(), bins=20, range=(0, 100), color="tab:green", alpha=0.8)
    ax[0].set(xlabel="Percentil de p* en la serie histórica de la lane", ylabel="Frecuencia",
              title=f"Posición de p* (n={len(df)})")
    ax[1].scatter(df["break_even"], df["p_estrella"], s=8, alpha=0.5, c=df["posicion_valiosa"].map({True: "tab:red", False: "tab:blue"}))
    lim = [df[["break_even", "p_estrella"]].min().min(), df[["break_even", "p_estrella"]].max().max()]
    ax[1].plot(lim, lim, "k--", lw=0.8)
    ax[1].set(xlabel="Break-even (USD/mi)", ylabel="p* (USD/mi)", title="p* vs break-even (rojo: posición valiosa)")
    im = ax[2].imshow(piv.to_numpy(dtype=float), aspect="auto", cmap="viridis", vmin=0, vmax=100)
    ax[2].set_xticks(range(len(piv.columns))); ax[2].set_xticklabels(piv.columns)
    ax[2].set_yticks(range(len(piv.index))); ax[2].set_yticklabels(piv.index)
    ax[2].set(xlabel="Tiempo restante", ylabel="Día de la semana", title="Percentil medio de p*")
    fig.colorbar(im, ax=ax[2])
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def _fig_aceptacion(res, path):
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    ax[0].barh(res["regla"], res["margen_medio_usd"], color="tab:blue")
    ax[0].set(xlabel="Margen medio por tour (USD)", title="Margen por regla de aceptación")
    ax[0].invert_yaxis(); ax[0].grid(alpha=0.3, axis="x")
    ax[1].barh(res["regla"], res["pct_ofertas_aceptadas"], color="tab:orange")
    ax[1].set(xlabel="% de ofertas aceptadas", title="Ofertas aceptadas")
    ax[1].invert_yaxis(); ax[1].grid(alpha=0.3, axis="x")
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Orquestación
# ─────────────────────────────────────────────────────────────────────────────

def run_pricing_validation(*args, **kwargs) -> dict:
    """Ver _run_pricing_validation. Siembra torch solo durante la validación (reproducible) y restaura el
    RNG global al terminar, para no alterar lo que corra después en el mismo proceso (main.py)."""
    devs = [torch.cuda.current_device()] if torch.cuda.is_available() else []
    with torch.random.fork_rng(devices=devs):
        torch.manual_seed(PRICING_VAL_SEED)
        return _run_pricing_validation(*args, **kwargs)


def _run_pricing_validation(
    agent, num_nodes, time_matrix, rate_stack, loads_stack, distance_arr, diesel_arr,
    ltr_stack, trucks_stack, avail_expected_arr, reward_global_p95, market, ev_esperado=None,
    output_dir=None, n_worlds=500, n_mc_states=300, n_mc_rollouts=50, n_pos=500, n_no_tomadas=100,
    verbose=True,
) -> dict:
    """Corre (a), (b) y (c); guarda tablas y figuras en <output_dir>/pricing_val y devuelve los DataFrames."""
    t0 = time.time()
    esc = Escenario(agent, num_nodes, time_matrix, rate_stack, loads_stack, distance_arr, diesel_arr,
                    ltr_stack, trucks_stack, avail_expected_arr, reward_global_p95, market, ev_esperado)
    out = output_dir or os.path.join(os.path.dirname(os.path.abspath(__file__)), "results",
                                     f"{num_nodes}nodes" + (f"_{RUN_TAG}" if RUN_TAG else ""))
    d_tab, d_fig = (os.path.join(out, "pricing_val", s) for s in ("tables", "figures"))
    os.makedirs(d_tab, exist_ok=True); os.makedirs(d_fig, exist_ok=True)

    rng = np.random.default_rng(PRICING_VAL_SEED)
    dis = disenar_mundos(num_nodes, n_worlds, market, num_dias=esc.rate_train.shape[0], semilla=MARKET_SEED_VAL)
    niveles = list(K_TRAINING_LEVELS)
    k_arr = [niveles[m % len(niveles)] for m in range(n_worlds)]

    if verbose:
        print(f"\n--- Validación del precio de reserva ({num_nodes} nodos, {n_worlds} mundos de validación) ---")
    recs = esc.recolectar(dis, k_arr, sample=True)
    if verbose:
        print(f"  trayectorias: {len(recs)} estados")

    cal, cal_res = validar_calibracion(esc, recs, rng, n_mc_states, n_mc_rollouts, n_no_tomadas)
    if verbose:
        print(f"  (a) calibración de Ṽ ({len(cal)} pares, {n_mc_rollouts} rollouts c/u): "
              f"{time.time() - t0:.0f} s")
        print(cal_res[cal_res["bucket"] == "global"].to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    pos, piv_pct, piv_n = validar_posicion(esc, recs, rng, n_pos)
    if verbose:
        print(f"  (b) posición de p* ({len(pos)} ofertas): percentil medio {pos['percentil_p_estrella'].mean():.1f}, "
              f"posición valiosa en {100 * pos['posicion_valiosa'].mean():.1f} %")
    acep, acep_mundo = validar_aceptacion(esc, dis, k_arr, PRICING_VAL_SEED)
    if verbose:
        print("  (c) política de aceptación:")
        print(acep.to_string(index=False, float_format=lambda x: f"{x:.2f}"))

    tablas = {
        "mc_calibracion_muestras": cal, "mc_calibracion_resumen": cal_res,
        "posicion_muestras": pos, "posicion_pivot_percentil_dia_tiempo": piv_pct.reset_index(),
        "posicion_pivot_n_dia_tiempo": piv_n.reset_index(),
        "aceptacion_resumen": acep, "aceptacion_por_mundo": acep_mundo,
    }
    for nombre, df in tablas.items():
        df.to_csv(os.path.join(d_tab, f"{nombre}.csv"), index=False)
    _fig_calibracion(cal, cal_res, os.path.join(d_fig, "mc_calibracion.png"))
    _fig_posicion(pos, piv_pct, os.path.join(d_fig, "posicion_p_estrella.png"))
    _fig_aceptacion(acep, os.path.join(d_fig, "aceptacion.png"))
    if verbose:
        print(f"  tablas y figuras → {os.path.join(out, 'pricing_val')}  ({time.time() - t0:.0f} s)")
    return tablas


def append_pricing_sheet(excel_path: str, tablas: dict) -> None:
    """Agrega la pestaña "Pricing" al Excel de resultados (sin tocar las demás hojas)."""
    secciones = [
        ("Calibración de Ṽ(s, j) contra Monte Carlo (USD) — sesgo = Ṽ cabeza − Ṽ MC", tablas["mc_calibracion_resumen"]),
        ("Posición de p* — percentil medio de p* en la serie histórica de la lane, por día y tiempo restante",
         tablas["posicion_pivot_percentil_dia_tiempo"]),
        ("Posición de p* — nº de ofertas por celda", tablas["posicion_pivot_n_dia_tiempo"]),
        ("Política de aceptación (supuesto: toda carga ofrecida está disponible a su tarifa observada)",
         tablas["aceptacion_resumen"]),
    ]
    nota = ("p* es el costo de oportunidad del transportista (tarifa mínima por milla a la que aceptar la lane "
            "iguala a su mejor alternativa); no es una probabilidad de aceptación del cliente.")
    with pd.ExcelWriter(excel_path, engine="openpyxl", mode="a", if_sheet_exists="replace") as writer:
        fila = 0
        pd.DataFrame({"Nota": [nota]}).to_excel(writer, sheet_name="Pricing", index=False, startrow=fila)
        fila += 3
        for titulo, df in secciones:
            pd.DataFrame({titulo: []}).to_excel(writer, sheet_name="Pricing", index=False, startrow=fila)
            df.to_excel(writer, sheet_name="Pricing", index=False, startrow=fila + 1)
            fila += len(df) + 4
    print(f"Pestaña Pricing agregada → {excel_path}")


def cargar_datos(num_nodes):
    """Datos y proceso de mercado para una corrida independiente (CLI)."""
    (time_matrix, rate_stack, loads_stack, distance_arr, diesel_arr,
     ltr_stack, trucks_stack, avail_expected_arr, reward_global_p95) = load_matrices(num_nodes)
    asegurar_parametros()
    market = MarketProcess(num_nodes)
    return dict(
        time_matrix=time_matrix, rate_stack=rate_stack, loads_stack=loads_stack, distance_arr=distance_arr,
        diesel_arr=diesel_arr, ltr_stack=ltr_stack, trucks_stack=trucks_stack,
        avail_expected_arr=avail_expected_arr, reward_global_p95=reward_global_p95, market=market,
        ev_esperado=calcular_ev_esperado(market, distance_arr, diesel_arr),
    )


def main():
    from transfer_learning import checkpoint_tiene_post_head, load_checkpoint_compatible
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--num-nodes", type=int, default=10)
    ap.add_argument("--checkpoint", help="ruta del .pt (default: checkpoints[/RUN_TAG]/am_checkpoint_<N>nodes[_best].pt)")
    ap.add_argument("--best", action="store_true", help="usa el checkpoint *_best.pt")
    ap.add_argument("--restringido", action="store_true", help="valida con K libre y K = K_MAX (default: solo K libre)")
    ap.add_argument("--output-dir", help="carpeta de salida (default: results/<N>nodes[_<RUN_TAG>])")
    ap.add_argument("--quick", action="store_true", help="tamaños reducidos para una prueba rápida")
    a = ap.parse_args()

    configurar_k(a.restringido)
    cwd = os.path.dirname(os.path.abspath(__file__))
    path = a.checkpoint or os.path.join(cwd, "checkpoints", RUN_TAG, f"am_checkpoint_{a.num_nodes}nodes{'_best' if a.best else ''}.pt")
    if not checkpoint_tiene_post_head(path):
        raise SystemExit(f"{path} no trae post_value: reentrena con USE_POST_HEAD=True antes de validar el pricing.")
    agent, _ = load_checkpoint_compatible(path, a.num_nodes, AM_D_H, AM_N_HEADS, AM_N_LAYERS, AM_D_FF, verbose=False)
    agent.eval()
    tam = dict(n_worlds=60, n_mc_states=30, n_mc_rollouts=20, n_pos=60, n_no_tomadas=10) if a.quick else {}
    run_pricing_validation(agent, a.num_nodes, **cargar_datos(a.num_nodes), output_dir=a.output_dir, **tam)


if __name__ == "__main__":
    main()
