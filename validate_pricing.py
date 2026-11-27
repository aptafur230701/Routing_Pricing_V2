"""
validate_pricing.py
===================
Validación de la decisión de precio del DRL ya entrenado (especificacion_fase0_ruteo_pricing.md §8).
Corre en mundos simulados con su propia semilla (PRICING_VAL_SEED, distinta de entrenamiento y de evaluación)
y con el mismo arnés de evaluation.py (`ejecutar_politica`), así que el margen es el mismo de la evaluación.

  (a) Nivel cotizado por el DRL según el estado. En cada cotización se registra el nivel elegido, el LTR del
      nodo de origen (y los competidores efectivos m), el momento del ciclo (fracción de tiempo restante y
      K restante) y el índice de mercado del origen y del destino. Se reporta el nivel medio y la tasa de
      aceptación por quintil de cada variable, con la correlación de Spearman. Lo esperado: un LTR más alto
      (menos competencia, m menor) lleva a cotizar más alto (Spearman positivo con el LTR).
  (b) Sensibilidad a κ: el modelo se entrenó con COMPETENCIA_KAPPA; se evalúa con κ ∈ KAPPAS (m = clip(1 +
      κ · LTR_REF / LTR, 1, M_MAX)), el DRL y DRL-dest+miope (referencia) en los mismos mundos. La regla miope
      conoce el κ del mercado (lee m del estado) y el DRL también lo observa como feature, pero no se entrenó
      con él. Las cotizaciones de (a) son las del DRL con κ = COMPETENCIA_KAPPA (la misma corrida).

  (c) Mismo registro en el REPLAY de los días reales de prueba (los mismos casos que la evaluación), para el DRL y para
      DRL-dest+miope, y una tabla que compara el régimen de mercado visto en mundos simulados y en replay (LTR, m,
      índice, nivel y aceptación). Sirve para ver si el precio del DRL se comporta igual fuera de muestra.

  (d) Margen de mejora del precio a ruta fija (`headroom_precio`): el DRL contra sus mismos destinos con (i) la regla
      miope y (ii) el precio óptimo de la tabla de B3 (diagnóstico "DRL-dest+dp"; solo donde la tabla existe). Si
      DRL-dest+dp supera al DRL, el precio del DRL tiene margen; si no, la brecha con B3 está en el ruteo.

La comparación B (DRL contra sus destinos con reglas fijas de precio) no se repite aquí: la hace la evaluación
de main.py, en los mundos de evaluación y en el replay (hoja "Comparación", familia "precio").

Salidas: <output_dir>/pricing_val/tables/*.csv y <output_dir>/pricing_val/figures/*.png
  (a)/(b): cotizaciones_drl, nivel_por_quintil, spearman_nivel, sensibilidad_kappa (mundos simulados, como siempre)
  (c)    : cotizaciones_comparacion (columnas modo y metodo), nivel_por_quintil_replay, spearman_nivel_replay,
           regimen_sim_vs_replay
(output_dir por defecto results/<N>nodes). main.py la corre al final de cada tamaño
(`validar_pricing`); no escribe en el Excel de resultados.

Uso aparte (sobre un checkpoint ya entrenado)
---------------------------------------------
    python validate_pricing.py --num-nodes 10 [--checkpoint RUTA] [--quick]
"""

import argparse
import os

import torch  # noqa: F401  (antes que pandas: en algunos Windows importar pandas primero rompe c10.dll de torch)
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import (
    TRAIN_DAYS, MAX_DURATION, PRICING_VAL_SEED, MARKET_M_EVAL_WORLDS, COMPETENCIA_KAPPA,
    AM_D_H, AM_N_HEADS, AM_N_LAYERS, AM_D_FF,
)
import evaluation as ev
from market_process import MarketProcess, asegurar_parametros
from pricing import N_NIVELES, U_NIVELES_ARR, gana
from problem_data import load_matrices
from baselines.dp_relajada import cargar_o_calcular_tabla
from transfer_learning import load_checkpoint_compatible

KAPPAS = (0.5, 1.0, 2.0)
K_LEVELS = (None,)            # la tesis reporta solo K libre
N_QUINTILES = 5


class _Registro:
    """Envuelve una política del arnés y anota el estado de cada cotización (no cambia sus decisiones)."""

    def __init__(self, politica, env, market, mundo, K, modo="simulado", metodo="DRL"):
        self.p, self.env, self.market = politica, env, market
        self.mundo, self.K = mundo, K
        self.modo, self.metodo = modo, metodo
        self.filas = []
        self.nombre = getattr(politica, "nombre", "politica")

    def actuar(self, obs) -> int:
        return self.p.actuar(obs)

    def cotizar(self, obs, j) -> int:
        o = self.p.cotizar(obs, j)
        if o < N_NIVELES:
            env, i = self.env, obs.nodo
            dia = int(env.current_day_idx[0])
            ind = env.market_index[0]
            self.filas.append(dict(
                modo=self.modo, metodo=self.metodo, dia_abs=dia,
                mundo=self.mundo, K=ev._etiqueta_k(self.K), nodo=i, destino=int(j), nivel=int(o),
                u=float(U_NIVELES_ARR[o]), ltr=float(self.market.ltr_hub[dia, i]), m=float(obs.m),
                frac_tiempo_restante=max(0.0, 1.0 - obs.tiempo / MAX_DURATION),
                k_restante=obs.k_restante if obs.k_restante is not None else np.nan,
                indice_origen=float(ind[i]), indice_destino=float(ind[j]),
                rechazos_previos=int(obs.n_rechazos),
                gano=bool(gana(U_NIVELES_ARR[o], env.u_comp_row[0, j])),
            ))
        return o


def _politicas(ctx, env) -> dict:
    """Las del DRL (comparación B) y, si la tabla de B3 existe, "DRL-dest+dp" (su destino con el precio de la tabla)."""
    pols = ctx.politicas_drl(env)
    dp = ctx.politica_dp(env)
    if dp is not None:
        pols["DRL-dest+dp"] = dp
    return pols


def _correr(ctx, env, dis, k_levels, metodos, registrar=False, registrar_metodos=("DRL",)):
    """Margen por (K, mundo, método) y, si `registrar`, las cotizaciones de `registrar_metodos`."""
    pols = _politicas(ctx, env)
    filas, cot = [], []
    for K in k_levels:
        for m in range(len(dis["nodo"])):
            h, dow = int(dis["nodo"][m]), int(dis["dow"][m])
            for nombre in metodos:
                pol = pols[nombre]
                if registrar and nombre in registrar_metodos:
                    pol = _Registro(pol, env, ctx.market, m, K, "simulado", nombre)
                r = ev.ejecutar_politica(pol, env, h, int(dis["start_day"][m]), K, int(dis["world_seed"][m]),
                                         ctx.dist, ctx.diesel, dow)
                filas.append(dict(modo="simulado", K=ev._etiqueta_k(K), mundo=m, metodo=nombre,
                                  margen=r["margen"], violacion=r["violacion"], n_cotizaciones=r["n_cotizaciones"],
                                  n_ganadas=r["n_ganadas"], nivel_medio=np.mean(r["niveles"]) if r["niveles"]
                                  else np.nan, horas_rechazo=r["horas_rechazo"]))
                if isinstance(pol, _Registro):
                    cot.extend(pol.filas)
    return pd.DataFrame(filas), pd.DataFrame(cot)


def _correr_replay(ctx, env, nodos, dias, semillas, k_levels, metodos, market):
    """(cotizaciones, márgenes) de `metodos` en el replay de los días reales de prueba (los mismos casos de la evaluación)."""
    pols = _politicas(ctx, env)
    cot, marg = [], []
    for K in k_levels:
        for c, (nodo, dia, ws) in enumerate(zip(nodos, dias, semillas)):
            nodo, dia, ws = int(nodo), int(dia), int(ws)
            dow = int(market.dia_semana(TRAIN_DAYS + dia))
            for nombre in metodos:
                pol = _Registro(pols[nombre], env, market, c, K, "replay", nombre)
                r = ev.ejecutar_politica(pol, env, nodo, dia, K, ws, ctx.dist, ctx.diesel, dow)
                cot.extend(pol.filas)
                marg.append(dict(modo="replay", K=ev._etiqueta_k(K), mundo=c, dia_abs=TRAIN_DAYS + dia, metodo=nombre,
                                 margen=r["margen"]))
    return pd.DataFrame(cot), pd.DataFrame(marg)


_COMPARACIONES_PRECIO = (("DRL", "DRL-dest+miope"), ("DRL-dest+dp", "DRL"), ("DRL-dest+dp", "DRL-dest+miope"))


def headroom_precio(marg: pd.DataFrame, B: int = 2000, semilla: int = 20261127) -> pd.DataFrame:
    """Diferencia pareada de margen (USD por ciclo) entre variantes de precio sobre las MISMAS rutas del DRL.

    marg : columnas modo, mundo, metodo, margen y, en replay, dia_abs. IC 95 % bootstrap; en replay se remuestrean los
    DÍAS (los escenarios de un mismo día comparten mercado y no son independientes), en simulado cada mundo.
    Filas: A − B para (DRL, miope), (dp, DRL) y (dp, miope); las de "dp" solo existen si la tabla de B3 está."""
    rng = np.random.default_rng(semilla)
    filas = []
    for modo, g in marg.groupby("modo", sort=False):
        p = g.pivot_table(index="mundo", columns="metodo", values="margen")
        if modo == "replay" and "dia_abs" in g:
            cluster = g.drop_duplicates("mundo").set_index("mundo")["dia_abs"].reindex(p.index)
        else:
            cluster = pd.Series(p.index.to_numpy(), index=p.index)
        for a, b in _COMPARACIONES_PRECIO:
            if a not in p or b not in p:
                continue
            dif = (p[a] - p[b]).dropna()
            s = dif.groupby(cluster.loc[dif.index].to_numpy()).agg(["sum", "count"])
            idx = rng.integers(0, len(s), size=(B, len(s)))
            est = s["sum"].to_numpy()[idx].sum(1) / s["count"].to_numpy()[idx].sum(1)
            filas.append(dict(modo=modo, comparacion=f"{a} − {b}", n=len(dif), n_clusters=len(s),
                              margen_a=float(p.loc[dif.index, a].mean()), margen_b=float(p.loc[dif.index, b].mean()),
                              dif_media_usd=float(dif.mean()), ic95_inf=float(np.percentile(est, 2.5)),
                              ic95_sup=float(np.percentile(est, 97.5)), frac_a_mejor=float((dif > 0).mean())))
    return pd.DataFrame(filas)


def regimen_sim_vs_replay(cot: pd.DataFrame) -> pd.DataFrame:
    """Régimen de mercado y conducta de precio por (modo, método): LTR, m, índice, nivel cotizado y aceptación."""
    g = cot.groupby(["modo", "metodo"], sort=False)
    return g.agg(n_cotizaciones=("nivel", "size"), nivel_medio=("nivel", "mean"), tasa_aceptacion=("gano", "mean"),
                 ltr_mediana=("ltr", "median"), m_medio=("m", "mean"), m_mediana=("m", "median"),
                 indice_origen_medio=("indice_origen", "mean"), indice_destino_medio=("indice_destino", "mean"),
                 frac_tiempo_restante_media=("frac_tiempo_restante", "mean")).reset_index()


def niveles_por_estado(cot: pd.DataFrame) -> tuple:
    """(tabla por quintil, tabla de Spearman) del nivel cotizado según cada variable de estado."""
    variables = ("ltr", "m", "frac_tiempo_restante", "k_restante", "indice_origen", "indice_destino")
    por_q, rho = [], []
    for v in variables:
        d = cot.dropna(subset=[v])
        if d[v].nunique() < 2:
            continue
        q = pd.qcut(d[v].rank(method="first"), min(N_QUINTILES, d[v].nunique()), labels=False)
        for b, g in d.groupby(q):
            por_q.append(dict(variable=v, quintil=int(b) + 1, desde=g[v].min(), hasta=g[v].max(), n=len(g),
                              nivel_medio=g["nivel"].mean(), tasa_aceptacion=g["gano"].mean()))
        r, p = spearmanr(d[v], d["nivel"])
        rho.append(dict(variable=v, n=len(d), spearman_nivel=float(r), p=float(p)))
    return pd.DataFrame(por_q), pd.DataFrame(rho)


def resumen_por_metodo(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby(["K", "metodo"], sort=False)
    out = g.agg(n_mundos=("margen", "size"), margen_medio=("margen", "mean"),
                violaciones=("violacion", "mean"), nivel_medio=("nivel_medio", "mean"),
                horas_rechazo=("horas_rechazo", "mean"),
                cot=("n_cotizaciones", "sum"), gan=("n_ganadas", "sum")).reset_index()
    out["tasa_aceptacion"] = out["gan"] / out["cot"].where(out["cot"] > 0)
    return out.drop(columns=["cot", "gan"])


def _figuras(cot, por_q, fig_dir):
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.hist(cot["nivel"], bins=np.arange(N_NIVELES + 1) - 0.5, rwidth=0.85)
    ax.set_xticks(range(N_NIVELES), [f"{u:.1f}" for u in U_NIVELES_ARR])
    ax.set_xlabel("nivel cotizado (percentil u)")
    ax.set_ylabel("cotizaciones")
    ax.set_title("Niveles cotizados por el DRL")
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "histograma_niveles.png"), dpi=150)
    plt.close(fig)

    vs = [v for v in ("ltr", "frac_tiempo_restante", "k_restante", "indice_destino") if v in set(por_q["variable"])]
    if not vs:
        return
    fig, axes = plt.subplots(1, len(vs), figsize=(3.2 * len(vs), 3), sharey=True)
    for ax, v in zip(np.atleast_1d(axes), vs):
        g = por_q[por_q["variable"] == v]
        ax.plot(g["quintil"], g["nivel_medio"], marker="o")
        ax.set_xlabel(f"quintil de {v}")
        ax.set_xticks(g["quintil"])
    np.atleast_1d(axes)[0].set_ylabel("nivel medio")
    fig.suptitle("Nivel cotizado por el DRL según el estado")
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "nivel_por_estado.png"), dpi=150)
    plt.close(fig)


def validar_pricing(agent, n, tm, rate, loads, dist, diesel, ltr, avail, p95, market, out,
                    M=MARKET_M_EVAL_WORLDS, quick=False, tabla=None) -> dict:
    """Corre (a) y (b) con `agent` y escribe <out>/pricing_val/{tables,figures}. Devuelve las tablas."""
    tab_dir, fig_dir = (os.path.join(out, "pricing_val", d) for d in ("tables", "figures"))
    os.makedirs(tab_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)
    dis = ev.disenar_mundos(n, M, market, semilla=PRICING_VAL_SEED)
    print(f"\n--- Validación del precio ({n} nodos, {M} mundos, semilla PRICING_VAL_SEED) ---")

    def contexto():
        c = ev.ContextoEval(agent, market, tabla, tm, dist, diesel, n, ltr, avail, p95)
        e = ev.make_sim_env(tm, rate[:TRAIN_DAYS], loads[:TRAIN_DAYS], dist, diesel, n, market)
        return c, e

    # (b) sensibilidad a κ, mismos mundos (el contexto se rehace porque las features usan m). Con el κ del
    # entrenamiento se registran además las cotizaciones del DRL para (a).
    base = ("DRL", "DRL-dest+miope")
    todos = base + (("DRL-dest+dp",) if tabla is not None else ())   # dp: solo si la tabla de B3 existe
    if tabla is None:
        print("  (sin tabla de UB-DP a este tamaño: no se calcula DRL-dest+dp)")
    sens, cot, marg_sim = [], None, None
    try:
        for kappa in KAPPAS:
            market.kappa = kappa
            c, e = contexto()
            # dp solo con el κ de entrenamiento: la tabla de B3 se calculó con ese mercado
            metodos = todos if kappa == COMPETENCIA_KAPPA else base
            d, registro = _correr(c, e, dis, K_LEVELS, metodos,
                                  registrar=(kappa == COMPETENCIA_KAPPA), registrar_metodos=metodos)
            if kappa == COMPETENCIA_KAPPA:
                cot, marg_sim = registro, d
            r = resumen_por_metodo(d[d["metodo"].isin(base)])
            r.insert(0, "kappa", kappa)
            sens.append(r)
    finally:
        market.kappa = COMPETENCIA_KAPPA
    sens = pd.concat(sens, ignore_index=True)
    if cot is None:                       # COMPETENCIA_KAPPA fuera de KAPPAS: una corrida más para (a)
        c, e = contexto()
        marg_sim, cot = _correr(c, e, dis, K_LEVELS, todos, registrar=True, registrar_metodos=todos)

    # (a) nivel cotizado por el DRL según el estado (mundos simulados)
    cot_sim = cot[cot["metodo"] == "DRL"]
    por_q, rho = niveles_por_estado(cot_sim)

    # (c) lo mismo en el replay de los días reales de prueba, con los casos de la evaluación
    rate_ev, loads_ev = rate[TRAIN_DAYS:], loads[TRAIN_DAYS:]
    cot_rep = pd.DataFrame()
    if rate_ev.shape[0] > 0:
        c_rep = ev.ContextoEval(agent, market, tabla, tm, dist, diesel, n, ltr[:, TRAIN_DAYS:], avail, p95)
        e_rep = ev.make_replay_env(tm, rate_ev, loads_ev, dist, diesel, n, day_offset=TRAIN_DAYS, market=market)
        nodos, dias, semillas = ev.casos_replay(n, rate_ev.shape[0])
        if quick:
            idx = np.linspace(0, len(nodos) - 1, 28).astype(int)       # repartido: los casos vienen ordenados por nodo
            nodos, dias, semillas = nodos[idx], dias[idx], semillas[idx]
        cot_rep, marg_rep = _correr_replay(c_rep, e_rep, nodos, dias, semillas, K_LEVELS, todos, market)
    else:
        marg_rep = pd.DataFrame()
    cot_todo = pd.concat([cot, cot_rep], ignore_index=True)
    reg = regimen_sim_vs_replay(cot_todo)
    if len(cot_rep):
        por_q_rep, rho_rep = niveles_por_estado(cot_rep[cot_rep["metodo"] == "DRL"])
    else:
        por_q_rep, rho_rep = pd.DataFrame(), pd.DataFrame()

    # (d) margen de mejora del precio a ruta fija: DRL vs miope vs precio de la tabla de B3, sobre las rutas del DRL
    head = headroom_precio(pd.concat([marg_sim, marg_rep], ignore_index=True))

    tablas = {"cotizaciones_drl": cot_sim.drop(columns=["modo", "metodo", "dia_abs"]),
              "nivel_por_quintil": por_q, "spearman_nivel": rho, "sensibilidad_kappa": sens,
              "cotizaciones_comparacion": cot_todo, "nivel_por_quintil_replay": por_q_rep,
              "spearman_nivel_replay": rho_rep, "regimen_sim_vs_replay": reg, "headroom_precio": head}
    for nombre, t in tablas.items():
        t.to_csv(os.path.join(tab_dir, f"{nombre}.csv"), index=False)
    if len(cot_sim):
        _figuras(cot_sim, por_q, fig_dir)

    with pd.option_context("display.width", 160, "display.max_columns", 20):
        print("\nNivel cotizado por el DRL vs estado (Spearman):\n", rho.to_string(index=False))
        print("\nRégimen y precio, simulado vs replay:\n", reg.to_string(index=False))
        print("\nMargen de mejora del precio a ruta fija (USD/ciclo; A − B):\n", head.to_string(index=False))
        print("\nSensibilidad a κ (K libre):\n",
              sens[["kappa", "metodo", "margen_medio", "tasa_aceptacion", "nivel_medio"]].to_string(index=False))
    print(f"\nTablas → {tab_dir}\nFiguras → {fig_dir}")
    return tablas


def main():
    ap = argparse.ArgumentParser(description="Validación de la decisión de precio del DRL.")
    ap.add_argument("--num-nodes", type=int, default=10)
    ap.add_argument("--checkpoint", help="default: checkpoints/am_checkpoint_<N>nodes.pt")
    ap.add_argument("--output-dir", help="default: results/<N>nodes")
    ap.add_argument("--quick", action="store_true", help="pocos mundos (prueba rápida)")
    args = ap.parse_args()
    n = args.num_nodes
    cwd = os.path.dirname(os.path.abspath(__file__))
    ckpt = args.checkpoint or os.path.join(cwd, "checkpoints", f"am_checkpoint_{n}nodes.pt")
    out = args.output_dir or os.path.join(cwd, "results", f"{n}nodes")

    asegurar_parametros()
    tm, rate, loads, dist, diesel, ltr, avail, p95 = load_matrices(n)
    market = MarketProcess(n)
    agent, _ = load_checkpoint_compatible(ckpt, n, AM_D_H, AM_N_HEADS, AM_N_LAYERS, AM_D_FF, verbose=False)
    agent.eval()
    print(f"Checkpoint: {ckpt}")
    # K libre (K = 0), como main.py con K_RESTRINGIDO = False: la firma del caché de la tabla incluye K, y con otro K se
    # recalcularía y se sobrescribiría la tabla que usa main.py.
    tabla = cargar_o_calcular_tabla(market, tm, dist, diesel, K=0)           # None si UB-DP no es tratable a este tamaño
    validar_pricing(agent, n, tm, rate, loads, dist, diesel, ltr, avail, p95, market, out, tabla=tabla,
                    M=28 if args.quick else MARKET_M_EVAL_WORLDS, quick=args.quick)


if __name__ == "__main__":
    main()
