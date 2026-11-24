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

La comparación B (DRL contra sus destinos con reglas fijas de precio) no se repite aquí: la hace la evaluación
de main.py, en los mundos de evaluación y en el replay (hoja "Comparación precio (B)").

Salidas: <output_dir>/pricing_val/tables/*.csv y <output_dir>/pricing_val/figures/*.png
(output_dir por defecto results/<N>nodes[_<RUN_TAG>]). main.py la corre al final de cada tamaño
(`validar_pricing`) y agrega al Excel las hojas "Precio vs estado" y "Sensibilidad κ".

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
    AM_D_H, AM_N_HEADS, AM_N_LAYERS, AM_D_FF, RUN_TAG,
)
import evaluation as ev
from market_process import MarketProcess, asegurar_parametros
from pricing import N_NIVELES, U_NIVELES_ARR, gana
from problem_data import load_matrices
from transfer_learning import load_checkpoint_compatible

KAPPAS = (0.5, 1.0, 2.0)
K_LEVELS = (None,)            # la tesis reporta solo K libre
N_QUINTILES = 5


class _Registro:
    """Envuelve una política del arnés y anota el estado de cada cotización (no cambia sus decisiones)."""

    def __init__(self, politica, env, market, mundo, K):
        self.p, self.env, self.market = politica, env, market
        self.mundo, self.K = mundo, K
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
                mundo=self.mundo, K=ev._etiqueta_k(self.K), nodo=i, destino=int(j), nivel=int(o),
                u=float(U_NIVELES_ARR[o]), ltr=float(self.market.ltr_hub[dia, i]), m=float(obs.m),
                frac_tiempo_restante=max(0.0, 1.0 - obs.tiempo / MAX_DURATION),
                k_restante=obs.k_restante if obs.k_restante is not None else np.nan,
                indice_origen=float(ind[i]), indice_destino=float(ind[j]),
                rechazos_previos=int(obs.n_rechazos),
                gano=bool(gana(U_NIVELES_ARR[o], env.u_comp_row[0, j])),
            ))
        return o


def _correr(ctx, env, dis, k_levels, metodos, registrar=False):
    """Margen por (K, mundo, método) y, si `registrar`, las cotizaciones del DRL."""
    pols = ctx.politicas_drl(env)
    filas, cot = [], []
    for K in k_levels:
        for m in range(len(dis["nodo"])):
            h, dow = int(dis["nodo"][m]), int(dis["dow"][m])
            for nombre in metodos:
                pol = pols[nombre]
                if registrar and nombre == "DRL":
                    pol = _Registro(pol, env, ctx.market, m, K)
                r = ev.ejecutar_politica(pol, env, h, int(dis["start_day"][m]), K, int(dis["world_seed"][m]),
                                         ctx.dist, ctx.diesel, dow)
                filas.append(dict(modo="simulado", K=ev._etiqueta_k(K), mundo=m, metodo=nombre,
                                  margen=r["margen"], violacion=r["violacion"], n_cotizaciones=r["n_cotizaciones"],
                                  n_ganadas=r["n_ganadas"], nivel_medio=np.mean(r["niveles"]) if r["niveles"]
                                  else np.nan, horas_rechazo=r["horas_rechazo"]))
                if isinstance(pol, _Registro):
                    cot.extend(pol.filas)
    return pd.DataFrame(filas), pd.DataFrame(cot)


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
                    M=MARKET_M_EVAL_WORLDS, excel_path=None) -> dict:
    """Corre (a) y (b) con `agent` y escribe <out>/pricing_val/{tables,figures}. Si `excel_path`, agrega al
    Excel de resultados las hojas "Precio vs estado" (Spearman y quintiles) y "Sensibilidad κ". Devuelve las tablas."""
    tab_dir, fig_dir = (os.path.join(out, "pricing_val", d) for d in ("tables", "figures"))
    os.makedirs(tab_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)
    dis = ev.disenar_mundos(n, M, market, semilla=PRICING_VAL_SEED)
    print(f"\n--- Validación del precio ({n} nodos, {M} mundos, semilla PRICING_VAL_SEED) ---")

    def contexto():
        c = ev.ContextoEval(agent, market, None, tm, dist, diesel, n, ltr, avail, p95)
        e = ev.make_sim_env(tm, rate[:TRAIN_DAYS], loads[:TRAIN_DAYS], dist, diesel, n, market)
        return c, e

    # (b) sensibilidad a κ, mismos mundos (el contexto se rehace porque las features usan m). Con el κ del
    # entrenamiento se registran además las cotizaciones del DRL para (a).
    sens, cot = [], None
    try:
        for kappa in KAPPAS:
            market.kappa = kappa
            c, e = contexto()
            d, registro = _correr(c, e, dis, K_LEVELS, ("DRL", "DRL-dest+miope"),
                                  registrar=(kappa == COMPETENCIA_KAPPA))
            if kappa == COMPETENCIA_KAPPA:
                cot = registro
            r = resumen_por_metodo(d)
            r.insert(0, "kappa", kappa)
            sens.append(r)
    finally:
        market.kappa = COMPETENCIA_KAPPA
    sens = pd.concat(sens, ignore_index=True)
    if cot is None:                       # COMPETENCIA_KAPPA fuera de KAPPAS: una corrida más para (a)
        c, e = contexto()
        _, cot = _correr(c, e, dis, K_LEVELS, ("DRL",), registrar=True)

    # (a) nivel cotizado por el DRL según el estado
    por_q, rho = niveles_por_estado(cot)

    tablas = {"cotizaciones_drl": cot, "nivel_por_quintil": por_q, "spearman_nivel": rho,
              "sensibilidad_kappa": sens}
    for nombre, t in tablas.items():
        t.to_csv(os.path.join(tab_dir, f"{nombre}.csv"), index=False)
    if len(cot):
        _figuras(cot, por_q, fig_dir)
    if excel_path is not None and os.path.exists(excel_path):
        with pd.ExcelWriter(excel_path, engine="openpyxl", mode="a", if_sheet_exists="replace") as w:
            rho.to_excel(w, sheet_name="Precio vs estado", index=False)
            por_q.to_excel(w, sheet_name="Precio vs estado", index=False, startrow=len(rho) + 3)
            sens.to_excel(w, sheet_name="Sensibilidad κ", index=False)

    with pd.option_context("display.width", 160, "display.max_columns", 20):
        print("\nNivel cotizado por el DRL vs estado (Spearman):\n", rho.to_string(index=False))
        print("\nSensibilidad a κ (K libre):\n",
              sens[["kappa", "metodo", "margen_medio", "tasa_aceptacion", "nivel_medio"]].to_string(index=False))
    print(f"\nTablas → {tab_dir}\nFiguras → {fig_dir}")
    return tablas


def main():
    ap = argparse.ArgumentParser(description="Validación de la decisión de precio del DRL.")
    ap.add_argument("--num-nodes", type=int, default=10)
    ap.add_argument("--checkpoint", help="default: checkpoints[/RUN_TAG]/am_checkpoint_<N>nodes.pt")
    ap.add_argument("--output-dir", help="default: results/<N>nodes[_<RUN_TAG>]")
    ap.add_argument("--quick", action="store_true", help="pocos mundos (prueba rápida)")
    args = ap.parse_args()
    n = args.num_nodes
    cwd = os.path.dirname(os.path.abspath(__file__))
    ckpt = args.checkpoint or os.path.join(cwd, "checkpoints", RUN_TAG, f"am_checkpoint_{n}nodes.pt")
    out = args.output_dir or os.path.join(cwd, "results", f"{n}nodes" + (f"_{RUN_TAG}" if RUN_TAG else ""))

    asegurar_parametros()
    tm, rate, loads, dist, diesel, ltr, avail, p95 = load_matrices(n)
    market = MarketProcess(n)
    agent, _ = load_checkpoint_compatible(ckpt, n, AM_D_H, AM_N_HEADS, AM_N_LAYERS, AM_D_FF, verbose=False)
    agent.eval()
    print(f"Checkpoint: {ckpt}")
    validar_pricing(agent, n, tm, rate, loads, dist, diesel, ltr, avail, p95, market, out,
                    M=28 if args.quick else MARKET_M_EVAL_WORLDS)


if __name__ == "__main__":
    main()
