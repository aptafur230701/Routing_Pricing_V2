"""
validate_pricing.py
===================
Validación de la decisión de precio del DRL ya entrenado (especificacion_fase0_ruteo_pricing.md §8).
Corre en mundos simulados con salida en el período de PRUEBA (octubre, como la evaluación) y su propia semilla
(PRICING_VAL_SEED, distinta de entrenamiento y de evaluación), con el mismo arnés de evaluation.py (`ejecutar_lote`,
en la GPU), así que el margen es el mismo de la evaluación.

  (a) Nivel cotizado por el DRL según el estado. En cada cotización se registra el nivel elegido, el LTR del
      nodo de origen (y los competidores efectivos m), el momento del ciclo (fracción de tiempo restante y
      K restante) y el índice de mercado del origen y del destino. Se reporta el nivel medio y la tasa de
      aceptación por quintil de cada variable, con la correlación de Spearman. Lo esperado: un LTR más alto
      (menos competencia, m menor) lleva a cotizar más alto (Spearman positivo con el LTR).
  (b) Sensibilidad a κ: el modelo se entrenó con COMPETENCIA_KAPPA; se evalúa con κ ∈ KAPPAS (m = clip(1 +
      κ · LTR_REF / LTR, 1, M_MAX)), el DRL y DRL-dest+miope (referencia) en los mismos mundos. La regla miope
      conoce el κ del mercado (lee m del estado) y el DRL también lo observa como feature, pero no se entrenó
      con él. Las cotizaciones de (a) son las del DRL con κ = COMPETENCIA_KAPPA (la misma corrida).

  (c) Margen de mejora del precio a ruta fija (`headroom_precio`): el DRL contra sus mismos destinos con (i) la regla
      miope y (ii) el precio óptimo de la tabla de B3 (diagnóstico "DRL-dest+dp"; solo donde la tabla existe). Si
      DRL-dest+dp supera al DRL, el precio del DRL tiene margen; si no, la brecha con B3 está en el ruteo.

La comparación B (DRL contra sus destinos con reglas fijas de precio) no se repite aquí: la hace la evaluación
de main.py, en los mundos de prueba (hoja "Comparación", familia "precio").

Salidas: <output_dir>/pricing_val/tables/*.csv y <output_dir>/pricing_val/figures/*.png
  (a)/(b): cotizaciones_drl, nivel_por_quintil, spearman_nivel, sensibilidad_kappa
  (c)    : headroom_precio
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
    MAX_DURATION, PRICING_VAL_SEED, MARKET_M_EVAL_WORLDS, COMPETENCIA_KAPPA,
    AM_D_H, AM_N_HEADS, AM_N_LAYERS, AM_D_FF,
)
import evaluation as ev
from pricing import N_NIVELES, U_NIVELES_ARR
from market_process import MarketProcess, asegurar_parametros
from problem_data import load_matrices
from baselines.dp_relajada import cargar_o_calcular_tabla
from transfer_learning import load_checkpoint_compatible

KAPPAS = (0.5, 1.0, 2.0)
K_LEVELS = (None,)            # la tesis reporta solo K libre
N_QUINTILES = 5


def _cotizaciones(res: list, casos, modo: str, metodo: str) -> list:
    """Una fila por cotización (nivel elegido y estado del mercado en la decisión) de los resultados del lote `res`."""
    filas = []
    for n, K in enumerate(casos.k_levels):
        for m in range(casos.M):
            for p in res[n * casos.M + m]["pasos"]:
                if not p["accion"].startswith("cotiza"):
                    continue
                filas.append(dict(
                    modo=modo, metodo=metodo, dia_abs=p["dia_abs"], mundo=m, K=ev._etiqueta_k(K), nodo=p["desde"],
                    destino=p["hacia"], nivel=p["nivel"], u=p["u"], ltr=p["ltr"], m=p["m"],
                    frac_tiempo_restante=max(0.0, 1.0 - p["horas_transcurridas"] / MAX_DURATION),
                    k_restante=p["k_restante"] if p["k_restante"] is not None else np.nan,
                    indice_origen=p["indice_origen"], indice_destino=p["indice_destino"],
                    rechazos_previos=p["rechazos_previos"], gano=p["accion"] == "cotiza y gana",
                ))
    return filas


def _correr(ctx, casos, metodos, modo="simulado", registrar_metodos=()) -> tuple:
    """Margen por (K, mundo, método) y las cotizaciones de `registrar_metodos`, con el arnés de la evaluación."""
    filas, cot = [], []
    for nombre in metodos:
        res = ev.ejecutar_lote(ctx.politica(nombre), ctx.env, casos.lote_mundos, casos.lote_nodos, casos.lote_k, ctx.dist)
        for n, K in enumerate(casos.k_levels):
            for m in range(casos.M):
                r = res[n * casos.M + m]
                filas.append(dict(modo=modo, K=ev._etiqueta_k(K), mundo=m, dia_abs=int(casos.dia0[m]), metodo=nombre,
                                  margen=r["margen"], violacion=r["violacion"], n_cotizaciones=r["n_cotizaciones"],
                                  n_ganadas=r["n_ganadas"], nivel_medio=np.mean(r["niveles"]) if r["niveles"]
                                  else np.nan, horas_rechazo=r["horas_rechazo"]))
        if nombre in registrar_metodos:
            cot.extend(_cotizaciones(res, casos, modo, nombre))
    return pd.DataFrame(filas), pd.DataFrame(cot)


_COMPARACIONES_PRECIO = (("DRL", "DRL-dest+miope"), ("DRL-dest+dp", "DRL"), ("DRL-dest+dp", "DRL-dest+miope"))


def headroom_precio(marg: pd.DataFrame, B: int = 2000, semilla: int = 20261127) -> pd.DataFrame:
    """Diferencia pareada de margen (USD por ciclo) entre variantes de precio sobre las MISMAS rutas del DRL.

    marg : columnas modo, mundo, metodo y margen. IC 95 % bootstrap sobre mundos.
    Filas: A − B para (DRL, miope), (dp, DRL) y (dp, miope); las de "dp" solo existen si la tabla de B3 está."""
    rng = np.random.default_rng(semilla)
    filas = []
    for modo, g in marg.groupby("modo", sort=False):
        p = g.pivot_table(index="mundo", columns="metodo", values="margen")
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


def validar_pricing(agent, n, tm, dist, diesel, avail, p95, market, out,
                    M=MARKET_M_EVAL_WORLDS, quick=False, tabla=None) -> dict:
    """Corre (a) a (c) con `agent` y escribe <out>/pricing_val/{tables,figures}. Devuelve las tablas."""
    tab_dir, fig_dir = (os.path.join(out, "pricing_val", d) for d in ("tables", "figures"))
    os.makedirs(tab_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)
    dis = ev.disenar_mundos(n, M, market, periodo="test", semilla=PRICING_VAL_SEED)
    # Los mundos se sortean una vez: el competidor de cada carga es el mismo uniforme con cualquier κ (solo cambia m).
    mundos = ev.mundos_eval(market, dis["start_day"], dis["dia_max"], PRICING_VAL_SEED)
    casos = ev._Casos(dis["nodo"], dis["start_day"], dis["dow"], mundos, K_LEVELS)
    print(f"\n--- Validación del precio ({n} nodos, {M} mundos de prueba, semilla PRICING_VAL_SEED) ---")

    def contexto():
        return ev.ContextoEval(agent, market, tabla, tm, dist, diesel, n, avail, p95)

    # (b) sensibilidad a κ, mismos mundos (el contexto se rehace porque el entorno lee m con el κ del mercado). Con el
    # κ del entrenamiento se registran además las cotizaciones para (a).
    base = ("DRL", "DRL-dest+miope")
    todos = base + (("DRL-dest+dp",) if tabla is not None else ())   # dp: solo si la tabla de B3 existe
    if tabla is None:
        print("  (sin tabla de B3 a este tamaño: no se calcula DRL-dest+dp)")
    sens, cot, marg_sim = [], None, None
    try:
        for kappa in KAPPAS:
            market.kappa = kappa
            # dp solo con el κ de entrenamiento: la tabla de B3 se calculó con ese mercado
            metodos = todos if kappa == COMPETENCIA_KAPPA else base
            d, registro = _correr(contexto(), casos, metodos,
                                  registrar_metodos=metodos if kappa == COMPETENCIA_KAPPA else ())
            if kappa == COMPETENCIA_KAPPA:
                cot, marg_sim = registro, d
            r = resumen_por_metodo(d[d["metodo"].isin(base)])
            r.insert(0, "kappa", kappa)
            sens.append(r)
    finally:
        market.kappa = COMPETENCIA_KAPPA
    sens = pd.concat(sens, ignore_index=True)
    if cot is None:                       # COMPETENCIA_KAPPA fuera de KAPPAS: una corrida más para (a)
        marg_sim, cot = _correr(contexto(), casos, todos, registrar_metodos=todos)

    # (a) nivel cotizado por el DRL según el estado
    cot_sim = cot[cot["metodo"] == "DRL"]
    por_q, rho = niveles_por_estado(cot_sim)

    # (c) margen de mejora del precio a ruta fija: DRL vs miope vs precio de la tabla de B3, sobre las rutas del DRL
    head = headroom_precio(marg_sim)

    tablas = {"cotizaciones_drl": cot_sim.drop(columns=["modo", "metodo", "dia_abs"]),
              "nivel_por_quintil": por_q, "spearman_nivel": rho, "sensibilidad_kappa": sens,
              "headroom_precio": head}
    for nombre, t in tablas.items():
        t.to_csv(os.path.join(tab_dir, f"{nombre}.csv"), index=False)
    if len(cot_sim):
        _figuras(cot_sim, por_q, fig_dir)

    with pd.option_context("display.width", 160, "display.max_columns", 20):
        print("\nNivel cotizado por el DRL vs estado (Spearman):\n", rho.to_string(index=False))
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
    tm, _, _, dist, diesel, _, avail, p95 = load_matrices(n)
    market = MarketProcess(n)
    agent, _ = load_checkpoint_compatible(ckpt, n, AM_D_H, AM_N_HEADS, AM_N_LAYERS, AM_D_FF, verbose=False)
    agent.eval()
    print(f"Checkpoint: {ckpt}")
    # K libre (K = 0), como main.py con K_RESTRINGIDO = False: la firma del caché de la tabla incluye K, y con otro K se
    # recalcularía y se sobrescribiría la tabla que usa main.py.
    tabla = cargar_o_calcular_tabla(market, tm, dist, diesel, K=0)           # None si la DP no es tratable a este tamaño
    validar_pricing(agent, n, tm, dist, diesel, avail, p95, market, out, tabla=tabla,
                    M=28 if args.quick else MARKET_M_EVAL_WORLDS, quick=args.quick)


if __name__ == "__main__":
    main()
