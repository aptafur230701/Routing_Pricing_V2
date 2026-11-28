"""
evaluar_b2p.py
==============
Compara B2 y B2P (B2 con pronósticos del índice de mercado, baselines.PoliticaB2P) sin entrenar nada:

  · mundos simulados: los mismos M mundos de evaluación de main.py (disenar_mundos con MARKET_SEED_EVAL)
    para cada N, K libre;
  · replay de los días reales de evaluación (solo con --replay), con los mismos casos de evaluar_replay.

Reporta el margen medio de cada política, la diferencia pareada B2P − B2 con IC95 por bootstrap sobre
mundos, los ms por decisión y la factibilidad. Escribe results/b2p/.

Uso
---
    python evaluar_b2p.py --nodos 10 35 --replay 10
"""

import argparse
import os
import time

import torch  # noqa: F401  (antes que pandas: en Windows, pandas primero rompe la carga de las DLL de torch)
import pandas as pd

from config import MARKET_SEED_EVAL, MARKET_M_EVAL_WORLDS, TRAIN_DAYS, REPLAY_MAX_CASOS
from evaluation import ContextoEval, _Casos, casos_replay, disenar_mundos, ejecutar_lote, mundos_eval
from market_process import MarketProcess, asegurar_parametros
from problem_data import load_matrices
from stats_analysis import ic_bootstrap_media

_CWD = os.path.dirname(os.path.abspath(__file__))


def _correr(ctx, casos, etiqueta) -> pd.DataFrame:
    """B2 y B2P sobre todos los casos a la vez (arnés en lote de la evaluación, K libre)."""
    filas = []
    for nombre in ("B2", "B2P"):
        t0 = time.time()
        res = ejecutar_lote(ctx.politica(nombre), ctx.env, casos.lote_mundos, casos.lote_nodos, casos.lote_k, ctx.dist)
        for c, m in enumerate(res):
            filas.append(dict(caso=c, metodo=nombre, margen=m["margen"], violacion=m["violacion"],
                              ms_decision=m["ms_decision"], ruta="-".join(map(str, m["ruta"]))))
        print(f"  {etiqueta}: {nombre} en {time.time() - t0:.0f} s", flush=True)
    return pd.DataFrame(filas)


def _contexto(N):
    market = MarketProcess(N)
    tm, _, _, dist, diesel, _, avail, p95 = load_matrices(N)
    return ContextoEval(None, market, None, tm, dist, diesel, N, avail, p95), market


def evaluar_simulado(N: int, M: int) -> pd.DataFrame:
    ctx, market = _contexto(N)
    dis = disenar_mundos(N, M, market, num_dias=TRAIN_DAYS, semilla=MARKET_SEED_EVAL)
    mundos = mundos_eval(market, dis["start_day"], TRAIN_DAYS - 1, MARKET_SEED_EVAL)
    casos = _Casos(dis["nodo"], dis["start_day"], dis["dow"], dis["start_day"], mundos, (None,))
    return _correr(ctx, casos, f"N={N} simulado")


def evaluar_replay(N: int) -> pd.DataFrame:
    ctx, market = _contexto(N)
    nodos, dias = casos_replay(N, market.num_dias_total - TRAIN_DAYS, REPLAY_MAX_CASOS)
    dia0 = TRAIN_DAYS + dias
    mundos = mundos_eval(market, dia0, market.num_dias_total - 1, MARKET_SEED_EVAL + 1, replay=True)
    casos = _Casos(nodos, dia0, market.dia_semana(dia0), dia0, mundos, (None,))
    return _correr(ctx, casos, f"N={N} replay")


def resumir(df: pd.DataFrame, modo: str, N: int) -> dict:
    ancho = df.pivot(index="caso", columns="metodo", values="margen")
    dif = (ancho["B2P"] - ancho["B2"]).to_numpy()
    lo, hi = ic_bootstrap_media(dif)
    ms = df.groupby("metodo")["ms_decision"].mean()
    viol = df.groupby("metodo")["violacion"].mean() * 100
    igual = (df[df.metodo == "B2"].set_index("caso")["ruta"] == df[df.metodo == "B2P"].set_index("caso")["ruta"]).mean()
    return {
        "modo": modo, "N": N, "casos": len(dif),
        "B2 margen medio": ancho["B2"].mean(), "B2P margen medio": ancho["B2P"].mean(),
        "dif pareada B2P − B2": dif.mean(), "IC95 inf": lo, "IC95 sup": hi,
        "% casos B2P > B2": 100 * (dif > 1e-9).mean(), "% casos B2P < B2": 100 * (dif < -1e-9).mean(),
        "% rutas iguales": 100 * igual,
        "B2 ms/decisión": ms["B2"], "B2P ms/decisión": ms["B2P"],
        "B2 violación %": viol["B2"], "B2P violación %": viol["B2P"],
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--nodos", type=int, nargs="+", default=[10, 35], help="tamaños para los mundos simulados")
    ap.add_argument("--replay", type=int, nargs="*", default=[10], help="tamaños para el replay")
    ap.add_argument("--mundos", type=int, default=MARKET_M_EVAL_WORLDS)
    ap.add_argument("--salida", default=os.path.join(_CWD, "results", "b2p"))
    args = ap.parse_args()
    asegurar_parametros()
    os.makedirs(args.salida, exist_ok=True)
    filas = []
    for N in args.nodos:
        df = evaluar_simulado(N, args.mundos)
        df.to_csv(os.path.join(args.salida, f"b2p_simulado_{N}nodos.csv"), index=False)
        filas.append(resumir(df, "simulado", N))
    for N in args.replay:
        df = evaluar_replay(N)
        df.to_csv(os.path.join(args.salida, f"b2p_replay_{N}nodos.csv"), index=False)
        filas.append(resumir(df, "replay", N))
    res = pd.DataFrame(filas)
    res.to_csv(os.path.join(args.salida, "b2p_resumen.csv"), index=False)
    with pd.option_context("display.width", 250, "display.max_columns", None, "display.float_format", "{:.3f}".format):
        print(res.T.to_string(header=False))


if __name__ == "__main__":
    main()
