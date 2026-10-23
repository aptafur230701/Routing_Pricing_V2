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

import numpy as np
import pandas as pd

from config import MARKET_SEED_EVAL, MARKET_M_EVAL_WORLDS, TRAIN_DAYS, REPLAY_MAX_CASOS
from baselines import PoliticaB2, PoliticaB2P
from evaluation import (
    disenar_mundos, ejecutar_politica, make_replay_env, make_sim_env, casos_replay, _UnHilo,
)
from market_process import MarketProcess, asegurar_parametros
from problem_data import load_matrices
from stats_analysis import ic_bootstrap_media

_CWD = os.path.dirname(os.path.abspath(__file__))


def _correr(politicas, caso, verbose_cada, etiqueta) -> pd.DataFrame:
    filas = []
    t0 = time.time()
    for c, args in enumerate(caso):
        for nombre, pol in politicas.items():
            m = args(pol)
            filas.append(dict(caso=c, metodo=nombre, margen=m["margen"], violacion=m["violacion"],
                              ms_decision=m["ms_decision"], ruta="-".join(map(str, m["ruta"]))))
        if (c + 1) % verbose_cada == 0:
            print(f"  {etiqueta}: {c + 1}/{len(caso)}  ({time.time() - t0:.0f} s)", flush=True)
    return pd.DataFrame(filas)


def evaluar_simulado(N: int, M: int) -> pd.DataFrame:
    market = MarketProcess(N)
    tm, rate, loads, dist, diesel, *_ = load_matrices(N)
    T = np.asarray(tm, float)
    politicas = {"B2": PoliticaB2(market, T, dist, diesel), "B2P": PoliticaB2P(market, T, dist, diesel)}
    env = make_sim_env(tm, rate[:TRAIN_DAYS], loads[:TRAIN_DAYS], dist, diesel, N, market)
    dis = disenar_mundos(N, M, market, num_dias=TRAIN_DAYS, semilla=MARKET_SEED_EVAL)
    casos = [
        (lambda pol, m=m: ejecutar_politica(
            pol, env, int(dis["nodo"][m]), int(dis["start_day"][m]), None, int(dis["world_seed"][m]),
            dist, diesel, int(dis["dow"][m])))
        for m in range(M)
    ]
    with _UnHilo():
        return _correr(politicas, casos, 100, f"N={N} simulado")


def evaluar_replay(N: int) -> pd.DataFrame:
    market = MarketProcess(N)
    tm, rate, loads, dist, diesel, *_ = load_matrices(N)
    T = np.asarray(tm, float)
    politicas = {"B2": PoliticaB2(market, T, dist, diesel), "B2P": PoliticaB2P(market, T, dist, diesel)}
    rate_e, loads_e = rate[TRAIN_DAYS:], loads[TRAIN_DAYS:]
    env = make_replay_env(tm, rate_e, loads_e, dist, diesel, N, day_offset=TRAIN_DAYS, market=market)
    nodos, dias = casos_replay(N, rate_e.shape[0], REPLAY_MAX_CASOS)
    casos = [
        (lambda pol, n=int(n), d=int(d): ejecutar_politica(
            pol, env, n, d, None, None, dist, diesel, int(market.dia_semana(TRAIN_DAYS + d)), loads_e, rate_e))
        for n, d in zip(nodos, dias)
    ]
    with _UnHilo():
        return _correr(politicas, casos, 100, f"N={N} replay")


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
