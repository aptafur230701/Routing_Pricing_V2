"""
resumen_semillas.py
===================
Resume el DRL entrenado con varias semillas contra los baselines, mundo a mundo (mismos mundos y casos de
replay en todas las corridas: los baselines no dependen de la semilla del DRL).

Lee los Excel de resultados de cada semilla (hojas "Mundos simulados" y "Replay") y reporta, por modo:
  · margen medio del DRL de cada semilla y su promedio (el DRL promedio de un mundo es el promedio de las
    semillas en ese mundo);
  · margen medio de los baselines;
  · diferencia pareada DRL − baseline (cada semilla y el DRL promediado) con IC95 por bootstrap sobre mundos.

Uso
---
    python resumen_semillas.py --carpetas results/10nodes results/10nodes_c43 results/10nodes_c44 \
        --semillas 42 43 44
"""

import argparse
import os

import numpy as np
import pandas as pd

from stats_analysis import ic_bootstrap_media

_CWD = os.path.dirname(os.path.abspath(__file__))
_HOJAS = {"simulado": "Mundos simulados", "replay": "Replay"}
_BASELINES = ("B1", "B2", "B2P", "B3")


def cargar(carpetas, semillas, K="libre") -> dict:
    """{modo: DataFrame ancho (mundo × [DRL_s42, ..., B1, B2, B2P, B3])}; comprueba que los baselines coinciden."""
    out = {}
    for modo, hoja in _HOJAS.items():
        ancho, base = {}, None
        for carpeta, semilla in zip(carpetas, semillas):
            df = pd.read_excel(os.path.join(_CWD, carpeta, "DRL_Routing_Summary_AM.xlsx"), sheet_name=hoja)
            df = df[df["K"].astype(str) == K]
            p = df.pivot(index="mundo", columns="metodo", values="margen")
            ancho[f"DRL_s{semilla}"] = p["DRL"]
            b = p[list(_BASELINES)]
            if base is None:
                base = b
            elif not np.allclose(base.to_numpy(), b.to_numpy()):
                raise ValueError(f"{modo}: los baselines de {carpeta} no coinciden con los de la primera corrida")
        out[modo] = pd.concat([pd.DataFrame(ancho), base], axis=1)
    return out


def resumir(anchos: dict, semillas) -> tuple:
    medias, difs = [], []
    for modo, a in anchos.items():
        cols = [f"DRL_s{s}" for s in semillas]
        a = a.assign(**{"DRL_promedio": a[cols].mean(axis=1)})
        fila = {"modo": modo, "mundos": len(a)}
        for c in cols + ["DRL_promedio", *_BASELINES]:
            fila[c] = a[c].mean()
        fila["DRL_sd_entre_semillas"] = a[cols].mean().std(ddof=1)
        medias.append(fila)
        for c in cols + ["DRL_promedio"]:
            for b in reversed(_BASELINES):
                d = (a[c] - a[b]).to_numpy()
                lo, hi = ic_bootstrap_media(d)
                difs.append({"modo": modo, "DRL": c.replace("DRL_", ""), "baseline": b,
                             "dif_media_usd": d.mean(), "ic95_inf": lo, "ic95_sup": hi,
                             "% mundos DRL > baseline": 100 * (d > 1e-9).mean()})
    return pd.DataFrame(medias), pd.DataFrame(difs)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--carpetas", nargs="+", required=True)
    ap.add_argument("--semillas", type=int, nargs="+", required=True)
    ap.add_argument("--salida", default=os.path.join(_CWD, "results", "10nodes"))
    args = ap.parse_args()
    medias, difs = resumir(cargar(args.carpetas, args.semillas), args.semillas)
    medias.to_csv(os.path.join(args.salida, "semillas_margenes.csv"), index=False)
    difs.to_csv(os.path.join(args.salida, "semillas_diferencias_pareadas.csv"), index=False)
    with pd.option_context("display.width", 250, "display.max_columns", None, "display.float_format", "{:.1f}".format):
        print("Margen medio (USD)"); print(medias.to_string(index=False))
        print("\nDiferencia pareada DRL − baseline (USD), IC95 bootstrap sobre mundos"); print(difs.to_string(index=False))


if __name__ == "__main__":
    main()
