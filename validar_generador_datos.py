"""
validar_generador_datos.py
==========================
Validación estadística del generador de realizaciones diarias (metodologia_realizaciones_mercado_reefer.md)
con los 194 hubs y las 37,442 lanes. Produce las tablas y figuras de validacion_generador_datos.md.

  A. Disponibilidad de cargas (binomial negativa / Poisson) — prueba de Kolmogorov-Smirnov de dos muestras,
     lane por lane: las cargas reales por día de la lane contra las cargas que sortea el generador
     (m del Paso 4b, r del grupo de volumen). Dentro de la muestra (jun-ago) y fuera de ella (oct).
  B. Rate ($/milla) — la media del precio sorteado (calculada exacta, sin Monte Carlo) contra la media
     real del día, que NO se usa para construir los sorteos. Mejores y peores lanes.
  C. Corrimiento δ — normalidad del cambio diario del factor nacional (KS con parámetros estimados,
     versión Lilliefors por simulación).

Uso:  python validar_generador_datos.py        (salida en results/market_validation/validacion_metodologia)
"""

import os

import torch  # noqa: F401  (antes que numpy/pandas/scipy: en este Windows el orden contrario rompe c10.dll)
import numpy as np
import pandas as pd
from scipy import stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import TRAIN_DAYS, TEST_START
from fit_market_process import estimar_m, factor_nacional, params_path
from problem_data import cargar_dias_mercado

_CWD = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(_CWD, "results", "market_validation", "validacion_metodologia")
TAB, FIG = os.path.join(OUT, "tables"), os.path.join(OUT, "figures")
SEED = 20261126
SORTEOS_POR_DIA = 20        # sorteos del generador por lane-día para armar la muestra sintética de una lane
MIN_DIAS_CASOS = 30         # SOLO para elegir qué lanes se dibujan como mejor/mediana/peor: días con cargas (una lane vacía da D = 0)
ALFA = 0.05
_PESOS_TRAMO = np.array([0.25, 0.25, 0.25, 0.15, 0.10])   # masa entre min-q1, q1-q2, q2-q3, q3-p90, p90-max


# ───────────────────────── A. cargas ─────────────────────────

def sortear_cargas(m, r, S, rng):
    """(D, S) sorteos del generador (Paso 4b) para una lane con medias diarias m (D,) y dispersión r:
    NegBin(m, r) si m >= 1, Poisson(m) si no."""
    mm = np.maximum(m, 1e-12)[:, None]
    forma = (len(m), S)
    return np.where(m[:, None] < 1.0, rng.poisson(mm, forma), rng.negative_binomial(r, r / (r + mm), forma))


def ks_por_lane(y, m, r_lane, rng):
    """KS de dos muestras por lane: cargas reales de sus días contra las sorteadas por el generador.
    Se prueban TODAS las lanes (fuera de la diagonal); una lane sin cargas reales ni generadas da D = 0."""
    N = y.shape[1]
    filas = []
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            sim = sortear_cargas(m[:, i, j], r_lane[i, j], SORTEOS_POR_DIA, rng)
            res = stats.ks_2samp(y[:, i, j], sim.ravel())
            filas.append((i, j, y.shape[0], int((y[:, i, j] > 0).sum()), float(m[:, i, j].mean()),
                          float(res.statistic), float(res.pvalue)))
    return pd.DataFrame(filas, columns=["i", "j", "n_dias", "dias_con_cargas", "m_medio", "D", "p_valor"])


def ks_por_grupo(y, m, r_lane, grupo, r_grupo, rng):
    """KS de dos muestras por grupo de volumen: todas las cargas reales del grupo contra un sorteo del
    generador por cada lane-día. Con cientos de miles de datos el p-valor es ~0 con cualquier desvío: se lee D."""
    filas = []
    for h in range(len(r_grupo)):
        sel = np.broadcast_to(grupo == h, y.shape)
        real = y[sel]
        mm = np.maximum(m[sel], 1e-12)
        rr = np.broadcast_to(r_lane, y.shape)[sel]
        sim = np.where(m[sel] < 1.0, rng.poisson(mm), rng.negative_binomial(rr, rr / (rr + mm)))
        res = stats.ks_2samp(real, sim)
        filas.append(dict(grupo=h, r=r_grupo[h], n=int(sel.sum()), D=float(res.statistic)))
    return pd.DataFrame(filas)


def metricas_error_cargas(y, m, r_lane):
    """Por lane (fuera de la diagonal), lo real contra lo que implica el generador, en forma EXACTA (sin Monte Carlo):
    media = promedio de m; varianza = promedio de las varianzas diarias (m + m²/r, o m si m < 1) + varianza de m entre
    días (ley de la varianza total); fracción de días sin cargas = promedio de P(0) diaria (NegBin: (r/(r+m))^r; Poisson: e^-m)."""
    N = y.shape[1]
    rr = r_lane[None]
    var_cond = np.where(m < 1.0, m, m + m ** 2 / rr)
    p0_dia = np.where(m < 1.0, np.exp(-m), (rr / (rr + m)) ** rr)
    ii, jj = np.nonzero(~np.eye(N, dtype=bool))
    cols = dict(media_real=y.mean(0), media_gen=m.mean(0), std_real=y.std(0),
                std_gen=np.sqrt(var_cond.mean(0) + m.var(0)), ceros_real=(y == 0).mean(0), ceros_gen=p0_dia.mean(0))
    return pd.DataFrame({"i": ii, "j": jj, **{k: v[ii, jj] for k, v in cols.items()}})


def resumen_metricas_cargas(t):
    """Percentiles del error relativo (generado − real)/real por lane. Los errores relativos solo existen si el
    valor real es > 0 (lanes sin cargas, o sin variación, quedan fuera de ese error y se cuentan aparte)."""
    c, cs = t[t.media_real > 0], t[t.std_real > 0]
    e_media = (c.media_gen - c.media_real) / c.media_real
    e_std = (cs.std_gen - cs.std_real) / cs.std_real
    d_ceros = 100 * (t.ceros_gen - t.ceros_real)
    return dict(
        lanes=len(t), lanes_sin_cargas=int((t.media_real == 0).sum()),
        media_real_por_dia=t.media_real.mean(), media_generada_por_dia=t.media_gen.mean(),
        error_ponderado_media_pct=100 * (t.media_gen - t.media_real).abs().sum() / t.media_real.sum(),
        media_error_abs_mediana_pct=100 * e_media.abs().median(), media_error_abs_p90_pct=100 * e_media.abs().quantile(.9),
        media_sesgo_mediana_pct=100 * e_media.median(),
        std_error_abs_mediana_pct=100 * e_std.abs().median(), std_error_abs_p90_pct=100 * e_std.abs().quantile(.9),
        std_sesgo_mediana_pct=100 * e_std.median(),
        ceros_real_pct=100 * t.ceros_real.mean(), ceros_gen_pct=100 * t.ceros_gen.mean(),
        ceros_dif_abs_mediana_pp=d_ceros.abs().median(), ceros_dif_abs_p90_pp=d_ceros.abs().quantile(.9))


def figura_media_desv(t, path):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4))
    for ax, (real, gen, nombre) in zip(axes, (("media_real", "media_gen", "Media de cargas por día"),
                                              ("std_real", "std_gen", "Desviación estándar de cargas por día"))):
        s = t[(t[real] > 0) & (t[gen] > 0)]
        hb = ax.hexbin(s[real], s[gen], gridsize=60, xscale="log", yscale="log", bins="log", cmap="Oranges", mincnt=1)
        lim = [min(s[real].min(), s[gen].min()), max(s[real].max(), s[gen].max())]
        ax.plot(lim, lim, color="black", ls="--", lw=1, label="real = generado")
        ax.set(xlabel=f"{nombre}: real", ylabel=f"{nombre}: generador", title=nombre)
        ax.legend(fontsize=8, loc="upper left")
    fig.suptitle("Una lane por punto (jun–ago). Colores más oscuros = más lanes", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def figura_ecdf_grupos(y, m, r_lane, grupo, r_grupo, path, rng):
    fig, axes = plt.subplots(2, 5, figsize=(14, 5.2))
    for h, ax in enumerate(axes.ravel()):
        sel = np.broadcast_to(grupo == h, y.shape)
        real = y[sel]
        mm = np.maximum(m[sel], 1e-12)
        rr = np.broadcast_to(r_lane, y.shape)[sel]
        sim = np.where(m[sel] < 1.0, rng.poisson(mm), rng.negative_binomial(rr, rr / (rr + mm)))
        tope = np.percentile(real, 99)
        x = np.arange(0, tope + 1)
        ax.step(x, [(real <= v).mean() for v in x], where="post", color="black", label="real")
        ax.step(x, [(sim <= v).mean() for v in x], where="post", color="tab:orange", label="generador")
        ax.set_title(f"grupo {h}  (r = {r_grupo[h]:.2f})", fontsize=9)
        ax.set_xlabel("cargas por día", fontsize=8)
        ax.set_ylim(0, 1.02)
    axes[0, 0].legend(fontsize=8)
    axes[0, 0].set_ylabel("proporción acumulada")
    axes[1, 0].set_ylabel("proporción acumulada")
    fig.suptitle("Distribución acumulada de las cargas por día: real contra generador, por grupo de volumen (jun–ago)",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def figura_lanes_cargas(y, m, r_lane, casos, hubs_txt, path, titulo):
    """casos: lista de (etiqueta, i, j, D, p). Fila superior: cargas reales por día contra la media m y el rango
    5–95 % del generador. Fila inferior: distribución acumulada real contra la del generador (lo que compara el KS)."""
    n = len(casos)
    fig, axes = plt.subplots(2, n, figsize=(4.6 * n, 6.2))
    rng = np.random.default_rng(SEED)
    for c, (etq, i, j, D, p) in enumerate(casos):
        real, mm, r = y[:, i, j], m[:, i, j], r_lane[i, j]
        ax_titulo_extra = f"{real.mean():.1f} cargas/día"
        sim = sortear_cargas(mm, r, 200, rng)
        ax = axes[0, c]
        ax.fill_between(np.arange(len(real)), np.percentile(sim, 5, axis=1), np.percentile(sim, 95, axis=1),
                        color="tab:orange", alpha=0.25, label="generador: rango 5–95 %")
        ax.plot(mm, color="tab:orange", lw=1.4, label="media m")
        ax.plot(real, color="black", lw=1, marker="o", ms=2.5, label="real")
        ax.set(title=f"{etq}: {hubs_txt(i, j)}\n{ax_titulo_extra} · D = {D:.3f}, p = {p:.3f}", xlabel="día", ylabel="cargas")
        if c == 0:
            ax.legend(fontsize=8)
        ax = axes[1, c]
        tope = max(np.percentile(real, 99), np.percentile(sim, 99))
        x = np.arange(0, tope + 1)
        ax.step(x, [(real <= v).mean() for v in x], where="post", color="black", label="real")
        ax.step(x, [(sim <= v).mean() for v in x], where="post", color="tab:orange", label="generador")
        ax.set(xlabel="cargas por día", ylabel="proporción acumulada", ylim=(0, 1.02))
        if c == 0:
            ax.legend(fontsize=8)
    fig.suptitle(titulo, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# ───────────────────────── B. rate ─────────────────────────

def media_sorteada(q):
    """Media EXACTA de un precio sorteado por inversión lineal por tramos de [min, q1, q2, q3, p90, max]
    (cada tramo aporta su masa por el punto medio). q: (..., 6) ordenado. Sin Monte Carlo."""
    return ((q[..., :-1] + q[..., 1:]) / 2 * _PESOS_TRAMO).sum(axis=-1)


def verificar_media_exacta(rng):
    """La fórmula exacta debe coincidir con la media de miles de sorteos de pricing.interp_cuantiles."""
    from pricing import interp_cuantiles
    q = np.sort(rng.uniform(1.5, 5, size=(6,)))
    u = rng.random(400_000)
    mc = interp_cuantiles(u, np.tile(q, (len(u), 1))).mean()
    return float(media_sorteada(q)), float(mc)


MIN_CARGAS_RATE = 5   # con menos cargas los 6 cuantiles son las propias cargas (1 carga: los 6 coinciden) y el error es ~0 por construcción


def tabla_rate(dias, min_cargas=MIN_CARGAS_RATE):
    q = np.sort(np.stack([dias[c] for c in ("min", "q1", "q2", "q3", "p90", "max")], axis=-1), axis=-1)
    q = q.astype(np.float64)
    real = dias["mean"].astype(np.float64)
    con = dias["count_loads"] >= min_cargas
    N = real.shape[1]
    con &= ~np.eye(N, dtype=bool)[None]
    sint = media_sorteada(q)
    err = np.where(con, (sint - real) / real, np.nan)
    n_dias = con.sum(axis=0)
    mape = np.nanmean(np.abs(err), axis=0)
    sesgo = np.nanmean(err, axis=0)
    lanes = np.argwhere(n_dias >= 20)
    t = pd.DataFrame({"i": lanes[:, 0], "j": lanes[:, 1], "n_dias": n_dias[lanes[:, 0], lanes[:, 1]],
                      "mape": mape[lanes[:, 0], lanes[:, 1]], "sesgo": sesgo[lanes[:, 0], lanes[:, 1]],
                      "tarifa_media": np.nanmean(np.where(con, real, np.nan), axis=0)[lanes[:, 0], lanes[:, 1]]})
    return t, err, con, sint, real, q


def figura_lanes_rate(dias, dist, casos, sint, real, q, con, hubs_txt, fechas, path, titulo):
    n = len(casos)
    fig, axes = plt.subplots(1, n, figsize=(4.6 * n, 4.2), sharex=True)
    axes = np.atleast_1d(axes)
    for ax, (etq, i, j, mape) in zip(axes, casos):
        k = con[:, i, j]
        # los días que no cumplen el filtro quedan como huecos (NaN): no se unen con líneas días muy separados
        ax.fill_between(fechas, q[:, i, j, 1], q[:, i, j, 3], where=k, color="0.6", alpha=0.25, label="real: [q1, q3] del día")
        ax.plot(fechas, np.where(k, real[:, i, j], np.nan), color="black", lw=1.2, marker="o", ms=3, label="real: tarifa media")
        ax.plot(fechas, np.where(k, sint[:, i, j], np.nan), color="tab:blue", lw=1.2, marker="o", ms=3, label="sintético: media del sorteo")
        ax.set(title=f"{etq}: {hubs_txt(i, j)} ({dist[i, j]:.0f} mi)\nerror relativo medio = {100 * mape:.1f}%",
               ylabel="USD/milla")
        ax.tick_params(axis="x", rotation=45)
    axes[0].legend(fontsize=8, loc="upper left")
    fig.suptitle(titulo, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# ───────────────────────── C. δ ─────────────────────────

def lilliefors_normal(x, n_sim=20000, rng=None):
    """KS contra una normal con media y desviación ESTIMADAS (Lilliefors 1967); p-valor por simulación."""
    rng = rng or np.random.default_rng(SEED)

    def D(v):
        z = np.sort((v - v.mean()) / v.std(ddof=1))
        F = stats.norm.cdf(z)
        n = len(z)
        return max((np.arange(1, n + 1) / n - F).max(), (F - np.arange(0, n) / n).max())

    d_obs = D(x)
    sim = np.array([D(rng.standard_normal(len(x))) for _ in range(n_sim)])
    return float(d_obs), float((sim >= d_obs).mean())


def main():
    os.makedirs(TAB, exist_ok=True)
    os.makedirs(FIG, exist_ok=True)
    rng = np.random.default_rng(SEED)
    dias = cargar_dias_mercado()
    fechas = dias["fechas"].astype("datetime64[D]")
    dow = (fechas.astype(np.int64) + 3) % 7
    cl = dias["count_loads"]
    N = cl.shape[1]
    with np.load(params_path(), allow_pickle=False) as z:
        f_w, grupo, r_grupo = z["f_w"].astype(np.float64), z["grupo_lane"].astype(np.int64), z["r"].astype(np.float64)
        s_delta = float(z["s_delta"])
    r_lane = r_grupo[np.clip(grupo, 0, len(r_grupo) - 1)]

    from problem_data import orden_hubs
    import validate_market_process as vmp
    hubs = orden_hubs()
    estados = vmp._estados_por_hub()
    dist = pd.read_csv(os.path.join(_CWD, "datos", "distance.csv"), index_col=0).to_numpy()

    tabla_hubs = pd.read_excel(os.path.join(_CWD, "hubs_id.xlsx"))
    nombres_hub = dict(zip(tabla_hubs["id"].astype(int), tabla_hubs["name"]))
    faltan = [int(h) for h in hubs if int(h) not in nombres_hub]
    assert not faltan, f"hubs sin nombre en hubs_id.xlsx: {faltan}"

    def hubs_txt(i, j):
        # nombre de ciudad (y estado) de cada hub, de hubs_id.xlsx; si faltara el ID se usa el estado
        def nom(k):
            h = int(hubs[k])
            return nombres_hub.get(h, f"{vmp._nombre_hub(h, estados)} (hub {h})")
        return f"{nom(i)} → {nom(j)}"

    pd.set_option("display.width", 200, "display.max_columns", 30, "display.float_format", "{:.4f}".format)

    # ── A ──
    print("== A. Disponibilidad de cargas (KS de dos muestras: real vs. generador) ==")
    # m como en el ajuste: entrenamiento con los vecinos de jun-ago; prueba con los vecinos de oct (r y f_w de
    # entrenamiento). Septiembre (validación) no entra en ninguno de los dos.
    periodos = {"entrenamiento": slice(0, TRAIN_DAYS), "prueba": slice(TEST_START, None)}
    datos = {}
    for nombre, sl in periodos.items():
        y = cl[sl].astype(np.float64)
        datos[nombre] = (y, estimar_m(y, dow[sl], f_w))

    y_tr, m_tr = datos["entrenamiento"]
    por_grupo = ks_por_grupo(y_tr, m_tr, r_lane, grupo, r_grupo, rng)
    y_te, m_te = datos["prueba"]
    por_grupo["D_prueba"] = ks_por_grupo(y_te, m_te, r_lane, grupo, r_grupo, rng)["D"]
    por_grupo = por_grupo.rename(columns={"D": "D_entrenamiento"})
    por_grupo.to_csv(os.path.join(TAB, "A_ks_por_grupo.csv"), index=False)
    print(por_grupo.to_string(index=False))
    figura_ecdf_grupos(y_tr, m_tr, r_lane, grupo, r_grupo, os.path.join(FIG, "A_ecdf_por_grupo.png"), rng)

    resumen, por_lane = [], {}
    for periodo, (y, m) in datos.items():
        t = ks_por_lane(y, m, r_lane, rng)
        por_lane[periodo] = t
        t.to_csv(os.path.join(TAB, f"A_ks_por_lane_{periodo}.csv"), index=False)
        resumen.append(dict(periodo=periodo, dias_por_lane=y.shape[0], lanes=len(t),
                            pct_rechazo_5=100 * (t["p_valor"] < ALFA).mean(),
                            D_mediana=t["D"].median(), D_p90=t["D"].quantile(0.9),
                            D_critico_5=1.36 * np.sqrt((y.shape[0] + y.shape[0] * SORTEOS_POR_DIA)
                                                       / (y.shape[0] * y.shape[0] * SORTEOS_POR_DIA))))
    resumen = pd.DataFrame(resumen)
    resumen.to_csv(os.path.join(TAB, "A_ks_por_lane_resumen.csv"), index=False)
    print("\nKS por lane:")
    print(resumen.to_string(index=False))

    t = por_lane["entrenamiento"]
    sin_cargas = int((t["dias_con_cargas"] == 0).sum())
    print(f"\nLanes sin ninguna carga en el período (D = 0 por construcción): {sin_cargas} de {len(t)}")
    pct_grupo = t.assign(g=grupo[t["i"], t["j"]], rech=t["p_valor"] < ALFA).groupby("g")["rech"].agg(["mean", "size"])
    pct_grupo["mean"] *= 100
    print("\n% de lanes rechazadas por grupo de volumen (entrenamiento):\n", pct_grupo.round(1).to_string())
    pct_grupo.to_csv(os.path.join(TAB, "A_rechazo_por_grupo.csv"))
    # Mejor / mediana / peor lane DENTRO de cada nivel de volumen (decil de volumen = grupo 0-9), para que el mejor caso
    # no sea una ruta casi vacía. Dentro de cada nivel solo se excluyen, para DIBUJAR, las lanes con menos de
    # MIN_DIAS_CASOS días con cargas (una lane vacía da D = 0 trivialmente). El KS de la sección 2.3 sí usa todas las lanes.
    t = t.assign(grupo=grupo[t["i"], t["j"]])
    filas_nivel = []
    for g in range(10):
        tg = t[(t["grupo"] == g) & (t["dias_con_cargas"] >= MIN_DIAS_CASOS)].sort_values("D").reset_index(drop=True)
        if len(tg) == 0:
            continue
        for etq, r in (("mejor", tg.iloc[0]), ("mediana", tg.iloc[len(tg) // 2]), ("peor", tg.iloc[-1])):
            filas_nivel.append(dict(grupo=g, caso=etq, origen=hubs_txt(int(r.i), int(r.j)).split(" → ")[0],
                                    destino=hubs_txt(int(r.i), int(r.j)).split(" → ")[1], i=int(r.i), j=int(r.j),
                                    cargas_por_dia=float(y_tr[:, int(r.i), int(r.j)].mean()), D=r.D, p_valor=r.p_valor,
                                    lanes_en_nivel=len(tg)))
    por_nivel = pd.DataFrame(filas_nivel)
    por_nivel.to_csv(os.path.join(TAB, "A_casos_por_volumen.csv"), index=False)
    print("\nMejor / mediana / peor lane en cada nivel de volumen (D del KS, jun–ago):")
    print(por_nivel.drop(columns=["i", "j"]).to_string(index=False))
    niveles = {"alto": (9, "Alto volumen (decil 9, ≥ 53 cargas/día en promedio)"),
               "medio": (5, "Volumen medio (decil 5, 7–11 cargas/día)"),
               "bajo": (1, "Bajo volumen (decil 1, 0.3–1.1 cargas/día)")}
    for clave, (g, texto) in niveles.items():
        sub = por_nivel[por_nivel["grupo"] == g]
        casos = [(f"{r.caso.capitalize()}", int(r.i), int(r.j), r.D, r.p_valor) for r in sub.itertuples()]
        figura_lanes_cargas(y_tr, m_tr, r_lane, casos, hubs_txt, os.path.join(FIG, f"A_lanes_{clave}_volumen.png"),
                            f"Cargas por día, {texto}: mejor, mediana y peor lane según D de KS (jun–ago)")

    met = {}
    for periodo, (y, m) in datos.items():
        met[periodo] = metricas_error_cargas(y, m, r_lane)
        met[periodo].to_csv(os.path.join(TAB, f"A_metricas_por_lane_{periodo}.csv"), index=False)
    resumen_met = pd.DataFrame([dict(periodo=p, **resumen_metricas_cargas(t)) for p, t in met.items()])
    resumen_met.to_csv(os.path.join(TAB, "A_metricas_error_resumen.csv"), index=False)
    print("\nMétricas de error de las cargas (generado vs real, por lane):")
    print(resumen_met.T.to_string(header=False))
    figura_media_desv(met["entrenamiento"], os.path.join(FIG, "A_media_desv_real_vs_generado.png"))

    # Referencia: la MISMA métrica sobre datos que sí vienen del generador (una realización por lane-día). Es el error
    # que se vería solo por ruido de muestreo (122 o 31 días por lane) si el modelo fuera exacto.
    rng_ref = np.random.default_rng(SEED + 1)
    ref = []
    for periodo, (y, m) in datos.items():
        mm = np.maximum(m, 1e-12)
        rr = np.broadcast_to(r_lane, m.shape)
        y_sim = np.where(m < 1.0, rng_ref.poisson(mm), rng_ref.negative_binomial(rr, rr / (rr + mm))).astype(np.float64)
        ref.append(dict(periodo=periodo, **resumen_metricas_cargas(metricas_error_cargas(y_sim, m, r_lane))))
    resumen_ref = pd.DataFrame(ref)
    resumen_ref.to_csv(os.path.join(TAB, "A_metricas_error_referencia.csv"), index=False)
    print("\nReferencia: mismas métricas con datos simulados por el propio generador (ruido de muestreo si el modelo fuera exacto):")
    print(resumen_ref.T.to_string(header=False))

    # ── B ──
    print("\n== B. Rate: media del precio sorteado vs. media real del día ==")
    exacta, mc = verificar_media_exacta(rng)
    print(f"Verificación fórmula exacta vs. Monte Carlo (400,000 sorteos): {exacta:.5f} vs {mc:.5f}")
    filas_g = []
    for etq, minc in (("≥1 carga (todos)", 1), (f"≥{MIN_CARGAS_RATE} cargas", MIN_CARGAS_RATE)):
        _, e, c, *_ = tabla_rate(dias, minc)
        filas_g.append(dict(lane_dias_incluidos=etq, n=int(c.sum()), sesgo_pct=100 * np.nanmean(e),
                            mape_pct=100 * np.nanmean(np.abs(e)), mediana_abs_pct=100 * np.nanmedian(np.abs(e)),
                            p95_abs_pct=100 * np.nanpercentile(np.abs(e), 95)))
    pd.DataFrame(filas_g).to_csv(os.path.join(TAB, "B_rate_global.csv"), index=False)
    print(pd.DataFrame(filas_g).to_string(index=False))
    tr, err, con, sint, real, q = tabla_rate(dias)       # a partir de aquí, solo lane-días con ≥ MIN_CARGAS_RATE cargas
    pct = tr["mape"].quantile([.05, .25, .5, .75, .9, .95, .99]) * 100
    print(f"Lanes con ≥20 días con ≥{MIN_CARGAS_RATE} cargas: {len(tr)}. Percentiles del error relativo medio por lane (%):\n{pct.round(2).to_string()}")
    pct.to_csv(os.path.join(TAB, "B_rate_percentiles_lane.csv"))
    tr = tr.sort_values("mape").reset_index(drop=True)
    sel = pd.concat([tr.head(3).assign(caso=["Mejor 1", "Mejor 2", "Mejor 3"]),
                     tr.iloc[[len(tr) // 2]].assign(caso="Mediana"),
                     tr.tail(3)[::-1].assign(caso=["Peor 1", "Peor 2", "Peor 3"])])
    sel["origen"] = [hubs_txt(i, j).split(" → ")[0] for i, j in zip(sel.i, sel.j)]
    sel["destino"] = [hubs_txt(i, j).split(" → ")[1] for i, j in zip(sel.i, sel.j)]
    sel["distancia_mi"] = [dist[i, j] for i, j in zip(sel.i, sel.j)]
    sel.to_csv(os.path.join(TAB, "B_casos_rate.csv"), index=False)
    print(sel.to_string(index=False))
    # Mejor / mediana / peor lane DENTRO de cada nivel de volumen (decil de volumen = grupo 0-9)
    tr["grupo"] = grupo[tr["i"], tr["j"]]
    filas_r = []
    for g in range(10):
        tg = tr[tr["grupo"] == g].sort_values("mape").reset_index(drop=True)
        if len(tg) == 0:
            continue
        for etq, r in (("mejor", tg.iloc[0]), ("mediana", tg.iloc[len(tg) // 2]), ("peor", tg.iloc[-1])):
            filas_r.append(dict(grupo=g, caso=etq, origen=hubs_txt(int(r.i), int(r.j)).split(" → ")[0],
                                destino=hubs_txt(int(r.i), int(r.j)).split(" → ")[1], i=int(r.i), j=int(r.j),
                                cargas_por_dia=float(dias["count_loads"][:, int(r.i), int(r.j)].mean()),
                                n_dias=int(r.n_dias), mape_pct=100 * r.mape, sesgo_pct=100 * r.sesgo,
                                lanes_en_nivel=len(tg)))
    rate_nivel = pd.DataFrame(filas_r)
    rate_nivel.to_csv(os.path.join(TAB, "B_casos_rate_por_volumen.csv"), index=False)
    print("\nRate: mejor / mediana / peor lane en cada nivel de volumen:")
    print(rate_nivel.drop(columns=["i", "j"]).to_string(index=False))
    g_bajo = int(tr.groupby("grupo").size().loc[lambda s: s >= 50].index.min())     # nivel más bajo con al menos 50 lanes
    for clave, g in (("alto", 9), ("medio", 6), ("bajo", g_bajo)):
        sub = rate_nivel[rate_nivel["grupo"] == g]
        cs = [(r.caso.capitalize(), int(r.i), int(r.j), r.mape_pct / 100) for r in sub.itertuples()]
        figura_lanes_rate(dias, dist, cs, sint, real, q, con, hubs_txt, fechas,
                          os.path.join(FIG, f"B_rate_{clave}_volumen.png"),
                          f"Rate, {clave} volumen (decil {g}): mejor, mediana y peor lane (≥ 20 días con ≥ 5 cargas)")
    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.hist(100 * tr["mape"], bins=40, color="tab:blue", edgecolor="white")
    ax.set(xlabel="error relativo medio por lane (%)", ylabel="lanes", title=f"{len(tr)} lanes con ≥ 20 días con ≥ 5 cargas")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "B_rate_hist_error.png"), dpi=130)
    plt.close(fig)

    # ── C ──
    print("\n== C. δ: normalidad del cambio diario del factor nacional (días de entrenamiento) ==")
    tr_cl = dias["count_loads"][:TRAIN_DAYS]
    universo = (tr_cl.sum(axis=0) > 0) & ~np.eye(N, dtype=bool)
    F = factor_nacional(tr_cl, dias["mean"][:TRAIN_DAYS], universo)
    dF = np.diff(F)
    D_l, p_l = lilliefors_normal(dF)
    t_stat, p_t = stats.ttest_1samp(dF, 0.0)
    resC = dict(n=len(dF), media=dF.mean(), desviacion=dF.std(ddof=1), s_usada=s_delta, asimetria=stats.skew(dF),
                curtosis_exceso=stats.kurtosis(dF), D_lilliefors=D_l, p_lilliefors=p_l, p_media_cero=float(p_t))
    pd.DataFrame([resC]).to_csv(os.path.join(TAB, "C_delta_normalidad.csv"), index=False)
    print(resC)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    axes[0].hist(dF, bins=25, density=True, color="0.5", edgecolor="white", label="ΔF real")
    xs = np.linspace(dF.min(), dF.max(), 200)
    axes[0].plot(xs, stats.norm.pdf(xs, 0, s_delta), color="tab:green", label=f"N(0, s²), s = {s_delta:.4f}")
    axes[0].set(xlabel="cambio diario del factor nacional (log)", ylabel="densidad")
    axes[0].legend(fontsize=8)
    stats.probplot((dF - dF.mean()) / dF.std(ddof=1), dist="norm", plot=axes[1])
    axes[1].set_title("QQ-plot de ΔF frente a la normal")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "C_delta_normalidad.png"), dpi=130)
    plt.close(fig)
    print(f"\nSalida en {OUT}")


if __name__ == "__main__":
    main()
