"""
fit_market_process.py
=====================
Estimación de lo que necesita el generador de realizaciones diarias
(metodologia_realizaciones_mercado_reefer.md) y del índice de mercado por nodo
(nivel lento + rápido AR(1), estimado con los datos nuevos).

El generador de cargas/tarifas de la metodología nueva NO tiene un modelo de tarifas que ajustar
(usa los cuantiles reales del día) ni una tasa global de cargas por lane (usa la media móvil de 6
vecinos de cada día, calculada en `market_process.py` a partir del stack real). Lo que SÍ hay que
estimar sobre los días de entrenamiento es:

  Paso 4b (cargas) — r: dispersión de la binomial negativa, por grupo de volumen (deciles de la
           media de cargas de cada lane en los días de ajuste). Lanes con m < 1 se sortean con
           Poisson, sin r.
  Fase 3 (índice, conservada) — índice de mercado por nodo, lento (ventana de 21 días), ρ y σ_i del
           rápido AR(1), frac_var_lento, indice_std. Se usa como feature del encoder y por B2P
           para pronosticar (ver decisión del plan).
  Paso 3 (δ) — s = desviación estándar del cambio diario del factor nacional F_t (promedio
           ponderado por cargas de log(tarifa media), sobre las 37,442 lanes sin diagonal).

Uso
---
    python fit_market_process.py

Escribe datos/market_params.npz. El ajuste usa los primeros TRAIN_DAYS días (jun-ago 2026) y los
194 nodos; las fechas salen de los nombres de archivo de datos/datos_finales_jun_oct_2026 (ver
problem_data.cargar_dias_mercado).
"""

import os
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import gammaln

from config import (
    DIAS_EXCLUIDOS, FECHA_INICIO, TRAIN_DAYS, MARKET_R_MAX, MARKET_N_GRUPOS, MARKET_PARAMS_PATH,
    VENTANA_LENTO_DIAS, MARKET_VENTANA_VECINOS, periodos,
)
from problem_data import cargar_dias_mercado

_CWD = os.path.dirname(os.path.abspath(__file__))

# 1970-01-01 fue jueves → con lunes = 0, dow = (días desde la época + 3) % 7.
_DOW_EPOCA = 3


def params_path() -> str:
    """Ruta absoluta de datos/market_params.npz."""
    return os.path.join(_CWD, MARKET_PARAMS_PATH)


def dia_de_semana(dias, fecha_inicio: str = FECHA_INICIO):
    """Día de la semana (lunes = 0) del día de índice `dias` (escalar o array) desde `fecha_inicio`."""
    dow0 = (int(np.datetime64(fecha_inicio, "D").astype(np.int64)) + _DOW_EPOCA) % 7
    return (dow0 + np.asarray(dias)) % 7


def seleccionar_dias(fechas: np.ndarray, dias_excluidos) -> np.ndarray:
    """Índices de los días de ajuste (de `fechas`, ya completo y continuo) salvo los de `dias_excluidos`.

    Una fecha excluida fuera del rango del archivo es un error (probablemente una fecha mal escrita).
    """
    excl = {np.datetime64(f, "D") for f in (dias_excluidos or [])}
    fuera = [str(f) for f in excl if f not in set(fechas)]
    if fuera:
        raise ValueError(f"DIAS_EXCLUIDOS fuera del rango del histórico: {fuera}")
    usar = np.array([f not in excl for f in fechas])
    return np.where(usar)[0]


# ─────────────────────────────────────────────────────────────────────────────
# Paso 4b — cargas: media móvil de 6 vecinos y dispersión r por grupo de volumen
# ─────────────────────────────────────────────────────────────────────────────

def estimar_factores_dow(total_por_fecha: np.ndarray, dow: np.ndarray) -> np.ndarray:
    """f_w (7,): promedio de las cargas totales por día de la semana entre el
    promedio de todas las fechas, reescalado para que los siete promedien 1."""
    promedio_global = total_por_fecha.mean()
    f = np.array([total_por_fecha[dow == k].mean() for k in range(7)]) / promedio_global
    return f / f.mean()


def estimar_m(count_loads: np.ndarray, dow: np.ndarray, f_w: np.ndarray,
              vecinos: int = MARKET_VENTANA_VECINOS, por_periodo: bool = False) -> np.ndarray:
    """m_{d,l} (D, N, N): media de cargas esperada en cada lane-día, según el Paso 4b de la
    metodología — media móvil de orden 2·`vecinos`+1 (`vecinos` días antes y después, sin el propio
    día), ajustada por día de la semana, con los vecinos disponibles en los bordes de la ventana
    (primeros/últimos días de `count_loads`, que puede ser la ventana completa jun-oct o solo los
    días de ajuste).

    por_periodo : True (con la ventana completa jun-oct) ⇒ la media móvil se calcula dentro de cada período
    (entrenamiento, validación, prueba; `config.periodos`): los días del borde usan solo los vecinos de su propio
    período, así que, p. ej., la m del 31 de agosto no ve las cargas de septiembre.
    """
    D = count_loads.shape[0]
    if por_periodo:
        partes = [estimar_m(count_loads[ini:fin], dow[ini:fin], f_w, vecinos)
                  for ini, fin in periodos(D).values() if fin > ini]
        return np.concatenate(partes, axis=0)
    ajustado = count_loads / f_w[dow][:, None, None]
    suma = np.zeros_like(ajustado)
    cuenta_full = np.zeros_like(ajustado)
    for k in [-v for v in range(1, vecinos + 1)] + list(range(1, vecinos + 1)):
        if k < 0:
            suma[-k:D] += ajustado[0:D + k]
            cuenta_full[-k:D] += 1.0
        else:
            suma[0:D - k] += ajustado[k:D]
            cuenta_full[0:D - k] += 1.0
    prom = suma / np.maximum(cuenta_full, 1.0)
    return f_w[dow][:, None, None] * prom


def _nll_nb(log_r: float, y: np.ndarray, m: np.ndarray) -> float:
    """−log-verosimilitud conjunta NegBin(media m, dispersión r); var = m + m²/r."""
    r = np.exp(log_r)
    ll = (
        gammaln(y + r) - gammaln(r) - gammaln(y + 1.0)
        + r * (np.log(r) - np.log(r + m))
        + y * (np.log(m) - np.log(r + m))
    )
    return -float(ll.sum())


def estimar_r_grupo(y: np.ndarray, m: np.ndarray, r_max: float = MARKET_R_MAX) -> float:
    """r de un grupo por máxima verosimilitud conjunta sobre sus lane-días (y, m con m > 0).

    Si el óptimo tiende a infinito (equivale a Poisson) se fija r_max.
    """
    lo, hi = np.log(1e-2), np.log(r_max)
    res = minimize_scalar(_nll_nb, bounds=(lo, hi), args=(y, m), method="bounded",
                          options={"xatol": 1e-4})
    if res.x > hi - 1e-2:
        return float(r_max)
    return float(np.exp(res.x))


def ajustar_cargas(count_loads: np.ndarray, dow: np.ndarray, n_grupos: int = MARKET_N_GRUPOS,
                    r_max: float = MARKET_R_MAX) -> dict:
    """f_w, grupo de volumen por lane (deciles de la m promedio de cada lane en los días de ajuste,
    sin la diagonal) y r por grupo.

    count_loads, dow : (D, N, N), (D,) — SOLO los días de ajuste (TRAIN_DAYS). La lane se agrupa por
    su m promedio (no por su m de cada día: da un grupo estable por lane, usado también para generar),
    y r se estima por MLE conjunta sobre todos los lane-días (m >= 1) de las lanes de ese grupo.
    """
    D, N, _ = count_loads.shape
    f_w = estimar_factores_dow(count_loads.sum(axis=(1, 2)), dow)
    m = estimar_m(count_loads, dow, f_w)                       # (D, N, N)
    sin_diag = ~np.eye(N, dtype=bool)
    m_prom_lane = m.mean(axis=0)                               # (N, N)

    grupo = np.full((N, N), -1, dtype=np.int8)
    valores = m_prom_lane[sin_diag]
    cortes = np.quantile(valores, np.linspace(0, 1, n_grupos + 1)[1:-1])
    grupo[sin_diag] = np.digitize(m_prom_lane[sin_diag], cortes).astype(np.int8)

    r = np.full(n_grupos, r_max)
    n_lanes_grupo = np.zeros(n_grupos, dtype=np.int64)
    for h in range(n_grupos):
        lanes_h = (grupo == h)
        n_lanes_grupo[h] = int(lanes_h.sum())
        y = count_loads[:, lanes_h]                            # (D, L_h)
        mm = m[:, lanes_h]                                      # (D, L_h)
        usar = mm >= 1.0
        if usar.sum() < 30:                                     # muy pocos datos: deja Poisson (r_max)
            continue
        r[h] = estimar_r_grupo(y[usar], mm[usar], r_max)

    return {"f_w": f_w, "grupo_lane": grupo, "cortes_grupo": cortes, "r": r,
            "n_lanes_grupo": n_lanes_grupo}


# ─────────────────────────────────────────────────────────────────────────────
# Fase 3 (conservada) — índice de mercado por nodo
# ─────────────────────────────────────────────────────────────────────────────

def indice_observado(rate: np.ndarray, media_lane: np.ndarray, universo_indice: np.ndarray) -> np.ndarray:
    """Índice de mercado observado de todos los nodos en cada día, (D, N).

    I(i, d)     = promedio, sobre las lanes salientes de i en `universo_indice`, de
                  tarifa(i, j, d) − media_lane(ij)
    nac(d)      = promedio de I(i, d) sobre los N nodos
    índice(i,d) = I(i, d) − nac(d)

    rate : (D, N, N) tarifa media diaria. media_lane, universo_indice : (N, N).
    """
    rate = np.asarray(rate, dtype=np.float64)
    cuenta = universo_indice.sum(axis=1)
    if (cuenta == 0).any():
        raise ValueError(f"nodos sin lanes salientes en el universo: {np.flatnonzero(cuenta == 0).tolist()}")
    desvio = (rate - media_lane[None]) * universo_indice[None]
    bruto = desvio.sum(axis=2) / cuenta[None]
    return bruto - bruto.mean(axis=1, keepdims=True)


def ajustar_indice(rates: np.ndarray, universo: np.ndarray, ventana: int = VENTANA_LENTO_DIAS,
                    dias: np.ndarray = None) -> dict:
    """Estima media_lane, escenarios de lento, ρ y σ_i del índice (antes `ajustar_tarifas`; ya no
    calcula residuos — el generador de precios usa los cuantiles reales, no media_lane+índice+residuo).

    rates : (D, N, N) tarifa MEDIA diaria de los días de ajuste. universo : (N, N) bool, lanes con al
    menos una carga. El índice usa el universo sin la diagonal.
    """
    rates = np.asarray(rates, dtype=np.float64)
    D, N, _ = rates.shape
    dias = np.arange(D) if dias is None else np.asarray(dias)
    if ventana % 2 != 1 or ventana < 1:
        raise ValueError(f"la ventana del lento debe ser impar, recibí {ventana}")
    mitad = ventana // 2

    media_lane = rates.mean(axis=0)
    universo_indice = universo & ~np.eye(N, dtype=bool)
    indice = indice_observado(rates, media_lane, universo_indice)

    if D < ventana:
        raise ValueError(f"{D} días no alcanzan para la ventana de {ventana}")
    ventanas = np.lib.stride_tricks.sliding_window_view(indice, ventana, axis=0)
    inicio = np.arange(D - ventana + 1)
    completa = (dias[inicio + ventana - 1] - dias[inicio]) == ventana - 1
    centros = inicio[completa] + mitad
    lento = ventanas.mean(axis=-1)[completa]
    rapido = indice[centros] - lento

    seguido = (centros[1:] - centros[:-1]) == 1
    if not seguido.any():
        raise ValueError(f"{D} días no alcanzan para la ventana de {ventana}: se necesitan ≥ 2 días de escenario seguidos")
    x, y = rapido[:-1][seguido], rapido[1:][seguido]
    rho = float((x * y).sum() / (x * x).sum())
    sigma = (y - rho * x).std(axis=0, ddof=1)

    var_indice = float(indice[centros].var())
    return {
        "media_lane": media_lane,
        "universo_indice": universo_indice,
        "indice": indice,
        "indice_std": np.float64(indice.std()),
        "lento_escenarios": lento,
        "dias_escenario": dias[centros].astype(np.int64),
        "rho": np.float64(rho),
        "sigma": sigma,
        "frac_var_lento": np.float64(float(lento.var()) / var_indice),
        "ventana_lento": np.int64(ventana),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Paso 3 — factor nacional F_t y dispersión de δ
# ─────────────────────────────────────────────────────────────────────────────

def factor_nacional(count_loads: np.ndarray, mean_rate: np.ndarray, universo_indice: np.ndarray) -> np.ndarray:
    """F_t (D,): promedio ponderado por cargas de log(tarifa media), sobre las lanes de
    `universo_indice` (sin diagonal), para cada día. Paso 3 de la metodología."""
    peso = count_loads * universo_indice[None]
    num = (peso * np.log(mean_rate)).sum(axis=(1, 2))
    den = peso.sum(axis=(1, 2))
    return num / den


def estimar_dispersion_delta(count_loads: np.ndarray, mean_rate: np.ndarray, universo_indice: np.ndarray) -> float:
    """s = desviación estándar del cambio diario de F_t en los días de ajuste (tamaño de δ)."""
    F = factor_nacional(count_loads, mean_rate, universo_indice)
    return float(np.diff(F).std())


# ─────────────────────────────────────────────────────────────────────────────
# Ajuste completo
# ─────────────────────────────────────────────────────────────────────────────

def ajustar(dias_excluidos=None, n_grupos: int = MARKET_N_GRUPOS, r_max: float = MARKET_R_MAX,
            ventana_lento: int = VENTANA_LENTO_DIAS, dias_data: dict = None,
            train_days: int = TRAIN_DAYS) -> dict:
    """Estima f_w, grupo_lane/r (cargas), el índice (media_lane, lento, ρ, σ_i) y s (δ)
    sobre los primeros `train_days` días de `dias_data` (por defecto, los de
    datos/datos_finales_jun_oct_2026 vía `cargar_dias_mercado`, con los 194 nodos).

    `dias_data` acepta un dict con la misma forma que devuelve `cargar_dias_mercado` (claves
    `count_loads`, `mean`, `fechas`) para poder probar el ajuste con un panel sintético pequeño,
    sin tocar disco.
    """
    dias_data = dias_data if dias_data is not None else cargar_dias_mercado()
    fechas_todas = dias_data["fechas"]
    idx_ajuste = seleccionar_dias(fechas_todas[:train_days], dias_excluidos)

    count_loads = dias_data["count_loads"][idx_ajuste]
    mean_rate   = dias_data["mean"][idx_ajuste]
    fechas      = fechas_todas[idx_ajuste]
    fecha_inicio = str(fechas_todas[0])
    dow = (fechas.astype("datetime64[D]").astype(np.int64) + _DOW_EPOCA) % 7

    n_ij = count_loads.sum(axis=0)
    universo = n_ij > 0
    N = universo.shape[0]
    universo_indice = universo & ~np.eye(N, dtype=bool)

    cargas = ajustar_cargas(count_loads, dow, n_grupos, r_max)
    indice = ajustar_indice(mean_rate, universo, ventana_lento, idx_ajuste)
    s_delta = estimar_dispersion_delta(count_loads, mean_rate, universo_indice)

    return {
        **cargas,
        **indice,
        "universo": universo,
        "n_ij": n_ij,
        "s_delta": np.float64(s_delta),
        "dias_usados": idx_ajuste.astype(np.int64),
        "fechas_ajuste": fechas.astype("datetime64[D]").astype(str),
        "fecha_inicio": np.array(fecha_inicio),
        "num_dias_total": np.int64(len(fechas_todas)),
        "train_days": np.int64(train_days),    # con cuántos días se ajustó: MarketProcess exige que sea TRAIN_DAYS
    }


def guardar(params: dict, path: str = None) -> str:
    """Guarda los parámetros en datos/market_params.npz."""
    path = path or params_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(path, **params)
    return path


def imprimir_resumen(params: dict) -> None:
    dias = ["lun", "mar", "mié", "jue", "vie", "sáb", "dom"]
    print(f"Días de ajuste : {len(params['dias_usados'])} "
          f"({params['fechas_ajuste'][0]} → {params['fechas_ajuste'][-1]})")
    print(f"Lanes en el universo (≥1 carga): {int(params['universo'].sum())}")
    print("f_w (factor del día de la semana, promedio 1):")
    for k in range(7):
        print(f"   {dias[k]}: {params['f_w'][k]:.4f}")
    print("r_h (dispersión por grupo de volumen, h = 0 menor volumen):")
    for h, (r, n) in enumerate(zip(params["r"], params["n_lanes_grupo"])):
        print(f"   h={h}: r = {r:10.3f}   ({n} lanes)")
    print(f"Índice — ρ (persistencia del rápido)   : {float(params['rho']):.4f}")
    print(f"         σ_i del rápido                : media {params['sigma'].mean():.4f}  "
          f"[{params['sigma'].min():.4f}, {params['sigma'].max():.4f}]")
    print(f"         fracción de varianza que es lento: {float(params['frac_var_lento']):.4f}")
    print(f"         desviación del índice histórico  : {float(params['indice_std']):.4f} $/milla")
    print(f"δ — desviación del cambio diario del factor nacional (s): {float(params['s_delta']):.4f}")


def main() -> dict:
    params = ajustar(DIAS_EXCLUIDOS)
    imprimir_resumen(params)
    path = guardar(params)
    print(f"Parámetros guardados en: {path}")
    return params


if __name__ == "__main__":
    main()
