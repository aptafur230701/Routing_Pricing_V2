"""
fit_market_process.py
=====================
Estimación del proceso exógeno de cargas y tarifas (fases 1 a 3 de
proceso_exogeno_cargas_tarifas.md).

  Fase 1 — datos de ajuste: las matrices [día, origen, destino] de 194 nodos
           de los primeros TRAIN_DAYS días (los de entrenamiento), menos las fechas
           de DIAS_EXCLUIDOS (vacía por defecto). Los días de evaluación quedan
           fuera del ajuste para que el replay sea fuera de muestra.
  Fase 2 — cargas: binomial negativa NegBin(mu_ij · s_d, phi_h)
             · s_d  : factor del día de la semana, promedio 1
             · mu_ij: tasa diaria base de la lane = n_ij / E  (E = Σ s_d sobre las fechas)
             · phi_h: dispersión por grupo de volumen (cuartiles de mu_ij), por
                      máxima verosimilitud conjunta; tope MARKET_PHI_MAX
  Fase 3 — tarifas: se guarda el histórico de tarifa por milla de cada lane
           (todos los días, con o sin cargas) para remuestrearlo con reemplazo.

Uso
---
    python fit_market_process.py

Escribe datos/market_params.npz (indexable por [origen, destino]). El ajuste usa los
primeros TRAIN_DAYS días; las fechas salen de FECHA_INICIO + índice del día.
"""

import os
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import gammaln

from config import (
    FECHA_INICIO, DIAS_EXCLUIDOS, TRAIN_DAYS, MARKET_PHI_MAX, MARKET_N_GRUPOS, MARKET_PARAMS_PATH,
)

_CWD = os.path.dirname(os.path.abspath(__file__))

# 1970-01-01 fue jueves → con lunes = 0, dow = (días desde la época + 3) % 7.
_DOW_EPOCA = 3


def params_path() -> str:
    """Ruta absoluta de datos/market_params.npz."""
    return os.path.join(_CWD, MARKET_PARAMS_PATH)


def dia_de_semana(dias, fecha_inicio: str = FECHA_INICIO):
    """Día de la semana (lunes = 0) del día de índice `dias` (escalar o array).

    dow = (FECHA_INICIO.weekday() + día) % 7. Con FECHA_INICIO = 2026-01-01
    (jueves) equivale a (día + 3) % 7.
    """
    dow0 = (int(np.datetime64(fecha_inicio, "D").astype(np.int64)) + _DOW_EPOCA) % 7
    return (dow0 + np.asarray(dias)) % 7


# ─────────────────────────────────────────────────────────────────────────────
# Fase 1 — datos de ajuste
# ─────────────────────────────────────────────────────────────────────────────

def cargar_matrices_completas(max_dias: int = None) -> tuple:
    """Lee loads_stack y rate_stack (194 nodos); los primeros `max_dias` días, o todos si es None."""
    data_dir = os.path.join(_CWD, "datos")
    loads = np.load(os.path.join(data_dir, "load_availability.npy"))
    rate  = np.load(os.path.join(data_dir, "rate.npy"))
    num_dias = min(loads.shape[0], rate.shape[0], max_dias or loads.shape[0])
    return loads[:num_dias].astype(np.float64), rate[:num_dias].astype(np.float64)


def seleccionar_dias(num_dias: int, fecha_inicio: str, dias_excluidos) -> tuple:
    """Índices de los días de ajuste y su fecha (np.datetime64[D]).

    Usa todos los días salvo los de `dias_excluidos`. Una fecha excluida fuera
    del rango del archivo es un error (probablemente una fecha mal escrita).
    """
    fechas = np.datetime64(fecha_inicio, "D") + np.arange(num_dias)
    excl = {np.datetime64(f, "D") for f in (dias_excluidos or [])}
    fuera = [str(f) for f in excl if f not in set(fechas)]
    if fuera:
        raise ValueError(f"DIAS_EXCLUIDOS fuera del rango del histórico: {fuera}")
    usar = np.array([f not in excl for f in fechas])
    return np.where(usar)[0], fechas[usar]


# ─────────────────────────────────────────────────────────────────────────────
# Fase 2 — cargas
# ─────────────────────────────────────────────────────────────────────────────

def estimar_factores_dow(total_por_fecha: np.ndarray, dow: np.ndarray) -> np.ndarray:
    """s_d (7,): promedio de las cargas totales por día de la semana entre el
    promedio de todas las fechas, reescalado para que los siete promedien 1."""
    promedio_global = total_por_fecha.mean()
    s = np.array([total_por_fecha[dow == k].mean() for k in range(7)]) / promedio_global
    return s / s.mean()


def _nll_nb(log_phi: float, y: np.ndarray, m: np.ndarray) -> float:
    """−log-verosimilitud conjunta NegBin(media m, dispersión phi); var = m + m²/phi."""
    phi = np.exp(log_phi)
    ll = (
        gammaln(y + phi) - gammaln(phi) - gammaln(y + 1.0)
        + phi * (np.log(phi) - np.log(phi + m))
        + y * (np.log(m) - np.log(phi + m))
    )
    return -float(ll.sum())


def estimar_phi_grupo(y: np.ndarray, m: np.ndarray, phi_max: float = MARKET_PHI_MAX) -> float:
    """phi de un grupo por máxima verosimilitud conjunta sobre sus conteos lane-día.

    y, m: conteos y medias (mu_ij · s_d), todos > 0 en m. Si el óptimo tiende a
    infinito (equivale a Poisson) se fija phi_max.
    """
    lo, hi = np.log(1e-2), np.log(phi_max)
    res = minimize_scalar(_nll_nb, bounds=(lo, hi), args=(y, m), method="bounded",
                          options={"xatol": 1e-4})
    # Óptimo pegado al tope ⇒ la verosimilitud sigue creciendo hacia Poisson.
    if res.x > hi - 1e-2:
        return float(phi_max)
    return float(np.exp(res.x))


def ajustar(
    loads_stack: np.ndarray,
    rate_stack:  np.ndarray,
    fecha_inicio: str = FECHA_INICIO,
    dias_excluidos=None,
    phi_max: float = MARKET_PHI_MAX,
    n_grupos: int = MARKET_N_GRUPOS,
) -> dict:
    """Estima s_d, mu_ij, phi_h y reúne las tarifas históricas por lane.

    Parámetros
    ----------
    loads_stack, rate_stack : [num_días, N, N] — cargas y tarifa por milla.

    Retorna un dict con los arrays que se guardan en market_params.npz.
    """
    num_dias = loads_stack.shape[0]
    idx, fechas = seleccionar_dias(num_dias, fecha_inicio, dias_excluidos)
    loads = loads_stack[idx]
    rates = rate_stack[idx]
    D = len(idx)

    # Fase 1: día de la semana de cada fecha usada (desde la fecha, no del índice).
    dow = (fechas.astype(np.int64) + _DOW_EPOCA) % 7

    # Paso 1: factores de día de la semana.
    s = estimar_factores_dow(loads.sum(axis=(1, 2)), dow)

    # Paso 2: tasa por lane, mu_ij = n_ij / E con E = Σ s_d sobre las fechas.
    exposicion = float(s[dow].sum())
    n_ij = loads.sum(axis=0)
    mu = n_ij / exposicion
    universo = n_ij > 0            # lanes con al menos una carga en el histórico

    # Paso 3: phi por grupo de volumen (cuartiles de mu_ij entre las lanes del universo).
    cortes = np.quantile(mu[universo], np.linspace(0, 1, n_grupos + 1)[1:-1])
    grupo = np.digitize(mu, cortes).astype(np.int8)   # 0..n_grupos-1
    media_dia = s[dow]                                 # (D,) factor de cada fecha
    phi = np.zeros(n_grupos)
    n_lanes_grupo = np.zeros(n_grupos, dtype=np.int64)
    for h in range(n_grupos):
        sel = universo & (grupo == h)
        n_lanes_grupo[h] = sel.sum()
        y = loads[:, sel]                                  # (D, L_h)
        m = media_dia[:, None] * mu[sel][None, :]          # (D, L_h)
        phi[h] = estimar_phi_grupo(y.ravel(), m.ravel(), phi_max)

    return {
        "s": s,
        "mu": mu,
        "phi": phi,
        "grupo": grupo,
        "universo": universo,
        "tarifas": rates.astype(np.float32),          # (D, N, N) histórico por lane
        "n_ij": n_ij,
        "exposicion": np.float64(exposicion),
        "cortes_grupo": cortes,
        "n_lanes_grupo": n_lanes_grupo,
        "dias_usados": idx.astype(np.int64),
        "fechas_ajuste": fechas.astype("datetime64[D]").astype(str),
        "fecha_inicio": np.array(fecha_inicio),
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
    print("s_d (factor del día de la semana, promedio 1):")
    for k in range(7):
        print(f"   {dias[k]}: {params['s'][k]:.4f}")
    print(f"Exposición E = Σ s_d = {float(params['exposicion']):.3f}")
    print("phi_h (dispersión por grupo de volumen, h = 0 menor volumen):")
    for h, (p, n) in enumerate(zip(params["phi"], params["n_lanes_grupo"])):
        print(f"   h={h}: phi = {p:10.3f}   ({n} lanes)")


def main() -> dict:
    # Solo los días de entrenamiento: los de evaluación no deben filtrarse al proceso.
    loads, rate = cargar_matrices_completas(TRAIN_DAYS)
    params = ajustar(loads, rate, FECHA_INICIO, DIAS_EXCLUIDOS)
    imprimir_resumen(params)
    path = guardar(params)
    print(f"Parámetros guardados en: {path}")
    return params


if __name__ == "__main__":
    main()
