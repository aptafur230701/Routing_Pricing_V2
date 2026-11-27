"""
test_baselines.py
=================
Pruebas de los baselines de baselines_drl_ruteo_estocastico.md: UB-DP, UB-PI, B1, B2, B3
y el arnés de evaluación en mundos simulados.

  · La tabla V vectorizada (dos intentos de cotización, mismo tablero) coincide con una enumeración ingenua.
  · B1, B2 y B3 nunca eligen una acción fuera de la máscara (el arnés lanza si ocurre).
  · Con el mismo world_seed, cada método da exactamente el mismo resultado dos veces.
  · En cada mundo, UB-PI es mayor o igual que el margen de cada política.
  · B3 ≤ UB-DP ≤ UB-PI en promedio (salvo el error de Monte Carlo).
  · Si la tabla es "no tratable" (tope de tiempo), se guarda el marcador y B3 / UB-DP se reportan así.
  · UB-PI está alineado con el entorno: el entorno acepta su ruta cotizando el nivel más alto que gana y
    el arnés mide su margen. (El margen = Σ lane_reward con el precio cotizado: test_env_cotizacion.py.)
    (UB-PI contra fuerza bruta, B2 contra enumeración ingenua y la máscara con revisitas: test_revisit.py.)
  · Ningún nombre borrado sigue referenciado en el repositorio.
"""

import glob
import os
import re

import numpy as np
import pandas as pd
import pytest
import torch

from model.agent import AMRoutingAgent
from baselines.politicas import Observacion, PoliticaB1, PoliticaB2
from pricing import N_NIVELES, U_NIVELES_ARR, gana
from routing_env import OPCION_VACIO
from config import (
    MAX_DURATION, TRAIN_DAYS, K_MAX, MARKET_SEED_TRAIN, MARKET_SEED_EVAL, MARKET_SEED_EVAL_FINAL,
)
from baselines.dp_relajada import (calcular_tabla, cargar_o_calcular_tabla, franjas_rechazo, _sortear_tableros,
                                   PENALIZACION_USD)
from baselines.ub_pi import franjas_por_dia
from routing_env import vacio_permitido
import evaluation as ev
from config import COTIZACIONES_MAX_POR_NODO
from market_process import MarketProcess, asegurar_parametros
from problem_data import load_matrices, lane_reward
import stats_analysis as st

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _agente(n):
    torch.manual_seed(0)
    return AMRoutingAgent(num_nodes=n, d_h=32, n_heads=4, n_layers=1, d_ff=64,
                          device=torch.device("cpu"))


def _opciones(loads_row, mascara):
    """Opciones de una observación sintética: niveles donde hay carga, VACÍO donde no."""
    op = np.zeros((len(loads_row), N_NIVELES + 1), dtype=bool)
    op[:, :N_NIVELES] = (np.asarray(loads_row) > 0)[:, None]
    op[:, OPCION_VACIO] = np.asarray(loads_row) == 0
    return op & (np.asarray(mascara) > 0)[:, None]


class Instancia:
    """Datos, proceso, tabla V y agente aleatorio de un tamaño de grafo."""

    def __init__(self, n, S=50):
        asegurar_parametros()
        self.n, self._S, self._tabla = n, S, None
        (self.tm, self.rate, self.loads, self.dist, self.diesel,
         self.ltr, self.avail, self.p95) = load_matrices(n)
        self.T = np.asarray(self.tm, dtype=float)
        self.market = MarketProcess(n)
        self.agent = _agente(n)

    @property
    def tabla(self):
        """Se calcula solo si un test la usa (con dos intentos cuesta minutos en CPU)."""
        if self._tabla is None:
            self._tabla = calcular_tabla(self.market, self.tm, self.dist, self.diesel, K=K_MAX, S=self._S)
        return self._tabla

    def simular(self, M, k_levels=(K_MAX, None)):
        return ev.evaluar_mundos_simulados(
            self.agent, self.market, self.tabla, self.tm,
            self.rate[:TRAIN_DAYS], self.loads[:TRAIN_DAYS], self.dist, self.diesel, self.n,
            self.ltr, self.avail, self.p95, M=M, k_levels=k_levels, verbose=False,
        )


@pytest.fixture(scope="module")
def inst10():
    return Instancia(10, S=50)


@pytest.fixture(scope="module")
def sim10(inst10):
    return inst10.simular(M=100)


# ─────────────────────────────────────────────────────────────────────────────
# UB-DP: tabla vectorizada vs recursión ingenua
# ─────────────────────────────────────────────────────────────────────────────

def test_tabla_dp_coincide_con_enumeracion_ingenua_de_los_dos_intentos():
    """Instancia chica (4 nodos, viajes de 6-10 h, franjas de 15 min, S = 4) donde sí hay revisitas y rechazos.

    La enumeración es la ecuación de especificacion_fase0_ruteo_pricing.md §7.3 escrita con loops: por cada
    tablero y r rechazos, la mejor de E (la mejor jugada sin carga: casa vacío, reposicionarse ≤ 300 mi o esperar
    al día siguiente) y (cotizar j al nivel k, si la lane tiene más de r cargas; si se pierde, lo mismo con r + 1). En casa sin haber salido (H): sin "volver"; sin ninguna jugada, 0 en s = 0 y la penalización
    después."""
    asegurar_parametros()
    n, S, dt, K = 4, 4, 0.25, 3
    _, _, _, dist, diesel, *_ = load_matrices(n)
    market = MarketProcess(n)
    rng = np.random.default_rng(3)
    T = rng.uniform(6.0, 10.0, size=(n, n))
    np.fill_diagonal(T, 0.0)
    tabla = calcular_tabla(market, T, dist, diesel, K=K, S=S, dt=dt, seed=11, device=torch.device("cpu"),
                           max_horas=None)

    tab = _sortear_tableros(market, dist, diesel, S, 11)
    A, D, Rk, P, R0 = tab["A"], tab["D"], tab["Rk"], tab["P"], tab["R0"]
    f = np.floor(T / dt + 1e-9).astype(int)
    smax = int(np.floor(MAX_DURATION / dt + 1e-9))
    tau = franjas_rechazo(dt)
    assert tau == 1
    C = COTIZACIONES_MAX_POR_NODO
    pd_ = franjas_por_dia(dt)
    rep = vacio_permitido(dist)
    niveles = range(Rk.shape[-1])
    memo, memo_h = {}, {}
    vuelta = f + f.T
    np.fill_diagonal(vuelta, smax + 1)

    def E_de(k, d0, h, i, t):
        """La mejor jugada sin carga desde el intermedio i en la franja t."""
        e = float(R0[i, h])
        sw = (t // pd_ + 1) * pd_
        if sw + f[i, h] <= smax:
            e = max(e, V(k, d0, h, i, sw))
        if k is None or k >= 2:
            for j in range(n):
                if rep[i, j] and j not in (i, h) and t + f[i, j] + f[j, h] <= smax:
                    e = max(e, float(R0[i, j]) + V(None if k is None else k - 1, d0, h, j, t + f[i, j]))
        return e

    def V(k, d0, h, i, s):
        clave = (k, d0, h, i, s)
        if clave in memo:
            return memo[clave]
        dw = (d0 + int((s * dt) // 14)) % 7
        ts = [min(s + r * tau, smax) for r in range(C + 1)]
        Es = [E_de(k, d0, h, i, t) for t in ts]
        vals = []
        for m in range(S):
            d = D[dw, i, m]

            def valido(j, t, r):
                if A[dw, i, m, j] <= r:
                    return False
                if t + tau + f[i, h] > smax:                        # si se pierde, no alcanza a volver
                    return False
                if j == h:
                    return True
                return (k is None or k >= 2) and j != i and t + f[i, j] + f[j, h] <= smax

            def G(j, lvl, t):
                cont = 0.0 if j == h else V(None if k is None else k - 1, d0, h, j, t + f[i, j])
                return float(Rk[dw, d, i, j, lvl]) + cont

            def intento(j, t, respaldo):
                return max(P[dw, d, i, l] * G(j, l, t) + (1 - P[dw, d, i, l]) * respaldo for l in niveles)

            b = Es[C]
            for r in range(C - 1, -1, -1):
                b = max([Es[r]] + [intento(j, ts[r], b) for j in range(n) if valido(j, ts[r], r)])
            vals.append(b)
        memo[clave] = float(np.mean(vals))
        return memo[clave]

    def H(k, d0, h, s):
        """Valor en casa sin haber salido, en la franja s (k arcos; None = libre)."""
        clave = (k, d0, h, s)
        if clave in memo_h:
            return memo_h[clave]
        ks = None if k is None else k - 1
        dw = (d0 + int((s * dt) // 14)) % 7
        ok = lambda j, t: j != h and t + f[h, j] + f[j, h] <= smax

        def sin_carga(t):
            e = -np.inf
            sw = (t // pd_ + 1) * pd_
            if sw + vuelta[h].min() <= smax:
                e = H(k, d0, h, sw)
            for j in range(n):
                if rep[h, j] and ok(j, t):
                    e = max(e, float(R0[h, j]) + V(ks, d0, h, j, t + f[h, j]))
            return e

        ts = [min(s + r * tau, smax) for r in range(C + 1)]
        Es = [sin_carga(t) for t in ts]
        falla = lambda x: x if np.isfinite(x) else PENALIZACION_USD   # después de perder, sin jugadas: falla
        vals = []
        for m in range(S):
            d = D[dw, h, m]
            G = lambda j, l, t: float(Rk[dw, d, h, j, l]) + V(ks, d0, h, j, t + f[h, j])
            intento = lambda j, t, resp: max(P[dw, d, h, l] * G(j, l, t) + (1 - P[dw, d, h, l]) * resp
                                             for l in niveles)
            mejor = falla(Es[C])
            for r in range(C - 1, -1, -1):
                mejor = max([Es[r]] + [intento(j, ts[r], mejor) for j in range(n)
                                       if A[dw, h, m, j] > r and ok(j, ts[r])])
                if r > 0:
                    mejor = falla(mejor)
            if not np.isfinite(mejor):
                mejor = 0.0 if s == 0 else PENALIZACION_USD        # no sale (s = 0) o falla (después)
            vals.append(mejor)
        memo_h[clave] = float(np.mean(vals))
        return memo_h[clave]

    def V0(k, d0, h):
        return H(k, d0, h, 0)

    for d0, h in ((0, 0), (3, 2)):
        for k in (2, 3):
            assert tabla.ub_dp(h, d0, k) == pytest.approx(V0(k, d0, h), rel=1e-4, abs=1e-2)
    assert tabla.ub_dp(1, 5, None) == pytest.approx(V0(None, 5, 1), rel=1e-4, abs=1e-2)
    for k, d0, h, i, s in ((1, 0, 0, 1, 150), (2, 2, 1, 3, 80), (3, 5, 2, 0, 0), (3, 4, 3, 2, 120)):
        assert float(tabla.V_k[k, d0, h, i, s]) == pytest.approx(V(k, d0, h, i, s), rel=1e-4, abs=1e-2)
    for d0, h, i, s in ((1, 0, 2, 60), (6, 2, 1, 160)):
        assert float(tabla.V_libre[d0, h, i, s]) == pytest.approx(V(None, d0, h, i, s), rel=1e-4, abs=1e-2)


def test_tabla_no_tratable_deja_el_marcador_y_b3_se_reporta_asi(tmp_path):
    asegurar_parametros()
    n = 4
    tm, rate, loads, dist, diesel, ltr, avail, p95 = load_matrices(n)
    market = MarketProcess(n)
    ruta = str(tmp_path / "ubdp_tabla_4nodos.npz")
    assert cargar_o_calcular_tabla(market, tm, dist, diesel, S=20, path=ruta, verbose=False, max_horas=0.0) is None
    marcador = ruta.replace(".npz", ".no_tratable.json")
    import json
    info = json.load(open(marcador, encoding="utf-8"))
    assert info["motivo"] == "tiempo" and info["num_nodes"] == n and 0 < info["fraccion"] <= 1
    # Un segundo intento con la misma firma no recalcula: lee el marcador.
    assert cargar_o_calcular_tabla(market, tm, dist, diesel, S=20, path=ruta, verbose=False) is None
    df = ev.evaluar_mundos_simulados(_agente(n), market, None, tm, rate[:TRAIN_DAYS], loads[:TRAIN_DAYS], dist,
                                     diesel, n, ltr, avail, p95, M=4, verbose=False)
    assert "B3" not in set(df["metodo"]) and df["ub_dp"].isna().all()
    res = ev.resumen_metodos(df)
    nt = res[res["metodo"].isin(["B3", "UB-DP"])]
    assert len(nt) == 2 * df["K"].nunique() and (nt["nota"] == ev.NOTA_NO_TRATABLE).all()


# ─────────────────────────────────────────────────────────────────────────────
# Políticas: máscara, determinismo, cotas
# ─────────────────────────────────────────────────────────────────────────────

def test_b1_desempata_por_indice_menor(inst10):
    n = inst10.n
    pol = PoliticaB1(inst10.dist, inst10.diesel)
    mascara = np.zeros(n, dtype=np.int8)
    mascara[[2, 5, 7]] = 1
    obs = Observacion(
        nodo=0, casa=0, tiempo=0.0, k_restante=None,
        mascara=mascara, loads_row=np.zeros(n), rate_row=np.zeros(n),
        precios=np.zeros((n, N_NIVELES)), m=1.0, opciones=_opciones(np.zeros(n), mascara), dia_semana_salida=0,
    )
    # Sin cargas (todo VACÍO) el valor es -costo; con distancias distintas gana la lane más barata.
    costos = -lane_reward(obs.rate_row, obs.loads_row, inst10.dist[0], inst10.diesel[0])
    esperado = int(np.flatnonzero(mascara)[np.argmin(costos[mascara == 1])])
    assert pol.actuar(obs) == esperado
    obs.loads_row = np.zeros(n)
    obs.rate_row = np.zeros(n)
    plano = PoliticaB1(np.ones((n, n)), np.ones((n, n)))
    assert plano.actuar(obs) == 2          # todo empata: el índice menor


def test_b2_libre_equivale_a_un_tope_de_cardinalidad_inalcanzable(inst10):
    """Con k_restante None, B2 elige lo mismo que con un K tan grande que nunca limita."""
    n = inst10.n
    pol = PoliticaB2(inst10.market, inst10.T, inst10.dist, inst10.diesel)
    rng = np.random.default_rng(3)
    for nodo in range(n):
        mascara = np.ones(n, dtype=np.int8)
        mascara[nodo] = 0
        loads_row = rng.integers(0, 4, size=n).astype(float)
        rate_row = rng.uniform(1.5, 3.0, size=n)
        kw = dict(nodo=nodo, casa=nodo, tiempo=0.0, mascara=mascara,
                  loads_row=loads_row, rate_row=rate_row, precios=rate_row[:, None] * (0.8 + 0.5 * U_NIVELES_ARR),
                  m=2.0, opciones=_opciones(loads_row, mascara), dia_semana_salida=2)
        libre = pol.actuar(Observacion(k_restante=None, **kw))
        grande = pol.actuar(Observacion(k_restante=n + 5, **kw))
        assert libre == grande and mascara[libre]


def test_conjunto_de_evaluacion_final_no_comparte_mundos_con_el_de_desarrollo(inst10):
    a = ev.disenar_mundos(inst10.n, 1000, inst10.market, semilla=MARKET_SEED_EVAL)
    b = ev.disenar_mundos(inst10.n, 1000, inst10.market, semilla=MARKET_SEED_EVAL_FINAL)
    assert MARKET_SEED_EVAL_FINAL not in (MARKET_SEED_EVAL, MARKET_SEED_TRAIN)
    assert not set(a["world_seed"].tolist()) & set(b["world_seed"].tolist())
    c = ev.disenar_mundos(inst10.n, 1000, inst10.market, semilla=MARKET_SEED_EVAL_FINAL)
    assert (b["world_seed"] == c["world_seed"]).all()          # reproducible


def _sin_tiempos(df):
    return df.drop(columns=["ms_decision"]).assign(ruta=df["ruta"].astype(str))


def test_politicas_respetan_mascara_y_son_deterministas(inst10):
    # ejecutar_politica lanza RuntimeError si una política elige fuera de la máscara.
    a = inst10.simular(M=8)
    b = inst10.simular(M=8)
    assert set(a["metodo"]) == set(ev._METODOS)
    assert set(ev._METODOS) == {"DRL", "DRL-dest+mediana", "DRL-dest+be_margen", "DRL-dest+miope",
                                "B1", "B2", "B2P", "B3"}
    assert ((a["metodo"] == "B2") & (a["K"] == "libre")).any()          # B2 también con K libre
    pd.testing.assert_frame_equal(_sin_tiempos(a), _sin_tiempos(b))


def test_ub_pi_mayor_o_igual_que_cada_politica(sim10):
    for df in (sim10,):
        ok = ~df["violacion"]
        assert ok.any() and df["ub_pi"].notna().all()
        assert (df.loc[ok, "margen"] <= df.loc[ok, "ub_pi"] + 1e-6).all()


def test_b3_no_supera_ub_dp_mas_error_monte_carlo(sim10):
    for k in (str(K_MAX), "libre"):
        g = sim10[(sim10["metodo"] == "B3") & (sim10["K"] == k)]
        dif = (g["margen"] - g["ub_dp"]).to_numpy()
        se = dif.std(ddof=1) / np.sqrt(len(dif))
        assert dif.mean() <= 3.0 * se, f"K={k}: B3 − UB-DP = {dif.mean():.1f} > 3·{se:.1f}"
        # UB-DP no puede quedar por debajo del mejor baseline por una distancia grande.
        assert g["ub_dp"].mean() >= g["margen"].mean() - 3.0 * se
        # La información perfecta rinde al menos lo que el óptimo no anticipativo (en promedio).
        mundos = sim10[(sim10["K"] == k) & (sim10["metodo"] == "B3")]
        assert mundos["ub_pi"].mean() >= mundos["ub_dp"].mean(), (
            f"K={k}: UB-PI {mundos['ub_pi'].mean():.1f} < UB-DP {mundos['ub_dp'].mean():.1f}")


# ─────────────────────────────────────────────────────────────────────────────
# UB-PI: alineación con el entorno
# ─────────────────────────────────────────────────────────────────────────────

class _RutaConInformacionPerfecta:
    """Repite una ruta cotizando en cada arco el nivel más alto que gana (lee el competidor del entorno: solo
    para alinear UB-PI con el entorno, ninguna política lo puede ver)."""

    def __init__(self, ruta, env):
        self._ruta, self._pos, self._env = ruta, 0, env

    def actuar(self, obs) -> int:
        self._pos += 1
        return int(self._ruta[self._pos])                     # el mismo nodo = esperar

    def cotizar(self, obs, j) -> int:
        n = int(obs.intentos[j])
        if j == obs.nodo or n == 0:
            return OPCION_VACIO                               # esperar, o la lane se recorre vacía
        u = self._env.u_comp_cargas[0, j, int(self._env.perdidas[0, j]):][:n]
        if not gana(U_NIVELES_ARR, u.max()).any():
            return OPCION_VACIO
        if u.argmax() > 0 and not gana(U_NIVELES_ARR[-1], u[0]):
            self._pos -= 1                                    # pierde esta carga (p90) para llegar a la mejor
            return N_NIVELES - 1
        return int(np.flatnonzero(gana(U_NIVELES_ARR, u[0])).max())


def test_ub_pi_alineado_con_el_entorno(inst10):
    """El entorno acepta la ruta de UB-PI (en cada lane pierde a propósito las cargas peores y cotiza la mejor en el
    nivel más alto que gana) y su margen no supera el valor de UB-PI, que no cobra esos 15 min por carga.

    UB-PI redondea los tiempos hacia abajo (franjas de 1 min) y usa la envolvente de días cerca de
    las fronteras, así que su ruta puede exceder MAX_DURATION en unos minutos (entonces el entorno
    no la permite) y su valor puede ser mayor que el margen de la propia ruta; nunca menor."""
    inst = inst10
    env = ev.make_sim_env(inst.tm, inst.rate[:TRAIN_DAYS], inst.loads[:TRAIN_DAYS],
                          inst.dist, inst.diesel, inst.n, inst.market)
    dis = ev.disenar_mundos(inst.n, 12, inst.market)
    comprobadas = 0
    for m in range(12):
        h, dow = int(dis["nodo"][m]), int(dis["dow"][m])
        dia, ws = int(dis["start_day"][m]), int(dis["world_seed"][m])
        mundo = ev.mundo_ub_pi(inst.market.mundo_cotizacion(ws, 6, dia))
        for K in (K_MAX, None):
            rew, ruta = ev.ub_pi(inst.T, inst.dist, inst.diesel, h, K, mundo)
            if len(ruta) == 1:
                continue                                                       # no sale
            viajes = [n for k, n in enumerate(ruta) if k == 0 or n != ruta[k - 1]]
            duracion = sum(inst.T[a, b] for a, b in zip(viajes[:-1], viajes[1:]))
            if duracion > MAX_DURATION or ruta != viajes:
                continue          # redondeo de franjas o esperas: el día real puede diferir de la franja
            res = ev.ejecutar_politica(_RutaConInformacionPerfecta(ruta, env), env, h, dia, K, ws,
                                       inst.dist, inst.diesel, dow)
            if res["violacion"] or res["ruta"] != viajes or duracion + res["horas_rechazo"] > MAX_DURATION:
                continue                                                       # los 15 min por carga perdida no cupieron
            assert res["margen"] <= rew + 1e-6
            comprobadas += 1
    assert comprobadas >= 6


# ─────────────────────────────────────────────────────────────────────────────
# Diseño de mundos y estadística
# ─────────────────────────────────────────────────────────────────────────────

def test_diseno_de_mundos_balanceado(inst10):
    n, M = inst10.n, 100
    dis = ev.disenar_mundos(n, M, inst10.market)
    cuenta_nodos = np.bincount(dis["nodo"], minlength=n)
    cuenta_dow = np.bincount(dis["dow"], minlength=7)
    assert cuenta_nodos.min() >= M // n and cuenta_nodos.max() <= -(-M // n)
    assert cuenta_dow.min() >= M // 7 and cuenta_dow.max() <= -(-M // 7)
    assert (inst10.market.dia_semana(dis["start_day"]) == dis["dow"]).all()
    assert (dis["start_day"] <= TRAIN_DAYS - ev.MARKET_DIAS_EPISODIO).all()      # ningún episodio se queda sin días
    # Reproducible y con semillas distintas de las de entrenamiento.
    otra = ev.disenar_mundos(n, M, inst10.market)
    assert all((dis[k] == otra[k]).all() for k in dis)
    entren = np.random.default_rng(MARKET_SEED_TRAIN).integers(0, 2**31 - 1, size=10_000)
    assert not np.isin(dis["world_seed"], entren).any()
    # El día de la semana no queda atado al nodo.
    assert len(set(zip(dis["nodo"], dis["dow"]))) > n


def test_holm_y_comparacion_pareada():
    p = np.array([0.01, 0.04, 0.03])
    assert st.holm(p) == pytest.approx([0.03, 0.06, 0.06])
    rng = np.random.default_rng(0)
    base = rng.normal(1000, 100, size=60)
    filas = []
    for k in ("5", "libre"):
        for i, b in enumerate(base):
            filas.append(dict(modo="simulado", K=k, mundo=i, metodo="DRL", margen=b + 50 + rng.normal(0, 20)))
            filas.append(dict(modo="simulado", K=k, mundo=i, metodo="B1", margen=b))
            filas.append(dict(modo="simulado", K=k, mundo=i, metodo="B3", margen=b + 100))
            filas.append(dict(modo="simulado", K=k, mundo=i, metodo="B2", margen=b + 20))
            filas.append(dict(modo="simulado", K=k, mundo=i, metodo="DRL-dest+miope", margen=b + 40))
    df = pd.DataFrame(filas)
    tabla = st.comparar_con_baselines(df, B=2000)
    assert set(tabla["comparacion"]) == {"DRL − B1", "DRL − B2", "DRL − B3", "DRL − DRL-dest+miope"}
    assert len(tabla[tabla["K"] == "libre"]) == 4                 # B2 también con K libre
    # Comparación B: familia aparte, con su propio Holm (una sola comparación ⇒ p_holm = p).
    b = tabla[tabla["familia"] == "precio"]
    assert set(b["comparacion"]) == {"DRL − DRL-dest+miope"}
    assert np.allclose(b["p_holm"], b["p_una_cola"])
    assert set(tabla.loc[tabla["familia"] == "ruteo", "comparacion"]) == {"DRL − B1", "DRL − B2", "DRL − B3"}
    fila = tabla[(tabla["K"] == "5") & (tabla["comparacion"] == "DRL − B1")].iloc[0]
    assert fila["ic95_inf"] < fila["dif_media_usd"] < fila["ic95_sup"]
    assert fila["p_una_cola"] < 0.001 and fila["p_holm"] >= fila["p_una_cola"]
    assert tabla[tabla["comparacion"] == "DRL − B3"]["dif_media_usd"].max() < 0       # B3 le gana
    pd.testing.assert_frame_equal(tabla, st.comparar_con_baselines(df, B=2000))        # semilla fija


def test_marca_de_ub_dp_floja_y_cotas_separadas():
    filas = []
    for k, dp in (("5", 90.0), ("libre", 150.0)):          # en "libre" UB-DP > UB-PI
        for i in range(10):
            for metodo, m in (("DRL", 50.0), ("B3", 60.0)):
                filas.append(dict(modo="simulado", K=k, mundo=i, metodo=metodo, margen=m, violacion=False,
                                  frac_millas_cargadas=1.0, millas_totales=500.0, millas_cargadas=500.0, n_cargas=2, duracion=50.0, ms_decision=1.0,
                                  ruta=[0, 1, 0], n_cotizaciones=3, n_ganadas=2, horas_rechazo=0.25,
                                  niveles=[1, 4, 2], usd_mi_sobre_mediana=0.1, dinero_en_la_mesa=30.0,
                                  n_esperas=1, horas_espera=5.0, n_reposiciones=0, n_bajo_be=1, usd_bajo_be=-20.0,
                                  ub_dp=dp, ub_pi=100.0))
    df = pd.DataFrame(filas)
    res = ev.resumen_metodos(df)
    assert set(res.loc[res["K"] == "libre", "nota"].dropna()) == {st.NOTA_UB_DP_FLOJA}
    assert res.loc[res["K"] == "5", "nota"].isna().all()
    # Cada gap contra su propia cota; las cotas aparecen como filas separadas.
    b3 = res[(res["K"] == "5") & (res["metodo"] == "B3")].iloc[0]
    assert b3["gap_ub_dp_pct"] == pytest.approx((90 - 60) / 90 * 100)
    assert b3["gap_ub_pi_pct"] == pytest.approx((100 - 60) / 100 * 100)
    assert {"UB-PI", "UB-DP"} <= set(res["metodo"])
    assert b3["tasa_aceptacion_pct"] == pytest.approx(200 / 3) and b3["cotizaciones_por_arco"] == pytest.approx(1.5)
    assert b3["nivel_medio"] == pytest.approx(7 / 3) and b3["horas_rechazo"] == pytest.approx(0.25)
    # IC bootstrap de la media: contiene la media y es reproducible (semilla fija).
    f = res[(res["K"] == "5") & (res["metodo"] == "B3")].iloc[0]
    assert f["margen_ic95_low"] <= f["margen_medio"] <= f["margen_ic95_high"]
    pd.testing.assert_frame_equal(res, ev.resumen_metodos(df))
    rng = np.random.default_rng(1)
    x = rng.normal(500, 100, size=200)
    lo, hi = st.ic_bootstrap_media(x)
    assert lo < x.mean() < hi and hi - lo == pytest.approx(2 * 1.96 * 100 / np.sqrt(200), rel=0.25)


# ─────────────────────────────────────────────────────────────────────────────
# Limpieza del repositorio
# ─────────────────────────────────────────────────────────────────────────────

_NOMBRES_BORRADOS = [
    r"solve_HGA_LNS_metaheuristic", r"solve_label_setting_exact", r"solve_heuristic_rolling_horizon\w*",
    r"solve_mc_rollout_stochastic", r"run_solver_comparison\w*", r"N_DAYS_PER_NODE",
    r"solve_heuristic\b", r"_label_setting_core\b", r"_close_cycle_and_finalize", r"_rh_greedy_stoch_from_state",
    r"solve_label_setting_oracle", r"_label_setting_core_stochastic", r"simulate_route_reward",
    r"\bvisited_(set|mask)\b", r"_visited_(set|mask)\b",
    r"_repair_route_lns_best_position", r"_generate_route_nearest_neighbor", r"_generate_random_feasible_route",
    r"_crossover_routes", r"_apply_2opt_mutation", r"_apply_lns_mutation", r"\bRHResult\b", r"\bMetaResult\b",
    r"DRL Det\b", r"\bLS-Exact\b", r"\bHGA-LNS\b", r"\bMC-Rollout\b", r"RH-Greedy", r"RH-Lookahead",
]


def test_ningun_nombre_borrado_sigue_referenciado():
    esta = os.path.abspath(__file__)
    archivos = [f for f in glob.glob(os.path.join(RAIZ, "**", "*.py"), recursive=True) if os.path.abspath(f) != esta]
    archivos.append(os.path.join(RAIZ, "README.md"))
    patron = re.compile("|".join(_NOMBRES_BORRADOS))
    hallazgos = []
    for f in archivos:
        with open(f, encoding="utf-8") as fh:
            for n, linea in enumerate(fh, 1):
                if patron.search(linea):
                    hallazgos.append(f"{os.path.relpath(f, RAIZ)}:{n}: {linea.strip()[:100]}")
    assert not hallazgos, "Referencias a baselines borrados:\n" + "\n".join(hallazgos)


# ─────────────────────────────────────────────────────────────────────────────
# Lo que cuesta perder la cotización y las fallas del ciclo
# ─────────────────────────────────────────────────────────────────────────────

def _obs_de_prueba(nodo, casa, n=3, vacio=True):
    from baselines.politicas import Observacion
    opciones = np.zeros((n, N_NIVELES + 1), dtype=bool)
    opciones[:, :N_NIVELES] = True
    opciones[nodo] = False
    if vacio:
        opciones[casa, OPCION_VACIO] = nodo != casa
        opciones[nodo, OPCION_VACIO] = True
    return Observacion(nodo=nodo, casa=casa, tiempo=0.0, k_restante=None, mascara=np.ones(n, dtype=np.int8),
                       loads_row=np.ones(n), rate_row=np.full(n, 2.0),
                       precios=np.full((n, 1), 2.0) * (0.7 + 0.6 * U_NIVELES_ARR), m=2.0,
                       opciones=opciones, dia_semana_salida=0)


def test_precio_de_la_vuelta_cuenta_volver_vacio():
    """En la vuelta a casa todas las reglas maximizan el ingreso esperado P·p (el costo se paga igual); en los
    demás tramos cada regla usa su nivel y perder vale 0."""
    from baselines.politicas import costo_de_perder, cotizar_regla, valor_inmediato
    from pricing import lane_cost_row, nivel_regla, prob_ganar
    dist, diesel = np.full((3, 3), 500.0), np.full((3, 3), 2.5)
    obs = _obs_de_prueba(nodo=1, casa=0)
    costo = float(lane_cost_row(dist[1, 0], diesel[1, 0]))
    P = prob_ganar(U_NIVELES_ARR, obs.m)
    esperado = int(np.argmax(P * obs.precios[0]))
    assert costo_de_perder(obs, 0, costo) == -costo and costo_de_perder(obs, 2, costo) == 0.0
    for regla in ("mediana", "be_margen", "miope"):
        assert cotizar_regla(obs, 0, dist, diesel, regla) == esperado
        assert cotizar_regla(obs, 2, dist, diesel, regla) == int(nivel_regla(regla, obs.precios[2], 500.0, costo, obs.m))
    v = valor_inmediato(obs, dist, diesel, cierre=True)                                # B2 / B2P
    assert v[0] == pytest.approx(float((P * obs.precios[0] * 500.0).max() - costo))
    v1 = valor_inmediato(obs, dist, diesel)                                            # B1: contra no moverse
    assert v1[0] == pytest.approx(float((P * (obs.precios[0] * 500.0 - costo)).max()))
    assert costo_de_perder(_obs_de_prueba(nodo=0, casa=0), 2, costo) == 0.0            # en casa puede esperar


def test_ciclo_fallido_se_reporta_con_la_penalizacion(sim10):
    """margen de un ciclo fallido = −3000 USD (la penalización de la DP y del entrenamiento): nunca 0."""
    assert PENALIZACION_USD == -3000.0
    f = sim10[sim10["violacion"]]
    assert (f["margen"] <= PENALIZACION_USD + 1e-6).all()
    sin_salir = sim10[(sim10["duracion"] == 0) & ~sim10["violacion"]]
    assert (sin_salir["margen"] == 0).all()


# ─────────────────────────────────────────────────────────────────────────────
# Diagnóstico "rutas del DRL + precio de la tabla de B3" (DRL-dest+dp)
# ─────────────────────────────────────────────────────────────────────────────

def test_precio_en_de_b3_coincide_con_su_opcion_y_dp_respeta_la_mascara(inst10):
    i = inst10
    ctx = ev.ContextoEval(i.agent, i.market, i.tabla, i.tm, i.dist, i.diesel, i.n, i.ltr, i.avail, i.p95)
    env = ev.make_sim_env(i.tm, i.rate[:TRAIN_DAYS], i.loads[:TRAIN_DAYS], i.dist, i.diesel, i.n, i.market)
    b3 = ctx.politicas["B3"]
    dis = ev.disenar_mundos(i.n, 6, i.market, semilla=7)

    class Espia:
        """B3 con una aserción: fijar desde afuera el destino que B3 elegiría da la misma opción de precio."""
        nombre = "espia"
        pasos = 0

        def actuar(self, obs):
            a = b3.actuar(obs)
            assert b3.precio_en(obs, a) == b3._opcion
            Espia.pasos += 1
            return a

        def cotizar(self, obs, j):
            return b3._opcion

    for m in range(6):
        args = (env, int(dis["nodo"][m]), int(dis["start_day"][m]), None, int(dis["world_seed"][m]),
                i.dist, i.diesel, int(dis["dow"][m]))
        ev.ejecutar_politica(Espia(), args[0], *args[1:])
    assert Espia.pasos > 0

    # DRL-dest+dp pasa por el arnés (que lanza si la opción queda fuera de la máscara) y existe solo con tabla
    dp = ctx.politica_dp(env)
    assert dp is not None and dp.nombre == "DRL-dest+dp"
    for m in range(6):
        ev.ejecutar_politica(dp, env, int(dis["nodo"][m]), int(dis["start_day"][m]), None,
                             int(dis["world_seed"][m]), i.dist, i.diesel, int(dis["dow"][m]))
    sin_tabla = ev.ContextoEval(i.agent, i.market, None, i.tm, i.dist, i.diesel, i.n, i.ltr, i.avail, i.p95)
    assert sin_tabla.politica_dp(env) is None
