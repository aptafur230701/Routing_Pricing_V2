"""
market_process.py
=================
Proceso exógeno de cargas y tarifas (fase 4 de proceso_exogeno_cargas_tarifas.md).

Cuando el agente llega al nodo i el día d, el proceso sortea para cada lane
(i, j) que sale del nodo:
  1. cuántas cargas hay: NegBin(mu_ij · s_d, phi_h)                    → loads_row (N,)
  2. la tarifa por milla del modelo con estado de mercado por nodo     → rate_row  (N,)

         tarifa(i→j, d) = media_lane(ij) + índice(i, d) + residuo(ij)
         índice(i, d)   = lento(i) + rápido(i, d)
         rápido(i, d)   = ρ · rápido(i, d−1) + ε(i, d),   ε ~ Normal(0, σ_i²)

  · índice : estado del mercado de salida del nodo i, neto del nivel nacional (fit_market_process.py).
  · lento  : estado de fondo del nodo. En cada mundo se toma el vector de todos los nodos de un día de
             escenario al azar y queda fijo durante todo el episodio.
  · rápido : AR(1) con un solo ρ y una σ_i por nodo; los nodos evolucionan de forma independiente, desde
             el día 0 con la distribución estacionaria Normal(0, σ_i² / (1 − ρ²)).
  · residuo: un día al azar del histórico de residuos de esa lane (independiente por lane); incluye la
             variación nacional y la propia de la lane.
Las tarifas por debajo del mínimo histórico de la lane se recortan a ese mínimo (se cuentan en
`estadisticas_recorte`). La tarifa se sortea en todas las lanes; el entorno solo la usa donde loads_row > 0.

El proceso NO depende de las acciones del agente. Tres modos de uso:

  · Entrenamiento : `sample_arrivals_por_mundo(nodos, días, semillas)` — un "mundo" por episodio. Cada
    sorteo es función pura de (semilla del mundo, nodo, día), así que (a) la salida no depende de qué decida
    el agente ni de cuántos episodios sigan activos, y (b) el rollout muestreado y el greedy del baseline
    self-critical, con la misma semilla de mundo, ven la misma realización. El mundo (lento y trayectoria
    del rápido) sale solo de la semilla; el índice de todos los nodos en un día es `indice_mundo`.
  · Evaluación / baselines : `sample_world(semilla, num_días, día_inicial)` genera un mundo completo
    [D, N, N]; es el mismo mundo que ve el entorno con esa semilla de mundo.
  · Política que ignora el mercado (DP de UB-DP y B2): `sample_arrivals(nodos, días, rng)` sortea en CADA
    llegada un escenario lento al azar y un rápido de su distribución estacionaria, sin memoria. Representa
    una política que conoce la distribución de las tarifas pero ignora el estado del mercado: el índice
    marginal es el mismo que el de los mundos, pero no hay persistencia ni lento fijo entre llegadas.

Replay con días reales: `indice_historico` calcula el índice observado de cada día con la definición del
ajuste (194 nodos, media_lane de entrenamiento, tarifas reales de ese día), recortado a los N nodos.

Toda la aleatoriedad sale de numpy Generators propios (nunca del estado global).
"""

import hashlib
import os
from collections import OrderedDict
from dataclasses import dataclass

import numpy as np

from config import MARKET_DIA_MAX, TRAIN_DAYS
from fit_market_process import dia_de_semana, indice_observado, params_path, main as _ajustar

# Etiqueta de la clave del generador de un mundo ([semilla, _TAG_MUNDO]); mayor que cualquier nodo para no
# coincidir nunca con la clave de una fila [semilla, nodo, día].
_TAG_MUNDO = 1_000_000_007
_VERSION_GENERADOR = "mercado-v1"          # entra en la firma: cambiar al modificar el generador
_CACHE_MUNDOS_BYTES = 256_000_000          # tope del caché de mundos (LRU)
_CLAVES_AJUSTE = ("media_lane", "min_lane", "lento_escenarios", "rho", "sigma", "residuos", "indice_std",
                  "universo_indice", "ventana_lento", "frac_var_lento")


@dataclass
class MarketParams:
    """Parámetros del proceso, ya recortados a num_nodes."""
    s:        np.ndarray   # (7,)        factor del día de la semana (lunes = 0), promedio 1
    mu:       np.ndarray   # (N, N)      tasa diaria base de la lane
    phi:      np.ndarray   # (G,)        dispersión por grupo de volumen
    grupo:    np.ndarray   # (N, N) int  grupo de volumen h de cada lane
    media_lane: np.ndarray  # (N, N)     tarifa media de la lane (días de ajuste)
    min_lane:   np.ndarray  # (N, N)     tarifa mínima histórica de la lane (piso de las tarifas simuladas)
    residuos:   np.ndarray  # (D, N, N)  residuos históricos tarifa − media_lane − índice, por lane
    lento_escenarios: np.ndarray  # (E, N) vector lento de cada día de escenario
    rho:        float       # persistencia del rápido
    sigma:      np.ndarray  # (N,)       desviación de la innovación del rápido por nodo
    indice_std: float       # desviación del índice histórico (normaliza la feature del agente)
    frac_var_lento: float   # fracción de la varianza del índice que es lento (ajuste)
    fecha_inicio: str


class MarketProcess:
    """Proceso exógeno de cargas y tarifas estimado del histórico.

    Parámetros
    ----------
    num_nodes : se usan los primeros `num_nodes` nodos (igual que load_matrices).
                Los parámetros se estiman con los 194 y se recortan aquí; el mundo (lento y rápido) se
                genera para los 194 y se recorta, así que el índice de un nodo no depende de num_nodes.
    path      : ruta del .npz (por defecto datos/market_params.npz).
    """

    def __init__(self, num_nodes: int, path: str = None):
        path = path or params_path()
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"{path} no existe. Corre `python fit_market_process.py` primero."
            )
        with np.load(path) as z:
            falta = [k for k in _CLAVES_AJUSTE if k not in z.files]
            if falta:
                raise ValueError(f"{path} es de un ajuste anterior (faltan {falta}). "
                                 "Corre `python fit_market_process.py`.")
            n_total = z["mu"].shape[0]
            if num_nodes > n_total:
                raise ValueError(f"num_nodes={num_nodes} excede los {n_total} nodos ajustados")
            n = num_nodes
            self.params = MarketParams(
                s=z["s"].astype(np.float64),
                mu=z["mu"][:n, :n].astype(np.float64),
                phi=z["phi"].astype(np.float64),
                grupo=z["grupo"][:n, :n].astype(np.int64),
                media_lane=z["media_lane"][:n, :n].astype(np.float64),
                min_lane=z["min_lane"][:n, :n].astype(np.float64),
                residuos=np.ascontiguousarray(z["residuos"][:, :n, :n]),
                lento_escenarios=z["lento_escenarios"][:, :n].astype(np.float64),
                rho=float(z["rho"]),
                sigma=z["sigma"][:n].astype(np.float64),
                indice_std=float(z["indice_std"]),
                frac_var_lento=float(z["frac_var_lento"]),
                fecha_inicio=str(z["fecha_inicio"]),
            )
            # Arreglos de los 194 nodos: el mundo se genera completo y se recorta a n (ver _mundo) y el
            # índice de un día real se calcula con todos los nodos (indice_historico).
            self._n_total = n_total
            self._lento_total = z["lento_escenarios"].astype(np.float64)       # (E, 194)
            self._sigma_total = z["sigma"].astype(np.float64)                  # (194,)
            self._media_total = z["media_lane"].astype(np.float64)             # (194, 194)
            self._universo_indice_total = z["universo_indice"].astype(bool)    # (194, 194)
        self.num_nodes = num_nodes
        self.num_dias_residuos = self.params.residuos.shape[0]
        self.num_escenarios = self.params.lento_escenarios.shape[0]
        self.n_dias_mundo = MARKET_DIA_MAX + 1          # días 0 … MARKET_DIA_MAX de la trayectoria del rápido
        # phi de cada lane según su grupo de volumen (N, N).
        self._phi_lane = self.params.phi[self.params.grupo]
        self._cols = np.arange(num_nodes)
        # desviación estacionaria del rápido por nodo: σ_i / sqrt(1 − ρ²)
        self._sd_estacionaria = self._sigma_total / np.sqrt(1.0 - self.params.rho ** 2)
        self._universo_lane = self._universo_indice_total[:num_nodes, :num_nodes]
        self._mundos = OrderedDict()                    # semilla → (lento (N,), índice (n_dias_mundo, N))
        self._mundo_bytes = (num_nodes * (self.n_dias_mundo + 1)) * 8
        self.reiniciar_recorte()

    # ── utilidades ───────────────────────────────────────────────────────────

    def dia_semana(self, dias) -> np.ndarray:
        """Día de la semana (lunes = 0) de los índices de día `dias`."""
        return dia_de_semana(dias, self.params.fecha_inicio)

    def firma_params(self) -> str:
        """Huella de TODO lo que determina el proceso (cargas, tarifas y generador).

        La usan la tabla V de UB-DP y el caché de la evaluación para recalcularse solos cuando cambian los
        datos o el generador."""
        p, h = self.params, hashlib.sha256()
        for a in (p.s, p.mu, p.phi, p.grupo, p.media_lane, p.min_lane, p.residuos, p.lento_escenarios,
                  p.sigma, np.array([p.rho, p.indice_std, p.frac_var_lento])):
            h.update(np.ascontiguousarray(a).tobytes())
        h.update(repr((_VERSION_GENERADOR, self.num_nodes, self._n_total, self.n_dias_mundo,
                       p.fecha_inicio)).encode())
        return h.hexdigest()

    def reiniciar_recorte(self) -> None:
        """Pone en cero los contadores de tarifas recortadas al mínimo histórico de la lane."""
        self.n_tarifas = self.n_recortadas = 0                    # todas las lanes sorteadas
        self.n_tarifas_universo = self.n_recortadas_universo = 0  # solo lanes del universo (≥ 1 carga, sin diagonal)

    def estadisticas_recorte(self) -> dict:
        """Tarifas sorteadas y recortadas desde el último `reiniciar_recorte` (todas las lanes y las del universo)."""
        return dict(
            tarifas=self.n_tarifas, recortadas=self.n_recortadas,
            pct=100.0 * self.n_recortadas / max(self.n_tarifas, 1),
            tarifas_universo=self.n_tarifas_universo, recortadas_universo=self.n_recortadas_universo,
            pct_universo=100.0 * self.n_recortadas_universo / max(self.n_tarifas_universo, 1),
        )

    # ── Mundo: lento fijo + trayectoria del rápido ───────────────────────────

    def _mundo(self, semilla: int) -> tuple:
        """(lento (N,), índice (n_dias_mundo, N)) del mundo `semilla`; función pura de la semilla.

        Elige un día de escenario al azar (su vector lento queda fijo en todo el episodio) y genera el
        rápido de todos los nodos desde el día 0 con la distribución estacionaria. Se genera para los 194
        nodos y se recorta, así que no depende de num_nodes. El caché solo evita recalcular."""
        semilla = int(semilla)
        hit = self._mundos.get(semilla)
        if hit is not None:
            self._mundos.move_to_end(semilla)
            return hit
        rng = np.random.Generator(np.random.PCG64([semilla, _TAG_MUNDO]))
        esc = int(rng.integers(0, self.num_escenarios))
        lento = self._lento_total[esc]                                            # (194,)
        z = rng.standard_normal((self.n_dias_mundo, self._n_total))
        rapido = np.empty_like(z)
        rapido[0] = self._sd_estacionaria * z[0]
        rho, sigma = self.params.rho, self._sigma_total
        for d in range(1, self.n_dias_mundo):
            rapido[d] = rho * rapido[d - 1] + sigma * z[d]
        n = self.num_nodes
        res = (lento[:n].copy(), lento[None, :n] + rapido[:, :n])
        self._mundos[semilla] = res
        if len(self._mundos) * self._mundo_bytes > _CACHE_MUNDOS_BYTES:
            self._mundos.popitem(last=False)
        return res

    def lento_mundo(self, semilla: int) -> np.ndarray:
        """(N,) vector lento del mundo (fijo durante todo el episodio)."""
        return self._mundo(semilla)[0]

    def indice_mundo(self, semilla: int, dia: int) -> np.ndarray:
        """(N,) índice de mercado de TODOS los nodos el día `dia` del mundo `semilla`."""
        dia = int(dia)
        if not 0 <= dia <= MARKET_DIA_MAX:
            raise ValueError(f"día {dia} fuera del mundo simulado (0 … {MARKET_DIA_MAX})")
        return self._mundo(semilla)[1][dia]

    def indice_mundo_lote(self, semillas, dias) -> np.ndarray:
        """(B, N) índice de todos los nodos para cada par (semilla de mundo, día)."""
        return np.stack([self.indice_mundo(s, d) for s, d in zip(semillas, dias)])

    def indice_historico(self, dia_ini: int = 0, dia_fin: int = None) -> np.ndarray:
        """(D, N) índice observado de los días reales [dia_ini, dia_fin) del histórico.

        Definición del ajuste (fit_market_process.indice_observado): promedio sobre las lanes salientes del
        universo de tarifa − media_lane de entrenamiento, menos el nivel nacional sobre los 194 nodos, con
        las tarifas reales del día. Después se recorta a los N nodos."""
        rate = np.load(os.path.join(os.path.dirname(params_path()), "rate.npy"), mmap_mode="r")
        rate = np.asarray(rate[dia_ini:dia_fin], dtype=np.float64)
        return indice_observado(rate, self._media_total, self._universo_indice_total)[:, :self.num_nodes]

    # ── Tarifa ───────────────────────────────────────────────────────────────

    def _armar_tarifa(self, nodos, indice, dia_res):
        """Tarifa = media_lane + índice + residuo, recortada al mínimo histórico; float32.

        nodos (B,), indice (B,), dia_res (B, N) día de residuo por lane. Devuelve (B, N)."""
        p = self.params
        n = nodos[:, None]
        r = p.media_lane[nodos] + indice[:, None] + p.residuos[dia_res, n, self._cols[None, :]]
        piso = p.min_lane[nodos]
        bajo = r < piso
        self.n_tarifas += r.size
        self.n_recortadas += int(bajo.sum())
        uni = self._universo_lane[nodos]
        self.n_tarifas_universo += int(uni.sum())
        self.n_recortadas_universo += int((bajo & uni).sum())
        return np.maximum(r, piso).astype(np.float32)

    # ── Llegada a un nodo (una fila) ─────────────────────────────────────────

    def _sortear_cargas(self, i: int, dow: int, rng: np.random.Generator) -> np.ndarray:
        m   = self.params.mu[i] * self.params.s[dow]                  # (N,) media por lane
        phi = self._phi_lane[i]                                       # (N,) dispersión por lane
        # m = 0 (lane sin cargas en el histórico) ⇒ p = 1 ⇒ siempre 0 cargas.
        return rng.negative_binomial(phi, phi / (phi + m))            # (N,)

    def _sortear_fila(self, i: int, dow: int, dia: int, semilla: int, rng: np.random.Generator) -> tuple:
        """llegada_a_nodo en un mundo: (loads_row, rate_row) del nodo i el día `dia` del mundo `semilla`.

        `rng` es el generador de la fila [semilla, nodo, día]: primero las cargas y después el día de residuo
        de cada lane; así las cargas son las del generador anterior para la misma terna."""
        loads_row = self._sortear_cargas(i, dow, rng)
        dia_res = rng.integers(0, self.num_dias_residuos, size=self.num_nodes)       # (N,)
        indice = self.indice_mundo(semilla, dia)[i]
        rate_row = self._armar_tarifa(np.array([i]), np.array([indice]), dia_res[None])[0]
        return loads_row, rate_row

    def llegada_a_nodo(self, i: int, dia: int, rng: np.random.Generator) -> tuple:
        """Devuelve (loads_row, rate_row) para las lanes que salen del nodo i el día `dia`, sin memoria
        (como `sample_arrivals` con un solo nodo)."""
        loads, rates = self.sample_arrivals([int(i)], [int(dia)], rng)
        return loads[0], rates[0]

    # ── Política que ignora el mercado (DP y B2) ─────────────────────────────

    def sample_arrivals(self, nodos, dias, rng: np.random.Generator) -> tuple:
        """Llegadas por lotes con un único Generator, SIN estado de mercado.

        En cada llegada se sortea un escenario lento al azar y un rápido de su distribución estacionaria
        Normal(0, σ_i² / (1 − ρ²)), independientes entre llegadas: representa una política que conoce la
        distribución de las tarifas pero ignora el estado del mercado (la que supone la DP de UB-DP y B2).
        El índice marginal coincide con el de los mundos de `sample_world`; no hay persistencia.

        nodos, dias : (B,) nodo de llegada y día (índice) de llegada (solo el día de la semana importa).
        Devuelve loads (B, N) int64 y rates (B, N) float32.
        El resultado depende del orden y tamaño del lote (consume `rng`); para sorteos independientes de las
        decisiones del agente usa `sample_arrivals_por_mundo`.
        """
        nodos = np.asarray(nodos, dtype=np.int64)
        dow   = self.dia_semana(dias)                                       # (B,)
        m     = self.params.mu[nodos] * self.params.s[dow][:, None]         # (B, N)
        phi   = self._phi_lane[nodos]                                       # (B, N)
        loads = rng.negative_binomial(phi, phi / (phi + m))                 # (B, N)
        dia_res = rng.integers(0, self.num_dias_residuos, size=m.shape)     # (B, N)
        esc = rng.integers(0, self.num_escenarios, size=len(nodos))         # (B,)
        indice = (self.params.lento_escenarios[esc, nodos]
                  + self._sd_estacionaria[nodos] * rng.standard_normal(len(nodos)))
        return loads.astype(np.int64), self._armar_tarifa(nodos, indice, dia_res)

    # ── Modo entrenamiento ───────────────────────────────────────────────────

    def sample_arrivals_por_mundo(self, nodos, dias, semillas) -> tuple:
        """Llegadas por lotes, cada fila determinista dada (semilla de mundo, nodo, día).

        nodos, dias, semillas : (B,). `semillas` es la semilla del mundo de cada
        episodio (entero ≥ 0). Dos filas con la misma terna ven exactamente el
        mismo sorteo, sin importar el resto del lote, el orden de las llamadas ni
        las acciones del agente.
        Devuelve loads (B, N) int64 y rates (B, N) float32.
        """
        nodos    = np.asarray(nodos,    dtype=np.int64)
        dias     = np.asarray(dias,     dtype=np.int64)
        semillas = np.asarray(semillas, dtype=np.int64)
        dow = self.dia_semana(dias)
        B, N = len(nodos), self.num_nodes
        loads = np.empty((B, N), dtype=np.int64)
        rates = np.empty((B, N), dtype=np.float32)
        for b in range(B):
            rng = np.random.Generator(np.random.PCG64(
                [int(semillas[b]), int(nodos[b]), int(dias[b])]
            ))
            loads[b], rates[b] = self._sortear_fila(int(nodos[b]), int(dow[b]), int(dias[b]), int(semillas[b]), rng)
        return loads, rates

    # ── Modo evaluación / baselines ──────────────────────────────────────────

    def sample_world(self, semilla: int, num_dias: int, dia_inicial: int = 0, devolver_indice: bool = False):
        """Mundo completo: loads_sim y rate_sim de forma [D, N, N], determinista dada la semilla.

        loads_sim[d, i] y rate_sim[d, i] son la fila que ve el agente al llegar
        al nodo i el día `dia_inicial + d`: el mismo sorteo que entrega
        `sample_arrivals_por_mundo` con esa semilla de mundo. Sustituto directo
        de loads_stack / rate_stack (el día 0 del mundo es `dia_inicial`).
        Con devolver_indice=True devuelve además el índice de mercado usado, (D, N)
        (el de `indice_mundo`), como tercer elemento.
        """
        N = self.num_nodes
        nodos = np.arange(N)
        loads_sim = np.empty((num_dias, N, N), dtype=np.float64)
        rate_sim  = np.empty((num_dias, N, N), dtype=np.float64)
        for d in range(num_dias):
            l, r = self.sample_arrivals_por_mundo(
                nodos, np.full(N, dia_inicial + d), np.full(N, semilla),
            )
            loads_sim[d], rate_sim[d] = l, r
        if not devolver_indice:
            return loads_sim, rate_sim
        indice = np.stack([self.indice_mundo(semilla, dia_inicial + d) for d in range(num_dias)])
        return loads_sim, rate_sim, indice


def asegurar_parametros() -> None:
    """Corre el ajuste (fit_market_process) si datos/market_params.npz no existe, es de un ajuste anterior
    (sin los parámetros del modelo de tarifas o con otra ventana del lento) o se ajustó con días distintos
    de los TRAIN_DAYS de entrenamiento (p. ej. un archivo viejo con los 120)."""
    from config import VENTANA_LENTO_DIAS
    if os.path.exists(params_path()):
        with np.load(params_path()) as z:
            if (all(k in z.files for k in _CLAVES_AJUSTE)
                    and len(z["dias_usados"]) == TRAIN_DAYS
                    and int(z["ventana_lento"]) == VENTANA_LENTO_DIAS):
                return
        print("market_params.npz no corresponde al ajuste actual (días, ventana o modelo de tarifas): reajustando...")
    else:
        print("market_params.npz no existe: corriendo el ajuste del proceso exógeno...")
    _ajustar()
