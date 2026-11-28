"""
test_modelo.py
==============
Entradas y salidas del agente (model/, greedy_rollout.py, transfer_learning.py), en tensores:

  · features por nodo (B, N, N_NODE_FEATURES) finitas y en rango; las de continuación (6–13) coinciden con una
    referencia con loops; el conjunto factible excluye lanes sin vuelta; casos especiales (espera, casa)
  · índice de mercado normalizado (simulado y replay), features de cotización recalculadas a mano, día de la semana
  · decoder: la cabeza de precio respeta las opciones; log π(j, o) = log π(j) + log π(o | j); una sola opción no
    suma log-prob; greedy elige opciones válidas; la cabeza de precio arranca en la miope y no consume el RNG global
  · greedy_rollout: K se respeta (y manda si la política nunca vuelve), el margen es Σ lane_reward, env_reward sin
    shaping es margen / escala, mismo mundo ⇒ mismo resultado
  · transfer de columnas de entrada y checkpoints viejos
"""

import inspect

import numpy as np
import pytest
import torch

import model.encoder as encoder_mod
from config import (MAX_DURATION, LOADS_EXPECTED_CLIP, COMPETENCIA_M_MAX, COTIZACIONES_MAX_POR_NODO, K_LEVELS, K_MAX,
                    REWARD_SCALE_FACTOR, TIME_VIOLATION_PENALTY, DEVICE, AM_D_H, AM_N_HEADS, AM_N_LAYERS, AM_D_FF)
from conftest import agente_chico, escenario_sintetico, mundos_sim
from greedy_rollout import EntradasAgente, construir_features, contexto_agente, greedy_rollout
from market_process import EstadisticasLane
from model.critic_head import CriticHead
from model.decoder import PriceHead
from model.encoder import (COT_COLS, COT_P, COT_V, INDEX_COL, N_COT_FEATURES, N_NODE_FEATURES,
                           build_continuation_features_batch, build_dow_features_batch, ContextNetwork)
from pricing import N_NIVELES, U_NIVELES_ARR, valor_reintentos
from problem_data import lane_reward
from routing_env import OPCION_VACIO, VectorRoutingEnv, fin_espera

N = 10
CONT = slice(6, 14)


def np_(x):
    return x.detach().cpu().numpy()


def _estados(d, B=60, pasos=2, semilla=0, replay=False):
    """Entorno con B episodios avanzados `pasos` pasos con acciones válidas al azar (K = 5)."""
    env = d.env()
    mundos = mundos_sim(d.market, B, semilla=semilla, replay=replay)
    nodos = np.random.default_rng(semilla).integers(0, N, size=B)
    env.reset(mundos, nodos, [K_MAX] * B)
    gen = torch.Generator(device=DEVICE).manual_seed(semilla)
    from conftest import accion_al_azar
    activo = env.active.clone()
    for _ in range(pasos):
        a, o = accion_al_azar(env, gen)
        _, term, _ = env.step(a, o)
        activo &= ~term
    feats = construir_features(env, d.entradas(env))
    return env, np.flatnonzero(np_(activo)), feats


def _referencia(d, env, b):
    """Columnas 6–13 de la fila b con loops de Python (definición literal del docstring del encoder)."""
    T, est, p95 = d.T, d.est, d.p95
    i, h, t = int(env.current_node[b]), int(env.start_node[b]), float(env.time_elapsed[b])
    dow0 = int(env.start_dow[b])
    clip = LOADS_EXPECTED_CLIP
    out = np.zeros((N, 8))
    for j in range(N):
        espera = j == i
        termina = j == h and not espera
        tj = float(fin_espera(t)) if espera else t + T[i, j]
        Tjh = 0.0 if j == h else T[j, h]
        dow_j = (dow0 + int(tj // 14)) % 7
        out[j, 0] = Tjh / MAX_DURATION
        out[j, 1] = np.clip((MAX_DURATION - tj - Tjh) / MAX_DURATION, 0, 1)
        if j != h:
            out[j, 2] = np.clip(est.G[dow_j, j, h] / p95, -1, 1)
            out[j, 3] = min(est.L[dow_j, j, h], clip) / clip
        if termina:
            continue
        F = [l for l in range(N) if l not in (i, j, h) and tj + T[j, l] + T[l, h] <= MAX_DURATION]
        if F:
            out[j, 4] = np.clip(max(est.P1[dow_j, j, l] * est.G[dow_j, j, l] for l in F) / p95, -1, 1)
            out[j, 5] = np.mean([min(est.L[dow_j, j, l], clip) / clip for l in F])
        out[j, 6] = len(F) / N
        out[j, 7] = np.clip(np.log(max(est.M[dow_j, j], 1.0)) / np.log(COMPETENCIA_M_MAX), 0, 1)
    return out


# ── Features por nodo ────────────────────────────────────────────────────────

def test_n_features_y_el_encoder_no_importa_de_baselines():
    assert N_NODE_FEATURES == 6 + 8 + 1 + 3 + 2 * N_NIVELES + 2 and INDEX_COL == 14
    assert "baselines" not in inspect.getsource(encoder_mod)


def test_forma_rango_y_finitud(d10):
    env, _, f = _estados(d10)
    nf = f["node_feats"]
    assert nf.shape == (60, N, N_NODE_FEATURES) and nf.dtype == torch.float32 and nf.device.type == DEVICE.type
    assert torch.isfinite(nf).all()
    cont = np_(nf[:, :, CONT])
    assert cont[:, :, 0].min() >= 0 and cont[:, :, 1].min() >= 0 and cont[:, :, 1].max() <= 1
    assert cont[:, :, [2, 4]].min() >= -1 and cont[:, :, [2, 4]].max() <= 1
    assert cont[:, :, [3, 5, 6, 7]].min() >= 0 and cont[:, :, [3, 5, 6, 7]].max() <= 1
    for k, forma in (("temporal", (60, 3)), ("market", (60, 1)), ("budget", (60, 2)), ("dow", (60, 7)), ("cot", (60, 2))):
        assert f[k].shape == forma and torch.isfinite(f[k]).all()


@pytest.mark.parametrize("pasos", [0, 1, 2])
def test_columnas_de_continuacion_coinciden_con_referencia_con_loops(d10, pasos):
    env, activas, f = _estados(d10, pasos=pasos, semilla=pasos)
    assert len(activas) > 0
    for b in activas[:15]:
        np.testing.assert_allclose(np_(f["node_feats"][b, :, CONT]), _referencia(d10, env, b), atol=1e-6)


def test_conjunto_factible_excluye_lanes_desde_las_que_no_se_alcanza_a_volver():
    """Con N = 4: ir de j a l = 2 y volver a casa pasa del tiempo máximo, así que su valor (enorme) no cuenta."""
    T = np.full((4, 4), 10.0)
    np.fill_diagonal(T, 0.0)
    T[2, 0] = 70.0
    G = np.zeros((7, 4, 4))
    G[:, 1, 2], G[:, 1, 3] = 100.0, 20.0
    est = EstadisticasLane(L=np.full((7, 4, 4), 5.0), P1=np.ones((7, 4, 4)), G=G, M=np.full((7, 4), 2.0)).a(DEVICE)
    z = torch.zeros(1, dtype=torch.long, device=DEVICE)
    f = np_(build_continuation_features_batch(z, z, torch.zeros(1, device=DEVICE), z,
                                              torch.as_tensor(T, device=DEVICE), est, MAX_DURATION, 100.0)[0])
    assert f[1, 4] == pytest.approx(0.2) and f[1, 6] == pytest.approx(1 / 4)
    assert f[1, 5] == pytest.approx(5.0 / LOADS_EXPECTED_CLIP)
    assert f[1, 7] == pytest.approx(np.log(2.0) / np.log(COMPETENCIA_M_MAX))


def test_casos_especiales_espera_y_casa(d10):
    env, activas, f = _estados(d10, pasos=1, semilla=3)
    nf = np_(f["node_feats"])
    fuera = [b for b in activas if int(env.current_node[b]) != int(env.start_node[b])]
    assert fuera
    for b in fuera:
        i, h, t = int(env.current_node[b]), int(env.start_node[b]), float(env.time_elapsed[b])
        assert nf[b, i, 1] * MAX_DURATION == pytest.approx(float(fin_espera(t) - t), abs=1e-4)
        assert nf[b, h, 6] == 0 and (nf[b, h, 8:14] == 0).all()              # volver a casa termina el ciclo
        assert nf[b, h, 7] == pytest.approx(float(np.clip((MAX_DURATION - t - d10.T[i, h]) / MAX_DURATION, 0, 1)),
                                            abs=1e-6)
    env0, _, f0 = _estados(d10, pasos=0, semilla=9)
    casa = np_(f0["node_feats"])[np.arange(60), np_(env0.start_node), CONT]
    assert (casa[:, [0, 2, 3]] == 0).all() and (casa[:, 6] > 0).any()
    assert (casa[casa[:, 6] == 0][:, [4, 5]] == 0).all()


@pytest.mark.parametrize("replay", [False, True])
def test_el_indice_es_el_de_mercado_normalizado(d10, replay):
    env, _, f = _estados(d10, pasos=1, semilla=4, replay=replay)
    assert env.index_std == d10.market.indice_std
    esperado = d10.market.indice_historico(0, None)[np_(env.current_day)] / env.index_std
    np.testing.assert_allclose(np_(f["node_feats"][:, :, INDEX_COL]), esperado, atol=1e-6)


def test_features_de_cotizacion(d10):
    env, _, f = _estados(d10, pasos=0, semilla=1)
    cot = np_(f["node_feats"][..., COT_COLS])
    assert cot[..., :3].min() >= 0 and cot.min() >= -1 and cot.max() <= 1
    b, l = np.argwhere(cot[..., 2] > 0)[0]
    P = (1.0 - U_NIVELES_ARR) ** float(env.competencia[b])
    d_, c_, p_ = float(env.dist_rows[b, l]), float(env.cost_rows[b, l]), np_(env.precios_row[b, l]).astype(float)
    np.testing.assert_allclose(cot[b, l, COT_V], np.clip(P * (p_ * d_ - c_) / d10.p95, -1, 1), atol=1e-5)
    np.testing.assert_allclose(cot[b, l, COT_P], P, atol=1e-6)
    n = int(env.intentos[b, l])
    assert cot[b, l, 2] == pytest.approx(n / COTIZACIONES_MAX_POR_NODO)
    for col, base in ((-2, 0.0), (-1, -c_)):
        V = valor_reintentos(P, p_ * d_ - c_, base, COTIZACIONES_MAX_POR_NODO)[0][n] / d10.p95
        assert cot[b, l, col] == pytest.approx(float(np.clip(V, -1, 1)), abs=1e-5)
    assert (cot[..., 3:][cot[..., 2] == 0] == 0).all()
    assert (cot[np.arange(60), np_(env.current_node)] == 0).all()                       # self-loop
    cotizable = np_(env.opciones_validas[..., :N_NIVELES].any(-1)) & (np.arange(N)[None] != np_(env.current_node)[:, None])
    np.testing.assert_array_equal(cot[..., 2] > 0, cotizable)


def test_dia_de_la_semana(d10):
    oh = build_dow_features_batch(torch.tensor([0, 6, 3], device=DEVICE))
    assert oh.shape == (3, 7) and (oh.sum(dim=1) == 1).all() and oh[1, 6] == 1
    torch.manual_seed(0)
    ctx = ContextNetwork(32).to(DEVICE)
    g, c = torch.randn(4, 32, device=DEVICE), torch.randn(4, 32, device=DEVICE)
    t, m, b = torch.rand(4, 3, device=DEVICE), torch.rand(4, 1, device=DEVICE), torch.rand(4, 2, device=DEVICE)
    sin = ctx(g, c, t, m, b)
    assert torch.equal(sin, ctx(g, c, t, m, b, torch.zeros(4, 7, device=DEVICE)))     # dow None ⇒ ceros
    uno = torch.zeros(4, 7, device=DEVICE)
    uno[:, 5] = 1
    assert not torch.allclose(sin, ctx(g, c, t, m, b, uno))
    # el entorno avanza el día de la semana con los días de mercado
    env, _, _ = _estados(d10, pasos=2, semilla=6)
    dow = (np_(env.start_dow) + np_(env.time_elapsed).astype(np.float64) // 14).astype(int) % 7
    np.testing.assert_array_equal(np_(env.current_dow), dow)


# ── Decoder ──────────────────────────────────────────────────────────────────

def _decisiones(d):
    agent = agente_chico(N)
    env, _, f = _estados(d, B=32, pasos=0, semilla=2)
    with torch.no_grad():
        h, emb = contexto_agente(agent, f)
    mask = env.opciones_validas.any(-1)
    return agent, h, emb, mask, f["opciones"], f["node_feats"][..., COT_COLS]


def test_act_respeta_opciones_y_logprob_es_la_suma(d10):
    agent, h, emb, mask, op, cot = _decisiones(d10)
    B = h.shape[0]
    filas = torch.arange(B, device=DEVICE)
    with torch.no_grad():
        j, o, lp, h_j, h_o = agent.decoder.act_cotizacion(h, emb, mask, op, cot)
        assert op[filas, j, o].all()
        lp2, _, _ = agent.decoder.evaluate_cotizacion(h, emb, mask, op, cot, j, o)
        torch.testing.assert_close(lp, lp2)
        logits, glimpse = agent.decoder._logits_y_glimpse(h, emb, ~mask)
        lp_j = torch.log_softmax(logits, -1)[filas, j]
        lp_o = agent.decoder._dist_precio(h, glimpse, emb, j, op, cot).log_prob(o)
        torch.testing.assert_close(lp, lp_j + lp_o, atol=1e-5, rtol=1e-5)
        assert (h_j >= 0).all() and (h_o >= 0).all()
        jg, og = agent.decoder.greedy_cotizacion(h, emb, mask, op, cot)
        assert op[filas, jg, og].all()
        # con una sola opción válida, log π(o | j) = 0
        solo = torch.zeros_like(op)
        jj = mask.float().argmax(dim=-1)
        solo[filas, jj, OPCION_VACIO] = True
        lp_o = agent.decoder._dist_precio(h, glimpse, emb, jj, solo, cot).log_prob(
            torch.full((B,), OPCION_VACIO, device=DEVICE))
        torch.testing.assert_close(lp_o, torch.zeros(B, device=DEVICE), atol=1e-6, rtol=0)


def test_act_mixto_muestrea_las_primeras_y_argmax_el_resto(d10):
    """act_mixto con n_muestra = B es act_cotizacion (misma semilla); con 0 es greedy_cotizacion; con n intermedio,
    las primeras n filas samplean (log π y entropías solo de ellas) y el resto es el argmax."""
    agent, h, emb, mask, op, cot = _decisiones(d10)
    B, n = h.shape[0], 12
    dec = agent.decoder
    with torch.no_grad():
        torch.manual_seed(5)
        ref = dec.act_cotizacion(h, emb, mask, op, cot)
        torch.manual_seed(5)
        mix = dec.act_mixto(h, emb, mask, op, cot, B)
        for x, y in zip(ref, mix):
            assert torch.equal(x, y)
        jg, og = dec.greedy_cotizacion(h, emb, mask, op, cot)
        j0, o0, lp0, hj0, ho0 = dec.act_mixto(h, emb, mask, op, cot, 0)
        assert torch.equal(j0, jg) and torch.equal(o0, og) and lp0.numel() == hj0.numel() == ho0.numel() == 0
        torch.manual_seed(9)
        ref_n = dec.act_cotizacion(h[:n], emb[:n], mask[:n], op[:n], cot[:n])
        torch.manual_seed(9)
        j, o, lp, hj, ho = dec.act_mixto(h, emb, mask, op, cot, n)
        assert torch.equal(ref_n[0], j[:n]) and torch.equal(ref_n[1], o[:n])
        for x, y in zip(ref_n[2:], (lp, hj, ho)):                 # otro tamaño de batch: igual hasta redondeo
            torch.testing.assert_close(x, y, atol=1e-5, rtol=1e-5)
        assert torch.equal(j[n:], jg[n:]) and torch.equal(o[n:], og[n:])


def test_cabeza_de_precio_no_consume_rng_y_arranca_en_la_miope():
    torch.manual_seed(123)
    antes = torch.get_rng_state().clone()
    cab = PriceHead(8, N_NIVELES + 1)
    assert torch.equal(antes, torch.get_rng_state())
    with torch.no_grad():
        cab.net[-1].weight.zero_(), cab.net[-1].bias.zero_()
    P = (1.0 - U_NIVELES_ARR) ** 2.0
    v = P * (0.4 + 0.5 * U_NIVELES_ARR - 0.3)
    cot = torch.zeros(1, N_COT_FEATURES)
    cot[0, COT_V], cot[0, COT_P] = torch.tensor(v, dtype=torch.float32), torch.tensor(P, dtype=torch.float32)
    z = torch.zeros(1, 8)
    with torch.no_grad():
        k0 = int(cab(z, z, z, cot)[0, :N_NIVELES].argmax())
        cab.delta[-1].bias.fill_(1.0)                       # Δ > 0: ganar la carga vale más ⇒ cotiza más bajo
        k1 = int(cab(z, z, z, cot)[0, :N_NIVELES].argmax())
    assert k0 == int(np.argmax(v)) and k1 == int(np.argmax(v + P)) and k1 < k0


# ── Rollout greedy (escenario sintético: 1 h por arco, K manda) ──────────────

NS, MAX_SYN = 9, 30.0


@pytest.fixture(scope="module")
def sint():
    s = escenario_sintetico(NS, precio=5.0, dist=10.0, m=2.0)
    r = np.random.RandomState(0)
    s["est"] = EstadisticasLane(L=r.uniform(0, 30, (7, NS, NS)), P1=r.uniform(0, 1, (7, NS, NS)),
                                G=r.uniform(-50, 200, (7, NS, NS)), M=r.uniform(1, 5, (7, NS)))
    s["agent"] = agente_chico(NS)
    return s


def _rollout(s, k, registrar=True):
    env = VectorRoutingEnv(s["T"], s["dist"], s["diesel"], s["mercado"], device=DEVICE, max_duration=MAX_SYN)
    entradas = EntradasAgente.crear(env, None, s["est"], 1.0, max_duration=MAX_SYN)
    mundos = s["mercado"].mundos(s["loads"], NS, semilla=100, u_min=0.15)          # p10 gana siempre aquí
    return greedy_rollout(s["agent"], env, mundos, np.arange(NS), entradas, k_max=[k] * NS, registrar=registrar)


def _nunca_vuelve_antes(agent):
    """Cotiza p10 en una lane que no vaya a casa mientras haya una; si no, vuelve: obliga a que K cierre el ciclo."""
    starts = torch.arange(NS, device=DEVICE)

    def pick(h_t, embeddings, mask_t, opciones_t, cot_t):
        rows = torch.arange(mask_t.shape[0], device=DEVICE)
        cotizable = opciones_t[:, :, :-1].any(dim=-1) & mask_t.bool()
        cotizable[rows, starts] = False
        casa = torch.zeros_like(cotizable)
        casa[rows, starts] = True
        j = torch.where(cotizable.any(dim=1, keepdim=True), cotizable, casa & mask_t.bool()).float().argmax(dim=1)
        return j, opciones_t[rows, j].float().argmax(dim=1)
    return pick


def test_greedy_respeta_k_y_el_libre_cierra_o_falla(sint):
    for k in K_LEVELS:
        res = _rollout(sint, k)
        for r, ok in zip(res.rutas(), np_(res.valid)):
            if ok:
                assert len(r) - 1 <= k and r[0] == r[-1]
    res = _rollout(sint, None)
    assert bool(res.valid.any())
    for s, r, ok, e, raw in zip(range(NS), res.rutas(), np_(res.valid), np_(res.env_reward), np_(res.raw_reward)):
        if ok:
            assert r[0] == r[-1] == s
        else:                                                   # margen recorrido + la penalización
            assert e == pytest.approx(raw / REWARD_SCALE_FACTOR + TIME_VIOLATION_PENALTY, abs=1e-4)
    assert (res.n_ganadas <= res.n_cotizaciones).all()


def test_k_manda_si_la_politica_nunca_vuelve_y_el_margen_es_la_suma(sint, monkeypatch):
    monkeypatch.setattr(sint["agent"].decoder, "greedy_cotizacion", _nunca_vuelve_antes(sint["agent"]))
    d, c, p = sint["dist"], sint["diesel"], 5.0 * (1.0 + U_NIVELES_ARR[0])
    for k in K_LEVELS:
        res = _rollout(sint, k)
        assert bool(res.valid.all()) and [len(r) - 1 for r in res.rutas()] == [k] * NS
        assert torch.equal(res.n_ganadas, res.n_cotizaciones)
        for r, raw in zip(res.rutas(), np_(res.raw_reward)):
            assert raw == pytest.approx(sum(float(lane_reward(p, 1, d[a, b], c[a, b])) for a, b in zip(r[:-1], r[1:])))


def test_env_reward_sin_shaping_es_margen_sobre_escala(sint, monkeypatch):
    import routing_env
    monkeypatch.setattr(routing_env, "USE_REWARD_SHAPING", False)
    for k in (*K_LEVELS, None):
        res = _rollout(sint, k, registrar=False)
        v = res.valid
        assert bool(v.any())
        torch.testing.assert_close(res.env_reward[v].double(), res.raw_reward[v] / REWARD_SCALE_FACTOR,
                                   atol=1e-5, rtol=0)


def test_greedy_real_determinista_y_no_cambia_el_modo_del_agente(d10):
    agent = agente_chico(N).train()
    mundos = mundos_sim(d10.market, 32, semilla=8)
    out = []
    for _ in range(2):
        env = d10.env()
        out.append(greedy_rollout(agent, env, mundos, np.arange(32) % N, d10.entradas(env), k_max=[K_MAX] * 32))
    assert agent.training
    assert torch.equal(out[0].raw_reward, out[1].raw_reward) and torch.equal(out[0].valid, out[1].valid)


# ── Transfer y checkpoints ───────────────────────────────────────────────────

def test_transfer_de_columnas_de_entrada():
    from transfer_learning import _reconcile_state_dict, _tiene_entradas_actuales
    torch.manual_seed(0)
    clave = "encoder.input_proj.proj.weight"
    viejo = torch.randn(32, 7)
    sd, _, omitidos = _reconcile_state_dict({clave: torch.randn(32, 11), "encoder.input_proj.proj.bias": torch.randn(32)},
                                            {clave: viejo, "encoder.input_proj.proj.bias": torch.ones(32)})
    w = sd[clave]
    assert torch.equal(w[:, :7], viejo) and (w[:, 7:] == 0).all() and not omitidos
    x = torch.randn(4, 11)
    assert torch.allclose(x @ w.T, x[:, :7] @ viejo.T)                 # las columnas nuevas no cambian la salida
    with pytest.raises(ValueError, match=r"encoder\.input_proj\.proj\.weight.*\(32, 12\).*\(32, 11\)"):
        _reconcile_state_dict({clave: torch.zeros(32, 11)}, {clave: torch.randn(32, 12)})
    with pytest.raises(ValueError):
        _reconcile_state_dict({"b": torch.zeros(32)}, {"b": torch.zeros(16)})
    sd, _, omitidos = _reconcile_state_dict({"a": torch.zeros(2)}, {"a": torch.ones(2), "extra": torch.ones(3)})
    assert omitidos == ["extra"] and torch.equal(sd["a"], torch.ones(2))
    assert _tiene_entradas_actuales({"agent": {clave: torch.zeros(32, N_NODE_FEATURES)}})
    assert not _tiene_entradas_actuales({"agent": {clave: torch.zeros(32, 34)}})


def test_checkpoints_viejos(tmp_path):
    from model.agent import AMRoutingAgent
    from transfer_learning import load_checkpoint_compatible
    torch.manual_seed(0)
    agent, critic = AMRoutingAgent(N, AM_D_H, AM_N_HEADS, AM_N_LAYERS, AM_D_FF, device=DEVICE), CriticHead(AM_D_H)
    assert not any(k.startswith("post_value") for k in agent.state_dict())
    sd = dict(agent.state_dict())
    sd["post_value_a.net.0.weight"] = torch.zeros(AM_D_H, 2 * AM_D_H)          # cabezas Ṽ de los *_pricing.pt viejos
    torch.save({"agent": sd, "critic": critic.state_dict()}, tmp_path / "viejo.pt")
    cargado, _ = load_checkpoint_compatible(str(tmp_path / "viejo.pt"), N, AM_D_H, AM_N_HEADS, AM_N_LAYERS, AM_D_FF,
                                            verbose=False)
    for k, v in agent.state_dict().items():
        assert torch.equal(cargado.state_dict()[k].to(v.device), v), k
    sin_precio = {k: v for k, v in agent.state_dict().items() if not k.startswith("decoder.precio.")}
    torch.save({"agent": sin_precio, "critic": critic.state_dict()}, tmp_path / "sin_precio.pt")
    with pytest.raises(ValueError, match="anterior a la cabeza de precio"):
        load_checkpoint_compatible(str(tmp_path / "sin_precio.pt"), N, AM_D_H, AM_N_HEADS, AM_N_LAYERS, AM_D_FF,
                                   verbose=False)
