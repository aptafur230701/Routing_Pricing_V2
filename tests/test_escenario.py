"""
test_escenario.py
=================
Escenario (validate_pricing.py): reproducción del prefijo, selector de DRL-Q, regla por umbral y la prueba
contrafactual, con un agente aleatorio en CPU sobre los datos reales de N=10 (K libre).
"""

import numpy as np
import pandas as pd
import pytest
import torch

import validate_pricing as vp
from model.agent import AMRoutingAgent
from config import (
    AM_D_H, AM_N_HEADS, AM_N_LAYERS, AM_D_FF, TRAIN_DAYS, MARKET_SEED_VAL, REWARD_SCALE_FACTOR as S, configurar_k,
)
from evaluation import disenar_mundos
from greedy_rollout import greedy_rollout
from pricing import crear_selector_q, q_values

NUM_NODES = 10


@pytest.fixture(scope="module", autouse=True)
def k_libre():
    configurar_k(False)


@pytest.fixture(scope="module")
def datos():
    return vp.cargar_datos(NUM_NODES)


@pytest.fixture(scope="module")
def esc(datos):
    torch.manual_seed(0)
    agent = AMRoutingAgent(NUM_NODES, AM_D_H, AM_N_HEADS, AM_N_LAYERS, AM_D_FF, device=torch.device("cpu"),
                           use_post_head=True).eval()
    agent.post_entrenadas = True
    d = datos
    return vp.Escenario(agent, NUM_NODES, d["time_matrix"], d["rate_stack"][:TRAIN_DAYS], d["loads_stack"][:TRAIN_DAYS],
                        d["distance_arr"], d["diesel_arr"], d["ltr_stack"], d["avail_expected_arr"],
                        d["reward_global_p95"], d["market"], d["ev_esperado"])


@pytest.fixture(scope="module")
def mundos(datos):
    M = 40
    return disenar_mundos(NUM_NODES, M, datos["market"], num_dias=TRAIN_DAYS, semilla=MARKET_SEED_VAL), [None] * M


@pytest.mark.parametrize("modo", ["greedy", "q"])
def test_reproducir_con_la_accion_original_reproduce_la_trayectoria(esc, mundos, modo):
    """Prefijo + la acción tomada en el paso k ⇒ misma lane en k y mismo margen posterior (el actor greedy
    es determinista; para DRL-Q la continuación es la del actor, así que solo se compara el primer paso)."""
    dis, k = mundos
    recs = esc.recolectar(dis, k, modo=modo, con_valores=esc.con_cabezas)
    acc = {}
    for r in recs:
        acc.setdefault(r["b"], []).append(r["j"])
    out = esc.reproducir([r["start"] for r in recs], [r["start_day"] for r in recs], [r["world_seed"] for r in recs],
                         [r["k"] for r in recs], [acc[r["b"]][:r["step"]] for r in recs], [r["j"] for r in recs])
    np.testing.assert_allclose(out["r_forzada"], [r["r_raw"] for r in recs], atol=1e-3)
    if modo == "greedy":
        np.testing.assert_allclose(out["cont"], [r["futuro_usd"] for r in recs], atol=1e-3)
    assert out["valida"].all()


def test_rama_distinta_cambia_solo_desde_el_estado(esc, mundos):
    """Forzar otra acción en el paso k no altera lo anterior: la lane cobrada en k es la de esa acción."""
    dis, k = mundos
    otras = lambda r: np.where((r["mask"] == 1) & (np.arange(NUM_NODES) != r["start"]) & (np.arange(NUM_NODES) != r["j"]))[0]
    recs = [r for r in esc.recolectar(dis, k, modo="greedy", con_valores=False) if r["step"] == 0 and len(otras(r))][:15]
    alt = [int(otras(r)[0]) for r in recs]
    out = esc.reproducir([r["start"] for r in recs], [r["start_day"] for r in recs], [r["world_seed"] for r in recs],
                         [r["k"] for r in recs], [[] for _ in recs], alt, capturar=True)
    for r, a, rf in zip(recs, alt, out["r_forzada"]):
        i = r["i"]
        esperado = np.round((r["rate"][a] * esc.dist[i, a] if r["loads"][a] > 0 else 0.0) - esc.cost[i, a], 0)
        assert rf == pytest.approx(esperado, abs=1.01)
    assert out["h"].shape == (len(recs), AM_D_H) and torch.isfinite(out["h"]).all() and torch.isfinite(out["e"]).all()


def test_selector_q_es_el_argmax_de_q_hecho_a_mano(esc, mundos):
    dis, k = mundos
    env = esc.make_env()
    masks = env.reset(dis["nodo"], dis["start_day"], k_max_arr=k, world_seeds=dis["world_seed"])
    h_t, emb, _ = esc.encode(env, masks)
    sel = crear_selector_q(esc.agent, esc.dist, esc.diesel, S)(h_t, emb, masks, env)
    va, vb = esc.valores(h_t, emb)
    for b in range(len(sel)):
        i = int(env.current_node[b])
        q = q_values(env.rate_row_obs[b].astype(float), env.loads_row_obs[b], esc.dist[i], esc.cost[i], S,
                     0.5 * (va[b] + vb[b]), int(env.start_node[b]))
        q[masks[b] == 0] = -np.inf
        assert sel[b] == int(np.argmax(q))
        assert masks[b][sel[b]] == 1


def test_recolectar_q_coincide_con_greedy_rollout_con_selector(esc, mundos, datos):
    """El rollout vectorizado de DRL-Q (recolectar) y el de greedy_rollout con el selector dan el mismo margen."""
    dis, k = mundos
    recs = esc.recolectar(dis, k, modo="q")
    margen = np.zeros(len(k))
    for r in recs:
        margen[r["b"]] += r["r_raw"]
    d = datos
    res = greedy_rollout(esc.agent, esc.make_env(), dis["nodo"], dis["start_day"], np.asarray(d["time_matrix"], float),
                         d["distance_arr"], d["ltr_stack"], d["reward_global_p95"], NUM_NODES,
                         k_max_arr=k, avail_expected_arr=d["avail_expected_arr"], world_seeds=dis["world_seed"],
                         ev_esperado=d["ev_esperado"], selector=esc.selector_q, device=torch.device("cpu"))
    np.testing.assert_allclose(margen, res.raw_reward, atol=1e-3)


def test_umbral_devuelve_un_episodio_valido_por_mundo(esc, mundos):
    dis, k = mundos
    for tipo in ("be", "be+margen", "pstar"):
        df = esc.umbral(dis, k, tipo)
        assert len(df) == len(k) and df["valido"].mean() > 0.9 and np.isfinite(df["margen"]).all()


def test_contrafactual_pequeno(esc):
    cf, res = vp.validar_contrafactual(esc, 60, 25, np.random.default_rng(0))
    assert 0 < len(cf) <= 25
    assert {"p_pred", "p_real", "sesgo", "sesgo_usd_lane", "tramo", "ancho_rango"} <= set(cf.columns)
    assert np.isfinite(cf[["p_pred", "p_real"]].to_numpy()).all()
    np.testing.assert_allclose(cf["sesgo_usd_lane"], cf["sesgo"] * cf["dist_j"])
    assert list(res["grupo"]) == ["global", "< 300 mi", "300-800 mi", "> 800 mi"]
    assert res.loc[0, "n_estados"] == len(cf)
    assert res.loc[0, "sesgo_ic95_inf"] <= res.loc[0, "sesgo_usd_milla"] <= res.loc[0, "sesgo_ic95_sup"]
    assert (cf["p_min"] <= cf["p_pred"] + 1e-9).all() and (cf["p_pred"] <= cf["p_max"] + 1e-9).all()


def test_p_real_algebra(esc):
    """A la tarifa p*_realizado la lane j rinde lo mismo que la alternativa: (p·d − costo_j) + C_j = G_alt."""
    cf, _ = vp.validar_contrafactual(esc, 60, 15, np.random.default_rng(1))
    costo = cf["break_even"] * cf["dist_j"]
    np.testing.assert_allclose(cf["p_real_ab"] * cf["dist_j"] - costo + cf["c_j"], cf["g_alt_ab"], atol=1e-6)
    np.testing.assert_allclose(cf["p_real_ba"] * cf["dist_j"] - costo + cf["c_j"], cf["g_alt_ba"], atol=1e-6)
    np.testing.assert_allclose(cf["p_real"], (cf["p_real_ab"] + cf["p_real_ba"]) / 2)
    mismas = cf["alt_a"] == cf["alt_b"]
    np.testing.assert_allclose(cf.loc[mismas, "p_real_ab"], cf.loc[mismas, "p_real_ba"])


def test_generar_ejemplos_y_ajuste_pequeno(esc):
    import post_training as pt
    ej = pt.generar_ejemplos(esc, 300, 12345, mundos_por_bloque=120, verbose=False)
    assert ej["h"].shape == (300, AM_D_H) and ej["e"].shape == (300, AM_D_H) and ej["y"].shape == (300,)
    assert torch.isfinite(ej["y"]).all() and len(ej["mundo"]) == 300
    val = pt.generar_ejemplos(esc, 100, 999, mundos_por_bloque=60, verbose=False)
    antes = esc.agent.post_value_a.state_dict()["net.0.weight"].clone()
    curva = pt.ajustar_cabezas(esc.agent, ej, val, 0, fracs=(0.5, 1.0), verbose=False)
    assert len(curva) == 2 and np.isfinite(curva["rmse_promedio_usd"]).all()
    assert not torch.equal(antes, esc.agent.post_value_a.state_dict()["net.0.weight"])
    # A y B se entrenaron con mundos distintos
    assert curva["n_A"].iloc[-1] > 0 and curva["n_B"].iloc[-1] > 0


def test_huella_del_actor_detecta_otro_actor_y_reutilizacion(esc, tmp_path):
    import post_training as pt
    antes = pt.huella_actor(esc.agent.state_dict())
    ruta = tmp_path / "p.pt"
    torch.save({"agent": esc.agent.state_dict(), "critic": {}, "post": {"actor_sha": antes}}, ruta)
    assert pt.pricing_vigente(str(ruta), esc.agent)                       # mismo actor
    otro = AMRoutingAgent(NUM_NODES, AM_D_H, AM_N_HEADS, AM_N_LAYERS, AM_D_FF, device=torch.device("cpu"), use_post_head=True)
    assert pt.huella_actor(otro.state_dict()) != antes
    assert not pt.pricing_vigente(str(ruta), otro)                        # actor reentrenado/distinto ⇒ no vigente
    # cambiar solo las cabezas no cambia la huella
    c = AMRoutingAgent(NUM_NODES, AM_D_H, AM_N_HEADS, AM_N_LAYERS, AM_D_FF, device=torch.device("cpu"), use_post_head=True)
    c.load_state_dict({k: v for k, v in esc.agent.state_dict().items()})
    with torch.no_grad():
        c.post_value_a.net[0].weight.add_(1.0)
    assert pt.huella_actor(c.state_dict()) == antes
