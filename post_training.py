"""
post_training.py
================
Entrenamiento de las cabezas de valor post-decisión Ṽ A y B (PRICING_PLAN.md), DESPUÉS del PPO y con el encoder,
el contexto y el actor congelados. Ṽ(s, a) es el margen esperado desde la llegada a `a` hasta volver a casa
siguiendo el actor en modo greedy (determinista).

Datos
-----
  · Estados: rutas del actor (muestreado) en mundos nuevos de entrenamiento (MARKET_SEED_POST_TRAIN, distinta de
    la del PPO, la de validación y la de evaluación), K libre.
  · En cada estado se elige una acción al azar entre las válidas excepto volver a casa y, desde la llegada, se
    sigue con el actor greedy hasta casa en el mismo mundo (Escenario.reproducir).
  · Objetivo: el margen real posterior a llegar a la acción (suma de last_raw_reward de los pasos siguientes) / S.
  · Cada MUNDO va al azar a A o a B (mitades distintas de los datos); se reparte por mundo y no por ejemplo
    porque los ejemplos de un mismo mundo comparten el ruido de la realización.
  · Validación: ejemplos de mundos retenidos (MARKET_SEED_POST_VAL).

Se guardan en un checkpoint nuevo (am_checkpoint_<N>nodes_pricing.pt); el del PPO queda intacto.

Uso
---
    python post_training.py --num-nodes 10 [--n-ejemplos 200000] [--checkpoint RUTA]
"""

import argparse
import copy
import os
import time

import numpy as np
import pandas as pd
import torch

from config import (
    DEVICE, SEED, TRAIN_DAYS, REWARD_SCALE_FACTOR as S, RUN_TAG, configurar_k, AM_D_H, AM_N_HEADS, AM_N_LAYERS,
    AM_D_FF, POST_N_EJEMPLOS, POST_N_VAL, POST_LR, POST_BATCH, POST_EPOCHS, POST_PACIENCIA,
    MARKET_SEED_POST_TRAIN, MARKET_SEED_POST_VAL, POST_HIDDEN, POST_SEED_A, POST_SEED_B,
)
from critic_head import PostDecisionValueHead
from evaluation import disenar_mundos
from validate_pricing import Escenario, cargar_datos


# ─────────────────────────────────────────────────────────────────────────────
# Vigencia del checkpoint con las cabezas
# ─────────────────────────────────────────────────────────────────────────────

def huella_actor(state_dict: dict) -> str:
    """SHA-1 de los pesos del actor del PPO (state_dict del agente, sin las cabezas Ṽ): identifica contra qué actor congelado se
    entrenaron las cabezas. Un checkpoint _pricing.pt solo sirve si coincide con el actor que se va a usar."""
    import hashlib
    h = hashlib.sha1()
    for k in sorted(k for k in state_dict if not k.startswith("post_value")):
        h.update(k.encode())
        h.update(state_dict[k].detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def pricing_vigente(pricing_path: str, agent) -> bool:
    """True si `pricing_path` trae las cabezas Ṽ entrenadas contra EXACTAMENTE los pesos del actor `agent`."""
    from transfer_learning import checkpoint_tiene_post_head
    if not checkpoint_tiene_post_head(pricing_path):
        return False
    meta = torch.load(pricing_path, map_location="cpu").get("post", {})
    return meta.get("actor_sha") == huella_actor(agent.state_dict())


# ─────────────────────────────────────────────────────────────────────────────
# Generación de ejemplos
# ─────────────────────────────────────────────────────────────────────────────

def generar_ejemplos(esc: Escenario, n_ejemplos: int, semilla: int, mundos_por_bloque: int = 4000,
                     verbose: bool = True) -> dict:
    """n_ejemplos (h_t, e_a, y) de estados de rutas del actor muestreado; K libre.

    Retorna dict con h (n, d_h), e (n, d_h) float32 en CPU, y (n,) = margen posterior a llegar a la acción / S,
    mundo (n,) identificador global del mundo, valida (n,) la continuación cerró en casa a tiempo.
    """
    rng = np.random.default_rng(semilla)
    h, e, y, mundo, valida = [], [], [], [], []
    n, bloque, offset = 0, 0, 0
    t0 = time.time()
    while n < n_ejemplos:
        dis = disenar_mundos(esc.N, mundos_por_bloque, esc.market, num_dias=TRAIN_DAYS, semilla=semilla + 1 + bloque)
        k_arr = [None] * mundos_por_bloque
        recs = esc.recolectar(dis, k_arr, modo="sample", con_valores=False)
        acciones = {}
        for r in recs:
            acciones.setdefault(r["b"], []).append(r["j"])
        filas, elegida = [], []
        for r in recs:
            cand = np.where((r["mask"] == 1) & (np.arange(esc.N) != r["start"]))[0]     # válidas excepto volver a casa
            if len(cand):
                filas.append(r)
                elegida.append(int(rng.choice(cand)))
        out = esc.reproducir(
            [r["start"] for r in filas], [r["start_day"] for r in filas], [r["world_seed"] for r in filas],
            [r["k"] for r in filas], [acciones[r["b"]][:r["step"]] for r in filas], elegida, capturar=True,
        )
        h.append(out["h"]); e.append(out["e"])
        y.append(torch.as_tensor(out["cont"] / S, dtype=torch.float32))
        mundo.append(np.array([offset + r["b"] for r in filas]))
        valida.append(out["valida"])
        n += len(filas); offset += mundos_por_bloque; bloque += 1
        if verbose:
            print(f"    ejemplos: {min(n, n_ejemplos)}/{n_ejemplos}  ({time.time() - t0:.0f} s)", flush=True)
    cut = slice(0, n_ejemplos)
    return dict(h=torch.cat(h)[cut], e=torch.cat(e)[cut], y=torch.cat(y)[cut],
                mundo=np.concatenate(mundo)[cut], valida=np.concatenate(valida)[cut])


# ─────────────────────────────────────────────────────────────────────────────
# Ajuste de las cabezas
# ─────────────────────────────────────────────────────────────────────────────

def _nueva_cabeza(semilla: int, device) -> PostDecisionValueHead:
    return PostDecisionValueHead(AM_D_H, POST_HIDDEN, seed=semilla).to(device)


@torch.no_grad()
def _predecir(cabeza, H, E, lote: int = 16384) -> torch.Tensor:
    return torch.cat([cabeza.forward_pairs(H[i:i + lote], E[i:i + lote]) for i in range(0, len(H), lote)])


def _entrenar_cabeza(cabeza, H, E, y, Hv, Ev, yv, gen, verbose=False, nombre="") -> dict:
    """Adam con parada temprana sobre el MSE de validación; deja en `cabeza` los mejores pesos."""
    opt = torch.optim.Adam(cabeza.parameters(), lr=POST_LR)
    mejor, mejor_sd, sin_mejora, hist = float("inf"), None, 0, []
    for ep in range(POST_EPOCHS):
        cabeza.train()
        perm = torch.randperm(len(H), generator=gen, device="cpu").to(H.device)
        for i in range(0, len(H), POST_BATCH):
            ix = perm[i:i + POST_BATCH]
            loss = torch.nn.functional.mse_loss(cabeza.forward_pairs(H[ix], E[ix]), y[ix])
            opt.zero_grad(); loss.backward(); opt.step()
        cabeza.eval()
        mse_v = float(torch.nn.functional.mse_loss(_predecir(cabeza, Hv, Ev), yv))
        hist.append(mse_v)
        if mse_v < mejor - 1e-6:
            mejor, mejor_sd, sin_mejora = mse_v, copy.deepcopy(cabeza.state_dict()), 0
        else:
            sin_mejora += 1
            if sin_mejora >= POST_PACIENCIA:
                break
    cabeza.load_state_dict(mejor_sd); cabeza.eval()
    if verbose:
        print(f"      cabeza {nombre}: {len(hist)} épocas, MSE val {mejor:.4f} (RMSE {np.sqrt(mejor) * S:.1f} USD)")
    return dict(epocas=len(hist), mse_val=mejor)


def metricas_val(va: torch.Tensor, vb: torch.Tensor, yv: torch.Tensor) -> dict:
    """Error de Ṽ en validación (USD): cada cabeza y el promedio de las dos (el que usa DRL-Q)."""
    y = yv.cpu().numpy().astype(np.float64) * S
    out = dict(n_val=len(y), sd_objetivo_usd=float(y.std()), media_objetivo_usd=float(y.mean()))
    for nombre, p in (("A", va), ("B", vb), ("promedio", 0.5 * (va + vb))):
        p = p.cpu().numpy().astype(np.float64) * S
        out[f"rmse_{nombre}_usd"] = float(np.sqrt(np.mean((p - y) ** 2)))
        out[f"sesgo_{nombre}_usd"] = float(np.mean(p - y))
        out[f"mae_{nombre}_usd"] = float(np.mean(np.abs(p - y)))
        out[f"r2_{nombre}"] = float(1 - np.mean((p - y) ** 2) / np.var(y))
    out["rmse_dif_AB_usd"] = float(np.sqrt(np.mean(((va - vb).cpu().numpy() * S) ** 2)))
    return out


def ajustar_cabezas(agent, train: dict, val: dict, semilla: int, fracs=(0.25, 0.5, 1.0), verbose=True) -> pd.DataFrame:
    """Entrena A y B (mundos repartidos al azar entre las dos) y devuelve la curva de error de validación por n.

    Para cada fracción de los datos se entrena de cero; la última (1.0) queda en el agente. El error de
    validación que sigue bajando entre fracciones indica que conviene generar más ejemplos.
    """
    dev = next(agent.parameters()).device
    rng = np.random.default_rng(semilla)
    mundos = np.unique(train["mundo"])
    a_A = set(rng.permutation(mundos)[: len(mundos) // 2].tolist())              # mitad de los MUNDOS a A, el resto a B
    es_A = torch.as_tensor(np.array([m in a_A for m in train["mundo"]]))
    H, E, y = train["h"].to(dev), train["e"].to(dev), train["y"].to(dev)
    Hv, Ev, yv = val["h"].to(dev), val["e"].to(dev), val["y"].to(dev)
    gen = torch.Generator().manual_seed(semilla)
    filas = []
    for frac in fracs:
        cab = {"A": _nueva_cabeza(POST_SEED_A, dev), "B": _nueva_cabeza(POST_SEED_B, dev)}
        n_cab = {}
        for nombre, sel in (("A", es_A), ("B", ~es_A)):
            ix = torch.where(sel)[0]
            ix = ix[: max(1, int(len(ix) * frac))].to(dev)
            n_cab[nombre] = len(ix)
            _entrenar_cabeza(cab[nombre], H[ix], E[ix], y[ix], Hv, Ev, yv, gen, verbose=verbose, nombre=nombre)
        m = metricas_val(_predecir(cab["A"], Hv, Ev), _predecir(cab["B"], Hv, Ev), yv)
        filas.append(dict(frac=frac, n_A=n_cab["A"], n_B=n_cab["B"], n_total=n_cab["A"] + n_cab["B"], **m))
        if verbose:
            print(f"    n={filas[-1]['n_total']:>7}: RMSE val (USD) A {m['rmse_A_usd']:.1f} | B {m['rmse_B_usd']:.1f} | "
                  f"promedio {m['rmse_promedio_usd']:.1f}  (σ objetivo {m['sd_objetivo_usd']:.1f}, R² {m['r2_promedio']:.3f}, "
                  f"sesgo {m['sesgo_promedio_usd']:+.2f})", flush=True)
    agent.post_value_a.load_state_dict(cab["A"].state_dict())
    agent.post_value_b.load_state_dict(cab["B"].state_dict())
    agent.post_entrenadas = True
    return pd.DataFrame(filas)


# ─────────────────────────────────────────────────────────────────────────────
# Orquestación
# ─────────────────────────────────────────────────────────────────────────────

def entrenar_cabezas_post(agent, critic, datos: dict, num_nodes: int, destino: str, n_ejemplos: int = POST_N_EJEMPLOS,
                          n_val: int = POST_N_VAL, output_dir: str = None, verbose: bool = True) -> pd.DataFrame:
    """Fase posterior al PPO: genera ejemplos, ajusta A/B, guarda el checkpoint `destino` y la curva de error."""
    torch.manual_seed(SEED)                                      # actor muestreado y mezclas reproducibles
    agent.eval()
    esc = Escenario(agent, num_nodes, datos["time_matrix"], datos["rate_stack"][:TRAIN_DAYS],
                    datos["loads_stack"][:TRAIN_DAYS], datos["distance_arr"], datos["diesel_arr"], datos["ltr_stack"],
                    datos["trucks_stack"], datos["avail_expected_arr"], datos["reward_global_p95"], datos["market"],
                    datos["ev_esperado"])
    t0 = time.time()
    if verbose:
        print(f"\n--- Ṽ posterior al PPO: {n_ejemplos} ejemplos de entrenamiento y {n_val} de validación ---")
    train = generar_ejemplos(esc, n_ejemplos, MARKET_SEED_POST_TRAIN, verbose=verbose)
    val = generar_ejemplos(esc, n_val, MARKET_SEED_POST_VAL, verbose=verbose)
    if verbose:
        print(f"  continuaciones que no cierran en casa a tiempo: train {100 * (~train['valida']).mean():.2f} %, "
              f"val {100 * (~val['valida']).mean():.2f} %  ({time.time() - t0:.0f} s)")
    curva = ajustar_cabezas(agent, train, val, SEED, verbose=verbose)
    meta = dict(actor_sha=huella_actor(agent.state_dict()), n_ejemplos=n_ejemplos, n_val=n_val, semilla_train=MARKET_SEED_POST_TRAIN, semilla_val=MARKET_SEED_POST_VAL,
                rmse_val_promedio_usd=float(curva["rmse_promedio_usd"].iloc[-1]), segundos=time.time() - t0)
    os.makedirs(os.path.dirname(destino), exist_ok=True)
    torch.save({"agent": agent.state_dict(), "critic": critic.state_dict(), "post": meta}, destino)
    if verbose:
        print(f"  checkpoint con las cabezas Ṽ → {destino}  ({time.time() - t0:.0f} s)")
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        curva.to_csv(os.path.join(output_dir, "post_training_curva.csv"), index=False)
    return curva


def ruta_checkpoint_pricing(ruta_actor: str) -> str:
    """am_checkpoint_<N>nodes[_best].pt → am_checkpoint_<N>nodes[_best]_pricing.pt (misma carpeta)."""
    base, ext = os.path.splitext(ruta_actor)
    return f"{base}_pricing{ext}"


def main():
    from transfer_learning import load_checkpoint_compatible
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--num-nodes", type=int, default=10)
    ap.add_argument("--checkpoint", help="checkpoint del PPO (default: checkpoints[/RUN_TAG]/am_checkpoint_<N>nodes.pt)")
    ap.add_argument("--n-ejemplos", type=int, default=POST_N_EJEMPLOS)
    ap.add_argument("--n-val", type=int, default=POST_N_VAL)
    ap.add_argument("--output-dir", help="carpeta de la curva de error (default: results/<N>nodes[_<RUN_TAG>])")
    a = ap.parse_args()

    configurar_k(False)                                           # K libre
    cwd = os.path.dirname(os.path.abspath(__file__))
    path = a.checkpoint or os.path.join(cwd, "checkpoints", RUN_TAG, f"am_checkpoint_{a.num_nodes}nodes.pt")
    agent, critic = load_checkpoint_compatible(path, a.num_nodes, AM_D_H, AM_N_HEADS, AM_N_LAYERS, AM_D_FF, verbose=False)
    out = a.output_dir or os.path.join(cwd, "results", f"{a.num_nodes}nodes" + (f"_{RUN_TAG}" if RUN_TAG else ""))
    entrenar_cabezas_post(agent, critic, cargar_datos(a.num_nodes), a.num_nodes, ruta_checkpoint_pricing(path),
                          a.n_ejemplos, a.n_val, out)


if __name__ == "__main__":
    main()
