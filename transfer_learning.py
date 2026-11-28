"""
transfer_learning.py
====================
Transfer learning estandarizado para la cadena de escalado:

    20 → 35 → 50 → 75 nodos

Por qué el transfer es siempre 100% en este modelo
---------------------------------------------------
Ninguna capa del AMRoutingAgent ni del CriticHead tiene dimensiones
que dependan de num_nodes. El Transformer opera sobre conjuntos de
embeddings de tamaño variable — lo que cambia entre tamaños es el
número de nodos en runtime, no los pesos del modelo.

  encoder.input_proj   : N_NODE_FEATURES → d_h  (independiente de N; igual en el encoder del crítico)
  encoder.layers[i]    : d_h → d_h        (independiente de N)
  context_net.proj     : 2·d_h+3+n_market+n_budget → d_h  (independiente de N)
  decoder.glimpse      : d_h → d_h        (independiente de N)
  decoder.pointer      : d_h → logit      (independiente de N)
  critic.*             : encoder + context propios, d_h → 1  (independiente de N)

Uso desde main.py
-----------------
  from transfer_learning import get_source_checkpoint, transfer_checkpoint

  # Obtener checkpoint del tamaño anterior automáticamente
  source = get_source_checkpoint(cwd, target_num_nodes=20)
  agent, critic = transfer_checkpoint(source, target_num_nodes=20)

Uso standalone
--------------
  python transfer_learning.py
  → Verifica que la cadena completa de transfers es válida.
"""

import os
import torch
from model.agent import AMRoutingAgent
from model.critic_head import CriticHead
from config import DEVICE, CRITIC_N_LAYERS


# ─────────────────────────────────────────────────────────────────────────────
# Secuencia canónica de escalado
# ─────────────────────────────────────────────────────────────────────────────

NODE_SEQUENCE = [20, 35, 50, 75]


# ─────────────────────────────────────────────────────────────────────────────
# Reconciliación de state_dicts (compatibilidad de checkpoints)
# ─────────────────────────────────────────────────────────────────────────────

def _reconcile_state_dict(target_sd: dict, source_sd: dict) -> tuple:
    """
    Fusiona un state_dict de checkpoint (source_sd) en un state_dict destino
    ya inicializado (target_sd), tolerando cambios de arquitectura.

    Reglas, en orden:
      1. Shape idéntica → se copia el tensor del checkpoint tal cual.
      2. Caso especial de expansión de entrada: mismo out_features y
         in_features del checkpoint < in_features del destino (ej.
         context_net.proj.0.weight al añadir el bloque de budget features, o
         encoder.input_proj.proj.weight al pasar de 7 a 11 features de nodo con
         USE_CONTINUATION_FEATURES, o de 11 a 12 al añadir el índice de mercado como última columna:
         copia las columnas viejas y deja las nuevas en cero) →
         se conserva el tensor destino (ya inicializado) pero se ponen TODAS
         sus columnas en cero y se copian las columnas viejas en las primeras
         posiciones. Con las columnas nuevas en cero, la contribución de las
         features añadidas es nula hasta que el fine-tuning las active, así
         que el forward pass es idéntico al del checkpoint original.
      3. Clave del checkpoint ausente en el destino → se omite (skipped).
      4. Clave presente en ambos con una forma incompatible que no es una
         expansión válida (p. ej. input_proj de un checkpoint con más
         features de nodo que el modelo actual, o una capa con otra
         dimensión) → ValueError: el checkpoint no corresponde a la
         arquitectura actual y omitir la capa en silencio dejaría el
         modelo con esa capa aleatoria sin aviso.
      Las claves del destino que el checkpoint no trae conservan su
      inicialización.

    Retorna (merged_sd, transferred: list[str], skipped: list[str]).
    """
    merged      = dict(target_sd)
    transferred = []
    skipped     = []

    for k, v in source_sd.items():
        if k not in merged:
            skipped.append(k)
            continue

        target_shape = merged[k].shape

        if v.shape == target_shape:
            merged[k] = v
            transferred.append(k)
        elif (
            v.dim() == 2
            and target_shape[0] == v.shape[0]
            and v.shape[1] < target_shape[1]
        ):
            new_w = torch.zeros_like(merged[k])
            new_w[:, : v.shape[1]] = v
            merged[k] = new_w
            transferred.append(f"{k} (expandido, columnas nuevas en cero)")
        else:
            raise ValueError(
                f"Checkpoint incompatible con la arquitectura actual: '{k}' tiene forma "
                f"{tuple(v.shape)} en el checkpoint y {tuple(target_shape)} en el modelo "
                f"(no es una expansión de columnas de entrada). Reentrena desde cero o "
                f"usa un checkpoint de esta arquitectura."
            )

    return merged, transferred, skipped


def _tiene_cabeza_de_precio(checkpoint: dict) -> bool:
    """La cabeza de precio actual (con Δ, model/decoder.PriceHead) y las opciones actuales (N_NIVELES + VACÍO)."""
    from pricing import N_NIVELES
    agente = checkpoint["agent"]
    salida = agente.get("decoder.precio.net.2.weight")
    return "decoder.precio.log_tau" in agente and salida is not None and salida.shape[0] == N_NIVELES + 1


def _tiene_entradas_actuales(checkpoint: dict) -> bool:
    """Los encoders del agente y del crítico tienen las columnas de nodo actuales (model.encoder.N_NODE_FEATURES). Un
    checkpoint con menos columnas (34 antes del 2026-11-27, 38 antes del 2026-11-28, sin el tablero de hoy) NO se
    expande: sus columnas de continuación e índice significan otra cosa y `_reconcile_state_dict` las copiaría
    desalineadas en silencio. Tampoco sirve uno cuyo crítico no tiene encoder propio (antes del 2026-11-28)."""
    from model.encoder import N_NODE_FEATURES
    pesos = checkpoint["agent"].get("encoder.input_proj.proj.weight")
    pesos_c = checkpoint.get("critic", {}).get("encoder.input_proj.proj.weight")   # crítico con encoder propio
    return (pesos is not None and pesos.shape[1] == N_NODE_FEATURES
            and pesos_c is not None and pesos_c.shape[1] == N_NODE_FEATURES)


def es_checkpoint_con_cotizacion(checkpoint_path: str) -> bool:
    """True si el .pt existe, trae la cabeza de precio actual (ver `_tiene_cabeza_de_precio`) y las columnas de nodo
    actuales (ver `_tiene_entradas_actuales`); un checkpoint anterior al 2026-11-25 o al 2026-11-27 da False."""
    if not os.path.exists(checkpoint_path):
        return False
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    return _tiene_cabeza_de_precio(checkpoint) and _tiene_entradas_actuales(checkpoint)


def _instanciar_y_transferir(
    checkpoint_path: str,
    num_nodes:       int,
    d_h:      int,
    n_heads:  int,
    n_layers: int,
    d_ff:     int,
) -> tuple:
    """Carga el .pt, instancia agent+critic con la arquitectura actual y les transfiere los
    pesos vía _reconcile_state_dict. Compartido por load_checkpoint_compatible (mismo tamaño
    de grafo) y transfer_checkpoint (tamaño mayor); solo cambia num_nodes y el mensaje verbose.

    Retorna
    -------
    agent, critic, checkpoint, transferred (lista combinada), skipped (lista combinada)
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint no encontrado: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    if not _tiene_cabeza_de_precio(checkpoint):
        # Checkpoint de antes de la cotización (sin PriceHead): cargarlo dejaría la cabeza de precio aleatoria
        # y las features nuevas en cero sin aviso, y el transfer o la evaluación usarían un modelo viejo.
        raise ValueError(
            f"{checkpoint_path} es anterior a la cabeza de precio actual (decoder.precio). "
            f"Bórralo o vuelve a entrenar ese tamaño."
        )
    if not _tiene_entradas_actuales(checkpoint):
        raise ValueError(
            f"{checkpoint_path} tiene otras columnas de nodo que el modelo actual, o un crítico sin encoder propio "
            f"(checkpoints de antes del 2026-11-28). Bórralo o vuelve a entrenar ese tamaño."
        )

    agent  = AMRoutingAgent(
        num_nodes=num_nodes, d_h=d_h, n_heads=n_heads, n_layers=n_layers, d_ff=d_ff,
        device=DEVICE,
    )
    critic = CriticHead(d_h, n_heads, CRITIC_N_LAYERS, d_ff).to(DEVICE)

    agent_sd,  transferred_a, skipped_a = _reconcile_state_dict(agent.state_dict(),  checkpoint["agent"])
    critic_sd, transferred_c, skipped_c = _reconcile_state_dict(critic.state_dict(), checkpoint["critic"])

    agent.load_state_dict(agent_sd)
    critic.load_state_dict(critic_sd)

    return agent, critic, checkpoint, transferred_a + transferred_c, skipped_a + skipped_c


def load_checkpoint_compatible(
    checkpoint_path: str,
    num_nodes:       int,
    d_h:      int = 128,
    n_heads:  int = 8,
    n_layers: int = 3,
    d_ff:     int = 512,
    verbose:  bool = True,
) -> tuple:
    """
    Carga un checkpoint del MISMO tamaño de grafo, tolerando la expansión de
    ContextNetwork.proj introducida por el condicionamiento en K (columnas
    nuevas inicializadas en cero — ver _reconcile_state_dict). Para un
    checkpoint ya compatible con la arquitectura actual, el resultado es
    observacionalmente idéntico a un torch.load + load_state_dict estricto.

    Uso: reemplaza el torch.load + load_state_dict directo en
    main.py._load_agent e inference.py.load_agent.
    """
    agent, critic, checkpoint, transferred, skipped = _instanciar_y_transferir(
        checkpoint_path, num_nodes, d_h, n_heads, n_layers, d_ff,
    )

    if verbose:
        total = len(transferred) + len(skipped)
        pct   = 100 * len(transferred) / max(total, 1)
        print(f"  [load_checkpoint_compatible] {os.path.basename(checkpoint_path)} "
              f"({len(transferred)}/{total} tensores, {pct:.1f}%)", end="")
        print(f" — omitidos: {skipped}" if skipped else " ✓")

    return agent, critic


def get_source_checkpoint(base_dir: str, target_num_nodes: int) -> str | None:
    """Dado target_num_nodes, devuelve la ruta al checkpoint del tamaño anterior en
    NODE_SEQUENCE, o None si target es el primero (20 nodos).

    Busca primero en results/{N}nodes/ y luego en la raíz del proyecto.
    """
    if target_num_nodes not in NODE_SEQUENCE:
        raise ValueError(
            f"{target_num_nodes} no está en NODE_SEQUENCE {NODE_SEQUENCE}.\n"
            f"Añádelo si quieres usarlo en la cadena de transfer."
        )

    idx = NODE_SEQUENCE.index(target_num_nodes)

    # Primer tamaño de la cadena — no tiene predecesor
    if idx == 0:
        return None

    source_nodes = NODE_SEQUENCE[idx - 1]
    fname = f"am_checkpoint_{source_nodes}nodes.pt"

    # Buscar: 1) raíz  2) carpeta compartida checkpoints/  3) results/Nnodes/
    candidates = [
        os.path.join(base_dir, fname),
        os.path.join(os.path.dirname(base_dir), "checkpoints", fname),
        os.path.join(base_dir, "results", f"{source_nodes}nodes", fname),
    ]

    for path in candidates:
        if os.path.exists(path):
            return path

    raise FileNotFoundError(
        f"No se encontró el checkpoint de {source_nodes} nodos en ninguna de:\n"
        + "\n".join(f"  · {p}" for p in candidates)
        + f"\nCopia el archivo a checkpoints/ antes de entrenar {target_num_nodes} nodos."
    )


# ─────────────────────────────────────────────────────────────────────────────
# Transfer principal
# ─────────────────────────────────────────────────────────────────────────────

def transfer_checkpoint(
    source_checkpoint: str,
    target_num_nodes:  int,
    d_h:      int = 128,
    n_heads:  int = 8,
    n_layers: int = 3,
    d_ff:     int = 512,
    verbose:  bool = True,
) -> tuple:
    """Transfiere los pesos de un checkpoint entrenado a un agente de mayor tamaño.

    Los pesos independientes de N se copian tal cual. Si la arquitectura del
    checkpoint fuente es más antigua que la actual, _reconcile_state_dict lo
    tolera: las capas lineales con menos columnas de entrada (p. ej.
    context_net.proj al añadir budget features) se expanden con columnas nuevas
    en cero, y cualquier otro mismatch de forma se omite y queda con su
    inicialización aleatoria. d_h, n_heads, n_layers, d_ff deben coincidir con la fuente.
    """
    agent, critic, _checkpoint, transferred, skipped = _instanciar_y_transferir(
        source_checkpoint, target_num_nodes, d_h, n_heads, n_layers, d_ff,
    )

    total = len(agent.state_dict()) + len(critic.state_dict())
    pct   = 100 * len(transferred) / total

    if verbose:
        source_name = os.path.basename(source_checkpoint)
        print(f"\n  [Transfer] {source_name} → {target_num_nodes} nodos")
        print(f"             {len(transferred)}/{total} tensores ({pct:.1f}%)", end="")
        if skipped:
            print(f" — omitidos: {skipped}")
        else:
            print(" ✓")

    return agent, critic


# ─────────────────────────────────────────────────────────────────────────────
# Integración con main.py
# ─────────────────────────────────────────────────────────────────────────────

def build_agent_for_training(
    base_dir:        str,
    num_nodes:       int,
    use_transfer:    bool = True,
    d_h:      int = 128,
    n_heads:  int = 8,
    n_layers: int = 3,
    d_ff:     int = 512,
    verbose:  bool = True,
) -> tuple:
    """Función de conveniencia para main.py: si use_transfer=True y existe un checkpoint
    del tamaño anterior en NODE_SEQUENCE, inicializa el agente con esos pesos (transfer
    learning); si no, crea un agente con pesos aleatorios.
    """
    if use_transfer and num_nodes in NODE_SEQUENCE:
        try:
            source = get_source_checkpoint(base_dir, num_nodes)
            if source is not None:
                return transfer_checkpoint(
                    source, num_nodes, d_h, n_heads, n_layers, d_ff, verbose
                )
        except FileNotFoundError as e:
            print(f"\n  ⚠ Transfer omitido: {e}")
            print(f"    Iniciando {num_nodes} nodos con pesos aleatorios.\n")

    # Sin transfer: pesos aleatorios
    if verbose:
        print(f"\n  [Transfer] {num_nodes} nodos — pesos aleatorios (sin fuente).")
    agent  = AMRoutingAgent(num_nodes, d_h, n_heads, n_layers, d_ff, device=DEVICE)
    critic = CriticHead(d_h, n_heads, CRITIC_N_LAYERS, d_ff).to(DEVICE)
    return agent, critic


# ─────────────────────────────────────────────────────────────────────────────
# Verificación de la cadena completa (standalone)
# ─────────────────────────────────────────────────────────────────────────────

def verify_chain(base_dir: str):
    """
    Verifica qué checkpoints existen y cuáles transfers son posibles
    en la cadena completa NODE_SEQUENCE.
    """
    print("\n" + "=" * 55)
    print("  Transfer Learning — Estado de la cadena")
    print(f"  Secuencia: {' → '.join(str(n) for n in NODE_SEQUENCE)}")
    print("=" * 55)

    for i, n in enumerate(NODE_SEQUENCE):
        # Buscar checkpoint de este tamaño
        candidates = [
            os.path.join(base_dir, "results", f"{n}nodes", f"am_checkpoint_{n}nodes.pt"),
            os.path.join(base_dir, f"am_checkpoint_{n}nodes.pt"),
        ]
        exists = next((p for p in candidates if os.path.exists(p)), None)

        if i == 0:
            origen = "base"
        else:
            prev = NODE_SEQUENCE[i - 1]
            origen = f"← {prev} nodos"

        estado = f"✓  {exists}" if exists else "✗  (no entrenado aún)"
        print(f"  {n:>3} nodos  [{origen:>12}]  {estado}")

    print("=" * 55 + "\n")


def main():
    cwd = os.path.dirname(os.path.abspath(__file__))
    verify_chain(cwd)

    # Demo: simular transfer de 10 → 20 si existe el checkpoint
    try:
        source = get_source_checkpoint(cwd, target_num_nodes=20)
        agent, critic = transfer_checkpoint(source, target_num_nodes=20, verbose=True)
        total_params = sum(p.numel() for p in agent.parameters())
        print(f"  Parámetros transferidos al agente de 20 nodos: {total_params:,}")
    except FileNotFoundError as e:
        print(f"  Demo omitida: {e}")


if __name__ == "__main__":
    main()
