"""
test_post_value_head.py
=======================
Cabezas de valor post-decisión Ṽ A y B (PostDecisionValueHead) y su integración en AMRoutingAgent.
"""

import numpy as np
import pytest
import torch

import config
from model.agent import AMRoutingAgent
from model.encoder import N_NODE_FEATURES
from model.critic_head import PostDecisionValueHead

D_H = 32


def _agent(n=10, use_post_head=True):
    return AMRoutingAgent(n, d_h=D_H, n_heads=4, n_layers=1, d_ff=64, device=torch.device("cpu"),
                          use_post_head=use_post_head)


def _context(agent, B, N):
    """h_t y embeddings reales (con grafo de gradiente hacia el encoder/context_net)."""
    nf = torch.randn(B, N, N_NODE_FEATURES)
    emb, graph = agent.encoder(nf)
    cur = emb[torch.arange(B), torch.zeros(B, dtype=torch.long), :]
    h_t = agent.context_net(graph, cur, torch.rand(B, 3), torch.rand(B, 1), torch.rand(B, 2),
                            torch.nn.functional.one_hot(torch.zeros(B, dtype=torch.long), 7).float())
    return h_t, emb


def test_agent_has_two_heads_when_enabled():
    ag = _agent()
    assert isinstance(ag.post_value_a, PostDecisionValueHead) and isinstance(ag.post_value_b, PostDecisionValueHead)
    assert ag.has_post_heads and ag.post_entrenadas is False
    off = _agent(use_post_head=False)
    assert off.post_value_a is None and off.post_value_b is None and not off.has_post_heads
    assert not any(k.startswith("post_value") for k in off.state_dict())


def test_default_follows_config():
    ag = AMRoutingAgent(10, d_h=D_H, n_heads=4, n_layers=1, d_ff=64)
    assert ag.has_post_heads == config.USE_POST_HEAD


def test_heads_share_architecture_but_not_initialization():
    ag = _agent()
    sa, sb = ag.post_value_a.state_dict(), ag.post_value_b.state_dict()
    assert sa.keys() == sb.keys() and all(sa[k].shape == sb[k].shape for k in sa)
    assert not torch.equal(sa["net.0.weight"], sb["net.0.weight"])


@pytest.mark.parametrize("N", [10, 100])
def test_shapes_same_weights(N):
    head = PostDecisionValueHead(D_H, hidden=D_H)
    out = head(torch.randn(4, D_H), torch.randn(4, N, D_H))
    assert out.shape == (4, N)


def test_output_finite():
    head = PostDecisionValueHead(D_H, hidden=D_H)
    out = head(torch.randn(8, D_H) * 10, torch.randn(8, 20, D_H) * 10)
    assert torch.isfinite(out).all()


def test_permutation_equivariance():
    head = PostDecisionValueHead(D_H, hidden=D_H)
    h, emb = torch.randn(4, D_H), torch.randn(4, 12, D_H)
    perm = torch.randperm(12)
    out, out_perm = head(h, emb), head(h, emb[:, perm])
    assert torch.allclose(out[:, perm], out_perm, atol=1e-6)


def test_forward_pairs_matches_forward():
    head = PostDecisionValueHead(D_H, hidden=D_H)
    h, emb = torch.randn(5, D_H), torch.randn(5, 9, D_H)
    a = torch.tensor([0, 3, 8, 2, 2])
    full = head(h, emb)[torch.arange(5), a]
    assert torch.allclose(head.forward_pairs(h, emb[torch.arange(5), a]), full, atol=1e-6)


def test_post_values_returns_both_heads():
    ag = _agent()
    h_t, emb = _context(ag, 3, 10)
    va, vb = ag.post_values(h_t, emb)
    assert va.shape == vb.shape == (3, 10)
    assert torch.equal(va, ag.post_value_a(h_t, emb)) and not torch.equal(va, vb)


def test_detach_blocks_gradient_to_policy_params():
    agent = _agent()
    h_t, emb = _context(agent, 4, 10)
    va, vb = agent.post_values(h_t, emb)
    (va.sum() + vb.sum()).backward()
    for name, p in agent.named_parameters():
        if name.startswith("post_value"):
            assert p.grad is not None and p.grad.abs().sum() > 0, name
        else:
            assert p.grad is None or float(p.grad.abs().sum()) == 0.0, name


def test_creating_heads_does_not_consume_global_rng():
    torch.manual_seed(7)
    a = torch.randn(3)
    torch.manual_seed(7)
    PostDecisionValueHead(D_H, hidden=D_H)
    PostDecisionValueHead(D_H, hidden=D_H, seed=5)
    assert torch.equal(a, torch.randn(3))


def test_creating_head_does_not_consume_cuda_rng():
    if not torch.cuda.is_available():
        pytest.skip("sin CUDA")
    torch.manual_seed(7)
    a = torch.randn(3, device="cuda")
    torch.manual_seed(7)
    PostDecisionValueHead(D_H, hidden=D_H)
    assert torch.equal(a, torch.randn(3, device="cuda"))
