"""
test_post_value_head.py
=======================
Fase P0 de PRICING_PLAN.md: PostDecisionValueHead y su integración en AMRoutingAgent.
"""

import pytest
import torch

import config
from am_agent import AMRoutingAgent
from attention_encoder import N_NODE_FEATURES
from critic_head import PostDecisionValueHead

D_H = 32


def _agent(n=10):
    return AMRoutingAgent(n, d_h=D_H, n_heads=4, n_layers=1, d_ff=64, device=torch.device("cpu"),
                          use_post_head=True)


def _context(agent, B, N):
    """h_t y embeddings reales (con grafo de gradiente hacia el encoder/context_net)."""
    nf = torch.randn(B, N, N_NODE_FEATURES)
    emb, graph = agent.encoder(nf)
    cur = emb[torch.arange(B), torch.zeros(B, dtype=torch.long), :]
    h_t = agent.context_net(graph, cur, torch.rand(B, 3), torch.rand(B, 1), torch.rand(B, 2),
                            torch.nn.functional.one_hot(torch.zeros(B, dtype=torch.long), 7).float())
    return h_t, emb


def test_agent_has_post_value_when_enabled():
    assert isinstance(_agent().post_value, PostDecisionValueHead)
    off = AMRoutingAgent(10, d_h=D_H, n_heads=4, n_layers=1, d_ff=64, device=torch.device("cpu"),
                         use_post_head=False)
    assert off.post_value is None
    assert not any(k.startswith("post_value") for k in off.state_dict())


def test_default_follows_config():
    assert (AMRoutingAgent(10, d_h=D_H, n_heads=4, n_layers=1, d_ff=64).post_value is not None) == config.USE_POST_HEAD


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


def test_detach_blocks_gradient_to_policy_params():
    agent = _agent()
    h_t, emb = _context(agent, 4, 10)
    agent.post_value(h_t, emb).sum().backward()
    for name, p in agent.named_parameters():
        if name.startswith("post_value"):
            assert p.grad is not None and p.grad.abs().sum() > 0, name
        else:
            assert p.grad is None or float(p.grad.abs().sum()) == 0.0, name


def test_creating_head_does_not_consume_global_rng():
    torch.manual_seed(7)
    a = torch.randn(3)
    torch.manual_seed(7)
    PostDecisionValueHead(D_H, hidden=D_H)
    assert torch.equal(a, torch.randn(3))


def test_creating_head_does_not_consume_cuda_rng():
    if not torch.cuda.is_available():
        pytest.skip("sin CUDA")
    torch.manual_seed(7)
    a = torch.randn(3, device="cuda")
    torch.manual_seed(7)
    PostDecisionValueHead(D_H, hidden=D_H)
    assert torch.equal(a, torch.randn(3, device="cuda"))
