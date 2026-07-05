import torch
from snnai.core.synapses.stdp import stdp_update


def test_stdp_potentiation():
    T = 20
    pre = torch.zeros(T, 5)
    post = torch.zeros(T, 3)
    pre[5, :] = 1.0
    post[10, :] = 1.0
    weights = torch.zeros(3, 5)
    new_weights = stdp_update(pre, post, weights, A_plus=0.1, A_minus=0.1, tau=5.0)
    assert new_weights.sum().item() > 0


def test_stdp_depression():
    T = 20
    pre = torch.zeros(T, 5)
    post = torch.zeros(T, 3)
    pre[10, :] = 1.0
    post[5, :] = 1.0
    weights = torch.ones(3, 5) * 0.5
    new_weights = stdp_update(pre, post, weights, A_plus=0.1, A_minus=0.1, tau=5.0)
    assert new_weights.sum().item() < weights.sum().item()
