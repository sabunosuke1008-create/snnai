import torch
from snnai.core.reservoir.liquid_state_machine import LiquidStateMachine


def test_lsm_forward_shape():
    lsm = LiquidStateMachine(input_size=4, reservoir_size=16, beta=0.9)
    x = torch.rand(30, 2, 4)
    spikes = lsm(x)
    assert spikes.shape == (30, 2, 16)


def test_lsm_reservoir_sparsity():
    lsm = LiquidStateMachine(input_size=4, reservoir_size=16, sparsity=0.1)
    w = lsm.W_rec.weight.data
    nonzero = (w != 0).float().mean().item()
    assert 0.0 < nonzero < 0.2
