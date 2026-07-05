import torch
from snnai.core.neurons.lif import create_lif_neuron, simulate_lif


def test_lif_spikes_with_high_input():
    neuron = create_lif_neuron(beta=0.9, threshold=1.0)
    current = torch.ones(20, 1) * 1.5
    spikes, _ = simulate_lif(neuron, current, 20)
    assert spikes.sum().item() > 0


def test_lif_no_spikes_with_low_input():
    neuron = create_lif_neuron(beta=0.9, threshold=1.0)
    current = torch.ones(20, 1) * 0.1
    spikes, _ = simulate_lif(neuron, current, 20)
    assert spikes.sum().item() == 0


def test_simulate_lif_shape():
    neuron = create_lif_neuron(beta=0.9, threshold=1.0)
    current = torch.ones(15, 3, 4)
    spikes, mems = simulate_lif(neuron, current, 15)
    assert spikes.shape == (15, 3, 4)
    assert mems.shape == (15, 3, 4)
