"""Environment sanity tests for Phase0."""
import torch
import snntorch
import brian2
import norse
import bindsnet


def test_imports():
    assert torch is not None
    assert snntorch is not None
    assert brian2 is not None
    assert norse is not None
    assert bindsnet is not None


def test_lif_neuron_spikes():
    from snnai.core.neurons.lif import create_lif_neuron

    lif = create_lif_neuron(beta=0.9, threshold=1.0)
    mem = lif.init_leaky()
    spikes = []
    for _ in range(20):
        current = torch.ones(1) * 1.5
        spk, mem = lif(current, mem)
        spikes.append(spk.item())
    assert sum(spikes) > 0, "LIF neuron should spike with constant high input"
