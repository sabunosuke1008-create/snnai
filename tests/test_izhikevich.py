import torch
from snnai.core.neurons.izhikevich import create_izhikevich, PRESETS


def test_izhikevich_preset():
    neuron = create_izhikevich("RS")
    assert neuron.a == PRESETS["RS"]["a"]


def test_izhikevich_spike():
    neuron = create_izhikevich("RS")
    v, u = neuron.init_state(batch_size=1)
    spikes = []
    for _ in range(200):
        I = torch.tensor([10.0])
        spk, v, u = neuron(I, v, u)
        spikes.append(spk.item())
    assert sum(spikes) > 0


def test_izhikevich_batch():
    neuron = create_izhikevich("FS")
    v, u = neuron.init_state(batch_size=4)
    I = torch.ones(4) * 15.0
    spk, v, u = neuron(I, v, u)
    assert spk.shape == (4,)
