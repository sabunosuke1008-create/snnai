import torch
from snnai.integration import EncodingBridge, Hub, SNNAIPipeline, SpikeEvent


def test_hub_routes_events():
    hub = Hub()
    received = []

    def callback(event):
        received.append(event)

    hub.subscribe("reflex", callback)
    hub.publish(SpikeEvent(timestamp=0, source="reflex", neuron_id=1))
    hub.publish(SpikeEvent(timestamp=1, source="honeybee", neuron_id=0))

    assert len(received) == 1
    assert received[0].source == "reflex"
    assert received[0].neuron_id == 1


def test_bridge_spikes_to_rates():
    bridge = EncodingBridge(in_dim=4, out_dim=4, time_steps=10)
    spikes = torch.bernoulli(torch.ones(10, 2, 4) * 0.5)
    rates = bridge.spikes_to_rates(spikes)
    assert rates.shape == (2, 4)


def test_bridge_rates_to_spikes():
    bridge = EncodingBridge(in_dim=4, out_dim=4, time_steps=10)
    rates = torch.ones(2, 4) * 0.5
    spikes = bridge.rates_to_spikes(rates)
    assert spikes.shape == (10, 2, 4)


def test_pipeline_runs():
    pipeline = SNNAIPipeline(time_steps=10)
    reflex_input = torch.zeros(10, 1, 2)
    reflex_input[:, 0, 0] = 2.0
    agent_pos = torch.tensor([[0.0, 0.0]])
    action = pipeline.forward(reflex_input, agent_pos)
    assert isinstance(action, int)
    assert 0 <= action < 4
