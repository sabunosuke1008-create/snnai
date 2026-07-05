import torch
from snnai.modules.c_elegans import ReflexModule


def test_reflex_avoidance_latency():
    module = ReflexModule(beta=0.9, threshold=1.0)
    x = torch.zeros(30, 1, 2)
    x[5:, 0, 0] = 2.0  # strong left stimulus from t=5

    spikes, latency = module(x)
    right_spikes = spikes[:, 0, 1].sum().item()  # right motor should spike
    assert right_spikes > 0, "Reflex should trigger avoidance right motor"
    assert latency < 30, "Reflex latency should be within simulation window"


def test_reflex_direction():
    module = ReflexModule(beta=0.9, threshold=1.0)
    x = torch.zeros(30, 1, 2)
    x[5:, 0, 1] = 2.0  # strong right stimulus

    spikes, _ = module(x)
    left_spikes = spikes[:, 0, 0].sum().item()
    right_spikes = spikes[:, 0, 1].sum().item()
    assert left_spikes > right_spikes, "Right stimulus should prefer left motor"
