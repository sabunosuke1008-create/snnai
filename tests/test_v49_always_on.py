import torch
from snnai.benchmarks.always_on_demo import AlwaysOnMonitor


def test_always_on_monitor():
    monitor = AlwaysOnMonitor(input_size=4, hidden_size=16, threshold=3)
    stream = torch.randn(100, 1, 4) * 0.2
    stream[40:60, :, 0] += 2.0  # anomaly
    events = monitor.monitor(stream, window_size=20)
    assert isinstance(events, list)


def test_always_on_energy():
    monitor = AlwaysOnMonitor(input_size=4, hidden_size=16, threshold=3)
    stream = torch.randn(40, 1, 4) * 0.2
    energy = monitor.energy_estimate(stream, window_size=20)
    assert "joules" in energy
