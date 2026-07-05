import torch
from snnai.modules.edge import SensorFilter


def test_sensor_filter_suppresses_noise():
    filt = SensorFilter(input_size=4, beta=0.9, threshold=1.0)
    # Weak noise
    x_weak = torch.randn(20, 2, 4) * 0.2
    spikes_weak = filt(x_weak)
    # Strong signal
    x_strong = torch.zeros(20, 2, 4)
    x_strong[5:15, :, 0] = 2.0
    spikes_strong = filt(x_strong)

    assert spikes_weak.sum().item() <= spikes_strong.sum().item()
    assert spikes_strong.shape == (20, 2, 4)
