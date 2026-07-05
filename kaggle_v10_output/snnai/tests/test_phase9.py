import torch
from torchvision import datasets, transforms

from snnai.benchmarks.event_utils import events_to_spike_tensor
from snnai.benchmarks.nmnist_loader import (
    load_nmnist_like_from_mnist,
    simulate_events_from_frames,
)


def test_simulate_events_from_frames():
    frames = torch.zeros(10, 1, 28, 28)
    frames[5:, 0, 10:15, 10:15] = 0.5
    events = simulate_events_from_frames(frames, threshold=0.1)
    assert len(events) > 0
    assert all("t" in e and "x" in e and "y" in e and "pol" in e for e in events)


def test_events_to_spike_tensor():
    events = [
        {"t": 0, "x": 5, "y": 5, "pol": 1},
        {"t": 10, "x": 20, "y": 20, "pol": -1},
    ]
    spikes = events_to_spike_tensor(events, height=28, width=28, time_steps=20)
    assert spikes.shape == (20, 1, 28, 28)
    assert spikes[0, 0, 5, 5] == 1.0
    assert spikes[19, 0, 20, 20] == 1.0


def test_load_nmnist_like_from_mnist():
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    image, _ = dataset[0]
    events = load_nmnist_like_from_mnist(image, n_frames=10, seed=1)
    assert isinstance(events, list)
    spikes = events_to_spike_tensor(events, height=28, width=28, time_steps=20)
    assert spikes.shape == (20, 1, 28, 28)
