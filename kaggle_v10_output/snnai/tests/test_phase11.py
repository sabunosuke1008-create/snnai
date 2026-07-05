import os
import tempfile

import torch

from snnai.benchmarks.large_mnist_snn import LargeMNISTSNN
from snnai.utils import auto_select_device, load_checkpoint, save_checkpoint


def test_auto_select_device():
    device = auto_select_device()
    assert device.type in ("cuda", "mps", "cpu")


def test_checkpoint_roundtrip():
    model = LargeMNISTSNN(time_steps=5)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "checkpoint.pt")
        save_checkpoint(model, optimizer, epoch=3, path=path)

        model2 = LargeMNISTSNN(time_steps=5)
        optimizer2 = torch.optim.Adam(model2.parameters(), lr=0.001)
        epoch = load_checkpoint(model2, optimizer2, path)

        assert epoch == 3
        for p1, p2 in zip(model.parameters(), model2.parameters()):
            assert torch.allclose(p1, p2)


def test_large_mnist_snn_forward():
    model = LargeMNISTSNN(time_steps=5)
    x = torch.bernoulli(torch.ones(5, 2, 1, 28, 28) * 0.3)
    out = model(x)
    assert out.shape == (5, 2, 10)
