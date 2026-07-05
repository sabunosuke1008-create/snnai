import torch
from torch.utils.data import DataLoader, Subset

from snnai.benchmarks.data_loader import get_mnist_datasets, images_to_spikes
from snnai.benchmarks.mnist_snn import MNISTSNN, evaluate_mnist_snn, train_mnist_snn


def test_images_to_spikes_shape():
    images = torch.rand(8, 1, 28, 28)
    spikes = images_to_spikes(images, time_steps=10)
    assert spikes.shape == (10, 8, 1, 28, 28)


def test_mnist_snn_training_runs():
    train, test = get_mnist_datasets(root="./data")
    # Use small subset for fast test
    train_subset = Subset(train, range(500))
    test_subset = Subset(test, range(100))
    train_loader = DataLoader(train_subset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_subset, batch_size=32)

    model = MNISTSNN(time_steps=10)
    loss = train_mnist_snn(
        model, train_loader, device="cpu", epochs=1, time_steps=10
    )
    acc = evaluate_mnist_snn(model, test_loader, device="cpu", time_steps=10)
    print(f"Phase8 test loss: {loss:.3f}, accuracy: {acc:.3f}")
    assert acc > 0.0
