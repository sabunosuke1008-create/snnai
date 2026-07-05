"""Data loaders for MNIST and Fashion-MNIST benchmarks."""
import torch
from torchvision import datasets, transforms


def get_mnist_datasets(root="./data"):
    """Return MNIST train/test datasets as tensors."""
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    train = datasets.MNIST(root=root, train=True, download=True, transform=transform)
    test = datasets.MNIST(root=root, train=False, download=True, transform=transform)
    return train, test


def get_fashion_mnist_datasets(root="./data"):
    """Return Fashion-MNIST train/test datasets as tensors."""
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.2860,), (0.3530,))]
    )
    train = datasets.FashionMNIST(
        root=root, train=True, download=True, transform=transform
    )
    test = datasets.FashionMNIST(
        root=root, train=False, download=True, transform=transform
    )
    return train, test


def images_to_spikes(images, time_steps=20):
    """Convert a batch of images to Poisson spike trains.

    Parameters
    ----------
    images : torch.Tensor
        Tensor of shape (batch, 1, 28, 28) with pixel values in [0, 1].
    time_steps : int
        Number of time steps.

    Returns
    -------
    torch.Tensor
        Spike tensor of shape (time_steps, batch, 1, 28, 28).
    """
    # Normalize to [0, 1] probabilities
    probs = images.clamp(0, 1)
    shape = (time_steps,) + probs.shape
    return torch.bernoulli(probs.unsqueeze(0).expand(shape))
