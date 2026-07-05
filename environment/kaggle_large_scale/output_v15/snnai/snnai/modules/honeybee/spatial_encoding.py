"""Spatial encoding helpers for the honeybee navigation module."""
import numpy as np
import torch


def place_cell_encode(pos, size=5, n_cells=25, sigma=0.8):
    """Encode a 2-D position into place-cell firing probabilities.

    A set of Gaussian radial basis functions is placed on a regular grid
    covering the environment. The response of each place cell decays
    exponentially with the squared distance from the agent position.

    Parameters
    ----------
    pos : tuple[float, float]
        The (x, y) agent position.
    size : int, optional
        Side length of the square environment (default 5).
    n_cells : int, optional
        Total number of place cells. Must be a perfect square (default 25).
    sigma : float, optional
        Width of each Gaussian place field (default 0.8).

    Returns
    -------
    torch.Tensor
        1-D tensor of shape ``(n_cells,)`` containing normalized firing
        probabilities in [0, 1].
    """
    grid_dim = int(np.sqrt(n_cells))
    xs = np.linspace(0, size - 1, grid_dim)
    ys = np.linspace(0, size - 1, grid_dim)
    centers = [(x, y) for x in xs for y in ys]
    rates = []
    for cx, cy in centers:
        d2 = (pos[0] - cx) ** 2 + (pos[1] - cy) ** 2
        rates.append(np.exp(-d2 / (2 * sigma**2)))
    rates = np.array(rates)
    rates = rates / (rates.max() + 1e-8)
    return torch.tensor(rates, dtype=torch.float32)
