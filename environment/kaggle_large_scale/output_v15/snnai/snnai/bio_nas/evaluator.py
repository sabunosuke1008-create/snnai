"""Lightweight evaluator for Bio-NAS architectures.

The evaluator uses deterministic surrogate modules so that architecture
search can run quickly without training real SNNs. Each module type is
modelled by a tiny fixed-weight linear layer.
"""
import torch
import torch.nn as nn

from .search_space import MODULES


class SurrogateModule(nn.Module):
    """Deterministic surrogate for a biological module.

    Maps a hidden vector to another hidden vector using a fixed random
    weight matrix.
    """

    _weights = {}

    def __init__(self, module_type, dim, seed=0):
        super().__init__()
        self.module_type = module_type
        key = (module_type, dim, seed)
        if key not in SurrogateModule._weights:
            g = torch.Generator().manual_seed(seed + hash(module_type) % 1000)
            SurrogateModule._weights[key] = torch.randn(dim, dim, generator=g) * 0.5
        self.weight = nn.Parameter(SurrogateModule._weights[key].clone(), requires_grad=False)

    def forward(self, x):
        # x: (batch, dim)
        return torch.matmul(x, self.weight.T)


def make_surrogate_graph(arch, feature_dim=4, hidden_dim=4, output_dim=4, seed=0):
    """Build an executable surrogate graph from an architecture."""
    order = arch.topological_order()
    modules = {}
    for node in order:
        if node == "input" or node == "output":
            continue
        mod_type = arch.module_types[node]
        modules[node] = SurrogateModule(mod_type, hidden_dim, seed=seed)

    # Input projection to hidden_dim
    input_key = ("input_projection", feature_dim, hidden_dim, seed)
    if input_key not in SurrogateModule._weights:
        g = torch.Generator().manual_seed(seed)
        SurrogateModule._weights[input_key] = torch.randn(hidden_dim, feature_dim, generator=g) * 0.5
    input_weight = SurrogateModule._weights[input_key]

    def forward(x):
        # x: (batch, time, feature_dim)
        cache = {"input": torch.matmul(x.mean(dim=1), input_weight.T)}
        for node in order:
            if node == "input":
                continue
            preds = [src for src, dst in arch.edges if dst == node]
            if not preds:
                continue
            combined = sum(cache[p] for p in preds)
            if node == "output":
                cache[node] = combined
            else:
                cache[node] = modules[node](combined)
        return cache["output"]

    return forward


def generate_synthetic_data(n_samples=200, feature_dim=4, n_classes=4, seed=42):
    """Generate a simple synthetic multi-modal classification dataset."""
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n_samples, 10, feature_dim, generator=g)
    y = torch.zeros(n_samples, dtype=torch.long)
    for i in range(n_samples):
        ch0 = x[i, :, 0].mean().item()
        ch1 = x[i, :, 1].mean().item()
        ch2 = x[i, :, 2].mean().item()
        if abs(ch0) > 0.3 and abs(ch0) > abs(ch1) and abs(ch0) > abs(ch2):
            y[i] = 1
        elif abs(ch1) > 0.3 and abs(ch1) > abs(ch0) and abs(ch1) > abs(ch2):
            y[i] = 2
        elif abs(ch2) > 0.3 and abs(ch2) > abs(ch0) and abs(ch2) > abs(ch1):
            y[i] = 3
        else:
            y[i] = 0
    return x, y


def evaluate_architecture(arch, n_samples=200, seed=42):
    """Return classification accuracy on the synthetic task."""
    if not arch.is_valid():
        return 0.0
    x, y = generate_synthetic_data(n_samples=n_samples, seed=seed)
    forward = make_surrogate_graph(arch, seed=seed)
    with torch.no_grad():
        logits = forward(x)
    preds = torch.argmax(logits, dim=1)
    accuracy = (preds == y).float().mean().item()
    return accuracy
