"""MLP baseline for comparison with SNNAI."""
import torch
import torch.nn as nn

from .synthetic_benchmark import generate_synthetic_dataset


class MLPBaseline(nn.Module):
    """Small MLP baseline matching SNNAI scale."""

    def __init__(self, input_dim=4, hidden_dim=16, output_dim=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.net(x)


def train_mlp_baseline(n_samples=400, epochs=20, seed=42):
    """Train and evaluate the MLP baseline on the synthetic benchmark."""
    x, y = generate_synthetic_dataset(n_samples=n_samples, seed=seed)
    x_mean = x.mean(dim=1)

    model = MLPBaseline(input_dim=4, hidden_dim=16, output_dim=4)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    for _ in range(epochs):
        optimizer.zero_grad()
        logits = model(x_mean)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        logits = model(x_mean)
        preds = torch.argmax(logits, dim=1)
        accuracy = (preds == y).float().mean().item()

    param_count = sum(p.numel() for p in model.parameters())
    return {"accuracy": accuracy, "parameters": param_count}
