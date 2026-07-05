"""Trainable end-to-end integration pipeline."""
import torch
import torch.nn as nn
import snntorch as snn

from snnai.modules.c_elegans import ReflexModule
from snnai.modules.crow import WorkingMemory


class DifferentiableBridge(nn.Module):
    """Learnable bridge that maps spike trains to spike trains.

    During training, it returns continuous rates so that gradients can
    flow through. During evaluation, it can return stochastic spikes.
    """

    def __init__(self, in_dim, out_dim, time_steps=20):
        super().__init__()
        self.time_steps = time_steps
        self.fc = nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, spikes, training=True):
        # spikes: (time, batch, in_dim)
        rates = torch.sigmoid(self.fc(spikes))
        if training:
            return rates
        return torch.bernoulli(rates)


class TrainablePipeline(nn.Module):
    """End-to-end trainable pipeline connecting reflex and memory modules.

    Input: sensory stimulus of shape (time, batch, 2)
    Output: logits of shape (batch, 4)
    """

    def __init__(self, time_steps=20, beta=0.9, threshold=1.0):
        super().__init__()
        self.time_steps = time_steps
        self.reflex = ReflexModule(beta=beta, threshold=threshold)
        self.bridge = DifferentiableBridge(2, 2, time_steps=time_steps)
        self.memory = WorkingMemory(beta=beta, threshold=threshold)
        self.readout = nn.Linear(2, 4, bias=False)

    def forward(self, x):
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input of shape (time, batch, 2).

        Returns
        -------
        torch.Tensor
            Logits of shape (batch, 4).
        """
        reflex_spikes, _ = self.reflex(x)
        bridge_out = self.bridge(reflex_spikes, training=self.training)
        mem_spikes = self.memory(bridge_out)
        rates = mem_spikes.mean(dim=0)  # (batch, 2)
        return self.readout(rates)


def generate_integration_data(n_samples=400, time_steps=20, seed=42):
    """Generate a synthetic task for the trainable pipeline.

    Classes:
    0: low both
    1: strong left
    2: strong right
    3: strong both
    """
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n_samples, time_steps, 2, generator=g) * 0.5
    y = torch.zeros(n_samples, dtype=torch.long)
    for i in range(n_samples):
        left = x[i, :, 0].mean().item()
        right = x[i, :, 1].mean().item()
        if left > 0.3 and right > 0.3:
            y[i] = 3
        elif left > 0.3:
            y[i] = 1
        elif right > 0.3:
            y[i] = 2
        else:
            y[i] = 0
    # Add strong stimulus pulses
    for i in range(n_samples):
        label = y[i].item()
        if label == 1 or label == 3:
            x[i, 5:15, 0] += 1.5
        if label == 2 or label == 3:
            x[i, 5:15, 1] += 1.5
    return x, y


def train_pipeline(n_samples=400, epochs=5, lr=0.01, seed=42):
    """Train the pipeline on the synthetic integration task."""
    x, y = generate_integration_data(n_samples=n_samples, seed=seed)
    model = TrainablePipeline(time_steps=x.shape[0])
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    # Split train/test
    split = int(n_samples * 0.8)
    x_train, y_train = x[:split], y[:split]
    x_test, y_test = x[split:], y[split:]

    for _ in range(epochs):
        model.train()
        optimizer.zero_grad()
        logits = model(x_train.permute(1, 0, 2))
        loss = criterion(logits, y_train)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        logits = model(x_test.permute(1, 0, 2))
        preds = torch.argmax(logits, dim=1)
        accuracy = (preds == y_test).float().mean().item()
    return model, accuracy
