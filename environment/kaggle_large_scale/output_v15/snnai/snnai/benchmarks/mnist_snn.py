"""SNN classifier for MNIST / Fashion-MNIST."""
import torch
import torch.nn as nn
import snntorch as snn

from .data_loader import images_to_spikes


class MNISTSNN(nn.Module):
    """Small convolutional SNN for MNIST/Fashion-MNIST."""

    def __init__(self, beta=0.9, threshold=1.0, time_steps=20):
        super().__init__()
        self.time_steps = time_steps
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, padding=1, bias=False)
        self.lif1 = snn.Leaky(beta=beta, threshold=threshold, learn_threshold=False)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1, bias=False)
        self.lif2 = snn.Leaky(beta=beta, threshold=threshold, learn_threshold=False)
        self.fc1 = nn.Linear(16 * 14 * 14, 64, bias=False)
        self.lif3 = snn.Leaky(beta=beta, threshold=threshold, learn_threshold=False)
        self.fc2 = nn.Linear(64, 10, bias=False)
        self.lif4 = snn.Leaky(beta=beta, threshold=threshold, learn_threshold=False)
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input spike tensor of shape (time, batch, 1, 28, 28).

        Returns
        -------
        torch.Tensor
            Output spikes of shape (time, batch, 10).
        """
        time_steps, batch_size, *_ = x.shape
        mem1 = self.lif1.init_leaky()
        if mem1.dim() == 0:
            mem1 = torch.zeros(batch_size, 8, 28, 28)
        mem2 = torch.zeros(batch_size, 16, 14, 14)
        mem3 = torch.zeros(batch_size, 64)
        mem4 = torch.zeros(batch_size, 10)

        out_spikes = []
        for t in range(time_steps):
            cur1 = self.conv1(x[t])
            spk1, mem1 = self.lif1(cur1, mem1)
            cur2 = self.pool(self.conv2(spk1))
            spk2, mem2 = self.lif2(cur2, mem2)
            flat = spk2.view(batch_size, -1)
            cur3 = self.fc1(flat)
            spk3, mem3 = self.lif3(cur3, mem3)
            cur4 = self.fc2(spk3)
            spk4, mem4 = self.lif4(cur4, mem4)
            out_spikes.append(spk4)
        return torch.stack(out_spikes)


def train_mnist_snn(
    model,
    train_loader,
    device="cpu",
    epochs=1,
    lr=0.001,
    time_steps=20,
):
    """Train the SNN on MNIST/Fashion-MNIST."""
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        total_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            spikes = images_to_spikes(images, time_steps=time_steps).to(device)

            optimizer.zero_grad()
            output = model(spikes)
            # Sum spikes over time and classify
            logits = output.sum(dim=0)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
    return total_loss


def evaluate_mnist_snn(model, test_loader, device="cpu", time_steps=20):
    """Evaluate the SNN."""
    model.to(device)
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            spikes = images_to_spikes(images, time_steps=time_steps).to(device)
            output = model(spikes)
            logits = output.sum(dim=0)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total if total > 0 else 0.0
