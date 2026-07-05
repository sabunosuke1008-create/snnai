"""Larger-scale SNN classifier for MNIST / Fashion-MNIST."""
import torch
import torch.nn as nn
import snntorch as snn


class LargeMNISTSNN(nn.Module):
    """Larger convolutional SNN for MNIST/Fashion-MNIST."""

    def __init__(self, beta=0.9, threshold=1.0, time_steps=20):
        super().__init__()
        self.time_steps = time_steps
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1, bias=False)
        self.lif1 = snn.Leaky(beta=beta, threshold=threshold, learn_threshold=False)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1, bias=False)
        self.lif2 = snn.Leaky(beta=beta, threshold=threshold, learn_threshold=False)
        self.fc1 = nn.Linear(32 * 14 * 14, 128, bias=False)
        self.lif3 = snn.Leaky(beta=beta, threshold=threshold, learn_threshold=False)
        self.fc2 = nn.Linear(128, 10, bias=False)
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
            mem1 = torch.zeros(batch_size, 16, 28, 28)
        mem2 = torch.zeros(batch_size, 32, 14, 14)
        mem3 = torch.zeros(batch_size, 128)
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
