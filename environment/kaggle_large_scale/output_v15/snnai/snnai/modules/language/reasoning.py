"""Mammal-cortex inspired reasoning module."""
import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate


class ReasoningModule(nn.Module):
    """Small reasoning layer that integrates context and produces outputs.

    Parameters
    ----------
    input_dim : int
        Input feature dimension.
    hidden_dim : int, optional
        Hidden layer size (default 64).
    output_dim : int, optional
        Output dimension (default input_dim).
    """

    def __init__(self, input_dim, hidden_dim=64, output_dim=None):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim or input_dim
        self.fc1 = nn.Linear(input_dim, hidden_dim, bias=False)
        self.fc2 = nn.Linear(hidden_dim, self.output_dim, bias=False)
        self.lif1 = snn.Leaky(beta=0.9, threshold=1.0, learn_threshold=False,
                              spike_grad=surrogate.fast_sigmoid())
        self.lif2 = snn.Leaky(beta=0.9, threshold=1.0, learn_threshold=False,
                              spike_grad=surrogate.fast_sigmoid())

    def forward(self, x, time_steps=20):
        """Run reasoning.

        Parameters
        ----------
        x : torch.Tensor
            Input of shape (batch, input_dim).
        time_steps : int, optional
            Number of simulation steps.

        Returns
        -------
        torch.Tensor
            Output spike counts of shape (batch, output_dim).
        """
        batch_size = x.size(0)
        mem1 = torch.zeros(batch_size, self.hidden_dim)
        mem2 = torch.zeros(batch_size, self.output_dim)
        out_spikes = []
        for _ in range(time_steps):
            cur1 = self.fc1(x)
            spk1, mem1 = self.lif1(cur1, mem1)
            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)
            out_spikes.append(spk2)
        return torch.stack(out_spikes).sum(dim=0)
