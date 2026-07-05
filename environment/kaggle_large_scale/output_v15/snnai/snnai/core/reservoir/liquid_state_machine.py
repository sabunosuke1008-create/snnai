"""Liquid State Machine (LSM) reservoir for SNNAI core."""
import torch
import torch.nn as nn
import snntorch as snn


class LiquidStateMachine(nn.Module):
    """Spiking liquid state machine built on snntorch LIF neurons.

    The reservoir consists of a recurrent population of leaky
    integrate-and-fire neurons driven by input weights and sparse random
    recurrent weights scaled to a target spectral radius.
    """

    def __init__(
        self,
        input_size,
        reservoir_size,
        output_size=None,
        beta=0.9,
        threshold=1.0,
        sparsity=0.1,
        spectral_radius=0.9,
    ):
        super().__init__()
        self.input_size = input_size
        self.reservoir_size = reservoir_size
        self.output_size = output_size
        self.lif = snn.Leaky(
            beta=beta, threshold=threshold, learn_threshold=False, learn_beta=False
        )
        self.W_in = nn.Linear(input_size, reservoir_size, bias=False)
        self.W_rec = nn.Linear(reservoir_size, reservoir_size, bias=False)

        # Initialize random sparse recurrent weights and scale by spectral radius.
        with torch.no_grad():
            w = torch.randn(reservoir_size, reservoir_size)
            mask = torch.rand(reservoir_size, reservoir_size) < sparsity
            w = w * mask.float()
            w.fill_diagonal_(0)
            eigvals = torch.linalg.eigvals(w)
            sr = torch.max(torch.abs(eigvals)).item()
            if sr > 0:
                w = w * (spectral_radius / sr)
            self.W_rec.weight.copy_(w)
            self.W_in.weight.normal_(0, 0.5)

    def forward(self, x):
        """Run the LSM reservoir on an input sequence.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (time, batch, input_size).

        Returns
        -------
        torch.Tensor
            Reservoir spike tensor of shape (time, batch, reservoir_size).
        """
        batch_size = x.shape[1]
        mem = torch.zeros(batch_size, self.reservoir_size)
        spikes = []
        for t in range(x.shape[0]):
            current = self.W_in(x[t]) + self.W_rec(mem)
            spk, mem = self.lif(current, mem)
            spikes.append(spk)
        return torch.stack(spikes)
