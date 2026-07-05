"""Encoding bridges between rate and temporal spike codes."""
import torch


class EncodingBridge:
    """Convert between rate-coded vectors and spike trains.

    Modules in SNNAI may use different coding schemes. This bridge lets
    them talk to each other by translating representations.

    Parameters
    ----------
    in_dim : int
        Input feature dimension.
    out_dim : int
        Output feature dimension.
    time_steps : int
        Number of time steps in generated spike trains.
    """

    def __init__(self, in_dim, out_dim, time_steps=20):
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.time_steps = time_steps
        self.weight = torch.randn(out_dim, in_dim) * 0.3

    def spikes_to_rates(self, spikes):
        """Convert a spike tensor to a rate vector.

        Parameters
        ----------
        spikes : torch.Tensor
            Tensor of shape (time, batch, neurons) or (time, neurons).

        Returns
        -------
        torch.Tensor
            Rate vector of shape (batch, neurons) or (neurons,).
        """
        return spikes.float().mean(dim=0)

    def rates_to_spikes(self, rates):
        """Convert a rate vector to a Poisson spike train.

        Parameters
        ----------
        rates : torch.Tensor
            Rate vector of shape (batch, neurons) or (neurons,).

        Returns
        -------
        torch.Tensor
            Spike tensor of shape (time, batch, neurons) or (time, neurons).
        """
        if rates.dim() == 1:
            rates = rates.unsqueeze(0)
            squeeze = True
        else:
            squeeze = False
        # Project rates to out_dim
        projected = torch.matmul(rates, self.weight.T)
        probs = torch.sigmoid(projected)
        # Repeat across time and sample Bernoulli
        probs_t = probs.unsqueeze(0).repeat(self.time_steps, 1, 1)
        spikes = torch.bernoulli(probs_t)
        if squeeze:
            spikes = spikes.squeeze(1)
        return spikes

    def project_rates(self, rates):
        """Project a rate vector to the output dimension."""
        return torch.matmul(rates, self.weight.T)
