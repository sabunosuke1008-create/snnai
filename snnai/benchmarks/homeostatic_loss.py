"""Homeostatic regularization for SNN language models.

Encourages each LIF layer to maintain a target firing rate, preventing the
network from collapsing into inactive states (which often correspond to
newline/space tokens in character-level language modeling).
"""
import torch
import torch.nn as nn


class HomeostaticRegularizer(nn.Module):
    """Penalty on deviation of layer-wise mean firing rates from a target.

    Parameters
    ----------
    target_firing_rate : float
        Desired average firing rate (e.g. 0.12 for 12%).
    homeostatic_weight : float
        Scalar multiplier applied to the summed layer penalty.
    """

    def __init__(self, target_firing_rate=0.12, homeostatic_weight=1e-3):
        super().__init__()
        self.target_firing_rate = target_firing_rate
        self.homeostatic_weight = homeostatic_weight

    def forward(self, spike_tensors):
        """Compute regularization loss from a list of spike tensors.

        Parameters
        ----------
        spike_tensors : list[torch.Tensor]
            Each tensor contains binary spikes (0 or 1). Typical shapes are
            ``(time_steps, batch, seq_len, hidden)`` or ``(batch, hidden)``.

        Returns
        -------
        torch.Tensor
            Scalar regularization loss.
        """
        if not spike_tensors:
            return torch.tensor(0.0, device=next(iter(spike_tensors), torch.tensor(0.0)).device)

        total_loss = torch.tensor(0.0, device=spike_tensors[0].device)
        for spk in spike_tensors:
            # Binary spikes: mean over all dimensions gives the firing rate.
            firing_rate = spk.float().mean()
            total_loss = total_loss + (firing_rate - self.target_firing_rate) ** 2

        return self.homeostatic_weight * total_loss
