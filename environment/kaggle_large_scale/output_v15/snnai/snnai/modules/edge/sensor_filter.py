"""Event-based sensor filter for low-power edge processing."""
import torch
import torch.nn as nn
import snntorch as snn


class SensorFilter(nn.Module):
    """Filter noisy sensor streams using a leaky integrate-and-fire layer.

    The module suppresses weak or transient sensor readings and only
    forwards significant events as spikes.

    Parameters
    ----------
    input_size : int
        Number of sensor channels.
    beta : float, optional
        Membrane time constant (default 0.9).
    threshold : float, optional
        Firing threshold (default 1.0).
    """

    def __init__(self, input_size, beta=0.9, threshold=1.0):
        super().__init__()
        self.input_size = input_size
        self.lif = snn.Leaky(beta=beta, threshold=threshold, learn_threshold=False)

    def forward(self, x):
        """Filter sensor input.

        Parameters
        ----------
        x : torch.Tensor
            Sensor readings of shape (time, batch, input_size).

        Returns
        -------
        torch.Tensor
            Filtered spikes of shape (time, batch, input_size).
        """
        time_steps, batch_size, _ = x.shape
        mem = torch.zeros(batch_size, self.input_size)
        spikes = []
        for t in range(time_steps):
            spk, mem = self.lif(x[t], mem)
            spikes.append(spk)
        return torch.stack(spikes)
