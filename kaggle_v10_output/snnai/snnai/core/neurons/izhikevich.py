"""Izhikevich neuron model for SNNAI core."""
import torch
import torch.nn as nn


PRESETS = {
    "RS": {"a": 0.02, "b": 0.2, "c": -65.0, "d": 8.0},
    "IB": {"a": 0.02, "b": 0.2, "c": -55.0, "d": 4.0},
    "CH": {"a": 0.02, "b": 0.2, "c": -50.0, "d": 2.0},
    "FS": {"a": 0.1, "b": 0.2, "c": -65.0, "d": 2.0},
}


class Izhikevich(nn.Module):
    """Izhikevich spiking neuron model.

    Parameters
    ----------
    a, b, c, d : float
        Model parameters controlling recovery and reset dynamics.
    threshold : float
        Spike threshold (mV).
    """

    def __init__(self, a=0.02, b=0.2, c=-65.0, d=8.0, threshold=30.0):
        super().__init__()
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.threshold = threshold

    def init_state(self, batch_size=1):
        """Return initial membrane potential v and recovery variable u."""
        v = torch.full((batch_size,), -65.0)
        u = torch.full((batch_size,), self.b * -65.0)
        return v, u

    def forward(self, I, v, u, dt=1.0):
        """Single Euler step.

        Parameters
        ----------
        I : torch.Tensor
            Input current.
        v, u : torch.Tensor
            Membrane potential and recovery variable.
        dt : float
            Time step.

        Returns
        -------
        spike, v, u
        """
        v = v + dt * (0.04 * v**2 + 5.0 * v + 140.0 - u + I)
        u = u + dt * self.a * (self.b * v - u)
        spike = (v >= self.threshold).float()
        v = torch.where(spike.bool(), torch.full_like(v, self.c), v)
        u = torch.where(spike.bool(), u + self.d, u)
        return spike, v, u


def create_izhikevich(preset="RS"):
    """Create an Izhikevich neuron from a named preset."""
    if preset not in PRESETS:
        raise ValueError(f"Unknown preset {preset}. Choose from {list(PRESETS.keys())}")
    return Izhikevich(**PRESETS[preset])
