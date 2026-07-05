"""Context retention combining crow-like working memory and hippocampus."""
import torch
import torch.nn as nn
import snntorch as snn

from snnai.modules.hippocampus.associative_memory import HippocampalMemory


class SimpleWorkingMemory(nn.Module):
    """Generic slot-based working memory.

    Parameters
    ----------
    dim : int
        Feature dimension.
    slots : int, optional
        Number of memory slots (default 8).
    """

    def __init__(self, dim, slots=8):
        super().__init__()
        self.dim = dim
        self.slots = slots
        self.register_buffer("memory", torch.zeros(slots, dim))
        self.register_buffer("usage", torch.zeros(slots))
        self.lif = snn.Leaky(beta=0.9, threshold=1.0, learn_threshold=False)

    def forward(self, x):
        """Store and read working memory.

        Parameters
        ----------
        x : torch.Tensor
            Input of shape (batch, dim).

        Returns
        -------
        torch.Tensor
            Readout of shape (batch, dim).
        """
        # Use oldest slot for new input
        idx = int(torch.argmin(self.usage).item())
        mem = self.memory[idx]
        spk, _ = self.lif(x, mem.unsqueeze(0).expand(x.size(0), -1))
        self.memory[idx] = spk.mean(dim=0)
        self.usage[idx] = self.usage.max() + 1
        # Read aggregated memory
        return self.memory.mean(dim=0).unsqueeze(0).expand(x.size(0), -1)


class ContextRetention(nn.Module):
    """Combine short-term (crow-like) and long-term (hippocampus) memory.

    Parameters
    ----------
    dim : int
        Feature dimension.
    memory_slots : int, optional
        Working memory capacity (default 8).
    capacity : int, optional
        Hippocampal memory capacity (default 64).
    """

    def __init__(self, dim, memory_slots=8, capacity=64):
        super().__init__()
        self.dim = dim
        self.working_memory = SimpleWorkingMemory(dim=dim, slots=memory_slots)
        self.hippocampus = HippocampalMemory(dim=dim, capacity=capacity)

    def forward(self, features):
        """Process a sequence of features and retain context.

        Parameters
        ----------
        features : torch.Tensor
            Input features of shape (time, batch, seq_len, dim).

        Returns
        -------
        torch.Tensor
            Working memory readout of shape (batch, dim).
        torch.Tensor
            Hippocampal retrieval of shape (batch, dim).
        """
        # Average over time and seq_len for a compact representation
        context = features.mean(dim=(0, 2))  # (batch, dim)
        # Update working memory
        wm_out = self.working_memory(context)
        # Store and retrieve from hippocampus
        self.hippocampus.store(context, context)
        hp_out = self.hippocampus(context)
        return wm_out, hp_out
