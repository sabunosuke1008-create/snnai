"""Hippocampus gate layer for LargeScaleSNNLM (Phase 6.6).

Integrates the existing `HippocampalMemory` from
`snnai.modules.hippocampus.associative_memory` into the SNN language
model as an optional gated memory pathway. The gate learns to blend
retrieved episodic context with the current hidden state.

Usage::

    from snnai.modules.language.hippocampus_gate import HippocampusGate
    gate = HippocampusGate(hidden_dim=512, capacity=128)
    out = gate(hidden)          # (B, L, H) gated output
    gate.reset_memory()         # clear stored episodes between runs
"""
import torch
import torch.nn as nn

from snnai.modules.hippocampus.associative_memory import HippocampalMemory


class HippocampusGate(nn.Module):
    """Gated integration of hippocampal memory into a hidden state.

    For each position in the sequence:
    1. Use the hidden state as a query to retrieve from HippocampalMemory.
    2. Gate the retrieved memory into the output with a learned sigmoid gate.

    The memory stores mean-pooled hidden states from previous forward
    passes as episodes, enabling retrieval of relevant context.

    Parameters
    ----------
    hidden_dim : int
        Dimension of the hidden state.
    capacity : int, optional
        Maximum number of stored episodes (default 64).
    """

    def __init__(self, hidden_dim, capacity=64):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.memory = HippocampalMemory(hidden_dim, capacity=capacity)
        self.gate_proj = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(self, x, store=True):
        """Apply hippocampus gate.

        Parameters
        ----------
        x : Tensor (B, L, H)
            Input hidden state.
        store : bool
            Whether to store x in memory for future retrieval.

        Returns
        -------
        out : Tensor (B, L, H)
            Gated output blending retrieved memory with current state.
        """
        B, L, H = x.shape

        if store:
            episode_key = x.mean(dim=1).detach()
            episode_value = x.mean(dim=1).detach()
            self.memory.store(episode_key, episode_value)

        flat_x = x.reshape(-1, H)
        if self.memory.count.item() == 0:
            retrieved = torch.zeros_like(flat_x)
        else:
            retrieved, _ = self.memory.retrieve(flat_x, top_k=1)
            retrieved = retrieved.squeeze(1)

        combined = torch.cat([flat_x, retrieved], dim=-1)
        g = torch.sigmoid(self.gate_proj(combined))
        out = g * retrieved + (1.0 - g) * flat_x
        return out.reshape(B, L, H)

    def reset_memory(self):
        """Clear all stored episodes."""
        self.memory.keys.zero_()
        self.memory.values.zero_()
        self.memory.count.zero_()
