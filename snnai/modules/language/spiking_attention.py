"""Spiking Self-Attention (SSA) for SNNAI Phase 6.7.

Implements a softmax-free, spike-based self-attention following the
Spikformer / SpikeBERT line of work:

    scores = Q @ K^T / sqrt(d)
    attn   = S(normalize(scores))   # spike activation, no softmax
    out     = attn @ V
    out     = W_o(out)

where ``S`` is a self-contained spiking (IF) activation with a
sigmoid straight-through surrogate gradient. This gives Transformer-like
inter-dependence between sequence positions while staying inside the
SNN paradigm, directly addressing the structural weakness called out in
``docs/roadmap_v65.md`` (no positional inter-dependence in the
recurrent / feed-forward SNN LM).

Design notes
---------------
* The spiking activation is **stateless** (membrane-free IF spike) and
  holds **no ``mem`` attribute**, so it is unaffected by
  ``LargeScaleSNNLM._reset_lif_states()`` (which monkey-clears
  snntorch Leaky membrane states). This keeps SSA compatible with
  the existing LIF stack and DataParallel.
* Causal masking (lower-triangular) is supported for language modelling.
* Normalization is applied per-row over the key axis in a mask-aware,
  shape-safe way (mean/variance computed each forward), so it works
  for arbitrary, dynamic sequence lengths.
"""
import math

import torch
import torch.nn as nn


def _spike_surrogate(mem, threshold, slope=25.0):
    """Hard spike forward, sigmoid surrogate backward (straight-through).

    Forward returns ``1`` where ``mem >= threshold`` else ``0``; the
    backward gradient flows through a sigmoid of ``mem - threshold``.
    """
    thr = threshold.to(mem.dtype) if torch.is_tensor(threshold) else torch.tensor(
        threshold, dtype=mem.dtype, device=mem.device
    )
    # Straight-through estimator: forward returns the hard binary spike,
    # but the gradient flows through the surrogate sigmoid. We add the
    # soft surrogate and subtract its detached copy so the forward value
    # stays binary while the backward graph connects to ``mem``.
    spk = (mem >= thr).to(mem.dtype)
    gate = torch.sigmoid(slope * (mem - thr))
    return spk + (gate - gate.detach())


class _SpikeActivation(nn.Module):
    """Stateless IF spike with a sigmoid straight-through gradient."""

    def __init__(self, threshold=1.0, learn_threshold=False, slope=25.0):
        super().__init__()
        self.slope = slope
        if learn_threshold:
            self.threshold = nn.Parameter(torch.tensor(threshold, dtype=torch.float32))
        else:
            self.register_buffer("threshold", torch.tensor(threshold, dtype=torch.float32))

    def forward(self, mem):
        return _spike_surrogate(mem, self.threshold, self.slope)


class SpikingSelfAttention(nn.Module):
    """Softmax-free spiking self-attention over the sequence axis."""

    def __init__(self, hidden_dim, num_heads=1, dropout=0.0, causal=True,
                 use_spike=True, threshold=1.0, learn_threshold=False, slope=25.0):
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.causal = causal
        self.use_spike = use_spike

        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.drop = nn.Dropout(dropout)
        if use_spike:
            self.spike = _SpikeActivation(
                threshold=threshold, learn_threshold=learn_threshold, slope=slope
            )

    def reset(self):
        """No persistent state; provided for API symmetry."""
        return

    @staticmethod
    def _normalize(scores, mask):
        """Mask-aware per-row (key-axis) normalization to zero-mean/unit-var."""
        if mask is not None:
            valid = (~mask).unsqueeze(0).to(scores.dtype)  # (1, L, L)
            s2 = scores.masked_fill(mask, 0.0)
            denom = valid.sum(-1, keepdim=True).clamp_min(1.0)
            mean = (s2 * valid).sum(-1, keepdim=True) / denom
            var = (((s2 - mean) * valid) ** 2).sum(-1, keepdim=True) / denom
            s2 = (s2 - mean) / var.sqrt().clamp_min(1e-5)
            return s2.masked_fill(mask, -1e9)
        mean = scores.mean(-1, keepdim=True)
        std = scores.std(-1, keepdim=True).clamp_min(1e-5)
        return (scores - mean) / std

    def _attn_from_scores(self, scores, mask):
        """Turn raw scores into an attention map (spike or softmax)."""
        if self.use_spike:
            scores = self._normalize(scores, mask)
            return self.spike(scores)
        if mask is not None:
            scores = scores.masked_fill(mask, float("-inf"))
        return torch.softmax(scores, dim=-1)

    def forward(self, x, return_attn=False):
        """Forward pass over ``(B, L, H)`` hidden states.

        Returns ``(B, L, H)``. When ``return_attn`` is True also returns
        the ``(B, L, L)`` (single-head) or ``(B, heads, L, L)``
        attention map.
        """
        B, L, H = x.shape
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        if self.causal:
            mask = torch.triu(
                torch.ones(L, L, device=x.device), diagonal=1
            ).bool()
        else:
            mask = None

        if self.num_heads > 1:
            q = q.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
            k = k.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
            v = v.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
            # (B, heads, L, head_dim)
            scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_dim)
            Bh = B * self.num_heads
            scores = scores.reshape(Bh, L, L)
            attn = self._attn_from_scores(scores, mask)  # (Bh, L, L)
            attn = attn.reshape(B, self.num_heads, L, L)
            out = torch.matmul(attn, v)  # (B, heads, L, head_dim)
            out = out.transpose(1, 2).reshape(B, L, H)
        else:
            scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_dim)
            attn = self._attn_from_scores(scores, mask)  # (B, L, L)
            out = torch.matmul(attn, v)

        out = self.drop(out)
        out = self.out_proj(out)
        if return_attn:
            return out, attn
        return out