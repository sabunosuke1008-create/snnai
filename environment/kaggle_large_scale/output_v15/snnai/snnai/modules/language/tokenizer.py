"""Character-level tokenizer and spike encoder for text."""
import torch
import torch.nn as nn


class CharTokenizer:
    """Simple character-level tokenizer."""

    def __init__(self, vocab):
        self.vocab = list(sorted(set(vocab)))
        self.char_to_idx = {c: i for i, c in enumerate(self.vocab)}
        self.idx_to_char = {i: c for i, c in enumerate(self.vocab)}
        self.vocab_size = len(self.vocab)

    def encode(self, text):
        """Convert text to list of token indices."""
        return [self.char_to_idx.get(c, 0) for c in text]

    def decode(self, indices):
        """Convert token indices to text."""
        return "".join(self.idx_to_char.get(i, " ") for i in indices)

    def __len__(self):
        return self.vocab_size


class SpikeEncoder(nn.Module):
    """Encode token indices into time-step spike tensors.

    Each token is represented by a one-hot vector over time steps using
    rate coding: the active neuron fires at a higher rate.

    Parameters
    ----------
    vocab_size : int
        Vocabulary size.
    time_steps : int, optional
        Number of time steps (default 20).
    """

    def __init__(self, vocab_size, time_steps=20):
        super().__init__()
        self.vocab_size = vocab_size
        self.time_steps = time_steps

    def forward(self, indices):
        """Encode token indices to spikes.

        Parameters
        ----------
        indices : torch.Tensor
            Token indices of shape (batch, seq_len) or (seq_len,).

        Returns
        -------
        torch.Tensor
            Spike tensor of shape (time_steps, batch, seq_len, vocab_size).
        """
        if indices.dim() == 1:
            indices = indices.unsqueeze(0)
        batch_size, seq_len = indices.shape
        # One-hot base firing probability per token
        base = torch.zeros(batch_size, seq_len, self.vocab_size, device=indices.device)
        base.scatter_(2, indices.unsqueeze(2), 1.0)
        # Repeat over time and add noise to make it spike-like
        base = base.unsqueeze(0).repeat(self.time_steps, 1, 1, 1)
        noise = torch.rand_like(base)
        spikes = (base > noise).float()
        return spikes
