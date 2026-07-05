"""Simple word-level tokenizer for SNNAI."""
import torch
import torch.nn as nn


class WordTokenizer:
    """Simple whitespace-based word tokenizer."""

    def __init__(self, texts):
        words = set()
        for t in texts:
            words.update(t.lower().split())
        self.vocab = ["<unk>", "<pad>"] + sorted(words)
        self.word_to_idx = {w: i for i, w in enumerate(self.vocab)}
        self.idx_to_word = {i: w for i, w in enumerate(self.vocab)}
        self.vocab_size = len(self.vocab)

    def encode(self, text):
        return [self.word_to_idx.get(w, 0) for w in text.lower().split()]

    def decode(self, indices):
        return " ".join(self.idx_to_word.get(i, "<unk>") for i in indices)


class WordSpikeEncoder(nn.Module):
    """Encode word indices into rate-coded spikes."""

    def __init__(self, vocab_size, time_steps=20):
        super().__init__()
        self.vocab_size = vocab_size
        self.time_steps = time_steps

    def forward(self, indices):
        """Encode word indices to spikes.

        Parameters
        ----------
        indices : torch.Tensor
            Shape (batch, seq_len).

        Returns
        -------
        torch.Tensor
            Spikes of shape (time_steps, batch, seq_len, vocab_size).
        """
        if indices.dim() == 1:
            indices = indices.unsqueeze(0)
        batch_size, seq_len = indices.shape
        base = torch.zeros(batch_size, seq_len, self.vocab_size)
        base.scatter_(2, indices.unsqueeze(2), 1.0)
        base = base.unsqueeze(0).repeat(self.time_steps, 1, 1, 1)
        noise = torch.rand_like(base)
        return (base > noise).float()
