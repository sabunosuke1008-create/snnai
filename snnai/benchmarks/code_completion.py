"""Minimal code completion concept using SNNAI text modules."""
import torch

from snnai.modules.language import CharTokenizer, SpikeEncoder
from snnai.modules.language.next_token import NextTokenSNN


class SimpleCodeCompleter:
    """Predict the next character in a code snippet.

    Uses the same next-token SNN trained on code-like token sequences.
    """

    def __init__(self, vocab, hidden_size=32, time_steps=10):
        self.tokenizer = CharTokenizer(vocab)
        self.encoder = SpikeEncoder(vocab_size=self.tokenizer.vocab_size, time_steps=time_steps)
        self.model = NextTokenSNN(vocab_size=self.tokenizer.vocab_size, hidden_size=hidden_size)

    def complete(self, snippet, max_chars=5):
        """Complete a code snippet character by character.

        Parameters
        ----------
        snippet : str
            Partial code snippet.
        max_chars : int, optional
            Number of characters to generate.

        Returns
        -------
        str
            Completed snippet.
        """
        text = snippet
        for _ in range(max_chars):
            indices = torch.tensor([self.tokenizer.encode(text[-20:])])
            if indices.size(1) == 0:
                break
            spikes = self.encoder(indices)
            next_idx = self.model.predict_next(spikes)
            text += self.tokenizer.idx_to_char[next_idx]
        return text
