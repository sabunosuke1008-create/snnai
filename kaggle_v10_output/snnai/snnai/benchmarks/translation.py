"""Minimal character-level translation concept using SNNAI text modules."""
import torch
import torch.nn as nn

from snnai.modules.language import CharTokenizer, SpikeEncoder
from snnai.modules.language.next_token import NextTokenSNN


class SimpleTranslator:
    """Character-level translator using a shared SNN next-token model.

    Maps characters from a source vocabulary to a target vocabulary one
    character at a time. The target vocabulary is assumed to be the same
    set of characters for simplicity.
    """

    def __init__(self, vocab, hidden_size=32, time_steps=10):
        self.tokenizer = CharTokenizer(vocab)
        self.encoder = SpikeEncoder(vocab_size=self.tokenizer.vocab_size, time_steps=time_steps)
        self.model = NextTokenSNN(vocab_size=self.tokenizer.vocab_size, hidden_size=hidden_size)
        # Learnable output mapping simulates translation decoding
        self.output_map = nn.Linear(self.tokenizer.vocab_size, self.tokenizer.vocab_size, bias=False)
        with torch.no_grad():
            self.output_map.weight.copy_(torch.eye(self.tokenizer.vocab_size))

    def translate(self, text):
        """Translate text character by character.

        Parameters
        ----------
        text : str
            Source text.

        Returns
        -------
        str
            Translated text.
        """
        indices = torch.tensor([self.tokenizer.encode(text)])
        spikes = self.encoder(indices)
        out = self.model(spikes)
        mapped = self.output_map(out[:, -1, :])
        next_idx = int(torch.argmax(mapped, dim=1).item())
        return text + self.tokenizer.idx_to_char[next_idx]
