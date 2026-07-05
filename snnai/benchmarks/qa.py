"""Minimal question-answering benchmark using SNNAI text modules."""
import torch

from snnai.modules.language import CharTokenizer, NextTokenSNN, SpikeEncoder


class SimpleQA:
    """Simple QA model that predicts the next character given question + context.

    The model encodes the concatenated question and context as spikes, runs
    a small next-token SNN, and returns the predicted answer token.
    """

    def __init__(self, vocab, hidden_size=32, time_steps=10):
        self.tokenizer = CharTokenizer(vocab)
        self.encoder = SpikeEncoder(vocab_size=self.tokenizer.vocab_size, time_steps=time_steps)
        self.model = NextTokenSNN(vocab_size=self.tokenizer.vocab_size, hidden_size=hidden_size)

    def answer(self, question, context):
        """Generate a short answer string.

        Parameters
        ----------
        question : str
            Question text.
        context : str
            Context text.

        Returns
        -------
        str
            Predicted answer character.
        """
        text = question + " " + context
        indices = torch.tensor([self.tokenizer.encode(text)])
        spikes = self.encoder(indices)
        next_idx = self.model.predict_next(spikes)
        return self.tokenizer.idx_to_char[next_idx]
