"""Minimal dialogue response system using SNNAI."""
import torch

from snnai.modules.language import CharTokenizer, SpikeEncoder
from snnai.modules.language.next_token import NextTokenSNN


class SimpleDialogue:
    """Simple dialogue model that appends a response to user input.

    The model encodes the conversation history and generates a short reply.
    """

    def __init__(self, vocab, hidden_size=32, time_steps=10, max_response=10):
        self.tokenizer = CharTokenizer(vocab)
        self.encoder = SpikeEncoder(vocab_size=self.tokenizer.vocab_size, time_steps=time_steps)
        self.model = NextTokenSNN(vocab_size=self.tokenizer.vocab_size, hidden_size=hidden_size)
        self.max_response = max_response

    def respond(self, user_input, history=""):
        """Generate a response.

        Parameters
        ----------
        user_input : str
            User message.
        history : str, optional
            Previous conversation history.

        Returns
        -------
        str
            Generated response.
        """
        context = (history + " " + user_input).strip()
        text = context
        for _ in range(self.max_response):
            indices = torch.tensor([self.tokenizer.encode(text[-30:])])
            spikes = self.encoder(indices)
            out = self.model(spikes)
            next_idx = int(torch.argmax(out[0, -1, :]).item())
            text += self.tokenizer.idx_to_char[next_idx]
        return text[len(context):].strip()
