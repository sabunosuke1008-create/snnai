"""SNNAI postprocessor for LLM output."""
import torch

from snnai.modules.language.reasoning import ReasoningModule


class LLMPostProcessor:
    """Lightweight SNN postprocessing for LLM outputs.

    Takes an LLM-generated draft, runs a small reasoning layer to shape
    the response, and returns a polished answer.
    """

    def __init__(self, vocab_size, feature_size=16, hidden_size=32):
        self.vocab_size = vocab_size
        self.reasoning = ReasoningModule(input_dim=vocab_size, hidden_dim=hidden_size, output_dim=vocab_size)

    def process(self, draft_text, tokenizer):
        """Process an LLM draft and return a shaped response.

        Parameters
        ----------
        draft_text : str
            Raw LLM output.
        tokenizer : CharTokenizer
            Tokenizer.

        Returns
        -------
        str
            Post-processed text.
        """
        indices = torch.tensor([tokenizer.encode(draft_text[-30:])])
        # One-hot input for reasoning
        one_hot = torch.zeros(1, indices.size(1), tokenizer.vocab_size)
        one_hot.scatter_(2, indices.unsqueeze(2), 1.0)
        context = one_hot.mean(dim=1)  # (1, vocab_size)
        out = self.reasoning(context, time_steps=5)
        next_idx = int(torch.argmax(out[0, :tokenizer.vocab_size]).item())
        return draft_text + tokenizer.idx_to_char[next_idx]
