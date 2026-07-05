"""Small modular language model integrating SNNAI bio-inspired modules."""
import torch
import torch.nn as nn

from snnai.modules.language import CharTokenizer
from snnai.modules.language.context_retention import ContextRetention
from snnai.modules.language.next_token import NextTokenSNN
from snnai.modules.language.preprocess_pipeline import TextPreprocessPipeline
from snnai.modules.language.reasoning import ReasoningModule


class SmallLanguageModel(nn.Module):
    """Modular SNN language model.

    Pipeline:
        text -> tokenizer -> preprocessing (C. elegans + insect)
        -> context retention (crow + hippocampus)
        -> reasoning (mammal) -> next-token prediction

    Parameters
    ----------
    vocab : str
        Character vocabulary.
    feature_size : int, optional
        Feature dimension (default 32).
    hidden_size : int, optional
        Hidden size for next-token and reasoning (default 64).
    time_steps : int, optional
        Spike time steps (default 10).
    """

    def __init__(self, vocab, feature_size=32, hidden_size=64, time_steps=10):
        super().__init__()
        self.tokenizer = CharTokenizer(vocab)
        self.vocab_size = self.tokenizer.vocab_size
        self.feature_size = feature_size
        self.preprocess = TextPreprocessPipeline(
            vocab_size=self.vocab_size, time_steps=time_steps, feature_size=feature_size
        )
        self.context = ContextRetention(dim=feature_size, memory_slots=8, capacity=32)
        self.reasoning = ReasoningModule(
            input_dim=feature_size, hidden_dim=hidden_size, output_dim=hidden_size
        )
        self.reasoning_proj = nn.Linear(hidden_size, self.vocab_size, bias=False)
        self.next_token = NextTokenSNN(vocab_size=self.vocab_size, hidden_size=hidden_size)

    def forward(self, text):
        """Run the full modular pipeline.

        Parameters
        ----------
        text : str
            Input text.

        Returns
        -------
        torch.Tensor
            Output spike counts of shape (1, seq_len, vocab_size).
        """
        indices = torch.tensor([self.tokenizer.encode(text)])
        features, _ = self.preprocess(indices)
        wm_out, hp_out = self.context(features)
        # Combine working memory and hippocampal context
        combined = wm_out + hp_out
        reasoned = self.reasoning(combined, time_steps=features.size(0))
        bias = self.reasoning_proj(reasoned)
        # Use reasoned representation to bias next-token model
        spikes = self.preprocess.encoder(indices)
        next_token_out = self.next_token(spikes)
        return next_token_out + bias.unsqueeze(1)

    def generate(self, prompt, max_chars=10):
        """Generate text continuation.

        Parameters
        ----------
        prompt : str
            Prompt text.
        max_chars : int, optional
            Number of characters to generate.

        Returns
        -------
        str
            Generated text.
        """
        text = prompt
        for _ in range(max_chars):
            out = self.forward(text)
            next_idx = int(torch.argmax(out[0, -1, :]).item())
            text += self.tokenizer.idx_to_char[next_idx]
        return text
