"""Practical text generation release API for SNNAI."""
import torch

from snnai.benchmarks.generation_metrics import _apply_repetition_penalty, _apply_token_bias, _sample_next_token
from snnai.modules.language import CharTokenizer, SpikeEncoder
from snnai.modules.language.large_lm import LargeNextTokenSNN


class TextGenerator:
    """High-level text generation API.

    Parameters
    ----------
    vocab : str
        Character vocabulary.
    embed_dim : int, optional
        Embedding dimension.
    hidden_dim : int, optional
        Hidden dimension.
    num_layers : int, optional
        Number of layers.
    time_steps : int, optional
        Spike time steps.
    """

    def __init__(self, vocab, embed_dim=64, hidden_dim=256, num_layers=2, time_steps=10):
        self.tokenizer = CharTokenizer(vocab)
        self.encoder = SpikeEncoder(vocab_size=self.tokenizer.vocab_size, time_steps=time_steps)
        self.model = LargeNextTokenSNN(vocab_size=self.tokenizer.vocab_size, embed_dim=embed_dim,
                                       hidden_dim=hidden_dim, num_layers=num_layers)

    def train(self, corpus, epochs=10, batch_size=8, lr=1e-3, seq_len=20):
        """Train on a corpus."""
        import torch.nn as nn
        data = torch.tensor([self.tokenizer.encode(corpus)])
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        self.model.train()
        max_start = max(1, data.size(1) - seq_len - 1)
        history = []
        for _ in range(epochs):
            starts = torch.randint(0, max_start, (batch_size,))
            inputs = torch.stack([data[0, s:s + seq_len] for s in starts])
            targets = torch.stack([data[0, s + 1:s + seq_len + 1] for s in starts])
            spikes = self.encoder(inputs)
            out = self.model(spikes)
            loss = criterion(out.reshape(-1, self.tokenizer.vocab_size), targets.reshape(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            history.append(loss.item())
        return history

    def generate(self, prompt, max_chars=50, temperature=1.0,
                 repetition_penalty=1.0, penalty_window=16,
                 top_p=None, generation_bias_tokens=None,
                 generation_bias_weight=0.0):
        """Generate text from a prompt."""
        self.model.eval()
        text = prompt
        generated_ids = []
        with torch.no_grad():
            for _ in range(max_chars):
                indices = torch.tensor([self.tokenizer.encode(text[-50:])])
                spikes = self.encoder(indices)
                out = self.model(spikes)
                logits = out[0, -1, :]
                logits = _apply_repetition_penalty(
                    logits,
                    recent_ids=generated_ids[-penalty_window:],
                    repetition_penalty=repetition_penalty,
                )
                logits = _apply_token_bias(
                    logits,
                    bias_token_ids=generation_bias_tokens,
                    bias_weight=generation_bias_weight,
                )
                next_idx = int(_sample_next_token(
                    logits,
                    temperature=temperature,
                    top_p=top_p,
                ).item())
                generated_ids.append(next_idx)
                text += self.tokenizer.idx_to_char[next_idx]
        return text
