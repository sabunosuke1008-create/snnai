"""Small-corpus next-character prediction trainer for SNNAI."""
import torch
import torch.nn as nn

from snnai.modules.language import CharTokenizer, SpikeEncoder
from snnai.modules.language.next_token import NextTokenSNN


class CharLMTrainer:
    """Train a next-character SNN on a tiny corpus.

    Parameters
    ----------
    corpus : str
        Training text.
    hidden_size : int, optional
        Hidden size (default 64).
    time_steps : int, optional
        Spike time steps (default 20).
    seq_len : int, optional
        Sequence length (default 20).
    """

    def __init__(self, corpus, hidden_size=64, time_steps=20, seq_len=20):
        self.corpus = corpus
        self.tokenizer = CharTokenizer(corpus)
        self.encoder = SpikeEncoder(vocab_size=self.tokenizer.vocab_size, time_steps=time_steps)
        self.model = NextTokenSNN(vocab_size=self.tokenizer.vocab_size, hidden_size=hidden_size)
        self.seq_len = seq_len
        self.data = torch.tensor([self.tokenizer.encode(corpus)])

    def _make_batch(self, batch_size):
        """Sample a batch of sequences from the corpus."""
        max_start = self.data.size(1) - self.seq_len - 1
        starts = torch.randint(0, max_start, (batch_size,))
        inputs = torch.stack([self.data[0, s:s + self.seq_len] for s in starts])
        targets = torch.stack([self.data[0, s + 1:s + self.seq_len + 1] for s in starts])
        return inputs, targets

    def train(self, epochs=10, batch_size=8, lr=1e-3):
        """Train the model.

        Returns
        -------
        list
            Loss history.
        """
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        history = []
        self.model.train()
        for _ in range(epochs):
            inputs, targets = self._make_batch(batch_size)
            spikes = self.encoder(inputs)
            out = self.model(spikes)
            loss = criterion(out.reshape(-1, self.tokenizer.vocab_size), targets.reshape(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            history.append(loss.item())
        return history

    def generate(self, prompt, max_chars=20):
        """Generate continuation from a prompt."""
        self.model.eval()
        text = prompt
        for _ in range(max_chars):
            indices = torch.tensor([self.tokenizer.encode(text[-self.seq_len:])])
            spikes = self.encoder(indices)
            out = self.model(spikes)
            next_idx = int(torch.argmax(out[0, -1, :]).item())
            text += self.tokenizer.idx_to_char[next_idx]
        return text
