"""Word-level next-token prediction model."""
import torch
import torch.nn as nn

from snnai.modules.language.next_token import NextTokenSNN
from snnai.modules.language.word_tokenizer import WordSpikeEncoder, WordTokenizer


class WordLMTrainer:
    """Train a word-level next-token SNN."""

    def __init__(self, texts, hidden_size=64, time_steps=10, seq_len=8):
        self.tokenizer = WordTokenizer(texts)
        self.encoder = WordSpikeEncoder(vocab_size=self.tokenizer.vocab_size, time_steps=time_steps)
        self.model = NextTokenSNN(vocab_size=self.tokenizer.vocab_size, hidden_size=hidden_size)
        self.seq_len = seq_len
        self.data = []
        for t in texts:
            self.data.extend(self.tokenizer.encode(t))
        self.data = torch.tensor([self.data])

    def _make_batch(self, batch_size):
        max_start = self.data.size(1) - self.seq_len - 1
        if max_start <= 0:
            max_start = 1
        starts = torch.randint(0, max_start, (batch_size,))
        inputs = torch.stack([self.data[0, s:s + self.seq_len] for s in starts])
        targets = torch.stack([self.data[0, s + 1:s + self.seq_len + 1] for s in starts])
        return inputs, targets

    def train(self, epochs=10, batch_size=4, lr=1e-3):
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

    def generate(self, prompt, max_words=3):
        self.model.eval()
        words = self.tokenizer.encode(prompt)
        for _ in range(max_words):
            seq = words[-self.seq_len:]
            if len(seq) == 0:
                break
            indices = torch.tensor([seq])
            spikes = self.encoder(indices)
            out = self.model(spikes)
            next_idx = int(torch.argmax(out[0, -1, :]).item())
            words.append(next_idx)
        return self.tokenizer.decode(words)
