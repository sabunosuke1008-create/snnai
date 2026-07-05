"""Trainer for large-corpus SNN language modeling experiments."""
import torch


class LargeCorpusTrainer:
    """Concept trainer that wraps a dataset loader and trains in epochs.

    Full-scale runs use the Kaggle notebook; this local version validates
    shapes and runs a few steps on synthetic data.
    """

    def __init__(self, model, optimizer, tokenizer, device="cpu"):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.tokenizer = tokenizer
        self.device = device
        self.loss_history = []

    def _batch_to_tensor(self, texts, seq_len=16):
        indices = []
        for text in texts:
            encoded = self.tokenizer.encode(text[:seq_len].ljust(seq_len))
            indices.append(encoded)
        tensor = torch.tensor(indices, dtype=torch.long)
        one_hot = torch.zeros(tensor.size(0), tensor.size(1), self.tokenizer.vocab_size)
        one_hot.scatter_(2, tensor.unsqueeze(2), 1.0)
        return one_hot.to(self.device)

    def train_step(self, batch_texts, time_steps=10, seq_len=16):
        """Run one training step and return loss."""
        self.model.train()
        self.optimizer.zero_grad()
        x = self._batch_to_tensor(batch_texts, seq_len=seq_len)
        x = x.unsqueeze(0).repeat(time_steps, 1, 1, 1)
        out = self.model(x)
        # Simple target: last token prediction
        target = x[-1, :, -1, :].argmax(dim=-1)
        loss = torch.nn.functional.cross_entropy(out[:, -1, :], target)
        loss.backward()
        self.optimizer.step()
        self.loss_history.append(float(loss.item()))
        return float(loss.item())
