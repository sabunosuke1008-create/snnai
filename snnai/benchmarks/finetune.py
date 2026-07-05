"""Fine-tuning base for SNNAI text models."""
import torch
import torch.nn as nn

from snnai.modules.language import CharTokenizer, SpikeEncoder


class FineTuner:
    """Fine-tune a pretrained next-token SNN on a downstream corpus.

    Parameters
    ----------
    model : nn.Module
        Pretrained SNN model.
    tokenizer : CharTokenizer
        Tokenizer.
    encoder : SpikeEncoder
        Spike encoder.
    """

    def __init__(self, model, tokenizer, encoder):
        self.model = model
        self.tokenizer = tokenizer
        self.encoder = encoder

    def finetune(self, corpus, epochs=5, batch_size=4, lr=1e-4, seq_len=20):
        """Fine-tune the model.

        Returns
        -------
        list
            Loss history.
        """
        data = torch.tensor([self.tokenizer.encode(corpus)])
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        history = []
        self.model.train()
        max_start = max(1, data.size(1) - seq_len - 1)
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
