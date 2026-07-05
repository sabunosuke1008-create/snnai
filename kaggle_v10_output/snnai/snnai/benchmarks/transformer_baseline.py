"""Minimal Transformer baseline for comparing with SNNAI SNN LM."""
import torch
import torch.nn as nn

from snnai.modules.language import CharTokenizer


class TransformerLM(nn.Module):
    """Tiny Transformer language model."""

    def __init__(self, vocab_size, d_model=64, nhead=2, num_layers=2, dim_feedforward=128):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        """x shape: (batch, seq_len). Returns logits (batch, seq_len, vocab_size)."""
        emb = self.embedding(x)
        out = self.transformer(emb)
        return self.fc(out)


def compare_models(corpus, snn_model, transformer_model, tokenizer, seq_len=10):
    """Compare perplexity-like loss between SNN and Transformer.

    Returns
    -------
    dict
        Loss values for both models.
    """
    data = torch.tensor([tokenizer.encode(corpus)])
    criterion = nn.CrossEntropyLoss()
    inputs = data[:, :seq_len]
    targets = data[:, 1:seq_len + 1]

    # SNN forward expects spikes
    from snnai.modules.language import SpikeEncoder
    encoder = SpikeEncoder(vocab_size=tokenizer.vocab_size, time_steps=10)
    spikes = encoder(inputs)
    snn_out = snn_model(spikes)
    snn_loss = criterion(snn_out.reshape(-1, tokenizer.vocab_size), targets.reshape(-1))

    transformer_out = transformer_model(inputs)
    transformer_loss = criterion(transformer_out.reshape(-1, tokenizer.vocab_size), targets.reshape(-1))

    return {
        "snn_loss": snn_loss.item(),
        "transformer_loss": transformer_loss.item(),
    }
