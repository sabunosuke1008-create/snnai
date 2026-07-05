"""Benchmark utilities comparing SNN LM and Transformer baselines."""
import time

import torch


class TransformerBaseline(torch.nn.Module):
    """Minimal Transformer decoder layer baseline for comparison."""

    def __init__(self, vocab_size, d_model=64, nhead=2, num_layers=2, dim_feedforward=128):
        super().__init__()
        self.embedding = torch.nn.Embedding(vocab_size, d_model)
        encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, batch_first=True
        )
        self.transformer = torch.nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = torch.nn.Linear(d_model, vocab_size)

    def forward(self, x):
        """x: (batch, seq_len) long tensor."""
        emb = self.embedding(x)
        out = self.transformer(emb)
        return self.fc(out)


def compare_models(snn_model, transformer_model, sample_input, tokenizer=None):
    """Compare SNN and Transformer on latency and parameter count."""
    snn_model.eval()
    transformer_model.eval()
    device = next(snn_model.parameters()).device

    # SNN latency
    start = time.perf_counter()
    with torch.no_grad():
        _ = snn_model(sample_input)
    snn_latency = time.perf_counter() - start

    # Transformer latency
    # sample_input for SNN is (time, batch, seq, vocab); derive token ids for transformer
    transformer_input = sample_input[-1].argmax(dim=-1).to(device)
    start = time.perf_counter()
    with torch.no_grad():
        _ = transformer_model(transformer_input)
    transformer_latency = time.perf_counter() - start

    snn_params = sum(p.numel() for p in snn_model.parameters())
    transformer_params = sum(p.numel() for p in transformer_model.parameters())
    return {
        "snn_latency": snn_latency,
        "transformer_latency": transformer_latency,
        "snn_parameters": snn_params,
        "transformer_parameters": transformer_params,
    }
