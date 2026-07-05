import torch
from snnai.benchmarks.energy_quantification import quantize_energy
from snnai.modules.language.large_lm import LargeNextTokenSNN


def test_quantize_energy():
    vocab = "abc"
    model = LargeNextTokenSNN(vocab_size=3, embed_dim=4, hidden_dim=8, num_layers=1, dropout=0.0)
    sample = torch.zeros(5, 1, 1, 3)  # (time_steps, batch, seq_len, vocab_size)
    report = quantize_energy(model, sample, time_steps=5)
    assert "joules" in report
    assert "latency_seconds" in report
    assert "total_spikes" in report
    assert "layer_spikes" in report
    assert report["total_spikes"] >= 0
    assert report["spikes_per_step"] >= 0
