import torch
from snnai.modules.language.large_scale_lm import LargeScaleSNNLM
from snnai.benchmarks.transformer_comparison import TransformerBaseline, compare_models


def test_transformer_comparison():
    vocab_size = 5
    snn_model = LargeScaleSNNLM(vocab_size=vocab_size, embed_dim=8, hidden_dim=16, num_layers=1)
    transformer_model = TransformerBaseline(vocab_size=vocab_size, d_model=8, nhead=1, num_layers=1, dim_feedforward=16)
    sample = torch.zeros(3, 1, 2, vocab_size)
    result = compare_models(snn_model, transformer_model, sample)
    assert "snn_latency" in result
    assert "transformer_latency" in result
