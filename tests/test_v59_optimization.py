import torch
from snnai.modules.language.large_scale_lm import LargeScaleSNNLM
from snnai.benchmarks.optimization import quantize_weights, prune_weights


def test_quantize_weights():
    model = LargeScaleSNNLM(vocab_size=5, embed_dim=4, hidden_dim=8, num_layers=1)
    scales = quantize_weights(model, bits=8)
    assert len(scales) > 0


def test_prune_weights():
    model = LargeScaleSNNLM(vocab_size=5, embed_dim=4, hidden_dim=8, num_layers=1)
    info = prune_weights(model, threshold=0.01)
    assert 0 <= info["sparsity"] <= 1
