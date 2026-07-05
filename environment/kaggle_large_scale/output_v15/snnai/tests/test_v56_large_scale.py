from snnai.modules.language.large_scale_lm import LargeScaleSNNLM, count_parameters


def test_large_scale_lm():
    model = LargeScaleSNNLM(vocab_size=5, embed_dim=8, hidden_dim=16, num_layers=2, dropout=0.0)
    import torch
    x = torch.zeros(3, 1, 2, 5)
    out = model(x)
    assert out.shape == (1, 2, 5)


def test_parameter_count():
    model = LargeScaleSNNLM(vocab_size=5, embed_dim=8, hidden_dim=16, num_layers=2, dropout=0.0)
    counts = count_parameters(model)
    assert counts["total"] > 0
    assert counts["trainable"] > 0
