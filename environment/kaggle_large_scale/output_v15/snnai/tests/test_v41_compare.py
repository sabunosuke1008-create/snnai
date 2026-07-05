import torch
from snnai.benchmarks.transformer_baseline import TransformerLM, compare_models
from snnai.modules.language import CharTokenizer
from snnai.modules.language.large_lm import LargeNextTokenSNN


def test_transformer_lm_runs():
    tokenizer = CharTokenizer("abcde")
    model = TransformerLM(vocab_size=tokenizer.vocab_size, d_model=16, nhead=2, num_layers=1)
    x = torch.tensor([[0, 1, 2, 3]])
    out = model(x)
    assert out.shape == (1, 4, tokenizer.vocab_size)


def test_compare_models():
    corpus = "abcde" * 10
    tokenizer = CharTokenizer(corpus)
    snn = LargeNextTokenSNN(vocab_size=tokenizer.vocab_size, embed_dim=8, hidden_dim=16, num_layers=1)
    transformer = TransformerLM(vocab_size=tokenizer.vocab_size, d_model=16, nhead=2, num_layers=1)
    result = compare_models(corpus, snn, transformer, tokenizer, seq_len=5)
    assert "snn_loss" in result
    assert "transformer_loss" in result
