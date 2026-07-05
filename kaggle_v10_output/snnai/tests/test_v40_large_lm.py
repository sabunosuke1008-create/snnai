import torch
from snnai.modules.language import CharTokenizer, SpikeEncoder
from snnai.modules.language.large_lm import LargeNextTokenSNN


def test_large_lm_shape():
    tokenizer = CharTokenizer("abcde")
    encoder = SpikeEncoder(vocab_size=tokenizer.vocab_size, time_steps=5)
    model = LargeNextTokenSNN(vocab_size=tokenizer.vocab_size, embed_dim=16,
                              hidden_dim=32, num_layers=2)
    indices = torch.tensor([tokenizer.encode("abc")])
    spikes = encoder(indices)
    out = model(spikes)
    assert out.shape == (1, 3, tokenizer.vocab_size)


def test_large_lm_parameter_count():
    tokenizer = CharTokenizer("abcdefghijklmnopqrstuvwxyz ")
    model = LargeNextTokenSNN(vocab_size=tokenizer.vocab_size, embed_dim=64,
                              hidden_dim=256, num_layers=2)
    params = sum(p.numel() for p in model.parameters())
    assert params > 80000
