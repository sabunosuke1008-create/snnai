import torch
from snnai.modules.language import CharTokenizer, SpikeEncoder
from snnai.modules.language.next_token import NextTokenSNN


def test_next_token_snn_shape():
    tokenizer = CharTokenizer("abcde")
    encoder = SpikeEncoder(vocab_size=tokenizer.vocab_size, time_steps=10)
    model = NextTokenSNN(vocab_size=tokenizer.vocab_size, hidden_size=16)
    text = "abc"
    indices = torch.tensor([tokenizer.encode(text)])
    spikes = encoder(indices)
    out = model(spikes)
    assert out.shape == (1, len(text), tokenizer.vocab_size)


def test_next_token_snn_predicts():
    tokenizer = CharTokenizer("abcde")
    encoder = SpikeEncoder(vocab_size=tokenizer.vocab_size, time_steps=10)
    model = NextTokenSNN(vocab_size=tokenizer.vocab_size, hidden_size=16)
    indices = torch.tensor([tokenizer.encode("abc")])
    spikes = encoder(indices)
    next_idx = model.predict_next(spikes)
    assert 0 <= next_idx < tokenizer.vocab_size
