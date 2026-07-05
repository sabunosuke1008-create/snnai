import torch
from snnai.modules.language import CharTokenizer, SpikeEncoder


def test_char_tokenizer_roundtrip():
    tokenizer = CharTokenizer("abc")
    encoded = tokenizer.encode("abc")
    decoded = tokenizer.decode(encoded)
    assert decoded == "abc"
    assert tokenizer.vocab_size == 3


def test_spike_encoder_shape():
    tokenizer = CharTokenizer("abcde")
    encoder = SpikeEncoder(vocab_size=tokenizer.vocab_size, time_steps=20)
    indices = torch.tensor([[0, 1, 2], [3, 4, 0]])
    spikes = encoder(indices)
    assert spikes.shape == (20, 2, 3, 5)
