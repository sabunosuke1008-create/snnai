import torch
from snnai.benchmarks.energy_benchmark import EnergyBenchmark
from snnai.benchmarks.transformer_baseline import TransformerLM
from snnai.modules.language import CharTokenizer, SpikeEncoder
from snnai.modules.language.large_lm import LargeNextTokenSNN


def test_energy_benchmark():
    tokenizer = CharTokenizer("abcde")
    encoder = SpikeEncoder(vocab_size=tokenizer.vocab_size, time_steps=5)
    snn = LargeNextTokenSNN(vocab_size=tokenizer.vocab_size, embed_dim=8, hidden_dim=16, num_layers=1)
    transformer = TransformerLM(vocab_size=tokenizer.vocab_size, d_model=16, nhead=2, num_layers=1)
    indices = torch.tensor([tokenizer.encode("abc")])
    spikes = encoder(indices)
    bench = EnergyBenchmark()
    result = bench.compare(snn, transformer, spikes, indices, seq_len=3)
    assert "snn" in result
    assert "transformer" in result
    assert "ratio" in result
    assert result["ratio"] > 0
