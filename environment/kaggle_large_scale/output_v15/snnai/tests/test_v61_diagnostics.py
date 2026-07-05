"""Tests for SNN diagnostic utilities."""
import torch

from snnai.benchmarks.diagnostic import SNNDiagnostic, diagnose_step
from snnai.modules.language.large_scale_lm import LargeScaleSNNLM
from snnai.modules.language.tokenizer import CharTokenizer, SpikeEncoder


def test_diagnostic_hooks_capture_gradients_and_spikes():
    vocab = list('abc')
    tokenizer = CharTokenizer(vocab)
    model = LargeScaleSNNLM(vocab_size=tokenizer.vocab_size, embed_dim=32, hidden_dim=64, num_layers=2)
    diag = SNNDiagnostic(model)

    encoder = SpikeEncoder(vocab_size=tokenizer.vocab_size, time_steps=5)
    x = torch.randint(0, tokenizer.vocab_size, (2, 4))
    spikes = encoder(x)
    out = model(spikes)
    loss = torch.nn.functional.cross_entropy(out.reshape(-1, tokenizer.vocab_size), x.reshape(-1))
    loss.backward()

    summary = diag.summarize()
    assert 'grad_norms' in summary
    assert 'spike_rates' in summary
    assert len(summary['grad_norms']) > 0
    diag.remove_hooks()


def test_diagnose_step_returns_loss_and_stats():
    vocab = list('abc')
    tokenizer = CharTokenizer(vocab)
    model = LargeScaleSNNLM(vocab_size=tokenizer.vocab_size, embed_dim=32, hidden_dim=64, num_layers=2)
    inputs = torch.randint(0, tokenizer.vocab_size, (2, 4))
    targets = inputs.clone()
    result = diagnose_step(model, inputs, targets, tokenizer)
    assert 'loss' in result
    assert 'ppl' in result
    assert 'grad_norms' in result
    assert 'spike_rates' in result
