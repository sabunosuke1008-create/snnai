"""Tests for v6.4.0 top-p (nucleus) sampling."""
import torch

from snnai.benchmarks.generation_metrics import _sample_next_token


def test_top_p_filters_low_probability_tokens():
    # Strongly favor token 0 (p ~0.89), then token 1 (p ~0.11).
    # With top_p=0.9, only token 0 should be in the nucleus.
    logits = torch.tensor([10.0, 8.0, 0.0, 0.0, 0.0])
    torch.manual_seed(0)
    samples = [_sample_next_token(logits, temperature=1.0, top_p=0.9).item() for _ in range(50)]
    assert all(s == 0 for s in samples)


def test_top_p_disabled_when_none():
    logits = torch.tensor([1.0, 1.0, 1.0])
    torch.manual_seed(1)
    sample = _sample_next_token(logits, temperature=1.0, top_p=None).item()
    assert 0 <= sample < 3


def test_greedy_when_temperature_zero():
    logits = torch.tensor([1.0, 5.0, 3.0])
    sample = _sample_next_token(logits, temperature=0.0).item()
    assert sample == 1
