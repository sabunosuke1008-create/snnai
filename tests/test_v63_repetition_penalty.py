"""Tests for repetition penalty in generation."""
import torch

from snnai.benchmarks.generation_metrics import _apply_repetition_penalty


def test_repetition_penalty_disabled():
    logits = torch.tensor([1.0, 2.0, 3.0, 4.0])
    penalized = _apply_repetition_penalty(logits, recent_ids=[0, 1], repetition_penalty=1.0)
    assert torch.allclose(penalized, logits)


def test_repetition_penalty_reduces_repeated_logits():
    logits = torch.tensor([1.0, 2.0, 3.0, 4.0])
    penalized = _apply_repetition_penalty(logits, recent_ids=[3], repetition_penalty=2.0)
    # Token 3 was repeated -> its logit should decrease.
    assert penalized[3] < logits[3]
    # Other logits unchanged.
    assert torch.allclose(penalized[:3], logits[:3])


def test_repetition_penalty_increases_with_count():
    logits = torch.tensor([1.0, 2.0, 3.0, 4.0])
    once = _apply_repetition_penalty(logits, recent_ids=[3], repetition_penalty=2.0)
    many = _apply_repetition_penalty(logits, recent_ids=[3, 3, 3], repetition_penalty=2.0)
    assert many[3] < once[3]
