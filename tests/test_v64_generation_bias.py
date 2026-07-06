"""Tests for v6.4.0 generation logit bias."""
import pytest
import torch

from snnai.benchmarks.generation_metrics import _apply_token_bias
from snnai.modules.language.tokenizer import CharTokenizer


def test_apply_token_bias_reduces_target_logits():
    vocab_size = 10
    logits = torch.zeros(vocab_size)
    bias_tokens = [0, 2]
    biased = _apply_token_bias(logits, bias_tokens, bias_weight=2.0)
    assert biased[0].item() == -2.0
    assert biased[2].item() == -2.0
    for i in range(vocab_size):
        if i not in bias_tokens:
            assert biased[i].item() == 0.0


def test_apply_token_bias_disabled_when_weight_zero():
    logits = torch.randn(10)
    biased = _apply_token_bias(logits, [0, 1], bias_weight=0.0)
    assert torch.allclose(biased, logits)


def test_apply_token_bias_empty_tokens():
    logits = torch.randn(10)
    biased = _apply_token_bias(logits, [], bias_weight=1.0)
    assert torch.allclose(biased, logits)


def test_char_tokenizer_newline_space_ids():
    vocab = "abc " + chr(10)
    tokenizer = CharTokenizer(vocab)
    assert tokenizer.encode(chr(10))[0] == tokenizer.char_to_idx[chr(10)]
    assert tokenizer.encode(" ")[0] == tokenizer.char_to_idx[" "]
