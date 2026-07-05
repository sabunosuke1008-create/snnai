"""Tests for generation metrics."""
import torch

from snnai.benchmarks.generation_metrics import (
    perplexity,
    token_accuracy,
    sequence_accuracy,
    bleu_1,
    char_error_rate,
)
from snnai.modules.language import CharTokenizer
from snnai.modules.language.large_scale_lm import LargeScaleSNNLM
from snnai.benchmarks.large_corpus_trainer import CharLMDataset, collate_fn


def test_perplexity_and_accuracy():
    logits = torch.zeros(2, 3, 5)
    logits[:, :, 0] = 10.0
    targets = torch.zeros(2, 3, dtype=torch.long)
    ppl = perplexity(logits, targets)
    acc = token_accuracy(logits, targets)
    assert ppl < 1.01
    assert acc == 1.0


def test_sequence_accuracy():
    logits = torch.zeros(2, 3, 5)
    logits[:, :, 0] = 10.0
    targets = torch.zeros(2, 3, dtype=torch.long)
    assert sequence_accuracy(logits, targets) == 1.0


def test_bleu_and_cer():
    assert bleu_1('abc', 'abc') == 1.0
    assert bleu_1('abc', 'def') == 0.0
    assert char_error_rate('abc', 'abc') == 0.0
    assert char_error_rate('abc', 'abcd') == 1.0 / 3.0


def test_dataset_and_collate():
    tokenizer = CharTokenizer('abc ')
    ds = CharLMDataset('abc abc abc', tokenizer, seq_len=4)
    one_hot, targets = collate_fn([ds[0]], tokenizer.vocab_size)
    assert one_hot.shape == (1, 4, tokenizer.vocab_size)
    assert targets.shape == (1, 4)
