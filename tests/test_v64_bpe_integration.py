"""Tests for v6.4.0 BPE tokenizer integration."""
import pytest
import torch

from snnai.modules.language import CharTokenizer
from snnai.modules.language.bpe_tokenizer import BPETokenizer
from snnai.benchmarks.large_corpus_trainer import LargeCorpusTrainer


class _DummyModel(torch.nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.vocab_size = vocab_size
        self.weight = torch.nn.Parameter(torch.zeros(vocab_size, vocab_size))

    def forward(self, x, return_spikes=False):
        t, b, s, v = x.shape
        avg = x.mean(dim=0)
        out = torch.matmul(avg, self.weight)
        if return_spikes:
            return out, [torch.rand(t, b, s, 4)]
        return out


def test_bpe_tokenizer_has_compatible_interface():
    corpus = "hello world hello snn ai"
    tokenizer = BPETokenizer([corpus], vocab_size=20)
    assert hasattr(tokenizer, 'vocab_size')
    assert hasattr(tokenizer, 'char_to_idx')
    assert hasattr(tokenizer, 'idx_to_char')
    assert tokenizer.vocab_size == len(tokenizer.vocab)
    ids = tokenizer.encode("hello")
    assert len(ids) > 0
    text = tokenizer.decode(ids)
    assert isinstance(text, str)


def test_bpe_trainer_runs():
    corpus = "hello world hello snn ai " * 20
    tokenizer = BPETokenizer([corpus], vocab_size=20)
    model = _DummyModel(tokenizer.vocab_size)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    trainer = LargeCorpusTrainer(model, opt, tokenizer, device='cpu',
                                 penalty_tokens=None, penalty_weight=0.0)
    history = trainer.train(corpus, epochs=1, seq_len=4, batch_size=2,
                            time_steps=2, verbose=False)
    assert 'train_loss' in history
    assert 'val_loss' in history
