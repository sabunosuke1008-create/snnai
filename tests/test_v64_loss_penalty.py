"""Tests for v6.4.0 training loss penalty for frequent tokens."""
import torch

from snnai.benchmarks.large_corpus_trainer import LargeCorpusTrainer
from snnai.modules.language.tokenizer import CharTokenizer


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


def test_loss_penalty_increases_for_penalty_tokens():
    vocab = "ab" + chr(10)
    tokenizer = CharTokenizer(vocab)
    text = ("ab" + chr(10)) * 16
    model = _DummyModel(tokenizer.vocab_size)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    newline_id = tokenizer.encode(chr(10))[0]

    torch.nn.init.zeros_(model.weight)
    trainer_no_penalty = LargeCorpusTrainer(
        model, opt, tokenizer, device='cpu',
        penalty_tokens=None, penalty_weight=0.0
    )
    model.train()
    loss_no, _, _ = trainer_no_penalty._run_epoch(
        trainer_no_penalty._make_loaders(text, seq_len=4, batch_size=2)[0],
        time_steps=2, mode='train'
    )

    torch.nn.init.zeros_(model.weight)
    trainer_with_penalty = LargeCorpusTrainer(
        model, opt, tokenizer, device='cpu',
        penalty_tokens=[newline_id], penalty_weight=1.0
    )
    loss_yes, _, _ = trainer_with_penalty._run_epoch(
        trainer_with_penalty._make_loaders(text, seq_len=4, batch_size=2)[0],
        time_steps=2, mode='train'
    )

    assert loss_yes > loss_no
