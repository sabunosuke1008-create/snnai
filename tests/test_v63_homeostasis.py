"""Tests for homeostatic regularization and return_spikes support."""
import torch

from snnai.benchmarks.homeostatic_loss import HomeostaticRegularizer
from snnai.modules.language import CharTokenizer
from snnai.modules.language.large_scale_lm import LargeScaleSNNLM
from snnai.benchmarks.large_corpus_trainer import LargeCorpusTrainer


def test_homeostatic_regularizer_low_loss_near_target():
    reg = HomeostaticRegularizer(target_firing_rate=0.12, homeostatic_weight=1.0)
    spikes = torch.zeros(4, 2, 3, 16)
    flat = spikes.view(-1)
    flat[:46] = 1
    loss = reg([spikes])
    assert loss.item() < 1e-3


def test_homeostatic_regularizer_high_loss_away_from_target():
    reg = HomeostaticRegularizer(target_firing_rate=0.12, homeostatic_weight=1.0)
    spikes = torch.zeros(4, 2, 3, 16)
    loss = reg([spikes])
    assert loss.item() > 0.01


def test_homeostatic_regularizer_multi_layer_scaling():
    reg = HomeostaticRegularizer(target_firing_rate=0.12, homeostatic_weight=0.1)
    spikes1 = torch.zeros(2, 8)
    spikes2 = torch.zeros(2, 8)
    loss = reg([spikes1, spikes2])
    expected = 0.1 * 2 * (0.12 ** 2)
    assert abs(loss.item() - expected) < 1e-4


def test_large_scale_snn_lm_return_spikes():
    vocab = "abc\n "
    tokenizer = CharTokenizer(vocab)
    model = LargeScaleSNNLM(
        vocab_size=tokenizer.vocab_size,
        embed_dim=8,
        hidden_dim=16,
        num_layers=2,
        output_mode='mem_last',
    )
    x = torch.zeros(3, 1, 4, tokenizer.vocab_size)
    x[:, :, 0, 0] = 1.0
    logits, spikes = model(x, return_spikes=True)
    assert logits.shape == (1, 4, tokenizer.vocab_size)
    assert len(spikes) == 2
    for s in spikes:
        assert s.shape == (3, 1, 4, 16)
        unique = set(s.flatten().tolist())
        assert unique <= {0, 1}


def test_large_corpus_trainer_includes_homeostatic_loss():
    vocab = "abc\n "
    tokenizer = CharTokenizer(vocab)
    model = LargeScaleSNNLM(
        vocab_size=tokenizer.vocab_size,
        embed_dim=8,
        hidden_dim=16,
        num_layers=1,
        output_mode='mem_last',
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    trainer = LargeCorpusTrainer(
        model, optimizer, tokenizer, val_ratio=0.0,
        homeostatic_weight=1e-2, target_firing_rate=0.12,
    )
    history = trainer.train(
        "abc abc abc cba cba cba", epochs=1, seq_len=4,
        batch_size=2, time_steps=3, save_path=None, verbose=False,
    )
    assert 'mean_firing_rate' in history
    assert len(history['mean_firing_rate']) == 1
    assert 0.0 <= history['mean_firing_rate'][0] <= 1.0
