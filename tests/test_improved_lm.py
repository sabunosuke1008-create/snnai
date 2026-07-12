"""Fast interface tests for ImprovedSNNLM (no training)."""
import torch

from snnai.modules.language.improved_lm import (
    ImprovedSNNLM, MushroomBodyEncoder, FixedReservoir, SynfireChain,
    count_parameters)


def _forward_ok(model, vocab=50, T=4, B=2, L=12):
    x = torch.zeros(T, B, L, vocab)
    x[:, :, :, 0] = 1.0  # one-hot on token 0
    out, spikes = model(x, return_spikes=True)
    assert out.shape == (B, L, vocab), out.shape
    assert torch.isfinite(out).all()
    assert all(s.shape[0] == T for s in spikes)
    fr = sum(s.float().mean().item() for s in spikes) / len(spikes)
    return fr


def test_mushroom_body_encoder_forward():
    enc = MushroomBodyEncoder(vocab_size=50, embed_dim=32, expand_dim=256, top_k=16)
    idx = torch.randint(0, 50, (3, 2, 8))
    out = enc(idx)
    assert out.shape == (3, 2, 8, 32)
    assert torch.isfinite(out).all()


def test_fixed_reservoir_forward():
    res = FixedReservoir(input_dim=32, readout_dim=32, reservoir_size=64)
    x = torch.randn(2, 10, 32)
    out = res(x)
    assert out.shape == (2, 10, 32)
    assert torch.isfinite(out).all()


def test_synfire_chain_forward():
    sf = SynfireChain(max_seq_len=48, output_dim=32)
    idx = torch.randint(0, 50, (4, 2, 12))
    out = sf(idx)
    assert out.shape == (4, 2, 12, 32)


def test_improved_current_like():
    model = ImprovedSNNLM(50, embed_dim=32, hidden_dim=32, num_layers=1, max_seq_len=64,
                          use_mushroom_body=False, use_fixed_reservoir=False,
                          use_synfire=False, use_sequence_recurrent=True)
    fr = _forward_ok(model)
    assert 0.01 < fr < 0.9


def test_improved_mb_synfire_gru():
    model = ImprovedSNNLM(50, embed_dim=32, hidden_dim=32, num_layers=1, max_seq_len=64,
                          use_mushroom_body=True, mushroom_expand=256, mushroom_topk=16,
                          use_synfire=True, use_sequence_recurrent=True,
                          use_fixed_reservoir=False)
    fr = _forward_ok(model)
    assert 0.01 < fr < 0.9


def test_improved_mb_synfire_reservoir():
    model = ImprovedSNNLM(50, embed_dim=32, hidden_dim=32, num_layers=1, max_seq_len=64,
                          use_mushroom_body=True, mushroom_expand=256, mushroom_topk=16,
                          use_synfire=True, use_sequence_recurrent=False,
                          use_fixed_reservoir=True, reservoir_size=64)
    fr = _forward_ok(model)
    assert 0.01 < fr < 0.9
    # Reservoir must freeze recurrent weights but keep readout trainable.
    p = count_parameters(model)
    assert p["trainable"] < p["total"]
