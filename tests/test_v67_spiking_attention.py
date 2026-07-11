"""Tests for Phase 6.7 Spiking Self-Attention (SSA)."""
import math

import torch

from snnai.modules.language import LargeScaleSNNLM, SpikingSelfAttention


def test_ssa_single_head_output_shape():
    sa = SpikingSelfAttention(hidden_dim=32, num_heads=1)
    x = torch.randn(2, 8, 32)
    out = sa(x)
    assert out.shape == (2, 8, 32)


def test_ssa_multi_head_output_shape():
    sa = SpikingSelfAttention(hidden_dim=32, num_heads=4)
    x = torch.randn(2, 8, 32)
    out = sa(x)
    assert out.shape == (2, 8, 32)


def test_ssa_return_attn_shape():
    sa = SpikingSelfAttention(hidden_dim=32, num_heads=1)
    x = torch.randn(2, 8, 32)
    out, attn = sa(x, return_attn=True)
    assert out.shape == (2, 8, 32)
    assert attn.shape == (2, 8, 8)


def test_ssa_head_mismatch_raises():
    try:
        SpikingSelfAttention(hidden_dim=32, num_heads=3)
    except ValueError:
        return
    raise AssertionError("expected ValueError for non-divisible heads")


def test_ssa_causal_no_future_leak():
    """Rows must not attend to future positions (causal mask)."""
    sa = SpikingSelfAttention(hidden_dim=16, num_heads=1, use_spike=False)
    x = torch.randn(1, 6, 16)
    _, attn = sa(x, return_attn=True)
    attn = attn[0].detach()
    for i in range(6):
        for j in range(i + 1, 6):
            assert attn[i, j].item() == 0.0


def test_ssa_is_nonnegative_spike():
    """Spike-based attention map entries are non-negative."""
    sa = SpikingSelfAttention(hidden_dim=16, num_heads=1, use_spike=True)
    x = torch.randn(1, 6, 16)
    _, attn = sa(x, return_attn=True)
    assert (attn >= 0).all()
    # Straight-through: forward is binary 0/1 (tiny soft residue
    # supplies the gradient).
    assert torch.allclose(attn, attn.round(), atol=1e-4)


def test_ssa_softmax_variant_sums_to_one():
    sa = SpikingSelfAttention(hidden_dim=16, num_heads=1, use_spike=False)
    x = torch.randn(1, 6, 16)
    _, attn = sa(x, return_attn=True)
    row = attn[0, 0]
    assert abs(row.sum().item() - 1.0) < 1e-5


def test_large_scale_lm_with_spiking_attention():
    model = LargeScaleSNNLM(
        vocab_size=32, embed_dim=16, hidden_dim=32, num_layers=1,
        use_spiking_attention=True, max_seq_len=16,
    )
    assert hasattr(model, "spiking_attention")
    x = torch.randint(0, 32, (1, 8))
    out = model(x)
    assert out.shape == (1, 8, 32)


def test_large_scale_lm_ssa_return_spikes():
    model = LargeScaleSNNLM(
        vocab_size=16, embed_dim=8, hidden_dim=16, num_layers=1,
        use_spiking_attention=True, max_seq_len=8,
    )
    x = torch.randint(0, 16, (1, 4))
    logits, spikes = model(x, return_spikes=True)
    assert logits.shape == (1, 4, 16)
    assert len(spikes) == 1


def test_large_scale_lm_ssa_backward():
    model = LargeScaleSNNLM(
        vocab_size=16, embed_dim=8, hidden_dim=16, num_layers=1,
        use_spiking_attention=True, max_seq_len=8,
    )
    x = torch.randint(0, 16, (1, 4))
    out = model(x)
    out.sum().backward()
    assert model.spiking_attention.q_proj.weight.grad is not None


def test_ssa_reset_has_no_mem_attribute():
    """SSA keeps no ``mem`` attr so it is safe under _reset_lif_states."""
    sa = SpikingSelfAttention(hidden_dim=16)
    sa.reset()
    assert not hasattr(sa, "mem")


def test_bionas_attention_layer_uses_real_ssa():
    from snnai.bio_nas.lm_evaluator import ProxyLm
    from snnai.bio_nas.lm_search_space import lm_serial_architecture
    arch = lm_serial_architecture(hidden_dim=32)
    model = ProxyLm(arch, vocab_size=32, embed_dim=16, hidden_dim=32, seq_len=8)
    attn_layer = None
    for name, module in model.layers.items():
        if arch.layers[name].layer_type == "attention":
            attn_layer = module
            break
    assert attn_layer is not None
    assert isinstance(attn_layer.attention, SpikingSelfAttention)
