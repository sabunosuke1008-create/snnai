"""Tests for Phase 6.6 Hippocampus gate integration."""
import torch

from snnai.modules.language import HippocampusGate, LargeScaleSNNLM


def test_hippocampus_gate_output_shape():
    """HippocampusGate preserves (B, L, H) shape."""
    gate = HippocampusGate(hidden_dim=32, capacity=16)
    x = torch.randn(2, 8, 32)
    out = gate(x, store=True)
    assert out.shape == x.shape


def test_hippocampus_gate_resets_memory():
    """reset_memory() clears stored episodes."""
    gate = HippocampusGate(hidden_dim=16, capacity=8)
    x = torch.randn(1, 4, 16)
    _ = gate(x, store=True)
    assert gate.memory.count.item() > 0
    gate.reset_memory()
    assert gate.memory.count.item() == 0


def test_hippocampus_gate_store_flag():
    """store=False prevents storing new episodes."""
    gate = HippocampusGate(hidden_dim=16, capacity=8)
    gate.reset_memory()
    x = torch.randn(1, 4, 16)
    _ = gate(x, store=False)
    assert gate.memory.count.item() == 0
    _ = gate(x, store=True)
    assert gate.memory.count.item() > 0


def test_hippocampus_gate_gradient_flow():
    """Gradients flow through the gate."""
    gate = HippocampusGate(hidden_dim=16, capacity=8)
    x = torch.randn(1, 4, 16, requires_grad=True)
    out = gate(x, store=True)
    loss = out.sum()
    loss.backward()
    assert x.grad is not None
    assert gate.gate_proj.weight.grad is not None


def test_large_scale_snn_lm_with_hippocampus_gate():
    """LargeScaleSNNLM accepts use_hippocampus_gate and runs."""
    model = LargeScaleSNNLM(
        vocab_size=32,
        embed_dim=16,
        hidden_dim=32,
        num_layers=2,
        use_hippocampus_gate=True,
        hippocampus_capacity=8,
        max_seq_len=16,
    )
    assert hasattr(model, "hippocampus_gate")
    x = torch.randint(0, 32, (1, 8))
    out = model(x)
    assert out.shape == (1, 8, 32)


def test_large_scale_snn_lm_without_hippocampus_gate():
    """Default LargeScaleSNNLM has no hippocampus_gate attribute."""
    model = LargeScaleSNNLM(
        vocab_size=16,
        embed_dim=8,
        hidden_dim=16,
        num_layers=1,
        max_seq_len=8,
    )
    assert not hasattr(model, "hippocampus_gate")


def test_large_scale_snn_lm_hippocampus_return_spikes():
    """use_hippocampus_gate is compatible with return_spikes=True."""
    model = LargeScaleSNNLM(
        vocab_size=16,
        embed_dim=8,
        hidden_dim=16,
        num_layers=1,
        use_hippocampus_gate=True,
        hippocampus_capacity=4,
        max_seq_len=8,
    )
    x = torch.randint(0, 16, (1, 4))
    logits, spikes = model(x, return_spikes=True)
    assert logits.shape == (1, 4, 16)
    assert len(spikes) == 1  # num_layers


def test_large_scale_snn_lm_hippocampus_backward_compat():
    """use_hippocampus_gate=False preserves old behavior."""
    model = LargeScaleSNNLM(
        vocab_size=16,
        embed_dim=8,
        hidden_dim=16,
        num_layers=1,
        use_hippocampus_gate=False,
        max_seq_len=8,
    )
    x = torch.randint(0, 16, (1, 4))
    out = model(x)
    assert out.shape == (1, 4, 16)


def test_bionas_hippocampus_gate_in_proxy_lm():
    """Bio-NAS hippocampus_gate layer type uses real HippocampusGate."""
    from snnai.bio_nas import lm_diverse_architecture
    from snnai.bio_nas.lm_evaluator import ProxyLm
    arch = lm_diverse_architecture(hidden_dim=32)
    model = ProxyLm(arch, vocab_size=32, embed_dim=16, hidden_dim=32, seq_len=8)
    # Check the hippocampus_gate layer in the model uses HippocampusGate
    hippo_layer = None
    for name, module in model.layers.items():
        if arch.layers[name].layer_type == "hippocampus_gate":
            hippo_layer = module
            break
    assert hippo_layer is not None
    assert hasattr(hippo_layer.gate, "memory")
    assert hasattr(hippo_layer.gate, "reset_memory")
