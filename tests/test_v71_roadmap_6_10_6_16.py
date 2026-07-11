"""Tests for roadmap phases 6.10 - 6.16 (v6.6.2+ stabilization & rigor).

Covers:
- 6.10 SSA stabilization (residual / layer-norm / relaxed normalize)
- 6.10/6.13 LargeScaleSNNLM all-features forward+backward (no NaN), ssa_input
- 6.11 HippocampalMemory FIFO + reset, trainer reset hook
- 6.12 Knowledge distillation runs
- 6.15 fair_compare seed loop + fairness matcher
- 6.14 Bio-NAS real evaluator path
- 6.16 evaluator KeyError fix
"""
import math

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F


def test_ssa_residual_and_layernorm_change_output():
    from snnai.modules.language.spiking_attention import SpikingSelfAttention

    x = torch.randn(2, 8, 16)
    sa_res = SpikingSelfAttention(16, num_heads=1, causal=True,
                                  enable_residual=True, enable_layernorm=True)
    sa_nor = SpikingSelfAttention(16, num_heads=1, causal=True,
                                  enable_residual=False, enable_layernorm=False)
    out_res = sa_res(x)
    out_nor = sa_nor(x)
    assert out_res.shape == (2, 8, 16)
    assert not torch.allclose(out_res, out_nor), "residual/ln should change output"
    assert torch.isfinite(out_res).all()


def test_ssa_score_scale_runs():
    from snnai.modules.language.spiking_attention import SpikingSelfAttention

    x = torch.randn(2, 6, 16)
    sa = SpikingSelfAttention(16, num_heads=2, causal=True, score_scale=0.5)
    out = sa(x)
    assert out.shape == (2, 6, 16)
    assert torch.isfinite(out).all()


def test_ssa_relaxed_normalize_floor():
    from snnai.modules.language.spiking_attention import SpikingSelfAttention

    sa = SpikingSelfAttention(8, num_heads=1, causal=False)
    scores = torch.zeros(2, 5, 5)
    out = sa._normalize(scores, None)
    assert torch.isfinite(out).all()


def test_ssa_normalize_backward_is_finite():
    # Regression: zero-variance rows previously produced a 0*Inf (0 x inf)
    # NaN in the backward pass through var.sqrt().clamp_min(eps), which made
    # the all-features SNN loss explode to NaN. (var + eps).sqrt() fixes it.
    from snnai.modules.language.spiking_attention import SpikingSelfAttention

    sa = SpikingSelfAttention(8, num_heads=1, causal=True)
    L = 6
    mask = torch.triu(torch.ones(L, L), diagonal=1).bool()
    scores = torch.randn(2, L, L, requires_grad=True)
    out = sa._normalize(scores, mask)
    (out ** 2).sum().backward()
    assert torch.isfinite(scores.grad).all()


def test_large_scale_all_features_forward_backward_no_nan():
    from snnai.modules.language.large_scale_lm import LargeScaleSNNLM

    model = LargeScaleSNNLM(
        vocab_size=20, embed_dim=32, hidden_dim=32, num_layers=1,
        use_hippocampus_gate=True, use_spiking_attention=True, num_heads=1,
        ssa_input='membrane', enable_ssa_residual=True, enable_ssa_layernorm=True,
    )
    x = torch.zeros(4, 2, 8, 20)
    logits, spikes = model(x, return_spikes=True)
    assert logits.shape == (2, 8, 20)
    loss = F.cross_entropy(logits.reshape(-1, 20),
                           torch.randint(0, 20, (2 * 8,)))
    loss.backward()
    assert torch.isfinite(loss).all(), "all-features loss must be finite (no NaN/Inf)"
    assert torch.isfinite(logits).all()
    rate = sum(s.float().mean().item() for s in spikes) / len(spikes)
    assert rate >= 0.0


def test_large_scale_reset_memory_clears_hippocampus():
    from snnai.modules.language.large_scale_lm import LargeScaleSNNLM

    model = LargeScaleSNNLM(vocab_size=20, embed_dim=16, hidden_dim=16, num_layers=1,
                            use_hippocampus_gate=True)
    x = torch.zeros(3, 2, 6, 20)
    model(x)
    count_before = model.hippocampus_gate.memory.count.item()
    assert count_before > 0
    model.reset_memory()
    assert model.hippocampus_gate.memory.count.item() == 0


def test_hippocampal_memory_fifo_and_reset():
    from snnai.modules.hippocampus.associative_memory import HippocampalMemory

    mem = HippocampalMemory(dim=4, capacity=4)
    for i in range(10):
        k = torch.full((1, 4), float(i))
        v = torch.full((1, 4), float(i * 10))
        mem.store(k, v)
    assert mem.count.item() == 4
    assert torch.isfinite(mem.keys).all()
    mem.reset()
    assert mem.count.item() == 0
    assert mem.keys.abs().sum().item() == 0.0


def test_distillation_runs():
    from snnai.modules.language.large_scale_lm import LargeScaleSNNLM
    from snnai.benchmarks.transformer_comparison import TransformerBaseline
    from snnai.benchmarks.distillation import run_distillation

    vocab = 30
    student = LargeScaleSNNLM(vocab, embed_dim=16, hidden_dim=16, num_layers=1)
    teacher = TransformerBaseline(vocab, d_model=16, nhead=2, num_layers=1, dim_feedforward=32)

    class DummyTok:
        vocab_size = vocab

        def encode(self, text):
            return [ord(c) % vocab for c in text]

    text = "the quick brown fox jumps " * 20
    history = run_distillation(student, teacher, text, DummyTok(), device='cpu',
                               epochs=1, seq_len=16, batch_size=8, time_steps=4,
                               lr=1e-3, verbose=False)
    assert 'val_loss' in history and 'val_ppl' in history
    assert math.isfinite(history['val_loss'][-1])


def _dummy_tokenizer(vocab):
    class DummyTok:
        vocab_size = vocab

        def encode(self, text):
            return [ord(c) % vocab for c in text]

        def decode(self, ids):
            if hasattr(ids, "tolist"):
                ids = ids.tolist()
            return "".join(chr((int(i) % 95) + 32) for i in ids)
    return DummyTok()


def test_fair_compare_seed_loop_reports_mean_std():
    from snnai.modules.language.large_scale_lm import LargeScaleSNNLM
    from snnai.benchmarks.transformer_comparison import TransformerBaseline, fair_compare

    vocab = 40
    snn = LargeScaleSNNLM(vocab, embed_dim=16, hidden_dim=16, num_layers=1)
    tf = TransformerBaseline(vocab, d_model=16, nhead=2, num_layers=1, dim_feedforward=32)
    tok = _dummy_tokenizer(vocab)
    text = "to be or not to be that is the question " * 15
    res = fair_compare(text, tok, snn, tf, epochs=1, seq_len=16, batch_size=8,
                       time_steps=4, device='cpu', seeds=(0, 1), verbose=False)
    assert 'snn_val_ppl_mean' in res and 'snn_val_ppl_std' in res
    assert len(res['snn_seeds']) == 2


def test_fair_compare_match_transformer():
    from snnai.modules.language.large_scale_lm import LargeScaleSNNLM
    from snnai.benchmarks.transformer_comparison import (
        TransformerBaseline, build_matched_transformer, fair_compare)

    vocab = 40
    snn = LargeScaleSNNLM(vocab, embed_dim=16, hidden_dim=32, num_layers=1)
    tok = _dummy_tokenizer(vocab)
    matched = build_matched_transformer(snn, vocab)
    snn_params = sum(p.numel() for p in snn.parameters())
    tf_params = sum(p.numel() for p in matched.parameters())
    assert tf_params <= snn_params * 3 + 1, f"tf_params={tf_params} snn_params={snn_params}"

    text = "all the world is a stage " * 15
    res = fair_compare(text, tok, snn, matched, epochs=1, seq_len=16, batch_size=8,
                       time_steps=4, device='cpu', match_transformer=True,
                       seeds=(0,), verbose=False)
    assert 'transformer_parameters' in res


def test_lm_evaluator_real_path():
    from snnai.bio_nas.lm_search_space import lm_serial_architecture
    from snnai.bio_nas.lm_evaluator import (
        evaluate_lm_architecture, evaluate_lm_architecture_real)

    arch = lm_serial_architecture(hidden_dim=16)
    real = evaluate_lm_architecture_real(arch, vocab_size=30, hidden_dim=16,
                                         seq_len=16, epochs=1, device='cpu')
    assert math.isfinite(real['val_ppl'])

    out = evaluate_lm_architecture(arch, vocab_size=30, use_real_eval=True,
                                   real_hidden_dim=16, real_epochs=1, seq_len=16)
    assert out['real_eval'] is True
    assert math.isfinite(out['val_ppl'])


def test_evaluator_surrogate_no_keyerror_on_source_less_node():
    from snnai.bio_nas.search_space import Architecture
    from snnai.bio_nas.evaluator import evaluate_architecture

    arch = Architecture()
    arch.add_module("m1", "honeybee")
    arch.add_module("m2", "honeybee")
    arch.add_edge("input", "m2")
    arch.add_edge("m2", "output")
    arch.add_edge("m1", "output")
    assert arch.is_valid()
    acc = evaluate_architecture(arch, seed=1)
    assert math.isfinite(acc)
