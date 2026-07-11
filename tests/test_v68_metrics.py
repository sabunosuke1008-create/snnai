"""Tests for Phase 6.8 expanded generation-quality metrics."""
import torch

import snnai.benchmarks.generation_metrics as gm


class _TinyTokenizer:
    """Minimal tokenizer matching evaluate_generation's API."""

    def __init__(self, vocab_size=32):
        self.vocab_size = vocab_size

    def encode(self, text):
        ids = [ord(c) % self.vocab_size for c in text]
        return ids if ids else [0]

    def decode(self, ids):
        return "".join(chr(i) for i in ids)


def _make_model():
    from snnai.modules.language import LargeScaleSNNLM
    return LargeScaleSNNLM(
        vocab_size=32, embed_dim=8, hidden_dim=16, num_layers=1, max_seq_len=16,
    )


def test_evaluate_generation_includes_phase68_metrics():
    model = _make_model()
    res = gm.evaluate_generation(model, _TinyTokenizer(), ["ab", "cd"], max_chars=10, device="cpu")
    for key in (
        "generated", "bleu_1_mean", "cer_mean", "avg_length",
        "ger_mean", "semantic_sim_mean", "topic_drift_mean",
        "repetition_rate_mean", "newline_rate_mean",
    ):
        assert key in res, f"missing metric {key}"


def test_repetition_rate_detects_loops():
    assert gm.repetition_rate("a a a a a") > 0.5
    assert gm.repetition_rate("a b c d e") == 0.0


def test_newline_rate():
    assert abs(gm.newline_rate("a\nb\nc") - 2 / 5) < 1e-6
    assert gm.newline_rate("") == 0.0


def test_topic_drift_short_is_zero():
    assert gm.topic_drift_rate("one two three") == 0.0


def test_topic_drift_nonnegative():
    r = gm.topic_drift_rate("apple banana cherry apple banana cherry dog elephant tiger dog elephant tiger")
    assert r >= 0.0


def test_ger_heuristic_runs():
    rate = gm.grammar_error_rate("this is a sentence. another one here.")
    assert 0.0 <= rate <= 1.0


def test_semantic_self_similarity_overlap():
    sim = gm.semantic_similarity("the cat sat", prompt="the cat ran")
    assert 0.0 <= sim <= 1.0


def test_text_quality_metrics_keys():
    res = gm.text_quality_metrics(
        ["hello world. hello world. hello world.", "a b c d e f g."],
        prompts=["hello", "a"],
    )
    for key in (
        "ger_mean", "semantic_sim_mean", "topic_drift_mean",
        "repetition_rate_mean", "newline_rate_mean",
    ):
        assert key in res
