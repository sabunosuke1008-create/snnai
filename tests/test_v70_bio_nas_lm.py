"""Tests for Bio-NAS Phase 7.0 LM architecture search."""
import math

import pytest

from snnai.bio_nas.lm_evaluator import evaluate_lm_architecture
from snnai.bio_nas.lm_evolution import LmEvolutionSearch, search_lm_architecture
from snnai.bio_nas.lm_search_space import (
    LM_LAYER_TYPES,
    LmArchitecture,
    LmLayer,
    lm_diverse_architecture,
    lm_serial_architecture,
)


def test_lm_layer_creation():
    """LmLayer validates layer_type."""
    layer = LmLayer("l1", "feedforward", hidden_dim=64)
    assert layer.name == "l1"
    assert layer.layer_type == "feedforward"
    assert layer.hidden_dim == 64
    with pytest.raises(ValueError):
        LmLayer("bad", "unknown_type")


def test_lm_architecture_serial_is_valid():
    """Predefined serial architecture is valid and uses 3 distinct LM types."""
    arch = lm_serial_architecture(hidden_dim=32)
    assert arch.is_valid()
    assert arch.count_lm_layer_types() == {"feedforward", "recurrent", "attention"}


def test_lm_architecture_diverse_is_valid():
    """Predefined diverse architecture uses all 4 LM types."""
    arch = lm_diverse_architecture(hidden_dim=32)
    assert arch.is_valid()
    assert arch.count_lm_layer_types() == {
        "feedforward",
        "recurrent",
        "attention",
        "hippocampus_gate",
    }


def test_lm_architecture_requires_at_least_one_lm_type():
    """An architecture with only biological modules is invalid."""
    arch = LmArchitecture()
    arch.add_layer("b1", "c_elegans")
    arch.add_edge("input", "b1")
    arch.add_edge("b1", "output")
    assert not arch.is_valid()


def test_lm_architecture_cycle_is_invalid():
    """A cyclic architecture is invalid."""
    arch = LmArchitecture()
    arch.add_layer("l1", "feedforward")
    arch.add_layer("l2", "recurrent")
    arch.add_edge("input", "l1")
    arch.add_edge("l1", "l2")
    arch.add_edge("l2", "l1")  # cycle
    arch.add_edge("l2", "output")
    assert not arch.is_valid()


def test_lm_architecture_mutate_produces_valid_child():
    """Mutating a valid arch yields a valid child most of the time."""
    base = lm_serial_architecture(hidden_dim=32)
    valid_children = 0
    for seed in range(20):
        import random
        rng = random.Random(seed)
        child = base.mutate(rng=rng)
        if child.is_valid():
            valid_children += 1
    # At least 50% of mutations should produce valid children
    assert valid_children >= 10


def test_lm_evaluate_returns_expected_metrics():
    """evaluate_lm_architecture returns the expected metric keys."""
    arch = lm_serial_architecture(hidden_dim=32)
    metrics = evaluate_lm_architecture(
        arch,
        vocab_size=64,
        embed_dim=16,
        hidden_dim=32,
        seq_len=16,
        epochs=1,
    )
    for key in [
        "val_ppl",
        "latency_sec",
        "energy_proxy_joules",
        "bleu1",
        "biological_penalty",
        "composite_score",
        "layer_type_count",
    ]:
        assert key in metrics
    assert math.isfinite(metrics["val_ppl"])
    assert metrics["latency_sec"] > 0
    assert metrics["layer_type_count"] == 3


def test_lm_evaluate_invalid_returns_infty():
    """Invalid architectures return infeasible metrics."""
    arch = LmArchitecture()
    # Empty architecture is invalid
    metrics = evaluate_lm_architecture(arch)
    assert metrics["val_ppl"] == float("inf")
    assert metrics["composite_score"] == -float("inf")
    assert metrics["layer_type_count"] == 0


def test_lm_evaluate_diverse_uses_more_types():
    """Diverse architecture reports 4 LM layer types."""
    arch = lm_diverse_architecture(hidden_dim=32)
    metrics = evaluate_lm_architecture(
        arch,
        vocab_size=64,
        embed_dim=16,
        hidden_dim=32,
        seq_len=16,
        epochs=1,
    )
    assert metrics["layer_type_count"] == 4


def test_lm_evolution_returns_best_arch():
    """LmEvolutionSearch returns a valid best architecture."""
    search = LmEvolutionSearch(
        population_size=4,
        n_generations=2,
        top_k=2,
        seed=0,
        eval_kwargs={
            "vocab_size": 64,
            "embed_dim": 16,
            "hidden_dim": 32,
            "seq_len": 16,
            "epochs": 1,
        },
    )
    best_arch, best_metrics, history, pareto = search.search()
    assert best_arch is not None
    assert best_arch.is_valid()
    assert isinstance(best_metrics, dict)
    assert len(history) == 2
    assert pareto  # non-empty pareto front


def test_lm_evolution_convenience_wrapper():
    """search_lm_architecture convenience wrapper works."""
    best_arch, best_metrics, history, pareto = search_lm_architecture(
        population_size=3,
        n_generations=1,
        top_k=2,
        seed=1,
        eval_kwargs={
            "vocab_size": 32,
            "embed_dim": 8,
            "hidden_dim": 16,
            "seq_len": 8,
            "epochs": 1,
        },
    )
    assert best_arch is not None
    assert best_arch.is_valid()


def test_lm_evolution_history_improves_or_stays():
    """Best score is non-decreasing across generations (elitism)."""
    search = LmEvolutionSearch(
        population_size=4,
        n_generations=3,
        top_k=2,
        seed=2,
        eval_kwargs={
            "vocab_size": 32,
            "embed_dim": 8,
            "hidden_dim": 16,
            "seq_len": 8,
            "epochs": 1,
        },
    )
    _, _, history, _ = search.search()
    scores = [h[1] for h in history]
    # Best score in gen i+1 should be >= best score in gen i
    for i in range(1, len(scores)):
        assert scores[i] >= scores[i - 1] - 1e-9


def test_lm_architecture_predecessors():
    """predecessors() returns correct predecessor list."""
    arch = lm_diverse_architecture()
    # "output" should have one predecessor ("l4")
    assert arch.predecessors("output") == ["l4"]
    # "l1" should have "input" as predecessor
    assert arch.predecessors("l1") == ["input"]
