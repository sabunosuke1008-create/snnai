import pytest
from snnai.bio_nas import (
    EvolutionSearch,
    hub_architecture,
    parallel_architecture,
    serial_architecture,
)
from snnai.bio_nas.evaluator import evaluate_architecture


def test_serial_architecture_is_valid():
    arch = serial_architecture()
    assert arch.is_valid()


def test_hub_architecture_is_valid():
    arch = hub_architecture()
    assert arch.is_valid()


def test_parallel_architecture_is_valid():
    arch = parallel_architecture()
    assert arch.is_valid()


def test_invalid_c_elegans_not_adjacent_to_input():
    from snnai.bio_nas.search_space import Architecture

    arch = Architecture()
    arch.add_module("reflex", "c_elegans")
    arch.add_module("only_honeybee", "honeybee")
    # c_elegans exists but is not directly connected from input
    arch.add_edge("input", "only_honeybee")
    arch.add_edge("only_honeybee", "reflex")
    arch.add_edge("reflex", "output")
    assert not arch.is_valid()


def test_evolution_finds_score_at_least_as_good_as_serial():
    search = EvolutionSearch(population_size=6, n_generations=4, top_k=3, seed=1)
    best_arch, best_score, history = search.search()
    serial_score = evaluate_architecture(serial_architecture(), seed=1)
    print(f"best score: {best_score:.3f}, serial score: {serial_score:.3f}")
    assert best_arch.is_valid()
    assert best_score >= serial_score - 0.05
