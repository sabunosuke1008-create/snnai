import torch
from snnai.benchmarks import run_synthetic_benchmark, train_mlp_baseline
from snnai.benchmarks.energy_estimation import estimate_energy


def test_energy_estimation():
    est = estimate_energy(1000)
    assert abs(est["joules"] - 1e-6) < 1e-15
    assert abs(est["nano_joules"] - 1000.0) < 1e-10


def test_synthetic_benchmark_runs():
    result = run_synthetic_benchmark(n_samples=200, seed=42)
    assert "accuracy" in result
    assert "avg_spikes" in result
    assert "energy_joules" in result
    assert 0.0 <= result["accuracy"] <= 1.0
    assert result["avg_spikes"] >= 0


def test_mlp_baseline_runs():
    result = train_mlp_baseline(n_samples=200, epochs=10, seed=42)
    assert "accuracy" in result
    assert "parameters" in result
    assert 0.0 <= result["accuracy"] <= 1.0
    assert result["parameters"] > 0
