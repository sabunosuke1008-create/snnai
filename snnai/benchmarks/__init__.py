"""Benchmarks and energy estimation for SNNAI."""
from .baseline_mlp import MLPBaseline, train_mlp_baseline
from .energy_estimation import estimate_energy
from .synthetic_benchmark import run_synthetic_benchmark

__all__ = ["MLPBaseline", "estimate_energy", "run_synthetic_benchmark", "train_mlp_baseline"]
