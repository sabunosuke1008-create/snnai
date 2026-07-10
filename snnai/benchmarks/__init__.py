"""Benchmarks and energy estimation for SNNAI."""
from .baseline_mlp import MLPBaseline, train_mlp_baseline
from .energy_estimation import estimate_energy
from .parallel_utils import (
    PARALLEL_STRATEGIES,
    cleanup_ddp_process_group,
    init_ddp_process_group,
    is_parallel,
    parallelize_model,
    unwrap_model,
)
from .synthetic_benchmark import run_synthetic_benchmark

__all__ = [
    "MLPBaseline",
    "estimate_energy",
    "run_synthetic_benchmark",
    "train_mlp_baseline",
    "parallelize_model",
    "unwrap_model",
    "is_parallel",
    "init_ddp_process_group",
    "cleanup_ddp_process_group",
    "PARALLEL_STRATEGIES",
]
