"""Quantitative energy-efficiency profiling for SNNAI inference."""
import time

import torch

from snnai.benchmarks.energy_estimation import estimate_energy


def quantize_energy(model, sample_input, time_steps=20, joules_per_spike=1e-9):
    """Profile a model and return detailed energy metrics.

    Parameters
    ----------
    model : torch.nn.Module
        SNN model.
    sample_input : torch.Tensor
        Input of shape (time_steps, batch, ...).
    time_steps : int
        Time steps for evaluation.
    joules_per_spike : float
        Energy per spike.

    Returns
    -------
    dict
        Detailed energy report.
    """
    model.eval()
    start = time.perf_counter()
    with torch.no_grad():
        out = model(sample_input)
    latency = time.perf_counter() - start
    spike_count = float((out > 0).sum().item())
    energy = estimate_energy(spike_count, joules_per_spike)
    energy["latency_seconds"] = latency
    energy["avg_latency_per_step"] = latency / time_steps
    energy["spikes_per_step"] = spike_count / time_steps
    return energy
