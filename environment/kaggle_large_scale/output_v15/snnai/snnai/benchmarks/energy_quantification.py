"""Quantitative energy-efficiency profiling for SNNAI inference."""
import time

import torch

from snnai.benchmarks.energy_estimation import estimate_energy


class SpikeCounter:
    """Attach to SNN modules and count spikes during a forward pass."""

    def __init__(self):
        self.handles = []
        self.spike_counts = {}
        self.total_spikes = 0

    def _make_hook(self, name):
        def hook(module, input, output):
            # snntorch Leaky returns (spk, mem) tuple
            if isinstance(output, tuple):
                spk = output[0]
            else:
                spk = output
            count = float((spk > 0).sum().item())
            self.spike_counts[name] = self.spike_counts.get(name, 0.0) + count
            self.total_spikes += count
        return hook

    def attach(self, model):
        """Register forward hooks on all LIF-like modules."""
        for name, module in model.named_modules():
            if hasattr(module, 'mem') and hasattr(module, 'threshold'):
                handle = module.register_forward_hook(self._make_hook(name))
                self.handles.append(handle)

    def remove(self):
        """Remove hooks."""
        for h in self.handles:
            h.remove()
        self.handles.clear()


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
    counter = SpikeCounter()
    counter.attach(model)
    start = time.perf_counter()
    with torch.no_grad():
        out = model(sample_input)
    latency = time.perf_counter() - start
    counter.remove()

    spike_count = counter.total_spikes
    energy = estimate_energy(spike_count, joules_per_spike)
    energy['latency_seconds'] = latency
    energy['avg_latency_per_step'] = latency / time_steps
    energy['spikes_per_step'] = spike_count / max(1, sample_input.size(1)) / time_steps
    energy['total_spikes'] = spike_count
    energy['layer_spikes'] = counter.spike_counts
    return energy
