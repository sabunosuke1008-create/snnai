"""Inference energy benchmark comparing SNN and Transformer."""
import time
import torch

from snnai.benchmarks.energy_estimation import estimate_energy


class EnergyBenchmark:
    """Estimate inference energy by counting spikes/ops."""

    def __init__(self, joules_per_spike=1e-9, joules_per_macs=1e-10):
        self.joules_per_spike = joules_per_spike
        self.joules_per_macs = joules_per_macs

    def measure_snn(self, model, spikes):
        """Measure SNN inference energy.

        Parameters
        ----------
        model : nn.Module
            SNN model.
        spikes : torch.Tensor
            Input spikes.

        Returns
        -------
        dict
            Energy and spike count.
        """
        model.eval()
        with torch.no_grad():
            start = time.time()
            out = model(spikes)
            elapsed = time.time() - start
        spike_count = float(out.sum().item())
        energy = estimate_energy(spike_count, self.joules_per_spike)
        return {
            "spike_count": spike_count,
            "time_seconds": elapsed,
            "joules": energy["joules"],
        }

    def measure_transformer(self, model, indices, seq_len=10):
        """Measure Transformer inference energy (rough MAC-based estimate).

        Parameters
        ----------
        model : nn.Module
            Transformer model.
        indices : torch.Tensor
            Input indices.
        seq_len : int
            Sequence length.

        Returns
        -------
        dict
            Estimated energy and time.
        """
        model.eval()
        with torch.no_grad():
            start = time.time()
            _ = model(indices)
            elapsed = time.time() - start
        # Rough estimate: 2 * params * seq_len MACs
        params = sum(p.numel() for p in model.parameters())
        macs = 2 * params * seq_len
        joules = macs * self.joules_per_macs
        return {
            "macs": macs,
            "time_seconds": elapsed,
            "joules": joules,
        }

    def compare(self, snn_model, transformer_model, spikes, indices, seq_len=10):
        """Compare SNN and Transformer inference energy."""
        snn_result = self.measure_snn(snn_model, spikes)
        transformer_result = self.measure_transformer(transformer_model, indices, seq_len)
        return {
            "snn": snn_result,
            "transformer": transformer_result,
            "ratio": transformer_result["joules"] / max(snn_result["joules"], 1e-12),
        }
