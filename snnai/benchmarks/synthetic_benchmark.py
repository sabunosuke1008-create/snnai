"""Synthetic multi-modal benchmark for SNNAI v1.0."""
import torch

from snnai.integration import EncodingBridge
from snnai.modules.c_elegans import ReflexModule
from snnai.modules.crow import WorkingMemory
from snnai.modules.honeybee import SNNAgent
from snnai.modules.octopus import SubModule


def generate_synthetic_dataset(n_samples=400, time_steps=20, feature_dim=4, seed=42):
    """Generate a 4-class synthetic time-series dataset."""
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n_samples, time_steps, feature_dim, generator=g)
    y = torch.zeros(n_samples, dtype=torch.long)
    for i in range(n_samples):
        means = x[i].mean(dim=0)
        label = int(torch.argmax(torch.abs(means)).item())
        y[i] = label
    return x, y


def run_synthetic_benchmark(n_samples=400, seed=42):
    """Run the SNNAI pipeline on the synthetic benchmark.

    Returns
    -------
    dict
        Dictionary with accuracy, avg_spikes, and energy estimate.
    """
    x, y = generate_synthetic_dataset(n_samples=n_samples, seed=seed)

    # Simple module assembly
    reflex = ReflexModule(beta=0.9, threshold=1.0)
    memory = WorkingMemory(beta=0.85, threshold=1.0)
    octopus_sub = SubModule(input_size=4, hidden_size=4)
    agent = SNNAgent(size=3, n_cells=25, hidden_size=20, n_steps=20)

    bridge = EncodingBridge(in_dim=4, out_dim=2, time_steps=20)

    total_spikes = 0
    correct = 0
    for i in range(n_samples):
        sample = x[i].unsqueeze(1)  # (time, 1, features)

        # Reflex signal (first 2 features)
        reflex_input = sample[:, :, :2]
        reflex_spikes, _ = reflex(reflex_input)
        reflex_rates = bridge.spikes_to_rates(reflex_spikes)
        total_spikes += int(reflex_spikes.sum().item())

        # Memory signal (first 2 features, split into cue/probe)
        mem_input = torch.zeros(50, 1, 2)
        mem_input[:20, 0, 0] = sample[:20, 0, 0]
        mem_input[30:40, 0, 1] = sample[10:20, 0, 1]
        mem_spikes = memory(mem_input)
        total_spikes += int(mem_spikes.sum().item())

        # Octopus signal
        oct_spikes = octopus_sub(sample)
        total_spikes += int(oct_spikes.sum().item())

        # Simple voting classifier based on module rates
        score = torch.zeros(4)
        score[0] = reflex_rates[0, 0]
        score[1] = reflex_rates[0, 1]
        score[2] = mem_spikes[:, 0, :].mean()
        score[3] = oct_spikes[:, 0, :].mean()
        pred = int(torch.argmax(score).item())
        if pred == y[i].item():
            correct += 1

    accuracy = correct / n_samples
    avg_spikes = total_spikes / n_samples
    # Rough energy estimate: 1 nJ per spike (arbitrary unit)
    energy = avg_spikes * 1e-9
    return {"accuracy": accuracy, "avg_spikes": avg_spikes, "energy_joules": energy}
