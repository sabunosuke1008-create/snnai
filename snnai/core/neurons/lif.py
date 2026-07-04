"""Minimal LIF neuron utilities for Phase0 validation."""
import torch
import snntorch as snn


def create_lif_neuron(beta=0.9, threshold=1.0):
    """Return a simple snntorch Leaky integrate-and-fire neuron."""
    return snn.Leaky(beta=beta, threshold=threshold)


def lif_spike_count(mem, spike):
    """Return total spike count from a spike tensor."""
    return spike.sum().item()
