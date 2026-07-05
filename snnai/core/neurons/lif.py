"""LIF neuron utilities for SNNAI core."""
import torch
import snntorch as snn


def create_lif_neuron(beta=0.9, threshold=1.0):
    """Return a simple snntorch Leaky integrate-and-fire neuron."""
    return snn.Leaky(beta=beta, threshold=threshold)


def lif_spike_count(mem, spike):
    """Return total spike count from a spike tensor."""
    return spike.sum().item()


def simulate_lif(neuron, input_current, num_steps, reset=True):
    """Simulate a single or population of LIF neurons over time.

    Parameters
    ----------
    neuron : snn.Leaky
        A snntorch Leaky integrate-and-fire neuron.
    input_current : torch.Tensor
        Tensor of shape (time, ...) where the leading dimension is the
        number of simulation steps.
    num_steps : int
        Number of time steps to simulate. Must be <= input_current.size(0).
    reset : bool, optional
        Whether to reset the neuron's state before simulation (default True).

    Returns
    -------
    spikes : torch.Tensor
        Stacked spike tensor of shape (num_steps, ...).
    mems : torch.Tensor
        Stacked membrane potential tensor of shape (num_steps, ...).
    """
    if reset:
        mem = neuron.init_leaky()
    else:
        mem = neuron.mem if hasattr(neuron, "mem") else neuron.init_leaky()
    if mem.dim() == 0:
        mem = torch.zeros(input_current.shape[1:])
    spikes = []
    mems = []
    for t in range(num_steps):
        spk, mem = neuron(input_current[t], mem)
        spikes.append(spk)
        mems.append(mem)
    return torch.stack(spikes), torch.stack(mems)
