"""Energy estimation utilities for SNNAI."""


def estimate_energy(spike_count, joules_per_spike=1e-9):
    """Estimate energy consumption from spike count.

    Parameters
    ----------
    spike_count : float
        Total number of spikes.
    joules_per_spike : float
        Energy per spike in joules. Default 1 nJ is a rough neuro-
        morphorphic estimate; real values depend heavily on hardware.

    Returns
    -------
    dict
        Estimated energy in joules and arbitrary energy units.
    """
    return {
        "joules": spike_count * joules_per_spike,
        "nano_joules": spike_count * joules_per_spike * 1e9,
    }
