"""Utilities for converting event streams to spike tensors."""
import torch


def events_to_spike_tensor(events, height=28, width=28, time_steps=20):
    """Convert an event stream to a spike tensor.

    Parameters
    ----------
    events : list[dict]
        Events with keys ``t``, ``x``, ``y``, ``pol``.
    height : int
        Image height.
    width : int
        Image width.
    time_steps : int
        Number of time bins.

    Returns
    -------
    torch.Tensor
        Spike tensor of shape (time_steps, 1, height, width).
    """
    spikes = torch.zeros(time_steps, 1, height, width)
    if not events:
        return spikes
    max_t = max(e["t"] for e in events)
    for e in events:
        if max_t > 0:
            bin_idx = int((e["t"] / max_t) * (time_steps - 1))
        else:
            bin_idx = 0
        x = min(max(e["x"], 0), width - 1)
        y = min(max(e["y"], 0), height - 1)
        if e["pol"] != 0:
            spikes[bin_idx, 0, y, x] = 1.0
    return spikes
