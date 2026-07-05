"""N-MNIST style event data loader and simulator."""
import torch


def simulate_events_from_frames(frames, threshold=0.1):
    """Generate pseudo-event streams from a sequence of frames.

    Parameters
    ----------
    frames : torch.Tensor
        Frame sequence of shape (time_steps, 1, H, W) with values in [0, 1].
    threshold : float
        Minimum absolute pixel change to generate an event.

    Returns
    -------
    list[dict]
        List of events with keys ``t``, ``x``, ``y``, ``pol``.
    """
    events = []
    time_steps = frames.shape[0]
    for t in range(1, time_steps):
        diff = frames[t] - frames[t - 1]
        on = (diff > threshold).nonzero(as_tuple=False)
        off = (diff < -threshold).nonzero(as_tuple=False)
        for y, x in on[:, 1:]:
            events.append({"t": t, "x": int(x), "y": int(y), "pol": 1})
        for y, x in off[:, 1:]:
            events.append({"t": t, "x": int(x), "y": int(y), "pol": -1})
    return events


def load_nmnist_like_from_mnist(image, n_frames=10, seed=0):
    """Create a pseudo N-MNIST event stream from a single MNIST image.

    Generates a short moving-frame sequence and returns events.

    Parameters
    ----------
    image : torch.Tensor
        MNIST image of shape (1, 28, 28).
    n_frames : int
        Number of frames to generate.
    seed : int
        Random seed.

    Returns
    -------
    list[dict]
        Event stream.
    """
    torch.manual_seed(seed)
    frames = []
    base = image.squeeze(0)
    for i in range(n_frames):
        # Simulate slight translation and scale jitter
        shift = (i - n_frames // 2) * 0.5
        noise = torch.randn_like(base) * 0.02
        frame = (base + shift * 0.01 + noise).clamp(0, 1)
        frames.append(frame.unsqueeze(0))
    frames = torch.stack(frames)  # (time, 1, H, W)
    return simulate_events_from_frames(frames, threshold=0.05)
