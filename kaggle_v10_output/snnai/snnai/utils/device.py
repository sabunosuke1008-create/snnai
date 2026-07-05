"""Device selection utilities."""
import torch


def auto_select_device(prefer="cuda"):
    """Automatically select the best available compute device.

    Parameters
    ----------
    prefer : str
        Preferred device type: "cuda", "mps", or "cpu".

    Returns
    -------
    torch.device
        Selected device.
    """
    if prefer == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if prefer == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
