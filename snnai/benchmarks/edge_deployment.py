"""Edge deployment experiment utilities."""
import json
import time

import torch


def edge_profile(model, sample_input, device="cpu"):
    """Profile model size and latency for edge deployment.

    Parameters
    ----------
    model : torch.nn.Module
        Model to profile.
    sample_input : torch.Tensor
        Sample input.
    device : str
        Target device.

    Returns
    -------
    dict
        Size and latency report.
    """
    model = model.to(device)
    sample_input = sample_input.to(device)
    model.eval()
    start = time.perf_counter()
    with torch.no_grad():
        model(sample_input)
    latency = time.perf_counter() - start
    param_count = sum(p.numel() for p in model.parameters())
    size_kb = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024
    return {
        "device": device,
        "parameters": param_count,
        "size_kb": size_kb,
        "latency_seconds": latency,
    }


def export_to_torchscript(model, sample_input, path):
    """Export model for edge inference.

    Falls back to ``torch.save`` if TorchScript tracing fails (e.g. due to
    snnTorch surrogate gradients).
    """
    model.eval()
    try:
        traced = torch.jit.trace(model, sample_input)
        traced.save(path)
    except Exception:
        torch.save(model.state_dict(), path)
    return path
