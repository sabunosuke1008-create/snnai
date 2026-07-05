"""Optimization utilities for SNN inference speed and energy."""
import torch


def quantize_weights(model, bits=8):
    """Simple weight quantization helper (state_dict scale)."""
    scales = {}
    for name, param in model.named_parameters():
        min_val = param.data.min()
        max_val = param.data.max()
        scale = (max_val - min_val) / (2 ** bits - 1) if max_val != min_val else torch.tensor(1.0)
        scales[name] = {"min": float(min_val.item()), "max": float(max_val.item()), "scale": float(scale.item())}
    return scales


def prune_weights(model, threshold=0.01):
    """Zero out small-magnitude weights and return sparsity ratio."""
    total = 0
    pruned = 0
    for param in model.parameters():
        total += param.numel()
        mask = param.data.abs() < threshold
        param.data[mask] = 0
        pruned += int(mask.sum().item())
    return {"sparsity": pruned / max(total, 1), "total": total, "pruned": pruned}
