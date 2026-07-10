"""Multi-GPU utilities for SNN/Transformer language model training.

Provides a single switch for DataParallel (DP), DistributedDataParallel (DDP),
or single-GPU training. DP is notebook-friendly; DDP is intended for script-based
launches where ``mp.spawn`` or ``torchrun`` is available.
"""
import os
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


PARALLEL_STRATEGIES = ('none', 'dp', 'ddp')


def is_parallel(model):
    """Return True if model is wrapped by DataParallel or DDP."""
    return isinstance(model, (nn.DataParallel, DDP))


def unwrap_model(model):
    """Return the inner model regardless of wrapping."""
    if isinstance(model, (nn.DataParallel, DDP)):
        return model.module
    return model


def parallelize_model(model, strategy='none', dim=0, device_ids=None,
                      find_unused=False, output_device=None):
    """Wrap a model for multi-GPU training.

    Parameters
    ----------
    model : torch.nn.Module
        Model already moved to the target device.
    strategy : {'none', 'dp', 'ddp'}
        'none' returns the model unchanged.
        'dp' wraps with ``nn.DataParallel`` (works in notebooks).
        'ddp' wraps with ``DistributedDataParallel`` (requires process group).
    dim : int
        Dimension along which inputs are scattered/gathered. For the SNN LM,
        the batch dimension is 1 because inputs are ``(T, B, L, V)``; for the
        Transformer it is 0 because inputs are ``(B, L)``.
    device_ids : list[int] or None
        GPUs to use. None means all available CUDA devices.
    find_unused : bool
        Passed to DDP ``find_unused_parameters``.
    output_device : int or None
        DP output device. None keeps the default (GPU 0).

    Returns
    -------
    torch.nn.Module
        Possibly wrapped model.
    """
    strategy = strategy.lower()
    if strategy not in PARALLEL_STRATEGIES:
        raise ValueError(f"strategy must be one of {PARALLEL_STRATEGIES}, got {strategy}")

    if strategy == 'ddp' and not dist.is_initialized():
        raise RuntimeError(
            "DDP strategy requires an initialized process group. "
            "Use init_ddp_process_group() or launch with torchrun."
        )

    if strategy == 'none' or not torch.cuda.is_available():
        return model

    if device_ids is None:
        device_ids = list(range(torch.cuda.device_count()))

    if strategy == 'dp':
        return nn.DataParallel(
            model,
            device_ids=device_ids,
            output_device=output_device,
            dim=dim,
        )

    if strategy == 'ddp':
        return DDP(
            model,
            device_ids=device_ids,
            output_device=output_device,
            dim=dim,
            find_unused_parameters=find_unused,
        )

    return model


def init_ddp_process_group(backend='nccl', init_method=None, rank=None,
                           world_size=None):
    """Initialize a process group for DDP.

    Safe to call from spawned notebook scripts. Environment variables
    MASTER_ADDR and MASTER_PORT are used as fallback when init_method is None.
    """
    if dist.is_initialized():
        return
    if rank is None:
        rank = int(os.environ.get('RANK', 0))
    if world_size is None:
        world_size = int(os.environ.get('WORLD_SIZE', 1))
    if init_method is None:
        init_method = 'env://'
    dist.init_process_group(
        backend=backend,
        init_method=init_method,
        rank=rank,
        world_size=world_size,
    )


def cleanup_ddp_process_group():
    """Destroy the current DDP process group if one exists."""
    if dist.is_initialized():
        dist.destroy_process_group()
