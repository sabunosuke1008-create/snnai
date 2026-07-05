"""Checkpoint save/load utilities for long-running SNN training."""
import torch


def save_checkpoint(model, optimizer, epoch, path):
    """Save a training checkpoint.

    Parameters
    ----------
    model : torch.nn.Module
        Model to save.
    optimizer : torch.optim.Optimizer
        Optimizer to save.
    epoch : int
        Current epoch.
    path : str
        Destination file path.
    """
    state = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    torch.save(state, path)


def load_checkpoint(model, optimizer, path):
    """Load a training checkpoint.

    Parameters
    ----------
    model : torch.nn.Module
        Model to populate.
    optimizer : torch.optim.Optimizer
        Optimizer to populate.
    path : str
        Checkpoint file path.

    Returns
    -------
    int
        Epoch number stored in the checkpoint.
    """
    checkpoint = torch.load(path, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint["epoch"]
