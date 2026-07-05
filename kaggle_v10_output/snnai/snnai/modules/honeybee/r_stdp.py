"""Reward-modulated spike-timing-dependent plasticity (R-STDP)."""
import torch

from snnai.core.synapses.stdp import stdp_update


def r_stdp_update(
    pre_spikes,
    post_spikes,
    weights,
    reward,
    A_plus=0.01,
    A_minus=0.01,
    tau=20.0,
):
    """Apply a reward-modulated STDP weight update.

    The standard STDP weight change is computed from the current weights,
    then scaled by the scalar reward signal. Positive rewards amplify
    Hebbian/anti-Hebbian updates; negative rewards invert them.

    Parameters
    ----------
    pre_spikes : torch.Tensor
        Presynaptic spike tensor of shape ``(time, pre_neurons)`` or
        ``(time, batch, pre_neurons)``.
    post_spikes : torch.Tensor
        Postsynaptic spike tensor of shape ``(time, post_neurons)`` or
        ``(time, batch, post_neurons)``.
    weights : torch.Tensor
        Current weight matrix of shape ``(post_neurons, pre_neurons)``.
    reward : float
        Scalar reward used to modulate the update magnitude and direction.
    A_plus : float, optional
        Maximum potentiation magnitude (default 0.01).
    A_minus : float, optional
        Maximum depression magnitude (default 0.01).
    tau : float, optional
        Time constant of the STDP window in time steps (default 20.0).

    Returns
    -------
    torch.Tensor
        Updated weight matrix of shape ``(post_neurons, pre_neurons)``.
    """
    updated = stdp_update(
        pre_spikes,
        post_spikes,
        weights,
        A_plus=A_plus,
        A_minus=A_minus,
        tau=tau,
    )
    dw = updated - weights
    return weights + reward * dw
