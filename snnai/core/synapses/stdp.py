"""Spike-timing-dependent plasticity (STDP) update for SNNAI core."""
import math
import torch


def stdp_update(pre_spikes, post_spikes, weights, A_plus=0.01, A_minus=0.01, tau=20.0):
    """Apply a simple pair-based STDP weight update.

    The update strengthens a synapse when a presynaptic spike precedes a
    postsynaptic spike and weakens it when the order is reversed. The
    magnitude decays exponentially with the spike timing difference.

    Parameters
    ----------
    pre_spikes : torch.Tensor
        Presynaptic spike tensor of shape (time, batch, pre_neurons) or
        (time, pre_neurons).
    post_spikes : torch.Tensor
        Postsynaptic spike tensor of shape (time, batch, post_neurons) or
        (time, post_neurons).
    weights : torch.Tensor
        Weight matrix of shape (post_neurons, pre_neurons).
    A_plus : float, optional
        Maximum potentiation magnitude (default 0.01).
    A_minus : float, optional
        Maximum depression magnitude (default 0.01).
    tau : float, optional
        Time constant of the STDP window in time steps (default 20.0).

    Returns
    -------
    torch.Tensor
        Updated weight matrix of shape (post_neurons, pre_neurons).
    """
    time_steps = pre_spikes.shape[0]

    # Ensure batch dimension exists for uniform handling.
    if pre_spikes.dim() == 2:
        pre_spikes = pre_spikes.unsqueeze(1)
    if post_spikes.dim() == 2:
        post_spikes = post_spikes.unsqueeze(1)

    dw = torch.zeros_like(weights)
    for t_post in range(time_steps):
        for t_pre in range(time_steps):
            dt = t_post - t_pre
            if dt == 0:
                continue
            elif dt > 0:
                factor = A_plus * math.exp(-dt / tau)
            else:
                factor = -A_minus * math.exp(dt / tau)

            pre = pre_spikes[t_pre].float()
            post = post_spikes[t_post].float()
            # post: (batch, post), pre: (batch, pre)
            # outer product across batch: post.T @ pre -> (post, pre)
            dw += factor * (post.T @ pre)

    return weights + dw
