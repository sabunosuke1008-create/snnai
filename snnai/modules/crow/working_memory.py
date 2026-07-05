"""Working-memory module using recurrent spiking neurons.

A minimal model of persistent activity for delayed-match-to-sample tasks.
Two memory neurons maintain a decision variable through recurrent
self-excitation and mutual inhibition.
"""
import torch
import torch.nn as nn
import snntorch as snn


class WorkingMemory(nn.Module):
    """Two-unit working memory with persistent activity.

    The network receives a brief cue that drives one of two memory
    populations. Recurrent dynamics then maintain the activity during a
    delay period, allowing a later probe to be compared against the
    remembered cue.

    Parameters
    ----------
    beta : float, optional
        Membrane time constant (default 0.85).
    threshold : float, optional
        Firing threshold (default 1.0).
    """

    def __init__(self, beta=0.85, threshold=1.0):
        super().__init__()
        self.beta = beta
        self.threshold = threshold
        self.input_size = 2
        self.memory_size = 2

        self.W_in = nn.Linear(self.input_size, self.memory_size, bias=False)
        self.W_rec = nn.Linear(self.memory_size, self.memory_size, bias=False)
        self.lif = snn.Leaky(beta=beta, threshold=threshold, learn_threshold=False)

        with torch.no_grad():
            # Strong input to each memory neuron from its corresponding cue.
            w_in = torch.eye(self.memory_size) * 2.0
            self.W_in.weight.copy_(w_in)

            # Self-excitation and mutual inhibition for winner-take-all memory.
            w_rec = torch.tensor([[0.6, -0.5],
                                  [-0.5, 0.6]])
            self.W_rec.weight.copy_(w_rec)

    def forward(self, x):
        """Run the working-memory network.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (time, batch, 2). The first channel
            drives memory A, the second drives memory B.

        Returns
        -------
        torch.Tensor
            Memory spikes of shape (time, batch, 2).
        """
        time_steps, batch_size, _ = x.shape
        mem = torch.zeros(batch_size, self.memory_size)
        spikes = []
        for t in range(time_steps):
            current = self.W_in(x[t]) + self.W_rec(mem)
            spk, mem = self.lif(current, mem)
            spikes.append(spk)
        return torch.stack(spikes)

    def classify_probe(self, memory_spikes, probe):
        """Compare a probe against maintained memory.

        Parameters
        ----------
        memory_spikes : torch.Tensor
            Spike tensor of shape (time, batch, 2) from the delay period.
        probe : torch.Tensor
            Probe input of shape (batch, 2).

        Returns
        -------
        torch.Tensor
            Match scores of shape (batch, 2). Higher first column
            indicates "match".
        """
        # Memory preference: which memory neuron fired more during delay.
        memory_pref = memory_spikes.mean(dim=0)  # (batch, 2)
        # Probe preference: which probe channel is active.
        probe_pref = probe  # (batch, 2)
        # Match score: dot product between memory and probe preferences.
        match = (memory_pref * probe_pref).sum(dim=1, keepdim=True)
        non_match = 1.0 - match
        return torch.cat([match, non_match], dim=1)
