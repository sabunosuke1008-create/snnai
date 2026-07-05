"""Minimal C. elegans reflex module.

A small feedforward spiking network with lateral inhibition that maps
left/right sensory input to left/right motor output.
"""
import torch
import torch.nn as nn
import snntorch as snn


class ReflexModule(nn.Module):
    """Reflex arc for immediate avoidance/approach behavior.

    Architecture:
        2 sensory neurons -> 4 interneurons (2 excitatory, 2 inhibitory)
                         -> 2 motor neurons

    Sensory inputs:
        [left_stimulus, right_stimulus]

    Motor outputs (spikes):
        [move_left, move_right]

    Default behavior: cross-inhibition reflex. Strong left stimulus
tends to produce right motor spikes (avoidance) and vice versa.
    """

    def __init__(self, beta=0.9, threshold=1.0, dt=1.0):
        super().__init__()
        self.dt = dt
        self.input_size = 2
        self.hidden_size = 4
        self.output_size = 2

        self.fc1 = nn.Linear(self.input_size, self.hidden_size, bias=False)
        self.fc2 = nn.Linear(self.hidden_size, self.output_size, bias=False)
        self.lif1 = snn.Leaky(beta=beta, threshold=threshold, learn_threshold=False)
        self.lif2 = snn.Leaky(beta=beta, threshold=threshold, learn_threshold=False)

        with torch.no_grad():
            # Sensory -> interneuron weights
            # hidden[0,1] excited by left, hidden[2,3] excited by right
            w1 = torch.zeros(self.hidden_size, self.input_size)
            w1[[0, 1], 0] = 1.0   # left input -> left-preferring interneurons
            w1[[2, 3], 1] = 1.0   # right input -> right-preferring interneurons
            self.fc1.weight.copy_(w1)

            # Interneuron -> motor weights (cross-inhibition)
            # left interneurons inhibit right motor, excite left motor
            # right interneurons inhibit left motor, excite right motor
            w2 = torch.zeros(self.output_size, self.hidden_size)
            # Cross-inhibition: each side excites the opposite motor
            w2[0, [0, 1]] = -1.2   # left motor <- left interneurons (inhibition)
            w2[0, [2, 3]] = 0.8    # left motor <- right interneurons (excitation)
            w2[1, [2, 3]] = -1.2   # right motor <- right interneurons (inhibition)
            w2[1, [0, 1]] = 0.8    # right motor <- left interneurons (excitation)
            self.fc2.weight.copy_(w2)

    def forward(self, x):
        """Run the reflex network on a time series.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (time, batch, 2).

        Returns
        -------
        spikes : torch.Tensor
            Motor spikes of shape (time, batch, 2).
        latency : int
            First time step where any motor spike occurs.
        """
        time_steps, batch_size, _ = x.shape
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        if mem1.dim() == 0:
            mem1 = torch.zeros(batch_size, self.hidden_size)
            mem2 = torch.zeros(batch_size, self.output_size)

        spikes = []
        for t in range(time_steps):
            cur1 = self.fc1(x[t])
            spk1, mem1 = self.lif1(cur1, mem1)
            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)
            spikes.append(spk2)

        spikes = torch.stack(spikes)
        nonzero = (spikes.sum(dim=(1, 2)) > 0).nonzero(as_tuple=True)[0]
        latency = int(nonzero[0].item()) if len(nonzero) > 0 else time_steps
        return spikes, latency
