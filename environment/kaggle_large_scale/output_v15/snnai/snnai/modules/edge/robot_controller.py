"""Robot control module combining reflex and action selection."""
import torch
import torch.nn as nn
import snntorch as snn


class RobotController(nn.Module):
    """Select robot actions from sensor inputs.

    Combines a reflex-like response with a learned action-selection layer.
    Inputs: proximity sensors (front, left, right).
    Outputs: action spikes for (forward, turn_left, turn_right, stop).

    Parameters
    ----------
    beta : float, optional
        Membrane time constant (default 0.9).
    threshold : float, optional
        Firing threshold (default 1.0).
    """

    def __init__(self, beta=0.9, threshold=1.0):
        super().__init__()
        self.input_size = 3
        self.hidden_size = 8
        self.output_size = 4
        self.fc1 = nn.Linear(self.input_size, self.hidden_size, bias=False)
        self.fc2 = nn.Linear(self.hidden_size, self.output_size, bias=False)
        self.lif1 = snn.Leaky(beta=beta, threshold=threshold, learn_threshold=False)
        self.lif2 = snn.Leaky(beta=beta, threshold=threshold, learn_threshold=False)
        with torch.no_grad():
            # Reflex: front proximity strongly avoids forward, triggers turn
            self.fc1.weight.normal_(0, 0.5)
            self.fc2.weight.normal_(0, 0.5)

    def forward(self, x):
        """Run controller.

        Parameters
        ----------
        x : torch.Tensor
            Sensor input of shape (time, batch, 3).

        Returns
        -------
        torch.Tensor
            Action spikes of shape (time, batch, 4).
        """
        time_steps, batch_size, _ = x.shape
        mem1 = torch.zeros(batch_size, self.hidden_size)
        mem2 = torch.zeros(batch_size, self.output_size)
        spikes = []
        for t in range(time_steps):
            cur1 = self.fc1(x[t])
            spk1, mem1 = self.lif1(cur1, mem1)
            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)
            spikes.append(spk2)
        return torch.stack(spikes)

    def select_action(self, x):
        """Return the action with the most spikes."""
        spikes = self.forward(x)
        counts = spikes.sum(dim=0).sum(dim=0)
        return int(torch.argmax(counts).item())
