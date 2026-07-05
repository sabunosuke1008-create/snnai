"""SNN agent for grid-world navigation using R-STDP."""
import torch
import torch.nn as nn
import snntorch as snn

from .spatial_encoding import place_cell_encode


class SNNAgent(nn.Module):
    """Spiking policy network for honeybee grid-world navigation.

    Architecture:
        place cells -> hidden LIF -> output LIF (4 actions)

    The agent runs its network for ``n_steps`` simulation steps per
    environment step, then selects the action with the highest output
    spike count. Weights are updated with reward-modulated STDP after
    each environment step.
    """

    def __init__(
        self,
        size=5,
        n_cells=25,
        hidden_size=20,
        n_actions=4,
        n_steps=10,
        beta=0.9,
        threshold=1.0,
        A_plus=0.05,
        A_minus=0.05,
        tau=10.0,
    ):
        super().__init__()
        self.size = size
        self.n_cells = n_cells
        self.hidden_size = hidden_size
        self.n_actions = n_actions
        self.n_steps = n_steps
        self.A_plus = A_plus
        self.A_minus = A_minus
        self.tau = tau

        self.fc1 = nn.Linear(n_cells, hidden_size, bias=False)
        self.fc2 = nn.Linear(hidden_size, n_actions, bias=False)
        self.lif1 = snn.Leaky(beta=beta, threshold=threshold, learn_threshold=False)
        self.lif2 = snn.Leaky(beta=beta, threshold=threshold, learn_threshold=False)

        with torch.no_grad():
            self.fc1.weight.normal_(0, 0.5)
            self.fc2.weight.normal_(0, 0.5)

    def _spike_encode(self, pos):
        if isinstance(pos, torch.Tensor):
            if pos.dim() == 2:
                pos = (pos[0, 0].item(), pos[0, 1].item())
            else:
                pos = (pos[0].item(), pos[1].item())
        rates = place_cell_encode(pos, size=self.size, n_cells=self.n_cells)
        # (n_cells,) -> repeat across time and apply Bernoulli sampling
        spikes = torch.bernoulli(rates.unsqueeze(0).repeat(self.n_steps, 1))
        return spikes

    def select_action(self, pos):
        """Choose an action from the current position.

        Returns
        -------
        int
            Selected action index.
        spikes_in : torch.Tensor
            Input spikes over time, shape (n_steps, n_cells).
        spikes_hid : torch.Tensor
            Hidden spikes over time, shape (n_steps, hidden_size).
        spikes_out : torch.Tensor
            Output spikes over time, shape (n_steps, n_actions).
        """
        x = self._spike_encode(pos)
        batch_size = 1
        mem1 = torch.zeros(batch_size, self.hidden_size)
        mem2 = torch.zeros(batch_size, self.n_actions)

        hid_spikes, out_spikes = [], []
        for t in range(self.n_steps):
            cur1 = self.fc1(x[t].unsqueeze(0))
            spk1, mem1 = self.lif1(cur1, mem1)
            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)
            hid_spikes.append(spk1.squeeze(0))
            out_spikes.append(spk2.squeeze(0))

        hid_spikes = torch.stack(hid_spikes)
        out_spikes = torch.stack(out_spikes)
        counts = out_spikes.sum(dim=0)
        action = int(torch.argmax(counts).item())
        return action, x, hid_spikes, out_spikes

    def select_action_from_spikes(self, x):
        """Choose an action from pre-computed input spikes.

        Parameters
        ----------
        x : torch.Tensor
            Input spikes of shape (n_steps, batch, n_cells).

        Returns
        -------
        int
            Selected action index.
        """
        time_steps, batch_size, _ = x.shape
        mem1 = torch.zeros(batch_size, self.hidden_size)
        mem2 = torch.zeros(batch_size, self.n_actions)
        out_spikes = []
        for t in range(time_steps):
            cur1 = self.fc1(x[t])
            spk1, mem1 = self.lif1(cur1, mem1)
            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)
            out_spikes.append(spk2)
        out_spikes = torch.stack(out_spikes)
        counts = out_spikes.sum(dim=0)
        return int(torch.argmax(counts, dim=1)[0].item())

    def update(self, spikes_in, spikes_hid, spikes_out, reward, action):
        """Apply R-STDP update favouring the selected action.

        A positive reward potentiates the pathway leading to the chosen
        action; a negative reward depresses it. The update is scaled by a
        fixed learning rate to keep weight changes stable.
        """
        lr = 0.1
        with torch.no_grad():
            pre = spikes_hid.mean(dim=0)  # (hidden_size,)
            post = torch.zeros(self.n_actions)
            post[action] = 1.0
            dw2 = lr * reward * torch.outer(post, pre)
            self.fc2.weight.add_(dw2)

            pre_in = spikes_in.mean(dim=0)  # (n_cells,)
            dw1 = lr * reward * torch.outer(pre, pre_in)
            self.fc1.weight.add_(dw1)
