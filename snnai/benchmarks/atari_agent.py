"""SNN agent for the minimal Atari catch environment."""
import torch
import torch.nn as nn
import snntorch as snn


class CatchSNNAgent(nn.Module):
    """Small SNN policy for the Catch game."""

    def __init__(self, input_size=3, hidden_size=16, n_actions=3, n_steps=10):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_actions = n_actions
        self.n_steps = n_steps
        self.fc1 = nn.Linear(input_size, hidden_size, bias=False)
        self.fc2 = nn.Linear(hidden_size, n_actions, bias=False)
        self.lif1 = snn.Leaky(beta=0.9, threshold=1.0, learn_threshold=False)
        self.lif2 = snn.Leaky(beta=0.9, threshold=1.0, learn_threshold=False)
        with torch.no_grad():
            self.fc1.weight.normal_(0, 0.5)
            self.fc2.weight.normal_(0, 0.5)

    def _encode(self, state):
        rates = torch.tensor(state, dtype=torch.float32)
        rates = rates / (rates.abs().max() + 1e-8)
        probs = rates.unsqueeze(0).repeat(self.n_steps, 1)
        return torch.bernoulli(probs.clamp(0, 1))

    def select_action(self, state):
        x = self._encode(state)
        mem1 = torch.zeros(1, self.hidden_size)
        mem2 = torch.zeros(1, self.n_actions)
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

    def update(self, spikes_in, spikes_hid, spikes_out, reward, action, lr=0.3):
        with torch.no_grad():
            pre = spikes_hid.mean(dim=0)
            post = torch.zeros(self.n_actions)
            post[action] = 1.0
            dw2 = lr * reward * torch.outer(post, pre)
            self.fc2.weight.add_(dw2)

            pre_in = spikes_in.mean(dim=0)
            dw1 = lr * reward * torch.outer(pre, pre_in)
            self.fc1.weight.add_(dw1)
