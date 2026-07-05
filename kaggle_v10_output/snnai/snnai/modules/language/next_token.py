"""Small SNN for next-token prediction."""
import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate


class NextTokenSNN(nn.Module):
    """Character-level next-token predictor.

    Parameters
    ----------
    vocab_size : int
        Vocabulary size.
    hidden_size : int, optional
        Hidden layer size (default 64).
    beta : float, optional
        Membrane time constant (default 0.9).
    """

    def __init__(self, vocab_size, hidden_size=64, beta=0.9):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.embed = nn.Linear(vocab_size, hidden_size, bias=False)
        self.lif1 = snn.Leaky(beta=beta, threshold=1.0, learn_threshold=False,
                              spike_grad=surrogate.fast_sigmoid())
        self.fc_out = nn.Linear(hidden_size, vocab_size, bias=False)
        self.lif2 = snn.Leaky(beta=beta, threshold=1.0, learn_threshold=False,
                              spike_grad=surrogate.fast_sigmoid())

    def forward(self, x):
        """Predict next token.

        Parameters
        ----------
        x : torch.Tensor
            Input spikes of shape (time_steps, batch, seq_len, vocab_size).

        Returns
        -------
        torch.Tensor
            Output spike counts of shape (batch, seq_len, vocab_size).
        """
        time_steps, batch_size, seq_len, _ = x.shape
        mem1 = torch.zeros(batch_size, seq_len, self.hidden_size)
        mem2 = torch.zeros(batch_size, seq_len, self.vocab_size)
        out_spikes = []
        for t in range(time_steps):
            cur1 = self.embed(x[t])
            spk1, mem1 = self.lif1(cur1, mem1)
            cur2 = self.fc_out(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)
            out_spikes.append(spk2)
        out = torch.stack(out_spikes).sum(dim=0)
        return out

    def predict_next(self, x):
        """Return most likely next token index."""
        out = self.forward(x)
        return int(torch.argmax(out[:, -1, :], dim=1).item())
