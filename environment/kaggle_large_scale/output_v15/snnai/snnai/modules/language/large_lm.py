"""Larger SNN text generation model with multiple layers."""
import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate


class LargeNextTokenSNN(nn.Module):
    """Multi-layer SNN for next-token prediction.

    Parameters
    ----------
    vocab_size : int
        Vocabulary size.
    embed_dim : int, optional
        Embedding dimension (default 256).
    hidden_dim : int, optional
        Hidden dimension (default 1024).
    num_layers : int, optional
        Number of hidden layers (default 2).
    dropout : float, optional
        Dropout rate (default 0.1).
    """

    def __init__(self, vocab_size, embed_dim=256, hidden_dim=1024, num_layers=2, dropout=0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.embed = nn.Linear(vocab_size, embed_dim, bias=False)
        layers = []
        for i in range(num_layers):
            in_dim = embed_dim if i == 0 else hidden_dim
            layers.append(nn.Linear(in_dim, hidden_dim, bias=False))
            layers.append(nn.Dropout(dropout))
            layers.append(snn.Leaky(beta=0.9, threshold=1.0, learn_threshold=False,
                                    spike_grad=surrogate.fast_sigmoid()))
        self.layers = nn.ModuleList(layers)
        self.fc_out = nn.Linear(hidden_dim, vocab_size, bias=False)
        self.lif_out = snn.Leaky(beta=0.9, threshold=1.0, learn_threshold=False,
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
        # List of membrane potentials per layer
        mems = [None] * self.num_layers
        out_spikes = []
        for t in range(time_steps):
            cur = self.embed(x[t])
            for i in range(self.num_layers):
                lin = self.layers[i * 3]
                drop = self.layers[i * 3 + 1]
                lif = self.layers[i * 3 + 2]
                cur = lin(cur)
                cur = drop(cur)
                if mems[i] is None:
                    mems[i] = torch.zeros(batch_size, seq_len, self.hidden_dim)
                spk, mems[i] = lif(cur, mems[i])
                cur = spk
            out_cur = self.fc_out(cur)
            if mems[-1] is None:
                mem_out = torch.zeros(batch_size, seq_len, self.vocab_size)
            else:
                mem_out = torch.zeros(batch_size, seq_len, self.vocab_size)
            spk_out, _ = self.lif_out(out_cur, mem_out)
            out_spikes.append(spk_out)
        return torch.stack(out_spikes).sum(dim=0)
