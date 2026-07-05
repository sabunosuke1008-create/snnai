"""Large-scale SNN language model architecture for scaling experiments."""
import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate


class LargeScaleSNNLM(nn.Module):
    """Configurable large SNN LM targeting 100M+ parameter scale.

    The local default is intentionally small for unit tests; larger configs
    are exercised on Kaggle.
    """

    def __init__(self, vocab_size, embed_dim=512, hidden_dim=2048, num_layers=6, dropout=0.1):
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
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            (time_steps, batch, seq_len, vocab_size).

        Returns
        -------
        torch.Tensor
            (batch, seq_len, vocab_size).
        """
        time_steps, batch_size, seq_len, _ = x.shape
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
                    mems[i] = torch.zeros(batch_size, seq_len, self.hidden_dim, device=cur.device)
                spk, mems[i] = lif(cur, mems[i])
                cur = spk
            out_cur = self.fc_out(cur)
            mem_out = torch.zeros(batch_size, seq_len, self.vocab_size, device=cur.device)
            spk_out, _ = self.lif_out(out_cur, mem_out)
            out_spikes.append(spk_out)
        return torch.stack(out_spikes, dim=0).sum(dim=0)


def count_parameters(model):
    """Return total and trainable parameter counts."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}
