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

    def __init__(self, vocab_size, embed_dim=512, hidden_dim=2048, num_layers=6,
                 dropout=0.1, beta=0.9, threshold=1.0, learn_threshold=False,
                 output_mode='mem_mean'):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_mode = output_mode

        self.embed = nn.Linear(vocab_size, embed_dim, bias=False)
        layers = []
        for i in range(num_layers):
            in_dim = embed_dim if i == 0 else hidden_dim
            layers.append(nn.Linear(in_dim, hidden_dim, bias=False))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.Dropout(dropout))
            layers.append(snn.Leaky(beta=beta, threshold=threshold,
                                    learn_threshold=learn_threshold,
                                    spike_grad=surrogate.fast_sigmoid()))
        self.layers = nn.ModuleList(layers)
        self.fc_out = nn.Linear(hidden_dim, vocab_size, bias=False)
        self.lif_out = snn.Leaky(beta=beta, threshold=threshold,
                                 learn_threshold=learn_threshold,
                                 spike_grad=surrogate.fast_sigmoid())

    def forward(self, x, return_spikes=False):
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            (time_steps, batch, seq_len, vocab_size).
        return_spikes : bool
            If True, also return a list of hidden-layer spike tensors for
            homeostatic regularization.

        Returns
        -------
        torch.Tensor or tuple
            (batch, seq_len, vocab_size) logits, optionally with a list of
            spike tensors of shape (time_steps, batch, seq_len, hidden).
        """
        time_steps, batch_size, seq_len, _ = x.shape
        mems = [None] * self.num_layers
        out_mems = []
        all_spikes = [[] for _ in range(self.num_layers)] if return_spikes else None
        for t in range(time_steps):
            cur = self.embed(x[t])
            for i in range(self.num_layers):
                lin = self.layers[i * 4]
                ln = self.layers[i * 4 + 1]
                drop = self.layers[i * 4 + 2]
                lif = self.layers[i * 4 + 3]
                cur = lin(cur)
                cur = ln(cur)
                cur = drop(cur)
                if mems[i] is None:
                    mems[i] = torch.zeros(batch_size, seq_len, self.hidden_dim, device=cur.device)
                spk, mems[i] = lif(cur, mems[i])
                if return_spikes:
                    all_spikes[i].append(spk)
                cur = spk
            out_cur = self.fc_out(cur)
            if t == 0:
                mem_out = torch.zeros(batch_size, seq_len, self.vocab_size, device=cur.device)
            else:
                mem_out = mem_out.detach() if not torch.is_grad_enabled() else mem_out
            _, mem_out = self.lif_out(out_cur, mem_out)
            out_mems.append(mem_out)

        if self.output_mode == 'spike_sum':
            # Legacy mode: sum output spikes over time.
            out_spikes = []
            mem_out = torch.zeros(batch_size, seq_len, self.vocab_size, device=cur.device)
            for t in range(time_steps):
                out_cur = self.fc_out(self.embed(x[t]))
                spk_out, mem_out = self.lif_out(out_cur, mem_out)
                out_spikes.append(spk_out)
            logits = torch.stack(out_spikes, dim=0).sum(dim=0)
        elif self.output_mode == 'mem_last':
            logits = out_mems[-1]
        else:  # mem_mean
            logits = torch.stack(out_mems, dim=0).mean(dim=0)

        if return_spikes:
            # Stack time dimension for each layer.
            spikes_list = [torch.stack(layer_spikes, dim=0) for layer_spikes in all_spikes]
            return logits, spikes_list
        return logits


def count_parameters(model):
    """Return total and trainable parameter counts."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {'total': total, 'trainable': trainable}
