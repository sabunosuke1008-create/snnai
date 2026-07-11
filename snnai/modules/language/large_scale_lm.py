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
                 output_mode='mem_mean', use_sequence_recurrent=True,
                 use_positional_encoding=True, max_seq_len=2048,
                 use_hippocampus_gate=False, hippocampus_capacity=64):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_mode = output_mode
        self.use_sequence_recurrent = use_sequence_recurrent
        self.use_positional_encoding = use_positional_encoding
        self.use_hippocampus_gate = use_hippocampus_gate
        self.max_seq_len = max_seq_len

        self.embed = nn.Embedding(vocab_size, embed_dim)
        if use_positional_encoding:
            self.pos_embed = nn.Embedding(max_seq_len, embed_dim)
        if use_sequence_recurrent:
            self.seq_recurrent = nn.GRU(
                embed_dim, embed_dim, batch_first=True
            )

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

        if use_hippocampus_gate:
            from snnai.modules.language.hippocampus_gate import HippocampusGate
            self.hippocampus_gate = HippocampusGate(
                hidden_dim=hidden_dim, capacity=hippocampus_capacity
            )

    def _prepare_input(self, x):
        """Convert input to token embeddings.

        Supports both one-hot (T,B,L,V) and index (T,B,L) or (B,L) inputs.
        """
        if x.dim() == 4:
            indices = x.argmax(dim=-1)
        elif x.dim() == 3:
            indices = x
        elif x.dim() == 2:
            indices = x.unsqueeze(0)
        else:
            raise ValueError(
                f"Expected input of shape (T,B,L,V), (T,B,L) or (B,L), got {x.shape}"
            )

        time_steps, batch_size, seq_len = indices.shape
        flat = indices.view(-1)
        embedded = self.embed(flat).view(time_steps, batch_size, seq_len, self.embed_dim)

        if self.use_positional_encoding:
            if seq_len > self.max_seq_len:
                raise ValueError(
                    f"Sequence length {seq_len} exceeds max_seq_len {self.max_seq_len}"
                )
            pos = torch.arange(seq_len, device=embedded.device)
            embedded = embedded + self.pos_embed(pos).unsqueeze(0).unsqueeze(0)

        return embedded

    def _reset_lif_states(self):
        """Reset internal snntorch LIF membrane states.

        snntorch LIF neurons cache ``self.mem`` as an instance attribute. When
        the model is used with DataParallel, each replica inherits the cached
        shape from a previous forward, which can cause size mismatches across
        different batch splits. Resetting forces the neuron to reinitialize
        ``self.mem`` from the current input shape.
        """
        for module in self.modules():
            if hasattr(module, 'mem') and module.mem is not None:
                module.mem = None

    def forward(self, x, return_spikes=False):
        """Forward pass.

        Returns (batch, seq_len, vocab_size) logits, optionally with spikes.
        """
        self._reset_lif_states()
        embedded = self._prepare_input(x)
        time_steps, batch_size, seq_len, _ = embedded.shape

        if self.use_sequence_recurrent:
            flat = embedded.view(time_steps * batch_size, seq_len, self.embed_dim)
            recurrent_out, _ = self.seq_recurrent(flat)
            embedded = recurrent_out.view(time_steps, batch_size, seq_len, self.embed_dim)

        mems = [None] * self.num_layers
        out_mems = []
        all_spikes = [[] for _ in range(self.num_layers)] if return_spikes else None
        for t in range(time_steps):
            cur = embedded[t]
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
            if self.use_hippocampus_gate:
                cur = self.hippocampus_gate(cur, store=True)
            out_cur = self.fc_out(cur)
            if t == 0:
                mem_out = torch.zeros(batch_size, seq_len, self.vocab_size, device=cur.device)
            else:
                mem_out = mem_out.detach() if not torch.is_grad_enabled() else mem_out
            _, mem_out = self.lif_out(out_cur, mem_out)
            out_mems.append(mem_out)

        if self.output_mode == 'spike_sum':
            out_spikes = []
            mem_out = torch.zeros(batch_size, seq_len, self.vocab_size, device=cur.device)
            for t in range(time_steps):
                out_cur = self.fc_out(embedded[t])
                spk_out, mem_out = self.lif_out(out_cur, mem_out)
                out_spikes.append(spk_out)
            logits = torch.stack(out_spikes, dim=0).sum(dim=0)
        elif self.output_mode == 'mem_last':
            logits = out_mems[-1]
        else:  # mem_mean
            logits = torch.stack(out_mems, dim=0).mean(dim=0)

        if return_spikes:
            spikes_list = [torch.stack(layer_spikes, dim=0) for layer_spikes in all_spikes]
            return logits, spikes_list
        return logits


def count_parameters(model):
    """Return total and trainable parameter counts."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {'total': total, 'trainable': trainable}
