"""Improved SNN LM combining biologically-motivated mechanisms.

This module implements an *improved* language model built from the
investigation findings (see docs/roadmap_v67_plus.md §11):

* **Mushroom Body (FlyHash) encoding** -- a fixed random expansion of the
  token one-hot followed by top-k sparsification, then a learned projection.
  Spike-native and (experiment A) slightly more accurate than a dense random
  projection at next-char 1-NN retrieval.
* **Fixed reservoir (Liquid State Machine)** -- replaces the fully-trained
  GRU recurrence with a randomly-fixed recurrent pool whose only learned
  component is a readout. Cuts BPTT through recurrent weights (experiment C:
  ~45% wall-clock reduction at a small accuracy cost).
* **Synfire chain timing skeleton** -- each sequence position deterministically
  activates a dedicated pool whose learnable "label line" projects to a timing
  vector (experiment B: ~3.7x fewer params and 76%->98% noise robustness vs
  GRU as a sequence generator).

The class mirrors ``LargeScaleSNNLM``'s ``(time, batch, seq, vocab)`` spike
interface and ``reset_memory()`` hook so it drops into the existing
``fair_compare`` / ``LargeCorpusTrainer`` pipeline.
"""
import math
import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate


class MushroomBodyEncoder(nn.Module):
    """FlyHash-style expand-then-sparsify token encoder.

    The token id is gathered through a *fixed* random projection
    ``(vocab, expand_dim)`` (no parameters), the top-``top_k`` entries are kept
    (sparse, spike-native), and a learned linear maps the sparse vector to
    ``embed_dim``.
    """

    def __init__(self, vocab_size, embed_dim, expand_dim=1024, top_k=64,
                 fixed_scale=0.5):
        super().__init__()
        self.expand_dim = expand_dim
        self.top_k = min(top_k, expand_dim)
        self.embed_dim = embed_dim
        # Fixed random projection (no grad).
        proj = torch.randn(vocab_size, expand_dim) * fixed_scale
        self.register_buffer("proj", proj)
        self.out = nn.Linear(expand_dim, embed_dim, bias=False)

    def forward(self, indices):
        """indices: (T, B, L) long tensor -> (T, B, L, embed_dim)."""
        T, B, L = indices.shape
        flat = indices.reshape(-1)
        expanded = self.proj[flat]  # (N, expand_dim)
        if self.top_k < self.expand_dim:
            vals, idx = expanded.topk(self.top_k, dim=-1)
            mask = torch.zeros_like(expanded)
            mask.scatter_(-1, idx, vals)
            expanded = mask
        embedded = self.out(expanded)  # (N, embed_dim)
        return embedded.reshape(T, B, L, self.embed_dim)


class FixedReservoir(nn.Module):
    """Fixed random recurrent reservoir with a learned readout.

    ``in_proj`` and ``rec`` are frozen (no gradient); only ``readout`` is
    trained. Runs over the sequence (L) dimension, mirroring the GRU usage in
    ``LargeScaleSNNLM``.
    """

    def __init__(self, input_dim, readout_dim, reservoir_size=256, beta=0.9,
                 threshold=1.0, sparsity=0.1, spectral_radius=0.9):
        super().__init__()
        self.reservoir_size = reservoir_size
        self.in_proj = nn.Linear(input_dim, reservoir_size, bias=False)
        self.rec = nn.Linear(reservoir_size, reservoir_size, bias=False)
        self.lif = snn.Leaky(beta=beta, threshold=threshold,
                             learn_threshold=False, learn_beta=False)
        self.readout = nn.Linear(reservoir_size, readout_dim, bias=False)
        with torch.no_grad():
            w = torch.randn(reservoir_size, reservoir_size)
            mask = torch.rand(reservoir_size, reservoir_size) < sparsity
            w = w * mask.float()
            w.fill_diagonal_(0)
            eig = torch.linalg.eigvals(w)
            sr = torch.max(torch.abs(eig)).item()
            if sr > 0:
                w = w * (spectral_radius / sr)
            self.rec.weight.copy_(w)
            self.in_proj.weight.normal_(0, 0.5)
        # Freeze the reservoir; train only the readout.
        self.in_proj.weight.requires_grad = False
        self.rec.weight.requires_grad = False

    def forward(self, x_seq):
        """x_seq: (B, L, input_dim) -> (B, L, readout_dim)."""
        B, L, _ = x_seq.shape
        xr = x_seq.transpose(0, 1)  # (L, B, input_dim)
        mem = torch.zeros(B, self.reservoir_size, device=x_seq.device)
        spikes = []
        for step in range(L):
            cur = self.in_proj(xr[step]) + self.rec(mem)
            spk, mem = self.lif(cur, mem)
            spikes.append(spk)
        sr = torch.stack(spikes, dim=0)  # (L, B, reservoir_size)
        out = self.readout(sr)  # (L, B, readout_dim)
        return out.transpose(0, 1)  # (B, L, readout_dim)


class SynfireChain(nn.Module):
    """Deterministic per-position timing skeleton.

    Position ``p`` deterministically fires pool ``p`` (one-hot over
    ``max_seq_len`` pools); a learned linear ("label line") maps the active
    pool to an ``output_dim`` timing vector that is added to the input. This is
    the biological HVC synfire-chain idea used as a *timing* signal rather than
    a standalone generator (experiment B).
    """

    def __init__(self, max_seq_len, output_dim):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.pool_out = nn.Linear(max_seq_len, output_dim, bias=False)

    def forward(self, indices):
        """indices: (T, B, L) -> (T, B, L, output_dim)."""
        T, B, L = indices.shape
        pos = torch.arange(L, device=indices.device)
        onehot = torch.nn.functional.one_hot(
            pos, num_classes=self.max_seq_len).float()  # (L, max_seq_len)
        timing = self.pool_out(onehot)  # (L, output_dim)
        return timing.unsqueeze(0).unsqueeze(0).expand(T, B, L, -1)


class ImprovedSNNLM(nn.Module):
    """Configurable SNN LM with mushroom-body encoding, fixed reservoir and
    synfire timing, reusing the SSA / hippocampus building blocks."""

    def __init__(self, vocab_size, embed_dim=512, hidden_dim=2048, num_layers=6,
                 dropout=0.1, beta=0.9, threshold=1.0, learn_threshold=False,
                 output_mode='mem_mean', use_sequence_recurrent=True,
                 use_fixed_reservoir=False, reservoir_size=256,
                 use_positional_encoding=True, max_seq_len=2048,
                 use_mushroom_body=False, mushroom_expand=1024, mushroom_topk=64,
                 use_synfire=False, use_hippocampus_gate=False,
                 hippocampus_capacity=64, use_spiking_attention=False,
                 num_heads=1, ssa_input='spike', ssa_score_scale=None,
                 enable_ssa_residual=True, enable_ssa_layernorm=True):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_mode = output_mode
        self.use_sequence_recurrent = use_sequence_recurrent
        self.use_fixed_reservoir = use_fixed_reservoir
        self.use_positional_encoding = use_positional_encoding
        self.use_mushroom_body = use_mushroom_body
        self.use_synfire = use_synfire
        self.use_hippocampus_gate = use_hippocampus_gate
        self.use_spiking_attention = use_spiking_attention
        self.num_heads = num_heads
        self.max_seq_len = max_seq_len
        self.ssa_input = ssa_input
        self.ssa_score_scale = ssa_score_scale
        self.enable_ssa_residual = enable_ssa_residual
        self.enable_ssa_layernorm = enable_ssa_layernorm

        if use_mushroom_body:
            self.mb_encoder = MushroomBodyEncoder(
                vocab_size, embed_dim, expand_dim=mushroom_expand, top_k=mushroom_topk)
        else:
            self.embed = nn.Embedding(vocab_size, embed_dim)

        if use_positional_encoding:
            self.pos_embed = nn.Embedding(max_seq_len, embed_dim)

        if use_sequence_recurrent and not use_fixed_reservoir:
            self.seq_recurrent = nn.GRU(embed_dim, embed_dim, batch_first=True)
        if use_fixed_reservoir:
            self.reservoir = FixedReservoir(
                embed_dim, embed_dim, reservoir_size=reservoir_size)

        if use_synfire:
            self.synfire = SynfireChain(max_seq_len, embed_dim)

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
                hidden_dim=hidden_dim, capacity=hippocampus_capacity)
        if use_spiking_attention:
            from snnai.modules.language.spiking_attention import SpikingSelfAttention
            self.spiking_attention = SpikingSelfAttention(
                hidden_dim=hidden_dim, num_heads=num_heads, causal=True,
                score_scale=ssa_score_scale,
                enable_residual=enable_ssa_residual,
                enable_layernorm=enable_ssa_layernorm)

    def _prepare_input(self, x):
        if x.dim() == 4:
            indices = x.argmax(dim=-1)
        elif x.dim() == 3:
            indices = x
        elif x.dim() == 2:
            indices = x.unsqueeze(0)
        else:
            raise ValueError(
                f"Expected input of shape (T,B,L,V), (T,B,L) or (B,L), got {x.shape}")
        time_steps, batch_size, seq_len = indices.shape
        if self.use_mushroom_body:
            embedded = self.mb_encoder(indices)
        else:
            flat = indices.view(-1)
            embedded = self.embed(flat).view(
                time_steps, batch_size, seq_len, self.embed_dim)
        if self.use_synfire:
            embedded = embedded + self.synfire(indices)
        if self.use_positional_encoding:
            if seq_len > self.max_seq_len:
                raise ValueError(
                    f"Sequence length {seq_len} exceeds max_seq_len {self.max_seq_len}")
            pos = torch.arange(seq_len, device=embedded.device)
            embedded = embedded + self.pos_embed(pos).unsqueeze(0).unsqueeze(0)
        return embedded

    def _reset_lif_states(self):
        for module in self.modules():
            if hasattr(module, 'mem') and module.mem is not None:
                module.mem = None

    def reset_memory(self):
        if self.use_hippocampus_gate and hasattr(self, 'hippocampus_gate'):
            self.hippocampus_gate.reset_memory()

    def forward(self, x, return_spikes=False):
        self._reset_lif_states()
        embedded = self._prepare_input(x)
        time_steps, batch_size, seq_len, _ = embedded.shape

        if self.use_fixed_reservoir:
            rec_out = torch.zeros(
                time_steps, batch_size, seq_len, self.embed_dim, device=embedded.device)
            for t in range(time_steps):
                rec_out[t] = self.reservoir(embedded[t])
            embedded = rec_out
        elif self.use_sequence_recurrent:
            flat = embedded.view(time_steps * batch_size, seq_len, self.embed_dim)
            recurrent_out, _ = self.seq_recurrent(flat)
            embedded = recurrent_out.view(
                time_steps, batch_size, seq_len, self.embed_dim)

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
            if self.use_spiking_attention:
                if self.ssa_input == 'membrane':
                    cur = self.spiking_attention(mems[i])
                else:
                    cur = self.spiking_attention(cur)
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
        else:
            logits = torch.stack(out_mems, dim=0).mean(dim=0)

        if return_spikes:
            spikes_list = [torch.stack(layer_spikes, dim=0) for layer_spikes in all_spikes]
            return logits, spikes_list
        return logits


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {'total': total, 'trainable': trainable}
