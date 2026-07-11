"""Proxy evaluator for LM architectures discovered by Bio-NAS.

This evaluator builds small surrogate PyTorch language models from an
`LmArchitecture` and returns a multi-objective score vector:
- perplexity (lower is better)
- latency in seconds (lower is better)
- energy estimate in joules (lower is better)
- BLEU-1 / character overlap (higher is better)

Biological constraints are applied as penalties:
- firing rate deviation from a target range
- total spike count
- energy budget
"""
import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from .lm_search_space import LmArchitecture, LM_LAYER_TYPES


class ProxyLmLayer(nn.Module):
    """A small surrogate layer for one LM layer type."""

    def __init__(self, layer_type, hidden_dim, vocab_size, seq_len):
        super().__init__()
        self.layer_type = layer_type
        self.hidden_dim = hidden_dim
        self.seq_len = seq_len

        if layer_type == "feedforward":
            self.net = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 2),
                nn.ReLU(),
                nn.Linear(hidden_dim * 2, hidden_dim),
            )
        elif layer_type == "recurrent":
            self.net = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        elif layer_type == "attention":
            self.qkv = nn.Linear(hidden_dim, hidden_dim * 3)
            self.proj = nn.Linear(hidden_dim, hidden_dim)
        elif layer_type == "hippocampus_gate":
            self.gate = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.Sigmoid(),
            )
            self.mem = nn.Linear(hidden_dim, hidden_dim)
        elif layer_type in ("c_elegans", "honeybee", "crow", "octopus"):
            # Biological modules are modelled as fixed random projections.
            self.net = nn.Linear(hidden_dim, hidden_dim, bias=False)
            with torch.no_grad():
                nn.init.orthogonal_(self.net.weight)
        else:
            raise ValueError(f"Unsupported layer type: {layer_type}")

    def forward(self, x, state=None):
        """Forward one layer.

        Parameters
        ----------
        x : Tensor (B, L, H)
        state : optional previous state for recurrent / hippocampus layers

        Returns
        -------
        out : Tensor (B, L, H)
        new_state : dict or None
        """
        if self.layer_type == "feedforward":
            return self.net(x), None
        elif self.layer_type == "recurrent":
            out, h = self.net(x, state["h"] if state else None)
            return out, {"h": h.detach()}
        elif self.layer_type == "attention":
            B, L, H = x.shape
            qkv = self.qkv(x).reshape(B, L, 3, H)
            q, k, v = qkv.unbind(dim=2)
            scores = torch.bmm(q, k.transpose(1, 2)) / math.sqrt(H)
            # Causal mask for language modelling
            mask = torch.triu(torch.ones(L, L, device=x.device), diagonal=1).bool()
            scores.masked_fill_(mask, float("-inf"))
            attn = F.softmax(scores, dim=-1)
            out = torch.bmm(attn, v)
            return self.proj(out), None
        elif self.layer_type == "hippocampus_gate":
            mem = state["mem"] if state else torch.zeros_like(x)
            combined = torch.cat([x, mem], dim=-1)
            g = self.gate(combined)
            out = g * x + (1 - g) * mem
            new_mem = self.mem(out)
            return out, {"mem": new_mem.detach()}
        else:
            return self.net(x), None


class ProxyLm(nn.Module):
    """Surrogate language model built from an LmArchitecture."""

    def __init__(self, arch, vocab_size, embed_dim, hidden_dim, seq_len):
        super().__init__()
        if not arch.is_valid():
            raise ValueError("Architecture is not valid")
        self.arch = arch
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.seq_len = seq_len
        self.order = arch.topological_order()

        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.input_proj = nn.Linear(embed_dim, hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, vocab_size)

        self.layers = nn.ModuleDict()
        for name in self.order:
            if name in ("input", "output"):
                continue
            layer_type = arch.layers[name].layer_type
            self.layers[name] = ProxyLmLayer(layer_type, hidden_dim, vocab_size, seq_len)

    def forward(self, token_ids):
        """Return logits (B, L, V) and a side-info dict."""
        B, L = token_ids.shape
        x = self.embedding(token_ids)
        x = self.input_proj(x)

        cache = {"input": x}
        states = {}
        spike_proxy = 0.0
        total_ops = 0.0

        for node in self.order:
            if node == "input":
                continue
            preds = self.arch.predecessors(node)
            if not preds:
                continue
            inp = sum(cache[p] for p in preds)
            if node == "output":
                logits = self.output_proj(inp)
                cache[node] = logits
            else:
                out, new_state = self.layers[node](inp, state=states.get(node))
                cache[node] = out
                states[node] = new_state
                # Proxy spike count: ReLU activation density
                spike_proxy += (out > 0).float().mean().item()
                total_ops += inp.numel() * self.hidden_dim

        # Energy proxy: joules ~ spike_proxy * total_ops scaled down
        energy_proxy = spike_proxy * total_ops * 1e-9
        side_info = {
            "spike_proxy": spike_proxy,
            "energy_proxy": energy_proxy,
            "total_ops": total_ops,
        }
        return cache["output"], side_info


def _make_dummy_corpus(seq_len, vocab_size, n_samples=400):
    """Generate a deterministic toy corpus for fast proxy evaluation."""
    g = torch.Generator().manual_seed(42)
    inputs = torch.randint(0, vocab_size, (n_samples, seq_len), generator=g)
    # Shifted targets for causal LM
    targets = torch.roll(inputs, shifts=-1, dims=1)
    targets[:, -1] = 0
    return inputs, targets


def evaluate_lm_architecture(
    arch,
    vocab_size=256,
    embed_dim=32,
    hidden_dim=64,
    seq_len=32,
    epochs=3,
    device="cpu",
    seed=42,
):
    """Train a proxy LM and return a metrics dict.

    Returns
    -------
    dict with keys: val_ppl, latency_sec, energy_proxy_joules, bleu1,
                    biological_penalty, composite_score, layer_type_count.
    """
    if not arch.is_valid():
        return {
            "val_ppl": float("inf"),
            "latency_sec": float("inf"),
            "energy_proxy_joules": float("inf"),
            "bleu1": 0.0,
            "biological_penalty": float("inf"),
            "composite_score": -float("inf"),
            "layer_type_count": 0,
        }

    torch.manual_seed(seed)
    model = ProxyLm(arch, vocab_size, embed_dim, hidden_dim, seq_len).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)

    inputs, targets = _make_dummy_corpus(seq_len, vocab_size)
    inputs, targets = inputs.to(device), targets.to(device)
    split = int(len(inputs) * 0.8)
    train_x, val_x = inputs[:split], inputs[split:]
    train_y, val_y = targets[:split], targets[split:]

    batch_size = 32
    for _ in range(epochs):
        model.train()
        for i in range(0, len(train_x), batch_size):
            bx = train_x[i : i + batch_size]
            by = train_y[i : i + batch_size]
            opt.zero_grad()
            logits, _ = model(bx)
            loss = F.cross_entropy(logits.reshape(-1, vocab_size), by.reshape(-1))
            loss.backward()
            opt.step()

    model.eval()
    with torch.no_grad():
        logits, side_info = model(val_x)
        val_loss = F.cross_entropy(logits.reshape(-1, vocab_size), val_y.reshape(-1)).item()
        val_ppl = math.exp(min(20.0, val_loss))

    # Latency: average over several forward passes
    sample = val_x[:1]
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(20):
        model(sample)
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    latency_sec = (time.perf_counter() - start) / 20.0

    energy_proxy = side_info["energy_proxy"]

    # BLEU-1 proxy: average 1-gram overlap between greedy generation and target
    with torch.no_grad():
        gen_logits, _ = model(val_x[:16])
        gen_ids = gen_logits.argmax(dim=-1)
        overlaps = []
        for pred, tgt in zip(gen_ids, val_y[:16]):
            pred_set = set(pred.tolist())
            tgt_set = set(tgt.tolist())
            if pred_set:
                overlaps.append(len(pred_set & tgt_set) / len(pred_set))
            else:
                overlaps.append(0.0)
        bleu1 = sum(overlaps) / len(overlaps)

    # Biological penalty: target firing rate ~0.12, energy budget
    target_rate = 0.12
    firing_rate = side_info["spike_proxy"] / max(1, len(arch.layers))
    rate_penalty = abs(firing_rate - target_rate)
    energy_penalty = max(0.0, energy_proxy - 1e-3)
    bio_penalty = rate_penalty + energy_penalty * 1000.0

    # Composite score: lower ppl/latency/energy, higher bleu, lower penalty
    composite = (
        -math.log(val_ppl + 1.0)
        - 10.0 * latency_sec
        - 1e3 * energy_proxy
        + 5.0 * bleu1
        - 5.0 * bio_penalty
    )

    return {
        "val_ppl": val_ppl,
        "latency_sec": latency_sec,
        "energy_proxy_joules": energy_proxy,
        "bleu1": bleu1,
        "biological_penalty": bio_penalty,
        "composite_score": composite,
        "layer_type_count": len(arch.count_lm_layer_types()),
    }
