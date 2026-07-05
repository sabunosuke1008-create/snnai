"""Diagnostic utilities for SNN language model training."""
import json
import torch
import torch.nn as nn


class SNNDiagnostic:
    """Collect gradient, spike firing rate, and activation statistics."""

    def __init__(self, model):
        self.model = model
        self.reset()
        self._register_hooks()

    def reset(self):
        self.grad_norms = {}
        self.spike_rates = {}
        self.activation_stats = {}
        self.layer_names = {}
        self._handles = []

    def _register_hooks(self):
        for name, module in self.model.named_modules():
            self.layer_names[id(module)] = name
            if isinstance(module, nn.Linear):
                handle = module.weight.register_hook(self._make_grad_hook(name))
                self._handles.append(handle)
            # snntorch Leaky neurons are detected by attribute presence
            if hasattr(module, 'mem') and hasattr(module, 'threshold'):
                module_name = name or type(module).__name__
                handle = module.register_forward_hook(self._make_spike_hook(module_name))
                self._handles.append(handle)

    def _make_grad_hook(self, name):
        def hook(grad):
            if grad is not None:
                self.grad_norms[name] = grad.norm(2).item()
        return hook

    def _make_spike_hook(self, name):
        def hook(module, input, output):
            # output may be a tuple (spk, mem) in some snntorch versions
            spk = output[0] if isinstance(output, tuple) else output
            if spk is not None:
                self.spike_rates[name] = spk.mean().item()
                self.activation_stats[name] = {
                    'mean': spk.mean().item(),
                    'std': spk.std().item(),
                    'max': spk.max().item(),
                    'min': spk.min().item(),
                }
        return hook

    def summarize(self):
        """Return current diagnostics as a dictionary."""
        return {
            'grad_norms': self.grad_norms,
            'spike_rates': self.spike_rates,
            'activation_stats': self.activation_stats,
        }

    def to_json(self, path):
        """Save diagnostics to JSON file."""
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.summarize(), f, indent=2)

    def remove_hooks(self):
        """Remove all registered hooks."""
        for h in self._handles:
            h.remove()
        self._handles.clear()


def diagnose_step(model, inputs, targets, tokenizer, device='cpu'):
    """Run one training step and return diagnostics.

    Parameters
    ----------
    model : torch.nn.Module
        SNN LM model.
    inputs : torch.Tensor
        (batch, seq_len) token ids.
    targets : torch.Tensor
        (batch, seq_len) target ids.
    tokenizer : object
        Tokenizer with vocab_size attribute.
    device : str or torch.device

    Returns
    -------
    dict
        Loss, perplexity, and diagnostic statistics.
    """
    from snnai.modules.language.tokenizer import SpikeEncoder

    model.to(device)
    model.train()
    diag = SNNDiagnostic(model)
    encoder = SpikeEncoder(vocab_size=tokenizer.vocab_size, time_steps=20)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    x = encoder(inputs.to(device))
    targets = targets.to(device)
    optimizer.zero_grad()
    out = model(x)
    loss = nn.functional.cross_entropy(out.reshape(-1, tokenizer.vocab_size), targets.reshape(-1))
    loss.backward()
    optimizer.step()

    result = {
        'loss': loss.item(),
        'ppl': torch.exp(loss).item(),
        **diag.summarize(),
    }
    diag.remove_hooks()
    return result
