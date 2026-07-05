"""Distributed parallel processing module.

Mimics octopus-like independent arm brains: several small SNN sub-modules
process distinct sensory channels in parallel, and an integration layer
combines their outputs into a single decision.
"""
from concurrent.futures import ThreadPoolExecutor

import torch
import torch.nn as nn
import snntorch as snn


class SubModule(nn.Module):
    """Independent sensory-processing sub-network.

    Maps a single sensory channel to an ``active`` / ``inactive`` spike
    representation.
    """

    def __init__(self, input_size=2, hidden_size=4, beta=0.9, threshold=1.0):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.fc = nn.Linear(input_size, hidden_size, bias=False)
        self.lif = snn.Leaky(beta=beta, threshold=threshold, learn_threshold=False)
        with torch.no_grad():
            self.fc.weight.normal_(0, 0.6)

    def forward(self, x):
        """Run the sub-module.

        Parameters
        ----------
        x : torch.Tensor
            Input of shape (time, batch, input_size).

        Returns
        -------
        torch.Tensor
            Output spikes of shape (time, batch, hidden_size).
        """
        time_steps, batch_size, _ = x.shape
        mem = torch.zeros(batch_size, self.hidden_size)
        spikes = []
        for t in range(time_steps):
            current = self.fc(x[t])
            spk, mem = self.lif(current, mem)
            spikes.append(spk)
        return torch.stack(spikes)


class DistributedModule(nn.Module):
    """Collection of independent sub-modules executed in parallel.

    Parameters
    ----------
    n_modules : int
        Number of parallel sub-modules.
    input_size : int
        Input dimensionality for each sub-module.
    hidden_size : int
        Output dimensionality for each sub-module.
    """

    def __init__(self, n_modules=2, input_size=2, hidden_size=4, max_workers=None):
        super().__init__()
        self.n_modules = n_modules
        self.submodules = nn.ModuleList(
            [SubModule(input_size, hidden_size) for _ in range(n_modules)]
        )
        self.max_workers = max_workers

    def _run_one(self, args):
        idx, x = args
        return idx, self.submodules[idx](x)

    def forward(self, inputs):
        """Run sub-modules in parallel.

        Parameters
        ----------
        inputs : list[torch.Tensor]
            List of ``n_modules`` input tensors, each
            (time, batch, input_size).

        Returns
        -------
        list[torch.Tensor]
            List of sub-module spike outputs.
        """
        if self.max_workers is None or self.max_workers == 1:
            return [self.submodules[i](inputs[i]) for i in range(self.n_modules)]

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            results = list(executor.map(self._run_one, enumerate(inputs)))
        results.sort(key=lambda r: r[0])
        return [r[1] for r in results]


class MultiModalClassifier(nn.Module):
    """Integrate parallel sub-module outputs into a categorical decision.

    Output classes:
        0 = safe, 1 = attention, 2 = warning
    """

    def __init__(self, n_modules=2, hidden_size=4, n_classes=3):
        super().__init__()
        self.n_modules = n_modules
        self.hidden_size = hidden_size
        self.fc = nn.Linear(n_modules * hidden_size, n_classes, bias=False)
        with torch.no_grad():
            self.fc.weight.normal_(0, 0.5)

    def forward(self, submodule_outputs):
        """Classify from parallel sub-module spike counts.

        Parameters
        ----------
        submodule_outputs : list[torch.Tensor]
            Output spikes from each sub-module.

        Returns
        -------
        torch.Tensor
            Logits of shape (batch, n_classes).
        """
        rates = [out.mean(dim=0) for out in submodule_outputs]  # (batch, hidden)
        features = torch.cat(rates, dim=1)
        return self.fc(features)


class OctopusNetwork(nn.Module):
    """Full octopus-inspired distributed multi-modal network."""

    def __init__(self, n_modules=2, input_size=2, hidden_size=4, n_classes=3, max_workers=None):
        super().__init__()
        self.distributed = DistributedModule(n_modules, input_size, hidden_size, max_workers)
        self.classifier = MultiModalClassifier(n_modules, hidden_size, n_classes)

    def forward(self, inputs):
        outputs = self.distributed(inputs)
        return self.classifier(outputs)
