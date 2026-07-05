# SNNAI Phase1 実装計画

> **For agentic workers:** REQUIRED SUB-LEVEL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** SNN の基本要素（LIF, Izhikevich, STDP, LSM）を `snnai/core/` に実装し、pytest テストと Kaggle 動作確認を行う。

**Architecture:** snnTorch ベース。Brian2 は検証用テスト。既存の `snnai/core/neurons/lif.py` を拡張する。

**Tech Stack:** Python, snnTorch, Brian2, torch, numpy, pytest, Kaggle CLI MCP, GitHub MCP.

---

## ファイル構造

```
snnai/
├── core/
│   ├── neurons/
│   │   ├── lif.py              # 拡張
│   │   └── izhikevich.py
│   ├── synapses/
│   │   └── stdp.py
│   └── reservoir/
│       └── liquid_state_machine.py
└── tests/
    ├── test_lif.py
    ├── test_izhikevich.py
    ├── test_stdp.py
    └── test_lsm.py
```

---

## Task 1: LIF ニューロン拡張

**Files:**
- Modify: `snnai/core/neurons/lif.py`
- Create: `tests/test_lif.py`

- [ ] **Step 1: `snnai/core/neurons/lif.py` を拡張する**

```python
import torch
import snntorch as snn


def create_lif_neuron(beta=0.9, threshold=1.0):
    return snn.Leaky(beta=beta, threshold=threshold)


def simulate_lif(neuron, input_current, num_steps, reset=True):
    """Simulate a single or population of LIF neurons over time."""
    mem = neuron.init_leaky()
    if mem.dim() == 0:
        mem = torch.zeros(input_current.shape[1:])
    spikes = []
    mems = []
    for t in range(num_steps):
        spk, mem = neuron(input_current[t], mem)
        spikes.append(spk)
        mems.append(mem)
    return torch.stack(spikes), torch.stack(mems)
```

- [ ] **Step 2: `tests/test_lif.py` を作成する**

```python
import torch
from snnai.core.neurons.lif import create_lif_neuron, simulate_lif


def test_lif_spikes_with_high_input():
    neuron = create_lif_neuron(beta=0.9, threshold=1.0)
    current = torch.ones(20, 1) * 1.5
    spikes, _ = simulate_lif(neuron, current, 20)
    assert spikes.sum().item() > 0


def test_lif_no_spikes_with_low_input():
    neuron = create_lif_neuron(beta=0.9, threshold=1.0)
    current = torch.ones(20, 1) * 0.1
    spikes, _ = simulate_lif(neuron, current, 20)
    assert spikes.sum().item() == 0
```

- [ ] **Step 3: コミット**

---

## Task 2: Izhikevich ニューロン

**Files:**
- Create: `snnai/core/neurons/izhikevich.py`
- Create: `tests/test_izhikevich.py`

- [ ] **Step 1: Izhikevich モデルを実装する**

```python
import torch
import torch.nn as nn


class Izhikevich(nn.Module):
    def __init__(self, a=0.02, b=0.2, c=-65.0, d=8.0, threshold=30.0):
        super().__init__()
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.threshold = threshold

    def init_state(self, batch_size=1):
        v = torch.full((batch_size,), -65.0)
        u = torch.full((batch_size,), self.b * -65.0)
        return v, u

    def forward(self, I, v, u, dt=1.0):
        v = v + dt * (0.04 * v**2 + 5 * v + 140 - u + I)
        u = u + dt * self.a * (self.b * v - u)
        spike = (v >= self.threshold).float()
        v = torch.where(spike.bool(), torch.full_like(v, self.c), v)
        u = torch.where(spike.bool(), u + self.d, u)
        return spike, v, u
```

- [ ] **Step 2: プリセットを追加する**

```python
PRESETS = {
    'RS':  {'a': 0.02, 'b': 0.2,  'c': -65.0, 'd': 8.0},
    'IB':  {'a': 0.02, 'b': 0.2,  'c': -55.0, 'd': 4.0},
    'CH':  {'a': 0.02, 'b': 0.2,  'c': -50.0, 'd': 2.0},
    'FS':  {'a': 0.1,  'b': 0.2,  'c': -65.0, 'd': 2.0},
}


def create_izhikevich(preset='RS'):
    return Izhikevich(**PRESETS[preset])
```

- [ ] **Step 3: テストを作成しコミットする**

---

## Task 3: STDP

**Files:**
- Create: `snnai/core/synapses/stdp.py`
- Create: `tests/test_stdp.py`

- [ ] **Step 1: STDP を実装する**

```python
import torch


def stdp_update(pre_spikes, post_spikes, weights, A_plus=0.01, A_minus=0.01, tau=20.0):
    """Simple STDP weight update.

    pre_spikes, post_spikes: (time, batch, neurons) or (time, neurons)
    weights: (post, pre)
    """
    time_steps = pre_spikes.shape[0]
    dw = torch.zeros_like(weights)
    for t_post in range(time_steps):
        for t_pre in range(time_steps):
            dt = t_post - t_pre
            if dt == 0:
                continue
            elif dt > 0:
                factor = A_plus * torch.exp(-dt / tau)
            else:
                factor = -A_minus * torch.exp(dt / tau)
            # Vectorized over batch and neurons
            pre = pre_spikes[t_pre].float()
            post = post_spikes[t_post].float()
            dw += factor * (post.T @ pre)
    return weights + dw
```

- [ ] **Step 2: テストを作成しコミットする**

---

## Task 4: LSM クラス化

**Files:**
- Create: `snnai/core/reservoir/liquid_state_machine.py`
- Create: `tests/test_lsm.py`

- [ ] **Step 1: Phase0.5 の LSM ノートブックをクラス化する**

```python
import torch
import torch.nn as nn
import snntorch as snn
from sklearn.linear_model import RidgeClassifier


class LiquidStateMachine(nn.Module):
    def __init__(self, input_size, reservoir_size, output_size=None,
                 beta=0.9, threshold=1.0, sparsity=0.1, spectral_radius=0.9):
        super().__init__()
        self.input_size = input_size
        self.reservoir_size = reservoir_size
        self.lif = snn.Leaky(beta=beta, threshold=threshold, learn_threshold=False)
        self.W_in = nn.Linear(input_size, reservoir_size, bias=False)
        self.W_rec = nn.Linear(reservoir_size, reservoir_size, bias=False)
        # Initialize random sparse recurrent weights
        with torch.no_grad():
            w = torch.randn(reservoir_size, reservoir_size)
            mask = torch.rand(reservoir_size, reservoir_size) < sparsity
            w = w * mask.float()
            w.fill_diagonal_(0)
            eigvals = torch.linalg.eigvals(w)
            sr = torch.max(torch.abs(eigvals)).item()
            if sr > 0:
                w = w * (spectral_radius / sr)
            self.W_rec.weight.copy_(w)
            self.W_in.weight.normal_(0, 0.5)

    def forward(self, x):
        # x: (time, batch, input_size)
        mem = self.lif.init_leaky()
        if mem.dim() == 0:
            mem = torch.zeros(x.shape[1], self.reservoir_size)
        spikes = []
        for t in range(x.shape[0]):
            current = self.W_in(x[t]) + self.W_rec(mem)
            spk, mem = self.lif(current, mem)
            spikes.append(spk)
        return torch.stack(spikes)
```

- [ ] **Step 2: テストを作成しコミットする**

---

## Task 5: Kaggle 動作確認

**Files:**
- Create: `environment/phase1_demo.ipynb`

- [ ] **Step 1: Phase1 デモノートブックを作成する**

Cells:
1. Install `pip install -q -e .` or add repo path
2. `import snnai`
3. Run each core component demo

- [ ] **Step 2: Kaggle 上で実行確認**

- [ ] **Step 3: コミットと GitHub push**

---

## Task 6: GitHub タグ v0.2 を打つ

- [ ] **Step 1: タグを作成・push する**

```bash
git tag -a v0.2 -m "Phase1 complete: LIF, Izhikevich, STDP, LSM core modules"
git push origin v0.2
```

---

## Self-Review Checklist

- [ ] Spec coverage: 全コンポーネントが計画に含まれている
- [ ] No placeholders: 各 Step に具体的なコードがある
- [ ] Type consistency: 関数名・ファイルパスが設計書と一致

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-07-05-snnai-phase1-plan.md`.

Execution options:
1. **Subagent-Driven (recommended)** — dispatch a fresh subagent per task.
2. **Inline Execution** — execute tasks in this session.

Proceeding with inline execution per user instruction to use self-judgment.
