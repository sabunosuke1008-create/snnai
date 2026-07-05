# SNNAI Phase3 実装計画

> **For agentic workers:** REQUIRED SUB-LEVEL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** ミツバチモジュール（空間認識・R-STDP 強化学習）を実装し、グリッドワールドでランダム方策を上回る学習曲線を示す。

**Architecture:** snnTorch + PyTorch。状態を place cell にエンコードし、SNN 方策ネットワークを R-STDP で学習。

---

## Task 1: グリッドワールド環境

**Files:**
- Create: `snnai/modules/honeybee/gridworld.py`

- [ ] **Step 1: GridWorld クラスを実装**

```python
import numpy as np

class GridWorld:
    def __init__(self, size=5, start=(0,0), goal=(4,4), obstacles=None):
        self.size = size
        self.start = start
        self.goal = goal
        self.obstacles = set(obstacles or [])
        self.reset()

    def reset(self):
        self.pos = self.start
        return self.pos

    def step(self, action):
        # action: 0=up, 1=down, 2=left, 3=right
        moves = [(-1,0), (1,0), (0,-1), (0,1)]
        next_pos = (self.pos[0]+moves[action][0], self.pos[1]+moves[action][1])
        if (0 <= next_pos[0] < self.size and 0 <= next_pos[1] < self.size
                and next_pos not in self.obstacles):
            self.pos = next_pos
        reward = 10.0 if self.pos == self.goal else -0.1
        done = self.pos == self.goal
        return self.pos, reward, done
```

---

## Task 2: Place Cell Encoding

**Files:**
- Create: `snnai/modules/honeybee/spatial_encoding.py`

- [ ] **Step 1: place_cell_encode を実装**

```python
import numpy as np
import torch


def place_cell_encode(pos, size=5, n_cells=25, sigma=0.8):
    """Encode (x,y) position into place-cell firing probabilities."""
    xs = np.linspace(0, size-1, int(np.sqrt(n_cells)))
    ys = np.linspace(0, size-1, int(np.sqrt(n_cells)))
    centers = [(x, y) for x in xs for y in ys]
    rates = []
    for cx, cy in centers:
        d2 = (pos[0]-cx)**2 + (pos[1]-cy)**2
        rates.append(np.exp(-d2 / (2 * sigma**2)))
    rates = np.array(rates)
    rates = rates / (rates.max() + 1e-8)
    return torch.tensor(rates, dtype=torch.float32)
```

---

## Task 3: R-STDP Update

**Files:**
- Create: `snnai/modules/honeybee/r_stdp.py`

- [ ] **Step 1: reward modulated STDP を実装**

```python
import torch
from snnai.core.synapses.stdp import stdp_update


def r_stdp_update(pre_spikes, post_spikes, weights, reward, A_plus=0.01, A_minus=0.01, tau=20.0):
    dw = stdp_update(pre_spikes, post_spikes, torch.zeros_like(weights),
                     A_plus=A_plus, A_minus=A_minus, tau=tau) - weights
    return weights + reward * dw
```

---

## Task 4: SNN 方策エージェント

**Files:**
- Create: `snnai/modules/honeybee/gridworld_agent.py`

- [ ] **Step 1: SNNAgent クラスを実装**

Architecture: place cells -> hidden LIF -> output LIF (4 actions)
Use softmax-like spike count for action selection.

---

## Task 5: 学習曲線テスト

**Files:**
- Create: `tests/test_honeybee_learning.py`

- [ ] **Step 1: 学習曲線が改善することを確認**

Compare first 10 episodes vs last 10 episodes average reward.

---

## Task 6: GitHub タグ v0.3

- [ ] **Step 1: コミット・push 後に v0.3 タグを打つ**
