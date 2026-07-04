# SNNAI Phase0 実装計画

> **For agentic workers:** REQUIRED SUB-LEVEL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** GitHub リポジトリ `snnai` を作成し、Kaggle Notebook 用の環境構築一式を含む Phase0 完了状態を作る。

**Architecture:** ロードマップに従い、メイン学習基盤は snnTorch、生物学的検証は Brian2、テストは pytest。Kaggle CLI で notebook/dataset を push し、GitHub MCP でリポジトリを管理する。

**Tech Stack:** Python, PyTorch, snnTorch, Brian2, Norse, BindsNET, pytest, Kaggle CLI, GitHub MCP.

---

## ファイル構造

```
snnai/
├── README.md
├── VERSION
├── requirements.txt
├── .gitignore
├── docs/
│   ├── roadmap.md
│   └── superpowers/
│       ├── specs/2026-07-05-snnai-phase0-design.md
│       └── plans/2026-07-05-snnai-phase0-plan.md
├── environment/
│   ├── kaggle_setup.ipynb
│   └── verify_imports.py
├── snnai/
│   ├── __init__.py
│   └── core/
│       ├── __init__.py
│       └── neurons/
│           ├── __init__.py
│           └── lif.py
└── tests/
    └── test_environment.py
```

---

## Task 1: GitHub リポジトリ作成

**Files:**
- Create: リモート GitHub リポジトリ `snnai`

- [ ] **Step 1: GitHub リポジトリを作成する**

Use MCP `github_create_repository`:
- name: `snnai`
- description: `Spiking Neural Network based AI - bio-inspired hybrid SNN modules`
- private: false
- autoInit: true

Expected result: Repository URL `https://github.com/gihuhi/snnai` created.

- [ ] **Step 2: ローカル git リポジトリを初期化し、リモートを追加する**

```bash
cd "C:\Users\otame\Downloads\SNN AI"
git init -b main
git remote add origin https://github.com/gihuhi/snnai.git
```

Expected result: `git remote -v` shows origin.

---

## Task 2: 基本ファイル追加

**Files:**
- Create: `.gitignore`
- Create: `VERSION`
- Create: `requirements.txt`
- Create: `README.md`

- [ ] **Step 1: `.gitignore` を書く**

```gitignore
# Python
__pycache__/
*.py[cod]
*.egg-info/
.ipynb_checkpoints/
*.log

# Virtual environments
venv/
.env/

# Kaggle
.kaggle/
*.kaggle.json

# OS
.DS_Store
Thumbs.db

# IDEs
.vscode/
.idea/

# Model artifacts
*.pth
*.pt
checkpoints/
```

- [ ] **Step 2: `VERSION` を書く**

```
v0.1.0-dev
```

- [ ] **Step 3: `requirements.txt` を書く**

```text
# Core
numpy==1.26.4
torch==2.3.0

# SNN frameworks
snntorch==0.9.4
brian2==2.7.0
norse==1.1.0
bindsnet==0.3.3

# Training / utils
matplotlib==3.9.0
jupyter==1.0.0

# Testing
pytest==8.2.0
```

- [ ] **Step 4: `README.md` を書く**

```markdown
# SNNAI

Spiking Neural Network based AI — 線虫・ミツバチ・カラス・タコなど複数の生物の脳構造を参考にした、省電力・モジュラー型 SNN の研究プロジェクト。

## ロードマップ

詳細は [docs/roadmap.md](docs/roadmap.md) を参照。

## クイックスタート

### ローカル

```bash
pip install -r requirements.txt
pytest tests/
```

### Kaggle

`environment/kaggle_setup.ipynb` を Kaggle Notebook としてインポートして実行してください。

## ライセンス

MIT
```

- [ ] **Step 5: 変更をコミットする**

```bash
git add .gitignore VERSION requirements.txt README.md docs/
git commit -m "chore: initialize Phase0 repository structure"
```

---

## Task 3: Kaggle セットアップファイル作成

**Files:**
- Create: `environment/kaggle_setup.ipynb`
- Create: `environment/verify_imports.py`

- [ ] **Step 1: `environment/verify_imports.py` を書く**

```python
"""環境検証スクリプト。ローカルでも Kaggle でも実行可能。"""
import sys


def check_import(module_name):
    try:
        __import__(module_name)
        print(f"[OK] {module_name}")
        return True
    except Exception as e:
        print(f"[NG] {module_name}: {e}")
        return False


def main():
    print("SNNAI environment verification")
    print("=" * 40)

    modules = ["torch", "snntorch", "brian2", "norse", "bindsnet", "numpy", "matplotlib", "pytest"]
    results = {m: check_import(m) for m in modules}

    try:
        import torch
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA device count: {torch.cuda.device_count()}")
            print(f"CUDA device name: {torch.cuda.get_device_name(0)}")
    except Exception as e:
        print(f"CUDA check failed: {e}")

    if all(results.values()):
        print("\nAll imports OK.")
        sys.exit(0)
    else:
        print("\nSome imports failed.")
        sys.exit(1)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: `environment/kaggle_setup.ipynb` を書く**

Notebook cells:
1. Markdown: `# SNNAI Kaggle Setup`
2. Code: `!pip install -q -r /kaggle/input/snnai-requirements/requirements.txt` （または `!pip install -q snntorch==0.9.4 brian2==2.7.0 ...`）
3. Code: `%run /kaggle/working/verify_imports.py` or inline verification
4. Code: single LIF neuron spike with snnTorch
5. Code: single LIF neuron spike with Brian2

Use `bash` to create the `.ipynb` JSON directly or use `jupyter nbformat` Python package if available.

Simplified inline version (full JSON not shown here): the notebook installs libraries, runs verify_imports.py, and prints LIF spike counts from both frameworks.

- [ ] **Step 3: 変更をコミットする**

```bash
git add environment/
git commit -m "feat: add Kaggle setup notebook and environment verification"
```

---

## Task 4: snnai パッケージ雛形

**Files:**
- Create: `snnai/__init__.py`
- Create: `snnai/core/__init__.py`
- Create: `snnai/core/neurons/__init__.py`
- Create: `snnai/core/neurons/lif.py`

- [ ] **Step 1: `__init__.py` ファイルを作成する**

All empty except `snnai/__init__.py` which sets `__version__`:

```python
__version__ = "0.1.0-dev"
```

- [ ] **Step 2: `snnai/core/neurons/lif.py` を書く**

```python
"""Minimal LIF neuron utilities for Phase0 validation."""
import torch
import snntorch as snn


def create_lif_neuron(beta=0.9, threshold=1.0):
    """Return a simple snnTorch Leaky integrate-and-fire neuron."""
    return snn.Leaky(beta=beta, threshold=threshold)


def lif_spike_count(mem, spike):
    """Return total spike count from a spike tensor."""
    return spike.sum().item()
```

- [ ] **Step 3: 変更をコミットする**

```bash
git add snnai/
git commit -m "feat: add snnai package skeleton with minimal LIF neuron"
```

---

## Task 5: 環境テスト

**Files:**
- Create: `tests/test_environment.py`

- [ ] **Step 1: テストを書く**

```python
"""Environment sanity tests for Phase0."""
import torch
import snntorch
import brian2
import norse
import bindsnet


def test_imports():
    assert torch is not None
    assert snntorch is not None
    assert brian2 is not None
    assert norse is not None
    assert bindsnet is not None


def test_lif_neuron_spikes():
    from snnai.core.neurons.lif import create_lif_neuron

    lif = create_lif_neuron(beta=0.9, threshold=1.0)
    mem = lif.init_leaky()
    spikes = []
    for _ in range(20):
        current = torch.ones(1) * 1.5
        spk, mem = lif(current, mem)
        spikes.append(spk.item())
    assert sum(spikes) > 0, "LIF neuron should spike with constant high input"
```

- [ ] **Step 2: テストを実行する**

```bash
pytest tests/test_environment.py -v
```

Expected: all tests pass.

- [ ] **Step 3: 変更をコミットする**

```bash
git add tests/
git commit -m "test: add Phase0 environment sanity tests"
```

---

## Task 6: GitHub へ push

**Files:**
- Modify: リモートブランチ `main`

- [ ] **Step 1: main ブランチを push する**

```bash
git push -u origin main
```

Expected: GitHub repository shows all files.

---

## Task 7: Kaggle へ notebook/dataset push（オプションだが推奨）

**Files:**
- Modify: Kaggle 上の notebook/kernel

- [ ] **Step 1: Kaggle dataset として requirements を push する**

```bash
mkdir -p kaggle_dataset
cp requirements.txt kaggle_dataset/
cp environment/verify_imports.py kaggle_dataset/
python -m kaggle datasets init -p kaggle_dataset
# edit kaggle_dataset/dataset-metadata.json title/slug
python -m kaggle datasets create -p kaggle_dataset
```

- [ ] **Step 2: Kaggle notebook として kaggle_setup.ipynb を push する**

```bash
python -m kaggle kernels init -p environment
# edit environment/kernel-metadata.json
python -m kaggle kernels push -p environment
```

- [ ] **Step 3: 不要になった `kaggle_dataset/` 作業ディレクトリを削除する**

```bash
rm -rf kaggle_dataset
```

Note: deletion is safe because files are already committed in `environment/` and pushed to GitHub.

---

## Self-Review Checklist

- [ ] Spec coverage: all Phase0 goals are covered.
- [ ] No placeholders:
 every step has concrete code or command.
- [ ] Type consistency: function names match across files.

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-07-05-snnai-phase0-plan.md`.

Execution options:
1. **Subagent-Driven (recommended)** — dispatch a fresh subagent per task.
2. **Inline Execution** — execute tasks in this session.

Proceeding with inline execution per user instruction to use self-judgment.
