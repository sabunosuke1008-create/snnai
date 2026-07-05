# SNNAI Phase0.5 実装計画

> **For agentic workers:** REQUIRED SUB-LEVEL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Phase0.5 の既存研究再現（LSM, LNN, Bee-Navigation）を実装し、Kaggle 上で検証して `v0.1` タグを打つ。

**Architecture:** snnTorch / Brian2 を使い、論文/公式実装を Kaggle Notebook 形式で再現。各再現は独立した `.ipynb` として作成し、結果を `reproduction_results.md` にまとめる。

**Tech Stack:** Python, snnTorch, Brian2, numpy, matplotlib, scikit-learn, Kaggle CLI MCP, GitHub MCP.

---

## ファイル構造

```
snnai/
├── reproductions/
│   ├── __init__.py
│   ├── lsm_reproduction.ipynb
│   ├── lnn_reproduction.ipynb
│   ├── bee_navigation_reproduction.ipynb
│   └── docs/
│       └── reproduction_results.md
└── docs/
    └── papers.md
```

---

## Task 1: 参考情報の整理

**Files:**
- Create: `docs/papers.md`

- [ ] **Step 1: 必要な論文・公式実装リンクを収集し `docs/papers.md` にまとめる**

```markdown
# 参考文献・実装リンク

## LSM
- Maass et al. (2002) "Real-time computing without stable states"
- snnTorch tutorial: https://snntorch.readthedocs.io/en/latest/tutorials/tutorial_6.html

## LNN
- Hasani et al. (2020) "Liquid Time-constant Networks"
- Official code: https://github.com/raminmh/liquid_time_constant_networks

## Bee Navigation
- Baddeley et al. path integration models
- 簡易実装参考: https://github.com/...
```

- [ ] **Step 2: 変更をコミットする**

```bash
git add docs/papers.md
git commit -m "docs: add Phase0.5 reference papers and implementations"
git push
```

---

## Task 2: LSM 再現ノートブック作成

**Files:**
- Create: `reproductions/lsm_reproduction.ipynb`

- [ ] **Step 1: ノートブックを作成する**

Cells:
1. Markdown: LSM 再現の目的
2. Code: 擬似時系列データ生成（2 クラスのパターン）
3. Code: 入力スパイク列へのエンコーディング
4. Code: snnTorch でリザバー層（LIF, ランダム結合）を構築
5. Code: リザバー状態を収集し scikit-learn で線形分類器を学習
6. Code: テスト精度を表示
7. Code: リザバー状態を可視化

- [ ] **Step 2: Kaggle 上で実行確認**

```bash
python -m kaggle kernels push -p reproductions/lsm
```

（MCP 使用の場合: `kaggle_cli_mcp_kernels_push`）

- [ ] **Step 3: 変更をコミットする**

```bash
git add reproductions/lsm_reproduction.ipynb
git commit -m "feat: add LSM reproduction notebook"
```

---

## Task 3: LNN 再現ノートブック作成

**Files:**
- Create: `reproductions/lnn_reproduction.ipynb`

- [ ] **Step 1: ノートブックを作成する**

Cells:
1. Markdown: LNN / Liquid Time-constant Network 再現の目的
2. Code: 正弦波時系列データ生成
3. Code: LTC ニューロンセルの実装（Hasani et al. 式を参考）
4. Code: snnTorch で LNN モデルを構築
5. Code: 時系列予測の学習ループ
6. Code: 予測誤差（MSE）を表示
7. Code: 予測結果をプロット

- [ ] **Step 2: Kaggle 上で実行確認**

- [ ] **Step 3: 変更をコミットする**

```bash
git add reproductions/lnn_reproduction.ipynb
git commit -m "feat: add LNN reproduction notebook"
```

---

## Task 4: Bee Navigation 再現ノートブック作成

**Files:**
- Create: `reproductions/bee_navigation_reproduction.ipynb`

- [ ] **Step 1: ノートブックを作成する**

Cells:
1. Markdown: ミツバチ経路積分モデルの再現目的
2. Code: 2D グリッド上でランダム walk を生成
3. Code: 速度・方向情報をスパイク列にエンコーディング
4. Code: Brian2 or snnTorch で積分器ニューロンを実装
5. Code: 推定位置と真の位置を比較
6. Code: 到達誤差を表示・可視化

- [ ] **Step 2: Kaggle 上で実行確認**

- [ ] **Step 3: 変更をコミットする**

```bash
git add reproductions/bee_navigation_reproduction.ipynb
git commit -m "feat: add bee navigation reproduction notebook"
```

---

## Task 5: 再現結果のまとめ

**Files:**
- Create: `reproductions/docs/reproduction_results.md`

- [ ] **Step 1: 結果をまとめる**

```markdown
# Phase0.5 再現結果

## LSM
| Metric | Reference | Ours | Status |
|---|---|---|---|
| Accuracy | ~95% | XX% | TBD |

## LNN
| Metric | Reference | Ours | Status |
|---|---|---|---|
| MSE | 0.01 | XX | TBD |

## Bee Navigation
| Metric | Reference | Ours | Status |
|---|---|---|---|
| Endpoint error | < 5% | XX% | TBD |
```

- [ ] **Step 2: 変更をコミットする**

```bash
git add reproductions/docs/reproduction_results.md
git commit -m "docs: add Phase0.5 reproduction results summary"
```

---

## Task 6: GitHub タグ v0.1 を打つ

**Files:**
- Modify: リモート GitHub リポジトリのタグ

- [ ] **Step 1: ローカルでタグを作成・push する**

```bash
git tag -a v0.1 -m "Phase0.5 complete: LSM, LNN, Bee-Nav reproductions"
git push origin v0.1
```

（MCP にタグ作成ツールがあればそちらを使用）

---

## Self-Review Checklist

- [ ] Spec coverage: 全再現対象が計画に含まれている
- [ ] No placeholders: 各 Step に具体的な内容がある
- [ ] Type consistency: ファイルパス・関数名が設計書と一致

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-07-05-snnai-phase0.5-plan.md`.

Execution options:
1. **Subagent-Driven (recommended)** — dispatch a fresh subagent per task.
2. **Inline Execution** — execute tasks in this session.

Proceeding with inline execution per user instruction to use self-judgment.
