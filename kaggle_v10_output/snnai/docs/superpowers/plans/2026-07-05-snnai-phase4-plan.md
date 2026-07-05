# SNNAI Phase4 実装計画

**Goal:** カラスモジュール（作業記憶）を実装し、Delayed Match-to-Sample タスクで一定の正解率を示す。

## Task 1: 作業記憶モジュール

**Files:**
- Create: `snnai/modules/crow/__init__.py`
- Create: `snnai/modules/crow/working_memory.py`

- [ ] 2 つの記憶ニューロン（A/B）を持つ小さな再帰 SNN を実装
- [ ] 自己興奮 + 相互抑制により遅延期間中に活動を維持
- [ ] 一致 / 不一致判定を行う出力層を追加

## Task 2: DMS タスクテスト

**Files:**
- Create: `tests/test_delayed_match.py`

- [ ] cue A -> delay -> probe A で「一致」出力が優位
- [ ] cue A -> delay -> probe B で「不一致」出力が優位
- [ ] 正解率 > 80% を目標

## Task 3: GitHub タグ v0.4

- [ ] コミット・push 後に v0.4 タグを打つ
