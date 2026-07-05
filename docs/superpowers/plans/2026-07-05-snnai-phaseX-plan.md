# SNNAI Phase X 実装計画

**Goal:** Bio-NAS（生物制約付きアーキテクチャ探索）を実装し、合成タスクで直列構造を上回る接続パターンを探索する。

## Task 1: 探索空間

**Files:**
- Create: `snnai/bio_nas/__init__.py`
- Create: `snnai/bio_nas/search_space.py`

- [ ] `Architecture` クラス：モジュール配置と接続を表現
- [ ] `serial`, `hub`, `parallel` の3つのテンプレートを生成
- [ ] 生物制約チェック関数を実装

## Task 2: 評価器

**Files:**
- Create: `snnai/bio_nas/evaluator.py`

- [ ] 合成多モーダル分類タスクを定義
- [ ] アーキテクチャをグラフ順に実行し、最終出力を取得
- [ ] 分類精度を評価指標として返す

## Task 3: 進化的探索

**Files:**
- Create: `snnai/bio_nas/evolution_search.py`

- [ ] 母集団を初期化（テンプレート＋ランダム突然変異）
- [ ] 世代ごとに評価・選択・突然変異を繰り返す
- [ ] 最良アーキテクチャを返す

## Task 4: テスト

**Files:**
- Create: `tests/test_bio_nas.py`

- [ ] 探索空間の制約チェックが機能すること
- [ ] 進化探索が直列構造より高い（または同等の）スコアを見つけること

## Task 5: GitHub タグ v0.6

- [ ] コミット・push 後に v0.6 タグを打つ
