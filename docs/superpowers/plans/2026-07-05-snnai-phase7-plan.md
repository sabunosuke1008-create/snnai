# SNNAI Phase7 実装計画

**Goal:** SNNAI v1.0 を完成させ、ベンチマーク・ベースライン比較・エネルギー推定を実装し、次期ロードマップを作成する。

## Task 1: ベンチマーク基盤

**Files:**
- Create: `snnai/benchmarks/__init__.py`
- Create: `snnai/benchmarks/synthetic_benchmark.py`
- Create: `snnai/benchmarks/baseline_mlp.py`

- [ ] 合成 4 クラス分類タスクを定義
- [ ] MLP ベースラインを実装
- [ ] SNNAI 統合パイプラインを使った分類器を実装

## Task 2: エネルギー推定

**Files:**
- Create: `snnai/benchmarks/energy_estimation.py`

- [ ] スパイク数に基づく簡易エネルギー推定
- [ ] ANN との比較指標を出力

## Task 3: テスト

**Files:**
- Create: `tests/test_phase7.py`

- [ ] ベンチマークが実行可能であること
- [ ] SNNAI の精度が 0 より大きいこと
- [ ] MLP ベースラインも実行可能であること

## Task 4: リリース準備

- [ ] `VERSION` を `v1.0` に更新
- [ ] `README.md` に v1.0 の成果を追記

## Task 5: GitHub タグ v1.0

- [ ] コミット・push 後に v1.0 タグを打つ

## Task 6: 次期ロードマップ作成

- [ ] `docs/roadmap_v2.md` を作成
