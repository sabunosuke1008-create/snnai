# SNNAI Phase11 実装計画

**Goal:** 大規模化・GPU 最適化・チェックポイント機構を実装する。

## Task 1: ユーティリティ
- [ ] `snnai/utils/device.py` を作成（CUDA/MPS/CPU 自動選択）
- [ ] `snnai/utils/checkpoint.py` を作成（保存・読み込み）

## Task 2: 大規模モデル
- [ ] `snnai/benchmarks/large_mnist_snn.py` を作成

## Task 3: テスト
- [ ] `tests/test_phase11.py` を作成
- [ ] checkpoint ラウンドトリップを確認
- [ ] デバイス選択を確認

## Task 4: Kaggle T4 例
- [ ] `environment/kaggle_mnist_t4/` を作成し、T4 GPU 指定の kernel-metadata を含める

## Task 5: GitHub タグ v1.4
- [ ] コミット・push 後に v1.4 タグを打つ
