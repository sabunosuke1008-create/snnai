# SNNAI Phase10 実装計画

**Goal:** End-to-End で学習可能な統合パイプラインを実装する。

## Task 1: 学習可能パイプライン
- [ ] `snnai/integration/trainable_pipeline.py` を作成
- [ ] ReflexModule + Bridge + WorkingMemory + readout を統合
- [ ] surrogate gradient による学習関数を実装

## Task 2: テスト
- [ ] `tests/test_phase10.py` を作成
- [ ] 学習が実行され、精度 > ランダムを確認

## Task 3: GitHub タグ v1.3
- [ ] コミット・push 後に v1.3 タグを打つ
