# SNNAI Phase9 実装計画

**Goal:** N-MNIST 形式のイベントデータ対応を実装する。

## Task 1: 疑似イベント生成
- [ ] `snnai/benchmarks/nmnist_loader.py` を作成
- [ ] MNIST フレームから ON/OFF イベントを生成する関数を実装

## Task 2: イベント → スパイク変換
- [ ] `snnai/benchmarks/event_utils.py` を作成
- [ ] イベント列を時系列スパイクテンソルに変換

## Task 3: テスト
- [ ] `tests/test_phase9.py` を作成
- [ ] イベント生成とスパイク変換の形状を確認

## Task 4: GitHub タグ v1.2
- [ ] コミット・push 後に v1.2 タグを打つ
