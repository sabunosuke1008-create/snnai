# SNNAI Phase11 設計書：大規模化・GPU 最適化・チェックポイント

## 概要

SNNAI v2 ロードマップの **Phase11: 大規模化・GPU 最適化・チェックポイント** を実装する。

## 目的

- 長時間学習における checkpoint 保存・再開機構を提供する。
- GPU（CUDA/MPS）自動選択ユーティリティを整備する。
- より大きな SNN モデル（LargeMNISTSNN）を実装し、スケールアップを実証する。
- Kaggle kernel-metadata で T4 GPU を指定する例を提供する。

## 構成

- `snnai/utils/device.py`: 最適なデバイスを自動選択
- `snnai/utils/checkpoint.py`: モデル・オプティマイザ・エポックの保存・読み込み
- `snnai/benchmarks/large_mnist_snn.py`: 拡張版 MNIST SNN（より多くのチャネル・ユニット）
- `tests/test_phase11.py`: checkpoint ラウンドトリップ・デバイス選択テスト
- `environment/kaggle_mnist_t4/`: T4 GPU 指定の Kaggle ノートブック例

## Checkpoint

```python
checkpoint.save(model, optimizer, epoch, path)
model, optimizer, epoch = checkpoint.load(path)
```

## 大規模化

- `LargeMNISTSNN`: 畳み込みチャネル数と全結合ユニット数を増やしたモデル
- Phase8 の `MNISTSNN` を継承・拡張

## GPU 最適化

- `auto_select_device()` で CUDA / MPS / CPU を自動選択
- Kaggle `kernel-metadata.json` に `"machine_shape": "NvidiaTeslaT4Highmem"` を設定した例
