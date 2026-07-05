# SNNAI Phase10 設計書：End-to-End 統合学習

## 概要

SNNAI v2 ロードマップの **Phase10: End-to-End 統合学習** を実装する。

## 目的

- Phase6 の統合パイプラインを勾配通過可能に拡張する。
- 複数モジュールを surrogate gradient 法で end-to-end に学習できる基盤を提供する。
- 簡易タスクで統合学習が性能向上に寄与することを示す。

## 構成

- `snnai/integration/trainable_pipeline.py`: 勾配可能な統合パイプライン
- `tests/test_phase10.py`: 統合学習テスト

## 設計

### TrainablePipeline

- 2 つのモジュール（ReflexModule と WorkingMemory）を直列に接続
- 間に学習可能な `EncodingBridge` を配置
- 最終出力は線形読み出し層で分類
- 全体を CrossEntropyLoss で学習

### 勾配フロー

- snnTorch の `snn.Leaky` は surrogate gradient に対応しているため、スパイクを通じて勾配が流れる
- Bridge は線形変換 + Poisson サンプリングだが、学習時は連続的な発火率を使用
- 各モジュールのパラメータも同時に更新可能

## 評価指標

- 統合モデルの学習が収束すること
- テスト精度がランダムより有意に高いこと
