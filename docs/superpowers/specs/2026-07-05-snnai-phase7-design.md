# SNNAI Phase7 設計書：SNNAI Version 1.0（最初の完成版）

## 概要

SNNAI ロードマップの **Phase7：SNNAI Version1** を実装する。Phase0〜Phase6 までに構築したモジュールを統合し、ベンチマークとエネルギー推定により客観的な性能評価を行う。

## 目的

- 統合された SNNAI パイプラインをベンチマークタスクで評価する。
- 同規模パラメータの ANN（MLP）ベースラインと比較する。
- 推定エネルギー効率を測定するための指標を提供する。
- GitHub タグ `v1.0` を打ち、README を更新する。

## ベンチマーク

本フェーズではロードマップのベンチマーク段階 1〜2（MNIST / Fashion-MNIST 相当の合成タスク）を実装する。

### Synthetic Multi-Modal Benchmark

- 入力：4 チャネルの時系列データ（各チャネルは異なるクラスに対応）
- タスク：4 クラス分類
- Phase6 の統合パイプラインを使用し、線虫・ミツバチ・カラス・タコの各モジュールを組み合わせる。

## 比較対象

### MLP Baseline

- 同程度のパラメータ数を持つ多層パーセプトロン
- 入力を時系列平均化して使用

### SNNAI Integrated Pipeline

- Phase6 の `SNNAIPipeline` を拡張し、複数モジュールの出力を統合
- 各モジュールのスパイク数に基づいて最終クラスを決定

## エネルギー推定

- スパイク数に比例したエネルギー消費モデル
- ANN ベースラインとの比較指標として使用

## 実装方針

- `snnai/benchmarks/synthetic_benchmark.py`: 合成タスクと評価
- `snnai/benchmarks/baseline_mlp.py`: MLP ベースライン
- `snnai/benchmarks/energy_estimation.py`: エネルギー推定ユーティリティ
- `tests/test_phase7.py`: ベンチマークが実行され、SNNAI のスコアが 0 より大きく、ベースラインとの差が一定範囲内であることを確認
- `VERSION` を `v1.0` に更新
- `README.md` に v1.0 の成果を追記

## 次期ロードマップ作成条件

- Phase7 のベンチマーク結果を `docs/superpowers/specs/2026-07-05-snnai-phase7-design.md` に記録
- 次期ロードマップ `docs/roadmap_v2.md` を作成し、以下を含める：
  - v1.0 の振り返り
  - v2.0 の目標（MNIST/Fashion-MNIST 実データ、より大規模なモジュール統合、Atari 検討）
  - 改善すべき課題と優先度
