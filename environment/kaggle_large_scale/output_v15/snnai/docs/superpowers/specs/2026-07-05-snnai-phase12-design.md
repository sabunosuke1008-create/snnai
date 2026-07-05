# SNNAI Phase12 設計書：Atari 実験・SNNAI v2.0 完成

## 概要

SNNAI v2 ロードマップの **Phase12: Atari 実験・SNNAI v2.0 完成** を実装する。

## 目的

- 簡易的な Atari 風強化学習タスクで SNN エージェントを評価する。
- SNNAI v2.0 として最初の完成版をリリースする。
- GitHub タグ `v2.0` を打つ。

## 構成

- `snnai/benchmarks/atari_env.py`: 簡易ピンポン/キャッチ環境
- `snnai/benchmarks/atari_agent.py`: SNN エージェント（Phase3 の R-STDP を流用）
- `snnai/benchmarks/atari_trainer.py`: 学習ループ
- `tests/test_phase12.py`: 学習曲線の改善を確認
- `VERSION` → `v2.0.0`
- `README.md` に v2.0 成果を追記

## 環境

簡易ピンポン（Pong-like）:
- 5x5 グリッド
- パドルは下端を左右に移動
- ボールが上から下に落ちてくる
- パドルでボールを受けると正報酬、ミスで負報酬

## エージェント

- Phase3 の `SNNAgent` を拡張
- 状態はボールとパドルの相対位置
- R-STDP で方策を学習

## 評価指標

- エピソードあたりの報酬が増加すること
- v2.0 タグが打てること
