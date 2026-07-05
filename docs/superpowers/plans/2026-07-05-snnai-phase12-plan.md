# SNNAI Phase12 実装計画

**Goal:** Atari 風簡易タスクで SNN エージェントを学習させ、v2.0 をリリースする。

## Task 1: Atari 風環境
- [ ] `snnai/benchmarks/atari_env.py` を作成
- [ ] 簡易ピンポン環境を実装

## Task 2: Atari エージェント
- [ ] `snnai/benchmarks/atari_agent.py` を作成
- [ ] Phase3 の SNNAgent を拡張

## Task 3: 学習ループ
- [ ] `snnai/benchmarks/atari_trainer.py` を作成
- [ ] エピソード単位の学習を実装

## Task 4: テスト
- [ ] `tests/test_phase12.py` を作成
- [ ] 学習曲線が改善することを確認

## Task 5: v2.0 リリース
- [ ] `VERSION` を `v2.0.0` に更新
- [ ] `README.md` に v2.0 成果を追記
- [ ] コミット・push 後に `v2.0` タグを打つ
