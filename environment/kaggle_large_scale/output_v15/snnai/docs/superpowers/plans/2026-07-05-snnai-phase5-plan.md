# SNNAI Phase5 実装計画

**Goal:** タコ分散モジュール（独立並列処理・多モーダル統合）を実装し、複数サブモジュールの並列動作と統合を示す。

## Task 1: 分散並列モジュール

**Files:**
- Create: `snnai/modules/octopus/__init__.py`
- Create: `snnai/modules/octopus/distributed_module.py`

- [ ] 複数の独立した LIF サブモジュールを定義
- [ ] 各サブモジュールを並列に実行（ThreadPoolExecutor）
- [ ] 統合層でサブモジュール出力を集約

## Task 2: 多モーダル統合テスト

**Files:**
- Create: `tests/test_octopus_integration.py`

- [ ] 両方の入力チャネルが強い場合は「警告」クラス
- [ ] 片方だけ強い場合は「注意」クラス
- [ ] 両方弱い場合は「安全」クラス
- [ ] 並列実行結果の出力形状を確認

## Task 3: GitHub タグ v0.5

- [ ] コミット・push 後に v0.5 タグを打つ
