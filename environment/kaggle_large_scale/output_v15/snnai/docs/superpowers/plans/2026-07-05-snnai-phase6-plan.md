# SNNAI Phase6 実装計画

**Goal:** 異種ニューラルネットワーク統合を実装し、複数の生物モジュールを1つのパイプラインで動作させる。

## Task 1: イベントフォーマットと Hub

**Files:**
- Create: `snnai/integration/__init__.py`
- Create: `snnai/integration/hub.py`

- [ ] `SpikeEvent` dataclass を定義
- [ ] `Hub` クラス：publish/subscribe によるイベントルーティング

## Task 2: 符号化変換層

**Files:**
- Create: `snnai/integration/encoding_bridge.py`

- [ ] `spikes_to_rates`: スパイク列から発火率への変換
- [ ] `rates_to_spikes`: 発火率からポアソンスパイク列への変換
- [ ] 次元変換（linear projection）

## Task 3: 統合パイプライン

**Files:**
- Create: `snnai/integration/pipeline.py`

- [ ] `SNNAIPipeline` クラス：モジュールを接続して順伝搬
- [ ] Phase2 反射モジュールと Phase3 ハニービーモジュールを統合した例

## Task 4: テスト

**Files:**
- Create: `tests/test_integration.py`

- [ ] Hub がイベントを正しくルーティングすること
- [ ] Bridge が双方向の変換を正しく行うこと
- [ ] 2 モジュール統合パイプラインが入力から出力まで動作すること

## Task 5: GitHub タグ v0.7

- [ ] コミット・push 後に v0.7 タグを打つ
