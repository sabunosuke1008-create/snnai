# SNNAI v2.1〜v6.0 成果詳細ドキュメント

**Version**: `v6.0.0`  
**Last updated**: 2026-07-05  
**Repository**: https://github.com/sabunosuke1008-create/snnai  
**Kaggle Notebook**: https://www.kaggle.com/code/gihuhi/snnai-v6-scale-training

---

## 1. はじめに

本ドキュメントは、SNNAI（Spiking Neural Network based AI）プロジェクトが `v2.1` から `v6.0.0` にかけて実装してきた全モジュール、ベンチマーク、実験を**詳細かつ具体的に**まとめたものです。各バージョンで「何を」「なぜ」「どのように」実装したか、ファイル構成、テスト方法、Kaggle との連携まで記述します。

### 全体の設計思想

- **バイオインスパイアード**: 線虫、ミツバチ、カラス、タコなどの神経機構を参考にしつつ、実用的な AI タスクへ応用する。
- **省電力・イベント駆動**: SNN（Spiking Neural Network）の特性を活かし、エッジデバイスや常時稼働システムでの利用を想定。
- **モジュラー構成**: 各生物モジュールを独立した Python モジュールとして実装し、必要に応じてパイプライン化できる。
- **段階的スケール**: 小規模な概念実証（ローカル）→ 大規模コーパス学習・Transformer 比較（Kaggle）という段階で拡張。

---

## 2. ロードマップ全体像

| Stage | バージョン | テーマ |
|---|---|---|
| Stage 1 | v2.1〜v2.5 | 省電力エッジ AI |
| Stage 2 | v2.6〜v3.5 | 小規模モジュール型言語パイプライン |
| Stage 3 | v3.6〜v4.5 | テキスト生成 |
| Stage 4 | v4.6〜v5.5 | LLM 支援・常時稼働 AI |
| Stage 5 | v5.6〜v6.0 | Transformer 並み性能への挑戦 |

---

## 3. Stage 1：省電力エッジ AI（v2.1〜v2.5）

### 3.1 概要

実世界のセンサーデータを低消費電力で処理し、ロボット制御や IoT 異常検知、自動運転デモまでを SNN で実現することを目指しました。

### 3.2 実装ファイルと責務

| ファイル | 責務 |
|---|---|
| `snnai/modules/edge/sensor_filter.py` | ノイズ抑制・移動平均 + SNN スパイクフィルタ |
| `snnai/modules/edge/robot_controller.py` | 差分エンコーディング + 線形層による簡易ロボット行動出力 |
| `snnai/modules/edge/iot_event_detector.py` | snnTorch Leaky LIF を使った異常検出器 |
| `snnai/modules/edge/edge_pipeline.py` | フィルタ→検出→制御をつなぐエッジパイプライン |
| `snnai/benchmarks/auto_driving.py` | 簡易 2D レーンキープデモ |

### 3.3 動作のイメージ

```text
センサー入力 → sensor_filter（ノイズ除去）
           → iot_event_detector（イベント検出）
           → robot_controller / auto_driving（行動決定）
```

### 3.4 テスト

```bash
pytest tests/test_v21_sensor_filter.py tests/test_v22_robot_controller.py \\
       tests/test_v23_iot_event_detector.py tests/test_v24_edge_pipeline.py \\
       tests/test_v25_auto_driving.py -v
```

---

## 4. Stage 2：小規模モジュール型言語パイプライン（v2.6〜v3.5）

### 4.1 概要

文字レベルから始まる小さな言語モデルを SNN で構築し、トークナイズ、前処理、文脈保持、推論、小規模 LM、さらに海馬（hippocampus）による連想記憶を統合しました。

### 4.2 実装ファイル

| ファイル | 責務 |
|---|---|
| `snnai/modules/language/tokenizer.py` | 文字トークナイザ + SpikeEncoder |
| `snnai/modules/language/next_token.py` | SNN による次文字予測 |
| `snnai/modules/language/preprocess_pipeline.py` | 注意マスク + トークン強調（C. elegans / 昆虫風前処理） |
| `snnai/modules/language/context_retention.py` | 作業記憶を使った文脈保持 |
| `snnai/modules/language/reasoning.py` | 小規模推論 SNN 層 |
| `snnai/modules/language/small_lm.py` | 単一 LIF 層の小規模言語モデル |
| `snnai/modules/hippocampus/associative_memory.py` | 海馬連想記憶 |
| `snnai/benchmarks/summarization.py` | 要約ベンチマーク（概念実証） |
| `snnai/benchmarks/qa.py` | QA ベンチマーク |
| `snnai/benchmarks/translation.py` | 翻訳ベンチマーク |
| `snnai/benchmarks/code_completion.py` | コード補完ベンチマーク |

### 4.3 パイプラインの流れ

```text
生テキスト → CharTokenizer（文字 ID 化）
          → TextPreprocessPipeline（顕著なトークンを強調）
          → WorkingMemory（直近文脈を保持）
          → SmallLM / NextTokenSNN（次トークン予測）
          → HippocampalMemory（長期連想を利用）
```

### 4.4 Kaggle 連携

- **Kernel**: `gihuhi/snnai-v3-5-small-lm-training`
- 小規模 LM の実学習を T4 GPU で実行します。

---

## 5. Stage 3：テキスト生成（v3.6〜v4.5）

### 5.1 概要

文字レベル、単語レベル、BPE/サブワード、長期文脈、数百万〜数千万パラメータ規模の SNN LM、Transformer 比較まで段階的に拡張しました。

### 5.2 実装ファイル

| ファイル | 責務 |
|---|---|
| `snnai/benchmarks/char_lm_trainer.py` | 文字レベル LM トレーナー |
| `snnai/modules/language/word_tokenizer.py` | 単語トークナイザ |
| `snnai/modules/language/word_lm.py` | 単語レベル SNN LM |
| `snnai/modules/language/bpe_tokenizer.py` | BPE/サブワードトークナイザ |
| `snnai/modules/language/long_context.py` | 海馬を使った長期文脈拡張 |
| `snnai/modules/language/large_lm.py` | 多層 LIF の大規模 SNN LM（`LargeNextTokenSNN`） |
| `snnai/benchmarks/transformer_baseline.py` | Transformer ベースライン |
| `snnai/benchmarks/energy_benchmark.py` | エネルギー効率ベンチマーク |
| `snnai/benchmarks/finetune.py` | ファインチューニング基盤 |
| `snnai/benchmarks/dialogue.py` | 対話応答 PoC |
| `snnai/benchmarks/text_generation_release.py` | 実用的テキスト生成 API（`TextGenerator`） |

### 5.3 テキスト生成 API の使い方

```python
from snnai.benchmarks.text_generation_release import TextGenerator

gen = TextGenerator("abcdefghijklmnopqrstuvwxyz ")
gen.train("hello world hello world", epochs=5)
print(gen.generate("hello", max_chars=20))
```

### 5.4 テストのポイント

- 生成結果の長さは `len(prompt) + max_chars` をアサート。
- 大規模 LM はローカルではパラメータ数が >80k であることを確認（大規模設定は Kaggle で検証）。

---

## 6. Stage 4：LLM 支援・常時稼働 AI（v4.6〜v5.5）

### 6.1 概要

外部 LLM と連携するための前処理・後処理、長期記憶、常時監視、マルチモーダル、パーソナライゼーション、エッジ展開までを実装しました。

### 6.2 実装ファイル

#### v4.6〜v5.0：LLM 連携パイプライン

| ファイル | 責務 |
|---|---|
| `snnai/modules/llm_bridge/preprocess.py` | LLM 入力前の圧縮・要約前処理 |
| `snnai/modules/llm_bridge/postprocess.py` | LLM 出力後の SNN 推論整形 |
| `snnai/modules/llm_bridge/memory_bridge.py` | 海馬記憶と LLM 文脈の橋渡し |
| `snnai/benchmarks/always_on_demo.py` | 常時監視デモ |
| `snnai/modules/llm_bridge/pipeline.py` | 前処理→LLM stub→記憶→後処理の統合パイプライン |

#### v5.1〜v5.5：省電力・マルチモーダル・エッジ・常時稼働

| ファイル | 責務 |
|---|---|
| `snnai/benchmarks/energy_quantification.py` | レイテンシ・スパイク数を含む詳細エネルギー報告 |
| `snnai/modules/multimodal/multimodal_input.py` | 画像風 + 音声風特徴の融合エンコーダ |
| `snnai/modules/personalization/user_adapter.py` | ユーザ埋め込みによるパーソナライズ |
| `snnai/benchmarks/edge_deployment.py` | エッジ向けプロファイリング + TorchScript/状態保存エクスポート |
| `snnai/benchmarks/always_on_ai_v1.py` | 常時監視 + LLM 応答を統合した v1.0 デモ |

### 6.3 LLM 連携パイプラインの流れ

```text
ユーザ入力 → LLMPreProcessor（圧縮）
          → MemoryBridge（関連する過去対話を検索）
          → LLM stub / 外部 LLM（下書き生成）
          → LLMPostProcessor（SNN で整形）
          → MemoryBridge（保存）
          → 応答
```

### 6.4 エッジ展開ユーティリティ

```python
from snnai.benchmarks.edge_deployment import edge_profile, export_to_torchscript
from snnai.modules.edge import IoTEventDetector

model = IoTEventDetector(input_size=4, hidden_size=8, threshold=1)
sample = torch.randn(5, 1, 4)
print(edge_profile(model, sample))
export_to_torchscript(model, sample, "edge_model.pt")
```

`export_to_torchscript` は snnTorch の Python surrogate 関数でトレースできない場合、`torch.save(state_dict)` に自動フォールバックします。

---

## 7. Stage 5：Transformer 並み性能への挑戦（v5.6〜v6.0）

### 7.1 概要

数億パラメータ規模を視野に入れた大規模 SNN アーキテクチャ、大規模コーパストレーナー、Transformer 比較スイート、最適化、そして Transformer 並み性能を測るチャレンジハーネスを実装しました。**実際の大規模学習は Kaggle で実行**し、ローカルではアーキテクチャと小規模データでの動作検証を行っています。

### 7.2 実装ファイル

| ファイル | 責務 |
|---|---|
| `snnai/modules/language/large_scale_lm.py` | 大規模多層 SNN LM（`LargeScaleSNNLM`）+ パラメータカウンタ |
| `snnai/benchmarks/large_corpus_trainer.py` | コーパスを使った学習ループ |
| `snnai/benchmarks/transformer_comparison.py` | Transformer ベースライン + SNN/Transformer 比較 |
| `snnai/benchmarks/optimization.py` | 量子化・枝刈り |
| `snnai/benchmarks/transformer_challenge.py` | v6.0 チャレンジ評価ハーネス |
| `environment/kaggle_large_scale/notebook.ipynb` | Kaggle 大規模学習 + 比較ノートブック |
| `environment/kaggle_large_scale/kernel-metadata.json` | Kaggle kernel メタデータ |

### 7.3 LargeScaleSNNLM

```python
from snnai.modules.language.large_scale_lm import LargeScaleSNNLM

model = LargeScaleSNNLM(
    vocab_size=65, embed_dim=256, hidden_dim=1024, num_layers=4
)
```

ローカルテストでは `embed_dim=8`, `hidden_dim=16` などの小設定で形状を検証。Kaggle では `embed_dim=256`, `hidden_dim=1024`, `num_layers=4` などを T4 GPU で学習します。

### 7.4 Kaggle ノートブックの内容

1. 依存関係インストール + リポジトリ clone
2. tiny Shakespeare コーパス取得
3. `LargeScaleSNNLM` の学習（perplexity 計測）
4. Transformer ベースラインの学習（同コーパス）
5. レイテンシ・パラメータ数・エネルギー比較
6. テキスト生成サンプル
7. 量子化・枝刈り適用
8. モデル保存

---

## 8. テスト方法

### 8.1 ローカル

```bash
# 推奨：brian2 互換性問題を除く
pytest tests/ --ignore=tests/test_environment.py -q
```

### 8.2 バージョン別

```bash
pytest tests/test_v56_large_scale.py -v
pytest tests/test_v57_large_corpus.py -v
pytest tests/test_v58_transformer_comparison.py -v
pytest tests/test_v59_optimization.py -v
pytest tests/test_v60_transformer_challenge.py -v
```

### 8.3 既知の注意点

- `tests/test_environment.py` は brian2 が NumPy の新しい API（`np.ndarray.ptp`）と互換性がなく、現状 import エラーとなります。これは v2.1〜v6.0 のロードマップとは無関係です。
- `tests/test_bio_nas.py` の 1 テストで `snnai/bio_nas/evaluator.py` の既存バグ（`KeyError: 'm3'`）により失敗します。これも新規モジュールとは無関係です。

---

## 9. GitHub タグ一覧

| タグ | 内容 |
|---|---|
| v2.1 | エッジセンサーフィルタ |
| v2.2 | ロボット制御 |
| v2.3 | IoT イベント検出 |
| v2.4 | エッジパイプライン統合 |
| v2.5 | 自動運転デモ |
| v2.6 | 文字トークナイザ |
| v2.7 | 次文字予測 |
| v2.8 | 前処理パイプライン |
| v2.9 | 文脈保持 |
| v3.0 | 推論モジュール |
| v3.1 | 小規模 SNN LM |
| v3.2 | 要約ベンチマーク |
| v3.3 | QA ベンチマーク |
| v3.4 | 翻訳ベンチマーク |
| v3.5 | コード補完ベンチマーク + Kaggle 小規模 LM 学習 |
| v3.6 | 文字レベル LM トレーナー |
| v3.7 | 単語レベル LM |
| v3.8 | BPE トークナイザ |
| v3.9 | 長期文脈（海馬） |
| v4.0 | 大規模 SNN LM（数百万〜数千万パラメータ） |
| v4.1 | Transformer ベースライン比較 |
| v4.2 | 推論時エネルギー測定 |
| v4.3 | ファインチューニング基盤 |
| v4.4 | 対話応答 PoC |
| v4.5 | テキスト生成リリース API |
| v4.6 | LLM 入力前処理 |
| v4.7 | LLM 出力後処理 |
| v4.8 | 長期記憶ブリッジ |
| v4.9 | 常時監視デモ |
| v5.0 | LLM 連携パイプラインデモ |
| v5.1 | 省電力推論の定量化 |
| v5.2 | マルチモーダル入力 |
| v5.3 | パーソナライゼーション |
| v5.4 | エッジ展開プロファイリング |
| v5.5 | 常時稼働 AI v1.0 |
| v5.6 | 大規模 SNN LM アーキテクチャ |
| v5.7 | 大規模コーパストレーナー |
| v5.8 | Transformer 比較スイート |
| v5.9 | 量子化・枝刈り最適化 |
| v6.0 | Transformer 並み性能チャレンジハーネス |
| v6.0.0 | v2.1〜v6.0 ロードマップ最終リリース |

---

## 10. 今後の展望

- Kaggle ノートブック `gihuhi/snnai-v6-scale-training` で実際の大規模学習を実行し、perplexity・生成品質・エネルギーを定量化する。
- `LargeScaleSNNLM` のハイパーパラメータを大規模 GPU 環境で探索する。
- `MemoryBridge` を実運用向けに拡張し、ベクトル DB や外部 LLM API との連携を強化する。
- `brian2` / `bio_nas` 周りの既存テスト失敗を別途修正する。

---

## 11. まとめ

SNNAI は v2.1 から v6.0.0 にかけて、**エッジ AI → 小規模言語パイプライン → テキスト生成 → LLM 連携・常時稼働 AI → Transformer 並み性能挑戦**という一貫した道筋を歩んできました。各バージョンは独立した Git タグとして管理され、ローカルでは pytest で検証、スケールが必要な部分は Kaggle T4 GPU ノートブックに委ねる構成としています。
