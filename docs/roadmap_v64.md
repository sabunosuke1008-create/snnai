# SNNAI v6.4 改良ロードマップ：BPE サブワードトークナイザ統合

## 背景

v6.3.1 では WikiText-2 のダウンロード問題を修正し、14M トークンの大規模コーパスで学習できるようになりました。しかし、vocab size が 65（Tiny Shakespeare のみ）から **1,153** まで拡大したことで、文字レベルモデルの予測難易度が急上昇し、生成テキストが再び改行・スペースに退化しました。

これは、文字レベルで 1,153 クラスを予測する負荷が、133 万パラメータの SNN にとって過大であったためです。モデルは「次の文字が何か確信が持てない」場合、最も出現頻度の高い改行・スペースを「安全牌」として選択するようになります。

## 目標

**BPE（Byte Pair Encoding）サブワードトークナイザを大規模学習パイプラインに統合し、文字レベルからサブワードレベルへの予測単位を変更することで、生成品質の根本的な改善を図ります。**

- 既存の `BPETokenizer`（`snnai/modules/language/bpe_tokenizer.py`）を活用
- 語彙サイズを約 2,000〜5,000 のサブワードに設定
- 実効系列長を短縮し、SNN が文脈関係性の学習にリソースを集中
- 補助対策として v6.4-dev で実装した logit bias / top-p / loss penalty も維持

## Phase 6.4.1：ロードマップ・補助機能の整理

- `docs/roadmap_v64.md` を BPE 統合中心に更新（本ファイル）
- logit bias、top-p、loss penalty は補助機能として維持
- テストハングの原因を調査・解消

## Phase 6.4.2：BPE トークナイザのパイプライン統合

### 内容

`snnai/modules/language/bpe_tokenizer.py` の `BPETokenizer` を、以下のコンポーネントで使用できるようにします。

- `snnai/benchmarks/large_corpus_trainer.py`
- `snnai/benchmarks/transformer_comparison.py`
- `snnai/benchmarks/generation_metrics.py`
- `snnai/benchmarks/text_generation_release.py`
- `environment/kaggle_large_scale/notebook.ipynb`

### 実装方針

- tokenizer が `CharTokenizer` でも `BPETokenizer` でも動作するよう、共通インターフェースを整備
- tokenizer は `vocab_size`、`encode(text)`、`decode(indices)`、`idx_to_char`（または `idx_to_token`）を提供
- BPE の場合は `decode` でサブワードを文字列に結合
- `SpikeEncoder` は tokenizer.vocab_size をそのまま利用

## Phase 6.4.3：Kaggle ノートブック更新

### 内容

`environment/kaggle_large_scale/notebook.ipynb` に BPE トークナイザを組み込みます。

### 変更点

- corpus 読み込み後、文字集合ではなく BPE 語彙を構築
- `CharTokenizer` の代わりに `BPETokenizer` を使用
- vocab_size を 2,000〜5,000 に設定
- `fair_compare()` に tokenizer を渡す
- 生成セルで BPE トークンをデコードして表示

### ファイル

- `environment/kaggle_large_scale/notebook.ipynb`

## Phase 6.4.4：テスト追加・更新

### 内容

BPE 統合を検証するテストを追加し、既存の v6.4 補助機能テストも維持します。

- `tests/test_v64_bpe_integration.py`
  - BPE tokenizer で encode/decode が対応可能
  - `LargeCorpusTrainer` が BPE tokenizer を受け入れる
  - `evaluate_generation()` が BPE tokenizer で動作
- 既存テスト
  - `tests/test_v64_generation_bias.py`
  - `tests/test_v64_topp_sampling.py`
  - `tests/test_v64_loss_penalty.py`
  - `tests/test_v64_version_sync.py`

## Phase 6.4.5：バージョン同期とリリース

### 内容

v6.4.0 としてリリースします。

- `VERSION` を `v6.4.0` に更新
- `README.md` に v6.4.0 の節を追加（BPE 統合を主題とする）
- `snnai/__init__.py` は `VERSION` ファイルを読むため自動的に同期
- `environment/kaggle_large_scale/kernel-metadata.json` を v6.4.0 に同期
- notebook.ipynb の clone branch を `v6.4.0` に更新
- Git tag `v6.4.0` を作成・push

## Phase 6.4.6：Kaggle 最終検証

### 内容

v6.4.0 を Kaggle で実行し、生成品質が改善するか確認します。

- Kaggle version 26 以降で v6.4.0 を push
- BPE vocab size 2,000〜5,000 で学習
- 生成テキストで改行・スペース以外のトークン（単語、サブワード）が出現するか確認
- BLEU-1 / CER / 生成テキストの多様性を評価
- `docs/snnai_test_results.md` を更新

## 期待される成果

- 貪欲生成でも改行・スペースに陥らず、英単語やサブワード単位のトークンが出現
- sampling 生成で流暢な英文の断片が出現
- perplexity は維持または改善（BPE により予測タスクが容易になるため）
- Kaggle 上で全セルが正常完了

## リスクと対策

| リスク | 対策 |
|---|---|
| BPE tokenizer の実装が大規模コーパスに耐えない | 既存実装を検証し、必要に応じてメモリ効率を改善 |
| BPE 語彙サイズが大きすぎると SNN spike 表現が希釈化 | 2,000〜5,000 程度から試行し、増減を検討 |
| 生成時の decode でサブワード間の結合が不自然 | スペース付きデコードルールを整備 |
| Kaggle 実行時間増加 | モデルサイズは維持、系列長短縮で対応 |
