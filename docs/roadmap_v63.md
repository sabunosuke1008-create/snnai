# SNNAI v6.3 改良ロードマップ

## 背景

v6.2.0 では過学習を抑制し、SNN LM が Tiny Shakespeare で実際に一般化できることを確認しました。しかし、生成テキストは依然として空白（スペース）や改行（`\n`）トークンに強いバイアスを持ち、連続する改行や空行が大半を占めています。また、Kaggle notebook 内のコーパスダウンロードがシェルコマンド（`!wget` / `!unzip`）に依存しており、実行環境によって失敗します。さらに、リリース時の `VERSION` ファイルと Git タグの整合性を自動化する仕組みがありません。

v6.3 では、以下を目指します。

## 目標

1. **空白・改行トークンの抑制**: 損失関数とデコーディングの両方から、空白・改行トークンにペナルティを与え、より多様な文字生成を促進する。
2. **Kaggle ダウンロードのクロスプラットフォーム化**: シェルコマンドに依存せず、Python 標準ライブラリ / `requests` / `zipfile` を使ってコーパスをダウンロード・解凍する。
3. **VERSION とタグの同期**: `VERSION` ファイル、Git タグ、`kernel-metadata.json` のバージョン記述を一元管理し、ズレを防ぐ。
4. **Kaggle T4 GPU で v6.3.0 として再検証・リリース**する。

---

## Phase 6.3.1: 空白・改行抑制ロス（Token-Level Penalty Loss）

### 問題

現在の `LargeCorpusTrainer` は `CrossEntropyLoss` をそのまま使用しており、空白（token id = スペース文字）や改行（`\n`）が頻出するコーパスでは、モデルがこれらの高頻度トークンに引きずられやすくなります。

### 対応

- `snnai/benchmarks/large_corpus_trainer.py` に `PenaltyCrossEntropy` を新規実装する。
  - コンストラクタで `penalty_ids`（空白・改行のトークン id リスト）と `penalty_weight`（デフォルト 1.2）を受け取る。
  - 通常の `cross_entropy` に加え、対象トークンの logit に `penalty_weight` を乗じた追加損失を課す。
  - 空白・改行の正解ラベルが来た場合でも損失を増やさず、モデルがこれらを選びにくくする方向に作用する。
- `LargeCorpusTrainer` に `loss_type='penalty_ce' | 'ce'` オプションを追加し、tokenizer から自動的に空白・改行 id を取得して `PenaltyCrossEntropy` に渡す。

### 期待される効果

- 学習中から空白・改行トークンの過剰な選択を抑制し、他の文字トークンの学習が進む。

---

## Phase 6.3.2: デコーディング時の繰り返しペナルティと空白・改行ペナルティ

### 問題

`evaluate_generation()` や `generate_text()` は argmax または temperature sampling を行うが、空白・改行トークンに対する抑制がないため、生成がこれらのトークンでループしやすい。

### 対応

- `snnai/benchmarks/generation_metrics.py` の `evaluate_generation()` に以下を追加する。
  - `repetition_penalty`（デフォルト 1.0 = 無効、例 1.2）: 既に生成されたトークンの logit を下げる。
  - `newline_space_penalty`（デフォルト 1.0 = 無効、例 1.5）: 空白・改行トークンの logit を下げる。
- 同様に `snnai/benchmarks/text_generation_release.py` の `generate_text()` も対応する。
- ペナルティは logit スケーリング後に適用し、温度による softmax より前に行う。

### 期待される効果

- 生成テキストの改行連続・空白連続が減り、文字列の多様性（BLEU/CER も含め）が向上する。

---

## Phase 6.3.3: Kaggle クロスプラットフォーム・コーパスダウンロード

### 問題

Kaggle notebook 内で `!wget` や `!unzip` を使うと、パーミッションエラーやパス解釈の違いで失敗します。v6.2.4 では WikiText-2 の解凍が失敗しました。

### 対応

- `environment/kaggle_large_scale/notebook.ipynb` のダウンロードセルを Python 実装に置き換える。
  - `requests`（Kaggle 環境にプリインストール）で zip をダウンロード。
  - `zipfile` で解凍。
  - `pathlib` でクロスプラットフォームなパス操作を行う。
  - ダウンロード失敗時は Tiny Shakespeare にフォールバックする。
- ローカルでも動作確認できるよう、`snnai/utils/download_corpus.py` として共通関数を切り出す。

### 期待される効果

- Kaggle 実行環境に依存せず、WikiText-2 などの外部コーパスを安定して取得できる。

---

## Phase 6.3.4: VERSION・タグ・ノートブックのバージョン同期

### 問題

`VERSION` ファイル、Git タグ、`kernel-metadata.json` のタイトル、notebook 内の clone タグが手動更新のため、リリース時に不整合が生じるリスクがあります。

### 対応

- `snnai/__init__.py`（または新規 `snnai/version.py`）で `__version__` を `VERSION` ファイルから読み込む。
- `environment/kaggle_large_scale/generate_notebook.py` に、現在の `VERSION` と Git タグを自動反映するオプションを追加する。
  - `--version` 指定時に `kernel-metadata.json` のタイトルと notebook 内の clone タグを更新。
- `kernel-metadata.json` の `title` を `SNNAI v{VERSION} Scale Training` 形式に統一する。

### 期待される効果

- リリース作業時に `VERSION` ファイルを更新するだけで、他のバージョン表記が自動的に追従する。

---

## Phase 6.3.5: ローカルテスト追加

### 対応

- `tests/test_v63_penalty_loss.py`:
  - `PenaltyCrossEntropy` が空白・改行トークンの選択を抑制することを確認。
  - 通常 CE と比較して、ペナルティ対象トークンの確率が下がることを検証。
- `tests/test_v63_generation_penalty.py`:
  - `repetition_penalty` と `newline_space_penalty` が生成結果に影響することを確認。
- `tests/test_v63_version_sync.py`:
  - `VERSION` ファイル、`snnai/__version__`、`kernel-metadata.json` の `title` が一致することを確認。
- `tests/test_v63_download_corpus.py`:
  - `download_corpus()` のフォールバック動作をモックで確認。

### 期待される効果

- 各機能がローカルで動作確認でき、Kaggle への持ち込み前に品質を担保できる。

---

## Phase 6.3.6: Kaggle 最終検証と v6.3.0 リリース

### 対応

- 全修正を反映した notebook を push する。
- `VERSION` を `v6.3.0` に更新する。
- tag `v6.3.0` を作成・push する。
- Kaggle T4 GPU で `COMPLETE` を確認する。
- `docs/snnai_test_results.md` に version N の結果を追記する。

### 成功基準

- Kaggle version N が `COMPLETE`
- 検証損失が 0 でない
- 生成テキストに空白・改行以外の文字が増加（CER 改善、BLEU 向上）
- ローカルテストが全て passed

---

## 参考: v6.2.0 からの教訓

- モデル容量をコーパス規模に合わせることで過学習は抑制できる。
- 時系列 split で val_loss が 0 に陥るのを防げる。
- temperature / top-k sampling は多様性を出すが、根本的なバイアス解消には損失関数・デコーディング両方のペナルティが必要。
- 外部コーパス取得はシェルコマンドに依存しない Python 実装にすべき。
