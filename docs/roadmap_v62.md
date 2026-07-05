# SNNAI v6.2 改良ロードマップ

## 背景

v6.1.0 では SNN 大規模言語モデルの学習パイプライン、評価、エネルギー推定を整備し、Kaggle T4 GPU で全セルが `COMPLETE` となることを確認しました。しかし、Tiny Shakespeare（約111万文字）に対して約350万パラメータのモデルを学習させた結果、**過学習・記憶**が発生し、生成テキストが改行のみに退化しました。

v6.2 では、この根本的な品質問題に取り組み、実用的な文字生成が可能な SNN LM を目指します。

## 目標

- 検証損失が意味のある値（0でない）になり、過学習を検知できること
- 貪欲デコーディング以外（temperature / top-k sampling）で多様なテキストを生成できること
- コーパス規模に見合ったモデルサイズで安定して学習すること
- Kaggle T4 GPU で v6.2.0 として再検証し、リリースすること

## Phase 6.2.1: 時系列バリデーション split の導入

### 問題

現在の `LargeCorpusTrainer` と `fair_compare()` は `random_split` を使用しています。これにより、検証データが学習データと同じ分布・同じ文脈を含み、モデルがテキストを記憶した場合に val_loss も 0 になってしまいます。

### 対応

- `LargeCorpusTrainer._make_loaders()` を変更し、ランダム split ではなく**末尾 hold-out**（時系列 split）を使用する。
- `fair_compare()` 内の Transformer 学習でも同様に時系列 split を使用する。
- デフォルトの `val_ratio=0.05` は維持し、最後の 5% を検証に回す。

### 期待される効果

- 検証損失が 0 にならず、過学習の早期発見が可能になる。

## Phase 6.2.2: サンプリング生成（temperature / top-k）の実装

### 問題

現在の `evaluate_generation()` は常に `argmax`（貪欲）を使用しているため、モデルが改行トークンに強いバイアスを持つと、生成が無限に改行を続けてしまいます。

### 対応

- `snnai/benchmarks/generation_metrics.py` の `evaluate_generation()` に以下のパラメータを追加する：
  - `temperature`（softmax の鋭さ調整）
  - `top_k`（上位 k トークンからサンプリング）
  - `do_sample`（True でサンプリング、False で貪欲）
- 同様に `snnai/benchmarks/text_generation_release.py` などの生成 API も拡張する。

### 期待される効果

- モデルのバイアスを緩和し、より多様な文字生成が可能になる。

## Phase 6.2.3: モデルサイズ・出力表現の調整

### 問題

Tiny Shakespeare（111万文字、語彙65）に対して `embed_dim=256, hidden_dim=1024, num_layers=4`（350万パラメータ）は大きすぎます。これは明らかな過学習要因です。

### 対応

- `LargeScaleSNNLM` のデフォルトを維持しつつ、Kaggle notebook ではより小さな設定を使用する：
  - `embed_dim=128, hidden_dim=512, num_layers=3`（約44万パラメータ）
- `output_mode` の比較：`mem_mean`、`mem_last`、`spike_sum` の効果を確認する。
- 必要に応じて `beta`、`threshold`、`learn_threshold` を調整する。

### 期待される効果

- コーパス規模に見合ったモデル容量で、過学習が抑制される。

## Phase 6.2.4: 大規模コーパスの利用

### 問題

Tiny Shakespeare は SNN LM の学習には小さすぎます。より多様な文脈が必要です。

### 対応

- Kaggle notebook で WikiText-2 または他の公開テキストコーパスをダウンロードし、統合する。
- `CharTokenizer` が複数ファイルに対応できるように拡張する。
- コーパス長が増えることで、より大きなモデルでも過学習しにくくなる。

### 期待される効果

- データの多様性が増し、モデルが一般化しやすくなる。

## Phase 6.2.5: 学習安定化

### 問題

学習が 1 epoch で損失 0 に到達しており、最適化が不安定またはデータが簡単すぎます。

### 対応

- Label smoothing（0.1）を `cross_entropy` に導入する。
- Dropout を維持・調整する（0.2〜0.3）。
- Weight decay（すでに AdamW 0.01）を維持。
- Gradient clipping を維持（max_grad_norm=1.0）。
- LR warmup 後のピークを抑え、base_lr を 5e-4 に試行する。

### 期待される効果

- 損失が急降下しにくくなり、学習が安定する。

## Phase 6.2.6: Kaggle 最終検証と v6.2.0 リリース

### 対応

- 全ての修正を反映した Kaggle notebook を `v6.2.x` タグで push する。
- T4 GPU で `COMPLETE` を確認する。
- `docs/snnai_test_results.md` を更新する。
- `VERSION` を `v6.2.0` に更新する。
- tag `v6.2.0` を作成し push する。

### 成功基準

- Kaggle version N が `COMPLETE`
- 検証損失が 0 でない（過学習が抑制されている証拠）
- 生成テキストが改行のみでない（sampling または model quality の改善）
- ローカルテストが 106 passed を維持

## 参考: v6.1.0 からの教訓

- モデル容量はコーパス規模に見合わせる必要がある。
- 検証 split は時系列で行う必要がある。
- 貪欲デコーディングは強いバイアスを露呈させやすい。
- SNN の spike カウントとエネルギー推定は正しく動作するようになった。
