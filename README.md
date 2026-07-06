# SNNAI

Spiking Neural Network based AI — 線虫・ミツバチ・カラス・タコなど複数の生物の脳構造を参考にした、省電力・モジュラー型 SNN の研究プロジェクト。

## バージョン v6.3.0（SNN 大規模言語モデルの生成品質根本改善）

SNNAI v6.3.0 では [docs/roadmap_v63.md](docs/roadmap_v63.md) に基づき、v6.2.0 で残った「生成テキストが改行・スペースに退化する」問題と「Kaggle での WikiText-2 解凍エラー」に取り組みます。

### v6.3.0 の主な改善点

- **Phase 6.3.1 恒常性正則化ロス**: `snnai/benchmarks/homeostatic_loss.py` を新規作成。`LargeScaleSNNLM` の各 LIF レイヤーの平均発火率をターゲット帯域（12%）に保ち、SNN が無活動な改行・スペーストークンに沈み込むのを防ぎます。
- **Phase 6.3.2 反復ペナルティ**: `snnai/benchmarks/generation_metrics.py` に `repetition_penalty` を導入。直近に生成されたトークンの logits を減算し、改行連続・空白連続を抑制します。
- **Phase 6.3.3 Python ネイティブコーパスダウンロード**: `snnai/utils/download_corpus.py` を新規作成。Kaggle notebook の bash 解凍セルを `requests` + `zipfile` に置き換えます。
- **Phase 6.3.4 バージョン同期**: `VERSION`、`kernel-metadata.json`、`README.md` を `v6.3.0` に統一します。
- **Phase 6.3.5 テスト追加**: `tests/test_v63_*.py` 群で恒常性ロス・反復ペナルティ・ダウンロード関数を検証します。

### 重要なファイル

- [docs/roadmap_v63.md](docs/roadmap_v63.md) — v6.3.0 改良ロードマップ
- [snnai/benchmarks/homeostatic_loss.py](snnai/benchmarks/homeostatic_loss.py) — 恒常性正則化ロス
- [snnai/utils/download_corpus.py](snnai/utils/download_corpus.py) — クロスプラットフォームコーパスダウンロード

## バージョン v6.2.0（SNN 大規模言語モデルの過学習抑制と生成多様性の改善）

SNNAI v6.2.0 では [docs/roadmap_v62.md](docs/roadmap_v62.md) に基づき、v6.1.0 で発覚した過学習・記憶問題に取り組みました。

### v6.2.0 の主な改善点

- **Phase 6.2.1 時系列バリデーション**: `LargeCorpusTrainer` と `fair_compare()` でランダム split から末尾 hold-out（時系列 split）に変更。検証損失が 0 に陥るのを防ぎました。
- **Phase 6.2.2 サンプリング生成**: `evaluate_generation()` に `temperature` / `top_k` / `do_sample` を追加。貪欲デコーディング以外での生成が可能になりました。
- **Phase 6.2.3 モデルサイズ調整**: Kaggle notebook で SNN を `embed_dim=128, hidden_dim=512, num_layers=3`（約 63 万パラメータ）に縮小。Tiny Shakespeare に対する過学習を抑制しました。
- **Phase 6.2.4 大規模コーパス**: WikiText-2 ダウンロードセルを追加（Kaggle 環境では unzip に失敗し Tiny Shakespeare fallback）。
- **Phase 6.2.5 学習安定化**: label smoothing（0.1）、lr 5e-4、AdamW(weight_decay=0.01)、dropout 0.2 を適用。
- **Phase 6.2.6 リリース**: v6.2.0 タグ、Kaggle T4 最終検証（version 18）。SNN val ppl 2.15、Transformer val ppl 2.09 を達成。

### 重要なファイル

- [docs/roadmap_v62.md](docs/roadmap_v62.md) — v6.2.0 改良ロードマップ
- [docs/snnai_test_results.md](docs/snnai_test_results.md) — version 18 の Kaggle 実行結果

## バージョン v6.1.0（SNN 大規模言語モデルの学習・評価・エネルギー推定の改善）

SNNAI v6.1.0 では [docs/roadmap_v6_improvement.md](docs/roadmap_v6_improvement.md) に基づき、v6.0.0 で発覚した「SNN が学習しない」「spike 発火がゼロ」「エネルギー推定が非現実的」といった課題に取り組みました。

### v6.1.0 の主な改善点

- **Phase 6.1 診断**: `snnai/benchmarks/diagnostic.py` による勾配・spike 発火率・中間活性の可視化。
- **Phase 6.2 アーキテクチャ修正**: `LargeScaleSNNLM` に `LayerNorm`、出力層の membrane mean、学習可能 threshold を導入。SNN が実際に学習するようになりました（tiny Shakespeare の小規模実験で loss 3.24 → 2.41、val ppl 12.85 → 11.05）。
- **Phase 6.3 学習強化**: `LargeCorpusTrainer` を全面的に書き換え、Dataset/DataLoader、train/val 分割、Warmup + Cosine Annealing、勾配クリッピング、最良 checkpoint 保存を実装。
- **Phase 6.4 評価**: `snnai/benchmarks/generation_metrics.py` で perplexity / accuracy / BLEU-1 / CER を計測。`transformer_comparison.py` の `fair_compare()` で SNN と Transformer を同一データ・同一条件で公平に比較。
- **Phase 6.5 エネルギー**: `energy_quantification.py` を LIF ニューロンの実際の spike をカウントする方式に変更し、非ゼロの spike ベース消費電力を推定。
- **Phase 6.6 リリース**: v6.1.0 タグ、Kaggle T4 最終検証、学習済みチェックポイント `models/snnai_v6_large_lm.pt` の更新。

### 重要なファイル

- [docs/roadmap_v6_improvement.md](docs/roadmap_v6_improvement.md) — v6.1.0 改良ロードマップ
- [docs/snnai_test_results.md](docs/snnai_test_results.md) — テスト結果と Kaggle 実行履歴
- [models/snnai_v6_large_lm.pt](models/snnai_v6_large_lm.pt) — Kaggle T4 で学習済みの v6 SNN LM チェックポイント

## バージョン v6.0.0（v2.1〜v6.0 ロードマップ完了）

SNNAI v6.0.0 では [docs/roadmap_v3.md](docs/roadmap_v3.md) に記載された v2.1〜v6.0 の全ステージを完了しました。ローカルではアーキテクチャ検証・単体テストを、大規模学習・Transformer 比較は Kaggle で実行します。

### 詳細ドキュメント

- [docs/snnai_v6_achievements.md](docs/snnai_v6_achievements.md) — v2.1〜v6.0 の各モジュール・ベンチマーク・Kaggle 連携を詳しく解説

### 主な成果

- **v2.1〜v2.5**: 省電力エッジ AI（センサフィルタ、ロボット制御、IoT イベント検出、自動運転デモ）
- **v2.6〜v3.5**: 小規模モジュール型言語パイプライン（トークナイザ、次トークン予測、前処理、文脈保持、推論、小規模 LM）
- **v3.6〜v4.5**: テキスト生成（文字/単語/BPE トークナイザ、大規模 SNN LM、Transformer 比較、ファインチューニング、対話、リリース API）
- **v4.6〜v5.5**: LLM 連携 / 常時稼働 AI（LLM 入力前処理、出力後処理、長期記憶、常時監視、パイプラインリリース、省電力定量、マルチモーダル、パーソナライゼーション、エッジ展開）
- **v5.6〜v6.0**: 大規模スケール実験（数億パラメータ規模アーキテクチャ、大規模コーパストレーナー、Transformer 比較スイート、量子化・枝刈り、Transformer 並み性能チャレンジ）

### GitHub タグ

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
| v3.5 | コード補完ベンチマーク / Kaggle 小規模 LM 学習 |
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
| v6.0.0 | v6.0.0 最終リリース |

### Kaggle ノートブック

- `gihuhi/snnai-v6-scale-training`: v5.6〜v6.0 大規模 SNN LM 学習 + Transformer 比較（T4 GPU、internet 有効）

## 今後の改良ロードマップ（v6.1 以降）

version 10 の Kaggle 実行結果（SNN が学習せず spike 発火もゼロ）を受け、v6.1 以降の厳密な改良計画を策定しました。

- [docs/roadmap_v6_improvement.md](docs/roadmap_v6_improvement.md) — 診断・アーキテクチャ修正・学習強化・評価・エネルギー推定・最終リリースの段階的計画
- [models/snnai_v6_large_lm.pt](models/snnai_v6_large_lm.pt) — Kaggle T4 で学習済みの v6 大規模 SNN LM チェックポイント（14 MB）
- [docs/snnai_test_results.md](docs/snnai_test_results.md) — テスト実行結果と Kaggle version 10 の詳細

## バージョン v2.0（Phase12 完了）

SNNAI v2.0 では Phase8〜Phase12 の実データ・大規模化・Atari 風実験を統合しました。

### Phase8〜Phase12 の成果

- **Phase8**: MNIST / Fashion-MNIST SNN 分類器（畳み込み + LIF）
- **Phase9**: N-MNIST 風イベントデータの疑似生成とスパイク変換
- **Phase10**: End-to-End で学習可能な統合パイプライン
- **Phase11**: 大規模 SNN、GPU 自動選択、checkpoint 機構、T4 GPU Kaggle ノートブック
- **Phase12**: 簡易 Atari 風キャッチゲームでの SNN 強化学習

### バージョン履歴

| タグ | 内容 |
|---|---|
| v0.1 | Phase0〜Phase0.5 |
| v0.2 | Phase2 線虫 |
| v0.3 | Phase3 ミツバチ |
| v0.4 | Phase4 カラス |
| v0.5 | Phase5 タコ |
| v0.6 | Phase X Bio-NAS |
| v0.7 | Phase6 統合 |
| v1.0 | Phase7 完成版 |
| v1.1 | Phase8 MNIST/Fashion-MNIST |
| v1.2 | Phase9 N-MNIST |
| v1.3 | Phase10 End-to-End 学習 |
| v1.4 | Phase11 大規模化・GPU・checkpoint |
| v2.0 | Phase12 Atari 風実験・v2 完成 |

## バージョン v1.0（Phase7 完了）

SNNAI v1.0 では以下のモジュールとベンチマーク基盤を統合しました。

### 含まれるモジュール

- **Phase1**: LIF / Izhikevich / STDP / LSM コア
- **Phase2**: 線虫モジュール（反射）
- **Phase3**: ミツバチモジュール（空間 RL + R-STDP）
- **Phase4**: カラスモジュール（作業記憶）
- **Phase5**: タコモジュール（分散並列処理）
- **Phase X**: Bio-NAS（生物制約付きアーキテクチャ探索）
- **Phase6**: 異種ニューラルネットワーク統合（Hub + Encoding Bridge + Pipeline）
- **Phase7**: ベンチマークとベースライン比較

### GitHub タグ

- `v0.1` Phase0〜Phase0.5
- `v0.2` Phase2 線虫
- `v0.3` Phase3 ミツバチ
- `v0.4` Phase4 カラス
- `v0.5` Phase5 タコ
- `v0.6` Phase X Bio-NAS
- `v0.7` Phase6 統合
- `v1.0` Phase7 完成版

### ベンチマーク

- `snnai/benchmarks/synthetic_benchmark.py`: 合成 4 クラス分類タスク
- `snnai/benchmarks/baseline_mlp.py`: 同規模 MLP ベースライン
- `snnai/benchmarks/energy_estimation.py`: スパイク数ベース簡易エネルギー推定

## ロードマップ

詳細は [docs/roadmap.md](docs/roadmap.md) を参照。

## クイックスタート

### ローカル

```bash
pip install -r requirements.txt
pytest tests/
```

### Kaggle

`environment/kaggle_setup.ipynb` を Kaggle Notebook としてインポートして実行してください。

```bash
# Kaggle CLI で push する例
python -m kaggle kernels push -p environment
```

## ライセンス

MIT
