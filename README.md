# SNNAI

Spiking Neural Network based AI — 線虫・ミツバチ・カラス・タコなど複数の生物の脳構造を参考にした、省電力・モジュラー型 SNN の研究プロジェクト。

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
