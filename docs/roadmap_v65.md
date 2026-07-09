# SNNAI v6.5+ 高性能化ロードマップ

**Version**: v6.5+  
**Date**: 2026-07-10  
**Repository**: https://github.com/sabunosuke1008-create/snnai

---

## 1. 現状と最重要課題

### 1.1 v6.4.6 の実測結果（Kaggle T4, BPE vocab=2048）

| 項目 | SNN | Transformer | 備考 |
|---|---|---|---|
| val perplexity | 161.18 | 47.90 | 同条件比較 |
| latency | 0.0607 s | 0.00189 s | SNN は約 32 倍遅い |
| parameters | 1,903,616 | 4,209,664 | SNN の方が少ないのに精度・速度とも劣る |
| energy | 7.33e-4 J | N/A | spike count = 733,240 |
| BLEU-1 | 0.107 | 0.259 | 生成品質にも大きな差 |

### 1.2 構造的弱点（クリティカル）

現行 `LargeScaleSNNLM` は各系列位置に対して `nn.Linear` を独立に適用しており、**time_steps 方向の LIF 膜電位減衰以外に、系列内の位置間の依存関係を学習する明示的な機構が存在しない**。これは Transformer との根本的な差に相当する。

| 機能 | Transformer | 現行 LargeScaleSNNLM |
|---|---|---|
| 位置間の依存 | Self-Attention | なし（位置独立） |
| 位置情報 | Positional Encoding | なし |
| リカレントな文脈 | 不要（Attention で直接参照） | なし |

**この問題を解決しない限り、モデルサイズの拡大や Bio-NAS によるモジュール探索の効果は限定的**。

### 1.3 クリティカルパス

```
[Phase 6.5] 系列軸リカレント + Embedding 置換
       ↓
[Phase 6.6] Hippocampus 外部記憶統合
       ↓
[Phase 6.7] Spiking Self-Attention 比較評価
       ↓
[Phase 6.8] 評価指標拡充（文法・意味的一貫性）
       ↓
[Phase 6.9] Kaggle ノートブック生成パイプライン信頼性向上
       ↓
[Phase 7.0] Bio-NAS による LM アーキテクチャ探索
```

**6.5 は次のフェーズ全ての前提条件**。

---

## 2. フェーズ構成

### Phase 6.5: 系列軸リカレント結合と入力表現の効率化

**目的**: `LargeScaleSNNLM` に系列内の位置間依存を導入し、入力表現を効率化する。

**変更内容**:
1. **`nn.Linear(vocab_size, embed_dim)` → `nn.Embedding(vocab_size, embed_dim)`**
   - BPE vocab=2048 では one-hot 行列演算が無駄。
2. **系列軸（seq_len 方向）リカレント結合の追加**
   - 最小限: `nn.RNN`/`nn.GRU`/`nn.LSTM` を SNN 前後に追加。
   - SNN らしい方法: 系列方向にも LIF を追加し、前トークンの spike を次トークンの入力に加算。
3. **位置エンコーディングの追加**
   - 学習可能な位置埋め込みまたは正弦波位置エンコーディング。

**成功基準**:
- Kaggle T4 で `COMPLETE`
- SNN val ppl ≤ 80（v6.4.6 の 161 から 50% 改善）
- latency ≤ 0.12 s

**想定リスク**:
- 系列方向のリカレントで勾配伝播が長くなり学習不安定化。
- LIF time_steps と series_steps の二重ループで遅くなる。

---

### Phase 6.6: Hippocampus 外部記憶の大規模 LM 統合

**目的**: 既存 `HippocampalMemory` を `LargeScaleSNNLM` に統合し、長距離文脈を活用する。

**変更内容**:
1. 各レイヤーの隠れ状態を key/value として保存。
2. 現在位置の hidden state で retrieve し、次トークン予測に gating 付きで加える。
3. 短い LIF リカレント + 長い Hippocampus の 2 階層記憶。

**成功基準**:
- Kaggle T4 で `COMPLETE`
- SNN val ppl ≤ 60
- 生成テキストで長距離の主語-述語一致が観測される
- latency ≤ 0.15 s

**想定リスク**:
- `HippocampalMemory` は現状バッチ非対応・GPU 非対応。
- Phase 6.5 依存。

---

### Phase 6.7: Spiking Self-Attention（SSA）の導入と比較評価

**目的**: Spikformer / SpikeBERT / SpikeLM の知見を取り入れ、softmax 不要の spike-based attention を実装し、リカレント案と比較する。

**変更内容**:
1. Q/K/V を spike に変換し、`S(BN(MLP(Q·K^T·V)))` で attention map を近似（参考: Spikformer）。
2. 3 アーキテクチャ比較: 系列軸リカレント / Hippocampus / SSA。
3. time_steps 方向と seq_len 方向の役割分離。

**成功基準**:
- Kaggle T4 で `COMPLETE`
- 最良の SNN val ppl ≤ 50
- 最良案の BLEU-1 ≥ 0.15
- latency は現状維持または 2 倍以内

**想定リスク**:
- SSA は GPU 上では疎性を活かしにくく遅くなる可能性。
- 因果言語モデルへのそのままの適用は困難。

**現実的限界**:
- 既存研究でも Transformer に追いついていない。知識蒸留なしで ppl 同等は未解決。

---

### Phase 6.8: 評価指標の拡充

**目的**: perplexity / BLEU-1 / CER だけでなく、文法的妥当性や意味的一貫性を測定する。

**変更内容**:
1. 文法エラー率 GER（language_tool_python / spacy）。
2. 意味的一貫性（sentence-transformers 自己類似度、topic drift rate）。
3. 繰り返し・退化指標（n-gram 繰り返し率、改行率）。
4. `evaluate_generation` 返り値への組み込み。

**成功基準**:
- すべての指標が Kaggle 出力 JSON に含まれる。
- 改行率 < 30%。
- GER を定量的に報告。

**想定リスク**:
- 外部ツールの Kaggle インストールが重い。
- 指標の再現性が低い可能性。

---

### Phase 6.9: Kaggle ノートブック生成パイプラインの信頼性向上

**目的**: 過去の失敗（IndexError, ImportError, IndentationError 等）を繰り返さない CI 的検証を導入する。

**変更内容**:
1. ノートブック生成後に `nbformat.validate()` と `ast.parse()` を実行。
2. JSON 内 `\n` が実際の改行に置き換わっていないかチェック。
3. kernel-metadata.json / title / 最初の markdown タイトルの整合性検証。
4. 新 kernel slug は短い名前（`snnai-v6-x-x`）に統一。

**成功基準**:
- push 後のエラー率を 30–40% から 10% 以下に削減。
- 全セルの `ast.parse()` 通過を push 前の必須条件とする。

**想定リスク**:
- 検証スクリプトのメンテナンスコスト。
- Kaggle API 挙動変更への依存。

---

### Phase 7.0: Bio-NAS による LM アーキテクチャ探索

**目的**: Bio-NAS を生物模倣モジュール群と LM アーキテクチャ選択を結びつける。

**変更内容**:
1. 探索空間を拡張: c_elegans/honeybee/crow/octopus に加え、LM 層タイプ（recurrent / attention / hippocampus gate / feed-forward-only）を探索対象に。
2. 目的関数: perplexity, latency, energy, BLEU-1, GER の多目的最適化。
3. 生物制約（発火率、spike 数、エネルギー）をペナルティに。

**成功基準**:
- Bio-NAS が少なくとも 3 種類の LM 層タイプから最適構成を選び出す。
- 探索された構成が手動設計を perplexity/latency/energy のいずれかで上回る。
- Kaggle 9 時間制限内で 1 世代の評価が完結（小規模 proxy モデル使用）。

**想定リスク**:
- 探索空間が広大で Kaggle では収束しない。
- Phase 6.5–6.7 で十分な性能向上がないと探索の意味が薄れる。

---

## 3. 優先順位と依存関係

| 優先度 | フェーズ | 依存 |
|---|---|---|
| **P0** | 6.5 系列軸リカレント + Embedding | なし |
| **P1** | 6.6 Hippocampus 統合 | 6.5 |
| **P1** | 6.7 Spiking Self-Attention | 6.5 |
| **P2** | 6.8 評価指標拡充 | 6.5 |
| **P2** | 6.9 Kaggle パイプライン信頼性 | 並行可能 |
| **P3** | 7.0 Bio-NAS 統合 | 6.5, 6.6, 6.7 |

**6.5 は絶対的な前提**。

---

## 4. 公平ベンチマーク設計

### 4.1 測定項目

| 指標 | 測定方法 |
|---|---|
| 推論レイテンシ | 1 バッチあたり wall-clock time（warmup 後平均） |
| スループット | tokens/sec |
| spike count | 各層 LIF spike 合計 |
| エネルギー | `snnai/benchmarks/energy_estimation.py` |
| パラメータ数 | `count_parameters(model)` |
| 精度 | perplexity, BLEU-1, CER, GER, topic drift |

### 4.2 公平性条件

- 同一コーパス、同一 tokenizer、同一 seq_len / batch_size。
- Transformer と SNN は同程度のパラメータ数で比較するグループを作成。
- GPU は T4 を固定。P100/CPU fallback は別枠記録。

### 4.3 SNN が遅い理由

現状は PyTorch + snntorch で GPU 上をシミュレーションしており、`time_steps=20` の for-loop を展開している。neuromorphic ハードウェア上のイベント駆動実行とは異なり、GPU では逆に遅くなる。

**現実的限界**: GPU シミュレーションで Transformer に速度で勝つのは極めて難しい。SNN の優位性は将来の neuromorphic チップ（Intel Loihi, IBM NorthPole 等）への展開を見据えた「理論エネルギー効率」として主張するのが妥当。

---

## 5. 現実的な限界と注意点

1. **SNN はまだ Transformer に追いついていない** — SpikeLM, SpikeLLM, NeuronSpark でも性能差は残る。
2. **SSA は GPU では必ずしも速くない** — 速度優位は専用ハードウェアに依存。
3. **Hippocampus のスケーラビリティ** — 長い WikiText 系列では容量と検索コストがボトルネック。
4. **Kaggle 制約** — 連続 9 時間、週 30 時間。Phase 7.0 は小規模 proxy モデルで実施。

---

## 6. 即座に始められる次のアクション

1. `LargeScaleSNNLM` に `nn.Embedding` と系列軸リカレントを追加。
2. `tests/test_v40_large_lm.py` / `tests/test_v57_large_corpus.py` を拡張。
3. Kaggle notebook の `LargeScaleSNNLM` 呼び出しを更新。
4. ローカルで最小設定で動作確認。
5. v6.5.0 タグ作成後、新規 kernel slug `gihuhi/snnai-v6-5-0` で Kaggle 実行。

---

## 7. リファレンス

- Zhou et al., "Spikformer: When Spiking Neural Network Meets Transformer", ICLR 2023.
- Lv et al., "SpikeBERT: A Language Spikformer Trained with Two-Stage Knowledge Distillation from BERT", OpenReview 2024.
- Xing et al., "SpikeLM: Towards General Spike-Driven Language Modeling via Elastic Bi-Spiking Mechanisms", ICML 2024.
- Xing et al., "SpikeLLM: Scaling up Spiking Neural Network to Large Language Models via Saliency-based Spiking", 2024.
- Gonzalez et al., "SAPU-LM: Synaptogenic Adaptive Processing Unit Language Models", Zenodo 2026.
- "NeuronSpark: A Spiking Neural Network Language Model with Selective State Space Dynamics", arXiv 2026.
