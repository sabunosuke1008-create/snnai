# SNNAI v6 大規模言語モデル改良ロードマップ

## 背景: version 10 の実行結果から分かった課題

Kaggle T4 GPU での version 10 実行結果を定量評価すると、以下の重大な問題が明らかになりました。

| 項目 | SNN | Transformer ベースライン | 評価 |
|---|---|---|---|
| parameters | 3,491,072 | 3,192,385 | 同等规模 |
| epoch 0 loss / ppl | 4.174 / 65.00 | 4.328 / 75.77 | SNN がやや良い出発点 |
| epoch 19 loss / ppl | **4.174 / 65.00** | **2.716 / 15.12** | **SNN が全く学習していない** |
| latency | 0.0523 s | 0.0221 s | SNN が 2.4 倍遅い |
| spikes_per_step | 0.0 | - | **SNN が spike を発火していない** |
| 生成テキスト | 空白/改行のみ | - | **全く文字を生成できていない** |
| 保存ファイル | `snnai_v6_large_lm.pt` | - | 14 MB、学習履歴付き |

**核心課題**: SNN は「動作する」が「学習しない」。損失が 20 epoch で完全に平坦であり、spike 発火もゼロです。これはアーキテクチャか学習ループ、あるいは spike 表現のどこかに根本原因があります。

---

## 目標

1. **SNN が実際に学習し、Transformer を追従/凌駕する**（version 6.x の最終目標）。
2. **spike 活動を可視化・定量化**し、SNN としての動作を保証する。
3. **レイテンシを Transformer 以下に抑える**（省エネ・高速推論の価値主張）。
4. **エネルギー推定を現実的な値にする**（spike ベースの消費電力モデル）。
5. **生成品質を客観的に測定**するベンチマークを整備する。

---

## 段階的改良計画

### Phase 6.1 — 診断と可視化（1〜2 週間）

**目的**: SNN が学習しない根本原因を特定する。

#### タスク

1. **勾配の可視化**
   - 各層の `p.grad` の L2 norm を epoch ごとに記録。
   - 勾配消失/爆発を検出。

2. **spike 発火率のモニタリング**
   - `snntorch.Leaky` から実際の spike tensor を取得し、time/step あたりの発火率を計算。
   - layer ごと・time step ごとの発火分布をログ出力。

3. **中間活性の分布調査**
   - `embed` 出力、`lin` 出力、`lif` 後の spike の平均/標準偏差を記録。
   - 入力が LIF のしきい値を超えていないケースを検出。

4. **損失曲面の簡易調査**
   - ランダムな重みと学習後の重みで損失を比較。
   - 損失が本当に変化しないか確認。

#### 成功基準

- 訓練中に spike 発火率が 0 より大きくなる層が少なくとも 1 つ存在すること。
- 勾配が数値的にゼロでないこと。
- 問題の原因が「勾配」「初期化」「しきい値」「loss 関数」「spike 表現」のどれかに絞られること。

#### 成果物

- `snnai/benchmarks/diagnostic.py`（診断ハーネス）
- 各種メトリックを CSV/JSON で保存

---

### Phase 6.2 — アーキテクチャ修正（2〜3 週間）

**目的**: SNN が spike を発火し、勾配が流れるアーキテクチャにする。

#### タスク

1. **LIF ニューロンの初期化・パラメータ調整**
   - `beta` を 0.9 からチューニング（0.5〜0.99）。
   - `threshold` を 1.0 から調整（0.1〜2.0）。
   - `learn_threshold=True` の実験。

2. **出力層の再設計**
   - 現状は spike を time 方向で sum しているが、これでは多くの場合 0 になる。
   - 代替案:
     - 最終 LIF 後の membrane potential を使う。
     - spike 割合を確率としてソフトマックスに入力。
     - time step ごとに logits を出して平均を取る。

3. **損失関数の見直し**
   - 現状: `cross_entropy` を (batch*seq, vocab) で計算。
   - 問題: spike がほとんど出ないため、softmax への入力が小さい/ゼロ。
   - 対策:
     - membrane potential ベースの損失を試す。
     - `cross_entropy` 前に適切なスケーリングを入れる。

4. **SpikeEncoder の改善**
   - 現状: one-hot を time 方向に複製しただけ。
   - 改善案:
     - 実際の rate coding や temporal coding。
     - 学習可能な spike encoder の導入。

5. **正規化の導入**
   - `LayerNorm` または `BatchNorm` の追加（SNN では LayerNorm が一般的）。
   - 勾配の安定化と spike 発火の促進。

#### 成功基準

- 5 epoch 以内に training loss が 4.17 より統計的に下がること。
- spike 発火率が平均 1% 以上あること。
- 生成テキストに少なくともランダムより高頻度で正しい文字が出現すること。

#### 成果物

- `snnai/modules/language/large_scale_lm.py` の改訂
- 新しい `SpikeEncoder` バリアント
- `snnai/benchmarks/losses.py`（複数損失関数）

---

### Phase 6.3 — 学習ループの強化（1〜2 週間）

**目的**: 安定的かつ高速な学習を実現する。

#### タスク

1. **学習率スケジュール**
   - Warmup + Cosine Annealing。
   - 初期 LR の感度実験（1e-4 〜 1e-2）。

2. **勾配クリッピング**
   - `torch.nn.utils.clip_grad_norm_` の導入。

3. **バッチサイズ・シーケンス長の最適化**
   - T4 16 GB に収まる範囲でバッチサイズを増やし、throughput を向上。
   - 勾配累積（gradient accumulation）の検討。

4. **検証セットの導入**
   - train/val 分割。
   - 過学習/学習不足の判定。

5. **チェックポイント戦略**
   - 最良 val loss で保存。
   - 学習再開用の checkpoint 形式整備。

#### 成功基準

- 20 epoch で val ppl < 40 を達成（現状 65 から大幅改善）。
- 学習曲線が単調減少または収束すること。

#### 成果物

- `snnai/benchmarks/large_corpus_trainer.py` の改訂
- `snnai/benchmarks/scheduler.py`
- 拡張された checkpoint 形式

---

### Phase 6.4 — 評価・ベンチマーク整備（1〜2 週間）

**目的**: 改良効果を客観的に測定する。

#### タスク

1. **生成品質メトリック**
   - Perplexity（既存）。
   - 文字/単語レベルの accuracy。
   - BLEU/ROUGE（短い生成テキスト向け）。
   - 人間が読める文字列が出現する割合。

2. **Transformer との公平な比較**
   - 同じパラメータ数、同じデータ、同じ epoch 数で比較。
   - 推論 latency、throughput、消費電力の同時測定。

3. **Ablasion study**
   - SpikeEncoder の有無。
   - LIF vs Leaky vs Synaptic ニューロン。
   - 出力層の spike sum vs membrane vs rate coding。

#### 成功基準

- SNN の val ppl が Transformer の 1.5 倍以内に収まること。
- SNN の推論 latency が Transformer 以下、または同等。

#### 成果物

- `snnai/benchmarks/transformer_comparison.py` の強化
- `snnai/benchmarks/generation_metrics.py`

---

### Phase 6.5 — エネルギー推定の現実化（1 週間）

**目的**: SNN の省エネ主張を数字で裏付ける。

#### タスク

1. **spike カウントの正確化**
   - `snntorch` の spike tensor から実際の発火数を取得。
   - layer ごと、time step ごとの発火数を計算。

2. **エネルギーモデルの改善**
   - 発火 1 回あたりのエネルギー係数を設定（文献値に基づく）。
   - 静的消費電力と動的消費電力を分離。

3. **Mac（乗算累加）ベースの代替推定**
   - Transformer の MAC 回数を計算。
   - SNN の spike ベース演算回数を計算し、比較。

#### 成功基準

- `spikes_per_step` が 0 でなくなること。
- SNN のエネルギー推定値が Transformer より小さくなること。

#### 成果物

- `snnai/benchmarks/energy_quantification.py` の改訂
- エネルギー計算用ユーティリティ

---

### Phase 6.6 — 性能最適化と最終リリース（2 週間）

**目的**: 実用的な速度・品質・効率を達成し、v6.x 最終版をリリースする。

#### タスク

1. **TorchScript / ONNX エクスポート**
   - 推論専用モデルの軽量化。

2. **量子化・枝刈りの統合**
   - 学習後に INT8 量子化と構造枝刈りを適用。

3. **常時稼働 AI パイプラインとの統合**
   - v5.5 の `AlwaysOnAIV1` への組み込み検討。

4. **最終ベンチマーク**
   - tiny Shakespeare での定量的最終評価。
   - WikiText-2 などのより大きなコーパスへの拡張。

#### 成功基準

- 推論 latency < Transformer latency。
- 生成テキストが文脈的に一貫した短い文章を出力すること。
- v6.1.0 タグを作成してリリース。

---

## 優先順位とマイルストーン

| フェーズ | 目標 | 目標到達指標 |
|---|---|---|
| 6.1 | 根本原因特定 | spike 発火率 > 0、勾配非ゼロを確認 |
| 6.2 | SNN 学習開始 | 5 epoch で loss < 4.17 |
| 6.3 | 安定的学習 | val ppl < 40 |
| 6.4 | 公平な比較 | SNN ppl ≤ 1.5 × Transformer ppl |
| 6.5 | エネルギー主張の裏付け | SNN energy < Transformer energy |
| 6.6 | 最終リリース | latency < Transformer、v6.1.0 リリース |

---

## テスト計画

各フェーズで以下のテストを追加/更新する。

- `tests/test_v61_diagnostics.py` — 診断メトリックの出力確認
- `tests/test_v62_architecture.py` — spike 発火と勾配の存在確認
- `tests/test_v63_training.py` — 学習曲線の単調減少確認
- `tests/test_v64_metrics.py` — 生成メトリックの計算確認
- `tests/test_v65_energy.py` — spike カウントとエネルギー > 0 の確認
- `tests/test_v66_release.py` — 推論 latency と TorchScript エクスポート確認

---

## Kaggle 実行計画

- 各フェーズの最終検証は **Kaggle T4** で実行。
- push 時は必ず `acc="NvidiaTeslaT4"` を指定。
- モデルが改善されたら `models/snnai_v6_large_lm.pt` を更新。
- 学習ログと生成サンプルを `docs/snnai_test_results.md` に追記。

---

## 参考: version 10 の教訓

- **GPU 指定**: `acc="NvidiaTeslaT4"` が唯一有効な方法。
- **デバイス処理**: SNN 内部で `torch.zeros(..., device=x.device)` を徹底。
- **学習しない症状**: loss 平坦 + spike 0 + 生成空白は、出力層や LIF パラメータに問題がある可能性が高い。
