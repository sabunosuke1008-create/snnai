# SNNAI v6.6.2+ 性能向上ロードマップ（厳密版）

**Version**: v6.6.2 以降（次期リリース候補）  
**Date**: 2026-07-11  
**Repository**: https://github.com/sabunosuke1008-create/snnai  
**作成根拠**: `docs/HANDOVER.md`、`docs/roadmap_v65.md`、`docs/snnai_test_results.md`（特に §18 / §21 / §22）の再検証、および Kaggle T4 での `fair_compare()` 実測（§0 検証結果）。

---

## 0. 作業開始前の事実確認（検証結果）

本ロードマップ策定前に、既存ドキュメントの「Phase 6.5〜7.0 完了」という記述に対して事実確認を行った。結論：**ローカルテスト（192 passed）と Bio-NAS プロキシは通るが、実コーパスでの「全部乗せ」性能は未測定どころか、組み合わせると学習不能であることが判明した。**

### 0.1 実コーパス・全部乗せ性能の測定（Kaggle T4 で実施）

`LargeScaleSNNLM(use_sequence_recurrent=True, use_positional_encoding=True, use_hippocampus_gate=True, use_spiking_attention=True)` の全機能 ON 構成で、`fair_compare()` を v6.5.7 と同一条件（BPE vocab=2048、WikiText-2 + Tiny Shakespeare、seq_len=128、batch=32、time_steps=20、epochs=5、T4）で実行。

**測定結果（2026-07-11, kernel `gihuhi/snnai-v6-6-2-fair-compare-all-features` v1, 中断時点）**:

| 項目 | 値 |
|---|---|
| corpus length | 14,467,591 |
| BPE vocab | 2048 |
| device | cuda（T4）|
| SNN parameters | 3,592,448 |
| Transformer parameters | 8,407,552（※SNN の 2.3 倍＝公平性条件違反、§4.2 参照）|
| Epoch 0 SNN train/val_loss | **nan / nan** |
| Epoch 0 SNN firing_rate | **0.0002（死んだ SNN）** |
| Epoch 1,2 | 同様に nan、firing_rate 0.0000 |

**結論**: 「全部乗せ」構成は学習初期から NaN 発散し、LIF が事実上発火しない（dead neuron）。既存ドキュメントの「Phase 6.6/6.7 完了」は**単体テスト・プロキシ評価のみの完了**であり、実コーパス統合検証は行われていなかった（懸念は的中）。

### 0.2 HippocampalMemory のバッチ/GPU 対応状況（コード読解で確認）

`associative_memory.py` / `hippocampus_gate.py` を確認：

- **GPU 対応**: `keys/values/count` は `register_buffer` で保持され、`model.to(device)` で正しく転送される。batched matmul（`torch.matmul(query, active.t())`）もバッチ次元を前提としている。**→ 技術的には GPU/バッチ対応済み**（roadmap_v65.md の「現状バッチ非対応・GPU非対応」リスクは Phase 6.6 で解消済み）。
- **ただし重大な安全性リスクあり**: メモリは `capacity`（既定 64）の**グローバル共有バッファ**で、`store=True` 毎に蓄積し、容量超過後は上書きされず破棄（`n<=0` で return）。`LargeCorpusTrainer` は forward 毎に `store=True` を呼ぶが**エポック・系列間で `reset_memory()` を呼ばない**。結果：
  1. 訓練中の全系列が同じメモリを共有 → **train/val リークの危険**。
  2. 容量 64 で飽和すると以降の retrieval が陳腐化。
  3. §0.1 の NaN/死ニューロンに Hippocampus が寄与している可能性が高い（retrieved が 0 固定→ゲートが恒等写像に近づく／勾配が不正に流れる）。

**結論**: GPU/バッチは動くが、**エポック・系列間リセットと容量設計が未解決**であり、大規模学習のボトルネック・数値不安定の原因になり得る。最優先で修正する。

### 0.3 `test_bio_nas.py` の pre-existing failure の扱い（決定）

`test_evolution_finds_score_at_least_as_good_as_serial` は `snnai/bio_nas/evaluator.py:62` で `KeyError`（Key は実行毎に `'m2'`/`'m4'`/… と変化＝非決定論的）を起こす。v6.0 から放置。

**決定: 新ロードマップの正式な修正タスク（Phase 6.16）として含める。** 理由：
- 本ロードマップは「3 シード平均・分散」による統計的信頼性を必須とするが、確率的に落ちるテストがあると CI が不安定になり、改善が誤差か否かの判定自体ができなくなる。
- 修正は小規模（進化で生成された DAG のトポロジカルソート順序の保証、または `cache` への存在チェック + フォールバック）。放置すると本ロードマップの信頼性基盤を蝕ぐ。

---

## 1. 現状分析（クリティカルなギャップ）

| 項目 | 実測 / 状態 | 備考 |
|---|---|---|
| v6.5.7（+recurrent,+pos のみ）SNN val_ppl | 68.74 / TF 33.04 | §18。5 epochs、実コーパス |
| v6.6.2 全部乗せ SNN val_ppl | **nan（学習不能）** | §0.1 本測定 |
| Bio-NAS プロキシ val_ppl | 4.85e8（非現実） | §22。合成ランダムデータのため無意味 |
| SNN firing_rate（全部乗せ） | ~0.0002（死） | §0.1 |
| 公平性（パラム数 SNN/TF） | 3.59M / 8.41M（不公平） | §0.1 |
| 統計的信頼性 | 全記録単一シード | 改善が誤差か判定不可 |
| Hippocampus メモリ安全性 | リーク・容量未解決 | §0.2 |

**核心的ボトルネック**: 個別機能は（単体テスト上）動くが、**統合すると数値発散・死ニューロン**になる。つまり「機能がある」≠「性能が出る」。したがって次のフェーズは「統合安定化 → 実測アブレーション → 知識蒸留で Transformer に近づける」の順で進める。

---

## 2. クリティカルパス

```
[Phase 6.10] 実コーパス・アブレーション基盤 + 安定化学習（NaN/死ニューロン修正）
       ↓（現在地の確定と各機構の寄与量化）
[Phase 6.11] Hippocampus メモリのエポック/バッチ安全性（リーク修正・容量見直し）
       ↓
[Phase 6.12] 知識蒸留（Transformer 教師からの soft-label / 中間表現蒸留）
       ↓
[Phase 6.13] SSA の安定化・高性能化（spike-attention の学習安定化・効率化）
       ↓
[Phase 6.14] Bio-NAS 本実行（実コーパス・実 LM + 蒸留込みアーキテクチャ探索）
       ↓
[Phase 6.15] 統計的信頼性運用（3 シード平均・分散の自動化 + 公平性チェック）
       ↓
[Phase 6.16] test_bio_nas 既存バグ修正（KeyError 順序依存）
```

6.10 と 6.11 は並行可能（6.11 のメモリ修正は 6.10 の安定化の一部としても組み込める）。

---

## 3. フェーズ構成

### Phase 6.10: 実コーパス・アブレーション基盤 + 安定化学習（最優先）

**目的**: §0.1 の NaN/死ニューロンを解消し、v6.5.7 と同一条件で 5 パターンのアブレーション表を作り、どの機構が val_ppl・BLEU-1・latency・energy に寄与するかを定量化する。これが以降の全フェーズの「現在地」となる。

**変更内容（対象ファイル・関数レベル）**:
1. `snnai/modules/language/large_scale_lm.py`
   - `forward()` の各 `t` ステップで `use_hippocampus_gate` 時に `store=True` する現状を修正：`LargeCorpusTrainer` が系列単位で `model.hippocampus_gate.reset_memory()` を呼ぶフックを追加（または `forward(x, reset_memory_every=None)` 引数）。
   - SSA 適用前の `cur` が「バイナリスパイク」であるため、`spiking_attention` への入力を **membrane（実数）出力**にするオプション `ssa_input='membrane'` を追加（バイナリ入力だと勾配が 0 に近く学習が停滞しやすい）。
   - 勾配崩壊対策：出力 `lif_out` の `spike_grad` を `fast_sigmoid()` から `atan()` / `super_spike()` に変更検討、または初期学習率を `1e-3`→`5e-4` に下げ、`max_grad_norm` を 1.0 維持。
2. `snnai/modules/language/spiking_attention.py`
   - `_normalize()` の `var.sqrt().clamp_min(1e-5)` によるゼロ除算／発散を `clamp_min(1e-3)` に緩和し、かつ初期 `threshold` を 1.0→`num_heads` 等にスケール。
3. `snnai/benchmarks/transformer_comparison.py`
   - `fair_compare()` に `seeds: list[int]` 引数を追加し、複数シードで SNN のみ（または両方）を回して mean±std を返すようにする。
   - **公平性**: `TransformerBaseline` の `d_model/num_layers` を SNN と**同パラメータ数グループ**になるよう自動決定するヘルパを追加（§4.2 遵守）。
4. 新規 `environment/kaggle_fair_compare/ablation.ipynb`（5 パターン × 3 シードを 1 カーネルでループ実行し JSON に集約）。

**アブレーション表（測定テンプレート）**:

| パターン | recurrent | positional | hippocampus | attention | 期待（測定待ち） |
|---|---|---|---|---|---|
| P0 baseline | – | – | – | – | val_ppl 基準 |
| P1 +recurrent+pos | ✓ | ✓ | – | – | v6.5.7 = 68.74 の再現確認 |
| P2 +hippocampus | ✓ | ✓ | ✓ | – | P1 比 ±? |
| P3 +attention | ✓ | ✓ | – | ✓ | P1 比 ±? |
| P4 all | ✓ | ✓ | ✓ | ✓ | **現状 nan → 修正後測定** |

**成功基準**:
- 5 パターン全てが NaN なしで学習完了すること（firing_rate が 0.05〜0.20 程度に維持）。
- 各パターンについて **3 シードの mean±std** を `val_ppl / BLEU-1 / latency / energy / GER` で報告。
- P1 が v6.5.7 の 68.74 と誤差 5% 以内で再現すること（パイプライン検証）。

**想定リスクと現実的限界**:
- P4 の安定化に複数回の Kaggle 試行が必要（SSA+Hippocampus の相互作用が非自明）。
- 1 パターン約 15 分 × 5 × 3 シード ≈ 225 分。Kaggle 9 時間制限内だが、1 セッションに収めるならエポック数を 5 に固定し、探索用は小パラメータ（embed=64,hidden=256,layers=2）で高速化し、最終確認のみフルサイズ。
- 既知の通り SNN は GPU シミュレーションで Transformer より数倍遅い（理論エネルギー効率での主張が主目的）。

**依存関係**: なし（最優先）。6.11 と並行可。

---

### Phase 6.11: Hippocampus メモリのエポック/バッチ安全性

**目的**: §0.2 の train/val リークと容量飽和を解消し、系列・エポック単位できれいにリセットできるようにする。

**変更内容**:
1. `snnai/modules/hippocampus/associative_memory.py`
   - `reset_memory()` を `LargeCorpusTrainer` の各系列（またはエポック）開始時に呼ぶ契約を明文化。
   - 容量飽和時に FIFO 上書き（最古を捨てる）か、retrieval 時に `count` ではなく直近 N 系列のみを対象にするオプションを追加。
2. `snnai/benchmarks/large_corpus_trainer.py`
   - `train()` 内で forward 前に `model.hippocampus_gate.reset_memory()` を呼ぶ（または `reset_memory_every='epoch'` 引数）。
3. `tests/test_v66_hippocampus_integration.py` に「バッチ>1 で reset 後にクリーンに再開できる」「train/val でメモリが分離される」ことを確認するテストを追加（batch=4, GPU 実行はローカル CUDA があれば、なければ Kaggle で確認）。

**成功基準**:
- `reset_memory()` 呼び出し後に `count==0` となり、同一入力で store→retrieve が確定的に動作する。
- train/val 間でメモリが漏れないこと（val 精度が train に依存しない）を確認。
- P2/P4 の val_ppl が P1 より悪化しない（リーク是正で寧ろ改善するはず）。

**想定リスク**: リセット頻度が高いと長距離文脈の恩恵が減る（トレードオフ）。容量・リセット単位は P4 のアブレーションでチューニング。

**依存関係**: 6.10（6.10 の安定化に内包可能）。

---

### Phase 6.12: 知識蒸留（Knowledge Distillation）

**目的**: SpikeBERT / SpikeLM / SpikeLLM 等の知見通り、Transformer baseline を**教師**とし、SNN に soft-label 蒸留または中間表現マッチングを行って Transformer 並みの性能に近づける。既存ロードマップで**完全に欠落していた観点**。

**変更内容**:
1. `snnai/benchmarks/distillation.py`（新規）
   - `TransformerBaseline` を教師とし、SNN のロジットと教師の `softmax(T=tau)` との KL 蒸留損失を実装。
   - 中間表現マッチング：`use_spiking_attention` 出力／各 LIF の membrane と教師の隠れ状態の MSE（hint layer）をオプションで追加。
   - タスク損失（CE）＋ `alpha * KL` の重み付け、温度 `tau` はハイパラ。
2. `snnai/benchmarks/transformer_comparison.py`
   - `fair_compare()` に `distill=True/teacher_model/tau/alpha` 引数を追加し、蒸留あり/なしを比較可能に。
3. 教師は Phase 6.10 で「同等パラメータ数グループ」にサイズ調整済みの `TransformerBaseline`。

**成功基準**（数値目標）:
- 蒸留あり SNN の val_ppl が非蒸留 SNN（P1/P4 の最良）から **30% 以上低下**（例: 68.74 → 48 目標、最終 50 以下を目指す）。
- BLEU-1 が 0.15 以上（現状 0.10 付近）に向上。
- latency は現状維持または 2 倍以内、energy は SNN のまま（理論優位を維持）。
- GER を定量的に報告（蒸留で文法的な生成が改善するか確認）。

**想定リスクと現実的限界**:
- 既存研究でも Transformer に完全同等は未達。目標は「近づける」こと（ppl 50 以下、BLEU 0.15+ を現実ライン）。
- 教師と生徒で出力形式が異なる（spike/real）ため、logit-level 蒸留が主眼。中間表現マッチングは membrane↔hidden の次元整合が必要。
- 蒸留は 2 模型を同時に保持するためメモリ 2 倍。Kaggle T4（15GB）ではフルサイズ同時保持が厳しい場合、教師を凍結・勾配非計算、または小モデルで検証。

**依存関係**: 6.10（生徒 SNN が安定学習できること）、6.11（任意）。

---

### Phase 6.13: SSA の安定化・高性能化

**目的**: Phase 6.7 の Spiking Self-Attention を、単に「動く」から「性能を出す」へ引き上げる。§0.1 の死ニューロンの主要因の疑いがあるため、ここで専用にチューニング。

**変更内容**:
1. `spiking_attention.py`
   - 入力をバイナリスパイクではなく membrane/real にする（`ssa_input='membrane'` オプション、6.10 で追加）。
   - 複数 head 時の head 間相互作用／残差接続（`cur = cur + attn_out`）の追加で勾配流を安定化。
   - 因果マスク下での学習安定性（初期スコアが全 0 に近いと softmax 的に平ら→勾配消失）への対処：スコアに学習可能バイアスを加える。
2. `large_scale_lm.py`
   - SSA の出力を次層へ渡す前のレイヤ正規化を追加。

**成功基準**:
- P3（+attention）が P1 に対して val_ppl で **10% 以上改善**（または同等以上を維持しつつ latency 増を 2 倍以内に抑える）。
- firing_rate が 0.05〜0.20 に維持され、死ニューロンが発生しない。

**想定リスク**: SSA は GPU 上では疎性を活かせず遅い。速度優位は neuromorphic ハード想定。精度寄与が小さい場合は Bio-NAS で自動除外される設計で良い。

**依存関係**: 6.10（安定化の基盤）。

---

### Phase 6.14: Bio-NAS 本実行（実コーパス・実 LM + 蒸留）

**目的**: Phase 7.0 のプレースホルダーを実の探索に昇格。合成ランダムプロキシ（val_ppl 4.8e8）ではなく、**実 WikiText-2 コーパスで実 `LargeScaleSNNLM` を評価**し、蒸留込みでアーキテクチャを探索。

**変更内容**:
1. `snnai/bio_nas/lm_evaluator.py`
   - `ProxyLm` の評価を「プロキシ・データセット」から「実コーパスでの短い `fair_compare` 相当（数エポック）」に切り替え（計算量のためエポック数を絞る）。
   - 目的関数に `val_ppl, latency, energy, BLEU-1, GER, distillation_gain` を組み込む。
2. `environment/kaggle_large_scale/notebook.ipynb` を Bio-NAS 本実行用に更新（蒸留あり設定）。

**成功基準**:
- Bio-NAS が少なくとも 3 種類の LM 層タイプから最適構成を選ぶ（既にプロキシで達成済み→実コーパスでも確認）。
- 探索構成が手動設計（P1/P4 最良）を val_ppl または BLEU-1 で上回る。
- Kaggle 9 時間制限内で 1 世代の評価が完結（小規模 proxy で探索、上位候補のみ実コーパス短評価）。

**想定リスク**: 実コーパス評価は重い。探索空間が広いと収束しない。→ 2 段階評価（proxy で絞り込み→実コーパスで最終確認）。

**依存関係**: 6.10, 6.12（蒸留モデルを探索対象に）。

---

### Phase 6.15: 統計的信頼性運用の自動化

**目的**: 単一シード実行の慣習を廃し、全ての性能主張を 3 シード mean±std で裏付ける。

**変更内容**:
1. `fair_compare()` / `distillation` に `seeds` 引数（6.10/6.12 で追加済み）の運用を標準化。
2. `docs/` への結果記録フォーマットを「mean±std」に統一（既存 `snnai_test_results.md` の過去記録は単一シードと明記）。
3. `tests/` に「同一シードで 2 回走らせて val_ppl が ±誤差内で再現する」軽量回帰テストを追加（ローカル小設定）。

**成功基準**: 以降の全フェーズ結果が 3 シード mean±std で報告されていること。改善が「1 シードのノイズ」でないことが検定可能。

**想定リスク**: 計算コスト 3 倍。Kaggle 制約内で小設定でのみ 3 シード、フル設定は 1 シード＋過去再現性で代替。

**依存関係**: 6.10 以降全て。

---

### Phase 6.16: `test_bio_nas` 既存バグ修正（KeyError 順序依存）

**目的**: v6.0 から放置の `test_evolution_finds_score_at_least_as_good_as_serial` の `KeyError` を修正し、CI を確定的にする。

**変更内容**:
- `snnai/bio_nas/evaluator.py:62` 付近：`evaluate_architecture` が `cache[p]` を参照する前に `p in cache` を確認し、不在時はスキップまたは順序をトポロジカルソートで保証。
- 非決定論（乱数シード依存で Key が変化）を除去するため、進化で生成された DAG の評価順序を `networkx.topological_sort` 等で決定論化。

**成功基準**: 該当テストが単独・フルスイート双方で確定的に PASS（§21 で記録済みの pre-existing failure が解消）。

**想定リスク**: 順序保証の変更が探索結果に微影響（許容範囲）。

**依存関係**: なし（いつでも可、最優先すべき低コスト修正）。

---

## 4. 優先順位と依存関係

| 優先度 | フェーズ | 依存 | 理由 |
|---|---|---|---|
| **P0** | 6.10 実コーパス・アブレーション＋安定化 | なし | 現在地確定と NaN 修正が全ての前提 |
| **P0** | 6.11 Hippocampus 安全性 | 6.10（内包可） | リーク解消が性能・信頼性の前提 |
| **P0** | 6.16 test_bio_nas 修正 | なし | 低コスト・CI 信頼性基盤 |
| **P1** | 6.12 知識蒸留 | 6.10 | Transformer 並み性能への主手段 |
| **P1** | 6.13 SSA 高性能化 | 6.10 | 死ニューロン解消後の本格活用 |
| **P2** | 6.14 Bio-NAS 本実行 | 6.10, 6.12 | 探索の実コーパス昇格 |
| **P2** | 6.15 統計的信頼性 | 6.10 以降 | 全主張の裏付け |

---

## 5. 公平ベンチマーク設計（遵守事項）

`docs/roadmap_v65.md` §4.2 の公平性条件を全フェーズで維持：

- **同一コーパス**: WikiText-2 raw + Tiny Shakespeare（14,467,591 文字）、**BPE vocab 2048**。
- **同一 tokenizer / seq_len(128) / batch_size(32) / time_steps(20)**。
- **同一パラメータ数グループ**: §0.1 で SNN 3.59M / TF 8.41M は不公平と判明。`TransformerBaseline` の `d_model/num_layers` を SNN と同規模（±10%）に自動調整して比較（Phase 6.10 で実装）。
- **GPU は T4 固定**。P100 割り当て時は CPU fallback（compute capability < 7.0 判定）。

### 測定項目（既存指標と整合）

| 指標 | 測定方法 |
|---|---|
| val_ppl | `fair_compare()` の val loss の exp |
| BLEU-1 / CER | `evaluate_generation()` |
| GER | `grammar_error_rate()`（Phase 6.8 追加済み）|
| latency | `compare_models()` の wall-clock（warmup 後平均）|
| energy | `quantize_energy()`（spike count ベース）|
| firing_rate | 各層 LIF の発火率（死ニューロン監視用、Phase 6.10 で必須ログ）|

---

## 6. Kaggle 制約内での実行計画

- **連続 9 時間 / 週 30 時間**制限。各フェーズは 1 セッション完結を目標。
- **計算量見積もり（T4, epochs=5）**:
  - 1 パターン SNN+TF 学習 ≈ 15 分。
  - アブレーション 5 パターン × 3 シード ≈ 225 分（約 3.75h）→ 1 セッション内。探索用は小パラメータ（embed=64,hidden=256,layers=2）で高速化し、最終確認のみフル（embed=128,hidden=512,layers=3）。
  - 蒸留（6.12）は教師＋生徒でメモリ 2 倍。T4 15GB では小モデルで検証し、フルはエポック数削減。
- **ノートブック検証**: push 前に `environment/kaggle_large_scale/validate_notebook.py`（Phase 6.9）で `nbformat.validate()` + `ast.parse()` + `\n` エスケープ検出を通す（§22 で導入済み）。

---

## 7. 現実的な限界

1. **SNN はまだ Transformer に追いついていない** — 蒸留でも ppl 同等は未達（目標は 50 以下、BLEU 0.15+）。
2. **GPU シミュレーションの遅さ** — time_steps=20 の for-loop 展開で Transformer の数倍遅い。速度優位は neuromorphic ハード（Loihi/NorthPole）想定の理論エネルギー効率として主張。
3. **Hippocampus のスケーラビリティ** — 長系列では容量/検索コストがボトルネック（6.11 で容量・リセット単位をチューニング）。
4. **3 シードでの計算コスト** — Kaggle 制約内でフル設定の 3 シードは困難。小設定 3 シード＋フル 1 シードで代替。
5. **NaN/死ニューロンの根本原因** — §0.1 の通り「全部乗せ」は現在学習不能。6.10 で解消するまで性能比較自体が成立しない（だから 6.10 が最優先）。

---

## 8. 即座に始められる次のアクション

1. **Phase 6.10 の安定化修正**を `large_scale_lm.py` / `spiking_attention.py` に適用（hippocampus リセットフック、`ssa_input='membrane'` オプション、normalize の clamp 緩和、学習率低下）。
2. `fair_compare()` に `seeds` と「同等パラメータ数 Transformer」ヘルパを追加。
3. `environment/kaggle_fair_compare/ablation.ipynb` を 5 パターン × 3 シードループ版に拡張し、T4 で実行（§0.1 のカーネルを update して再利用）。
4. 結果を `docs/snnai_test_results.md` の新セクションに mean±std で記録。
5. 並行して Phase 6.16（`test_bio_nas` 修正）を低コストで完了させ、CI を確定的にする。

---

## 9. リファレンス

- Zhou et al., "Spikformer", ICLR 2023.
- Lv et al., "SpikeBERT", OpenReview 2024.
- Xing et al., "SpikeLM", ICML 2024.
- Xing et al., "SpikeLLM", 2024.
- Gonzalez et al., "SAPU-LM", Zenodo 2026.
- "NeuronSpark", arXiv 2026.
- 既存: `docs/roadmap_v65.md`（Phase 6.5〜7.0）、`docs/snnai_test_results.md`（§18/§21/§22）。
