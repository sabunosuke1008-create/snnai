# SNNAI 開発引き継ぎ書

**作成日**: 2026-07-11
**作成時点でのバージョン**: v6.6.1
**対象読者**: このプロジェクトを一切知らない状態で引き継ぐ AI エージェント
**目的**: プロジェクトの全体像・現在の状態・進め方を完全に理解させ、次のステップ（Phase 6.7〜7.0 残り）にシームレスに移行できるようにする

---

## 0. 最速キャッチアップ（30秒で読む）

- **SNNAI** = Spiking Neural Network + AI。生物学的妥当性のあるスパイキングニューラルネットワーク（SNN）で Transformer レベルの言語モデルを目指す研究プロジェクト。
- **リポジトリ**: https://github.com/sabunosuke1008-create/snnai
- **作業ディレクトリ**: `C:\Users\otame\Downloads\SNN AI`
- **最新バージョン**: `v6.6.1`（タグ済み、GitHub に push 済み）
- **最新 Kaggle カーネル**: `gihuhi/snnai-v6-6-1-bio-nas-hippocampus-demo`（version 2、COMPLETE）
- **テスト結果**: 166 passed / 1 pre-existing failure（`test_bio_nas.py::test_evolution_finds_score_at_least_as_good_as_serial`、v6.0 から存在するバグ、本変更とは無関係）
- **ロードマップ**: `docs/roadmap_v65.md`（Phase 6.5〜7.0）、`docs/roadmap_v64.md`、...
- **過去の Kaggle 実行履歴**: `docs/snnai_test_results.md`（section 1〜19）

**次の未着手タスク（最優先）**:
1. Phase 6.7: Spiking Self-Attention (SSA) 実装
2. Phase 6.8: 評価指標拡充（GER, topic drift, repetition rate, newline rate）
3. Phase 6.9: Kaggle ノートブックパイプライン信頼性向上

---

## 1. プロジェクト概要

### 1.1 目標
- **SNN で Transformer レベルの言語モデルを実現する**ことが最終目標
- 現状は GPU シミュレーション上の PyTorch + snntorch 実装
- 将来的には neuromorphic ハードウェア（Intel Loihi、IBM NorthPole 等）への展開を見据えた「理論エネルギー効率」の主張

### 1.2 ロードマップ全体像

| フェーズ | 内容 | 状態 | バージョン |
|---|---|---|---|
| v2.0 | Phase 12（SNNAI v2.0 基盤） | ✅ 完了 | v2.0 |
| v2.1〜v2.5 | Stage 1: 省電力エッジ AI | ✅ 完了 | v2.1〜v2.5 |
| v2.6〜v3.5 | Stage 2: 小規模モジュール型言語パイプライン | ✅ 完了 | v2.6〜v3.5 |
| v3.6〜v4.5 | Stage 3: テキスト生成 | ✅ 完了 | v3.6〜v4.5 |
| v4.6〜v5.5 | Stage 4: LLM 支援・常時稼働 AI | ✅ 完了 | v4.6〜v5.5 |
| v5.6〜v6.0 | Stage 5: Transformer 並み性能への挑戦 | ✅ 完了 | v5.6〜v6.0 |
| Phase 6.5 | 系列軸リカレント + Embedding | ✅ 完了 | v6.5.0 |
| Phase 6.6 | Hippocampus 外部記憶統合 | ✅ 完了 | **v6.6.1** |
| Phase 6.7 | Spiking Self-Attention | ⏳ **次のタスク** | 未着手 |
| Phase 6.8 | 評価指標拡充 | ⏳ 次のタスク | 未着手 |
| Phase 6.9 | Kaggle パイプライン信頼性向上 | ⏳ 次のタスク | 未着手 |
| Phase 7.0 | Bio-NAS による LM アーキテクチャ探索 | ✅ 完了（v6.6.0） | v6.6.0 |

注: Phase 7.0 は実装上 v6.6.0 として先行リリース済み（プレースホルダー実装）。Phase 6.6〜6.9 で実レイヤ・実評価を段階的に置き換えていく。

### 1.3 主要な構造的弱点（v6.4.6 時点での実測）

| 項目 | SNN | Transformer | 備考 |
|---|---|---|---|
| val perplexity | 161.18 | 47.90 | 同条件比較 |
| latency | 0.0607 s | 0.00189 s | SNN は約 32 倍遅い |
| parameters | 1,903,616 | 4,209,664 | SNN の方が少ないのに精度・速度とも劣る |
| energy | 7.33e-4 J | N/A | spike count = 733,240 |
| BLEU-1 | 0.107 | 0.259 | 生成品質にも大きな差 |

→ この差を埋めるのが Phase 6.5〜7.0 の目的。

---

## 2. ファイル構造（重要ファイル一覧）

### 2.1 コアモデル

| ファイル | 役割 |
|---|---|
| `snnai/modules/language/large_scale_lm.py` | `LargeScaleSNNLM`（メインの SNN 言語モデル）。`use_sequence_recurrent`, `use_positional_encoding`, `use_hippocampus_gate` のフラグで機能を切り替えられる。 |
| `snnai/modules/language/hippocampus_gate.py` | `HippocampusGate`（Phase 6.6 で追加）。`HippocampalMemory` + 学習可能 sigmoid ゲート。`reset_memory()` で記憶をクリア。 |
| `snnai/modules/language/bpe_tokenizer.py` | `BPETokenizer`（v6.4 で導入）。文字レベルではなく BPE サブワード。 |
| `snnai/modules/language/tokenizer.py` | `CharTokenizer`, `SpikeEncoder`（旧来の文字レベル） |
| `snnai/modules/hippocampus/associative_memory.py` | `HippocampalMemory`（Phase 6.6 で利用）。キー・値ストア + Hopfield 風検索。 |

### 2.2 ベンチマーク・評価

| ファイル | 役割 |
|---|---|
| `snnai/benchmarks/large_corpus_trainer.py` | `LargeCorpusTrainer`, `CharLMDataset`, `WarmupCosineSchedule`, `collate_fn`。大規模コーパスでの学習ループ。 |
| `snnai/benchmarks/transformer_comparison.py` | `TransformerBaseline`, `compare_models`。Transformer ベースラインとの比較。`parallel_strategy='none'\|'dp'\|'ddp'` 対応（DP/DDP は stateful SNN と非互換のため使用不可）。 |
| `snnai/benchmarks/energy_quantification.py` | `quantize_energy`。SNN のエネルギー推定（joules, spikes, latency）。 |
| `snnai/benchmarks/generation_metrics.py` | `evaluate_generation`。BLEU-1、CER、生成サンプル。`repetition_penalty`, `top_k`, `top_p` 対応。 |
| `snnai/benchmarks/homeostatic_loss.py` | `HomeostaticRegularizer`。LIF 発火率を 0.12 に保つ正則化。 |
| `snnai/benchmarks/parallel_utils.py` | DataParallel/DDP ヘルパー（参考実装、Kaggle では動作不可）。 |

### 2.3 Bio-NAS（Phase 7.0）

| ファイル | 役割 |
|---|---|
| `snnai/bio_nas/search_space.py` | 元の生物学的モジュール検索空間（c_elegans, honeybee, crow, octopus）。 |
| `snnai/bio_nas/evaluator.py` | 元の surrogate エバリュエーター（合成データでアーキテクチャ評価）。 |
| `snnai/bio_nas/evolution_search.py` | 元の進化探索（分類タスク用）。 |
| `snnai/bio_nas/lm_search_space.py` | LM 層タイプ検索空間（feedforward, recurrent, attention, hippocampus_gate + 生物モジュール）。`LmArchitecture`, `LmLayer`。 |
| `snnai/bio_nas/lm_evaluator.py` | LM プロキシ評価器。`ProxyLm` + 多目的メトリクス（val_ppl, latency, energy, BLEU-1, bio_penalty）。**Phase 6.6 で hippocampus_gate が実 `HippocampusGate` に置き換え済み**。 |
| `snnai/bio_nas/lm_evolution.py` | `LmEvolutionSearch`（エリート保存 + スカラー化 + Pareto front）。 |
| `snnai/bio_nas/__init__.py` | すべての公開 API をエクスポート。 |

### 2.4 Kaggle 関連

| ファイル | 役割 |
|---|---|
| `environment/kaggle_large_scale/notebook.ipynb` | メインの Kaggle ノートブック（現在は Bio-NAS デモ）。v6.6.0 で Bio-NAS に書き換え済み。 |
| `environment/kaggle_large_scale/kernel-metadata.json` | カーネルメタデータ。**`id` は実際の Kaggle スラッグと完全一致必須**（不一致は HTTP 409 Conflict）。 |
| `environment/kaggle_large_scale/train_ddp.py` | DDP トレーニングスクリプト（Kaggle ノートブックでは動作不可、参考実装）。 |

### 2.5 ユーティリティ

| ファイル | 役割 |
|---|---|
| `snnai/utils/download_corpus.py` | `download_wikitext2()`（Hugging Face ミラー使用）。戻り値は `Path`（`str()` で文字列変換必要）。 |

### 2.6 テスト

| ファイル | テスト内容 |
|---|---|
| `tests/test_v65_large_scale_recurrent.py` | Phase 6.5（系列軸リカレント + Embedding） |
| `tests/test_v66_hippocampus_integration.py` | Phase 6.6（Hippocampus 統合、9 テスト） |
| `tests/test_v70_bio_nas_lm.py` | Phase 7.0（Bio-NAS LM、13 テスト） |
| `tests/test_v63_version_sync.py`, `tests/test_v64_version_sync.py` | バージョン同期チェック |
| `tests/test_bio_nas.py` | 元の Bio-NAS（**pre-existing failure**: `test_evolution_finds_score_at_least_as_good_as_serial`、v6.0 から存在） |

### 2.7 ドキュメント

| ファイル | 内容 |
|---|---|
| `docs/roadmap_v65.md` | Phase 6.5〜7.0 のロードマップ（**必読**） |
| `docs/snnai_test_results.md` | 全 Kaggle 実行履歴（section 1〜19） |
| `docs/snnai_v6_achievements.md` | v6.0 達成内容 |
| `README.md` | プロジェクト概要 |
| `VERSION` | 現在のバージョン（例: `v6.6.1`） |

---

## 3. 重要な規約・制約・落とし穴

### 3.1 バージョン管理

- **必ずバージョンを bump してから push する**
- bump 対象ファイル: `VERSION`, `README.md`（`## バージョン vX.Y.Z` の行）, `tests/test_v63_version_sync.py`, `tests/test_v64_version_sync.py`
- 新しいバージョンのタグを作成し、`git push origin main vX.Y.Z` で push
- **`docs/snnai_test_results.md` にも新しいセクションを追加する**

### 3.2 Kaggle CLI 規約

- **Kaggle CLI は必ず `kaggle-cli-mcp` を使う**（`python -m kaggle` は認証エラーになりがち）
- T4 GPU を指定: `kaggle_cli_mcp_kernels_push(folder="environment/kaggle_large_scale", acc="NvidiaTeslaT4")`
- **P100 が割り当てられた場合は CPU fallback**（`torch.cuda.get_device_capability(0)[0] < 7` で判定）
- カーネル `id` は実際のスラッグと完全一致必須（不一致は HTTP 409）
- 長いスラッグ（例: `snnai-v6-5-6-single-gpu-snn-lm-transformer-comparison`）は失敗しやすい → **短く保つ**（例: `snnai-v6-6-0`）
- 既存カーネルへの **update** で再実行すると環境セットアップが省略され、所要時間が短縮される

### 3.3 ノートブック JSON の落とし穴

- Python 文字列内の `\n` は JSON では `'\\n'` とエスケープ（**実際の改行を入れると Kaggle で `IndentationError`**）
- セル内の Python コードに複数行文字列を含む場合、JSON エディタで慎重に確認
- **`%%writefile` で外部スクリプトを書き出す**と `subprocess.run` で実行できる（Kaggle で独立した `.py` ファイルが確実に存在しない問題の回避策）

### 3.4 マルチ GPU の制約

- **DataParallel は snntorch LIF の stateful ニューロンと根本的に非互換**（v6.5.2/6.5.3 で確認済み、`tensor size 0 vs 512` エラー）
- **DDP via `torch.multiprocessing.spawn` は Kaggle ノートブック環境でハング**（v6.5.5 で 60+ 分経過しても出力ゼロ、NCCL 初期化で停止）
- **Kaggle では単一 GPU 実行が唯一の安定手段**

### 3.5 Bio-NAS の設計

- `is_valid()` は **DAG + 全ての非入力ノードが indegree >= 1 + 少なくとも 1 つの LM 層タイプ使用** を要求
- 突然変異でエッジ削除 → ノードが孤立 → `is_valid()` で除外される
- `ProxyLm.forward()` の `for node in self.order` ループで `preds` が空のノードはスキップされる（**`is_valid()` で防いでいる**）
- 評価関数は **スカラー化重み和**（デフォルト: `val_ppl: -0.4, latency_sec: -2.0, energy_proxy_joules: -200.0, bleu1: 2.0, biological_penalty: -1.0`）
- **Pareto front はスカラー化と独立に追跡**される

### 3.6 HippocampusGate の重要な実装詳細

```python
# 必ず .detach() してからメモリバッファに格納する
episode_key = x.mean(dim=1).detach()
episode_value = x.mean(dim=1).detach()
self.memory.store(episode_key, episode_value)

# メモリが空のときはゲートプロジェクションが勾配を受け取れるようゼロを使う
if self.memory.count.item() == 0:
    retrieved = torch.zeros_like(flat_x)
else:
    retrieved, _ = self.memory.retrieve(flat_x, top_k=1)
    retrieved = retrieved.squeeze(1)
```

`.detach()` を忘れると **「Trying to backward through the graph a second time」** エラーになる（バッファの in-place 変更が autograd グラフを破壊するため）。ゼロを使うのはゲートプロジェクションが常に勾配を受け取れるようにするため（Bio-NAS 学習で必須）。

### 3.7 ファイル書き込みの落とし穴

- `write` ツールは **既存ディレクトリ内のファイル作成で `EEXIST` エラー** を出すことがある（mkdir が既存で失敗）
- **回避策**: ヘルパースクリプトを `C:/Users/otame/snnai_temp/` に作成して実行、または bash heredoc で書き込む
- Windows でファイル名 `nul`, `nul.*` は予約デバイス名なので作成禁止

---

## 4. 現在の状態と次のステップ

### 4.1 直近で完了した作業（v6.6.1）

- ✅ Phase 6.6: `HippocampusGate` を `LargeScaleSNNLM` に統合（`use_hippocampus_gate=True` で有効化）
- ✅ Bio-NAS の `hippocampus_gate` プロキシをプレースホルダーから実 `HippocampusGate` に置き換え
- ✅ 9 つの新規テスト（`tests/test_v66_hippocampus_integration.py`）すべて pass
- ✅ Kaggle T4 で Bio-NAS デモ COMPLETE（version 2、~1 分）
- ✅ v6.6.1 を GitHub に commit/tag/push

### 4.2 Phase 6.7: Spiking Self-Attention (SSA) — **次のタスク**

**目標**: Spikformer / SpikeBERT / SpikeLM の知見を取り入れた、softmax 不要の spike-based attention を実装し、リカレント案・Hippocampus 案と比較する。

**やること**:
1. `snnai/modules/language/spiking_attention.py` を新規作成
   - Q/K/V をスパイクに変換し、`S(BN(MLP(Q·K^T·V)))` で attention map を近似（Spikformer 参考）
   - 因果マスク（言語モデル用）をサポート
2. `LargeScaleSNNLM` に `use_spiking_attention` フラグを追加
3. `snnai/bio_nas/lm_evaluator.py` の `attention` プレースホルダーを実 SSA に置き換え
4. `tests/test_v67_spiking_attention.py` を作成
5. Kaggle で Bio-NAS デモを再実行（v6.6.2 として）
6. `docs/snnai_test_results.md` にセクション追加

**成功基準**: SNN val_ppl ≤ 50、BLEU-1 ≥ 0.15、latency は現状維持または 2 倍以内。

### 4.3 Phase 6.8: 評価指標拡充

**目標**: perplexity / BLEU-1 / CER だけでなく、文法的妥当性や意味的一貫性を測定する。

**やること**:
1. `snnai/benchmarks/generation_metrics.py` に追加:
   - 文法エラー率 GER（`language_tool_python` / `spacy`）
   - 意味的一貫性（`sentence-transformers` 自己類似度、topic drift rate）
   - 繰り返し・退化指標（n-gram 繰り返し率、改行率）
2. `evaluate_generation` 返り値への組み込み
3. テスト追加
4. すべての指標が Kaggle 出力 JSON に含まれることを確認

### 4.4 Phase 6.9: Kaggle ノートブックパイプライン信頼性向上

**目標**: 過去の失敗（IndexError, ImportError, IndentationError 等）を繰り返さない CI 的検証を導入する。

**やること**:
1. `environment/kaggle_large_scale/validate_notebook.py` を作成:
   - `nbformat.validate()` で notebook JSON の妥当性チェック
   - 全セルの `ast.parse()` 実行
   - JSON 内 `\n` が実際の改行に置き換わっていないかチェック
   - `kernel-metadata.json` / title / 最初の markdown タイトルの整合性検証
2. `kaggle_cli_mcp_kernels_push` の **前** に自動実行する仕組み
3. テスト追加

---

## 5. 典型的な作業フロー

### 5.1 新しいフェーズの実装とリリース

```bash
# 1. ブランチ作成（main で直接作業しても OK）
cd "C:/Users/otame/Downloads/SNN AI"

# 2. コード変更（write ツールまたはヘルパースクリプト）

# 3. AST パースチェック（.py 変更後）
python -c "import ast; ast.parse(open('<file>.py', encoding='utf-8').read())"

# 4. テスト実行
python -m pytest tests/test_vXX_<feature>.py -v

# 5. 全テスト実行（リグレッション確認）
python -m pytest tests/ --ignore=tests/test_environment.py -q

# 6. バージョン bump（ヘルパースクリプト使用）
# VERSION, README.md, test_v63_version_sync.py, test_v64_version_sync.py を更新

# 7. notebook.ipynb を更新（必要に応じて）

# 8. Commit + tag + push
git add -A
git commit -m "feat(vX.Y.Z): Phase X.Y description"
git tag -a vX.Y.Z -m "Release vX.Y.Z: Phase X.Y description"
git push origin main vX.Y.Z

# 9. Kaggle push
# kernel-metadata.json の id を確認（実際のスラッグと一致）
kaggle_cli_mcp_kernels_push(
    folder="environment/kaggle_large_scale",
    acc="NvidiaTeslaT4"
)

# 10. ステータス監視（単一 MCP 呼び出しで十分）
kaggle_cli_mcp_kernels_status(kernel="gihuhi/snnai-vX-Y-Z")

# 11. COMPLETE 後、ログを取得して metrics を抽出
kaggle_cli_mcp_kernels_logs(kernel="gihuhi/snnai-vX-Y-Z")

# 12. docs/snnai_test_results.md にセクション追加

# 13. docs/snnai_test_results.md を commit/push
git add docs/snnai_test_results.md
git commit -m "docs: add vX.Y.Z results"
git push origin main
```

### 5.2 ヘルパースクリプトパターン

Windows + Git Bash 環境で `write` ツールが `EEXIST` を出す場合:

```python
# C:/Users/otame/snnai_temp/write_xxx.py
import os
content = File content here.