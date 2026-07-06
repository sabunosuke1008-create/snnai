# SNNAI v6.3.0 改良ロードマップ

## 背景

SNNAI v6.2.0 では Tiny Shakespeare に対する過学習を抑制し、検証損失が意味のある値（val ppl ≈ 2.15）に収まることを確認しました。しかし、生成テキストは依然として改行（`\n`）やスペースに強く偏っており、実用的な文字列生成には程遠い状態です。また、Kaggle notebook 内で WikiText-2 をダウンロードする際、`!wget` / `!unzip` による bash セルがパーミッションやパス指定の問題で失敗しています。さらにリポジトリ内のバージョン表記が `v6.2.x` のまま混在しており、`v6.3.0-dev` への移行が必要です。

v6.3.0 ではこれらの課題に対し、**恒常性正則化ロス**、**反復ペナルティ付きデコーディング**、**Python ネイティブなコーパス解凍**、**バージョン同期**の 4 本柱で取り組みます。

---

## 目標

1. **改行・スペース退化の克服**（最優先）
   - 学習時: LIF レイヤーの平均発火率をターゲット帯域（10%〜15%）に保つ `HomeostaticRegularizer` を追加し、SNN が「無活動」な改行・スペーストークンに沈み込むのを防ぐ。
   - 生成時: `repetition_penalty` を導入し、直近に生成されたトークン（特に改行・スペース）の logits を減算して多様性を向上させる。
2. **Kaggle WikiText-2 解凍エラーの解消**
   - `notebook.ipynb` の bash セルを廃止し、`requests` + `zipfile` によるクロスプラットフォームな Python 実装に置き換える。
3. **バージョン情報の同期**
   - `VERSION`、`README.md`、`environment/kaggle_large_scale/kernel-metadata.json`、notebook 内クローンタグを `v6.3.0-dev` に統一する。

---

## ステップ 1: 実装対象ファイルと変更計画

### 1.1 `snnai/benchmarks/homeostatic_loss.py`（新規）

- `HomeostaticRegularizer` クラスを新規作成。
  - 入力: LIF レイヤーから出力される `spk` テンソル（shape: `(batch, time_steps, hidden)` または `(time_steps, batch, hidden)`）。
  - 各レイヤーごとに時間方向の平均発火率 `firing_rate = spk.float().mean()` を計算。
  - 目標発火率 `target_firing_rate=0.12`（12%）からの二乗誤差 `loss = ((firing_rate - target) ** 2).mean()` を返す。
  - 全レイヤーの損失を合計し、`homeostatic_weight`（例: 1e-3）でスケーリングして主な `CrossEntropyLoss` に加算。

### 1.2 `snnai/benchmarks/large_corpus_trainer.py`（変更）

- `LargeCorpusTrainer.__init__` に以下を追加。
  - `homeostatic_weight: float = 1e-3`
  - `target_firing_rate: float = 0.12`
- 訓練ループ内で `model(..., return_spikes=True)` から各 LIF レイヤーの spike を取得。
  - `LargeScaleSNNLM.forward()` に `return_spikes=False` オプションを追加し、True の場合は `(logits, spikes_list)` を返す。
- `HomeostaticRegularizer` を呼び出し、CE loss に加算。
- `validation` 時も spike を取得して発火率のログを出力（損失には加算しない）。

### 1.3 `snnai/modules/language/large_scale_lm.py`（変更）

- `LargeScaleSNNLM.forward()` のシグネチャを `forward(x, return_spikes=False)` に拡張。
- 各 LIF レイヤーの spike をリスト `all_spikes` に蓄積。
- `return_spikes=False` の場合は従来通り `logits` を返す。
- `return_spikes=True` の場合は `(logits, all_spikes)` を返す。
- 既存の `output_mode`（`mem_mean` / `mem_last` / `spike_sum`）の動作は維持。

### 1.4 `snnai/benchmarks/generation_metrics.py`（変更）

- `evaluate_generation()` に `repetition_penalty: float = 1.0`（1.0 = 無効）を追加。
- 生成ループ内で、直近 `penalty_window`（例: 16）トークンの出現回数をカウント。
- 各ステップの logits に対し、出現済みトークンの logit から `log(count) * log(repetition_penalty)` を減算。
- `temperature`、`top_k`、`do_sample` との整合性を保つ（温度スケーリングの直前にペナルティを適用）。

### 1.5 `snnai/benchmarks/text_generation_release.py`（変更）

- `generate_text()` および `SNNTextGenerator.generate()` に `repetition_penalty` 引数を追加。
- 内部で `evaluate_generation()` または同等のロジックを呼び出す場合、引数を透過する。

### 1.6 `environment/kaggle_large_scale/notebook.ipynb`（変更）

- 既存の bash セル（`!wget ...` / `!unzip ...`）を削除または無効化。
- Python セルを追加:
  ```python
  import requests, zipfile, io, os
  url = "https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-v1.zip"
  r = requests.get(url, timeout=60)
  r.raise_for_status()
  z = zipfile.ZipFile(io.BytesIO(r.content))
  z.extractall("/kaggle/working/wikitext-2")
  # 成功時は wikitext-2/wikitext-2/train.txt 等を読み込み
  ```
- ダウンロード失敗時は Tiny Shakespeare にフォールバックする try/except ブロックを設ける。

### 1.7 `snnai/utils/download_corpus.py`（新規）

- `download_wikitext2(dest_dir="./data/wikitext-2", timeout=60) -> Path` を実装。
- `requests` + `zipfile` を使用し、Windows / Linux / Kaggle のいずれでも動作する。
- 戻り値として解凍先ディレクトリの `Path` を返す。
- 失敗時は例外を発生させず `None` を返し、呼び出し側で fallback を選択できる。

### 1.8 `VERSION` / `README.md` / `kernel-metadata.json` / `docs/snnai_test_results.md`（変更）

- `VERSION` を `v6.3.0-dev` に更新。
- `README.md` のトップバージョン表記を `v6.3.0-dev` に更新し、v6.3.0 の改善点セクションを追加。
- `kernel-metadata.json` の `title` を `SNNAI v6.3.0-dev Scale Training` に更新。
- notebook 内の `git clone -b ...` タグを `v6.3.0-dev`（または作業中は `main`）に更新。
- `docs/snnai_test_results.md` の「v6.3.0-dev 開発中」エントリを追加。

### 1.9 `tests/test_v63_homeostasis.py`（新規）

- `HomeostaticRegularizer` のテスト:
  - 発火率が目標値に近いテンソルでは損失が小さい。
  - 発火率が目標値から離れているテンソルでは損失が大きい。
  - 全レイヤー合計のスケーリングが正しい。
- `LargeScaleSNNLM` の `return_spikes=True` テスト:
  - 返却される `spikes_list` の長さが LIF レイヤー数と一致。
  - 各 spike テンソルに `0` または `1` の値のみが含まれる。

### 1.10 `tests/test_v63_repetition_penalty.py`（新規）

- `evaluate_generation()` において `repetition_penalty > 1.0` の場合、直前トークンが連続生成されにくいことを確認。
- `repetition_penalty = 1.0` の場合は通常の貪欲生成と一致することを確認。

### 1.11 `tests/test_v63_download_corpus.py`（新規）

- `download_wikitext2()` の成功時に `train.txt` / `valid.txt` / `test.txt` が存在することを検証（実際のネットワークアクセスは `pytest` 実行時に optional とし、失敗時は skip）。
- フォールバック動作をモックで確認。

### 1.12 `tests/test_v63_version_sync.py`（新規）

- `VERSION` ファイルの内容が `v6.3.0-dev` であることを確認。
- `kernel-metadata.json` の `title` に `v6.3.0-dev` が含まれることを確認。

---

## ステップ 2: 実装順序

1. **恒常性ロス**: `homeostatic_loss.py` → `large_scale_lm.py`（`return_spikes`） → `large_corpus_trainer.py`（統合）
2. **反復ペナルティ**: `generation_metrics.py` → `text_generation_release.py`
3. **コーパスダウンロード**: `download_corpus.py` → `notebook.ipynb`
4. **バージョン同期**: `VERSION` → `kernel-metadata.json` → `README.md` → `docs/snnai_test_results.md` → notebook clone タグ
5. **テスト追加**: `test_v63_*.py` 群
6. **ローカルテスト実行と Kaggle notebook push**

---

## ステップ 3: 成功基準

- ローカル pytest: 既存 105 tests + 新規 tests が全て passed
- `LargeCorpusTrainer` が `homeostatic_weight > 0` で学習可能
- `evaluate_generation()` が `repetition_penalty > 1.0` で改行連続を抑制
- `download_wikitext2()` が Kaggle 環境で `!unzip` なしに動作
- Kaggle notebook version N が T4 GPU で `COMPLETE`
- 生成テキストに改行・スペース以外の文字が増加（CER 改善を目指す）

---

## 参考: v6.2.0 からの教訓

- モデル容量をコーパス規模に合わせることで過学習は抑制できる。
- 時系列 split で val_loss が 0 に陥るのを防げる。
- SNN の「無活動」は改行・スペーストークンに学習が偏る要因の一つ。
- bash コマンドは Kaggle 環境で不安定である。
