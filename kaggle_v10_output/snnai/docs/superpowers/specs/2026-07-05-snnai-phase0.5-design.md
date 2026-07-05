# SNNAI Phase0.5 設計書

## 概要

SNNAI ロードマップの **Phase0.5：既存研究の再現** を実装する。本フェーズの目的は、新規実装に進む前に「自分の実装が悪いのか」「アイデアが悪いのか」を切り分けられる基準を作ること。再現結果は以降の全モジュールの検証基準（サニティチェック）として使い続ける。

## 背景

- Phase0 で環境構築と最小 LIF 実装は完了した。
- 批判的評価ドキュメントでは、個別の生物模倣 SNN は既存研究と重なるが、複数生物モジュールのハイブリッド統合に新規性の余地があるとされている。
- ロードマップは Phase0.5 で LSM、LNN、Bee-Navigation の再現を推奨している。

## 目標

Phase0.5 終了時点で以下を達成する：

1. `snnai/reproductions/` に LSM、LNN、Bee-Navigation の再現ノートブックを作成する。
2. 各再現が Kaggle Notebook 上でエラーなく完走する。
3. 元論文/公式実装の性能指標に対して許容誤差内（目安 ±10%）で一致すること。
4. 再現結果を `reproduction_results.md` にまとめ、以降のモジュール実装の基準とする。

## 採用アプローチ

**再現研究駆動型**: Phase0.5 の再現実装をそのまま後の生物モジュールの基盤に流用する。例えば LNN の連続時間動力学は線虫モジュールの基盤、ミツバチ経路積分は Phase3 の出発点となる。

## スコープ

| 再現対象 | 優先度 | 理由 | 目標ファイル |
|---|---|---|---|
| Liquid State Machine (LSM) | 高 | Phase2 線虫モジュール（少数ニューロンで複雑な振る舞い）の直接的基盤 | `lsm_reproduction.ipynb` |
| Liquid Neural Network (LNN) | 高 | 少パラメータで高性能という本プロジェクトの目標と直結 | `lnn_reproduction.ipynb` |
| Bee Navigation Model | 中 | Phase3 ミツバチモジュールの妥当性検証基準 | `bee_navigation_reproduction.ipynb` |
| HTM / Fly Connectome | 今回は対象外 | 拡張候補として将来検討 | - |

## ファイル構造

```
snnai/
├── reproductions/
│   ├── __init__.py
│   ├── lsm_reproduction.ipynb
│   ├── lnn_reproduction.ipynb
│   ├── bee_navigation_reproduction.ipynb
│   └── docs/
│       └── reproduction_results.md
├── docs/
│   └── papers.md          # 参考論文・公式実装リンク集
└── environment/
    └── kaggle_setup.ipynb # Phase0 で作成済み
```

## 各再現の詳細設計

### LSM (Liquid State Machine)

- **ライブラリ**: snnTorch
- **タスク**: 時系列パターン分類
- **構成**:
  - 入力層: スパイク列を入力
  - リザバー層: ランダム結合 LIF ニューロン（固定重み）
  - 読み出し層: 線形分類器（Logistic Regression など）
- **データセット**: 自前で生成した簡易時系列パターン（例：正弦波の周波数識別）または N-MNIST（時間があれば）
- **評価**: 分類精度。リザバー状態の可分離性も可視化。

### LNN (Liquid Neural Network / Liquid Time-constant Network)

- **ライブラリ**: snnTorch + カスタム LTC ニューロン
- **参考**: Hasani et al. (2020) "Liquid Time-constant Networks" 公式実装
- **タスク**: 時系列予測（例： sine wave prediction, MNIST 一筆書き軌道予測）
- **評価**: 元論文/公式実装の予測誤差と比較

### Bee Navigation（ミツバチ経路積分）

- **ライブラリ**: Brian2 or snntorch
- **参考**: Baddeley et al. 等のミツバチ経路積分モデル
- **タスク**: 自己運動情報（速度・方向）から位置を推定
- **評価**: 目標位置への到達誤差

## 実装・評価ルール

1. 各ノートブックは Kaggle 上で 1 から実行可能にする。
2. セルごとにコメントを入れ、何を計算しているかを明示する。
3. 再現結果は `reproduction_results.md` に以下の形式で記録する:
   - 元論文/公式実装の値
   - 自実装の値
   - 許容誤差と判定
4. 実行時間が長い場合は Kaggle 上で GPU を使用する。

## Kaggle 連携

- 各 `.ipynb` を `kaggle_cli_mcp_kernels_push` で個別に push する。
- タイトル例: `SNNAI Phase0.5 - LSM Reproduction`
- GPU: Kaggle 既定の torch/cuda を利用するため、torch/numpy は再インストールしない。

## GitHub 運用

- Phase0.5 完了時に GitHub タグ `v0.1` を打つ。
- コミットメッセージ: `feat: add Phase0.5 reproductions (LSM, LNN, Bee-Nav)`

## 次フェーズ条件

- 少なくとも LSM と LNN の再現が完了していること。
- `reproduction_results.md` に結果が記録されていること。
- 各再現ノートブックが Kaggle 上で完走していること。

## 制約

- 削除を伴う変更を行う場合は、影響範囲を考慮する。
- ユーザーが行き詰まりの際に質問を求めているため、重要な分岐点のみ確認する。
