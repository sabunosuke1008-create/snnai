# SNNAI Phase0 設計書

## 概要

SNNAI（Spiking Neural Network based AI）プロジェクトの**Phase0：開発環境構築とライブラリ選定**を実装する。
ロードマップ `SNNAI_roadmap.md` の方針に従い、GitHub リポジトリを作成し、Kaggle Notebook 用のセットアップ環境を構築する。

## 背景

- 批判的評価ドキュメントによれば、個別の生物模倣 SNN は既存研究と重なるが、**複数生物モジュールのハイブリッド統合**には新規性の余地がある。
- ロードマップは Phase0 から Phase7 まで定義されており、本設計書ではその入り口となる Phase0 に焦点を絞る。

## 目標

Phase0 終了時点で以下を達成する：

1. GitHub リポジトリ `snnai` を作成し、初期ファイルを push する。
2. 必要なライブラリ（snnTorch, Brian2, Norse, BindsNET, torch 等）を `requirements.txt` に定義する。
3. Kaggle Notebook 上でライブラリを導入し、GPU を認識できる `environment/kaggle_setup.ipynb` を作成する。
4. ライブラリ import と GPU 認識を自動検証するスクリプトを作成する。
5. ロードマップに沿ったディレクトリ構造の雛形を整える。

## 採用アプローチ

**アプローチ B：検証付きセットアップ**

- アプローチ A（最小構成）では動作確認が手動となるため、自動検証を含める。
- アプローチ C（Phase0＋0.5 同時）では作業量が大きいため、本設計では Phase0 のみをスコープとする。

## 初期リポジトリ構造

```
snnai/
├── README.md                     # プロジェクト概要、ロードマップへのリンク
├── VERSION                       # v0.1.0-dev
├── requirements.txt              # 依存ライブラリとバージョン固定
├── .gitignore
├── docs/
│   ├── roadmap.md                # ロードマップのコピー
│   └── superpowers/specs/2026-07-05-snnai-phase0-design.md
├── environment/
│   ├── kaggle_setup.ipynb        # Kaggle Notebook 用セットアップ
│   └── verify_imports.py         # ローカル/Kaggle 両方で使える検証スクリプト
├── snnai/
│   ├── __init__.py
│   └── core/
│       ├── __init__.py
│       └── neurons/
│           ├── __init__.py
│           └── lif.py            # 単一 LIF ニューロンの最小実装
└── tests/
    └── test_environment.py       # pytest による環境検証テスト
```

## 採用ライブラリ

| ライブラリ | 用途 |
|---|---|
| snnTorch | メイン学習基盤（代理勾配法） |
| Brian2 | 生物学的検証、線虫モジュール |
| Norse | snnTorch との比較実験 |
| BindsNET | 強化学習連携の参考実装 |
| torch | 基盤フレームワーク |
| numpy | 数値計算 |
| pytest | テスト |

## 検証項目

1. `import snntorch`, `import brian2`, `import norse`, `import bindsnet` が成功する。
2. `torch.cuda.is_available()` の結果を表示する。
3. 単一 LIF ニューロンで、一定電流入力に対してスパイクが発生することを確認する。

## Kaggle 連携

- `environment/kaggle_setup.ipynb` は `.ipynb` 形式で Kaggle Notebook としてそのままアップロード可能とする。
- Kaggle CLI を使用して push する際のコマンド例を README に記載する。
- 必要に応じて `kaggle datasets init` 用のメタデータも同梱する。

## GitHub 運用

- リポジトリ名: `snnai`
- 初期ブランチ: `main`
- タグ: Phase0 完了時に `v0.1.0-dev` を打つ（Phase0.5 完了時に `v0.1` とする）。

## 次フェーズ条件

- `pytest tests/test_environment.py` が全てパスすること。
- `environment/kaggle_setup.ipynb` が Kaggle 上でエラーなく実行できること。
- snnTorch と Brian2 それぞれで単一 LIF ニューロンのスパイクを再現できること。

## 制約

- 削除を伴う変更を行う場合は、影響範囲を考慮してから実行する。
- ユーザーが行き詰まりの際に質問を求めているため、不明確な点は自己判断で決定し、重要な分岐点のみ確認する。
