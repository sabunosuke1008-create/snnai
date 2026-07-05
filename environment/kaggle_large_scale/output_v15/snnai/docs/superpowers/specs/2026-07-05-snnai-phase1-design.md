# SNNAI Phase1 設計書

## 概要

SNNAI ロードマップの **Phase1：SNNの基礎実装** を実装する。本フェーズでは、snnTorch / Brian2 を使って SNN の基本要素（LIF, STDP, Izhikevich, Liquid State Machine）を `snnai/core/` モジュールとして整備し、以降の生物モジュール実装の基盤とする。

## 背景

- Phase0.5 で LSM / LNN / Bee-Navigation の再現は完了した。
- ただし、それらのノートブックは独立した再現であり、プロジェクト共通の `snnai.core` API には統合されていない。
- Phase1 では、これらの基本コンポーネントを再利用可能な Python モジュールとして実装する。

## 目標

Phase1 終了時点で以下を達成する：

1. `snnai/core/neurons/` に LIF, Izhikevich ニューロンを実装する。
2. `snnai/core/synapses/` に STDP 学習則を実装する。
3. `snnai/core/reservoir/` に Liquid State Machine を実装する。
4. 各コンポーネントに対する pytest テストを追加する。
5. Kaggle 上で `snnai` パッケージを import して動作確認する。

## 採用アプローチ

**snnTorch ベース**: Phase0.5 と同様に、メイン学習基盤は snnTorch。Brian2 は生物学的検証用に並行して小規模テストを作成する。

## ファイル構造

```
snnai/
├── core/
│   ├── __init__.py
│   ├── neurons/
│   │   ├── __init__.py
│   │   ├── lif.py              # 拡張版 LIF
│   │   └── izhikevich.py       # Izhikevich モデル
│   ├── synapses/
│   │   ├── __init__.py
│   │   └── stdp.py             # STDP 学習則
│   └── reservoir/
│       ├── __init__.py
│       └── liquid_state_machine.py
└── tests/
    ├── test_lif.py
    ├── test_izhikevich.py
    ├── test_stdp.py
    └── test_lsm.py
```

## 各コンポーネント設計

### LIF ニューロン

- snnTorch の `snn.Leaky` をラップしたファクトリ関数を提供。
- 追加: 複数ニューロン同時実行、時間方向のシミュレーション関数。

### Izhikevich ニューロン

- snnTorch 上で Izhikevich モデルを PyTorch モジュールとして実装。
- パラメータプリセット: Regular Spiking (RS), Intrinsically Bursting (IB), Chattering (CH), Fast Spiking (FS).
- 時間発展を `torch.Tensor` でベクトル化。

### STDP

- ポストシナプス発火とプレシナプス発火のタイミング差から重み更新。
- snnTorch のスパイクテンソル `(time, batch, neuron)` を入力とする。
- 対称窓 or 非対称窓を選択可能。

### LSM

- Phase0.5 のノートブックをリファクタリングし、クラス化。
- `LSM(input_size, reservoir_size, output_size)` として提供。
- リザバー重みは固定、読み出し層のみ学習可能。

## Kaggle 連携

- `environment/kaggle_setup.ipynb` に `import snnai` して動作確認するセルを追加。
- Phase1 完了時に `environment/phase1_demo.ipynb` を作成し、Kaggle 上で実行確認する。

## 次フェーズ条件

- `pytest tests/` が全てパスすること。
- `snnai` パッケージが Kaggle 上で import 可能であること。
- 単一 LIF / Izhikevich / STDP の動作が理論値と整合すること。

## 制約

- 削除を伴う変更を行う場合は、影響範囲を考慮する。
- Phase0 で作成した最小 LIF 実装を拡張する形で進める。
