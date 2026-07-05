# SNNAI Phase4 設計書：カラスモジュール

## 概要

SNNAI ロードマップの **Phase4：カラスモジュール（作業記憶・道具使用・因果推論）** を実装する。本モジュールは **Working Memory / Tool Use / Causal Reasoning** を目的とする。

## 背景

- カラスは類人猿に匹敵する認知能力を持つ。特に道具使用と因果推論が知られる。
- ニューロン規模を 1000〜5000 に拡大し、持続的活動（persistent activity）による作業記憶を実装する。

## 目標

1. `snnai/modules/crow/` に作業記憶モジュールを実装する。
2. Delayed Match-to-Sample（DMS）タスクを解く。
3. キュー提示後の遅延期間中に情報を保持し、プローブ提示時に正しく一致判定できること。

## 設計

### モジュール構造

```
snnai/modules/
└── crow/
    ├── __init__.py
    ├── working_memory.py
    └── tests/test_delayed_match.py
```

### 作業記憶モジュール

- 入力層：刺激 A / B をスパイク列で表現
- 記憶層：Izhikevich ニューロンによる持続的活動
- 出力層：一致 / 不一致判定

### Delayed Match-to-Sample タスク

1. キュー提示（A または B）
2. 遅延期間（何も提示しない）
3. プローブ提示（A または B）
4. ネットワークは「一致」なら left motor、「不一致」なら right motor を発火

## 評価指標

- 一致/不一致判定の正解率 > 80%

## 実装方針

- snnTorch の Izhikevich ニューロン（或いは独自 Izhikevich）を使用。
- キュー期間に対応する入力を与え、遅延期間は入力をゼロにする。
- 記憶ニューロンが自発発火を維持するよう、再帰結合を持たせる。

## 次フェーズ条件

- DMS タスクで一定以上の正解率が出ること。
- GitHub タグ `v0.4` を打つこと。
