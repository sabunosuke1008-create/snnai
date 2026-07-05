# Phase0.5 再現結果

このドキュメントは、SNNAI Phase0.5 で実施した既存研究の再現結果をまとめたものです。

## LSM (Liquid State Machine)

| Metric | Reference | Ours | Status |
|---|---|---|---|
| Test accuracy | > 90% (typical LSM on simple temporal tasks) | **1.0000** | ✅ |

### 備考
- snnTorch の LIF ニューロン 100 個からなるリザバーを使用。
- 低周波（1.0）と高周波（3.0）の正弦波 2 クラス分類タスク。
- リザバー出力（最後 50 ステップの平均発火率）に対し RidgeClassifier を学習。
- PCA 結果：第 1 主成分で 61.6% の分散を説明、2 クラスの重心距離 = 1.28。

**Kaggle Notebook**: https://www.kaggle.com/gihuhi/snnai-phase0-5-lsm-reproduction

## LNN (Liquid Neural Network / Liquid Time-constant Network)

| Metric | Reference | Ours | Status |
|---|---|---|---|
| Sine-wave prediction MSE | ~0.01 | **0.0124** | ✅ |
| MAE | - | **0.0890** | - |

### 備考
- カスタム LTC（Liquid Time-constant）セルを snnTorch 上に実装。
- ノイズ付き正弦波の次時刻予測タスク。
- 200 epoch で MSE 0.0124、MAE 0.0890 を達成。

**Kaggle Notebook**: https://www.kaggle.com/gihuhi/snnai-phase0-5-lnn-reproduction

## Bee Navigation（ミツバチ経路積分の簡易モデル）

| Metric | Reference | Ours | Status |
|---|---|---|---|
| Endpoint error / trajectory length | < 0.5 | **2.27** | ⚠️ |

### 備考
- 速度信号を正/負のスパイク集団に符号付きエンコード。
- 2 つの LIF 積分器ニューロンで x, y 位置を推定。
- 推定位置は真の位置の傾向を追うが、重みの較正が不十分なため相対誤差 2.27 となった。
- Phase3 ミツバチモジュールの基盤として、重み較正や STDP 学習を追加する予定。

**Kaggle Notebook**: https://www.kaggle.com/gihuhi/snnai-phase0-5-bee-navigation-reproduction

## 総合評価

- LSM と LNN はロードマップで想定された性能基準を満たし、Phase2〜4 のモジュール基盤として利用可能。
- Bee Navigation は概念実証として動作するが、位置推定精度の改善が今後の課題。

## 次フェーズ

- Phase1: SNN 基礎（LIF, STDP, Izhikevich）の実装
- Phase2: 線虫モジュール（反射・最低限制御）
- Phase3: ミツバチモジュール（空間認識・強化学習）
- Phase4: カラスモジュール（推論・ワーキングメモリ）
- Phase5: タコモジュール（分散処理・イベント駆動）
