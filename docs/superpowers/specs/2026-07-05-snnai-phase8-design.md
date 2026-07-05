# SNNAI Phase8 設計書：実データベンチマーク（MNIST / Fashion-MNIST）

## 概要

SNNAI v2 ロードマップの **Phase8: 実データベンチマーク基盤** を実装する。

## 目的

- MNIST / Fashion-MNIST を用いた SNN 分類器を実装する。
- snnTorch の surrogate gradient 学習を活用する。
- MLP ベースラインと同規模で性能比較できる基盤を整える。

## 構成

- `snnai/benchmarks/data_loader.py`: MNIST / Fashion-MNIST のダウンロード・前処理
- `snnai/benchmarks/mnist_snn.py`: SNN 分類モデル・学習・評価
- `tests/test_phase8.py`: 学習が実行され、精度が 0 より大きいことを確認

## モデル

- 入力画像を時系列スパイク列に変換（ポアソン符号化）
- 畳み込み + LIF ニューロン
- 全結合読み出し層
- Surrogate gradient で学習

## 評価指標

- テスト精度
- 学習時間
- エポックあたりの損失
