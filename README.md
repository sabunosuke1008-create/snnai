# SNNAI

Spiking Neural Network based AI — 線虫・ミツバチ・カラス・タコなど複数の生物の脳構造を参考にした、省電力・モジュラー型 SNN の研究プロジェクト。

## バージョン v1.0（Phase7 完了）

SNNAI v1.0 では以下のモジュールとベンチマーク基盤を統合しました。

### 含まれるモジュール

- **Phase1**: LIF / Izhikevich / STDP / LSM コア
- **Phase2**: 線虫モジュール（反射）
- **Phase3**: ミツバチモジュール（空間 RL + R-STDP）
- **Phase4**: カラスモジュール（作業記憶）
- **Phase5**: タコモジュール（分散並列処理）
- **Phase X**: Bio-NAS（生物制約付きアーキテクチャ探索）
- **Phase6**: 異種ニューラルネットワーク統合（Hub + Encoding Bridge + Pipeline）
- **Phase7**: ベンチマークとベースライン比較

### GitHub タグ

- `v0.1` Phase0〜Phase0.5
- `v0.2` Phase2 線虫
- `v0.3` Phase3 ミツバチ
- `v0.4` Phase4 カラス
- `v0.5` Phase5 タコ
- `v0.6` Phase X Bio-NAS
- `v0.7` Phase6 統合
- `v1.0` Phase7 完成版

### ベンチマーク

- `snnai/benchmarks/synthetic_benchmark.py`: 合成 4 クラス分類タスク
- `snnai/benchmarks/baseline_mlp.py`: 同規模 MLP ベースライン
- `snnai/benchmarks/energy_estimation.py`: スパイク数ベース簡易エネルギー推定

## ロードマップ

詳細は [docs/roadmap.md](docs/roadmap.md) を参照。

## クイックスタート

### ローカル

```bash
pip install -r requirements.txt
pytest tests/
```

### Kaggle

`environment/kaggle_setup.ipynb` を Kaggle Notebook としてインポートして実行してください。

```bash
# Kaggle CLI で push する例
python -m kaggle kernels push -p environment
```

## ライセンス

MIT
