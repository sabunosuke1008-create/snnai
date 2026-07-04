# SNNAI

Spiking Neural Network based AI — 線虫・ミツバチ・カラス・タコなど複数の生物の脳構造を参考にした、省電力・モジュラー型 SNN の研究プロジェクト。

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
