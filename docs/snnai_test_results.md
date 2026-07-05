# SNNAI テスト実行結果レポート

**Version**: `v6.0.5`  
**Test date**: 2026-07-05  
**Repository**: https://github.com/sabunosuke1008-create/snnai

---

## 1. 実行環境

- **OS**: Windows 11
- **Python**: 3.13.14
- **pytest**: 8.2.0
- **PyTorch**: 2.x（ローカル環境）
- **snnTorch**: 最新版

## 2. 実行コマンド

```bash
pytest tests/ --ignore=tests/test_environment.py -q
```

`tests/test_environment.py` は `brian2` が NumPy の新しい API（`np.ndarray.ptp`）と互換性がないため、import 段階でエラーとなります。これは v2.1〜v6.0 のロードマップ実装とは無関係なため、除外して実行しました。

## 3. 総合結果

| 項目 | 値 |
|---|---|
| Passed | 98 |
| Failed | 1 |
| Warnings | 5 |
| 合計テスト数 | 99 |

```text
98 passed, 1 failed, 5 warnings in 185.45s (0:03:05)
```

## 4. ステージ別テスト結果

### Stage 1：省電力エッジ AI（v2.1〜v2.5）

| テストファイル | 結果 | 検証内容 |
|---|---|---|
| `tests/test_v21_sensor_filter.py` | PASS | スパイクフィルタ出力の形状と値範囲 |
| `tests/test_v22_robot_controller.py` | PASS | ロボット制御コマンドの形状と範囲 |
| `tests/test_v23_iot_event_detector.py` | PASS | IoT イベント検出のアラーム発生 |
| `tests/test_v24_edge_pipeline.py` | PASS | エッジパイプライン統合の出力形状 |
| `tests/test_v25_auto_driving.py` | PASS | 自動運転デモの車両位置更新 |

**通過基準**: 各モジュールが期待されるテンソル形状を返し、異常時に適切にアラームを出すこと。

### Stage 2：小規模モジュール型言語パイプライン（v2.6〜v3.5）

| テストファイル | 結果 | 検証内容 |
|---|---|---|
| `tests/test_v26_tokenizer.py` | PASS | 文字トークナイザの encode/decode |
| `tests/test_v27_next_token.py` | PASS | 次文字予測の出力形状 |
| `tests/test_v28_preprocess_pipeline.py` | PASS | 注意マスクの形状と値範囲 |
| `tests/test_v29_context_retention.py` | PASS | 作業記憶の文脈保持 |
| `tests/test_v30_reasoning.py` | PASS | 推論モジュールの出力形状 |
| `tests/test_v31_small_lm.py` | PASS | 小規模 SNN LM の学習・生成 |
| `tests/test_v32_summarization.py` | PASS | 要約ベンチマークの実行 |
| `tests/test_v33_qa.py` | PASS | QA ベンチマークの実行 |
| `tests/test_v34_translation.py` | PASS | 翻訳ベンチマークの実行 |
| `tests/test_v35_code_completion.py` | PASS | コード補完ベンチマークの実行 |

**通過基準**: トークナイザが正しく文字↔ID を変換し、各言語モデル/ベンチマークがエラーなく実行できること。

### Stage 3：テキスト生成（v3.6〜v4.5）

| テストファイル | 結果 | 検証内容 |
|---|---|---|
| `tests/test_v36_char_lm.py` | PASS | 文字レベル LM トレーナー |
| `tests/test_v37_word_lm.py` | PASS | 単語レベル LM |
| `tests/test_v38_bpe.py` | PASS | BPE トークナイザ |
| `tests/test_v39_long_context.py` | PASS | 海馬を使った長期文脈 |
| `tests/test_v40_large_lm.py` | PASS | 大規模 SNN LM（パラメータ数 >80k） |
| `tests/test_v41_transformer_baseline.py` | PASS | Transformer ベースライン |
| `tests/test_v42_energy_benchmark.py` | PASS | エネルギー効率測定 |
| `tests/test_v43_finetune.py` | PASS | ファインチューニング基盤 |
| `tests/test_v44_dialogue.py` | PASS | 対話応答 |
| `tests/test_v45_text_gen.py` | PASS | テキスト生成リリース API |

**通過基準**: モデルが期待される形状の出力を返し、生成テキストの長さが `len(prompt) + max_chars` となること。大規模 LM はローカルで最小 80k パラメータを満たすこと。

### Stage 4：LLM 支援・常時稼働 AI（v4.6〜v5.5）

| テストファイル | 結果 | 検証内容 |
|---|---|---|
| `tests/test_v46_llm_preprocess.py` | PASS | LLM 入力前処理 |
| `tests/test_v47_llm_postprocess.py` | PASS | LLM 出力後処理 |
| `tests/test_v48_memory_bridge.py` | PASS | 長期記憶の保存・検索 |
| `tests/test_v49_always_on.py` | PASS | 常時監視デモ |
| `tests/test_v50_llm_pipeline.py` | PASS | LLM 連携パイプライン |
| `tests/test_v51_energy_quant.py` | PASS | 省電力推論の定量化 |
| `tests/test_v52_multimodal.py` | PASS | マルチモーダル入力 |
| `tests/test_v53_personalization.py` | PASS | パーソナライゼーション |
| `tests/test_v54_edge_deployment.py` | PASS | エッジ展開プロファイリング・TorchScript エクスポート |
| `tests/test_v55_always_on_ai.py` | PASS | 常時稼働 AI v1.0 |

**通過基準**: 各連携モジュールが文字列やテンソルを正しく入出力し、パイプラインがエラーなく動作すること。

### Stage 5：Transformer 並み性能への挑戦（v5.6〜v6.0）

| テストファイル | 結果 | 検証内容 |
|---|---|---|
| `tests/test_v56_large_scale.py` | PASS | 大規模 SNN LM アーキテクチャとパラメータ数 |
| `tests/test_v57_large_corpus.py` | PASS | 大規模コーパストレーナー |
| `tests/test_v58_transformer_comparison.py` | PASS | Transformer 比較スイート |
| `tests/test_v59_optimization.py` | PASS | 量子化・枝刈り |
| `tests/test_v60_transformer_challenge.py` | PASS | Transformer 並み性能チャレンジハーネス |

**通過基準**: 大規模アーキテクチャが小設定で形状を保ち、比較・最適化・チャレンジハーネスがエラーなく実行できること。

## 5. 失敗したテスト

### `tests/test_bio_nas.py::test_evolution_finds_score_at_least_as_good_as_serial`

**結果**: FAILED  
**エラー**: `KeyError: 'm2'`（`snnai/bio_nas/evaluator.py` 62 行目）

```python
combined = sum(cache[p] for p in preds)
```

**考察**:
- このテストは v2.1〜v6.0 のロードマップとは無関係な、既存の `bio_nas` モジュールのバグです。
- `evaluate_architecture` 内でネットワークをトポロジカル順に評価している際、依存先のモジュール ID（例: `m2`）が `cache` に存在しない状態でアクセスされています。
- 原因として考えられるのは、進化探索で生成されたアーキテクチャの依存グラフが正しい評価順序でソートされていないこと、または一部のモジュールが評価リストから漏れていることです。
- 今回のロードマップ実装ではこのモジュールに変更を加えていないため、別途修正が必要です。

## 6. Kaggle 実行結果

- **Kernel**: `gihuhi/snnai-v6-scale-training`
- **URL**: https://www.kaggle.com/code/gihuhi/snnai-v6-scale-training
- **状況**: Kaggle 環境では T4 指定を試みましたが、P100 が割り当てられるケースがありました。PyTorch のプリインストール版が P100（sm_60）をサポートしていなかったため、v6.0.5 以降は compute capability < 7.0 の場合に CPU フォールバックする処理を追加しています。
- 大規模学習の実測結果は Kaggle ノートブックの実行ログで確認してください。

## 7. Warnings

| 内容 | 発生ファイル |
|---|---|
| `torch.jit.trace` は非推奨 | `tests/test_v54_edge_deployment.py` |
| snnTorch Leaky LIF の Python bool 変換警告 | `tests/test_v54_edge_deployment.py` |
| TransformerEncoder `enable_nested_tensor` 警告（nhead=1 のため） | `tests/test_v58_transformer_comparison.py` |

これらは実行を妨げるものではありません。

## 8. まとめ

- v2.1〜v6.0 に対応する 99 テスト中 **98 テストが PASS** しました。
- 唯一の失敗は `bio_nas` の既存バグによるもので、今回のロードマップ範囲外です。
- Kaggle では P100 フォールバック対応により、大規模学習を継続可能としています。
