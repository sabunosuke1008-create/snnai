# SNNAI 実装ロードマップ（Kaggle Notebook専用・生物模倣型SNN）

**役割宣言**: 本ロードマップは、AI研究者・ニューロモルフィックコンピューティング研究者・SNN研究者・強化学習研究者・神経科学者・システムアーキテクトの視点を統合して作成した「実装ロードマップ」です。論文執筆ではなく、GitHubリポジトリとして動くコードを段階的に積み上げることを目的とします。

---

## 0. 全体方針

### 0.1 開発思想の評価と補強
「複数の生物の脳の長所を組み合わせる」という発想自体は、ニューロモルフィック分野では **heterogeneous / modular SNN** と呼ばれる正当なアプローチです。ただし、線虫→昆虫→鳥類→哺乳類という一直線の階層は、生物学的にも工学的にも「積み上げ型パイプライン」というより **並列モジュール+仲介ハブ（ブローカー）構造** の方が実装・デバッグが容易です。

**提案する改良アーキテクチャ**

```
                 ┌─────────────┐
   感覚入力 ───▶ │ 線虫モジュール │──反射出力(即時)
                 │ (数十ニューロン)│
                 └──────┬──────┘
                        │ スパイクイベント
                 ┌──────▼──────┐
   状態入力 ───▶ │ 昆虫モジュール  │──特徴表現
                 │(ミツバチ型・数百)│
                 └──────┬──────┘
                        │
                 ┌──────▼──────┐      ┌──────────────┐
                 │ ハブ/ルーター  │◀───▶│ タコ型分散モジュール│
                 │(イベントバス)  │      │(並列・独立サブネット)│
                 └──────┬──────┘      └──────────────┘
                        │
                 ┌──────▼──────┐
                 │ カラス型モジュール│──推論・計画・WM
                 │(再帰SNN+記憶) │
                 └──────┬──────┘
                        ▼
                     最終出力
```

理由：
- 線虫モジュールは**割り込み型の反射弓**として独立に常時稼働させ、他モジュールをバイパスできるようにする（生物でも反射は大脳を経由しない）。
- タコ型モジュールは「階層の1段」ではなく「並列に動く独立ワーカー群」として設計する方が、分散処理という本来の特性を活かせる。
- ハブはイベント駆動のPub/Sub的な設計にすると、Phase6の統合が大幅に楽になる。

この構造は例であり、Phase6で実測しながら調整することを前提とします。

### 0.2 Kaggle環境の制約整理
| リソース | スペック | 用途の目安 |
|---|---|---|
| CPU | 4コア / 約30GB RAM | Brian2でのニューロン科学的シミュレーション、小規模SNN |
| T4×2 | 16GB×2（計32GB） | snnTorch/Norseでの学習、中規模モジュール訓練 |
| P100 | 16GB | 単体学習、CUDA互換性確認 |
| TPU v3-8 | 128GB HBM | JAX系（Spyx）での大規模並列実験。ただしSNN対応ライブラリはPyTorch系ほど成熟していない |

**注意点（現実的制約）**:
- Kaggle Notebookはセッション時間制限（GPU: 週30時間、連続実行9時間程度）があるため、**チェックポイント保存を必須設計**にする。
- Lava（Intel Loihi向け）は専用ニューロモルフィックチップ前提のため、Kaggle上では**シミュレーションモードのみ**使用可能。実チップ性能は再現できない。
- TPUでのSNN実装はJAX/Spyxに限られ、情報・サンプルコードが少ない＝Phase0-1では避け、Phase6以降の高速化実験用に温存するのが現実的。

### 0.3 バージョン戦略（成果を残す設計）
「勉強フェーズ」の裏に「リリース単位」を必ず対応させ、途中で止まっても成果物が残る設計にします。

| リリース | 対応Phase | 成果物として示せる内容 |
|---|---|---|
| v0.1 | Phase0〜Phase0.5 | 環境構築＋既存研究（LSM/LNN等）の再現結果 |
| v0.2 | Phase2 | 線虫AI（反射モジュール単体） |
| v0.3 | Phase3 | 昆虫AI（線虫＋ミツバチ、2モジュール連携）＝MVP |
| v0.4 | Phase4 | 鳥AI（＋カラスモジュール、ワーキングメモリ追加） |
| v0.5 | Phase5 | タコAI（＋分散並列モジュール） |
| v0.6 | Phase X (Bio-NAS) | 最適モジュール配置の探索結果レポート |
| v1.0 | Phase6〜7 | Hybrid SNNAI（全モジュール統合＋ベンチマーク結果） |

各リリースにはGitHubのタグを打ち、READMEに「今のバージョンで何ができるか」のデモ動画/GIFかログを添付することを推奨します。

### 0.4 モジュール共通設計方針（目的の明確化）
各生物モジュールは「後から追加・差し替え可能な部品」として扱うため、実装前に**目的（Purpose）を3つのタグで固定**します。新しい生物を追加する際も、このタグ付けルールに従えばアーキテクチャを変えずに拡張できます。

| モジュール | 目的タグ |
|---|---|
| 線虫 | Reflex / Event Detection / Filtering |
| ミツバチ | Feature Learning / Sparse Memory / Reinforcement |
| カラス | Working Memory / Planning / Rule Learning |
| タコ | Parallel Processing / Independent Expert / Dynamic Routing |
| （拡張候補）ショウジョウバエ | Fast Sensory Filtering / Olfactory Coding |
| （拡張候補）海馬 | Episodic Memory / Spatial Map (Place Cells) |
| （拡張候補）アリ | Swarm Coordination / Pheromone-like Signaling |

各モジュールの実装ファイルには、コード先頭に目的タグをコメントとして明記するルールにすると、後から見返す際にも設計意図が失われません。

### 0.5 最終ベンチマークロードマップ（客観的な進捗判定）
「研究が進んでいるか」を主観ではなく指標で判定するため、以下の順でベンチマークをクリアしていく計画にします。難易度は易→難の順です。

| 段階 | ベンチマーク | 主な評価対象 |
|---|---|---|
| 1 | MNIST | 基礎的な分類能力（Phase1後） |
| 2 | Fashion-MNIST | 汎化性能の確認 |
| 3 | N-MNIST（イベントカメラ版MNIST） | SNN本来の時間符号化性能 |
| 4 | DVS Gesture | 時系列イベントデータへの対応（カラス/タコモジュール向き） |
| 5 | MiniGrid | 強化学習・空間認識（ミツバチモジュール向き） |
| 6 | Atari（一部タイトル） | 統合後のSNNAI v1の総合性能評価 |
| 7 | Minecraft（MineRL等） | 長期計画・複合タスク（v2以降の目標） |
| 8 | 実ロボット/シミュレータ（PyBullet等） | 身体性を伴う統合評価（v2の最終目標） |

**運用ルール**: 各Phase・各モジュールは、対応するベンチマーク段階を最低1つクリアしないと次のPhaseに進まない、というゲート条件として使います（例：Phase3終了条件にMiniGridでの有意な学習曲線改善を追加）。

---

## Phase0：開発環境構築とライブラリ選定

### ライブラリ比較
| ライブラリ | 基盤 | 強み | 弱み | Kaggle適性 |
|---|---|---|---|---|
| **Brian2** | 独自(Python) | 生物学的精度、微分方程式ベースのニューロンモデルが書きやすい | GPU学習には不向き、勾配学習不可 | ◎（Phase1-2の基礎学習・線虫モジュールに最適） |
| **snnTorch** | PyTorch | 逆伝播代理勾配法(surrogate gradient)が簡単、PyTorchエコシステムそのまま使える | 生物学的精度はBrian2より低い | ◎（Phase2以降のメイン学習フレームワーク候補） |
| **Norse** | PyTorch | STDP・LIFなど豊富なニューロンモデル、PyTorch Lightning対応 | ドキュメントがやや少ない | ○（snnTorchと併用・比較用） |
| **BindsNET** | PyTorch | 強化学習との統合サンプルが多い | 開発がやや停滞気味 | ○（Phase3強化学習モジュール参考実装用） |
| **Spyx** | JAX | TPUとの親和性、高速 | 情報量少、コミュニティ小 | △（Phase6以降のTPU実験用） |
| **Lava** | 独自 | Loihi向け設計、真のイベント駆動 | Kaggleでは実チップ不可、学習コストが高い | △（概念検証・将来の実チップ移行時に参照） |

### 採用方針
- **メイン学習基盤：snnTorch（PyTorch）** — Kaggle T4/P100との相性、サンプル・チュートリアルの豊富さから採用。
- **生物学的検証・線虫モジュール：Brian2** — LIF/Izhikevichモデルの正確な再現、CPUのみで完結。
- **比較実験：Norse** — snnTorchとの結果比較でバグ検出・妥当性確認に使う。
- Spyx/Lavaは正式採用せず、Phase6以降の性能検証・将来の実機移行検討時にのみ触れる。

### GitHubコミット内容（Phase0終了時）
```
snnai/
├── README.md              # プロジェクト概要、ロードマップへのリンク
├── environment/
│   ├── requirements.txt   # snntorch, norse, brian2, bindsnet, torch
│   └── kaggle_setup.ipynb # Kaggle上でのライブラリ導入手順
├── docs/
│   └── roadmap.md         # 本ロードマップ
└── .gitignore
```
**テスト項目**: 各ライブラリのimportとGPU認識確認（`torch.cuda.is_available()`等）。
**次フェーズ条件**: snnTorch+Brian2でそれぞれ1個のLIFニューロンのスパイクをKaggle上で再現できること。

---

## Phase0.5：既存研究の再現（v0.1の中核）

**目的**: 新規実装に進む前に、「自分の実装が悪いのか」「アイデアが悪いのか」を切り分けられる基準を作る。既存の確立された研究をKaggle上で完全再現し、それを以降の全モジュールの検証基準（サニティチェック）として使い続けます。

### 再現対象と優先順位
| 優先度 | 再現対象 | 再現する理由 |
|---|---|---|
| 高 | **Liquid State Machine (LSM)** | Phase2の線虫モジュール（少数ニューロンで複雑な振る舞い）の直接的基盤 |
| 高 | **Liquid Neural Network (LNN, Hasani et al.)** | 少パラメータで高性能という本プロジェクトの目標と直結 |
| 中 | **Bee Navigation Model（ミツバチの経路積分モデル）** | Phase3ミツバチモジュールの妥当性検証基準 |
| 中 | **Fly (Drosophila) Connectome ベースの簡易回路** | 将来のショウジョウバエモジュール拡張時の布石、かつSTDP検証にも使える |
| 低〜中 | **Numenta HTM (Hierarchical Temporal Memory)** | SNNとは異なる神経科学ベースモデルとの比較対象として。異種統合（Phase6）の設計の参考にもなる |

### 必要な論文・参考実装
- Maass et al. (2002) LSM原論文
- Hasani et al. (2020) "Liquid Time-constant Networks"（LNN、公式実装が公開されている）
- Numenta社のHTM公式ドキュメント・NuPICリポジトリ
- ミツバチナビゲーションの計算モデル（Baddeley et al.等の経路積分モデル）

### 実装評価
- Kaggle実装可否：◎（すべて既存OSSが存在しCPU/T4のみで完結）
- 難易度：中（再現がゴールなので実装難易度自体は低いが、再現性の検証に手間がかかる）
- 想定期間：3〜4週間
- 必要GPU：LNNのみT4推奨、他はCPU可
- 再利用OSS：LSM/LNNは公式GitHub実装あり、HTMはNuPIC/htm.core

### GitHubコミット内容
```
snnai/reproductions/
├── lsm_reproduction.ipynb
├── lnn_reproduction.ipynb
├── bee_navigation_reproduction.ipynb
├── htm_reproduction.ipynb
└── docs/reproduction_results.md   # 元論文の数値と自分の再現結果を並記
```
**テスト項目**: 各再現実装が元論文/公式実装の性能指標に対して許容誤差内（目安±10%）で一致すること。
**次フェーズ条件**: 少なくともLSMとLNNの再現が完了し、Phase1で実装した自作LIFニューロンとの動作比較ログが取れていること。

---

## Phase1：SNNの基礎実装

### 学習順序（推奨）
1. **LIFニューロン**（積分発火モデル）— 最も単純で理論と実装の対応が取りやすい
2. **シナプスモデル**（電流ベース／コンダクタンスベース）
3. **STDP**（スパイクタイミング依存可塑性）— 教師なし学習則の理解
4. **Izhikevichニューロン** — 多様な発火パターン（バースト等）を扱う際に必要
5. **Liquid State Machine / Liquid Neural Network** — リザバーコンピューティングの基礎。線虫モジュール（少数ニューロンで複雑な振る舞い）に直結

### 必要な論文・書籍（代表例）
- Gerstner & Kistler, *Spiking Neuron Models*（教科書、LIF/Izhikevich理論の基盤）
- Izhikevich (2003) "Simple model of spiking neurons"
- Bi & Poo (1998) STDPの実験的基盤論文
- Maass (2002) "Real-time computing without stable states"（Liquid State Machine）
- Neftci et al. (2019) "Surrogate Gradient Learning in Spiking Neural Networks"（snnTorchの理論的背景）

### Kaggle実装可否／難易度／期間
- 実装可否：◎（すべてCPUのみで可能）
- 難易度：中（微分方程式の数値積分の理解が必要）
- 想定期間：2〜3週間
- 必要GPU：不要（CPUのみ）
- サンプルコード：snnTorchの公式チュートリアル、Brian2のexample galleryが充実

### GitHubコミット内容
```
snnai/core/
├── neurons/
│   ├── lif.py
│   └── izhikevich.py
├── synapses/
│   └── stdp.py
├── reservoir/
│   └── liquid_state_machine.py
└── tests/
    ├── test_lif_spike_train.py
    └── test_stdp_weight_update.py
```
**テスト項目**: 既知の入力電流に対する発火頻度が理論値と一致するか（許容誤差5%以内）。
**ベンチマーク**: 単一ニューロンのシミュレーション速度（step/sec）をBrian2/snnTorchで比較。
**次フェーズ条件**: LIF・STDPを使い、単純な分類タスク（例：MNIST二値分類）でチャンスレート以上の精度が出ること。

---

## Phase2：線虫モジュール（反射・最低限制御）（v0.2）

**目的タグ**: Reflex / Event Detection / Filtering

- **目安ニューロン数**：C. elegansの神経系は約302ニューロン。モジュールとしては**20〜50ニューロン**程度に単純化した「反射弓サブセット」から始めるのが現実的。
- **評価タスク**：単純な走性行動の模倣（例：センサー入力に対する回避/接近の二値反射、あるいはOpenAI Gym的な単純環境での即時回避行動）。
- **成功条件**：反射応答の遅延が既定閾値以下、かつ誤反応率が一定以下であること。

### 必要な論文・書籍
- White et al. (1986) *The Structure of the Nervous System of the Nematode C. elegans*（コネクトームの原典）
- OpenWorm プロジェクトの技術文書（オープンソースC. elegansシミュレーション、参照実装として有用）

### 実装評価
- Kaggle実装可否：◎（CPUのみ、Brian2推奨）
- 難易度：低〜中
- 想定期間：1〜2週間
- 必要GPU：不要
- 再利用OSS：**OpenWorm**（コネクトームデータ・シミュレーションコードが公開されており、ニューロン数・結合パターンの参考に直接使える）

### GitHubコミット内容
```
snnai/modules/c_elegans/
├── connectome_subset.json   # OpenWormベースの簡略化結合
├── reflex_module.py
└── tests/test_reflex_latency.py
```
**次フェーズ条件**: 反射モジュール単体が他モジュールと独立に動作し、割り込み動作（他処理を中断して反射を優先）が実装できること。

---

## Phase3：ミツバチモジュール（空間認識・学習・強化学習）（v0.3 = MVP）

**目的タグ**: Feature Learning / Sparse Memory / Reinforcement

- **目安ニューロン数**：ミツバチの脳は約96万ニューロンだが、キノコ体（学習中枢）に相当する機能単位は**数百〜数千**規模で模擬可能。まずは**500〜2000ニューロン**規模から開始。
- **評価タスク**：グリッドワールドでの経路学習（強化学習）、簡易な花色・位置連合学習（パブロフ型条件づけの模倣）。
- **実装方法**：STDP＋報酬変調STDP（R-STDP）による強化学習則が生物学的妥当性と実装容易さのバランスが良い。

### 必要な論文・書籍
- Menzel & Giurfa (2001) "Cognitive architecture of a mini-brain: the honeybee"
- Izhikevich (2007) "Solving the Distal Reward Problem through Linkage of STDP and Dopamine Signaling"（R-STDPの基礎）
- Nowotny et al. によるミツバチ嗅覚学習の計算モデル論文群

### 実装評価
- Kaggle実装可否：◎（T4 1基で十分）
- 難易度：中
- 想定期間：3〜4週間
- 必要GPU：T4×1
- 推奨ライブラリ：snnTorch（R-STDPはBindsNETのサンプルも参考になる）
- サンプルコード：BindsNETに強化学習向けSNNサンプルあり

### GitHubコミット内容
```
snnai/modules/honeybee/
├── spatial_encoding.py     # グリッドワールド状態のスパイクエンコーディング
├── r_stdp.py
├── train_gridworld.py
└── tests/test_learning_curve.py
```
**ベンチマーク**: グリッドワールドでの収束エピソード数をQ学習ベースラインと比較。
**次フェーズ条件**: 未学習環境で有意な学習曲線の改善が確認できること（ランダム方策との有意差）。

---

## Phase4：カラスモジュール（推論・短期記憶・ワーキングメモリ）（v0.4）

**目的タグ**: Working Memory / Planning / Rule Learning

- **実装方法**：純粋なフィードフォワードSNNでは高次認知の再現が難しいため、**再帰結合（Recurrent SNN）＋短期可塑性（Short-Term Plasticity）**でワーキングメモリを模擬。
- **評価タスク**：遅延見本合わせ課題（delayed match-to-sample）— ワーキングメモリの標準的な評価パラダイム。

### 必要な論文・書籍
- Nieder & Miller (2002) 系のワーキングメモリ神経科学研究（カラス自体の研究としてはNieder研究室の一連の論文が有名）
- Bellec et al. (2020) "A solution to the learning dilemma for recurrent networks of spiking neurons"（LSNN, Long Short-Term Spiking Neural Networks — ワーキングメモリ実装の直接的参考）

### 実装評価
- Kaggle実装可否：○（T4×1〜2、メモリ管理に注意）
- 難易度：高
- 想定期間：4〜6週間
- 必要GPU：T4×2推奨（再帰SNNの学習は勾配計算が重い）
- 推奨ライブラリ：snnTorch（BPTT対応）
- 再利用OSS：Bellec et al.のLSNN公式実装（TensorFlow版だが移植の参考になる）

### GitHubコミット内容
```
snnai/modules/corvid/
├── recurrent_lsnn.py
├── working_memory_task.py
└── tests/test_delay_match_accuracy.py
```
**次フェーズ条件**: 遅延見本合わせ課題で遅延時間を延ばしても一定精度を維持できること。

---

## Phase5：タコモジュール（分散処理・モジュール通信・イベント駆動）（v0.5）

**目的タグ**: Parallel Processing / Independent Expert / Dynamic Routing

- **設計方針**：タコの神経系は腕ごとに自律した神経節を持つ分散構造が特徴。これを模して**独立したサブネット×複数（例：4〜8個）**を用意し、中央ハブと非同期にイベントをやり取りする設計にする。
- **評価タスク**：複数センサー入力の並列処理課題（例：異なる入力チャネルを異なるサブネットが独立処理し、統合結果が単一処理より速い/正確であることを示す）。

### 必要な論文・書籍
- Hochner (2012) "An embodied view of octopus neurobiology"
- Zullo & Hochner (2011) "A new perspective on the organization of an invertebrate brain"

### 実装評価
- Kaggle実装可否：○（マルチプロセスまたは非同期イベントループの実装が必要、CPU中心）
- 難易度：高（分散システム設計の要素が加わる）
- 想定期間：3〜5週間
- 必要GPU：各サブネットはCPU可、統合時のみT4
- 推奨ライブラリ：snnTorch＋Pythonの`asyncio`または`multiprocessing`でイベントバスを自作
- サンプルコード：分散SNN特化のOSSは少なく、ここは独自実装比率が高くなる（要注意点として明記）

### GitHubコミット内容
```
snnai/modules/octopus/
├── event_bus.py
├── subnet_worker.py
├── distributed_config.yaml
└── tests/test_parallel_throughput.py
```
**次フェーズ条件**: サブネット数を増やした際にスループットが単一ネット構成より向上する（線形ではなくとも改善が見える）こと。

---

## Phase X：Bio-NAS（生物制約付きアーキテクチャ探索）（v0.6）

**目的**: 「線虫→蜂→鳥」のような固定順序が本当に最適かは自明ではありません。モジュールが単体で動く状態（Phase5終了時点）になったら、Phase6で結線を固定する前に、**モジュール配置・接続トポロジー自体を探索対象にする**フェーズを設けます。これは本ロードマップの中で最も研究的独自性が高い部分です。

### 探索空間の定義
- **接続パターン**: 直列（線虫→蜂→鳥→タコ）／ハブ型（全モジュールがハブに接続）／部分並列（例：蜂とタコが並列、鳥がその後段）など、あらかじめ数パターンをテンプレート化。
- **モジュール数・組み合わせ**: 4モジュール全部を使うか、タスクによっては一部モジュールを省略するか。
- **生物制約**: 探索は完全自由なNASではなく、「線虫モジュールは必ず入力に最も近い位置に置く（反射は末端処理であるべき）」等、生物学的に妥当な制約をハード制約として与える（制約なしNASは組み合わせ爆発でKaggleの計算資源では非現実的なため）。

### 実装方法（Kaggleでの現実的な探索手法）
- 本格的な勾配ベースNAS（DARTS等）はSNNとの相性・実装コストの面でKaggleには重すぎるため、**進化的アルゴリズム（Evolution Strategy）またはランダムサーチ＋早期打ち切り（successive halving）**を採用するのが現実的。
- 各候補構造は、Phase0.5で作った再現実装を使った軽量タスク（MNIST/MiniGridの縮小版等）で高速スクリーニングし、有望な上位数個のみをフル評価する。

### 必要な論文
- Zoph & Le (2017) "Neural Architecture Search with Reinforcement Learning"（NASの基礎）
- Real et al. (2019) "Regularized Evolution for Image Classifier Architecture Search"（進化的NAS、Kaggleでも実装しやすい）
- 生物制約付きNASという枠組み自体は先行研究が非常に少なく、ここが独自性の核になります。

### 実装評価
- Kaggle実装可否：○（軽量タスクでのスクリーニングに限定すれば実行可能。フル評価は計算コスト大なので候補を絞ることが必須）
- 難易度：非常に高い
- 想定期間：4〜8週間
- 必要GPU：T4×2（並列に複数候補を評価するため）

### GitHubコミット内容
```
snnai/bio_nas/
├── search_space.py       # 接続パターンのテンプレート定義
├── evolution_search.py
├── constraints.py         # 生物制約（線虫は入力側固定 等）
├── results/
│   └── top5_architectures.md
└── tests/test_search_space_validity.py
```
**次フェーズ条件**: 探索によって選ばれた上位構造が、素朴な直列構造（元のアイデア）を軽量ベンチマークで上回るか、少なくとも同等以上であることを確認できること。Phase6の統合構造はこの結果を反映して決定する。

---

## Phase6：異種ニューラルネットワーク統合（v1.0候補）

**注**: 以下は0章で示した初期案の統合設計です。実際の最終構造はPhase X（Bio-NAS）の探索結果を優先し、必要に応じて置き換えてください。

### 統合設計
- **通信方式**：スパイクイベントを共通フォーマット（タイムスタンプ＋発火ニューロンID＋強度）にシリアライズし、ハブ経由でルーティング。
- **情報表現の橋渡し**：モジュール間でニューロン数・時間スケールが異なるため、**レートコーディング⇄タイミングコーディングの変換層**を各モジュール境界に挿入する。
- **学習方法**：モジュール単体は既存の学習則（STDP/R-STDP/BPTT代理勾配）を維持しつつ、モジュール間の結合重みのみ**end-to-endの強化学習（報酬信号）**で調整する二段階学習が現実的（全体を一度に学習すると勾配消失・組み合わせ爆発が起きやすい）。

### 実装評価
- Kaggle実装可否：△〜○（統合部分はT4×2フル活用、セッション制限との戦いになる）
- 難易度：非常に高
- 想定期間：6〜10週間
- 必要GPU：T4×2 or P100
- 参考論文：Yamazaki et al.系の脳型AIアーキテクチャ研究、Eliasmith (2012) "Spaun"（複数脳領域を統合した唯一の大規模事例、最重要参考）

### GitHubコミット内容
```
snnai/integration/
├── hub.py
├── encoding_bridge.py
├── two_stage_training.py
└── tests/test_end_to_end_pipeline.py
```
**ベンチマーク**: 統合後の総合タスク（複数モジュールが必要な複合課題）での成功率。
**次フェーズ条件**: 単体モジュールの性能を大きく損なわずに統合パイプラインが動作すること。

---

## Phase7：SNNAI Version1（最初の完成版）（v1.0）

**ベンチマーク到達目標**: 0.5節のベンチマーク段階のうち、少なくとも段階6（Atari一部タイトル）まで到達し、段階7（Minecraft）以降はv2の目標とする。

### 性能目標の考え方
Transformerとの直接比較は**推奨しません**。パラメータ規模・学習データ量が桁違いのため、公平な比較にならず、かつ本プロジェクトの目的（省電力・少数ニューロンでの機能実現）とも一致しません。

比較するなら以下の軸が妥当です：
- **同規模パラメータのANN（多層パーセプロン等）との比較**（公平な土俵）
- **消費電力・推定エネルギー効率**（SNNの本来の強み。Kaggle上では推定値ベースになる点に注意）
- **少数ニューロンでのタスク達成能力**（Transformerではなく、小型RNN/LSMとの比較が適切）

### GitHubコミット内容（v1.0タグ）
```
snnai/
├── VERSION (v1.0)
├── benchmarks/
│   ├── vs_mlp.py
│   ├── energy_estimation.py
│   └── results.md
└── docs/paper_draft_notes.md   # 新規性の整理（下記参照）
```

---

## 現実的制約まとめと代替案

| 課題 | 内容 | 代替案 |
|---|---|---|
| セッション時間制限 | Kaggleは連続実行に上限あり | 各Phaseでチェックポイント保存を必須化、Kaggle Datasetsに中間モデルを保存して再開 |
| Lava/実チップ不可 | Loihi等の専用ハードウェアなし | シミュレーションのみで概念検証を行い、将来的にIntel Neuromorphic Research CommunityへのアクセスかLoihi開発ボード購入を検討 |
| 分散処理（Phase5）はOSSが少ない | 独自実装比率が高くなる | 完全な分散システムを狙わず、まずはマルチスレッド疑似分散から始める |
| TPU×SNNの未成熟 | Spyxはドキュメント・実績が少ない | Phase0-5ではPyTorch系に統一し、TPUはPhase6以降の高速化検証にのみ限定利用 |
| 統合フェーズは研究段階 | Phase6は既存の確立手法がほぼ存在しない | Spaun等の先行研究を参考にしつつ、小規模な2モジュール統合から段階的に拡張する |

---

## MVPと長期ビジョン

### MVP（最小実用版）の定義
**v0.3**（Phase0〜Phase3：環境構築＋既存研究再現＋線虫モジュール＋ミツバチモジュール）が完了し、**2モジュールが連携して単純な「反射＋学習」タスク**（例：障害物回避しながら報酬のある方向へ学習的に移動する）をこなせる状態。これは数ヶ月規模で到達可能で、GitHub上で「動くデモ」として提示できます。

### 長期ビジョン
**v1.0**（Phase7、Bio-NASによる最適構造採用済み）までを完了し、4モジュール統合SNNAIが単体では実現困難な複合タスクをこなせる状態。ここまでで半年〜1年規模を想定するのが現実的です。v2以降はMinecraft・実ロボットタスクへの拡張、マルチモーダル対応（当初除外したアイデア2）が視野に入ります。

### 論文レベルの新規性が狙える段階
- **Phase5（タコ型分散モジュール）**: 分散SNN・イベント駆動通信のOSS実装例は少なく、実装自体が新規性を持ちうる。
- **Phase6（異種統合）**: 「複数の生物種の神経アーキテクチャを模した異種SNNの統合手法」は、Spaunのような先行研究はあるものの数が非常に少なく、独自の統合手法（特に二段階学習や情報表現の変換層）を定式化できれば新規性を主張しやすい。
- Phase0〜4は既存研究の再現・応用が中心のため、新規性よりも「基盤構築」と捉えるのが妥当です。

---

以上がロードマップです。次のステップとして、Phase0の`kaggle_setup.ipynb`の中身（ライブラリバージョン固定、GPU確認コード）を具体的に書き起こすこともできます。必要であれば教えてください。
