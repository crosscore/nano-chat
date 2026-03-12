# nanochat × DGX Spark 実行可能性分析

## 結論：**動作可能です。ただし規模に制限あり。**

DGX Sparkでnanochatの全パイプライン（pretraining → SFT → RL → 推論/チャットUI）を動作させることは**可能**です。自己学習ループ（RL）も含めて実行できます。

---

## DGX Spark スペック

| 項目 | スペック |
|------|---------|
| GPU | GB10 Grace Blackwell Superchip (Blackwell GPU) |
| CUDAコア | 6,144 |
| メモリ | 128GB LPDDR5x **統合メモリ**（CPU/GPU共有） |
| メモリ帯域幅 | 273 GB/s |
| AI性能 | 最大 1 PFLOPS（FP4、Tensor Core + Sparsity） |
| 対応精度 | BF16 / FP16 / FP32 / FP8 / FP4 |

> [!IMPORTANT]
> DGX Sparkは**単一GPU**です。nanochatのspeedrun.shは8×H100向けに設計されているため、そのまま実行すると**約8倍**の時間がかかります。

---

## nanochat 学習パイプライン概要

nanochatでは`--depth`パラメータ1つでモデルの全ハイパーパラメータが自動計算されます：

| depth | model_dim | パラメータ数（概算） | メモリ要件（概算）| 用途 |
|-------|-----------|---------------------|-------------------|------|
| 6 | 384 | ~30M | ~1GB | テスト・学習用 |
| 12 | 768 | ~125M | ~2-3GB | 研究用 (GPT-1規模) |
| 20 | 1280 | ~475M | ~8-10GB | デフォルト |
| 24 | 1536 | ~800M | ~15-20GB | GPT-2相当 |

### 各ステージ

```
1. Tokenizer訓練 → BPE tokenizer（vocab_size=32768）
2. Base Model Pretraining → GPTモデルの事前学習
3. SFT（Supervised Fine-Tuning） → 会話形式に微調整
4. RL（Reinforcement Learning） → GSM8K数学タスクでGRPO強化学習
5. 推論/チャット → CLI or Web UI
```

---

## DGX Sparkでの実行計画

### ✅ すぐに実行可能

nanochatは**単一GPU**で問題なく動作します。READMEにも明記されています：

> *All code will run just fine on even a single GPU by omitting `torchrun`, and will produce ~identical results (code will automatically switch to gradient accumulation)*

### 推奨設定

DGX SparkのBlackwell GPUはSM 100+であり、**BF16**が自動検出されます。128GBの統合メモリにより、かなり大きなモデルも訓練可能です。

```bash
# DGX Spark用 推奨実行例（depth=12, 研究用, ~5分）
python -m scripts.base_train \
    --depth=12 \
    --device-batch-size=16 \
    --run=dummy

# より大きなモデル（depth=20, デフォルトサイズ）
python -m scripts.base_train \
    --depth=20 \
    --device-batch-size=16 \
    --run=dummy
```

> [!TIP]
> 統合メモリ128GBあるため、`--device-batch-size`は比較的大きく設定できます。OOMが出たら16→8→4と減らしてください。

### GPT-2フルトレーニング（depth=24）

8×H100で約2時間 → DGX Spark単一GPUでは**約16-24時間**かかる見込みです。計算性能差（H100 989 TFLOPS vs DGX Spark推定~30-50 TFLOPS BF16）を考慮すると、さらに長くなる可能性があります。

---

## 自己学習ループ（RL）について

### 既に実装済み ✅

nanochatには[scripts/chat_rl.py](file:///Users/yuu/repos/nano-chat/nanochat/scripts/chat_rl.py)が含まれており、**GSM8K数学問題**を使ったGRPO（簡易版REINFORCE）ベースの強化学習が実装されています。

```bash
# SFT完了後にRL実行
python -m scripts.chat_rl
```

### RLの仕組み

1. SFTモデルを読み込み
2. GSM8K訓練問題からプロンプトを生成
3. モデルが複数のサンプルを生成（rollout）
4. 正解/不正解でリワードを計算
5. Policy Gradient (REINFORCE) で学習
6. 繰り返し → 数学問題の正答率が向上

### 自己学習ループの拡張可能性

現在のRLは**GSM8K（数学タスク）**に限定されていますが、nanochatの設計はシンプルで拡張しやすいです：

- `tasks/` ディレクトリに新しいタスクを追加可能
- `tasks/gsm8k.py` をテンプレートとして、新しいリワード関数を定義可能
- カスタムJSONLデータ（`tasks/customjson.py`）でオリジナルの学習データも使用可能

---

## セットアップ手順

```bash
# 1. DGX Sparkに接続
ssh spark

# 2. リポジトリをDGX Sparkにクローン（またはrsync）
git clone https://github.com/karpathy/nanochat.git
cd nanochat

# 3. 環境セットアップ
export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"
mkdir -p $NANOCHAT_BASE_DIR
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv
uv sync --extra gpu
source .venv/bin/activate

# 4. データダウンロード + Tokenizer訓練
python -m nanochat.dataset -n 8
python -m scripts.tok_train

# 5. Base Model訓練（まずdepth=12で試す、約5分）
python -m scripts.base_train --depth=12 --device-batch-size=16 --run=dummy

# 6. SFT
curl -L -o $NANOCHAT_BASE_DIR/identity_conversations.jsonl \
    https://karpathy-public.s3.us-west-2.amazonaws.com/identity_conversations.jsonl
python -m scripts.chat_sft --device-batch-size=16 --run=dummy

# 7. RL自己学習ループ
python -m scripts.chat_rl --device-batch-size=8

# 8. チャットUI
python -m scripts.chat_web
```

---

## 注意点

> [!WARNING]
> - DGX Sparkのメモリ帯域幅（273 GB/s）はH100（3.35 TB/s）の約1/12であり、特に推論時のスループットに影響します
> - FP8訓練はH100+向けに最適化されており、DGX SparkのBlackwell GPUとの互換性は要検証です
> - `torchrun`を使わず単一GPU実行する必要があります（`torchrun`はマルチGPU用）
> - `--fp8`フラグは除外することを推奨（Hopper以降向けのカスタム実装のため）

> [!NOTE]
> 128GBの統合メモリは大きな利点です。通常のGPU（80GB VRAM + 別途CPU RAM）と異なり、メモリ不足になりにくいため、`--device-batch-size`を比較的大きく設定できます。
