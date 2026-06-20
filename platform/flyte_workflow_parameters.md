# Flyte Workflow パラメータリファレンス

Flyte Console (`https://d1fk8c95f6ice9.cloudfront.net/console`) から各 workflow を Launch する際のパラメータ説明。

Project: `auto-e2e` / Domain: `development`

---

## wf_data_ingest

生データセットをダウンロードし、学習用 WebDataset shards に変換する。

| パラメータ | 型 | デフォルト | UI | 説明 |
|-----------|-----|-----------|-----|------|
| `dataset` | Enum | `L2D` | ドロップダウン | 取得元データセット。`L2D` = yaak-ai/L2D (HuggingFace)、`NVIDIA_PHYSICAL_AI` = nvidia/PhysicalAI |
| `version_tag` | str | `10hz-224px-v1` | テキスト | 加工バージョンのタグ。同じデータを別設定で再処理する時に変える (例: `20hz-256px-v2`) |
| `hz` | int | `10` | 数値 | エゴモーション/フレームのリサンプリング周波数 (Hz)。高いほどデータ量増 |
| `image_size` | int | `224` | 数値 | カメラ画像のリサイズ先 (正方形 px)。モデル入力サイズに合わせる |
| `episodes` | int | `5` | 数値 | 処理するエピソード数。全量なら `-1`、テストなら `1-5` |

**出力**: `FlyteDirectory` (S3上の WebDataset shards ディレクトリ)

---

## wf_train_il

Imitation Learning (教師あり学習)。Expert demonstrations から運転 policy を学習する。

| パラメータ | 型 | デフォルト | UI | 説明 |
|-----------|-----|-----------|-----|------|
| `shards` | FlyteDirectory | (必須) | URI入力 | `wf_data_ingest` の出力 URI。Flyte UI で前回実行の Outputs からコピー |
| `backbone` | Enum | `SWIN_V2_TINY` | ドロップダウン | 画像エンコーダ。`SWIN_V2_TINY` (22M params, 良バランス)、`CONVNEXT_V2_TINY` (28M, 高精度)、`RESNET_50` (25M, 高速) |
| `fusion_mode` | Enum | `CONCAT` | ドロップダウン | マルチカメラ特徴量統合方式。`CONCAT` (単純結合, 最速)、`CROSS_ATTN` (注意機構, 高精度)、`BEV` (Bird's Eye View, 空間理解) |
| `epochs` | int | `10` | 数値 | 学習エポック数。少ないと underfitting、多いと overfitting |
| `batch_size` | int | `4` | 数値 | ミニバッチサイズ。GPU メモリに依存 (g6e.4xlarge L40S 48GB なら 8-16 まで可) |
| `lr` | float | `0.001` | 数値 | 学習率。大きいと発散、小さいと収束遅い。通常 `0.0001` - `0.001` |

**出力**: `FlyteFile` (best checkpoint `.pt`)。MLflow に params/metrics/model 自動記録。

---

## wf_evaluate

学習済みモデルの Open-Loop 評価。予測軌道と正解軌道を比較。

| パラメータ | 型 | デフォルト | UI | 説明 |
|-----------|-----|-----------|-----|------|
| `checkpoint` | FlyteFile | (必須) | URI入力 | `wf_train_il` の出力 URI (checkpoint ファイル) |
| `shards` | FlyteDirectory | (必須) | URI入力 | 評価用データ。`wf_data_ingest` の出力 URI |

**出力**: `EvalMetrics` (NamedTuple)
- `ade` (float): Average Displacement Error (m) — 全タイムステップの平均ずれ。小さいほど良い
- `fde` (float): Final Displacement Error (m) — 最終地点のずれ。小さいほど良い
- `gate_pass` (bool): ade < 2.0 かつ fde < 4.0 なら True

MLflow に metrics 自動記録。

---

## wf_train_offline_rl

Offline RL (IQL) で IL policy を改善。シミュレータ不要、recorded data のみ使用。

| パラメータ | 型 | デフォルト | UI | 説明 |
|-----------|-----|-----------|-----|------|
| `pretrained` | FlyteFile | (必須) | URI入力 | `wf_train_il` の出力 URI (IL checkpoint) |
| `shards` | FlyteDirectory | (必須) | URI入力 | 学習データ。`wf_data_ingest` の出力 URI |
| `epochs` | int | `5` | 数値 | RL 学習エポック数 |
| `tau` | float | `0.7` | 数値 | IQL expectile parameter。高いほど保守的 (0.5=mean, 1.0=max)。通常 `0.7-0.9` |
| `beta` | float | `3.0` | 数値 | Advantage weight temperature。高いほど expert に近い行動を重視。通常 `1.0-10.0` |

**出力**: `FlyteFile` (RL-refined checkpoint `.pt`)。MLflow に metrics 自動記録。

---

## wf_full_pipeline

上記4ステージを直列実行 (Ingest → IL Train → Eval → Offline RL)。

| パラメータ | 型 | デフォルト | UI | 説明 |
|-----------|-----|-----------|-----|------|
| `dataset` | Enum | `L2D` | ドロップダウン | (wf_data_ingest と同じ) |
| `version_tag` | str | `10hz-224px-v1` | テキスト | (wf_data_ingest と同じ) |
| `backbone` | Enum | `SWIN_V2_TINY` | ドロップダウン | (wf_train_il と同じ) |
| `fusion_mode` | Enum | `CONCAT` | ドロップダウン | (wf_train_il と同じ) |
| `epochs_il` | int | `10` | 数値 | IL epochs |
| `epochs_rl` | int | `5` | 数値 | RL epochs |
| `batch_size` | int | `4` | 数値 | IL batch size |
| `lr` | float | `0.001` | 数値 | IL learning rate |

**出力**: `FlyteFile` (最終 RL-refined checkpoint)

---

## 初回実行ガイド

### 最も簡単: `wf_full_pipeline` をデフォルトのまま Launch

すべてデフォルト値でOK。L2D データ取得 → 学習 → 評価 → RL まで自動実行。

### 個別実行する場合の順序

1. `wf_data_ingest` → 出力 URI をメモ
2. `wf_train_il` → `shards` に 1 の出力 URI を貼り付け → 出力 URI をメモ
3. `wf_evaluate` → `checkpoint` に 2 の出力、`shards` に 1 の出力
4. `wf_train_offline_rl` → `pretrained` に 2 の出力、`shards` に 1 の出力

### FlyteDirectory / FlyteFile URI の見つけ方

Flyte Console → Executions → 該当実行をクリック → Outputs タブ → URI をコピー

形式例:
```
s3://auto-e2e-platform-artifacts-381491877296/data/abc123/...
```
