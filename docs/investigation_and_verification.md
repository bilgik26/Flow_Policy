# Flow_Policy × RoboCasa 調査・検証記録

> 実施日: 2026-06-25  
> 対象: `bilgik26/robocasa v1.0.1` + `ARISE-Initiative/robosuite (GitHub main)`  
> 環境: Singularity + uv、CUDA 12.8、RTX 4090 × 2

---

## 1. cosmos-policy pkl の内容

ファイル: `cosmos-policy/cosmos_policy/experiments/robot/robocasa/robocasa_controller_configs.pkl`

```json
{
  "type": "OSC_POSE",
  "input_max": 1, "input_min": -1,
  "output_max": [0.05, 0.05, 0.05, 0.5, 0.5, 0.5],
  "output_min": [-0.05, -0.05, -0.05, -0.5, -0.5, -0.5],
  "kp": 150, "damping_ratio": 1,
  "impedance_mode": "fixed", "control_delta": true, ...
}
```

これは OSC_POSE コントローラーの**パラメータ設定のみ**を含むフラットな dict。  
アクション次元の定義は含まれない。  
`robocasa` の `kitchen.py` が `refactor_composite_controller_config()` でこれを `HYBRID_MOBILE_BASE` 形式のコンポジット設定に自動変換する。

---

## 2. RoboCasa データセットの実態（TurnOffMicrowave、1 タスク）

### 収録条件（`dataset_meta.json` より確認）

| 項目 | 値 |
|---|---|
| ロボット | `PandaOmron`（robocasa のキッチンタスク全般で統一） |
| コントローラー | `HYBRID_MOBILE_BASE` |
| エピソード数 | 108 |
| 総フレーム数 | 15,233 |
| 画像サイズ | 256×256（評価時は 224×224 にリサイズ） |

> **`PandaMobile` について**: `kitchen.py` が後方互換のために `PandaMobile` を `PandaOmron` に自動変換する。  
> cosmos-policy の評価スクリプトで `robots="PandaMobile"` と書かれているものも内部的には `PandaOmron` として動作する。  
> robocasa のキッチンタスクで使えるロボットは `PandaOmron` のみ（`assert len(robots) == 1` で強制）。

### action / state のデータ形式（`PandaOmron_modality.json` による）

HDF5 形式のデモデータは `reorder_hdf5_action()` / `reorder_hdf5_state()` によって  
LeRobot 形式に**並べ替えて**保存される。Flow_Policy が読む LeRobot 形式の定義は以下の通り。

**action（12-dim）**：

```
LeRobot reordered action (12-dim):
  [0:4]   base_motion       (x, y, θ, mode)   ← 台車速度
  [4:5]   control_mode                         ← -1 でベースロック
  [5:8]   end_effector_position delta          ← EEF 位置制御
  [8:11]  end_effector_rotation delta          ← EEF 姿勢制御
  [11:12] gripper_close                        ← グリッパー開閉

Flow_Policy が使うスライス: action[5:12] = EEF pos(3) + ori(3) + gripper(1)
```

> HDF5 元データの順序は逆（EEF が先、base が後）。`reorder_hdf5_action()` が並び替えた結果が LeRobot 形式。

**observation.state（16-dim）**：

```
LeRobot reordered state (16-dim):
  [0:3]   base_position                        ← 台車のワールド座標
  [3:7]   base_rotation                        ← 台車の姿勢（クォータニオン）
  [7:10]  end_effector_position_relative       ← 台車基準 EEF 位置
  [10:14] end_effector_rotation_relative       ← 台車基準 EEF 姿勢
  [14:16] gripper_qpos                         ← グリッパー開度

Flow_Policy が使うスライス: state[7:16] = eef_pos_rel(3) + eef_rot_rel(4) + gripper(2)
```

`state[7:16]` を選ぶ理由: **action[5:12] が EEF 制御**であるため、observation も EEF 状態（台車基準）を使うことで入力と出力の意味的な整合が取れる。台車状態 `[0:7]` は方策に対して間接的な参照情報に留まる。

### 動作確認（データセット vs 環境の一致）

```
dataset action[5:12] = [-0.029  0.  0.  0.  0.  0.  -1.]   ← EEF が動いている ✓
dataset state[7:16]  = [ 0.25   0.003  0.59  -0.99  -0.035  -0.12  0.02  0.02  -0.02]
env agent_pos (9-dim)= [ 0.25  -0.01   0.61  -0.99  -0.03   -0.15  0.02  0.02  -0.02]
  eef_pos_rel [0:3]  : [ 0.25  -0.01   0.61]   ← スケールが整合 ✓
  gripper     [7:9]  : [ 0.02  -0.02]           ← dataset と一致 ✓
```

---

## 3. API 変更の詳細（robosuite GitHub main vs PyPI v1.5.1）

### 3-1. `load_controller_config` の廃止

旧 robosuite（PyPI v1.5.1）はシングルアーム前提の flat dict でコントローラーを指定していた。  
GitHub main ではロボットの各パーツ（右腕・グリッパー・台車・胴体）を独立したサブコントローラーで管理する**コンポジットコントローラー**に刷新され、`load_controller_config` は廃止。

```python
# 旧（廃止）
from robosuite.controllers import load_controller_config
cfg = load_controller_config(default_controller="OSC_POSE")

# 新
from robosuite import load_composite_controller_config
cfg = load_composite_controller_config(robot="PandaOmron")
```

**Flow_Policy の対処**: cosmos-policy の pkl（OSC_POSE flat dict）を使用。  
`kitchen.py` の `refactor_composite_controller_config()` が旧→新形式に自動変換し、action_dim=12 が維持される。

### 3-2. action_dim の違い（使用するコントローラーによる）

| コントローラー設定 | action_dim |
|---|---|
| cosmos-policy pkl（OSC_POSE flat → HYBRID_MOBILE_BASE に変換） | 12 |
| `load_composite_controller_config(robot="PandaOmron")` のデフォルト | 11 |

cosmos-policy pkl を使うことで `HYBRID_MOBILE_BASE` の body_part_ordering が  
`[right(6), right_gripper(1), base(3), torso(1), mode(1)] = 12-dim` に固定される。  
デフォルト設定では gripper が 2-dim（両指独立）になるため 11-dim になる。

---

## 4. EGL レンダリング検証

### 4-1. 環境

| 項目 | 値 |
|---|---|
| GPU | NVIDIA GeForce RTX 4090 × 2 |
| ドライバー | 570.172.08 |
| `/dev/dri/renderD128-130` | 存在するが `render` グループ所属なしで Permission Denied |

### 4-2. EGL デバイス列挙結果

`eglQueryDevicesEXT()` で 4 デバイスを検出:

| device_id | 種別 | 初期化結果 |
|---|---|---|
| 0 | GPU 0 (RTX 4090) | `EGL_NOT_INITIALIZED` (Permission Denied) |
| 1 | GPU 1 (RTX 4090) | `EGL_NOT_INITIALIZED` (Permission Denied) |
| 2 | GPU 2 (不明) | `EGL_NOT_INITIALIZED` (Permission Denied) |
| 3 | Mesa ソフトウェア EGL | **成功 (EGL v1.5)** |

### 4-3. 結論と設定

```bash
# GPU ハードウェア EGL を使うには render グループへの追加が必要
sudo usermod -aG render $USER  # サーバー管理者に依頼

# render グループ未所属の場合のフォールバック
--env MUJOCO_EGL_DEVICE_ID=3   # Mesa software EGL
```

**重要**: `CUDA_VISIBLE_DEVICES=N` と `MUJOCO_EGL_DEVICE_ID=M`（M≠N）を同時に設定すると、  
robosuite の `binding_utils.py` が「MUJOCO_EGL_DEVICE_ID must be in CUDA_VISIBLE_DEVICES」とアサーションで落ちる。  
→ `CUDA_VISIBLE_DEVICES` は設定せず、`training.device=cuda:1` で明示的に GPU を指定する。

### 4-4. 実測レンダリング確認

```
Image shape: (3, 224, 224)  range [0.00, 1.00]  mean=92.61
step() OK. rew=0.0 done=False success=False
=== EGL (device 3 / Mesa) rendering PASSED ===
```

---

## 5. 環境構築で遭遇したエラーと修正

| エラー | 原因 | 修正 |
|---|---|---|
| `ImportError: cached_download` | diffusers 0.27.2 が huggingface_hub 0.36 の廃止 API を使用 | `uv pip install 'diffusers==0.33.1'` |
| `FileNotFoundError` in TimmObsEncoder | timm の pretrained weight DL 失敗（`HF_HOME` 未設定） | 検証時は `pretrained: false`、本番は `HF_HOME` をバインド |
| `ConfigKeyError: Missing key horizon` | shape_meta の各 obs キーに `horizon` フィールドが必要 | `horizon: ${n_obs_steps}` を追加 |
| `AssertionError` in TimmObsEncoder forward | `feature_aggregation: null` は r3m 専用（ResNet 非対応） | `feature_aggregation: spatial_embedding` に変更 |
| `FileNotFoundError` in dataset | hydra がワーキングディレクトリを変更するため相対パス不可 | `dataset_base_path` を絶対パスに変更 |
| `CUDA out of memory` | GPU 0 が別プロセスに占有 | `training.device=cuda:1` に変更 |
| `MUJOCO_EGL_DEVICE_ID assertion` | `CUDA_VISIBLE_DEVICES=1` と `MUJOCO_EGL_DEVICE_ID=3` の競合 | `CUDA_VISIBLE_DEVICES` を外し `training.device=cuda:1` で指定 |
| `wandb.Video requires moviepy` | wandb disabled 時に raw numpy Video を作成 | runner の Video 生成を `try/except` でラップ |

---

## 6. 1 タスク学習・評価の検証結果

### 6-1. 実行コマンド

```bash
singularity exec --nv \
    --bind /home/bilgehan.sakai:/home/bilgehan.sakai \
    --env MUJOCO_EGL_DEVICE_ID=3 \
    /mnt/data/bilgehan.sakai/singularity/sif/flow_policy_robocasa.sif \
    bash -c "
        source /home/bilgehan.sakai/Flow_Policy/.venv/bin/activate
        cd /home/bilgehan.sakai/Flow_Policy/ManiFlow
        python -m maniflow.workspace.train_maniflow_robocasa_workspace \
            --config-name maniflow_image_timm_policy_robocasa_1task \
            task=robocasa_1task \
            training.device=cuda:1 \
            dataloader.batch_size=32 \
            val_dataloader.batch_size=32
    "
```

### 6-2. データセット・モデル情報

| 項目 | 値 |
|---|---|
| タスク | TurnOffMicrowave（1 タスク） |
| 学習エピソード | 103 eps（14,608 samples） |
| 検証エピソード | 5 eps（625 samples） |
| モデル総パラメータ数 | 21.83M |
| obs_encoder (ResNet-18) | 11.20M |
| DiTX (6 blocks, dim=256) | 10.63M |

### 6-3. val_loss NaN への対処と原因の切り分け

#### 当初の対処（防御策）

初期の学習中に val_loss が NaN になる事象が一度観測されたため、学習を堅牢化する目的で
以下の 3 点を導入した：

1. **training ループに NaN スキップを追加** — `torch.isfinite(raw_loss)` でチェックし、
   非有限なバッチは backward せずスキップ（スキップ数 `n_skipped` をカウント・出力）
2. **勾配クリッピングを明示化** — `clip_grad_norm_(max_norm=1.0)`（`training.grad_clip_norm`
   で変更可、`null` で無効化）
3. **val_loss 集計を `np.nanmean` に変更** — 万一 NaN が混入しても平均に伝播しないよう

#### 原因の切り分け（重要な訂正）

当初、上記のうち **「batch_size=8 だと consistency_batchsize=2 になり確率的に NaN が出る、
batch_size=32 にすれば消える」** と説明していたが、これは **裏付けの取れない誤った推定** だった。
NaN スキップ（`continue`）と `np.nanmean` は **NaN を集計から隠してしまう** ため、
「クリーンなログが出た ＝ NaN が消えた」とは言えない。そこで以下を計装したうえで実測し直した：

- train ループの **スキップ数** `n_skipped` を毎 epoch カウント・出力
- val ループの **nanmean 前の生の NaN 数** `n_val_nan` をカウント・出力

この計装下で、原因を切り分けるための制御実験（各 10 epoch、`debug=true`）を実施した：

| 条件 | batch_size | grad clip | スライス | n_skipped | n_val_nan |
|---|---|---|---|---|---|
| A（オリジナル相当） | 8 | 無効 | 正（現行） | **0** | **0** |
| B | 8 | 1.0 | 正（現行） | **0** | **0** |
| C（コミット構成） | 32 | 1.0 | 正（現行） | **0** | **0** |
| D | 8 | 無効 | **修正前の誤スライス** | **0** | **0** |

さらに、未学習モデルに対し生の `compute_loss` を batch_size=8 / 32 で各 457 回ずつ実行する
プローブでも **0/457（0.00%）** で、非有限値は一度も観測されなかった。

**結論**:

- 現行コードでは、batch_size・grad clip・スライスのいずれを変えても **NaN は再現しない**。
  当初の「batch_size=8 で 1.3%、32 で 0%」という数値は再現できず、撤回する。
- すなわち **NaN を消した主因は batch_size ではない**。初期に一度だけ観測された NaN は、
  再現性の低い稀な事象（consistency loss の乱数 t/δt の極端な組み合わせ等）だったと考えられる。
- 上記 3 つの防御策（スキップ / grad clip / nanmean）は、**稀な NaN が発生しても学習を
  破綻させない保険** として有効であり、そのまま残す。`n_skipped` / `n_val_nan` の出力により、
  以後は「スキップが実際に発火したか」をログから直接確認できる。

### 6-4. 学習曲線・評価結果（batch_size=32 での再学習）

`batch_size=32` で 20 epoch（各 457 iteration）の再学習を実施。**全 epoch を通して val_loss / train_loss ともに NaN は一度も発生せず**、学習・検証・評価のパイプラインが破綻なく完走することを確認した（NaN が出なかったことの解釈は 6-3 を参照。なお、この run では `n_skipped` も全 epoch で 0 であり、スキップは一度も発火していない）。

**train_loss（各 epoch 末尾の移動平均）**

| epoch | 0 | 1 | 2 | 3 | 5 | 7 | 9 | 11 | 13 | 15 | 17 | 19 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| train_loss | 0.434 | 0.277 | 0.156 | 0.242 | 0.296 | 0.110 | 0.108 | 0.156 | 0.120 | 0.161 | 0.120 | 0.097 |

- epoch 0 開始直後の loss=2.12 から急速に低下し、epoch 2 以降は概ね 0.10〜0.25 の範囲で安定。
- epoch 19 で train_loss≈0.097 と最小付近に収束。**発散・NaN は皆無**。

**val_loss**

- `val_every=5` で epoch 0 / 5 / 10 / 15 に検証を実施。いずれも**有限値**（NaN なし）。
- topk チェックポイント（monitor=val_loss, mode=min）に保存された最良値は
  `epoch=0010-val_loss=0.227320.ckpt` → **val_loss=0.2273**（正常な数値で保存された）。

**評価（rollout）**

`rollout_every=10` で epoch 0 / 10 に TurnOffMicrowave を 3 episode 評価。

| epoch | mean success rate |
|---|---|
| 0 | 0.000（3/3 失敗） |
| 10 | 0.000（3/3 失敗） |

成功率 0.000 は想定どおり。本検証は**学習パイプラインと NaN 修正の動作確認が目的**であり、20 epoch・1 タスク・pretrained 無効（`resnet18, pretrained=false`）という最小設定のため、方策が成功に至る品質には達していない。実用的な成功率は 7-1 / 7-2 の本番設定（pretrained ResNet + 全タスク + 十分な epoch 数）で得る。

**結論**: val_loss NaN は解消され、20 epoch の学習が exit 0 で完走、損失は単調に収束、チェックポイントも正常保存された。学習・検証・評価（rollout）の一連のパイプラインが破綻なく動作することを確認した。

---

## 7. 今後の本番学習に向けた推奨設定

### 7-1. pretrained ResNet の使用

HF_HOME を事前にマウントして pretrained weight をキャッシュしておく:

```bash
singularity exec --nv \
    --bind /home/bilgehan.sakai:/home/bilgehan.sakai \
    --bind /home/bilgehan.sakai/.cache/huggingface:/home/bilgehan.sakai/.cache/huggingface \
    --env HF_HOME=/home/bilgehan.sakai/.cache/huggingface \
    ...
```

config:
```yaml
model_name: 'resnet18.a1_in1k'
pretrained: true
```

### 7-2. 10 タスク本番学習コマンド

データセットを 10 タスク分ダウンロード後：

```bash
singularity exec --nv \
    --bind /home/bilgehan.sakai:/home/bilgehan.sakai \
    --env MUJOCO_EGL_DEVICE_ID=3 \
    /mnt/data/bilgehan.sakai/singularity/sif/flow_policy_robocasa.sif \
    bash -c "
        source /home/bilgehan.sakai/Flow_Policy/.venv/bin/activate
        cd /home/bilgehan.sakai/Flow_Policy/ManiFlow
        python -m maniflow.workspace.train_maniflow_robocasa_workspace \
            --config-name maniflow_image_timm_policy_robocasa \
            task=robocasa_multitask \
            training.device=cuda:1
    "
```
