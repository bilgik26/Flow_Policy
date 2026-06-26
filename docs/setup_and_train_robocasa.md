# Flow_Policy × RoboCasa セットアップ・学習・評価ガイド

> **対象:** `Flow_Policy` リポジトリで RoboCasa (bilgik26/robocasa v1.0.1) を使って  
> Flow Matching Policy の学習と評価を行うための完全手順書。  
> 環境構築は **Singularity + uv** で行い、conda は使用しない。

---

## 前提条件

| 項目 | 要件 |
|---|---|
| GPU | NVIDIA GPU (A100 推奨) + CUDA 12.8 ドライバ |
| Singularity | 3.x 以上 |
| ディスク空き容量 | 10 タスクで約 30 GB（全 65 タスクで 100 GB 超） |
| HuggingFace | アカウント不要（RoboCasa データセットは公開済み） |

---

## Step 1: Singularity イメージのビルド

```bash
export CODEBASE_DIR="/mnt/data/bilgehan.sakai"
export SINGULARITY_TMPDIR="$CODEBASE_DIR/singularity/tmp_build"
export SINGULARITY_CACHEDIR="$CODEBASE_DIR/singularity/singularity_cache"
mkdir -p "$SINGULARITY_TMPDIR" "$SINGULARITY_CACHEDIR"

singularity build \
    "$CODEBASE_DIR/singularity/sif/flow_policy_robocasa.sif" \
    "$CODEBASE_DIR/singularity/flow_policy_robocasa.def"
```

ビルド所要時間: 約 5〜10 分。

---

## Step 2: Python 仮想環境のセットアップ (uv)

Singularity コンテナ内で `uv` を使い、`Flow_Policy` ディレクトリに `.venv` を作成する。

```bash
export CODEBASE_DIR="/mnt/data/bilgehan.sakai"
SIF="$CODEBASE_DIR/singularity/sif/flow_policy_robocasa.sif"

singularity exec --nv \
    --bind "$CODEBASE_DIR:/mnt/data/bilgehan.sakai" \
    --bind "$HOME/.cache:/root/.cache" \
    "$SIF" \
    bash -c "
        cd /mnt/data/bilgehan.sakai/Flow_Policy
        uv sync --extra cu128 --python 3.10
    "
```

インストールされる主なパッケージ:

| パッケージ | バージョン |
|---|---|
| torch | 2.7.0+cu128 |
| torchvision | 0.22.0+cu128 |
| hydra-core | 1.2.0 |
| diffusers | 0.27.2 |
| einops | 0.8.1 |
| av (PyAV) | 最新安定版 |
| pandas / pyarrow | 最新安定版 |

---

## Step 3: RoboCasa のインストール

### 3-1. リポジトリのクローン（ホスト上）

```bash
cd /mnt/data/bilgehan.sakai/Flow_Policy
git clone -b dev https://github.com/bilgik26/robocasa.git robocasa
```

### 3-2. コンテナ内で依存パッケージをインストール

```bash
export CODEBASE_DIR="/mnt/data/bilgehan.sakai"
SIF="$CODEBASE_DIR/singularity/sif/flow_policy_robocasa.sif"

singularity exec --nv \
    --bind "$CODEBASE_DIR:/mnt/data/bilgehan.sakai" \
    --bind "$HOME/.cache:/root/.cache" \
    "$SIF" \
    bash -c "
        source /mnt/data/bilgehan.sakai/Flow_Policy/.venv/bin/activate

        # robosuite を GitHub main 版に差し替え
        # (新 robocasa の kitchen.py が load_model_on_init を使用しており
        #  PyPI 版 robosuite には含まれていないため)
        uv pip install 'robosuite @ git+https://github.com/ARISE-Initiative/robosuite.git'

        # 新 robocasa をインストール (editable)
        uv pip install -e /mnt/data/bilgehan.sakai/Flow_Policy/robocasa

        # numba を安定バージョンに固定
        # (新 robocasa の setup.py が numba==0.61.2 を要求するが
        #  当環境の LLVM で SIGSEGV クラッシュを起こすため)
        uv pip install 'numba==0.63.1' 'llvmlite==0.46.0'

        # Flow_Policy 自身をインストール (editable)
        uv pip install -e /mnt/data/bilgehan.sakai/Flow_Policy/ManiFlow
    "
```

### 3-3. パッチの適用（ホスト上）

#### パッチ 1: editable インストールの名前空間パッケージバグ修正

**問題:** `/workspace` が `sys.path` に入るため Python が robocasa を名前空間パッケージとして誤認識。

```bash
sed -i 's/sys.meta_path.append(_EditableFinder)/sys.meta_path.insert(0, _EditableFinder)/' \
    /mnt/data/bilgehan.sakai/Flow_Policy/.venv/lib/python3.10/site-packages/__editable___robocasa_1_0_1_finder.py
```

#### パッチ 2: NumPy バージョンアサーション修正

**問題:** `robocasa/__init__.py` が `numpy.__version__ in ["2.2.5"]` とアサートするが、インストールされる NumPy は 2.2.6。

```bash
sed -i 's/    "2\.2\.5",/    "2.2.5",\n    "2.2.6",/' \
    /mnt/data/bilgehan.sakai/Flow_Policy/robocasa/robocasa/__init__.py
```

### 3-4. Kitchen アセットのダウンロード

```bash
singularity exec --nv \
    --bind "$CODEBASE_DIR:/mnt/data/bilgehan.sakai" \
    --bind "$HOME/.cache:/root/.cache" \
    "$SIF" \
    bash -c "
        source /mnt/data/bilgehan.sakai/Flow_Policy/.venv/bin/activate
        cd /mnt/data/bilgehan.sakai/Flow_Policy

        # Kitchen アセット (約 3.8GB、初回のみ)
        echo 'y' | python robocasa/robocasa/scripts/download_kitchen_assets.py

        # プライベートマクロファイルのセットアップ
        python robocasa/robocasa/scripts/setup_macros.py
    "
```

### 3-5. 動作確認

```bash
singularity exec --nv \
    --bind "$CODEBASE_DIR:/mnt/data/bilgehan.sakai" \
    --bind "$HOME/.cache:/root/.cache" \
    --env MUJOCO_GL=osmesa \
    "$SIF" \
    bash -c "
        source /mnt/data/bilgehan.sakai/Flow_Policy/.venv/bin/activate
        python -c '
import torch, numba, numpy, robosuite, mujoco, robocasa
print(\"torch:\", torch.__version__, \"cuda:\", torch.cuda.is_available())
print(\"numba:\", numba.__version__)
print(\"numpy:\", numpy.__version__)
print(\"robosuite:\", robosuite.__version__)
print(\"mujoco:\", mujoco.__version__)
print(\"robocasa:\", robocasa.__version__)
print(\"robocasa path:\", robocasa.__file__)
'
    "
```

期待される出力:

```
torch: 2.7.0+cu128 cuda: True
numba: 0.63.1
numpy: 2.2.6
robosuite: 1.5.2
mujoco: 3.3.1
robocasa: 1.0.1
robocasa path: /mnt/data/bilgehan.sakai/Flow_Policy/robocasa/robocasa/__init__.py
```

> **注意:** `robocasa path` の末尾が `robocasa/robocasa/__init__.py` になっていることを確認する。  
> `robocasa/__init__.py` と表示される場合はパッチ 1 が正しく適用されていない。

---

## Step 4: 学習用データセットのダウンロード

### 4-1. データセット構造について

RoboCasa データセットは LeRobot v2.1 形式で配布されている:

```
datasets/v1.0/pretrain/atomic/<TaskName>/
├── meta/
│   ├── info.json
│   ├── tasks.jsonl    # タスク説明文
│   └── episodes.jsonl # エピソードメタデータ
├── data/
│   └── chunk-000/
│       └── episode_000000.parquet   # action(12), observation.state(9), ...
└── videos/
    └── chunk-000/
        └── observation.images.robot0_agentview_left/
            └── episode_000000.mp4
```

### 4-2. 10 タスクのダウンロード（動作確認用）

```bash
singularity exec --nv \
    --bind "$CODEBASE_DIR:/mnt/data/bilgehan.sakai" \
    --bind "$HOME/.cache:/root/.cache" \
    "$SIF" \
    bash -c "
        source /mnt/data/bilgehan.sakai/Flow_Policy/.venv/bin/activate
        for TASK in CloseBlenderLid CloseFridge OpenCabinet OpenDrawer \
                    OpenStandMixerHead PickPlaceCounterToCabinet PickPlaceCounterToStove \
                    PickPlaceDrawerToCounter PickPlaceSinkToCounter PickPlaceToasterToCounter; do
            echo \"=== Downloading: \$TASK ===\"
            python -c \"
from robocasa.scripts.download_datasets import download_datasets
download_datasets(
    split=['pretrain'],
    tasks=['\$TASK'],
    source=['human'],
    overwrite=False,
    output_dir='/mnt/data/bilgehan.sakai/Flow_Policy/data/robocasa/datasets',
)
\"
        done
    "
```

ダウンロード先: `Flow_Policy/data/robocasa/datasets/v1.0/pretrain/atomic/<TaskName>/`

10 タスクのデータ統計（目安）:

| 項目 | 数値 |
|---|---|
| エピソード数 | 約 1,000 |
| 総ステップ数 | 約 220,000 |
| 総データ量 | 約 15〜25 GB |

### 4-3. config のデータパス確認

`ManiFlow/maniflow/config/task/robocasa_multitask.yaml` の `dataset_base_path` がダウンロード先に合っていることを確認する（デフォルト: `data/robocasa/datasets/v1.0/pretrain/atomic`）。

別のパスを使う場合は hydra オーバーライドで指定できる:

```bash
task.dataset_base_path=/absolute/path/to/datasets/v1.0/pretrain/atomic
```

---

## Step 5: 学習の実行

### 5-1. 学習コマンド（1-GPU）

```bash
singularity exec --nv \
    --bind "/home/bilgehan.sakai:/home/bilgehan.sakai" \
    --bind "$CODEBASE_DIR:/mnt/data/bilgehan.sakai" \
    --env MUJOCO_GL=osmesa \
    --env PYOPENGL_PLATFORM=osmesa \
    --env CUDA_VISIBLE_DEVICES=0 \
    "$SIF" \
    bash -c "
        source /home/bilgehan.sakai/Flow_Policy/.venv/bin/activate
        export PYTHONPATH=/home/bilgehan.sakai/Flow_Policy/robocasa:\$PYTHONPATH
        cd /home/bilgehan.sakai/Flow_Policy/ManiFlow

        python -m maniflow.workspace.train_maniflow_robocasa_workspace \
            --config-name maniflow_image_timm_policy_robocasa \
            task=robocasa_multitask \
            training.seed=42 \
            training.device='cuda:0' \
            exp_name=robocasa_10tasks
    "
```

> **レンダリングについて:** コンテナは `PYOPENGL_PLATFORM=egl` をデフォルトで設定しているが、  
> `/dev/dri/renderD*` への権限がないと EGL ハードウェアレンダリングが失敗する。  
> `robosuite` の `binding_utils.py` の assertion により `MUJOCO_EGL_DEVICE_ID=3`（Mesa EGL）も使用不可  
> （`CUDA_VISIBLE_DEVICES` に含まれない値を指定すると `AssertionError`）。  
> **osmesa ソフトウェアレンダリング**（`MUJOCO_GL=osmesa` + `PYOPENGL_PLATFORM=osmesa`）が唯一の解決策。  
> GPU アクセス権を得るには `sudo usermod -aG render $USER` が必要（サーバー管理者に依頼）。

または付属のスクリプトを使用する:

```bash
singularity exec --nv \
    --bind "/home/bilgehan.sakai:/home/bilgehan.sakai" \
    --bind "$CODEBASE_DIR:/mnt/data/bilgehan.sakai" \
    --env MUJOCO_GL=osmesa \
    --env PYOPENGL_PLATFORM=osmesa \
    "$SIF" \
    bash /home/bilgehan.sakai/Flow_Policy/scripts/train_eval_robocasa.sh 0 42 robocasa_10tasks
```

### 5-2. 主要なハイパーパラメータ

| パラメータ | デフォルト値 | 説明 |
|---|---|---|
| `horizon` | 34 | ウィンドウ幅 (n_obs_steps + n_action_steps) |
| `n_obs_steps` | 2 | 観測履歴フレーム数 |
| `n_action_steps` | 32 | 予測アクションチャンクサイズ |
| `training.num_epochs` | 3001 | 学習エポック数 |
| `dataloader.batch_size` | 64 | バッチサイズ |
| `optimizer.lr` | 1e-4 | 学習率 |
| `training.rollout_every` | 500 | 環境ロールアウトの実行間隔（エポック） |
| `policy.num_inference_steps` | 10 | 推論時のフローステップ数 |

hydra オーバーライドで変更可能（例: `training.num_epochs=1000`）。

### 5-3. チェックポイントの保存先

```
Flow_Policy/ManiFlow/data/outputs/<DATE>/<TIME>_<exp_name>_robocasa_multitask/
├── checkpoints/
│   ├── latest.ckpt
│   └── epoch=XXXX-val_loss=X.XXXXXX.ckpt
└── logs.json.txt
```

### 5-4. wandb モニタリング

`WANDB_API_KEY` 環境変数を設定すると wandb でリアルタイムに学習の進捗を確認できる:

```bash
--env WANDB_API_KEY=<YOUR_KEY>
```

ログは `https://wandb.ai/<YOUR_USERNAME>/flow_policy_robocasa/` に記録される。

---

## Step 6: 評価の実行

### 6-1. 学習済みモデルの評価

```bash
singularity exec --nv \
    --bind "/home/bilgehan.sakai:/home/bilgehan.sakai" \
    --bind "$CODEBASE_DIR:/mnt/data/bilgehan.sakai" \
    --env MUJOCO_GL=osmesa \
    --env PYOPENGL_PLATFORM=osmesa \
    --env CUDA_VISIBLE_DEVICES=0 \
    "$SIF" \
    bash -c "
        source /home/bilgehan.sakai/Flow_Policy/.venv/bin/activate
        export PYTHONPATH=/home/bilgehan.sakai/Flow_Policy/robocasa:\$PYTHONPATH
        cd /home/bilgehan.sakai/Flow_Policy/ManiFlow

        python -m maniflow.workspace.eval_maniflow_robocasa_workspace \
            --config-name maniflow_image_timm_policy_robocasa \
            task=robocasa_multitask \
            training.seed=42 \
            training.device='cuda:0' \
            exp_name=robocasa_10tasks \
            'hydra.run.dir=data/outputs/<DATE>/<TIME>_robocasa_10tasks_robocasa_multitask'
    "
```

### 6-2. 評価対象タスクの変更

`config/task/robocasa_multitask.yaml` の `env_runner.task_name` を変更するか、コマンドラインでオーバーライドする:

```bash
task.env_runner.task_name=OpenCabinet \
task.env_runner.eval_episodes=50 \
task.env_runner.obj_instance_split=target
```

### 6-3. 利用可能な ATOMIC タスク（主要）

| 新タスク名 | 旧タスク名（cosmos-policy） |
|---|---|
| `TurnOffMicrowave` | — |
| `TurnOnMicrowave` | — |
| `OpenCabinet` | `OpenSingleDoor` |
| `CloseCabinet` | `CloseSingleDoor` |
| `StartCoffeeMachine` | `CoffeePressButton` |
| `PickPlaceCounterToCabinet` | `PnPCounterToCab` |
| `PickPlaceCounterToStove` | `PnPCounterToStove` |
| `OpenDrawer` | — |
| `CloseDrawer` | — |

`obj_instance_split`:
- `"target"` = テスト用オブジェクト（評価時）
- `"pretrain"` = 学習用オブジェクト（データ収集時）

---

## トラブルシューティング

### `robocasa.__file__` が `robocasa/__init__.py` を指す

**原因:** editable インストールの名前空間パッケージ誤認識。  
**修正:** Step 3-3 パッチ 1 を再適用する。

```bash
sed -i 's/sys.meta_path.append(_EditableFinder)/sys.meta_path.insert(0, _EditableFinder)/' \
    /mnt/data/bilgehan.sakai/Flow_Policy/.venv/lib/python3.10/site-packages/__editable___robocasa_1_0_1_finder.py
```

---

### `AssertionError: numpy version must be 2.2.5`

**原因:** `robocasa/__init__.py` の NumPy バージョンアサーション。  
**修正:** Step 3-3 パッチ 2 を再適用する。

```bash
sed -i 's/    "2\.2\.5",/    "2.2.5",\n    "2.2.6",/' \
    /mnt/data/bilgehan.sakai/Flow_Policy/robocasa/robocasa/__init__.py
```

---

### `TypeError: ManipulationEnv.__init__() got an unexpected keyword argument 'load_model_on_init'`

**原因:** PyPI 版 robosuite に `load_model_on_init` が存在しない。  
**修正:** robosuite を GitHub main 版に差し替える（Step 3-2 参照）。

```bash
source /mnt/data/bilgehan.sakai/Flow_Policy/.venv/bin/activate
uv pip install 'robosuite @ git+https://github.com/ARISE-Initiative/robosuite.git'
```

---

### SIGSEGV クラッシュ（`llvmlite.binding.passmanagers`）

**原因:** numba 0.61.x が当環境の LLVM で SIGSEGV を起こす。  
**修正:** Step 3-2 の numba 固定コマンドを再実行する。

```bash
source /mnt/data/bilgehan.sakai/Flow_Policy/.venv/bin/activate
uv pip install 'numba==0.63.1' 'llvmlite==0.46.0'
```

---

### EGL レンダリング：GPU ハードウェア EGL vs osmesa ソフトウェアレンダリング

コンテナは `PYOPENGL_PLATFORM=egl` をデフォルト設定しているが、権限問題で EGL が使えない場合がある。

**GPU ハードウェア EGL（推奨・高速）**

`/dev/dri/renderD128` 等のデバイスへの読み書き権限が必要。  
ユーザーが `render` グループに所属していない場合は以下を依頼する:

```bash
sudo usermod -aG render $USER
# 再ログインして反映
```

グループ追加後は `MUJOCO_GL` / `PYOPENGL_PLATFORM` の指定不要で GPU 直接レンダリングが有効になる。

**osmesa ソフトウェアレンダリング（代替・権限不要）**

`/dev/dri` へのアクセス権がない場合は osmesa（CPU ソフトウェアレンダリング）を使用する。  
物理シミュレーションと推論は引き続き GPU で実行されるため、学習・評価自体に支障はない。

> **注意:** `MUJOCO_EGL_DEVICE_ID=3`（Mesa EGL フォールバック）は使用不可。  
> `robosuite/utils/binding_utils.py` に  
> `assert MUJOCO_EGL_DEVICE_ID in CUDA_VISIBLE_DEVICES` があり、`AssertionError` でクラッシュする。  
> osmesa を使う場合は必ず **両方の環境変数**を設定すること（片方だけでは失敗する）。

```bash
# Singularity 実行時に両方追加（どちらか一方だけでは不可）
--env MUJOCO_GL=osmesa \
--env PYOPENGL_PLATFORM=osmesa
```

**確認方法:**

```bash
find /dev/dri -name "renderD*" -readable 2>/dev/null
# 何も表示されない → render グループ未所属、osmesa を使用
# renderD128 等が表示される → GPU ハードウェア EGL 使用可能
```

---

### DataLoader がデッドロックする

**原因:** `fork` による multiprocessing + PyAV のスレッドセーフ問題。  
**修正:** `dataloader.num_workers=0` でシングルスレッドに落とす（速度は低下する）:

```bash
dataloader.num_workers=0
```

または `maniflow/config/maniflow_image_timm_policy_robocasa.yaml` の `dataloader.persistent_workers` を `false` に変更する。

---

## ファイル構成（RoboCasa 関連）

```
Flow_Policy/
├── pyproject.toml                          # uv プロジェクト定義
├── robocasa/                               # bilgik26/robocasa@dev (git clone)
├── data/
│   └── robocasa/
│       └── datasets/
│           └── v1.0/pretrain/atomic/       # ダウンロードしたデータセット
├── docs/
│   └── setup_and_train_robocasa.md         # 本ドキュメント
├── scripts/
│   ├── download_robocasa_dataset.sh        # データセットダウンロード
│   └── train_eval_robocasa.sh              # 学習・評価スクリプト
└── ManiFlow/
    └── maniflow/
        ├── dataset/
        │   └── robocasa_dataset.py         # LeRobot 形式データローダー
        ├── env/
        │   └── robocasa/
        │       └── robocasa_wrapper.py     # robosuite 環境ラッパー
        ├── env_runner/
        │   └── robocasa_runner.py          # ロールアウト実行クラス
        ├── workspace/
        │   ├── train_maniflow_robocasa_workspace.py
        │   └── eval_maniflow_robocasa_workspace.py
        └── config/
            ├── maniflow_image_timm_policy_robocasa.yaml
            └── task/
                └── robocasa_multitask.yaml

/mnt/data/bilgehan.sakai/singularity/
└── flow_policy_robocasa.def                # Singularity ビルド定義
```
