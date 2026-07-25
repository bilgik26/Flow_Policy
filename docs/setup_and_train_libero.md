# Flow_Policy × LIBERO セットアップ・学習・評価ガイド

> **対象:** `Flow_Policy` リポジトリで [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO)
> ベンチマークを使って Flow Matching Policy の学習と評価を行うための手順書。
>
> 環境構築の方法論（Singularity/Apptainer + uv、osmesa ソフトウェアレンダリング等）は
> [`docs/setup_and_train_robocasa.md`](./setup_and_train_robocasa.md) を踏襲している。
> LIBERO 固有の環境ラッパー・データセット変換・評価ループの実装は
> [`openvla-oft`](https://github.com/moojink/openvla-oft) の LIBERO 統合
> (`LIBERO.md`, `experiments/robot/libero/`) を参考にしている。
>
> **重要:** LIBERO は `robosuite==1.4.1` を要求するが、RoboCasa は `robosuite@main`
> (GitHub 最新版) を要求するため、**両者は同じ venv に共存できない**。
> 本ガイドでは RoboCasa 用の `.venv` とは別に **`.venv-libero`** を新規作成する。
> Singularity イメージ（CUDA / MuJoCo 等のシステム依存）は `flow_policy_robocasa.sif` を
> そのまま再利用してよい。
>
> **本ドキュメントは 2026-07-08 に `p-team-3` サーバー向けに修正した
> `docs/setup_and_train_robocasa.md` をベースに作成し、2026-07-22 に本サーバーで
> Step 1〜6（環境構築 → LIBERO インストール → データセット変換 → 学習 → 評価）を
> 実機で最後まで動作確認済み。** 動作確認の詳細な実行ログ・実測値は
> [Step 7: 動作確認ログ](#step-7-動作確認ログ実施済み2026-07-22) を参照。

---

## 0. Flow_Policy 側の実装 (このガイドが前提とするコード)

RoboCasa 統合（`maniflow/env/robocasa/`, `maniflow/dataset/robocasa_dataset.py`,
`maniflow/env_runner/robocasa_runner.py` 等）と全く同じ構成パターンで、以下を新規実装済み:

```
Flow_Policy/
├── libero/                                      # git clone (Lifelong-Robot-Learning/LIBERO), editable install
├── data/libero/datasets/lerobot/<suite>/<task>/  # 変換後 LeRobot 形式データセット
├── scripts/
│   ├── convert_libero_to_lerobot.py             # 生 HDF5 → LeRobot 形式 変換スクリプト
│   ├── download_libero_dataset.sh               # 生データセットDL + 変換の一括実行
│   ├── train_eval_libero.sh                     # 学習・評価スクリプト
│   └── eval_checkpoint_libero.sh                # チェックポイント評価スクリプト
└── ManiFlow/
    └── maniflow/
        ├── env/libero/
        │   ├── __init__.py
        │   └── libero_wrapper.py                # LiberoEnv（robosuite/LIBERO ラッパー）
        ├── env_runner/libero_runner.py           # LiberoRunner（評価ロールアウト）
        ├── dataset/libero_dataset.py             # LiberoImageDataset（LeRobot形式ローダー）
        ├── workspace/
        │   ├── train_maniflow_libero_workspace.py
        │   └── eval_maniflow_libero_workspace.py
        └── config/
            ├── maniflow_image_timm_policy_libero.yaml
            └── task/
                ├── libero_spatial.yaml
                ├── libero_object.yaml
                ├── libero_goal.yaml
                ├── libero_10.yaml
                ├── libero_90.yaml
                └── libero_test.yaml              # 3エピソードのスモークテスト用
```

RoboCasa と同様、**新しい policy/encoder クラスは追加していない** —
既存の汎用 `ManiFlowTransformerImagePolicy` + `TimmObsEncoder` を
`shape_meta`（後述）経由でそのまま再利用する。LIBERO 固有のコードは
env wrapper・dataset・env_runner・workspace（train/eval）・hydra config のみ。

### 0-1. 観測・アクション形式（openvla-oft を参照して決定）

| 項目 | 内容 |
|---|---|
| `image` | (3, 256, 256) float32 [0,1] — agentview カメラ、180°回転済み |
| `wrist_image` | (3, 256, 256) float32 [0,1] — 手首カメラ、180°回転済み |
| `agent_pos` | (8,) float32 = eef_pos(3) + axis_angle(eef_quat)(3) + gripper_qpos(2) |
| `action` | (7,) float32 = eef delta_pos(3) + delta_axis_angle_ori(3) + gripper(1) |

- **画像180°回転**: LIBERO/robosuite はこの種の環境では画像を上下反転して描画するため、
  `img[::-1, ::-1]` を **`libero_wrapper.py`（オンライン推論時）** と
  **`libero_dataset.py`（オフラインデータ読み込み時）** の両方で適用している
  （openvla-oft の `libero_utils.py::get_libero_image` のコメント
  "IMPORTANT: rotate 180 degrees to match train preprocessing" と同じ理由）。
  どちらか片方だけ変更すると学習と推論で画像の向きが食い違うので注意。
- **Gripper 符号**: openvla-oft は自前の RLDS データローダーが gripper の符号を反転させる
  仕様のため、推論時に `normalize_gripper_action` + `invert_gripper_action` という
  補正が必要だった。Flow_Policy の変換スクリプトは生の LIBERO アクションをそのまま
  保存するため、この反転処理は**不要**（`libero_wrapper.py` のモジュール docstring 参照）。
  gripper は一貫して `-1=open, +1=close`（robosuite/LIBERO のネイティブ規約）。
- **アクション空間**: RoboCasa の PandaOmron（移動台車付き）は 12 次元の
  composite-controller アクションにゼロパディングする必要があったが、LIBERO は
  固定ベースの Panda + OSC_POSE コントローラーのため、7 次元アクションを
  そのまま `env.step()` に渡せる（パディング不要、`libero_wrapper.py` 参照）。
- **タスクスイート別 episode 上限ステップ数**（openvla-oft `run_libero_eval.py::TASK_MAX_STEPS` から転記）:

  | task_suite_name | max steps |
  |---|---|
  | libero_spatial | 220 |
  | libero_object | 280 |
  | libero_goal | 300 |
  | libero_10 | 520 |
  | libero_90 | 400 |

- **アクションチャンクサイズ**: openvla-oft の LIBERO 標準設定 (`NUM_ACTIONS_CHUNK=8`) に
  合わせ、`n_action_steps=8`, `num_open_loop_steps=8`, `horizon=10` をデフォルトにしている
  （RoboCasa の `n_action_steps=32` より短い）。

---

## 前提条件

RoboCasa と同じサーバー前提（[setup_and_train_robocasa.md の前提条件](./setup_and_train_robocasa.md#前提条件)参照）:
NVIDIA GPU + CUDA 12.8、Apptainer 1.4.3、fakeroot 設定済み、`render` グループ未所属
（→ osmesa ソフトウェアレンダリング固定）。

追加でLIBERO固有の要件:

| 項目 | 要件 |
|---|---|
| ディスク（生データ） | 1スイート（10タスク）あたり実測 約5.9GB（`libero_spatial` で確認）。`libero_spatial`+`libero_object`+`libero_goal` で約15〜20GB、`libero_100`（libero_10+libero_90）はさらに大きい |
| ディスク（チェックポイント） | 1 checkpoint（EMA込み・model+optimizer state 全体）で実測 約3.2GB。`checkpoint.topk.k=3` + `save_last_ckpt=true` により学習中は常時 4個前後（約13GB）が保持される |
| venv | RoboCasa 用 `.venv` とは別に `.venv-libero` を新規作成（robosuite バージョン競合のため） |

---

## Step 1: Singularity イメージ

RoboCasa 用に構築済みの `flow_policy_robocasa.sif` をそのまま利用する
（CUDA / MuJoCo など、システムレベルの依存は共通のため、LIBERO 専用イメージを
新規ビルドする必要はない）。未構築の場合は
[setup_and_train_robocasa.md の Step 1](./setup_and_train_robocasa.md#step-1-singularity-イメージのビルド)
を参照してビルドする。

---

## Step 2: LIBERO 用 Python 仮想環境のセットアップ (uv)

RoboCasa 用 `.venv` とは**別の venv** を、`UV_PROJECT_ENVIRONMENT` で明示的に切り替えて作成する。

```bash
export CODEBASE_DIR="/home/bilgehan.sakai"
SIF="$CODEBASE_DIR/singularity/sif/flow_policy_robocasa.sif"

singularity exec --nv \
    --bind "$CODEBASE_DIR:/home/bilgehan.sakai" \
    --bind "$HOME/.cache:/root/.cache" \
    "$SIF" \
    bash -c "
        cd /home/bilgehan.sakai/Flow_Policy
        UV_PROJECT_ENVIRONMENT=.venv-libero uv sync --extra cu128 --python 3.10
    "
```

これで RoboCasa と同じベース依存（torch, hydra-core, timm, diffusers 等）を持つ
`.venv-libero` が作成される。

---

## Step 3: LIBERO のインストール

### 3-1. リポジトリのクローン（ホスト上）

```bash
cd /home/bilgehan.sakai/Flow_Policy
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git libero
```

### 3-2. コンテナ内で依存パッケージをインストール

```bash
export CODEBASE_DIR="/home/bilgehan.sakai"
SIF="$CODEBASE_DIR/singularity/sif/flow_policy_robocasa.sif"

singularity exec --nv \
    --bind "$CODEBASE_DIR:/home/bilgehan.sakai" \
    --bind "$HOME/.cache:/root/.cache" \
    "$SIF" \
    bash -c "
        source /home/bilgehan.sakai/Flow_Policy/.venv-libero/bin/activate

        # LIBERO が要求する固定バージョンの robosuite
        # (RoboCasa の robosuite@main とは非互換 — 別 venv にする理由)
        uv pip install 'robosuite==1.4.1'

        # LIBERO 本体 (editable、'compat' モード必須。理由は下記パッチ1参照)
        uv pip install --config-settings editable_mode=compat \
            -e /home/bilgehan.sakai/Flow_Policy/libero

        # LIBERO の追加依存 (openvla-oft の libero_requirements.txt を参照)
        uv pip install bddl easydict cloudpickle gym 'imageio[ffmpeg]'

        # 高速な動画ランダムアクセスデコード (maniflow/dataset/libero_dataset.py が使用。
        # PyAV は動画の先頭から順にデコードしないと任意フレームへシークできないため、
        # エピソード終盤のフレームを読むたびにエピソード全体を再デコードしていた。
        # decord はキーフレーム索引を使った真のランダムアクセスができ、
        # 実測で __getitem__ 1回あたり約1.3倍、学習の1stepあたりも同程度高速化する)
        uv pip install decord

        # r3m (obs_encoder の model_name='r3m' が使用。third_party/r3m は
        # Flow_Policy リポジトリに同梱されている RoboCasa 用の第三者コード)
        uv pip install -e /home/bilgehan.sakai/Flow_Policy/third_party/r3m

        # diffusers==0.27.2 が使う huggingface_hub.cached_download は
        # huggingface_hub>=0.26 で削除されているため、RoboCasa の .venv と
        # 同じバージョンにピン留めする（パッチ2参照）
        uv pip install 'huggingface_hub==0.25.2'

        # Flow_Policy 自身をインストール (editable)
        uv pip install -e /home/bilgehan.sakai/Flow_Policy/ManiFlow

        # データセット変換スクリプトが使う h5py
        uv pip install h5py
    "
```

### 3-3. LIBERO 設定ファイルの事前作成（インタラクティブプロンプト回避）

**問題:** `libero.libero` を初めて import すると、`~/.libero/config.yaml` が
存在しない場合に **標準入力からの `input()` 呼び出し**（データセット保存先を
カスタマイズするかどうかの Y/N 質問）が実行される。非対話的なスクリプト実行では
これがハングまたは `EOFError` の原因になる。

**修正:** どのコマンドよりも先に、ホスト上で config.yaml を直接作成しておく
（`LIBERO_CONFIG_PATH` 環境変数が未設定の場合、常に `$HOME/.libero/config.yaml`
が読まれる。本サーバーの `$HOME` は `/home/bilgehan.sakai`）。

```bash
LIBERO_PKG="/home/bilgehan.sakai/Flow_Policy/libero/libero/libero"
mkdir -p ~/.libero
cat > ~/.libero/config.yaml <<EOF
benchmark_root: ${LIBERO_PKG}
bddl_files: ${LIBERO_PKG}/bddl_files
init_states: ${LIBERO_PKG}/init_files
datasets: ${LIBERO_PKG}/datasets
assets: ${LIBERO_PKG}/assets
EOF
```

> **パス構造の注意:** LIBERO のクローンルート（`git clone` した先、Step 3-1 の
> `libero/`）の中に、さらに Python パッケージ本体の `libero/` サブディレクトリがあり、
> その中にもう一段 `libero/`（`libero.libero` サブパッケージ、`get_libero_path` 等が
> 定義されている場所）がある — つまり **`libero/libero/libero/` で3段** になる。
> `datasets` のデフォルトパスもこの3段目基準（`.../libero/libero/libero/datasets`）。
> `scripts/download_libero_dataset.sh` の `RAW_BASE` もこのパスと一致させてある。

### 3-4. パッチの適用（ホスト上）

#### パッチ 1: LIBERO editable インストールの namespace パッケージ問題

**問題:** LIBERO の `setup.py` は `find_packages()` でパッケージを収集するが、
トップレベルの `libero/` ディレクトリ自体に `__init__.py` が無い（`libero.libero` /
`libero.lifelong` など各サブパッケージにはある）。この構造だと `find_packages()` は
**何も見つけられず**、`uv pip install -e .`（PEP 660 の import-hook 方式）で
インストールすると `MAPPING`/`NAMESPACES` が空のエディタブルファインダーが
生成され、`from libero.libero import benchmark` が
`ImportError: cannot import name 'benchmark' from 'libero.libero' (unknown location)`
で失敗する（RoboCasa の namespace パッケージ誤認識バグとは別種の問題で、
`sys.meta_path` の順序ではなく **マッピングが空** なのが原因）。

**修正:** Step 3-2 の通り `--config-settings editable_mode=compat` を付けて
インストールする。これは setuptools のレガシー互換エディタブルモードで、
単に `libero/` のクローンルートを sys.path に追加する `.pth` ファイルを
生成するだけになり、Python 標準の暗黙的 namespace パッケージ解決
（PEP 420）に委ねられるため正しく動作する。

適用済みか確認する方法:
```bash
python -c "import libero.libero as l; print(l.__file__)"
# .../libero/libero/libero/__init__.py が表示されれば正常
# None が表示される場合は namespace パッケージのまま → 上記コマンドで再インストール
```

#### パッチ 2: LIBERO 本体への `torch.load(weights_only=False)` パッチ

**問題:** `libero/libero/benchmark/__init__.py` の `Benchmark.get_task_init_states()` が
`torch.load(init_states_path)` を素の設定で呼んでいるが、PyTorch 2.6 以降
`torch.load` のデフォルトが `weights_only=True` に変わり、pickle 化された
numpy 配列（`.pruned_init` ファイル）を読み込めず
`_pickle.UnpicklingError: Weights only load failed ... numpy.core.multiarray._reconstruct`
で失敗する（本環境は torch==2.7.0）。評価時に `LiberoEnv.__init__` →
`task_suite.get_task_init_states(task_id)` を呼んだ際に発生する。

**修正:**
```bash
sed -i 's/init_states = torch.load(init_states_path)/init_states = torch.load(init_states_path, weights_only=False)/' \
    /home/bilgehan.sakai/Flow_Policy/libero/libero/libero/benchmark/__init__.py
```

### 3-5. 動作確認

```bash
singularity exec --nv \
    --bind "$CODEBASE_DIR:/home/bilgehan.sakai" \
    --bind "$HOME/.cache:/root/.cache" \
    --env MUJOCO_GL=osmesa \
    --env PYOPENGL_PLATFORM=osmesa \
    "$SIF" \
    bash -c "
        source /home/bilgehan.sakai/Flow_Policy/.venv-libero/bin/activate
        python -c '
import torch, robosuite
from libero.libero import benchmark
print(\"torch:\", torch.__version__, \"cuda:\", torch.cuda.is_available())
print(\"robosuite:\", robosuite.__version__)
bd = benchmark.get_benchmark_dict()
suite = bd[\"libero_spatial\"]()
print(\"libero_spatial n_tasks:\", suite.n_tasks)
task0 = suite.get_task(0)
print(\"task 0 name:\", task0.name)
print(\"task 0 language:\", task0.language)
'
    "
```

**実測出力（2026-07-22、本サーバーで実行確認済み）:**
```
torch: 2.7.0+cu128 cuda: True
robosuite: 1.4.1
libero_spatial n_tasks: 10
task 0 name: pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate
task 0 language: pick up the black bowl between the plate and the ramekin and place it on the plate
```

`libero.libero.benchmark` の実際の API（`libero/libero/benchmark/__init__.py` で確認済み）:
- `Task` は `NamedTuple(name, language, problem, problem_folder, bddl_file, init_states_file)`。
- `Benchmark.n_tasks`（プロパティではなく通常の int 属性）、`.get_task(i)`、
  `.get_task_init_states(i)`（`torch.load` でロード、パッチ2適用後は正常動作）、
  `.get_task_demonstration(i)` → `f"{problem_folder}/{name}_demo.hdf5"`（ダウンロードディレクトリ相対）。
- `libero.libero.envs.env_wrapper.ControlEnv` に `get_observation()` は **存在しない**。
  `reset()` 自体が observation dict を返す（内部で robosuite の `env.reset()` を呼ぶだけ）。
  `LiberoEnv.reset()` は必ず `self._env.reset()` を呼んでから、`init_state_idx` が
  指定されていれば `set_init_state()` の返り値で上書きする実装になっている
  （`maniflow/env/libero/libero_wrapper.py` 参照）。
- `Gym has been unmaintained since 2022...` という警告は無害（LIBERO が内部で
  `import gym`（0.26.2）しているだけで、Flow_Policy 側のコードには影響しない）。
- `[robosuite WARNING] No private macro file found!` も無害（RoboCasa と同じ、
  未設定でもデフォルト値で動作する）。

---

## Step 4: 学習用データセットのダウンロード・変換

LIBERO の生データセットは HDF5 形式（低解像度 128×128、手首カメラの解像度も低い）で
配布されているため、Flow_Policy では **シミュレータでリプレイして再描画**し、
RoboCasa と同じ LeRobot v2.1 形式（parquet + mp4）に変換してから学習に使う
（この手法は openvla-oft の `experiments/robot/libero/regenerate_libero_dataset.py` と
同じ考え方 — 生デモの no-op アクション除去、リプレイ失敗デモの除外を含む）。

### 4-1. 一括ダウンロード + 変換

```bash
singularity exec --nv \
    --bind "$CODEBASE_DIR:/home/bilgehan.sakai" \
    --bind "$HOME/.cache:/root/.cache" \
    --env MUJOCO_GL=osmesa \
    --env PYOPENGL_PLATFORM=osmesa \
    "$SIF" \
    bash -c "
        source /home/bilgehan.sakai/Flow_Policy/.venv-libero/bin/activate
        cd /home/bilgehan.sakai/Flow_Policy
        bash scripts/download_libero_dataset.sh libero_spatial libero_object libero_goal
    "
```

引数なしで実行すると、デフォルトで `libero_spatial libero_object libero_goal`
の3スイート（各10タスク）をダウンロード・変換する。

**`libero_10` / `libero_90` について（訂正: 2026-07-25 に確認、個別ダウンロード可能）:**
LIBERO の公式ダウンロードスクリプト `benchmark_scripts/download_libero_datasets.py`
の CLI は `--datasets` に `{all, libero_goal, libero_spatial, libero_object, libero_100}`
しか受け付けず、`libero_10`/`libero_90` は選択肢に無い。しかし実際に
Hugging Face リポジトリ（`yifengzhu-hf/LIBERO-datasets`）の中身を
`HfApi.list_repo_files` で確認したところ、`libero_100` というフォルダは
**存在せず**、`libero_10/`・`libero_90/` が最初から独立したトップレベル
フォルダとして存在している（`libero_goal`/`libero_object`/`libero_spatial`
と同列）。つまり CLI の choices が古い/box.com 時代の仕様を引きずっている
だけで、**`libero_10` だけを個別にダウンロードすることは実際には可能**
（`libero_100` 全体＝10+90=100 タスク分をダウンロードする必要はない）。

具体的には、CLI を経由せず `download_utils.download_from_huggingface()` を
直接呼び出せばよい:

```python
import libero.libero.utils.download_utils as download_utils
download_utils.download_from_huggingface(
    dataset_name="libero_10",
    download_dir="libero/libero/libero/datasets",
    check_overwrite=False,
)
```

実測（2026-07-25、本サーバー）: `libero_10` の10ファイルのみで約2分半・
合計13GB（`libero_100` 全体をダウンロードする場合の1/9程度で済む）。
`download_libero_dataset.sh` は現状この直接呼び出しに対応していない
（`libero_10`/`libero_90` は依然として `libero_100` にマッピングされる
実装のまま）ため、`libero_10` だけが必要な場合は上記のワンライナーを
手動で使うほうが速い。

**確認済みの実際の実行結果（2026-07-22、`libero_spatial` のみ）:**
```
Datasets downloaded to /home/bilgehan.sakai/Flow_Policy/libero/libero/libero/datasets
Downloading libero_spatial datasets
Using Hugging Face as the download source
Fetching 10 files: 100%|██████████| 10/10 [01:10<00:00,  7.01s/it]
Downloaded 10 files for libero_spatial
[X] Dataset libero_spatial is complete
```
10 タスク分の `*_demo.hdf5` がダウンロードされ、合計約 5.9GB
（1ファイルあたり約500MB〜750MB、各50デモ、生の観測は128×128解像度）。

> **`--use-huggingface` は必須。** これを付けないと元の box.com リンクを使うか
> 確認する対話式プロンプトが標準入力に出て非対話実行がハングする
> （リンク自体も失効している可能性がある）。`download_libero_dataset.sh` は
> 常にこのフラグを付けて呼び出す。

### 4-2. 変換のみ個別実行する場合

```bash
python scripts/convert_libero_to_lerobot.py \
    --libero_task_suite libero_spatial \
    --libero_raw_data_dir /home/bilgehan.sakai/Flow_Policy/libero/libero/libero/datasets/libero_spatial \
    --libero_target_dir  /home/bilgehan.sakai/Flow_Policy/data/libero/datasets/lerobot/libero_spatial \
    --task_ids 0                    # 省略時は全タスク（このスイートでは10個）
    --max_demos_per_task 20          # 省略時は各タスクの全デモ（このスイートでは50個）
```

**変換速度の実測値（本サーバー、osmesa ソフトウェアレンダリング、256×256×2カメラ）:**
1 エピソードあたり約 30〜60 秒（デモの長さに依存。リプレイ中は
MuJoCo の物理ステップ + 2カメラのソフトウェアレンダリングを毎ステップ実行するため）。
`--task_ids 0 --max_demos_per_task 20` で 18/20 デモが成功リプレイに終わり
（2件は物理エンジンのバージョン差異等でリプレイが `done=True` に到達せず除外）、
所要時間は約 9〜10 分だった。**全タスク・全デモ（10タスク×50デモ）を変換する場合は
数時間規模になる見込み**なので、大規模実行時はバックグラウンド実行を推奨する。

変換後、以下のようなディレクトリ構造が生成される（1タスク1ディレクトリ。
`maniflow/dataset/libero_dataset.py` の `_discover_lerobot_dirs` が自動検出する）:

```
data/libero/datasets/lerobot/libero_spatial/
├── <task_name_1>/
│   ├── meta/{info.json, tasks.jsonl, episodes.jsonl}
│   ├── data/chunk-000/episode_000000.parquet ...
│   └── videos/chunk-000/observation.images.{image,wrist_image}/episode_000000.mp4 ...
├── <task_name_2>/
│   └── ...
└── ...（スイート内の全タスク分）
```

### 4-3. config のデータパス確認

`ManiFlow/maniflow/config/task/libero_spatial.yaml`（他スイートも同様）の
`dataset_base_path` が変換先パスと一致していることを確認する。

別のパスを使う場合は hydra オーバーライドで指定できる:

```bash
task.dataset_base_path=/absolute/path/to/lerobot/libero_spatial
```

---

## Step 5: 学習の実行

### 5-1. 学習コマンド（1-GPU、libero_spatial の例）

```bash
singularity exec --nv \
    --bind "/home/bilgehan.sakai:/home/bilgehan.sakai" \
    --env MUJOCO_GL=osmesa \
    --env PYOPENGL_PLATFORM=osmesa \
    --env CUDA_VISIBLE_DEVICES=0 \
    "$SIF" \
    bash -c "
        source /home/bilgehan.sakai/Flow_Policy/.venv-libero/bin/activate
        cd /home/bilgehan.sakai/Flow_Policy/ManiFlow

        python -m maniflow.workspace.train_maniflow_libero_workspace \
            --config-name maniflow_image_timm_policy_libero \
            task=libero_spatial \
            training.seed=42 \
            training.device='cuda:0' \
            exp_name=libero_spatial_run
    "
```

他のタスクスイートで学習する場合は `task=libero_object` / `task=libero_goal` /
`task=libero_10` / `task=libero_90` に差し替える。

または付属のスクリプトを使用する:

```bash
singularity exec --nv \
    --bind "/home/bilgehan.sakai:/home/bilgehan.sakai" \
    --env MUJOCO_GL=osmesa \
    --env PYOPENGL_PLATFORM=osmesa \
    "$SIF" \
    bash /home/bilgehan.sakai/Flow_Policy/scripts/train_eval_libero.sh 0 42 libero_spatial_run train libero_spatial
```

スモークテスト（3エピソードのみ、短い上限ステップ数）で配線を確認したい場合:

```bash
bash scripts/train_eval_libero.sh 0 0 smoke_test train libero_test
```

### 5-1b. マルチGPU学習（DDP、global batch size を大きくしたい場合）

`train_maniflow_libero_workspace.py` は `torch.nn.parallel.DistributedDataParallel`
(DDP) による複数GPU学習に対応している。`training.num_gpus`（デフォルト `1`）に
GPU数を指定すると、`torch.multiprocessing.spawn` で GPU数と同じ数のプロセスを
起動し、各プロセスが1GPUを占有して勾配を all-reduce で同期する。

**`dataloader.batch_size` は GPU 1枚あたりのバッチサイズ**として扱われる
（`DistributedSampler` でデータセットを GPU 数に分割するため）。したがって

```
global batch size = dataloader.batch_size × training.num_gpus
```

となり、GPU を増やすほど実効的な batch size が大きくなる
（学習率を保つ場合は `optimizer.lr` を batch size に応じて調整することを推奨）。

付属スクリプトを使う場合、GPU 引数にカンマ区切りで GPU番号を渡すと自動的に
マルチGPU学習になる:

```bash
# 4GPU (物理GPU 0,1,2,3) で学習、per-GPU batch_size=128 なら global batch size=512
singularity exec --nv \
    --bind "/home/bilgehan.sakai:/home/bilgehan.sakai" \
    --env MUJOCO_GL=osmesa \
    --env PYOPENGL_PLATFORM=osmesa \
    "$SIF" \
    bash /home/bilgehan.sakai/Flow_Policy/scripts/train_eval_libero.sh 0,1,2,3 42 libero_spatial_4gpu train libero_spatial
```

Hydra を直接呼ぶ場合は `training.num_gpus=<N>` を渡す（`CUDA_VISIBLE_DEVICES` で
使用するGPUを絞り込んでおくこと。`training.device` はマルチGPU時は無視され、
各プロセスが `cuda:<local_rank>`（`CUDA_VISIBLE_DEVICES` 内でのインデックス）を
自動的に使う）:

```bash
singularity exec --nv \
    --bind "/home/bilgehan.sakai:/home/bilgehan.sakai" \
    --env MUJOCO_GL=osmesa \
    --env PYOPENGL_PLATFORM=osmesa \
    --env CUDA_VISIBLE_DEVICES=0,1,2,3 \
    "$SIF" \
    bash -c "
        source /home/bilgehan.sakai/Flow_Policy/.venv-libero/bin/activate
        cd /home/bilgehan.sakai/Flow_Policy/ManiFlow

        python -m maniflow.workspace.train_maniflow_libero_workspace \
            --config-name maniflow_image_timm_policy_libero \
            task=libero_spatial \
            training.seed=42 \
            training.num_gpus=4 \
            dataloader.batch_size=128 \
            exp_name=libero_spatial_4gpu
    "
```

**設計上のポイント（実装の詳細）:**
- 環境ロールアウト（osmesa レンダリング）・validation・wandb ログ・チェックポイント保存は
  rank 0（プロセス0）のみが行う。他の GPU は学習の forward/backward のみを担当し、
  各エポック末尾で `dist.barrier()` により rank 0 の処理完了を待ってから次エポックへ進む。
- チェックポイントの `state_dict` は常に DDP でラップされていない生の
  `ManiFlowTransformerImagePolicy` から保存されるため、単一GPUで保存した
  checkpoint をマルチGPUで resume する（あるいはその逆）ことができる
  （`module.` プレフィックスの有無を気にする必要がない）。
- 1バッチの loss が NaN/Inf になった場合のスキップ判定は全 rank 間で
  `dist.all_reduce`（MAX）を取って同期している。DDP は全 rank が同じ回数
  `backward()` を呼ぶことを要求するため、1 rank だけが NaN でスキップすると
  残りの rank が allreduce 待ちでハングしてしまう — これを避けるための処理。
- `training.ddp_find_unused_parameters`（デフォルト `true`）: `ManiFlowTransformerImagePolicy`
  は基底クラス `ModuleAttrMixin` の `_dummy_variable`（device/dtype 問い合わせ用の
  ダミー `nn.Parameter` で forward には一切使われない）を常に持つため、
  DDP は常に「使われないパラメータ」を検出する。`find_unused_parameters=false`
  にすると `RuntimeError: Expected to have finished reduction in the prior
  iteration...` で2ステップ目以降に失敗するため、基本的に `true` のまま使うこと。

### 5-1c. `dataloader.batch_size` / `num_workers` の実測上限（4GPU、本サーバー）

本サーバー（GPU 5枚、システム RAM 45GB + swap 8GB）で `libero_spatial`
フルデータセット・4GPU DDP・`dataloader.num_workers=1` を固定して実測した結果:

| per-GPU `batch_size` | global batch size | 結果 |
|---|---|---|
| 128 | 512 | ✅ 安全（RSS 33〜39GB で安定・推奨） |
| 160 | 640 | △ RSS が45.9GBまで到達する危険域（生き残ったが余裕なし） |
| 192以上 | 768以上 | ❌ カーネルの OOM Killer に強制終了される |

**GPUメモリ（1枚49GB）ではなく、4プロセス分のモデル/データセットの固定オーバーヘッド
＋ dataloader のプリフェッチバッファがホスト RAM を圧迫するのがボトルネック**。
`batch_size=128`（global 512）を安全な上限として推奨する。

`num_workers` は **1 を超えると危険**。4GPU DDP では `num_workers × 4` 個の
dataloader ワーカープロセスが同時に立つため、`num_workers=2`（合計8プロセス）に
しただけで `batch_size=96`（`num_workers=1`なら余裕で安全な設定）でもswapが
8GB上限まで完全に埋まり、数十秒〜数分でステップ進行が完全停止する
（原因不明のCPUメモリ増加のように見えるが、実体はswap枯渇によるI/Oスラッシング）。
`num_workers=1` を維持すること。

### 5-1d. decord による動画デコードの高速化（適用済み）

`maniflow/dataset/libero_dataset.py` の動画フレームデコードは PyAV から
[decord](https://github.com/dmlc/decord) に変更済み。PyAV の `container.decode()`
は先頭フレームから順にしかデコードできないため、エピソード終盤のフレームを
1つ読むだけでもエピソード全体を毎回再デコードしていた。decord は
キーフレーム索引を使った真のランダムアクセス（`VideoReader.get_batch()`）が
でき、さらに直近に開いた動画の `VideoReader` を worker プロセスごとに
LRU キャッシュ（最大16件）して再オープンのコストも削減している。

本サーバーでの実測（`libero_spatial`、ランダムアクセス50サンプル平均）:

| | 1サンプルあたり | 4GPU学習の1step（batch_size=128） |
|---|---|---|
| PyAV（旧） | 151ms | 約22秒/step |
| decord（新） | 117ms | **約17秒/step**（約1.3倍高速） |

91 step/epoch換算で、1epochの所要時間は約33分→**約26分**に短縮される。
`decord` は Step 3-2 の手順（`uv pip install decord`）で `.venv-libero` に
インストール済みであること。

### 5-1e. 既存データセットの GOP 短縮による追加高速化（オプション）

`scripts/convert_libero_to_lerobot.py --gop_size <N>` で、mp4 保存時のキーフレーム間隔
（デフォルトは実質1エピソードに1枚）を短くできる。LIBERO シミュレータを再実行せず、
**既存の変換済み mp4 だけを読み込んで再エンコードする** `scripts/reencode_libero_videos.py`
も用意してあり、`libero_spatial` 全体（451エピソード・902動画）で実測 **約5分**
（シミュレータ再生ありのフル変換=約5.5時間とは別物）で完了する:

```bash
python scripts/reencode_libero_videos.py \
    --source_dir data/libero/datasets/lerobot/libero_spatial \
    --target_dir data/libero/datasets/lerobot/libero_spatial_gop10 \
    --gop_size 10
```

`meta/`・`data/`（parquetのaction/state）はそのままコピーし、動画のみ再エンコードする
（action/stateはdecord読み込み結果と完全一致することを確認済み）。

**実測（`libero_spatial`全体、ランダムアクセス）:**

| | 1sample | 4GPU学習の1step（batch_size=128） |
|---|---|---|
| decord + 既存GOP | 118.7 ms | 約17秒 |
| decord + `gop_size=10`再エンコード | **72.9 ms**（1.63倍） | **約11.5秒** |

再エンコードで画質はごくわずかに劣化する（2回目の非可逆圧縮のため）が、
平均絶対差は0.006（256階調で約1.6）、diff>0.1の画素は全体の0.01〜0.02%のみで、
学習で使う`ColorJitter`等のaugmentationより遥かに小さく実用上無視できる。

有効にするには `task.dataset_base_path=data/libero/datasets/lerobot/libero_spatial_gop10`
をオーバーライドするか、`task/libero_spatial.yaml`の`dataset_base_path`をそちらに変更する。

### 5-1f. `wrist_image`（手首カメラ）を無効化してさらに高速化（オプション）

`task.use_wrist_image=false` を渡すと、`wrist_image`（手首カメラ）を
データセットのデコード対象からも policy の `shape_meta`（=obs_encoder の入力）
からも完全に除外する。両カメラのデコード時間はほぼ半々なので、データロードは
**約2倍**（73ms→35.5ms/sample、`gop_size=10`再エンコード済みデータの場合）
高速化する。

```bash
python -m maniflow.workspace.train_maniflow_libero_workspace \
    --config-name maniflow_image_timm_policy_libero \
    task=libero_spatial \
    task.use_wrist_image=false \
    ...
```

**重要な注意点:**
- これは純粋な速度最適化ではなく **policy への入力そのものを変える変更**。
  手首カメラは接近時の視野やオクルージョン耐性など、俯瞰カメラだけでは
  得られない情報を提供しているため、**タスク成功率が下がるリスクがある**。
  openvla-oft や LIBERO 標準設定でも通常2カメラ構成が使われている。
- `use_wrist_image=false` で学習したチェックポイントは、
  `obs_encoder`（ひいては全体のパラメータ数、実測 205.3M→193.4M）が
  構造的に変わるため、`use_wrist_image=true`（デフォルト）のチェックポイントと
  **互換性がない**（stateダイクトの形が一致しない）。学習・評価で必ず同じ値を使うこと。
- `task/*.yaml`（`libero_spatial`等）の`use_wrist_image`（デフォルト `true`）から
  `task.dataset.use_wrist_image` と `task.shape_meta.obs.wrist_image` の
  除去（`train_maniflow_libero_workspace.py::_apply_use_wrist_image`、
  `TrainManiFlowLiberoWorkspace.__init__` の最初で policy 構築前に実行）の
  両方に自動的に反映される — 手動で shape_meta を編集する必要はない。

### 5-1g. NVIDIA DALI（ハードウェアNVDEC）による動画デコードのさらなる高速化（オプション、新規追加）

`dataloader.use_dali=true` を渡すと、decord（CPU デコード）の代わりに
[NVIDIA DALI](https://github.com/NVIDIA/DALI) の `fn.readers.video`（ハードウェア
NVDEC）で動画を GPU 上に直接デコードする。既存の decord パスは変更しておらず、
この新しい `maniflow/dataset/libero_dataset_dali.py`（`LiberoDaliLoader`）を
config で選べるようにしただけ。

**実測（`libero_spatial_gop10`、単一ストリーム、256x256、horizon=16）:**

| | 1sample あたり（デコードのみ） |
|---|---|
| decord（warm cache） | 19.8 ms |
| DALI（GPU NVDEC） | **3.3 ms**（約6倍） |

境界サンプル（エピソード先頭/末尾でフレームを複製パディングする箇所）を含め
`train_maniflow_libero_workspace.py` 経由のフル学習ループ（`compute_loss`・
validation を含む）で10エポックの動作確認済み。

```bash
# インストール（別 extra、~400MB。NVDEC 対応 NVIDIA GPU が必要）
uv pip install nvidia-dali-cuda120   # or: uv sync --extra dali

python -m maniflow.workspace.train_maniflow_libero_workspace \
    --config-name maniflow_image_timm_policy_libero \
    task=libero_spatial \
    dataloader.use_dali=true \
    val_dataloader.use_dali=true \
    ...
```

**アーキテクチャ上の注意点:**
- `LiberoDaliLoader` は `LiberoImageDataset` を**置き換えない**。エピソード読み込み・
  train/val split・normalizer・action/state はそのまま `LiberoImageDataset` の
  ロジックを再利用し、画像デコードだけを DALI に差し替えている。
- DALI の `fn.readers.video` は PyTorch の `Dataset.__getitem__` のような
  ランダムアクセスではなく、`file_list`（動画パス + 開始フレームの一覧）を
  順番に読む「reader」方式。そのため `LiberoDaliLoader` は
  `torch.utils.data.DataLoader` を使わず、独自の `__iter__`/`set_epoch` を実装し、
  DDP のシャーディングも（`DistributedSampler` ではなく）ランク毎に
  同じシードで決定的にシャッフル+分割することで実現している
  （全ランクが必ず同じバッチ数になるよう保証 — `dist.all_reduce` が
  バッチ毎に呼ばれるため、ランク間でバッチ数がずれるとハングする）。
- `num_workers`（CPU ワーカープロセス）は不要 — DALI はデコードを GPU の
  NVDEC で行うため、`num_workers>1` で問題になっていたホストRAM/swap
  枯渇の懸念がそもそも発生しない。`dataloader.dali_num_threads`（デフォルト4）
  は DALI 自身のファイルI/O用スレッド数で、デコードとは無関係。
- pad_before/pad_after（エピソード境界でのフレーム複製）は DALI の
  `file_list` では直接表現できない（負の `start_frame` は「末尾から
  N番目」の意味になるため）ため、実在するフレーム範囲のみ DALI で
  読み、複製パディングは読み込み後に再構成している
  （`LiberoDaliLoader._pad_and_layout`）。
- NVDEC のハードウェアデコードは decord（ffmpeg ソフトウェアデコード）と
  YUV→RGB 変換が若干異なり、画素値に小さな差が出る（平均絶対差 約1.7/255、
  最大 約13/255）。学習に影響する範囲ではないが、`use_dali=true`/`false`
  それぞれで学習したモデルはビット単位では一致しない。
- 現状 `drop_last=True` 固定（最後の中途半端なバッチは毎エポック捨てる、
  最大 `batch_size-1` サンプル）。

### 5-1h. torch.compile によるモデル計算の高速化（検証結果: 現状は非推奨）

`training.compile=true` で DiTX transformer（policy の計算の89%を占める、
1.83億パラメータ）の forward を `torch.compile` する仕組みを追加した。
EMA（`ema_model`）は毎ステップ `ema_model.model` の forward を consistency
training のターゲット計算に使うため、こちらも同様にコンパイルする。

**重要な実装上の注意（EMA破壊バグの回避）:** `policy.model = torch.compile(policy.model)`
のように**モジュール属性を丸ごと差し替えるとEMAが壊れる**。
`EMAModel.step()`（`maniflow/model/diffusion/ema_model.py`）は
`zip(new_model.modules(), self.averaged_model.modules())` で両モデルの
モジュールツリーを**位置ベースで**ペアリングしてパラメータを平均するため、
片方だけ `torch.compile` の `OptimizedModule` ラッパーが挿入されてツリーの
形が変わると、ペアリングがずれて誤ったパラメータ同士が平均されてしまう
（サイレントに破損する、テストで気づきにくいバグ）。そのため
`train_maniflow_libero_workspace.py::_compile_model` は
`policy.model.forward = torch.compile(policy.model.forward)` と
**`forward`属性だけ**を差し替え、モジュールツリー自体（state_dict・EMA
ペアリング・チェックポイント互換性）には一切手を触れない実装にしている。

**実測結果（`libero_spatial`、batch_size=128、`compute_loss`のforward+backward、
`torch.compile`前に見つけた別バグ修正後 — 下記参照）:**

| | steady-state | 備考 |
|---|---|---|
| eager（コンパイルなし） | 772 ms/step | |
| DiTX のみ compile | 753 ms/step（**1.02倍**） | ほぼ誤差範囲、実用上の恩恵なし |
| DiTX + obs_encoder 両方 compile | 8564 ms/step（**0.09倍・11倍遅い**） | 毎ステップ再コンパイルが発生（後述） |

さらに、実際の学習ループ（DDPなし・1GPU・`compute_loss`を`train_maniflow_libero_workspace.py`
経由で複数ステップ実行）で試したところ、DiTXのみのcompileでも
**2ステップ目でクラッシュした**（`torch._inductor`の
`assert_size_stride`アサーションエラー、コンパイル済みbackwardグラフ内の
動的形状/ストライドの不整合）。単発の孤立ベンチマークでは再現しなかった
問題なので、実際の学習ループ固有の何か（バッチ構成やデータ依存の分岐）が
引き金になっている可能性がある。

**結論: 現状の構成では `torch.compile` は推奨しない**（デフォルト
`training.compile=false`のまま）。理由:
1. DiTX単体は速度向上がほぼゼロ（1.02倍）— transformerの計算はすでに
   cuBLASの行列積カーネルでほぼ最適化されており、`torch.compile`の
   カーネル融合が入り込む余地が小さいと考えられる。
2. `obs_encoder`（R3M/ResNet18、`self.training`によるaugmentation分岐や
   `img.max() > 1.0`のデータ依存分岐を含む）を追加でcompileすると、
   ガード失敗による**毎ステップ再コンパイル**が発生し、11倍以上遅くなる。
3. 実際の学習ループでは2ステップ目でクラッシュする、既知の不具合がある
   （原因未特定、pytorch/inductor側の制約の可能性）。

`training.compile=true` / `compile_mode`（`default`/`reduce-overhead`/
`max-autotune`）は将来のPyTorch/inductorバージョンで改善された場合に
再検証できるよう設定としては残してあるが、実運用の学習では使わないこと。

**副産物として見つかった別の問題（ベンチマークスクリプト側のバグ、
本番コードには影響なし）:** 最初にこの比較を行った際、`compute_loss`が
772msではなく2760ms/stepと表示され、`torch.profiler`で調べたところ
画像バッチ全体がGPU→CPU→GPUを毎ステップ往復していた
（`normalizer.py`の`_normalize`が`x = x.to(device=scale.device, ...)`で
normalizerのバッファ側にデータを合わせる実装のため、normalizerの
scale/offsetバッファがCPUに残っていると発生する）。原因は
筆者のベンチマークスクリプトが`model.set_normalizer(normalizer)`を
`model.to(device)`より**後**に呼んでいたこと
（`train_maniflow_libero_workspace.py::run()`本体は正しく
`set_normalizer`→`to(device)`の順で呼んでおり、本番コードにこの問題はない）。
`set_normalizer`は必ず`.to(device)`より前に呼ぶこと。

### 5-2. 主要なハイパーパラメータ

| パラメータ | デフォルト値 | 説明 |
|---|---|---|
| `horizon` | 10 | ウィンドウ幅 (n_obs_steps + n_action_steps) |
| `n_obs_steps` | 2 | 観測履歴フレーム数 |
| `n_action_steps` | 8 | 予測アクションチャンクサイズ（openvla-oft の LIBERO 標準に合わせた値） |
| `training.num_epochs` | 501 | 学習エポック数 |
| `dataloader.batch_size` | 128 | バッチサイズ（マルチGPU時は **GPU 1枚あたり**。global batch size = これ × `training.num_gpus`） |
| `training.num_gpus` | 1 | DDP で使用する GPU 数（[5-1b](#5-1b-マルチgpu学習ddpglobal-batch-size-を大きくしたい場合)参照） |
| `optimizer.lr` | 1e-4 | 学習率 |
| `training.rollout_every` | 500 | 環境ロールアウトの実行間隔（エポック） |
| `policy.num_inference_steps` | 10 | 推論時のフローステップ数 |
| `task.env_runner.task_ids` | `[0]` | 学習中の定期ロールアウトで評価するタスク（速度優先で1タスクのみ。全タスク評価は Step 6 参照） |
| `task.env_runner.eval_episodes_per_task` | 20 | 1タスクあたりのロールアウト試行数 |

hydra オーバーライドで変更可能（例: `training.num_epochs=1000`）。

### 5-3. チェックポイントの保存先

```
Flow_Policy/ManiFlow/data/outputs/<DATE>/<TIME>_<exp_name>_<task_name>/
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

ログは `https://wandb.ai/<YOUR_USERNAME>/flow_policy_libero/` に記録される
（`logging.project` = `flow_policy_libero`、RoboCasa の `flow_policy_robocasa` とは別プロジェクト）。

---

## Step 6: 評価の実行

### 6-1. 学習済みモデルの評価（学習中と同じ代表タスクのみ・高速）

```bash
singularity exec --nv \
    --bind "/home/bilgehan.sakai:/home/bilgehan.sakai" \
    --env MUJOCO_GL=osmesa \
    --env PYOPENGL_PLATFORM=osmesa \
    --env CUDA_VISIBLE_DEVICES=0 \
    "$SIF" \
    bash -c "
        source /home/bilgehan.sakai/Flow_Policy/.venv-libero/bin/activate
        cd /home/bilgehan.sakai/Flow_Policy/ManiFlow

        python -m maniflow.workspace.eval_maniflow_libero_workspace \
            --config-name maniflow_image_timm_policy_libero \
            task=libero_spatial \
            training.seed=42 \
            training.device='cuda:0' \
            exp_name=libero_spatial_run \
            'hydra.run.dir=data/outputs/<DATE>/<TIME>_libero_spatial_run_libero_spatial'
    "
```

### 6-2. 公式 LIBERO ベンチマーク相当の全タスク評価

`task.env_runner.task_ids` を `null` にすると、スイート内の全タスク
（各スイート10タスク、libero_90 は90タスク）を評価し、タスクごとの成功率と
全体平均成功率の両方をログに出力する（openvla-oft の評価プロトコルと同じ:
`num_trials_per_task` 試行ずつ全タスクを回す）。

```bash
task.env_runner.task_ids=null \
task.env_runner.eval_episodes_per_task=50
```

または付属スクリプトで:

```bash
bash scripts/eval_checkpoint_libero.sh 0 \
    ManiFlow/data/outputs/2026.07.22/10.00.00_libero_spatial_run_libero_spatial \
    best libero_spatial true   # 最後の "true" が全タスク評価を意味する
```

出力される主な指標（`eval_results/<epoch>/steps10_<tag>/metrics_<mode>.json`）:

| キー | 意味 |
|---|---|
| `mean_success_rates` | 評価対象タスク全体・全エピソードでの平均成功率 |
| `SR_task00`, `SR_task01`, ... | タスクごとの成功率 |
| `SR_test_L3` / `SR_test_L5` | 学習中に記録された上位3回/5回平均成功率（RoboCasa と同じ `LargestKRecorder` による） |

動画は `eval_results/<epoch>/steps10_<tag>/videos/task<NN>_ep<NNN>.mp4` に保存される。

---

## Step 7: 動作確認ログ（実施済み・2026-07-22）

本ガイドの Step 1〜6 を実際に本サーバーで最後まで実行し、
「LIBERO インストール → データセット変換 → 数エポック学習 → 数エピソード評価」を
end-to-end で動作確認した際の実行記録。スモークテストのため
`libero_spatial` の task 0（`pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate`）
1タスクのみ・少数デモに絞っている。全タスク・本格学習を行う場合は Step 4〜6 の
標準コマンド（`task=libero_spatial` 等、フル suite）を使うこと。

### 7-1. データセット変換（1タスク・20デモに限定）

```bash
python scripts/convert_libero_to_lerobot.py \
    --libero_task_suite libero_spatial \
    --libero_raw_data_dir libero/libero/libero/datasets/libero_spatial \
    --libero_target_dir data/libero/datasets/lerobot/libero_spatial \
    --task_ids 0 \
    --max_demos_per_task 20
```

**結果:** 20デモ中 18デモがリプレイ成功（2件は `done=True` に到達せず除外）。
所要時間 約9分（1エピソードあたり平均 30〜60秒、osmesa ソフトウェアレンダリングで
256×256 の agentview + wrist の2カメラを毎ステップ描画するため）。
`LiberoImageDataset` で読み込み確認済み: 18エピソード → train 17 / val 1、
`horizon=10, pad_before=1, pad_after=7` で 1336 学習サンプル（1エピソードあたり平均73ステップ）。

### 7-2. 学習（実質4エポック、`libero_test` タスク設定）

```bash
python -m maniflow.workspace.train_maniflow_libero_workspace \
    --config-name maniflow_image_timm_policy_libero \
    task=libero_test \
    training.seed=42 training.device=cuda:0 \
    training.num_epochs=5 training.rollout_every=1000 \
    training.checkpoint_every=1 training.val_every=1 training.sample_every=1 \
    dataloader.batch_size=32 dataloader.num_workers=0 dataloader.persistent_workers=false \
    val_dataloader.batch_size=32 val_dataloader.num_workers=0 val_dataloader.persistent_workers=false \
    logging.mode=disabled \
    exp_name=libero_smoketest hydra.run.dir=data/outputs/libero_smoketest
```

（`training.rollout_every=1000` でロールアウトをスキップし純粋な学習速度を確認、
`logging.mode=disabled` で wandb 認証なしに実行、`dataloader.num_workers=0` は
PyAV デッドロック回避の定石設定。1エポック約4〜5分、1GPU=RTX A6000。）

**loss 推移の実測値**（1バッチ約5秒、1エポック42ステップ）:

| epoch | train_loss | val_loss |
|---|---|---|
| 0 | (推移中) | 2.178647 |
| 1 | (推移中) | 1.777469 |
| 2（resume 後、実質3周目） | (推移中) | 1.339879 |
| 3（resume 後、実質4周目） | 1.0985 | 0.799635 |

（`training.resume=true` で `latest.ckpt` から再開して追加学習した際、
`self.epoch` が保存時点で1エポックずれる既知の挙動により、resume 後の
tqdm 表示は "epoch 1, epoch 2" から再開する — RoboCasa のワークスペース実装
由来の特性で、学習自体は正しく継続している。詳細は `train_maniflow_libero_workspace.py`
の `save_checkpoint`/`run()` のコメント参照。）

4回分のデータパス（実質エポック）で val_loss が 2.18 → 0.80 まで一貫して低下しており、
学習パイプライン（データ読み込み → 正規化 → ManiFlowTransformerImagePolicy の
flow-matching loss 計算 → 逆伝播 → EMA 更新 → チェックポイント保存）が
正しく機能していることを確認した。

### 7-3. 評価（3エピソード）

```bash
python -m maniflow.workspace.eval_maniflow_libero_workspace \
    --config-name maniflow_image_timm_policy_libero \
    task=libero_test \
    training.device=cuda:0 training.use_ema=true \
    +eval_mode=latest +eval_dir_tag=smoketest \
    task.env_runner.eval_episodes_per_task=3 \
    task.env_runner.task_ids='[0]' \
    task.env_runner.max_episode_steps=100 \
    hydra.run.dir=data/outputs/libero_smoketest
```

**結果（`eval_results/2/steps10_smoketest/metrics_latest.json`）:**
```json
{
    "SR_task00": 0.0,
    "mean_success_rates": 0.0,
    "test_mean_score": 0.0,
    "SR_test_L3": 0.0,
    "SR_test_L5": 0.0
}
```
3エピソードとも成功せず（SR=0%）。これは実質4エポック・17学習エピソードのみという
極小規模学習であるため予想通りの結果であり、**評価ループ自体が例外なく完走し、
ロールアウト・動画保存・メトリクス出力が正しく動作すること**を確認するのが目的。
動画は `eval_results/2/steps10_smoketest/videos/task00_ep000.mp4`〜`task00_ep002.mp4`
（各エピソード約36〜38秒で完了、1エピソードあたり実行時間の大半は
`num_open_loop_steps=8` ごとのポリシー推論と osmesa レンダリング）。

### 7-4. 本ログで判明し、対応済みの実装バグ

以下は実装当初想定していなかったが、実機検証で発見し修正したバグ
（コードは既に修正済み。ここでは記録として残す）:

1. **`LiberoEnv.reset()` が存在しない `get_observation()` を呼んでいた**
   （`maniflow/env/libero/libero_wrapper.py`）。LIBERO の `ControlEnv` には
   `get_observation()` が無く、`reset()` 自体が観測を返す設計だったため修正。
2. **`LiberoImageDataset._discover_lerobot_dirs` のパス解決バグ**
   （`maniflow/dataset/libero_dataset.py`）。`root.glob("*/meta/info.json")` の
   マッチ結果に対し `.parent`（→ `meta` ディレクトリ）ではなく
   `.parent.parent`（→ タスクディレクトリ本体）を取る必要があった。

### 7-5. `libero_spatial` 全タスク・全デモの変換（本番相当データセット）

Step 7-1〜7-3 のスモークテスト（1タスク・20デモ限定）とは別に、`libero_spatial`
10タスク・各50デモ（計500デモ）の**フル変換**もバックグラウンドで実行し、完走を確認済み
（`--task_ids`・`--max_demos_per_task` を指定せず、Step 4-2 の標準コマンドをそのまま実行）。

**所要時間:** 約5.5時間（1エピソードあたり平均約40秒 × 451件成功 + リプレイ失敗分のオーバーヘッド）。

**タスクごとの結果（`saved N/50`）:**

| task_id | タスク名 | 成功デモ数 |
|---|---|---|
| 0 | pick_up_the_black_bowl_between_the_plate_and_the_ramekin... | 48/50 |
| 1 | pick_up_the_black_bowl_next_to_the_ramekin... | 45/50 |
| 2 | pick_up_the_black_bowl_from_table_center... | 48/50 |
| 3 | pick_up_the_black_bowl_on_the_cookie_box... | 46/50 |
| 4 | pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet... | 43/50 |
| 5 | pick_up_the_black_bowl_on_the_ramekin... | 41/50 |
| 6 | pick_up_the_black_bowl_next_to_the_cookie_box... | 48/50 |
| 7 | pick_up_the_black_bowl_on_the_stove... | 40/50 |
| 8 | pick_up_the_black_bowl_next_to_the_plate... | 48/50 |
| 9 | pick_up_the_black_bowl_on_the_wooden_cabinet... | 44/50 |
| **合計** | | **451/500（90.2%）** |

タスクごとの成功率は80〜96%とばらつきがあるが、Step 7-1 で観測した「約90%」という
概算とおおむね一致する。変換後データはディスク上で約105MB（256×256の2カメラ動画+
parquetのみで、生HDF5の128×128・全観測フィールド含みの5.9GBよりはるかに小さい）。

`LiberoImageDataset(dataset_dirs="data/libero/datasets/lerobot/libero_spatial", ...)`
で読み込み確認済み: 全10タスクが自動検出され、**451エピソード → train 442 / val 9 →
46,540学習サンプル**（`horizon=10, pad_before=1, pad_after=7, val_ratio=0.02` の場合）。
`task=libero_spatial`（フルスイート設定）でそのまま本格学習に使用できる状態。

### 7-6. `libero_goal` / `libero_object` / `libero_10` 全タスク・全デモの変換（gop_size=10 を変換時に直接適用）

Step 7-5 の `libero_spatial`（変換後に別ディレクトリへ `reencode_libero_videos.py`
で再エンコード — [5-1e](#5-1e-既存データセットの-gop-短縮による追加高速化オプション)参照）
とは異なり、`libero_goal`・`libero_object`・`libero_10` は
`convert_libero_to_lerobot.py --gop_size 10` を**変換時に直接指定**し、
シミュレータのリプレイと同時に GOP 短縮済み mp4 を書き出した（2026-07-25 実施）。
`libero_goal`・`libero_object` は生 HDF5 が Step 4-1 で既にダウンロード済みだったため
変換のみ実行。`libero_10` は後述の通り生データも新規ダウンロードしている。

```bash
python scripts/convert_libero_to_lerobot.py \
    --libero_task_suite libero_goal \
    --libero_raw_data_dir libero/libero/libero/datasets/libero_goal \
    --libero_target_dir data/libero/datasets/lerobot/libero_goal \
    --gop_size 10

python scripts/convert_libero_to_lerobot.py \
    --libero_task_suite libero_object \
    --libero_raw_data_dir libero/libero/libero/datasets/libero_object \
    --libero_target_dir data/libero/datasets/lerobot/libero_object \
    --gop_size 10

python scripts/convert_libero_to_lerobot.py \
    --libero_task_suite libero_10 \
    --libero_raw_data_dir libero/libero/libero/datasets/libero_10 \
    --libero_target_dir data/libero/datasets/lerobot/libero_10 \
    --gop_size 10
```

**結果（`saved N/50`・全10タスク合計）:**

| スイート | 成功デモ数 | 変換後サイズ |
|---|---|---|
| `libero_goal` | 451/500（90.2%） | 145MB |
| `libero_object` | 459/500（91.8%） | 203MB |
| `libero_10` | 404/500（80.8%） | 272MB |

`libero_10` の成功率がやや低い（80.8%）のは、`libero_10` のデモが他スイートより
長い（`TASK_MAX_STEPS`: `libero_10`=520 vs `libero_goal`=300 / `libero_object`=280、
実際 episode 長も200〜280ステップ台とlibero_spatial/goal/objectの100〜180ステップ台より
長い）ため、リプレイが `done=True` に到達する前に物理エンジンの差異が蓄積しやすいと
考えられる（Step 4-2・トラブルシューティングの「データセット変換で成功デモが極端に
少ない」節と同じ原因）。

`ffprobe` で実際のキーフレーム間隔を確認済み（`libero_10` の1エピソード、
約460フレーム中 I-frame 44枚 ≈ 10フレームおきで `--gop_size 10` の指定通り）。
3スイートとも `ManiFlow/maniflow/config/task/{libero_goal,libero_object,libero_10}.yaml`
の `dataset_base_path` がそのままこの変換先を指しているため、追加のconfig変更や
hydraオーバーライドは不要で `task=libero_goal` 等を指定するだけで学習に使える。

---

## トラブルシューティング

### `robosuite` のバージョン競合（RoboCasa と LIBERO を同じ venv に入れてしまった）

**症状:** `TypeError` や `AttributeError` が robosuite の API 呼び出しで発生する、
あるいは `pip`/`uv` の依存解決時に robosuite のバージョンが意図しないものになる。

**原因:** RoboCasa は `robosuite@main`、LIBERO は `robosuite==1.4.1` を要求しており、
同一 venv では共存できない。

**修正:** 本ガイドの Step 2〜3 の通り、LIBERO は必ず `.venv-libero`
（`UV_PROJECT_ENVIRONMENT=.venv-libero uv sync ...`）に隔離してインストールする。
`.venv`（RoboCasa 用）に誤って `robosuite==1.4.1` を入れてしまった場合は、
`docs/setup_and_train_robocasa.md` の手順に従って robosuite を `@main` に戻す。

---

### `ImportError: cannot import name 'benchmark' from 'libero.libero' (unknown location)`

**原因:** LIBERO editable インストールの namespace パッケージ誤認識（Step 3-4 パッチ1）。
**修正:** `uv pip install --config-settings editable_mode=compat -e libero/` で再インストールする。
`python -c "import libero.libero as l; print(l.__file__)"` が `None` でなくパスを
表示すれば解消している。

---

### `ImportError: cannot import name 'cached_download' from 'huggingface_hub'`

**原因:** `diffusers==0.27.2` が `huggingface_hub.cached_download`（`huggingface_hub>=0.26`
で削除済み）を import しようとしている。`uv sync` が解決する `huggingface_hub` の
バージョンは固定されておらず、実行タイミングによって非互換な新しいバージョンが
入ってしまうことがある。

**修正:** Step 3-2 の通り `uv pip install 'huggingface_hub==0.25.2'` で固定する
（RoboCasa の `.venv` も同じバージョンで動作している）。

---

### `ModuleNotFoundError: No module named 'r3m'`

**原因:** `policy.obs_encoder`（`TimmObsEncoder`）の `model_name: 'r3m'` が
`third_party/r3m`（Flow_Policy リポジトリに同梱、RoboCasa 用に導入済みの第三者コード）
を要求するが、`.venv-libero` には未インストール。

**修正:** `uv pip install -e /home/bilgehan.sakai/Flow_Policy/third_party/r3m`（Step 3-2 参照）。

---

### `_pickle.UnpicklingError: Weights only load failed ... numpy.core.multiarray._reconstruct`

**原因:** PyTorch 2.6 以降 `torch.load` のデフォルトが `weights_only=True` に変わったが、
LIBERO 本体の `Benchmark.get_task_init_states()` が weights_only を指定せずに
`torch.load()` を呼んでいるため、pickle 化された numpy 配列（`.pruned_init` ファイル）
の読み込みで失敗する。`LiberoEnv.__init__` が `task_suite.get_task_init_states()` を
呼ぶタイミング（学習中のロールアウト・評価の両方）で発生する。

**修正:** Step 3-4 パッチ2の通り、LIBERO 本体にパッチを当てる:
```bash
sed -i 's/init_states = torch.load(init_states_path)/init_states = torch.load(init_states_path, weights_only=False)/' \
    /home/bilgehan.sakai/Flow_Policy/libero/libero/libero/benchmark/__init__.py
```

---

### EGL レンダリング（RoboCasa と同じ問題・同じ解決策）

本サーバーは `render` グループに未所属のため、`MUJOCO_GL=osmesa` と
`PYOPENGL_PLATFORM=osmesa` を**両方**設定する必要がある。詳細は
[setup_and_train_robocasa.md のレンダリングに関するトラブルシューティング](./setup_and_train_robocasa.md#egl-レンダリングgpu-ハードウェア-egl-vs-osmesa-ソフトウェアレンダリング)
を参照（LIBERO の `OffScreenRenderEnv` も内部的には同じ MuJoCo レンダリングバックエンド
設定に従うため、同一の対処で解決する）。

---

### データセット変換で成功デモが極端に少ない / 0 件になる

**原因候補:**
1. `scripts/convert_libero_to_lerobot.py` の `replay_demo` がデモの初期状態
   （`orig_states[0]`）から `orig_actions` を単純リプレイしているため、
   物理エンジンのバージョン差異等でオリジナルのデモと異なる挙動になり、
   `done=True` に到達しないデモが発生することがある（openvla-oft の
   `regenerate_libero_dataset.py` でも同様の理由で一部デモが除外される）。
   **実測値（Step 7-1 参照）:** `libero_spatial` task 0 の 20 デモ中 18 デモが成功
   （10%程度は正常な失敗率として想定内）。
2. `--libero_raw_data_dir` に指定したパスにタスクに対応する
   `<task_name>_demo.hdf5` が存在しない。パスの3段 `libero/` 構造
   （Step 3-3 のパス構造の注意を参照）を間違えている場合が多い。

**確認方法:** `convert_libero_to_lerobot.py` 実行時に出力される
`[WARNING] Skipping task ...` や `saved N/M successful demos` のログで
どのタスク・何件が失敗しているか確認する。

---

### DataLoader がデッドロックする（RoboCasa と同じ問題）

**原因:** `fork` による multiprocessing + PyAV のスレッドセーフ問題。
**修正:** `dataloader.num_workers=0` でシングルスレッドに落とす。

```bash
dataloader.num_workers=0
```

---

## ファイル構成（LIBERO 関連・まとめ）

```
Flow_Policy/
├── pyproject.toml                              # [dependency-groups].libero に手順コメントあり
├── libero/                                      # Lifelong-Robot-Learning/LIBERO (git clone)
├── .venv-libero/                                # LIBERO 専用 venv（.venv とは別）
├── data/
│   └── libero/
│       └── datasets/
│           └── lerobot/
│               ├── libero_spatial/<task_name>/...
│               ├── libero_object/<task_name>/...
│               ├── libero_goal/<task_name>/...
│               └── libero_10/<task_name>/...
├── docs/
│   └── setup_and_train_libero.md               # 本ドキュメント
├── scripts/
│   ├── convert_libero_to_lerobot.py            # HDF5 → LeRobot 形式 変換（--gop_size 対応）
│   ├── reencode_libero_videos.py               # 既存mp4のGOP再エンコード（シミュレータ不要、高速）
│   ├── download_libero_dataset.sh              # 生データDL + 変換 一括実行
│   ├── train_eval_libero.sh                    # 学習・評価
│   └── eval_checkpoint_libero.sh               # チェックポイント評価（動画保存つき）
└── ManiFlow/
    └── maniflow/
        ├── env/
        │   ├── __init__.py                     # RoboCasaEnv/LiberoEnv を optional import
        │   └── libero/
        │       ├── __init__.py
        │       └── libero_wrapper.py            # LiberoEnv
        ├── dataset/
        │   ├── libero_dataset.py                # LiberoImageDataset（decord/CPUデコード、デフォルト）
        │   └── libero_dataset_dali.py            # LiberoDaliLoader（DALI/NVDECデコード、use_dali=true時）
        ├── env_runner/
        │   └── libero_runner.py                 # LiberoRunner
        ├── workspace/
        │   ├── train_maniflow_libero_workspace.py
        │   └── eval_maniflow_libero_workspace.py
        └── config/
            ├── maniflow_image_timm_policy_libero.yaml
            └── task/
                ├── libero_spatial.yaml
                ├── libero_object.yaml
                ├── libero_goal.yaml
                ├── libero_10.yaml
                ├── libero_90.yaml
                └── libero_test.yaml
```
