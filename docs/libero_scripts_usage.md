# LIBERO 学習・評価スクリプトの使い方

`scripts/train_eval_libero.sh` と `scripts/eval_checkpoint_libero.sh` の使い方リファレンス。
環境構築そのもの（Singularity イメージ、venv、LIBERO のインストール等）は
[`docs/setup_and_train_libero.md`](./setup_and_train_libero.md) を参照。本ドキュメントは
セットアップ済みであることを前提に、2つのスクリプトの引数・挙動・具体例のみをまとめる。

両スクリプトとも Singularity コンテナの外から実行しても自動で
`singularity exec` に再実行される（`$SINGULARITY_CONTAINER` が未設定なら自動で
コンテナに入り直す）ので、コンテナに手動で入る必要はない。

---

## 1. `scripts/train_eval_libero.sh` — 学習・（学習と同じ設定での）評価

### 1-1. 構文

```bash
bash scripts/train_eval_libero.sh [gpu_id(s)] [seed] [exp_name] [mode] [task_suite] [extra hydra overrides...]
```

全引数省略可（位置引数なので、後ろの引数を指定したい場合は手前も埋める必要がある）。

| 引数 | 意味 | デフォルト |
|---|---|---|
| `gpu_id(s)` | GPU番号。`"0,1,2,3"` のようにカンマ区切りで複数指定すると `train` モードのみマルチGPU DDP学習になる（`eval` モードは常に1GPU、複数指定しても先頭のみ使用） | `0` |
| `seed` | `training.seed`。学習の乱数シード | `42` |
| `exp_name` | wandbのrun名 / ログディレクトリ名の一部 | `libero_<task_suite>` |
| `mode` | `train` または `eval` | `train` |
| `task_suite` | `maniflow/config/task/*.yaml` のtask設定名（下記1-2参照） | `libero_spatial` |
| 以降すべて | Hydraの追加オーバーライド（`key=value` 形式でいくつでも指定可） | なし |

マルチGPU時、`dataloader.batch_size` は **GPU1枚あたり** の値。global batch size は
`dataloader.batch_size × GPU数` になる。

### 1-2. `task_suite` の選択肢

| 値 | 内容 |
|---|---|
| `libero_spatial` / `libero_object` / `libero_goal` / `libero_10` | LIBERO標準4スイートのいずれか単体（`libero_10` は論文等で"LIBERO-Long"とも呼ばれる） |
| `libero_90` | LIBERO-90スイート |
| `libero_test` | 3エピソードのみのスモークテスト用（動作確認専用） |
| `libero_all4` | 上記4スイート（spatial/object/goal/10）を1つの学習セットに混合。詳細は1-4参照 |

### 1-3. `mode` の違い

- **`train`**: `maniflow.workspace.train_maniflow_libero_workspace` を実行。通常の学習ループ（`training.num_epochs` 分）を回し、`training.val_every` 毎に検証損失、`training.rollout_every` 毎に環境ロールアウトによる成功率評価を行い、両方ともwandbに記録する。チェックポイントは `training.checkpoint_every` 毎に保存される。
- **`eval`**: `maniflow.workspace.eval_maniflow_libero_workspace` を実行。指定した `hydra.run.dir`（既存の学習出力ディレクトリ）から「最良/最新チェックポイント」を読み込み、ロールアウトのみを1回実行してローカルに `metrics_*.json` と動画を保存する。**wandbには記録されない**（ローカルファイル出力のみ）。マルチGPU非対応。

`eval` モードで既存の学習結果を評価する場合、`hydra.run.dir` を追加オーバーライドで明示的に指定する必要がある（`exp_name`/`seed` だけでは学習時と同じ出力先が再現されるとは限らない）:

```bash
bash scripts/train_eval_libero.sh 0 42 my_eval eval libero_all4 \
    "hydra.run.dir=/storage/home/bilgehan.sakai/Flow_Policy/ManiFlow/data/outputs/2026.08.07/16.22.16_smoke_test_seen_unseen_v2_seed42"
```
（チェックポイント選択のロジックは `eval_checkpoint_libero.sh` の方が簡単に扱えるので、既存学習結果の評価には基本的にそちらを推奨。2章参照）

### 1-4. `libero_all4` の仕様（seen/unseen 評価・検証損失）

`libero_all4` は 4スイート×10タスクを1つの学習データセットに混合するが、**タスク単位**で
以下のような固定分割を行う（`maniflow/config/task/libero_all4.yaml` の
`task_split_seed` / `val_tasks_per_suite` / `seen_tasks_per_suite`）:

- 各スイートごとに、10タスク中 **2タスクを丸ごと学習から除外**（`val_tasks_per_suite: 2`）
- 残り8タスクのうち **2タスクを「seen」評価用に選出**（`seen_tasks_per_suite: 2`）

この分割は `task_split_seed`（デフォルト `0`）で決定され、**`--seed`（`training.seed`）を変えても
同じタスクが選ばれる**（学習試行間で再現性がある）。

記録される指標（wandb、`train` モードは毎epoch、`eval` モードは1回のみ）:

| 指標 | 内容 |
|---|---|
| `val_loss` / `val_loss_<suite>` | 除外した2タスク分のエピソードでのBC損失（全体プール値 / スイート別） |
| `unseen_mean_success_rate_<suite>` | 除外した2タスク（学習に一切使っていない）でのロールアウト成功率。`task.env_runner.unseen_episodes_per_task`（デフォルト10）エピソード/タスク |
| `seen_mean_success_rate_<suite>` | 学習に使った8タスクのうち選出した2タスクでのロールアウト成功率（損失計算はしない、ロールアウトのみ）。`task.env_runner.seen_episodes_per_task`（デフォルト10）エピソード/タスク |
| `mean_success_rates` / `test_mean_score` | seen/unseen 全体を合算した平均成功率（チェックポイント選定等には使われない参考値） |

`<suite>` は `libero_spatial` / `libero_object` / `libero_goal` / `libero_10` のいずれか。

エピソード数を変更したい場合:

```bash
bash scripts/train_eval_libero.sh 0 42 myrun train libero_all4 \
    task.env_runner.seen_episodes_per_task=20 \
    task.env_runner.unseen_episodes_per_task=20
```

単体スイート（`libero_spatial` 等）は従来通り `task.env_runner.task_ids` /
`task.env_runner.eval_episodes_per_task` で制御する（seen/unseenの概念はない）。

### 1-5. DALI（動画デコード高速化・CPUワーカー削減）

本サーバーはシステムRAMに余裕がないため、CPUマルチプロセスの `DataLoader`
（`num_workers>1`）はワーカープロセスがメモリ不足で異常終了しやすい。
`use_dali=true` にすると動画デコードがNVDEC（GPU側）に移り、CPUワーカープロセスを
一切使わなくなるため、この問題を回避できる。**RAMに余裕がない環境ではDALIの使用を推奨。**

```bash
bash scripts/train_eval_libero.sh 0 42 myrun train libero_all4 \
    dataloader.use_dali=true \
    val_dataloader.use_dali=true
```

`use_dali=true` のとき `num_workers` / `persistent_workers` は無視される（値を指定しても無害）。
libero_all4 のスイート別検証ローダー（`val_loss_<suite>`）も `val_dataloader.use_dali` の
設定に自動的に追従する。

### 1-6. 主要な `training.*` オーバーライド

| キー | デフォルト | 意味 |
|---|---|---|
| `training.num_epochs` | 501 | 学習エポック数 |
| `training.rollout_every` | 500 | ロールアウト評価の間隔（epoch） |
| `training.val_every` | 50 | 検証損失計算の間隔（epoch） |
| `training.checkpoint_every` | 100 | チェックポイント保存の間隔（epoch） |
| `training.max_train_steps` / `training.max_val_steps` | `null`（無制限） | 1epochあたりの最大step数を制限（デバッグ用。指定するとDataLoaderの反復が早期に打ち切られるため、`num_workers>0` 環境ではワーカーの異常終了ログが出ることがある。実運用では基本 `null` のまま） |
| `checkpoint.topk.monitor_key` | `val_loss` | チェックポイントのtop-K保存で使う指標。値を変える場合は下記に注意 |
| `checkpoint.topk.format_str` | `'epoch={epoch:04d}-val_loss={val_loss:.6f}.ckpt'` | チェックポイントファイル名の書式 |

**注意**: `checkpoint.topk.format_str` を変更してファイル名に別の指標（例:
`test_mean_score`）を入れても、`checkpoint.topk.monitor_key` は自動では変わらない。
`eval_checkpoint_libero.sh` の `best` モードはファイル名中の `{monitor_key}=<値>`
部分をパースして最良チェックポイントを探すため、`format_str` と `monitor_key`
が指す指標がずれていると `best` 選択が機能しなくなる。両方変更する場合は揃えること:

```bash
    checkpoint.topk.monitor_key=test_mean_score \
    checkpoint.topk.mode=max \
    "checkpoint.topk.format_str='epoch={epoch:04d}-test_mean_score={test_mean_score:.4f}.ckpt'"
```

### 1-7. 実行例

```bash
# libero_spatial を1GPUで学習（デフォルト設定）
bash scripts/train_eval_libero.sh 0 42

# libero_spatial を4GPU DDPで学習
bash scripts/train_eval_libero.sh 0,1,2,3 42 libero_spatial_4gpu train libero_spatial

# 動作確認用スモークテスト（3エピソードのみ）
bash scripts/train_eval_libero.sh 0 0 smoke train libero_test

# libero_all4 を1GPUで学習（DALI使用、RAM節約）
bash scripts/train_eval_libero.sh 0 42 libero_all4_main train libero_all4 \
    dataloader.use_dali=true \
    val_dataloader.use_dali=true

# libero_all4 を4GPU DDPで学習
bash scripts/train_eval_libero.sh 0,1,2,3 42 libero_all4_4gpu train libero_all4 \
    dataloader.use_dali=true \
    val_dataloader.use_dali=true

# 既存の学習結果ディレクトリに対して eval モードを実行
bash scripts/train_eval_libero.sh 0 42 my_eval eval libero_all4 \
    "hydra.run.dir=<既存の学習出力ディレクトリの絶対パス>"
```

---

## 2. `scripts/eval_checkpoint_libero.sh` — 既存チェックポイントの評価・動画保存

`train_eval_libero.sh eval` モードの薄いラッパー。既存の学習出力ディレクトリを指定するだけで
`hydra.run.dir` の組み立てやチェックポイント選択（latest/best）を自動化してくれる。
評価専用に別GPUを使いたい場合などにも便利（デフォルトGPUが `1` なのは、GPU0で学習中でも
並行して評価できるようにするため）。

### 2-1. 構文

```bash
bash scripts/eval_checkpoint_libero.sh [gpu_id] [training_dir] [eval_mode] [task_suite] [full_suite]
```

| 引数 | 意味 | デフォルト |
|---|---|---|
| `gpu_id` | GPU番号 | `1` |
| `training_dir` | 学習出力ディレクトリ（`checkpoints/` を含む）。相対パスは `Flow_Policy/` からの相対とみなされる | 最新の学習出力ディレクトリを自動検出 |
| `eval_mode` | `latest`（最終epochの`latest.ckpt`）または `best`（`checkpoint.topk.monitor_key` が最良のもの） | `latest` |
| `task_suite` | 学習時に使ったのと同じtask設定名 | `libero_spatial` |
| `full_suite` | `true` でより広い評価を実行（下記2-2参照）、`false` で学習中の定期ロールアウトと同じ軽量設定 | `false` |

### 2-2. `full_suite=true` の挙動（`task_suite` により異なる）

- **単体スイート**（`libero_spatial` 等）: `task.env_runner.task_ids=null`（全タスク）
  `task.env_runner.eval_episodes_per_task=50` を適用（LIBERO公式プロトコル相当）。
- **`libero_all4`**: タスク自体は変わらず（seen 2タスク・unseen 2タスク/スイートの固定分割のまま）、
  `task.env_runner.seen_episodes_per_task=50` / `task.env_runner.unseen_episodes_per_task=50`
  に引き上げてより多くの試行を行う（`libero_all4` に「全タスク評価」という概念はない。1-4参照）。

### 2-3. 環境変数（オプション）

| 変数 | 意味 |
|---|---|
| `TASK_IDS` | 例 `"[7,9]"`。特定タスクのみ評価したい場合の明示的なHydraリストオーバーライド。**単体スイート専用**（`libero_all4` では使えない。`libero_all4` で個別に調整したい場合は `EXTRA_OVERRIDES` で `task.env_runner.seen_episodes_per_task=...` 等を直接指定する） |
| `EPISODES_PER_TASK` | `TASK_IDS` 指定時のエピソード数/タスク（デフォルト50） |
| `EXTRA_OVERRIDES` | 任意の追加Hydraオーバーライド（スペース区切り文字列）。例: 学習時に `policy.use_sra=true` 等アーキテクチャを変えていた場合、state_dictのキーを合わせるために評価時にも同じ値を渡す必要がある |

### 2-4. 出力

```
<training_dir>/eval_results/<epoch>/steps<N>_<tag>/metrics_<eval_mode>.json
<training_dir>/eval_results/<epoch>/steps<N>_<tag>/videos/<...>.mp4
```

- `<epoch>`: 読み込んだチェックポイントのepoch番号
- `<N>`: 推論ステップ数（`eval_inference_steps` config、デフォルト10）
- `<tag>`: `canonical_task`（デフォルト） / `full_suite`（`full_suite=true`） /
  `tasks_<TASK_IDS>`（`TASK_IDS` 指定時）
- `metrics_*.json` の中身は `unseen_mean_success_rate_<suite>` 等、1章のwandb指標と同じキー
  （wandbには送られず、ローカルJSONのみ）

### 2-5. 実行例

```bash
# 学習中と同じ軽量設定で、最新チェックポイントを評価
bash scripts/eval_checkpoint_libero.sh 1 \
    ManiFlow/data/outputs/2026.07.22/10.00.00_libero_spatial_run_libero_spatial \
    latest libero_spatial false

# 最良チェックポイント（val_loss最小）で、libero_spatial の全10タスク×50エピソード評価
bash scripts/eval_checkpoint_libero.sh 0 \
    ManiFlow/data/outputs/2026.07.22/10.00.00_libero_spatial_run_libero_spatial \
    best libero_spatial true

# libero_all4 の最良チェックポイントで、seen/unseenを各50エピソードに引き上げて評価
bash scripts/eval_checkpoint_libero.sh 0 \
    ManiFlow/data/outputs/2026.08.07/16.22.16_smoke_test_seen_unseen_v2_seed42 \
    best libero_all4 true

# 学習時に use_sra=true だったチェックポイントを評価（state_dict整合のため必須）
EXTRA_OVERRIDES="policy.use_sra=true" \
bash scripts/eval_checkpoint_libero.sh 1 \
    ManiFlow/data/outputs/2026.08.01/12.00.00_sra_run_libero_all4 \
    best libero_all4 false
```

---

## 3. どちらを使うべきか

| やりたいこと | スクリプト |
|---|---|
| 新規学習を始める | `train_eval_libero.sh ... train ...` |
| 学習をゼロから再現しつつ、学習後1回だけ評価も流したい | `train_eval_libero.sh ... eval ...`（`hydra.run.dir` を手動指定） |
| 既存の学習済みチェックポイントを評価・動画保存したい（通常はこちら） | `eval_checkpoint_libero.sh` |
| 学習と並行して別GPUで随時チェックポイントを評価したい | `eval_checkpoint_libero.sh`（デフォルトGPUが学習用と別の `1` なのはこの用途のため） |

## 4. 以前使用した実行

```bash
# vanila学習
bash scripts/train_eval_libero.sh 0,1,2,3 42 libero_all4_dali_main train libero_all4 \
    dataloader.use_dali=true \
    val_dataloader.use_dali=true \
    dataloader.num_workers=0 \
    val_dataloader.num_workers=0 \
    dataloader.batch_size=128 \
    val_dataloader.batch_size=128 \
    optimizer.lr=2e-4 \
    policy.language_conditioned=true \
    policy.use_consistency=false \
    policy.use_sra=false \
    policy.obs_encoder.transforms=null \
    training.num_epochs=101 \
    training.rollout_every=10 \
    training.val_every=10 \
    training.checkpoint_every=50 \
    "checkpoint.topk.format_str='epoch={epoch:04d}-test_mean_score={test_mean_score:.4f}.ckpt'"

# SRAの時の引数
    policy.use_sra=true \
    policy.sra_block_out_s=3 \
    policy.sra_loss_weight=1.0 \
    policy.sra_mask_ratio=1.0 \
    policy.sra_dual_time_scheduling=false \
    policy.sra_attention_separation=false \
    policy.sra_teacher_mask=false \
    policy.sra_full_sample_prob=0.0 \
    policy.sra_teacher_t="sra" \
    policy.sra_loss_type="sml1" \
    policy.sra_weight_schedule=false \
    ema.use_decay_schedule=false \
    ema.fixed_decay=0.999 \

# 評価
# EXTRA_OVERRIDES："半角スペース区切りで学習時の'policy.'から始まる引数を入力。++eval_dir_tagで保存場所を指定"
EXTRA_OVERRIDES="policy.language_conditioned=true policy.use_consistency=false policy.use_sra=false ++eval_dir_tag=full_suite_libero_10" \
bash scripts/eval_checkpoint_libero.sh 1 ManiFlow/data/outputs/2026.08.01/02.44.42_libero_all4_dali_main_seed42 100 libero_10 true
```
