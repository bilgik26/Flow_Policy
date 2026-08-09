#!/usr/bin/env bash
# Train and evaluate Flow_Policy on a LIBERO task suite.
#
# Usage:
#   bash scripts/train_eval_libero.sh [gpu_id(s)] [seed] [exp_name] [mode] [task_suite]
#
# Arguments:
#   gpu_id(s)   : GPU index, or a comma-separated list for multi-GPU DDP
#                 training (default: 0). e.g. "0,1,2,3" trains with 4 GPUs
#                 and multiplies the global batch size by 4 (dataloader.batch_size
#                 is per-GPU). Multi-GPU is only used in "train" mode; "eval"
#                 always runs single-GPU (uses the first listed GPU).
#   seed        : random seed (default: 42)
#   exp_name    : wandb/run name (default: libero_<task_suite>)
#   mode        : "train" (default) or "eval"
#   task_suite  : "libero_spatial" (default) | "libero_object" | "libero_goal"
#                 | "libero_10" | "libero_90" | "libero_test" (3-episode smoke test)
#                 | "libero_all4" (libero_spatial + libero_object + libero_goal +
#                   libero_10 mixed into one training set — "libero_10" is the
#                   suite commonly called "LIBERO-Long"/"LIBERO-10" in papers;
#                   see maniflow/config/task/libero_all4.yaml)
#
# Examples:
#   bash scripts/train_eval_libero.sh 0 42                                   # libero_spatial, train, 1 GPU
#   bash scripts/train_eval_libero.sh 0,1,2,3 42 libero_spatial_4gpu train libero_spatial  # 4-GPU DDP
#   bash scripts/train_eval_libero.sh 0 42 libero_spatial_run eval libero_spatial
#   bash scripts/train_eval_libero.sh 0 0   smoke        train libero_test    # quick smoke test
#   bash scripts/train_eval_libero.sh 0,1,2,3 42 libero_all4_run train libero_all4  # all 4 suites mixed, 4-GPU DDP
#   bash scripts/train_eval_libero.sh 0 42 libero_all4_run eval libero_all4        # eval on all 4 suites
#
# libero_all4 specifics (maniflow/config/task/libero_all4.yaml,
# task_split_seed/val_tasks_per_suite/seen_tasks_per_suite): per suite (10
# LIBERO tasks each), a *fixed* split -- independent of --seed above, so it
# never changes across training runs -- holds out 2 whole tasks entirely
# from training and picks 2 of the remaining 8 trained-on tasks to probe:
#   - val_loss / val_loss_<suite>: BC loss on the 2 held-out tasks' episodes
#     (maniflow/dataset/libero_dataset.py's val_tasks_per_suite split).
#   - unseen_mean_success_rate_<suite>: rollout success rate on the 2
#     held-out (never-trained-on) tasks, unseen_episodes_per_task episodes
#     each.
#   - seen_mean_success_rate_<suite>: rollout success rate on 2 of the 8
#     trained-on tasks, seen_episodes_per_task episodes each -- no loss is
#     computed for these, only rollout.
# All four are logged to wandb every training.val_every / rollout_every
# epochs (train mode) or once (eval mode) -- same config drives both modes.
# See maniflow/env_runner/libero_runner.py's LiberoRunner docstring and
# maniflow/common/libero_task_split.py for how the split is computed.
#
# Override individual values via extra hydra args (appended after
# task_suite), e.g. to raise the "unseen" episode count for a more thorough
# eval-mode run:
#   bash scripts/train_eval_libero.sh 0 42 myrun eval libero_all4 \
#       task.env_runner.unseen_episodes_per_task=50
# (single-suite task configs keep using the plain task.env_runner.task_ids /
# task.env_runner.eval_episodes_per_task overrides as before.)

set -euo pipefail
set -a
source .env
set +a

GPU="${1:-0}"
SEED="${2:-42}"
TASK_SUITE="${5:-libero_spatial}"
EXP_NAME="${3:-libero_${TASK_SUITE}}"
MODE="${4:-train}"        # "train" or "eval"
EXTRA_ARGS=("${@:6}")     # additional hydra overrides, e.g. task.env_runner.task_ids=null

# GPU may be "0" or a comma-separated list like "0,1,2,3" for multi-GPU DDP.
IFS=',' read -ra GPU_LIST <<< "$GPU"
NUM_GPUS="${#GPU_LIST[@]}"
if [[ "$MODE" == "eval" && "$NUM_GPUS" -gt 1 ]]; then
    echo "NOTE: eval mode does not use DDP; using only the first GPU (${GPU_LIST[0]}) of: $GPU"
    GPU="${GPU_LIST[0]}"
    NUM_GPUS=1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE="$(dirname "$SCRIPT_DIR")"

# LIBERO needs robosuite==1.4.1, which conflicts with RoboCasa's
# robosuite@main requirement — see docs/setup_and_train_libero.md for why
# these two sims live in separate venvs.
VENV="$WORKSPACE/.venv-libero"

# If not inside Singularity, re-invoke via the container (osmesa requires it)
if [[ -z "${SINGULARITY_CONTAINER:-}" ]]; then
    SIF="/home/bilgehan.sakai/singularity/sif/flow_policy_robocasa.sif"
    if [[ ! -f "$SIF" ]]; then
        echo "ERROR: Singularity image not found at $SIF"
        exit 1
    fi
    echo "=== Re-invoking inside Singularity container ==="
    exec singularity exec --nv \
        --bind "/home/bilgehan.sakai:/home/bilgehan.sakai" \
        --bind "/storage/home/bilgehan.sakai:/storage/home/bilgehan.sakai" \
        --env MUJOCO_GL=osmesa \
        --env PYOPENGL_PLATFORM=osmesa \
        --env CUDA_VISIBLE_DEVICES="$GPU" \
        ${WANDB_API_KEY:+--env WANDB_API_KEY="$WANDB_API_KEY"} \
        "$SIF" \
        bash "$SCRIPT_DIR/train_eval_libero.sh" "$@"
fi

if [ ! -d "$VENV" ]; then
    echo "ERROR: venv not found at $VENV. Run setup first (see docs/setup_and_train_libero.md)."
    exit 1
fi
source "$VENV/bin/activate"

# MuJoCo headless rendering (both vars required; container sets PYOPENGL_PLATFORM=egl by default)
export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa
export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES="$GPU"

cd "$WORKSPACE/ManiFlow"

RUN_DIR="${RUN_DIR:-$WORKSPACE/ManiFlow/data/outputs/$(date +%Y.%m.%d)/$(date +%H.%M.%S)_${EXP_NAME}_seed${SEED}}"

if [[ "$MODE" == "eval" ]]; then
    echo "=== Evaluation only mode (task_suite=${TASK_SUITE}) ==="
    python -m maniflow.workspace.eval_maniflow_libero_workspace \
        --config-name maniflow_image_timm_policy_libero \
        task="$TASK_SUITE" \
        training.seed="$SEED" \
        training.device="cuda:0" \
        exp_name="$EXP_NAME" \
        "hydra.run.dir=$RUN_DIR" \
        "${EXTRA_ARGS[@]}"
else
    if [[ "$NUM_GPUS" -gt 1 ]]; then
        echo "=== Training mode (task_suite=${TASK_SUITE}, ${NUM_GPUS} GPUs: ${GPU}) ==="
    else
        echo "=== Training mode (task_suite=${TASK_SUITE}) ==="
    fi
    # training.device is only used for the num_gpus=1 path; with num_gpus>1 the
    # workspace picks cuda:<local_rank> per spawned process instead.
    python -m maniflow.workspace.train_maniflow_libero_workspace \
        --config-name maniflow_image_timm_policy_libero \
        task="$TASK_SUITE" \
        training.seed="$SEED" \
        training.device="cuda:0" \
        training.num_gpus="$NUM_GPUS" \
        exp_name="$EXP_NAME" \
        "hydra.run.dir=$RUN_DIR" \
        "${EXTRA_ARGS[@]}"
fi
