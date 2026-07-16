#!/usr/bin/env bash
# Train and evaluate Flow_Policy on RoboCasa tasks.
#
# Usage:
#   bash scripts/train_eval_robocasa.sh [gpu_id] [seed] [exp_name] [mode] [use_consistency] [use_sra] [task_type]
#
# Arguments:
#   task_type: "multitask" (default) or "1task"
#
# Examples:
#   # --- multitask (default) ---
#   bash scripts/train_eval_robocasa.sh 0 42 robocasa_multitask
#   bash scripts/train_eval_robocasa.sh 0 42 robocasa_multitask eval                          # eval only
#   bash scripts/train_eval_robocasa.sh 0 42 robocasa_flow      train false                   # flow only
#   bash scripts/train_eval_robocasa.sh 0 42 robocasa_sra       train false true              # flow + SRA
#
#   # --- 1task (PickPlaceCounterToCabinet) ---
#   bash scripts/train_eval_robocasa.sh 0 42 robocasa_1task     train true  false 1task
#   bash scripts/train_eval_robocasa.sh 0 42 robocasa_1task     train false false 1task       # flow only
#   bash scripts/train_eval_robocasa.sh 0 42 robocasa_1task     train false true  1task       # flow + SRA

set -euo pipefail

GPU="${1:-0}"
SEED="${2:-42}"
EXP_NAME="${3:-robocasa_multitask}"
MODE="${4:-train}"            # "train" or "eval"
USE_CONSISTENCY="${5:-true}"  # "true" or "false"
USE_SRA="${6:-false}"         # "true" or "false"
TASK_TYPE="${7:-multitask}"   # "multitask" or "1task"
EXTRA_ARGS=("${@:8}")         # additional hydra overrides, e.g. horizon=18 n_action_steps=16

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE="$(dirname "$SCRIPT_DIR")"

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
        --env MUJOCO_GL=osmesa \
        --env PYOPENGL_PLATFORM=osmesa \
        --env CUDA_VISIBLE_DEVICES="$GPU" \
        ${WANDB_API_KEY:+--env WANDB_API_KEY="$WANDB_API_KEY"} \
        "$SIF" \
        bash "$SCRIPT_DIR/train_eval_robocasa.sh" "$@"
fi

# Activate venv (same one used for robocasa installation)
VENV="$WORKSPACE/.venv"
if [ ! -d "$VENV" ]; then
    echo "ERROR: venv not found at $VENV. Run setup first."
    exit 1
fi
source "$VENV/bin/activate"

# MuJoCo headless rendering (both vars required; container sets PYOPENGL_PLATFORM=egl by default)
export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa
export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES="$GPU"

# robocasa editable install must be on PYTHONPATH
export PYTHONPATH="$WORKSPACE/robocasa:${PYTHONPATH:-}"

cd "$WORKSPACE/ManiFlow"

RUN_DIR="${RUN_DIR:-$WORKSPACE/ManiFlow/data/outputs/$(date +%Y.%m.%d)/$(date +%H.%M.%S)_${EXP_NAME}_seed${SEED}}"

if [[ "$TASK_TYPE" == "1task" ]]; then
    CONFIG_NAME="maniflow_image_timm_policy_robocasa_1task"
    TASK_NAME="robocasa_1task"
else
    CONFIG_NAME="maniflow_image_timm_policy_robocasa"
    TASK_NAME="robocasa_multitask"
fi

if [[ "$MODE" == "eval" ]]; then
    echo "=== Evaluation only mode (task_type=${TASK_TYPE}) ==="
    python -m maniflow.workspace.eval_maniflow_robocasa_workspace \
        --config-name "$CONFIG_NAME" \
        task="$TASK_NAME" \
        training.seed="$SEED" \
        training.device="cuda:0" \
        exp_name="$EXP_NAME" \
        "hydra.run.dir=$RUN_DIR" \
        "${EXTRA_ARGS[@]}"
else
    echo "=== Training mode (task_type=${TASK_TYPE}, use_consistency=${USE_CONSISTENCY}, use_sra=${USE_SRA}) ==="
    python -m maniflow.workspace.train_maniflow_robocasa_workspace \
        --config-name "$CONFIG_NAME" \
        task="$TASK_NAME" \
        training.seed="$SEED" \
        training.device="cuda:0" \
        exp_name="$EXP_NAME" \
        policy.use_consistency="$USE_CONSISTENCY" \
        policy.use_sra="$USE_SRA" \
        "hydra.run.dir=$RUN_DIR" \
        "${EXTRA_ARGS[@]}"
fi
