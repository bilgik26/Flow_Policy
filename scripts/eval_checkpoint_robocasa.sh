#!/usr/bin/env bash
# Evaluate a trained RoboCasa checkpoint and save videos locally.
#
# Usage:
#   bash scripts/eval_checkpoint_robocasa.sh [gpu_id] [training_dir] [eval_mode] [obj_split] [task_type]
#
# Arguments:
#   gpu_id       : GPU index (default: 1)
#   training_dir : Path to the training output directory that contains checkpoints/
#                  (default: most recent run under ManiFlow/data/outputs/)
#   eval_mode    : "latest" (epoch 500) or "best" (lowest val_loss) (default: latest)
#   obj_split    : "target" (unseen test objects) or "pretrain" (training objects) (default: target)
#   task_type    : "1task" or "multitask" (default: 1task)
#
# Output videos are saved to:
#   <training_dir>/eval_results/<epoch>/steps10_<obj_split>/videos/ep_NNN.mp4
#
# Examples:
#   # Evaluate latest checkpoint on test objects (default)
#   bash scripts/eval_checkpoint_robocasa.sh 1 \
#       ManiFlow/data/outputs/2026.07.03/19.52.03_robocasa_1task_seed42
#
#   # Evaluate latest checkpoint on training objects
#   bash scripts/eval_checkpoint_robocasa.sh 1 \
#       ManiFlow/data/outputs/2026.07.03/19.52.03_robocasa_1task_seed42 \
#       latest pretrain
#
#   # Evaluate best checkpoint on test objects
#   bash scripts/eval_checkpoint_robocasa.sh 1 \
#       ManiFlow/data/outputs/2026.07.03/19.52.03_robocasa_1task_seed42 \
#       best target

set -euo pipefail

GPU="${1:-1}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE="$(dirname "$SCRIPT_DIR")"

# Resolve training_dir: accept absolute or relative path
if [[ -n "${2:-}" ]]; then
    if [[ "${2}" = /* ]]; then
        TRAINING_DIR="${2}"
    else
        TRAINING_DIR="$WORKSPACE/${2}"
    fi
else
    # Default: most recent run directory
    TRAINING_DIR="$(ls -dt "$WORKSPACE/ManiFlow/data/outputs"/*/*/ 2>/dev/null | head -1)"
    if [[ -z "$TRAINING_DIR" ]]; then
        echo "ERROR: No training output found. Specify training_dir explicitly."
        exit 1
    fi
fi

EVAL_MODE="${3:-latest}"   # "latest" or "best"
OBJ_SPLIT="${4:-target}"   # "target" or "pretrain"
TASK_TYPE="${5:-1task}"    # "1task" or "multitask"

if [[ ! -d "$TRAINING_DIR/checkpoints" ]]; then
    echo "ERROR: No checkpoints/ found in: $TRAINING_DIR"
    exit 1
fi

# If not inside Singularity, re-invoke via the container (osmesa requires it)
if [[ -z "${SINGULARITY_CONTAINER:-}" ]]; then
    SIF="/mnt/data/bilgehan.sakai/singularity/sif/flow_policy_robocasa.sif"
    if [[ ! -f "$SIF" ]]; then
        echo "ERROR: Singularity image not found at $SIF"
        exit 1
    fi
    echo "=== Re-invoking inside Singularity container ==="
    exec singularity exec --nv \
        --bind "/home/bilgehan.sakai:/home/bilgehan.sakai" \
        --bind "/mnt/data/bilgehan.sakai:/mnt/data/bilgehan.sakai" \
        --env MUJOCO_GL=osmesa \
        --env PYOPENGL_PLATFORM=osmesa \
        --env CUDA_VISIBLE_DEVICES="$GPU" \
        ${WANDB_API_KEY:+--env WANDB_API_KEY="$WANDB_API_KEY"} \
        "$SIF" \
        bash "$SCRIPT_DIR/eval_checkpoint_robocasa.sh" "$@"
fi

# Activate venv
VENV="$WORKSPACE/.venv"
if [[ ! -d "$VENV" ]]; then
    echo "ERROR: venv not found at $VENV"
    exit 1
fi
source "$VENV/bin/activate"

export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa
export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES="$GPU"
export PYTHONPATH="$WORKSPACE/robocasa:${PYTHONPATH:-}"

if [[ "$TASK_TYPE" == "1task" ]]; then
    CONFIG_NAME="maniflow_image_timm_policy_robocasa_1task"
    TASK_NAME="robocasa_1task"
else
    CONFIG_NAME="maniflow_image_timm_policy_robocasa"
    TASK_NAME="robocasa_multitask"
fi

echo "=== RoboCasa Eval ==="
echo "  training_dir : $TRAINING_DIR"
echo "  eval_mode    : $EVAL_MODE  (latest=最終epoch, best=val_loss最小)"
echo "  obj_split    : $OBJ_SPLIT  (target=未知オブジェクト, pretrain=学習オブジェクト)"
echo "  task_type    : $TASK_TYPE"
echo "  GPU          : $GPU"
echo ""

cd "$WORKSPACE/ManiFlow"

python -m maniflow.workspace.eval_maniflow_robocasa_workspace \
    --config-name "$CONFIG_NAME" \
    task="$TASK_NAME" \
    training.device="cuda:0" \
    "+eval_mode=$EVAL_MODE" \
    "+eval_dir_tag=$OBJ_SPLIT" \
    "task.env_runner.obj_instance_split=$OBJ_SPLIT" \
    "hydra.run.dir=$TRAINING_DIR"
