#!/usr/bin/env bash
# Evaluate a trained LIBERO checkpoint and save videos locally.
#
# Usage:
#   bash scripts/eval_checkpoint_libero.sh [gpu_id] [training_dir] [eval_mode] [task_suite] [full_suite]
#
# Arguments:
#   gpu_id       : GPU index (default: 1)
#   training_dir : Path to the training output directory that contains checkpoints/
#                  (default: most recent run under ManiFlow/data/outputs/)
#   eval_mode    : "latest" or "best" (lowest val_loss) (default: latest)
#   task_suite   : task config name used at training time, e.g. "libero_spatial" (default: libero_spatial)
#   full_suite   : "true" to evaluate every task in the suite (task_ids=null, official
#                  LIBERO protocol), "false" to only run the single canonical task
#                  used for periodic training rollouts (default: false)
#
# Env vars (optional):
#   TASK_IDS           : explicit Hydra list override, e.g. "[7,9]", to eval specific
#                         task indices instead of the canonical task / full suite.
#   EPISODES_PER_TASK   : episodes per task when TASK_IDS is set (default: 50)
#
# Output videos are saved to:
#   <training_dir>/eval_results/<epoch>/steps10_<tag>/videos/taskNN_epNNN.mp4
#
# Example:
#   bash scripts/eval_checkpoint_libero.sh 1 \
#       ManiFlow/data/outputs/2026.07.22/10.00.00_libero_spatial_seed42 \
#       best libero_spatial true

set -euo pipefail

GPU="${1:-1}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE="$(dirname "$SCRIPT_DIR")"

if [[ -n "${2:-}" ]]; then
    if [[ "${2}" = /* ]]; then
        TRAINING_DIR="${2}"
    else
        TRAINING_DIR="$WORKSPACE/${2}"
    fi
else
    TRAINING_DIR="$(ls -dt "$WORKSPACE/ManiFlow/data/outputs"/*/*/ 2>/dev/null | head -1)"
    if [[ -z "$TRAINING_DIR" ]]; then
        echo "ERROR: No training output found. Specify training_dir explicitly."
        exit 1
    fi
fi

EVAL_MODE="${3:-latest}"
TASK_SUITE="${4:-libero_spatial}"
FULL_SUITE="${5:-false}"

if [[ ! -d "$TRAINING_DIR/checkpoints" ]]; then
    echo "ERROR: No checkpoints/ found in: $TRAINING_DIR"
    exit 1
fi

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
        bash "$SCRIPT_DIR/eval_checkpoint_libero.sh" "$@"
fi

VENV="$WORKSPACE/.venv-libero"
if [[ ! -d "$VENV" ]]; then
    echo "ERROR: venv not found at $VENV"
    exit 1
fi
source "$VENV/bin/activate"

export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa
export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES="$GPU"

TAG="canonical_task"
FULL_SUITE_OVERRIDE=()
if [[ "$FULL_SUITE" == "true" ]]; then
    TAG="full_suite"
    FULL_SUITE_OVERRIDE=("task.env_runner.task_ids=null" "task.env_runner.eval_episodes_per_task=50")
elif [[ -n "${TASK_IDS:-}" ]]; then
    TASK_IDS_CLEAN="${TASK_IDS//[\[\], ]/_}"
    TASK_IDS_CLEAN="$(echo "$TASK_IDS_CLEAN" | sed -e 's/^_*//' -e 's/_*$//')"
    TAG="tasks_${TASK_IDS_CLEAN}"
    FULL_SUITE_OVERRIDE=("task.env_runner.task_ids=${TASK_IDS}" "task.env_runner.eval_episodes_per_task=${EPISODES_PER_TASK:-50}")
fi

echo "=== LIBERO Eval ==="
echo "  training_dir : $TRAINING_DIR"
echo "  eval_mode    : $EVAL_MODE  (latest=最終epoch, best=val_loss最小)"
echo "  task_suite   : $TASK_SUITE"
echo "  full_suite   : $FULL_SUITE  (true=全タスク50試行/false=学習中と同じ代表タスクのみ)"
if [[ -n "${TASK_IDS:-}" ]]; then
    echo "  task_ids     : $TASK_IDS  (episodes_per_task=${EPISODES_PER_TASK:-50})"
fi
echo "  GPU          : $GPU"
echo ""

cd "$WORKSPACE/ManiFlow"

python -m maniflow.workspace.eval_maniflow_libero_workspace \
    --config-name maniflow_image_timm_policy_libero \
    task="$TASK_SUITE" \
    training.device="cuda:0" \
    "+eval_mode=$EVAL_MODE" \
    "+eval_dir_tag=$TAG" \
    "hydra.run.dir=$TRAINING_DIR" \
    "${FULL_SUITE_OVERRIDE[@]}"
