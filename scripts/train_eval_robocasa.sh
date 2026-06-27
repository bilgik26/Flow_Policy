#!/usr/bin/env bash
# Train and evaluate Flow_Policy on RoboCasa tasks.
#
# Usage:
#   bash scripts/train_eval_robocasa.sh [gpu_id] [seed] [exp_name] [mode] [use_consistency] [use_sra]
#
# Examples:
#   bash scripts/train_eval_robocasa.sh 0 42 robocasa_10tasks
#   bash scripts/train_eval_robocasa.sh 0 42 robocasa_10tasks eval            # eval only
#   bash scripts/train_eval_robocasa.sh 0 42 robocasa_flow train false        # flow matching only
#   bash scripts/train_eval_robocasa.sh 0 42 robocasa_sra   train false true  # flow + SRA

set -euo pipefail

GPU="${1:-0}"
SEED="${2:-42}"
EXP_NAME="${3:-robocasa_10tasks}"
MODE="${4:-train}"           # "train" or "eval"
USE_CONSISTENCY="${5:-true}"  # "true" or "false"
USE_SRA="${6:-false}"         # "true" or "false"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE="$(dirname "$SCRIPT_DIR")"

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

RUN_DIR="$WORKSPACE/ManiFlow/data/outputs/$(date +%Y.%m.%d)/$(date +%H.%M.%S)_${EXP_NAME}_seed${SEED}"

if [[ "$MODE" == "eval" ]]; then
    echo "=== Evaluation only mode ==="
    python -m maniflow.workspace.eval_maniflow_robocasa_workspace \
        --config-name maniflow_image_timm_policy_robocasa \
        task=robocasa_multitask \
        training.seed="$SEED" \
        training.device="cuda:0" \
        exp_name="$EXP_NAME" \
        "hydra.run.dir=$RUN_DIR"
else
    echo "=== Training mode (use_consistency=${USE_CONSISTENCY}, use_sra=${USE_SRA}) ==="
    python -m maniflow.workspace.train_maniflow_robocasa_workspace \
        --config-name maniflow_image_timm_policy_robocasa \
        task=robocasa_multitask \
        training.seed="$SEED" \
        training.device="cuda:0" \
        exp_name="$EXP_NAME" \
        policy.use_consistency="$USE_CONSISTENCY" \
        policy.use_sra="$USE_SRA" \
        "hydra.run.dir=$RUN_DIR"
fi
