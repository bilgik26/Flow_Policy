#!/usr/bin/env bash
# Download RoboCasa datasets (LeRobot format) inside the Singularity container.
#
# Usage (from Flow_Policy workspace):
#   singularity exec --nv --bind ... flow_policy_robocasa.sif \
#       bash /workspace/Flow_Policy/scripts/download_robocasa_dataset.sh
#
# Downloaded datasets are stored under:
#   Flow_Policy/data/robocasa/datasets/v1.0/pretrain/atomic/<TaskName>/
#
# Default: 10 representative tasks for initial training.
# To download all 65 ATOMIC tasks, set ALL_TASKS=1.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE="$(dirname "$SCRIPT_DIR")"
VENV="$WORKSPACE/../cosmos-policy/.venv"  # reuse cosmos-policy venv that has robocasa

if [ ! -d "$VENV" ]; then
    echo "ERROR: venv not found at $VENV"
    echo "Run the robocasa installation steps first (see docs/setup_and_train_robocasa.md)."
    exit 1
fi

source "$VENV/bin/activate"

export NEW_ROBOCASA_DS_BASE="$WORKSPACE/data/robocasa/datasets"

TASKS=(
    CloseBlenderLid
    CloseFridge
    OpenCabinet
    OpenDrawer
    OpenStandMixerHead
    PickPlaceCounterToCabinet
    PickPlaceCounterToStove
    PickPlaceDrawerToCounter
    PickPlaceSinkToCounter
    PickPlaceToasterToCounter
)

if [[ "${ALL_TASKS:-0}" == "1" ]]; then
    echo "Downloading ALL ATOMIC tasks (65 tasks, ~100GB) ..."
    python -c "
from robocasa.scripts.download_datasets import download_datasets
download_datasets(split=['pretrain'], source=['human'], overwrite=False,
                  output_dir='$NEW_ROBOCASA_DS_BASE')
"
else
    echo "Downloading 10 representative tasks ..."
    for TASK in "${TASKS[@]}"; do
        echo "=== $TASK ==="
        python -c "
from robocasa.scripts.download_datasets import download_datasets
download_datasets(split=['pretrain'], tasks=['$TASK'], source=['human'], overwrite=False,
                  output_dir='$NEW_ROBOCASA_DS_BASE')
"
    done
fi

echo "Done. Datasets are in: $NEW_ROBOCASA_DS_BASE/v1.0/pretrain/atomic/"
