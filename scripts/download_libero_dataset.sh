#!/usr/bin/env bash
# Download raw LIBERO HDF5 demonstrations and convert them to the LeRobot
# format Flow_Policy's LiberoImageDataset expects.
#
# Usage (run inside the Singularity container / activated venv):
#   bash scripts/download_libero_dataset.sh [suite ...]
#
# With no arguments, converts the three standard 10-task suites:
#   libero_spatial libero_object libero_goal
# (libero_10 and libero_90 are NOT separately downloadable — LIBERO's own
# download script only offers "libero_spatial", "libero_object", "libero_goal"
# and "libero_100" (== libero_10 + libero_90 combined, ~9x bigger). Pass
# "libero_100" explicitly if you need libero_10 or libero_90 data; after
# extraction it creates libero_10/ and libero_90/ subdirectories directly
# under the download dir, same as the other suites.)
#
# Raw datasets are downloaded to:   Flow_Policy/libero/libero/libero/datasets/<suite>/
# Converted datasets are saved to:  Flow_Policy/data/libero/datasets/lerobot/<suite>/
#
# Verified against Lifelong-Robot-Learning/LIBERO as cloned 2026-07-22:
# `benchmark_scripts/download_libero_datasets.py` takes `--download-dir` (not
# `--save_dir`) and `--datasets {all,libero_goal,libero_spatial,libero_object,libero_100}`.
# `--use-huggingface` is REQUIRED for non-interactive runs — without it the
# script prompts "Download from original links may lead to failures... (y/n)"
# on stdin (the original box.com links may also be expired).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE="$(dirname "$SCRIPT_DIR")"
LIBERO_DIR="$WORKSPACE/libero"

SUITES=("$@")
if [[ ${#SUITES[@]} -eq 0 ]]; then
    SUITES=(libero_spatial libero_object libero_goal)
fi

if [[ ! -d "$LIBERO_DIR" ]]; then
    echo "ERROR: $LIBERO_DIR not found. Clone LIBERO first:"
    echo "  git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git $LIBERO_DIR"
    exit 1
fi

# NOTE: this is libero/libero/libero/datasets (three "libero" segments) —
# LIBERO's own get_libero_path("datasets") default, since the installed
# package root is $LIBERO_DIR/libero/libero/ (see docs/setup_and_train_libero.md
# Step 3-3 for the ~/.libero/config.yaml this must match).
RAW_BASE="$LIBERO_DIR/libero/libero/datasets"
TARGET_BASE="$WORKSPACE/data/libero/datasets/lerobot"
mkdir -p "$RAW_BASE" "$TARGET_BASE"

# download_libero_datasets.py maps "libero_10"/"libero_90" -> "libero_100"
declare -A DOWNLOAD_NAME=(
    [libero_spatial]=libero_spatial
    [libero_object]=libero_object
    [libero_goal]=libero_goal
    [libero_10]=libero_100
    [libero_90]=libero_100
    [libero_100]=libero_100
)

if [[ "${SKIP_DOWNLOAD:-0}" != "1" ]]; then
    DOWNLOADED=()
    for SUITE in "${SUITES[@]}"; do
        DL_NAME="${DOWNLOAD_NAME[$SUITE]:-$SUITE}"
        if [[ " ${DOWNLOADED[*]:-} " == *" $DL_NAME "* ]]; then
            continue  # libero_10 + libero_90 both map to libero_100; download once
        fi
        echo "=== Downloading raw HDF5 demos: $DL_NAME (for $SUITE) ==="
        python "$LIBERO_DIR/benchmark_scripts/download_libero_datasets.py" \
            --datasets "$DL_NAME" --download-dir "$RAW_BASE" --use-huggingface
        DOWNLOADED+=("$DL_NAME")
    done
fi

for SUITE in "${SUITES[@]}"; do
    echo "=== Converting to LeRobot format: $SUITE ==="
    python "$SCRIPT_DIR/convert_libero_to_lerobot.py" \
        --libero_task_suite "$SUITE" \
        --libero_raw_data_dir "$RAW_BASE/$SUITE" \
        --libero_target_dir "$TARGET_BASE/$SUITE"
done

echo "Done. Converted datasets are in: $TARGET_BASE/"
