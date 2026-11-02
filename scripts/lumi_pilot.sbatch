#!/bin/bash
#SBATCH --job-name=uvi-vl-pilot
#SBATCH --account=project_462001391
#SBATCH --partition=small-g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=8
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
# IMPORTANT: The partition/GPU count/account above are templates; verify with your LUMI allocation.
set -euo pipefail
: "${METADATA_DIR:?set METADATA_DIR}"
: "${IMAGE_ROOT:?set IMAGE_ROOT}"
: "${OUTPUT_JSONL:?set OUTPUT_JSONL}"
# Activate an existing LUMI ROCm-compatible environment/container before submission.
python scripts/check_rocm.py
python scripts/pilot.py --metadata "$METADATA_DIR" --image-root "$IMAGE_ROOT" --out "$OUTPUT_JSONL" --limit "${LIMIT:-1}"
