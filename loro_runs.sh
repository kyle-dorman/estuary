#!/usr/bin/env bash
# run_loro.sh – Leave-One-Region-Out fine-tune driver
#
# Usage:
#   ./run_loro.sh /path/to/checkpoint.ckpt
#
# You can edit the REGIONS array below to match your 15 regions.
# The script fine-tunes for 5 epochs on each fold.

set -euo pipefail

# ---------------- USER CONFIG ----------------
CHECKPOINT_PATH="${1:?please provide a checkpoint path}"
EPOCHS=5

# List of region identifiers (case-sensitive, must match CSV)
REGIONS=(
big_sur_river
little_sur
los_penasquitos_lagoon
navarro_river
pismo_creek_lagoon
russian_river
san_mateo_lagoon
)
# ---------------------------------------------

echo "Checkpoint: ${CHECKPOINT_PATH}"
echo "Epochs per fold: ${EPOCHS}"
echo "Regions (${#REGIONS[@]}): ${REGIONS[*]}"
echo

for REGION in "${REGIONS[@]}"; do
  echo "=== Fine-tuning, holding out ${REGION} ==="
  uv run --env-file .env scripts/run_loro.py \
    --checkpoint-path "${CHECKPOINT_PATH}" \
    --holdout-region   "${REGION}" \
    --epochs           "${EPOCHS}"
  echo
done