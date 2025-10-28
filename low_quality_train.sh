#!/usr/bin/env bash

set -Eeuo pipefail

# ---------------- USER CONFIG ----------------
EPOCHS=1
DEBUG=true
BATCH_SIZE=128
WORKERS=4
PREFETCH=2

uv run --env-file .env scripts/train/train.py \
    data="/Volumes/x10pro/estuary/low_quality/quality_labels.csv" \
    normalization_path="/Users/kyledorman/data/estuary/dataset/normalization/stats.json" \
    epochs=${EPOCHS} \
    debug=${DEBUG} \
    batch_size=${BATCH_SIZE} \
    workers=${WORKERS} \
    prefetch_factor=${PREFETCH} \
    ;