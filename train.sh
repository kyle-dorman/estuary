#!/usr/bin/env bash

set -Eeuo pipefail

# ---------------- USER CONFIG ----------------
EPOCHS=25
DEBUG=false
BATCH_SIZE=128
WORKERS=4
PREFETCH=2

# region_splits="/Users/kyledorman/data/estuary/dataset/region_splits.csv" \

uv run --env-file .env scripts/train/train.py \
    data="/Users/kyledorman/data/estuary/dataset/labels.csv" \
    normalization_path="/Users/kyledorman/data/estuary/dataset/normalization/stats.json" \
    epochs=${EPOCHS} \
    debug=${DEBUG} \
    batch_size=${BATCH_SIZE} \
    workers=${WORKERS} \
    prefetch_factor=${PREFETCH} \
    split_method="yearly" \
    val_year=2022 \
    test_year=2024 \
    perch_smooth_factor=0.4 \
    ;