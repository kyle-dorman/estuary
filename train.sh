#!/usr/bin/env bash

set -Eeuo pipefail

# ---------------- USER CONFIG ----------------
DEBUG=false
BATCH_SIZE=64
WORKERS=4
PREFETCH=2
EPOCHS=25

uv run --env-file .env scripts/train/train.py \
    data="/Users/kyledorman/data/estuary/dataset/train.csv" \
    normalization_path="/Users/kyledorman/data/estuary/dataset/normalization_no_power/stats.json" \
    epochs=${EPOCHS} \
    debug=${DEBUG} \
    batch_size=${BATCH_SIZE} \
    base_lr_batch_size=${BATCH_SIZE} \
    workers=${WORKERS} \
    prefetch_factor=${PREFETCH} \
    val_year=2021 \
    test_year=2025 \
    use_class_weights=true \
    backbone_lr_scale=0.2 \
    aug_level="high" \
    classes="['closed','open']" \
    bands="RGB" \
    ;