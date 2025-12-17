#!/usr/bin/env bash

set -Eeuo pipefail

# ---------------- USER CONFIG ----------------
DEBUG=false
BATCH_SIZE=32
WORKERS=4
PREFETCH=2

# example
# lr=1e-4
# backbone_lr_scale=0.1
# drop_path=0.2

uv run --env-file .env scripts/train/train.py \
    data="/Users/kyledorman/data/estuary/dataset/open_closed.csv" \
    normalization_path="/Users/kyledorman/data/estuary/dataset/normalization/stats.json" \
    epochs=25 \
    debug=${DEBUG} \
    batch_size=${BATCH_SIZE} \
    base_lr_batch_size=${BATCH_SIZE} \
    workers=${WORKERS} \
    prefetch_factor=${PREFETCH} \
    split_method="yearly" \
    val_year=2021 \
    test_year=2024 \
    use_class_weights=true \
    aug_level="split" \
    train_size=384 \
    val_size=384 \
    model_name="convnext_tiny.in12k" \
    ;
