#!/usr/bin/env bash

set -Eeuo pipefail

# ---------------- USER CONFIG ----------------
DEBUG=false
BATCH_SIZE=128
WORKERS=4
PREFETCH=2

# example
# lr=1e-4
# backbone_lr_scale=0.1
# drop_path=0.2
uv run --env-file .env scripts/train/train.py \
    data="/Users/kyledorman/data/estuary/dataset/open_closed.csv" \
    normalization_path="/Users/kyledorman/data/estuary/dataset/normalization/stats.json" \
    epochs=20 \
    debug=${DEBUG} \
    batch_size=${BATCH_SIZE} \
    workers=${WORKERS} \
    prefetch_factor=${PREFETCH} \
    split_method="yearly" \
    val_year=2021 \
    test_year=2024 \
    use_class_weights=true \
    aug_level="split" \
    model_name="convnext_small.dinov3_lvd1689m"
    ;
