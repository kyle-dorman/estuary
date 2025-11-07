#!/usr/bin/env bash

set -Eeuo pipefail

# ---------------- USER CONFIG ----------------
DEBUG=false
BATCH_SIZE=128
WORKERS=4
PREFETCH=2

uv run --env-file .env scripts/train/train.py \
    data="/Users/kyledorman/data/estuary/dataset/train_dataset.csv" \
    normalization_path="/Users/kyledorman/data/estuary/dataset/normalization/stats.json" \
    epochs=25 \
    debug=${DEBUG} \
    batch_size=${BATCH_SIZE} \
    workers=${WORKERS} \
    prefetch_factor=${PREFETCH} \
    split_method="yearly" \
    val_year=2021 \
    test_year=2024 \
    perch_smooth_factor=0.4 \
    ;

uv run --env-file .env scripts/train/train.py \
    data="/Users/kyledorman/data/estuary/dataset/train_low_quality.csv" \
    normalization_path="/Users/kyledorman/data/estuary/dataset/normalization/stats.json" \
    epochs=20 \
    debug=${DEBUG} \
    batch_size=${BATCH_SIZE} \
    workers=${WORKERS} \
    prefetch_factor=${PREFETCH} \
    split_method="yearly" \
    val_year=2021 \
    test_year=2024 \
    perch_smooth_factor=0.4 \
    ;