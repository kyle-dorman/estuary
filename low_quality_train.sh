#!/usr/bin/env bash

set -Eeuo pipefail

# ---------------- USER CONFIG ----------------
EPOCHS=6
DEBUG=false
BATCH_SIZE=128
WORKERS=4
PREFETCH=2

uv run --env-file .env scripts/train/train_low_quality.py \
    data="/Users/kyledorman/data/estuary/dataset/quality_labels.csv" \
    test_data="/Users/kyledorman/data/estuary/dataset/time_series.csv" \
    val_data="/Users/kyledorman/data/estuary/dataset/low_quality.csv" \
    normalization_path="/Users/kyledorman/data/estuary/dataset/normalization/stats.json" \
    epochs=${EPOCHS} \
    debug=${DEBUG} \
    batch_size=${BATCH_SIZE} \
    workers=${WORKERS} \
    prefetch_factor=${PREFETCH} \
    lr=3e-5 \
    freeze_encoder=true \
    ;

uv run --env-file .env scripts/train/train_low_quality.py \
    data="/Users/kyledorman/data/estuary/dataset/quality_labels.csv" \
    test_data="/Users/kyledorman/data/estuary/dataset/time_series.csv" \
    val_data="/Users/kyledorman/data/estuary/dataset/low_quality.csv" \
    normalization_path="/Users/kyledorman/data/estuary/dataset/normalization/stats.json" \
    epochs=${EPOCHS} \
    debug=${DEBUG} \
    batch_size=${BATCH_SIZE} \
    workers=${WORKERS} \
    prefetch_factor=${PREFETCH} \
    lr=3e-5 \
    freeze_encoder=true \
    encoder_checkpoint_path="/Users/kyledorman/data/results/estuary/train/20251106-154000/checkpoints/last.ckpt" \
    ;
