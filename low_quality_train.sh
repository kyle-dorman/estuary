#!/usr/bin/env bash

set -Eeuo pipefail

# ---------------- USER CONFIG ----------------
DEBUG=false
EPOCHS=5
FINE_TUNE_EPOCHS=10
BATCH_SIZE=128
WORKERS=4
PREFETCH=2

# find the most recent training run folder with a 'checkpoints/epoch*.ckpt'
LATEST_CKPT=$(find /Users/kyledorman/data/results/estuary/train/*/checkpoints/epoch*.ckpt -type f -print0 \
  | xargs -0 ls -t | head -n1)

echo "Using latest base model checkpoint: $LATEST_CKPT"

uv run --env-file .env scripts/train/train_low_quality.py \
    data="/Users/kyledorman/data/estuary/dataset/cluster_labels.csv" \
    val_data="/Users/kyledorman/data/estuary/dataset/time_series.csv" \
    test_data="/Users/kyledorman/data/estuary/dataset/time_series.csv" \
    normalization_path="/Users/kyledorman/data/estuary/dataset/normalization/stats.json" \
    epochs=${EPOCHS} \
    debug=${DEBUG} \
    batch_size=${BATCH_SIZE} \
    workers=${WORKERS} \
    prefetch_factor=${PREFETCH} \
    lr=3e-5 \
    freeze_encoder=true \
    encoder_checkpoint_path=$LATEST_CKPT \
    use_class_weights=false \
    split_method="dataset" \
    ;

# find the most recent training run folder with a 'checkpoints/last.ckpt'
LATEST_CKPT=$(find /Users/kyledorman/data/results/estuary_quality/train/*/checkpoints/last.ckpt -type f -print0 \
  | xargs -0 ls -t | head -n1)

echo "Using latest checkpoint: $LATEST_CKPT"

uv run --env-file .env scripts/train/train_low_quality.py \
    data="/Users/kyledorman/data/estuary/dataset/train.csv" \
    test_data="/Users/kyledorman/data/estuary/dataset/time_series.csv" \
    normalization_path="/Users/kyledorman/data/estuary/dataset/normalization/stats.json" \
    epochs=${FINE_TUNE_EPOCHS} \
    debug=${DEBUG} \
    batch_size=${BATCH_SIZE} \
    workers=${WORKERS} \
    prefetch_factor=${PREFETCH} \
    lr=1e-5 \
    checkpoint_path=$LATEST_CKPT \
    pct_low_quality=0.0 \
    freeze_encoder=False \
    warmup_epochs=0 \
    use_class_weights=true \
    split_method="yearly" \
    val_year=2021 \
    ;
