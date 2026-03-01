#!/usr/bin/env bash

set -Eeuo pipefail

# ---------------- USER CONFIG ----------------
DEBUG=false
BATCH_SIZE=16
WORKERS=4
PREFETCH=2
EPOCHS=50

# Eight band
#     bands="EIGHT" \
#     pretrained=false \
#     preview_channels=[7,5,3] \
#     drop_path=0.05 \
#     dropout=0.1 \
#     epochs=100 \
#     warmup_epochs=5 \
#     lr=3e-4 \

# Four band
#     bands="FOUR" \
#     pretrained=false \
#     preview_channels=[3,2,1] \
#     drop_path=0.05 \
#     dropout=0.1 \
#     epochs=100 \
#     warmup_epochs=5 \
#     lr=3e-4 \
# vit_small_patch16_dinov3.lvd1689m

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
    backbone_lr_scale=0.05 \
    lr=1e-4 \
	warmup_epochs=10 \
    aug_level="low" \
    weight_decay=0.05 \
    dropout=0.0 \
	drop_path=0.05 \
    precision="32-true" \
    smooth_factor=0.05 \
    img_size=224 \
    model_name="vit_small_patch8_224.dino" \
    ;
