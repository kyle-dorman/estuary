#!/bin/bash

# ---------------- USER CONFIG ----------------
EPOCHS=20
DEBUG=false

uv run --env-file .env scripts/train.py \
    data="/Users/kyledorman/data/estuary/dataset/labels.csv" \
    region_splits="/Users/kyledorman/data/estuary/dataset/region_splits.csv" \
    normalization_path="/Users/kyledorman/data/estuary/dataset/normalization/stats.json" \
    epochs=${EPOCHS} \
    debug=${DEBUG} \
    batch_size=128 \
    bands="RGB" \
    workers=4 \
    prefetch_factor=2 \
    model_type="TIMM" \
    model_name="convnext_tiny.dinov3_lvd1689m" \
    train_size=224 \
    val_size=224 \
    loss_fn="ce" \
    use_class_weights=false \
    blur_p=0.05 \
    erasing_p=0.05 \
    sharpness_p=0.05 \
    brightness_p=0.1 \
    contrast_p=0.1 \
    channel_shift_p=0.05 \
    gauss_p=0.05 \
    horizontal_flip_p=0.5 \
    vertical_flip_p=0.5 \
    rotation_p=0.1 \
    lr=5e-5 \
    min_lr_scale=5e-2 \
    # backbone_lr_scale=2e-1 \
    dropout=0.15 \
    drop_path=0.1 \
    ;