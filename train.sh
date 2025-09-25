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
    model_name="mobilenetv4_conv_small_050.e3000_r224_in1k" \
    loss_fn="ce" \
    use_class_weights=false \
    drop_path=0.0 \
    dropout=0.0 \
    channel_shift_p=0.0 \
    blur_p=0.0 \
    gauss_p=0.0 \
    erasing_p=0.0 \
    sharpness_p=0.0 \
    brightness_p=0.0 \
    contrast_p=0.0 \
    rotation_p=0.0 \
    vertical_flip_p=0.0 \
    horizontal_flip_p=0.0 \
    ;
echo