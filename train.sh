#!/bin/bash

# ---------------- USER CONFIG ----------------
EPOCHS=1
DEBUG=true
patience=20

echo "=== Training base model ==="
uv run --env-file .env scripts/train.py \
    data="/Users/kyledorman/data/estuary/dataset/labels.csv" \
    region_splits="/Users/kyledorman/data/estuary/dataset/region_splits.csv" \
    normalization_apth="/Users/kyledorman/data/estuary/data/normalization/stats.json" \
    bands="rgb" \
    workers=0 \
    epochs=${EPOCHS} \
    debug=${DEBUG} \
    patience=${patience} \
    dropout=0.2 \
    weight_decay=2e-4 \
    loss_fn="ce" \
    use_class_weights=true \
    model_type="timm" \
    model_name="mobilenetv4_conv_small_050.e3000_r224_in1k" \
    ;
echo