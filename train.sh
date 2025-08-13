#!/bin/bash

# uv run --env-file .env scripts/train_clay.py \
#     data="/Users/kyledorman/data/estuary/label_studio/00025/labels.csv" \
#     region_crops_json="/Users/kyledorman/data/estuary/label_studio/region_crops.json" \
#     workers=4 \
#     epochs=40 \
#     decoder_name="transformer" \
#     decoder_depth=3 \
#     lr=5e-4 \
#     init_lr=1e-4 \
#     min_lr=5e-6 \
#     classes="['open','closed']" \
#     ;

uv run --env-file .env scripts/train_clay.py \
    data="/Users/kyledorman/data/estuary/label_studio/00025/labels.csv" \
    region_crops_json="/Users/kyledorman/data/estuary/label_studio/region_crops.json" \
    workers=4 \
    epochs=40 \
    classes="['open','closed']" \
    decoder_dim=192 \
    decoder_depth=4 \
    ;

uv run --env-file .env scripts/train_clay.py \
    data="/Users/kyledorman/data/estuary/label_studio/00025/labels.csv" \
    region_crops_json="/Users/kyledorman/data/estuary/label_studio/region_crops.json" \
    workers=4 \
    epochs=40 \
    classes="['open','closed']" \
    decoder_depth=4 \
    ;
