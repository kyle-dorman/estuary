#!/bin/bash

# ---------------- USER CONFIG ----------------
EPOCHS=40
DEBUG=false
patience=20

# List of region identifiers (case-sensitive, must match CSV)
REGIONS=(
# goleta
# malibu_lagoon
# ventura
topanga
big_sur_river
carmel
little_sur
los_penasquitos_lagoon
navarro_river
pismo_creek_lagoon
russian_river
san_dieguito_lagoon
san_elijo_lagoon
san_mateo_lagoon
santa_margarita
)
# ---------------------------------------------

echo "Epochs per fold: ${EPOCHS}"
echo "Regions (${#REGIONS[@]}): ${REGIONS[*]}"
echo

echo "=== Training base model ==="
uv run --env-file .env scripts/train_clay.py \
    data="/Users/kyledorman/data/estuary/label_studio/00025/labels.csv" \
    region_crops_json="/Users/kyledorman/data/estuary/label_studio/region_crops.json" \
    workers=4 \
    epochs=${EPOCHS} \
    debug=${DEBUG} \
    patience=${patience} \
    dropout=0.2 \
    weight_decay=2e-4 \
    loss_fn="focal" \
    use_class_weights=false \
    ;
echo

# for REGION in "${REGIONS[@]}"; do
#   echo "=== Holding out ${REGION} ==="
#     uv run --env-file .env scripts/train_clay.py \
#         data="/Users/kyledorman/data/estuary/label_studio/00025/labels.csv" \
#         region_crops_json="/Users/kyledorman/data/estuary/label_studio/region_crops.json" \
#         workers=4 \
#         epochs=${EPOCHS} \
#         debug=${DEBUG} \
#         patience=${patience} \
#         holdout_region=${REGION} \
#         dropout=0.2 \
#         weight_decay=2e-4 \
#         loss_fn="focal" \
#         use_class_weights=false \
#         ;
#   echo
# done