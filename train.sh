#!/bin/bash

# ---------------- USER CONFIG ----------------
EPOCHS=1

# List of region identifiers (case-sensitive, must match CSV)
REGIONS=(
goleta
malibu_lagoon
ventura
topanga
)
# big_sur_river
# little_sur
# los_penasquitos_lagoon
# navarro_river
# pismo_creek_lagoon
# russian_river
# san_mateo_lagoon
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
    ;
echo

for REGION in "${REGIONS[@]}"; do
  echo "=== Holding out ${REGION} ==="
    uv run --env-file .env scripts/train_clay.py \
        data="/Users/kyledorman/data/estuary/label_studio/00025/labels.csv" \
        region_crops_json="/Users/kyledorman/data/estuary/label_studio/region_crops.json" \
        workers=4 \
        epochs=${EPOCHS} \
        holdout_region=${REGION} \
        ;
  echo
done