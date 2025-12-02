#!/bin/bash

uv run --env-file .env tensorboard \
    --logdir /Users/kyledorman/data/results/estuary/train \
    --port 6006 \
    ;
