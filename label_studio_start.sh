#!/usr/bin/env bash

set -Eeuo pipefail

SQLITE_BUSY_TIMEOUT=30000 uv run --env-file .env label-studio start
