#!/bin/bash

# Check
uv run ruff check --fix

# format
uv run ruff format
