#!/usr/bin/env bash
set -euo pipefail

uv sync --extra dev
echo "Run: uv run pytest -q"
