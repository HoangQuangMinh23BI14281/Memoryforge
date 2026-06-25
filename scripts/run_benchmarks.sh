#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH="${PYTHONPATH:-.}"
uv run python benchmarks/longmemeval_benchmark.py "$@"
