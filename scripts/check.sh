#!/usr/bin/env bash
set -euo pipefail

PYTHONPATH=src uv run --no-sync --extra dev --extra experiments ruff check \
  src experiments scripts tests
PYTHONPATH=src uv run --no-sync --extra dev --extra experiments pytest -q
