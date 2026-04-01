#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
if [ "${1:-}" = "--fresh" ]; then
  rm -rf .venv
fi
uv sync --python 3.11
