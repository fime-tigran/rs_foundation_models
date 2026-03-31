#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
uv venv -p 3.11
uv pip install --python .venv/bin/python .
