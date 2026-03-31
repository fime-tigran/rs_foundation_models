#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
REPO_ROOT="$(cd .. && pwd)"
PY="${REPO_ROOT}/.venv/bin/python"
if [[ ! -x "${PY}" ]]; then
  echo "Run once from repo root: ./bootstrap_env.sh" >&2
  exit 1
fi
KMP_DUPLICATE_LIB_OK=TRUE exec "${PY}" -m pytest tests -q "$@"
