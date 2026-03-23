#!/usr/bin/env bash
set -e
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV:-rsfm}"
cd "$(dirname "$0")"
KMP_DUPLICATE_LIB_OK=TRUE python -m pytest tests -q "$@"
