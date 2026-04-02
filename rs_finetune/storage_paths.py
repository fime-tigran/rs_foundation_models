from __future__ import annotations

import os
from typing import Final

_DATASET_CONFIG_KEYS: frozenset[str] = frozenset(
    {
        "root",
        "base_dir",
        "splits_dir",
        "dataset_path",
        "metadata_dir",
        "metadata_path",
    }
)


def _getenv(name: str, default: str) -> str:
    value: str | None = os.environ.get(name)
    if value is None or value == "":
        return default
    return value


DATASETS_ROOT: Final[str] = _getenv(
    "GEOCROSSBENCH_DATASETS_ROOT",
    "/mnt/weka/akhosrovyan/geocrossbench/datasets",
)
BASE_MODELS_ROOT: Final[str] = _getenv(
    "GEOCROSSBENCH_BASE_MODELS_ROOT",
    "/mnt/weka/akhosrovyan/geocrossbench/rs-base-models",
)
RESULTS_ROOT: Final[str] = _getenv(
    "RS_FOUNDATION_RESULTS_ROOT",
    "/mnt/weka/tgrigoryan/rs_foundation",
)


def datasets_path(*parts: str) -> str:
    return os.path.join(DATASETS_ROOT, *parts)


def base_models_path(*parts: str) -> str:
    return os.path.join(BASE_MODELS_ROOT, *parts)


def results_path(*parts: str) -> str:
    return os.path.join(RESULTS_ROOT, *parts)


def resolve_dataset_path_value(value: str) -> str:
    if value == "":
        return value
    if os.path.isabs(value):
        return value
    return os.path.join(DATASETS_ROOT, value)


def resolve_dataset_config_dict(data: dict[str, object]) -> dict[str, object]:
    out: dict[str, object] = dict(data)
    for key in _DATASET_CONFIG_KEYS:
        if key not in out:
            continue
        raw = out[key]
        if not isinstance(raw, str):
            continue
        out[key] = resolve_dataset_path_value(raw)
    return out
