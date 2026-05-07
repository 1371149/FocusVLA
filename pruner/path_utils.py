"""Helpers for remapping legacy absolute paths after moving the repo across machines."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable


CURRENT_REPO_ROOT = Path(__file__).resolve().parents[1]
CURRENT_ROBOT_ROOT = CURRENT_REPO_ROOT.parent
LEGACY_ROBOT_ROOTS = ()


def normalize_legacy_robot_path(value):
    if value is None:
        return value

    raw_value = str(value).strip()
    if not raw_value or raw_value.lower() in {"default", "none", "null"}:
        return value
    if not raw_value.startswith("/"):
        return value

    current_robot_root = str(CURRENT_ROBOT_ROOT)
    if raw_value == current_robot_root or raw_value.startswith(f"{current_robot_root}{os.sep}"):
        return raw_value

    for legacy_root in LEGACY_ROBOT_ROOTS:
        if raw_value == legacy_root or raw_value.startswith(f"{legacy_root}{os.sep}"):
            remapped = f"{current_robot_root}{raw_value[len(legacy_root):]}"
            if Path(remapped).exists() or not Path(raw_value).exists():
                return remapped

    return raw_value


def normalize_cfg_paths(cfg, field_names: Iterable[str]) -> None:
    for field_name in field_names:
        if not hasattr(cfg, field_name):
            continue
        setattr(cfg, field_name, normalize_legacy_robot_path(getattr(cfg, field_name)))
