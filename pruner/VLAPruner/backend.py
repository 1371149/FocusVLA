"""Backend loading helpers for VLA-Pruner evaluation."""

import importlib
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Optional


DEFAULT_BACKEND_ROOT = Path(__file__).resolve().parents[2]


def resolve_backend_root(backend_root: Optional[str] = None) -> Path:
    candidate = backend_root or os.environ.get("VLAPRUNER_BACKEND_ROOT") or str(DEFAULT_BACKEND_ROOT)
    root = Path(candidate).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"VLA-Pruner backend root does not exist: {root}")
    return root


def ensure_backend_root_on_path(backend_root: Optional[str] = None) -> Path:
    root = resolve_backend_root(backend_root)
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    return root


def load_backend_modules(backend_root: Optional[str] = None) -> SimpleNamespace:
    root = ensure_backend_root_on_path(backend_root)
    return SimpleNamespace(
        backend_root=root,
        base_eval=importlib.import_module("experiments.robot.libero.run_libero_eval"),
        libero_utils=importlib.import_module("experiments.robot.libero.libero_utils"),
        openvla_utils=importlib.import_module("experiments.robot.openvla_utils"),
        robot_utils=importlib.import_module("experiments.robot.robot_utils"),
    )
