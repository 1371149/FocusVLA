"""Thin wrapper around the shared preset-based mug evaluation helper."""

from __future__ import annotations

import importlib.util
from pathlib import Path


SHARED_MUG_EVAL = Path(__file__).resolve().parents[1] / "ADP" / "mug_eval.py"
SPEC = importlib.util.spec_from_file_location("_specprune_shared_mug_eval", SHARED_MUG_EVAL)
if SPEC is None or SPEC.loader is None:
    raise ImportError(f"Failed to load shared mug_eval module from {SHARED_MUG_EVAL}")

MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)

MugEvalEnvWrapper = MODULE.MugEvalEnvWrapper
describe_eval_mug_spec = MODULE.describe_eval_mug_spec
get_libero_env_with_optional_mug = MODULE.get_libero_env_with_optional_mug
resolve_eval_mug_spec = MODULE.resolve_eval_mug_spec

__all__ = [
    "MugEvalEnvWrapper",
    "describe_eval_mug_spec",
    "get_libero_env_with_optional_mug",
    "resolve_eval_mug_spec",
]
