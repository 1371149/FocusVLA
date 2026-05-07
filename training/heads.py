"""Scene-specific attention heads used by FocusVLA training.

The values below are the eight lowest-importance attention heads per layer,
precomputed from OpenVLA-OFT LIBERO head-importance matrices. Keeping the
resolved head ids in source avoids depending on local analysis artifacts at
training time.
"""

from __future__ import annotations

import os
import re
from typing import Dict, Optional, Sequence


NUM_SELECTED_HEADS = 8
DEFAULT_SCENE = "libero_object"

_HEADS_BY_SCENE = {
    "libero_10": [
        [20, 27, 28, 10, 26, 17, 13, 12],
        [20, 31, 10, 12, 9, 0, 28, 16],
        [17, 18, 19, 9, 4, 0, 12, 24],
        [5, 6, 14, 10, 12, 24, 11, 18],
        [28, 25, 29, 15, 6, 30, 18, 19],
        [11, 8, 14, 5, 9, 10, 25, 7],
        [31, 4, 23, 22, 10, 29, 25, 19],
        [10, 11, 18, 26, 16, 19, 3, 22],
        [2, 20, 12, 5, 18, 19, 31, 29],
        [1, 21, 9, 14, 0, 7, 22, 26],
        [22, 23, 20, 30, 28, 5, 26, 21],
        [27, 16, 25, 13, 30, 0, 1, 15],
        [31, 23, 17, 27, 29, 6, 3, 12],
        [26, 0, 18, 1, 17, 25, 15, 28],
        [10, 22, 16, 8, 7, 12, 14, 28],
        [15, 16, 31, 25, 28, 3, 4, 18],
        [23, 28, 12, 15, 11, 16, 7, 20],
        [24, 19, 15, 14, 23, 26, 28, 7],
        [7, 6, 26, 29, 4, 19, 8, 0],
        [5, 27, 0, 25, 24, 20, 9, 26],
        [15, 13, 31, 2, 18, 26, 4, 20],
        [0, 11, 22, 3, 20, 14, 15, 8],
        [11, 26, 21, 29, 14, 13, 7, 15],
        [4, 3, 30, 27, 23, 11, 24, 12],
        [23, 26, 14, 27, 21, 28, 19, 12],
        [19, 10, 30, 15, 29, 8, 27, 20],
        [28, 22, 8, 17, 18, 11, 24, 5],
        [16, 7, 10, 6, 21, 15, 3, 24],
        [23, 6, 16, 24, 11, 2, 9, 29],
        [0, 11, 12, 16, 6, 27, 23, 24],
        [2, 4, 27, 28, 10, 30, 12, 26],
        [2, 9, 14, 16, 13, 0, 27, 3],
    ],
    "libero_goal": [
        [20, 27, 28, 10, 12, 17, 16, 9],
        [12, 31, 9, 7, 10, 20, 6, 16],
        [9, 14, 27, 7, 28, 4, 18, 0],
        [5, 18, 6, 13, 12, 24, 10, 16],
        [25, 15, 18, 6, 19, 29, 8, 9],
        [8, 11, 9, 14, 7, 29, 10, 25],
        [31, 29, 23, 25, 4, 19, 10, 22],
        [18, 16, 26, 11, 27, 10, 19, 3],
        [2, 20, 12, 18, 5, 14, 19, 29],
        [1, 21, 14, 26, 0, 22, 9, 7],
        [22, 20, 30, 5, 23, 28, 11, 7],
        [13, 27, 16, 25, 30, 0, 1, 8],
        [17, 29, 23, 31, 6, 27, 3, 12],
        [26, 0, 28, 15, 1, 17, 18, 25],
        [10, 22, 16, 6, 21, 28, 8, 14],
        [15, 16, 0, 28, 3, 25, 18, 4],
        [23, 12, 16, 15, 0, 8, 10, 20],
        [24, 15, 26, 4, 19, 23, 11, 7],
        [7, 6, 26, 27, 29, 4, 0, 19],
        [5, 0, 26, 25, 3, 24, 27, 20],
        [15, 13, 4, 31, 14, 18, 7, 2],
        [0, 11, 25, 20, 22, 21, 23, 24],
        [11, 26, 21, 25, 1, 7, 0, 15],
        [3, 30, 24, 4, 10, 26, 27, 2],
        [23, 10, 21, 19, 12, 26, 0, 28],
        [19, 15, 10, 29, 5, 26, 30, 24],
        [4, 29, 20, 22, 8, 17, 5, 9],
        [7, 16, 2, 6, 21, 24, 15, 26],
        [23, 6, 24, 16, 2, 11, 8, 5],
        [0, 10, 11, 16, 27, 25, 4, 1],
        [28, 8, 0, 10, 20, 25, 30, 14],
        [0, 23, 2, 29, 3, 8, 7, 14],
    ],
    "libero_object": [
        [27, 20, 28, 2, 10, 26, 7, 11],
        [12, 20, 10, 30, 9, 6, 16, 28],
        [14, 17, 9, 24, 18, 19, 10, 0],
        [5, 24, 6, 14, 22, 10, 15, 31],
        [28, 25, 15, 18, 30, 9, 16, 8],
        [8, 9, 11, 14, 24, 18, 10, 26],
        [31, 23, 4, 19, 29, 25, 10, 3],
        [16, 18, 11, 26, 10, 27, 19, 22],
        [2, 20, 12, 18, 29, 31, 19, 5],
        [21, 14, 1, 9, 0, 5, 22, 2],
        [22, 30, 20, 23, 28, 5, 7, 15],
        [25, 27, 30, 16, 28, 13, 1, 23],
        [29, 23, 17, 31, 27, 6, 12, 14],
        [0, 18, 26, 25, 1, 17, 15, 14],
        [10, 16, 22, 14, 8, 7, 23, 6],
        [15, 16, 31, 0, 25, 3, 18, 28],
        [23, 28, 12, 7, 15, 8, 11, 16],
        [24, 19, 4, 26, 23, 14, 25, 28],
        [6, 26, 0, 7, 29, 4, 19, 2],
        [5, 27, 0, 16, 25, 24, 31, 9],
        [15, 13, 11, 14, 0, 2, 4, 7],
        [22, 11, 0, 24, 3, 12, 21, 23],
        [11, 26, 15, 7, 21, 25, 30, 1],
        [3, 11, 4, 24, 10, 23, 25, 26],
        [23, 14, 26, 27, 21, 2, 22, 19],
        [10, 2, 20, 19, 26, 8, 12, 15],
        [4, 16, 5, 22, 29, 8, 11, 20],
        [7, 16, 21, 24, 6, 30, 2, 12],
        [23, 6, 24, 2, 21, 11, 13, 16],
        [12, 0, 23, 1, 25, 3, 19, 4],
        [4, 27, 2, 0, 20, 6, 10, 22],
        [0, 27, 25, 3, 2, 23, 31, 13],
    ],
    "libero_spatial": [
        [20, 27, 10, 16, 12, 17, 28, 13],
        [31, 12, 9, 10, 28, 6, 19, 7],
        [14, 7, 12, 9, 8, 27, 28, 15],
        [5, 10, 18, 15, 24, 4, 29, 14],
        [28, 25, 15, 6, 31, 19, 8, 29],
        [8, 14, 9, 11, 25, 29, 26, 24],
        [31, 16, 4, 19, 25, 23, 22, 29],
        [11, 18, 26, 10, 16, 3, 19, 22],
        [2, 20, 18, 5, 31, 12, 19, 1],
        [21, 1, 14, 0, 26, 22, 7, 9],
        [22, 20, 23, 30, 28, 21, 26, 7],
        [13, 27, 16, 30, 25, 1, 28, 8],
        [23, 29, 17, 31, 27, 6, 3, 22],
        [0, 26, 18, 1, 15, 28, 25, 17],
        [10, 16, 8, 28, 14, 21, 22, 9],
        [15, 16, 28, 3, 4, 25, 18, 31],
        [23, 28, 12, 15, 11, 24, 8, 16],
        [24, 25, 15, 19, 23, 4, 7, 26],
        [6, 26, 7, 0, 29, 4, 27, 8],
        [5, 27, 0, 17, 26, 16, 20, 9],
        [4, 15, 14, 13, 31, 18, 2, 11],
        [0, 22, 11, 3, 15, 9, 20, 12],
        [11, 26, 25, 21, 7, 15, 14, 0],
        [4, 24, 11, 3, 10, 23, 27, 30],
        [23, 10, 2, 22, 9, 27, 14, 21],
        [19, 29, 30, 5, 12, 2, 15, 10],
        [20, 29, 4, 22, 25, 17, 28, 14],
        [3, 21, 12, 24, 28, 10, 5, 1],
        [23, 6, 4, 13, 2, 5, 9, 11],
        [0, 23, 24, 16, 4, 25, 6, 14],
        [12, 2, 23, 26, 10, 22, 7, 19],
        [0, 23, 21, 2, 29, 16, 9, 3],
    ],
}


def _normalize_scene_text(value: object) -> str:
    text = str(value).lower()
    text = text.replace("-", "_")
    return re.sub(r"[^a-z0-9]+", "_", text)


def infer_libero_scene(*scene_hints: object, default: Optional[str] = None) -> str:
    """Infer the LIBERO suite from dataset/model/output names."""

    for hint in scene_hints:
        if hint is None:
            continue
        text = _normalize_scene_text(hint)
        if not text:
            continue

        if "libero_10" in text or "libero10" in text or re.search(r"(^|_)10(_|$)", text):
            return "libero_10"
        for suite in ("spatial", "goal", "object"):
            if f"libero_{suite}" in text or re.search(rf"(^|_){suite}(_|$)", text):
                return f"libero_{suite}"

    if default is not None:
        if default not in _HEADS_BY_SCENE:
            raise ValueError(f"Unknown default LIBERO scene: {default}")
        return default

    raise ValueError(
        "Could not infer LIBERO scene. Expected one of: "
        + ", ".join(sorted(_HEADS_BY_SCENE.keys()))
    )


def get_heads_to_finetune(
    *scene_hints: object,
    num_heads: int = NUM_SELECTED_HEADS,
    default_scene: Optional[str] = DEFAULT_SCENE,
) -> Dict[int, list[int]]:
    """Return the lowest-importance heads for the inferred LIBERO scene."""

    if num_heads <= 0:
        raise ValueError("num_heads must be positive")
    if num_heads > NUM_SELECTED_HEADS:
        raise ValueError(f"Only the lowest {NUM_SELECTED_HEADS} heads are embedded, got {num_heads}.")

    scene = infer_libero_scene(*scene_hints, default=default_scene)
    rows: Sequence[Sequence[int]] = _HEADS_BY_SCENE[scene]
    return {layer_idx: list(row[:num_heads]) for layer_idx, row in enumerate(rows)}


def set_heads_to_finetune(
    *scene_hints: object,
    num_heads: int = NUM_SELECTED_HEADS,
    default_scene: Optional[str] = DEFAULT_SCENE,
) -> Dict[int, list[int]]:
    """Update the legacy global dict in place and return it."""

    global SELECTED_SCENE
    SELECTED_SCENE = infer_libero_scene(*scene_hints, default=default_scene)
    selected = get_heads_to_finetune(SELECTED_SCENE, num_heads=num_heads, default_scene=SELECTED_SCENE)
    HEADS_TO_FINETUNE.clear()
    HEADS_TO_FINETUNE.update(selected)
    return HEADS_TO_FINETUNE


def summarize_heads(heads: Dict[int, list[int]]) -> str:
    if not heads:
        return "0 layers"
    first_layer = min(heads)
    return f"{len(heads)} layers x {len(heads[first_layer])} heads; layer0={heads[first_layer]}"


SELECTED_SCENE = infer_libero_scene(os.environ.get("FOCUSVLA_LIBERO_SCENE"), default=DEFAULT_SCENE)
HEADS_TO_FINETUNE = get_heads_to_finetune(SELECTED_SCENE, default_scene=SELECTED_SCENE)
