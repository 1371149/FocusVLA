"""Debug visualization helpers for SP pruning."""

import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image


def get_debug_output_dir(cfg: Any) -> Path:
    base_dir = Path(getattr(cfg, "local_log_dir", ".")).expanduser().resolve() / "debug_attention_pruning"
    run_label = _resolve_run_label(cfg)
    return base_dir / run_label if run_label else base_dir


def maybe_save_first_pruned_image(
    model: Any,
    cfg: Any,
    raw_image: np.ndarray,
    important_idx: Iterable[int],
    num_tokens: int,
    image_tag: str,
) -> None:
    if not bool(getattr(cfg, "sp_save_first_pruned_image", False)):
        return

    target = getattr(model, "_sp_owner", model)

    saved_tags = getattr(target, "_sp_debug_saved_tags", None)
    if saved_tags is None:
        saved_tags = set()
        target._sp_debug_saved_tags = saved_tags
    if image_tag in saved_tags:
        return

    if raw_image is None or num_tokens <= 0:
        return

    important_idx = np.asarray(list(important_idx), dtype=np.int64)
    important_idx = important_idx[(important_idx >= 0) & (important_idx < num_tokens)]

    grid_size = int(round(math.sqrt(num_tokens)))
    if grid_size * grid_size != num_tokens:
        return

    image = _to_rgb_ndarray(raw_image)
    keep_mask = np.zeros(num_tokens, dtype=np.uint8)
    keep_mask[important_idx] = 1
    if int(keep_mask.sum()) >= int(num_tokens):
        return
    output_dir = get_debug_output_dir(cfg)
    output_dir.mkdir(parents=True, exist_ok=True)

    task_id = int(getattr(target, "_sp_current_task_id", -1))
    episode_idx = int(getattr(target, "_sp_current_episode_idx", -1))
    task_name = _safe_name(str(getattr(target, "_sp_current_task_label", "sp_debug")).strip())
    if not task_name:
        task_name = "sp_debug"
    save_path = output_dir / f"task_{task_id:02d}_ep_{episode_idx:03d}_{task_name}_{image_tag}_first_pruned.png"

    _save_pruning_panel(image, keep_mask, grid_size, save_path)
    saved_tags.add(image_tag)
    print(f"[SP][DEBUG] first pruned image saved to: {save_path}", flush=True)


def _to_rgb_ndarray(image: Any) -> np.ndarray:
    arr = np.asarray(image)
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    elif arr.ndim == 3 and arr.shape[-1] == 4:
        arr = arr[..., :3]
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr


def _safe_name(text: str) -> str:
    chars = []
    for ch in text:
        if ch.isalnum():
            chars.append(ch)
        else:
            chars.append("_")
    safe = "".join(chars).strip("_")
    while "__" in safe:
        safe = safe.replace("__", "_")
    return safe or "sp_debug"


def _resolve_run_label(cfg: Any) -> str:
    custom_log_filepath = str(getattr(cfg, "custom_log_filepath", "") or "").strip()
    if custom_log_filepath:
        return _safe_name(Path(custom_log_filepath).stem)

    run_id_note = str(getattr(cfg, "run_id_note", "") or "").strip()
    if run_id_note:
        return _safe_name(run_id_note)

    return "default"


def _save_pruning_panel(
    image: np.ndarray,
    keep_mask: np.ndarray,
    grid_size: int,
    save_path: Path,
) -> None:
    orig = Image.fromarray(_to_rgb_ndarray(image), mode="RGB")

    mask_2d = keep_mask.reshape(grid_size, grid_size)
    mask_image = Image.fromarray(mask_2d * 255, mode="L").resize(orig.size, resample=Image.NEAREST)
    mask_np = (np.asarray(mask_image, dtype=np.uint8) > 0).astype(np.uint8)

    orig_np = np.asarray(orig, dtype=np.uint8)
    pruned_np = (orig_np * mask_np[..., None]).astype(np.uint8)

    panel = np.concatenate([orig_np, np.full((orig_np.shape[0], 8, 3), 255, dtype=np.uint8), pruned_np], axis=1)
    Image.fromarray(panel, mode="RGB").save(save_path)
