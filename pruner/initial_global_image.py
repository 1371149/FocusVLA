"""Helpers for saving the first warmed-up global-view frame during evaluation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


def safe_filename(value: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in str(value))
    return safe.strip("_") or "task"


def to_rgb_uint8(image):
    image = np.asarray(image)
    if image.ndim == 2:
        image = np.repeat(image[:, :, None], 3, axis=2)
    if image.ndim == 3 and image.shape[0] in {1, 3} and image.shape[-1] not in {1, 3}:
        image = np.transpose(image, (1, 2, 0))
    if image.ndim == 3 and image.shape[-1] == 1:
        image = np.repeat(image, 3, axis=2)
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"Expected RGB-like image, got shape={image.shape}")
    if image.dtype != np.uint8:
        image = image * 255.0 if float(np.nanmax(image)) <= 1.0 else image
        image = np.clip(image, 0, 255).astype(np.uint8)
    return image


def resolve_initial_global_image_dir(cfg, default_note: str) -> Path:
    configured_dir = str(getattr(cfg, "eval_initial_global_image_dir", "") or "").strip()
    if configured_dir:
        return Path(configured_dir).expanduser().resolve()

    if getattr(cfg, "custom_log_filepath", None):
        save_stem = Path(str(cfg.custom_log_filepath)).stem
    else:
        save_stem = str(getattr(cfg, "run_id_note", "") or default_note)
    return Path(cfg.local_log_dir).expanduser().resolve() / "initial_global_images" / safe_filename(save_stem)


def save_initial_global_image(
    cfg,
    image,
    task_name: str,
    episode_idx: int,
    total_episode_idx: int,
    *,
    default_note: str,
) -> Path | None:
    if not bool(getattr(cfg, "eval_save_initial_global_images", False)):
        return None

    output_dir = resolve_initial_global_image_dir(cfg, default_note)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_path = output_dir / f"task_{safe_filename(task_name)}_ep_{episode_idx:03d}_global_initial.png"
    if save_path.exists():
        save_path = output_dir / (
            f"episode_{total_episode_idx:04d}_task_{safe_filename(task_name)}_ep_{episode_idx:03d}_global_initial.png"
        )
    Image.fromarray(to_rgb_uint8(image)).save(save_path)
    return save_path
