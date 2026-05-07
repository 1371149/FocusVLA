"""Debug visualization helpers for ADP pruning."""

from __future__ import annotations

import math
import os
import types
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image


def _to_rgb_ndarray(x) -> np.ndarray:
    try:
        if hasattr(x, "detach"):
            x = x.detach().cpu().numpy()
        arr = np.array(x)
        if arr.ndim == 2:
            arr = np.stack([arr, arr, arr], axis=-1)
        elif arr.ndim == 3:
            if arr.shape[0] in (3, 4) and arr.shape[-1] not in (3, 4):
                arr = np.transpose(arr[:3], (1, 2, 0))
            if arr.shape[-1] == 4:
                arr = arr[..., :3]
            if arr.shape[-1] > 4:
                arr = arr[..., :3]
        if arr.dtype != np.uint8:
            arr = arr.astype(np.float32)
            arr = np.clip(arr, 0, 255)
            arr = arr.astype(np.uint8)
        return arr
    except Exception:
        try:
            return np.array(Image.fromarray(x).convert("RGB"))
        except Exception:
            return np.zeros((64, 64, 3), dtype=np.uint8)


def _clone_observation(observation):
    cloned = {}
    for key, value in observation.items():
        if hasattr(value, "copy"):
            try:
                cloned[key] = value.copy()
                continue
            except Exception:
                pass
        try:
            cloned[key] = np.array(value, copy=True)
        except Exception:
            cloned[key] = value
    return cloned


def get_debug_output_dir(cfg) -> str:
    base_dir = Path(getattr(cfg, "local_log_dir", ".")).expanduser().resolve() / "debug_attention_pruning"
    custom_log_filepath = str(getattr(cfg, "custom_log_filepath", "") or "").strip()
    if custom_log_filepath:
        run_label = Path(custom_log_filepath).stem
    else:
        run_label = str(getattr(cfg, "run_id_note", "") or "").strip() or "default"
    safe_label = _safe_name(run_label)
    return str(base_dir / safe_label)


def maybe_debug_first_step(
    cfg,
    t: int,
    action_queue,
    model,
    observation,
    task_description: str,
    get_action_fn,
    *,
    processor=None,
    action_head=None,
    proprio_projector=None,
    noisy_action_projector=None,
):
    if not bool(getattr(cfg, "qk_debug", False)):
        return False

    if not (t == cfg.num_steps_wait and len(action_queue) == 0):
        return False

    debug_dir = get_debug_output_dir(cfg)
    os.makedirs(debug_dir, exist_ok=True)
    print("\n[ADP][DEBUG] hooking _compute_qk_importance_generic for pruning visualization...")

    base_model = model.module if hasattr(model, "module") else model
    original_compute = base_model._compute_qk_importance_generic
    captured_importance = None
    original_qk_keep_enabled = bool(getattr(cfg, "qk_keep_enabled", False))
    debug_actions = None

    def hooked_compute(self_obj, lm, text_emb, vis_emb, layer):
        nonlocal captured_importance
        imp = original_compute(lm, text_emb, vis_emb, layer)
        captured_importance = imp.detach().clone().cpu()
        return imp

    base_model._compute_qk_importance_generic = types.MethodType(hooked_compute, base_model)
    try:
        if not original_qk_keep_enabled:
            print("[ADP][DEBUG] pruning disabled for this chunk; running sidecar pruned pass for visualization only.")
            cfg.qk_keep_enabled = True
        debug_actions = get_action_fn(
            cfg,
            model,
            _clone_observation(observation),
            task_description,
            processor=processor,
            action_head=action_head,
            proprio_projector=proprio_projector,
            noisy_action_projector=noisy_action_projector,
            use_film=cfg.use_film,
        )
    finally:
        cfg.qk_keep_enabled = original_qk_keep_enabled
        base_model._compute_qk_importance_generic = original_compute

    if original_qk_keep_enabled:
        action_queue.extend(debug_actions)
    else:
        actions = get_action_fn(
            cfg,
            model,
            _clone_observation(observation),
            task_description,
            processor=processor,
            action_head=action_head,
            proprio_projector=proprio_projector,
            noisy_action_projector=noisy_action_projector,
            use_film=cfg.use_film,
        )
        action_queue.extend(actions)

    if captured_importance is None:
        print("[ADP][DEBUG] failed to capture pruning importance; check qk_keep_enabled.")
        return True

    keep_ratio = float(getattr(cfg, "qk_keep_ratio", 0.5))
    total_vis_tokens = captured_importance.shape[0]
    num_images = max(1, int(getattr(cfg, "num_images_in_input", 1)))
    tokens_per_image = total_vis_tokens // num_images if num_images > 0 else total_vis_tokens
    if tokens_per_image <= 0:
        print("[ADP][DEBUG] invalid token layout; skip visualization.")
        return True

    grid_size = int(math.sqrt(tokens_per_image))
    if grid_size * grid_size != tokens_per_image:
        print("[ADP][DEBUG] vision token count is not a square grid; skip visualization.")
        return True

    safe_task_name = task_description.replace(" ", "_")
    image_names = ["full", "wrist"]
    image_arrays = [
        _to_rgb_ndarray(observation["full_image"]),
        _to_rgb_ndarray(observation["wrist_image"]),
    ]

    for image_idx in range(min(num_images, len(image_arrays))):
        start = image_idx * tokens_per_image
        end = start + tokens_per_image
        importance = captured_importance[start:end]
        save_path = os.path.join(debug_dir, f"{safe_task_name}_{image_names[image_idx]}_pruned.png")
        _save_pruning_overlay(
            image_arrays[image_idx],
            importance,
            grid_size,
            keep_ratio,
            save_path,
            title_prefix=image_names[image_idx],
        )

    print(f"[ADP][DEBUG] pruning visualization saved to: {debug_dir}")
    return True


def _save_pruning_overlay(img_array, importance_tensor, grid_size: int, keep_ratio: float, save_path: str, title_prefix: str):
    k_i = max(1, int(round(grid_size * grid_size * keep_ratio)))
    if k_i < len(importance_tensor):
        threshold_val = torch.topk(importance_tensor, k=k_i)[0][-1]
        keep_mask = (importance_tensor >= threshold_val).float().numpy()
    else:
        keep_mask = np.ones(len(importance_tensor))

    mask_2d = keep_mask.reshape(grid_size, grid_size)
    mask_img = Image.fromarray((mask_2d * 255).astype(np.uint8), mode="L")
    orig_img = Image.fromarray(_to_rgb_ndarray(img_array))
    mask_img = mask_img.resize(orig_img.size, resample=Image.NEAREST)
    mask_np = np.array(mask_img) / 255.0
    mask_3c = np.stack([mask_np] * 3, axis=-1)
    pruned_np = (np.asarray(orig_img, dtype=np.uint8) * mask_3c).astype(np.uint8)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(orig_img)
    axes[0].set_title(f"{title_prefix} original")
    axes[0].axis("off")
    axes[1].imshow(pruned_np)
    axes[1].set_title(f"{title_prefix} pruned ({keep_ratio * 100:.0f}%)")
    axes[1].axis("off")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def _safe_name(text: str) -> str:
    chars = []
    for ch in str(text):
        chars.append(ch if ch.isalnum() else "_")
    safe = "".join(chars).strip("_")
    while "__" in safe:
        safe = safe.replace("__", "_")
    return safe or "adp_debug"
