"""Utilities for VLA-Cache-style static patch discovery and attention filtering."""

from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple

import numpy as np
import torch
from PIL import Image


PATCH_SIZE = 14
PATCHES_PER_IMAGE = 16 * 16


def to_numpy_rgb(image) -> np.ndarray:
    if isinstance(image, Image.Image):
        return np.asarray(image.convert("RGB"))
    arr = np.asarray(image)
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    if arr.ndim == 3 and arr.shape[-1] == 4:
        arr = arr[..., :3]
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr


def patchify(image: np.ndarray, patch_size: int = PATCH_SIZE) -> np.ndarray:
    array = to_numpy_rgb(image)
    height, width = array.shape[:2]
    assert height % patch_size == 0 and width % patch_size == 0, "Image size must be divisible by patch size."
    rows = height // patch_size
    cols = width // patch_size
    patches = array.reshape(rows, patch_size, cols, patch_size, 3).transpose(0, 2, 1, 3, 4)
    return patches.reshape(rows * cols, patch_size, patch_size, 3)


def calculate_patch_similarity(patches_a: np.ndarray, patches_b: np.ndarray) -> np.ndarray:
    flat_a = patches_a.reshape(len(patches_a), -1).astype(np.float32)
    flat_b = patches_b.reshape(len(patches_b), -1).astype(np.float32)
    dot = np.sum(flat_a * flat_b, axis=1)
    norm_a = np.linalg.norm(flat_a, axis=1)
    norm_b = np.linalg.norm(flat_b, axis=1)
    return dot / (norm_a * norm_b + 1e-8)


def find_static_patches(
    current_image,
    previous_image,
    patch_size: int = PATCH_SIZE,
    top_k: int = 150,
    sim_threshold: float = 0.996,
) -> List[int]:
    if previous_image is None:
        return []

    current_patches = patchify(current_image, patch_size=patch_size)
    previous_patches = patchify(previous_image, patch_size=patch_size)
    similarity = calculate_patch_similarity(current_patches, previous_patches)

    patch_scores = [(idx, float(score)) for idx, score in enumerate(similarity) if float(score) >= sim_threshold]
    patch_scores.sort(key=lambda item: item[1], reverse=True)
    return [idx for idx, _ in patch_scores[:top_k]]


@torch.no_grad()
def get_layer_mask_schedule(
    multihead_attention: Sequence[torch.Tensor],
    apply_weighted_growth: bool = True,
    growth_factor: float = 0.55,
) -> torch.Tensor:
    if not multihead_attention:
        return torch.empty((0,), dtype=torch.float32)

    device = multihead_attention[0].device
    if len(multihead_attention) == 1:
        return torch.ones((1,), device=device, dtype=torch.float32)

    entropies = []
    for attn in multihead_attention[:-1]:
        # Attention shape: [B, heads, q_len, k_len] or reduced [q_len, k_len]
        if attn.dim() == 4:
            attn = attn.mean(dim=1)[0]
        elif attn.dim() != 2:
            raise ValueError(f"Unexpected attention tensor shape for schedule: {tuple(attn.shape)}")
        attn = attn / (attn.sum(dim=-1, keepdim=True) + 1e-10)
        attn = torch.nan_to_num(attn, nan=0.0)
        token_entropy = -torch.sum(attn * torch.log(attn + 1e-10), dim=-1)
        entropies.append(token_entropy.mean())

    entropies = torch.stack(entropies)
    norm_entropy = (entropies - entropies.min()) / (entropies.max() - entropies.min() + 1e-10)
    reuse = 1.0 - norm_entropy

    if apply_weighted_growth:
        reuse = reuse.tolist()
        for idx in range(1, len(reuse)):
            delta = reuse[idx] - reuse[idx - 1]
            if delta > 0:
                reuse[idx] = reuse[idx - 1] + delta * growth_factor
        reuse = torch.tensor(reuse, dtype=torch.float32, device=device)

    return reuse


@torch.no_grad()
def select_reusable_visual_positions(
    prev_attentions: Sequence[torch.Tensor],
    query_positions: torch.Tensor,
    text_position_range: Tuple[int, int],
    stable_patch_indices_per_image: Sequence[Sequence[int]],
    attention_layer: int = 15,
    top_k: int = 100,
    patches_per_image: int = PATCHES_PER_IMAGE,
) -> Tuple[List[int], List[int]]:
    if not prev_attentions or query_positions is None:
        return [], [0 for _ in stable_patch_indices_per_image]

    num_layers = len(prev_attentions)
    layer_idx = attention_layer if attention_layer >= 0 else num_layers + attention_layer
    layer_idx = max(0, min(layer_idx, num_layers - 1))

    layer_attn = prev_attentions[layer_idx]
    if layer_attn.dim() == 4:
        layer_attn = layer_attn.to(torch.float32).mean(dim=1)[0]
    elif layer_attn.dim() == 2:
        layer_attn = layer_attn.to(torch.float32)
    else:
        raise ValueError(f"Unexpected attention tensor shape for reuse selection: {tuple(layer_attn.shape)}")

    text_start, text_end = text_position_range
    query_positions = query_positions.to(layer_attn.device)
    text_mask = (query_positions >= int(text_start)) & (query_positions < int(text_end))
    if int(text_mask.sum().item()) == 0:
        return [], [0 for _ in stable_patch_indices_per_image]

    reusable_positions: List[int] = []
    reusable_counts: List[int] = []
    for image_idx, stable_indices in enumerate(stable_patch_indices_per_image):
        if not stable_indices:
            reusable_counts.append(0)
            continue

        absolute_start = 1 + image_idx * patches_per_image
        absolute_end = absolute_start + patches_per_image
        score = layer_attn[text_mask, absolute_start:absolute_end].mean(dim=0)
        keep_top = min(int(top_k), patches_per_image)
        top_patch_indices = torch.topk(score, k=max(keep_top, 1)).indices.detach().cpu().tolist() if keep_top > 0 else []
        stable_set = set(int(idx) for idx in stable_indices)
        task_relevant = set(int(idx) for idx in top_patch_indices)
        reusable_local = sorted(stable_set - task_relevant)
        reusable_counts.append(len(reusable_local))
        reusable_positions.extend([absolute_start + idx for idx in reusable_local])

    return sorted(reusable_positions), reusable_counts


def summarize_reuse_ratio(reusable_counts: Sequence[int], patches_per_image: int = PATCHES_PER_IMAGE) -> List[float]:
    ratios = []
    for count in reusable_counts:
        ratios.append(0.0 if patches_per_image <= 0 else float(count) / float(patches_per_image))
    return ratios
