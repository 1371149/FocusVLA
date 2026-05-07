"""Runtime monkeypatches for a self-contained VLA-Cache-style evaluator.

This implementation stays inside `pruner/VLACache` and avoids modifying the
main `openvla-oft` tree on disk. It reproduces the core VLA-Cache idea under
the constraints of the stock `openvla-oft` conda environment:

1. cache previous KV states across action queries
2. detect static visual patches between frames
3. reuse cached KV slots for non-task-relevant static patches
4. recompute only dynamic patches plus the trailing prompt/action tokens

The original repo relies on a custom `transformers` fork. Under the "modify
only pruner/VLACache" constraint we implement the closest compatible runtime
version with stock `transformers==4.40.x`.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from types import MethodType
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from transformers.cache_utils import Cache, DynamicCache

from backend import load_backend_modules
from vla_cache_utils import (
    get_layer_mask_schedule,
    find_static_patches,
    select_reusable_visual_positions,
    summarize_reuse_ratio,
)


_backend = load_backend_modules()
openvla_utils = _backend.openvla_utils
robot_utils = _backend.robot_utils

from prismatic.vla.constants import ACTION_DIM, IGNORE_INDEX, NUM_ACTIONS_CHUNK


PATCHES_PER_IMAGE = 16 * 16


@dataclass
class EpisodeCacheMetrics:
    model_calls: int = 0
    model_time_seconds: float = 0.0
    reused_primary_tokens: int = 0
    reused_wrist_tokens: int = 0

    def snapshot(self) -> Dict[str, float]:
        avg_ms = 0.0 if self.model_calls <= 0 else (self.model_time_seconds / float(self.model_calls) * 1000.0)
        primary_ratio = 0.0 if self.model_calls <= 0 else self.reused_primary_tokens / float(self.model_calls * PATCHES_PER_IMAGE)
        wrist_ratio = 0.0 if self.model_calls <= 0 else self.reused_wrist_tokens / float(self.model_calls * PATCHES_PER_IMAGE)
        return {
            "model_calls": float(self.model_calls),
            "model_time_seconds": float(self.model_time_seconds),
            "avg_model_time_ms": float(avg_ms),
            "reuse_ratio_primary": float(primary_ratio),
            "reuse_ratio_wrist": float(wrist_ratio),
        }


class PositionAwareCache(Cache):
    """Cache that supports overwriting arbitrary token positions in-place."""

    def __init__(self) -> None:
        self.key_cache: List[torch.Tensor] = []
        self.value_cache: List[torch.Tensor] = []
        self._seen_tokens: int = 0

    def __len__(self) -> int:
        return len(self.key_cache)

    def __getitem__(self, layer_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    def _ensure_layer_shape(
        self,
        cache_list: List[torch.Tensor],
        layer_idx: int,
        template: torch.Tensor,
        seq_len: int,
    ) -> None:
        if len(cache_list) <= layer_idx:
            shape = list(template.shape)
            shape[-2] = seq_len
            cache_list.append(template.new_zeros(shape))
            return

        existing = cache_list[layer_idx]
        if existing.shape[-2] >= seq_len:
            return

        pad_shape = list(existing.shape)
        pad_shape[-2] = seq_len - existing.shape[-2]
        padding = existing.new_zeros(pad_shape)
        cache_list[layer_idx] = torch.cat([existing, padding], dim=-2)

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        cache_position = None if cache_kwargs is None else cache_kwargs.get("cache_position")
        if cache_position is None:
            if layer_idx == 0:
                self._seen_tokens += key_states.shape[-2]
            if len(self.key_cache) <= layer_idx:
                self.key_cache.append(key_states)
                self.value_cache.append(value_states)
            else:
                self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim=-2)
                self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states], dim=-2)
            return self.key_cache[layer_idx], self.value_cache[layer_idx]

        positions = cache_position.to(device=key_states.device, dtype=torch.long)
        required_seq_len = int(positions.max().item()) + 1 if positions.numel() > 0 else 0

        self._ensure_layer_shape(self.key_cache, layer_idx, key_states, required_seq_len)
        self._ensure_layer_shape(self.value_cache, layer_idx, value_states, required_seq_len)

        self.key_cache[layer_idx][:, :, positions, :] = key_states
        self.value_cache[layer_idx][:, :, positions, :] = value_states

        if layer_idx == 0:
            self._seen_tokens = max(self._seen_tokens, required_seq_len)

        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        if layer_idx is None:
            layer_idx = 0
        if len(self.key_cache) <= layer_idx:
            return 0
        return int(self.key_cache[layer_idx].shape[-2])

    def get_max_length(self) -> Optional[int]:
        return None

    def reorder_cache(self, beam_idx: torch.LongTensor) -> None:
        for idx in range(len(self.key_cache)):
            self.key_cache[idx] = self.key_cache[idx].index_select(0, beam_idx)
            self.value_cache[idx] = self.value_cache[idx].index_select(0, beam_idx)

    def to_legacy_cache(self):
        return self


def apply_runtime_patches() -> None:
    if getattr(openvla_utils, "_vla_cache_runtime_patch_applied", False):
        return
    openvla_utils._vla_cache_runtime_patch_applied = True

    if not hasattr(DynamicCache, "_vla_cache_original_from_legacy_cache"):
        DynamicCache._vla_cache_original_from_legacy_cache = DynamicCache.from_legacy_cache

        @classmethod
        def _patched_from_legacy_cache(cls, past_key_values=None):
            if isinstance(past_key_values, Cache):
                return past_key_values
            return cls._vla_cache_original_from_legacy_cache(past_key_values)

        DynamicCache.from_legacy_cache = _patched_from_legacy_cache


def patch_loaded_model(model: torch.nn.Module, cfg: Optional[Any] = None) -> None:
    target = model.module if hasattr(model, "module") else model
    if cfg is not None:
        target._vla_cache_cfg = cfg
    if not hasattr(target, "_vla_cache_episode_metrics"):
        target._vla_cache_episode_metrics = EpisodeCacheMetrics()

    if not hasattr(target, "_vla_cache_original_predict_action"):
        target._vla_cache_original_predict_action = target.predict_action
        target._vla_cache_original_regression_prediction = target._regression_or_discrete_prediction
        target.predict_action = MethodType(predict_action_with_vla_cache, target)
        target._regression_or_discrete_prediction = MethodType(
            _regression_or_discrete_prediction_with_vla_cache,
            target,
        )
        target.vla_cache_reset_episode = MethodType(vla_cache_reset_episode, target)

    target.vla_cache_reset_episode()


def get_episode_metrics(model: torch.nn.Module) -> Dict[str, float]:
    target = model.module if hasattr(model, "module") else model
    metrics = getattr(target, "_vla_cache_episode_metrics", None)
    if metrics is None:
        return EpisodeCacheMetrics().snapshot()
    return metrics.snapshot()


def vla_cache_reset_episode(self) -> None:
    self._vla_cache_episode_metrics = EpisodeCacheMetrics()


def _build_absolute_masks(
    all_actions_mask: torch.Tensor,
    projected_patch_embeddings: torch.Tensor,
    num_visual_patch_tokens: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    device = projected_patch_embeddings.device
    batch_size = projected_patch_embeddings.shape[0]
    total_projected_tokens = projected_patch_embeddings.shape[1]
    prefix_false = torch.zeros((batch_size, total_projected_tokens), dtype=torch.bool, device=device)
    final_action_mask = torch.cat([all_actions_mask[:, :1], prefix_false, all_actions_mask[:, 1:]], dim=1)
    visual_positions = torch.arange(1, 1 + num_visual_patch_tokens, device=device, dtype=torch.long)
    trailing_positions = torch.arange(1 + total_projected_tokens, final_action_mask.shape[1], device=device, dtype=torch.long)
    return final_action_mask, visual_positions, trailing_positions


def _regression_or_discrete_prediction_with_vla_cache(
    self,
    input_embeddings,
    all_actions_mask,
    projected_patch_embeddings,
    attention_mask,
    labels,
    num_patches,
    num_prompt_tokens,
    action_head=None,
    past_key_values=None,
    reusable_positions=None,
):
    all_actions_mask_bool = all_actions_mask
    all_actions_mask = all_actions_mask.unsqueeze(-1)
    input_embeddings = input_embeddings * ~all_actions_mask

    multimodal_embeddings, multimodal_attention_mask = self._build_multimodal_attention(
        input_embeddings, projected_patch_embeddings, attention_mask
    )

    device = multimodal_embeddings.device
    num_visual_patch_tokens = self.vision_backbone.get_num_patches() * self.vision_backbone.get_num_images_in_input()
    final_action_mask, visual_positions, trailing_positions = _build_absolute_masks(
        all_actions_mask_bool,
        projected_patch_embeddings,
        num_visual_patch_tokens,
    )

    if past_key_values is None:
        past_key_values = PositionAwareCache()

    use_cached_path = (
        isinstance(past_key_values, PositionAwareCache)
        and past_key_values.get_seq_length() > 0
        and reusable_positions is not None
        and len(reusable_positions) > 0
    )

    if use_cached_path:
        reusable_positions_tensor = torch.as_tensor(reusable_positions, device=device, dtype=torch.long)
        reusable_positions_tensor = reusable_positions_tensor[
            (reusable_positions_tensor >= 1) & (reusable_positions_tensor < 1 + num_visual_patch_tokens)
        ]
        dynamic_visual_positions = visual_positions[
            ~torch.isin(visual_positions, reusable_positions_tensor)
        ]
        # Recompute proprio token plus all trailing text/action tokens every step.
        proprio_and_trailing_start = 1 + num_visual_patch_tokens
        proprio_and_trailing_positions = torch.arange(
            proprio_and_trailing_start,
            multimodal_embeddings.shape[1],
            device=device,
            dtype=torch.long,
        )
        query_positions = torch.cat([dynamic_visual_positions, proprio_and_trailing_positions], dim=0)
        selected_embeddings = multimodal_embeddings.index_select(1, query_positions)
        current_attention_mask = None
    else:
        query_positions = torch.arange(multimodal_embeddings.shape[1], device=device, dtype=torch.long)
        selected_embeddings = multimodal_embeddings
        current_attention_mask = multimodal_attention_mask

    language_model_output = self.language_model(
        input_ids=None,
        attention_mask=current_attention_mask,
        position_ids=None,
        past_key_values=past_key_values,
        inputs_embeds=selected_embeddings,
        labels=None,
        use_cache=True,
        output_attentions=True,
        output_hidden_states=True,
        return_dict=True,
        cache_position=query_positions,
    )

    last_hidden_states = language_model_output.hidden_states[-1]
    action_positions = torch.where(final_action_mask[0])[0]
    action_query_mask = torch.isin(query_positions, action_positions)
    actions_hidden_states = last_hidden_states[:, action_query_mask, :]

    reduced_attentions = tuple(
        att.detach().mean(dim=1)[0].to(dtype=torch.float16).cpu() for att in language_model_output.attentions
    )
    last_caches = {
        "past_key_values": language_model_output.past_key_values,
        "attentions": reduced_attentions,
        "query_positions": query_positions.detach().cpu(),
        "full_sequence_length": int(multimodal_embeddings.shape[1]),
    }

    if action_head is not None:
        normalized_actions = action_head.predict_action(actions_hidden_states)
        normalized_actions = normalized_actions.reshape(NUM_ACTIONS_CHUNK, ACTION_DIM)
        normalized_actions = normalized_actions.float().cpu().detach().numpy()
    else:
        action_logits = language_model_output.logits[:, action_query_mask, :]
        predicted_action_token_ids = action_logits.argmax(dim=2).cpu().numpy()
        discretized_actions = self.vocab_size - predicted_action_token_ids
        discretized_actions = np.clip(discretized_actions - 1, a_min=0, a_max=self.bin_centers.shape[0] - 1)
        normalized_actions = self.bin_centers[discretized_actions].reshape(NUM_ACTIONS_CHUNK, ACTION_DIM)

    return normalized_actions, actions_hidden_states, last_caches


def predict_action_with_vla_cache(
    self,
    input_ids: Optional[torch.LongTensor] = None,
    unnorm_key: Optional[str] = None,
    proprio=None,
    proprio_projector=None,
    action_head=None,
    noisy_action_projector=None,
    use_film: bool = False,
    past_key_values=None,
    reusable_positions=None,
    **kwargs: str,
) -> np.ndarray:
    if not torch.all(input_ids[:, -1] == 29871):
        input_ids = torch.cat(
            (input_ids, torch.unsqueeze(torch.tensor([29871]).long(), dim=0).to(input_ids.device)),
            dim=1,
        )

    pixel_values = kwargs["pixel_values"]
    attention_mask = kwargs["attention_mask"]

    labels = input_ids.clone()
    labels[:] = IGNORE_INDEX
    num_prompt_tokens = input_ids.shape[-1] - 1

    input_ids, attention_mask = self._prepare_input_for_action_prediction(input_ids, attention_mask)
    labels = self._prepare_labels_for_action_prediction(labels, input_ids)

    input_embeddings = self.get_input_embeddings()(input_ids)
    all_actions_mask = self._process_action_masks(labels)
    language_embeddings = input_embeddings[~all_actions_mask].reshape(
        input_embeddings.shape[0], -1, input_embeddings.shape[2]
    )

    projected_patch_embeddings = self._process_vision_features(pixel_values, language_embeddings, use_film)
    use_proprio = proprio_projector is not None and proprio is not None
    if use_proprio:
        proprio = torch.as_tensor(proprio).to(projected_patch_embeddings.device, dtype=projected_patch_embeddings.dtype)
        projected_patch_embeddings = self._process_proprio_features(projected_patch_embeddings, proprio, proprio_projector)

    use_diffusion = noisy_action_projector is not None and hasattr(action_head, "noise_scheduler")
    if use_diffusion:
        # The stock openvla-oft eval path uses L1 regression; keep a safe fallback for diffusion.
        actions, actions_hidden_states = self._vla_cache_original_predict_action(
            input_ids=input_ids[:, : num_prompt_tokens + 1],
            unnorm_key=unnorm_key,
            proprio=proprio,
            proprio_projector=proprio_projector,
            action_head=action_head,
            noisy_action_projector=noisy_action_projector,
            use_film=use_film,
            pixel_values=pixel_values,
            attention_mask=attention_mask[:, : num_prompt_tokens + 1],
        )
        return actions, actions_hidden_states, None

    normalized_actions, actions_hidden_states, last_caches = self._regression_or_discrete_prediction(
        input_embeddings,
        all_actions_mask,
        projected_patch_embeddings,
        attention_mask,
        labels,
        self.vision_backbone.get_num_patches() * self.vision_backbone.get_num_images_in_input() + (1 if use_proprio else 0),
        num_prompt_tokens,
        action_head,
        past_key_values,
        reusable_positions,
    )

    actions = self._unnormalize_actions(normalized_actions, unnorm_key)
    return actions, actions_hidden_states, last_caches


@torch.inference_mode()
def get_cached_vla_action(
    cfg: Any,
    vla: torch.nn.Module,
    processor: Any,
    obs: Dict[str, Any],
    task_label: str,
    action_head: Optional[torch.nn.Module] = None,
    proprio_projector: Optional[torch.nn.Module] = None,
    noisy_action_projector: Optional[torch.nn.Module] = None,
    use_film: bool = False,
    last_caches: Optional[Dict[str, Any]] = None,
) -> Tuple[List[np.ndarray], Optional[Dict[str, Any]], Dict[str, float]]:
    patch_loaded_model(vla, cfg)
    target = vla.module if hasattr(vla, "module") else vla

    all_images = [obs["full_image"]]
    if cfg.num_images_in_input > 1:
        all_images.extend([obs[k] for k in obs.keys() if "wrist" in k])
    all_images = openvla_utils.prepare_images_for_vla(all_images, cfg)

    previous_images = obs.get("prev_images") or [None for _ in all_images]
    previous_images = [
        None if image is None else openvla_utils.prepare_images_for_vla([image], cfg)[0]
        for image in previous_images
    ]

    primary_image = all_images[0]
    prompt = f"In: What action should the robot take to {task_label.lower()}?\nOut:"
    inputs = processor(prompt, primary_image).to(openvla_utils.DEVICE, dtype=torch.bfloat16)
    prompt_token_count = int(inputs["input_ids"].shape[1])

    if len(all_images) > 1:
        wrist_inputs = [processor(prompt, image).to(openvla_utils.DEVICE, dtype=torch.bfloat16) for image in all_images[1:]]
        primary_pixel_values = inputs["pixel_values"]
        wrist_pixel_values = [item["pixel_values"] for item in wrist_inputs]
        inputs["pixel_values"] = torch.cat([primary_pixel_values] + wrist_pixel_values, dim=1)

    proprio = None
    if cfg.use_proprio:
        proprio = obs["state"]
        proprio_norm_stats = vla.norm_stats[cfg.unnorm_key]["proprio"]
        proprio = openvla_utils.normalize_proprio(proprio, proprio_norm_stats)

    past_key_values = None
    reusable_positions: Optional[List[int]] = None
    reused_counts = [0 for _ in all_images]
    layer_schedule = torch.empty((0,), dtype=torch.float32)
    if cfg.use_vla_cache:
        past_key_values = last_caches["past_key_values"] if last_caches is not None else PositionAwareCache()
        prev_attentions = None if last_caches is None else last_caches.get("attentions")
        prev_query_positions = None if last_caches is None else last_caches.get("query_positions")
        if prev_attentions is not None and prev_query_positions is not None:
            stable_patch_indices = []
            for image, prev_image in zip(all_images, previous_images):
                stable_patch_indices.append(
                    find_static_patches(
                        image,
                        prev_image,
                        top_k=int(cfg.vla_cache_static_top_k),
                        sim_threshold=float(cfg.vla_cache_similarity_threshold),
                    )
                )

            total_projected_tokens = target.vision_backbone.get_num_patches() * len(all_images) + (1 if cfg.use_proprio else 0)
            text_start = 1 + total_projected_tokens
            text_end = text_start + max(prompt_token_count - 1, 0)
            reusable_positions, reused_counts = select_reusable_visual_positions(
                prev_attentions=prev_attentions,
                query_positions=prev_query_positions,
                text_position_range=(text_start, text_end),
                stable_patch_indices_per_image=stable_patch_indices,
                attention_layer=int(cfg.vla_cache_attention_layer),
                top_k=int(cfg.vla_cache_attention_top_k),
                patches_per_image=target.vision_backbone.get_num_patches(),
            )
            layer_schedule = get_layer_mask_schedule(
                prev_attentions,
                apply_weighted_growth=bool(cfg.vla_cache_apply_weighted_growth),
                growth_factor=float(cfg.vla_cache_growth_factor),
            )

    start_time = time.perf_counter()
    actions, _, next_caches = vla.predict_action(
        **inputs,
        unnorm_key=cfg.unnorm_key,
        do_sample=False,
        proprio=proprio,
        proprio_projector=proprio_projector,
        noisy_action_projector=noisy_action_projector,
        action_head=action_head,
        use_film=use_film,
        past_key_values=past_key_values,
        reusable_positions=reusable_positions,
    )
    elapsed = time.perf_counter() - start_time

    if cfg.use_vla_cache and next_caches is not None:
        next_caches["layer_schedule"] = layer_schedule.detach().cpu() if isinstance(layer_schedule, torch.Tensor) else layer_schedule
    else:
        next_caches = None

    ratios = summarize_reuse_ratio(reused_counts, patches_per_image=target.vision_backbone.get_num_patches())
    metrics = {
        "time_elapsed": float(elapsed),
        "num_static_tokens_primary": float(reused_counts[0] if len(reused_counts) > 0 else 0),
        "num_static_tokens_wrist": float(reused_counts[1] if len(reused_counts) > 1 else 0),
        "reuse_ratio_primary": float(ratios[0] if len(ratios) > 0 else 0.0),
        "reuse_ratio_wrist": float(ratios[1] if len(ratios) > 1 else 0.0),
    }

    target._vla_cache_episode_metrics.model_calls += 1
    target._vla_cache_episode_metrics.model_time_seconds += float(elapsed)
    if len(reused_counts) > 0:
        target._vla_cache_episode_metrics.reused_primary_tokens += int(reused_counts[0])
    if len(reused_counts) > 1:
        target._vla_cache_episode_metrics.reused_wrist_tokens += int(reused_counts[1])

    action_list = [actions[idx] for idx in range(min(len(actions), cfg.num_open_loop_steps))]
    return action_list, next_caches, metrics
