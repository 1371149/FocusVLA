"""Runtime monkeypatches that transplant AdaptPrune into OpenVLA-OFT.

All method-specific logic lives in this directory. The patch operates on the
already loaded OpenVLA model and avoids modifying the upstream source tree.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from types import MethodType
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F

from backend import load_backend_modules

_backend = load_backend_modules()
openvla_utils = _backend.openvla_utils

from prismatic.vla.constants import ACTION_DIM, NUM_ACTIONS_CHUNK


@dataclass
class AdaptPruneMetrics:
    model_calls: int = 0
    prune_events: int = 0
    tokens_before: int = 0
    tokens_after: int = 0
    fallback_calls: int = 0
    last_visual_tokens: int = 0
    last_keep_ratio: float = 1.0
    last_layer: int = 2
    last_keep_per_image: int = 0
    last_query_mode: str = "text_last"

    def snapshot(self) -> Dict[str, float]:
        weighted_prune_rate = 0.0
        if self.tokens_before > 0:
            weighted_prune_rate = 1.0 - (self.tokens_after / float(self.tokens_before))
        return {
            "model_calls": float(self.model_calls),
            "prune_events": float(self.prune_events),
            "tokens_before": float(self.tokens_before),
            "tokens_after": float(self.tokens_after),
            "weighted_prune_rate": float(weighted_prune_rate),
            "fallback_calls": float(self.fallback_calls),
            "last_visual_tokens": float(self.last_visual_tokens),
            "last_keep_ratio": float(self.last_keep_ratio),
            "last_layer": float(self.last_layer),
            "last_keep_per_image": float(self.last_keep_per_image),
            "last_query_mode": self.last_query_mode,
        }


@dataclass
class AdaptPruneForwardResult:
    hidden_states: torch.Tensor
    logits: Optional[torch.Tensor]
    current_visual_tokens: int
    used_adaptprune: bool


def apply_runtime_patches() -> None:
    if getattr(openvla_utils, "_adaptprune_runtime_patch_applied", False):
        return
    openvla_utils._adaptprune_runtime_patch_applied = True


def patch_loaded_model(model: torch.nn.Module, cfg: Optional[Any] = None) -> None:
    target = model.module if hasattr(model, "module") else model
    if cfg is not None:
        target._adaptprune_cfg = cfg

    if not hasattr(target, "_adaptprune_metrics"):
        target._adaptprune_metrics = AdaptPruneMetrics()
    if not hasattr(target, "_adaptprune_episode_metrics"):
        target._adaptprune_episode_metrics = AdaptPruneMetrics()
    if not hasattr(target, "_adaptprune_logged_fallback"):
        target._adaptprune_logged_fallback = False

    if not hasattr(target, "_adaptprune_original_regression_prediction"):
        target._adaptprune_original_regression_prediction = target._regression_or_discrete_prediction
        target._regression_or_discrete_prediction = MethodType(_regression_or_discrete_prediction, target)

    if not hasattr(target, "_adaptprune_original_diffusion_prediction"):
        target._adaptprune_original_diffusion_prediction = target._run_diffusion_prediction
        target._run_diffusion_prediction = MethodType(_run_diffusion_prediction, target)

    if not hasattr(target, "adaptprune_reset_episode"):
        target.adaptprune_reset_episode = MethodType(adaptprune_reset_episode, target)
    if not hasattr(target, "adaptprune_get_metrics"):
        target.adaptprune_get_metrics = MethodType(adaptprune_get_metrics, target)

    target.adaptprune_reset_episode()


def get_runtime_metrics(model: torch.nn.Module) -> Dict[str, float]:
    target = model.module if hasattr(model, "module") else model
    metrics = getattr(target, "_adaptprune_metrics", None)
    if metrics is None:
        return AdaptPruneMetrics().snapshot()
    return metrics.snapshot()


def get_episode_runtime_metrics(model: torch.nn.Module) -> Dict[str, float]:
    target = model.module if hasattr(model, "module") else model
    metrics = getattr(target, "_adaptprune_episode_metrics", None)
    if metrics is None:
        return AdaptPruneMetrics().snapshot()
    return metrics.snapshot()


def adaptprune_reset_episode(self) -> None:
    self._adaptprune_episode_metrics = AdaptPruneMetrics()


def adaptprune_get_metrics(self) -> Dict[str, float]:
    return self._adaptprune_metrics.snapshot()


def _record_metrics(
    self,
    visual_tokens_before: int,
    visual_tokens_after: int,
    used_adaptprune: bool,
    fallback: bool = False,
) -> None:
    cfg = getattr(self, "_adaptprune_cfg", None)
    layer = 2 if cfg is None else int(getattr(cfg, "adaptprune_layer", 2))
    query_mode = "text_last" if cfg is None else str(getattr(cfg, "adaptprune_query_mode", "text_last"))
    keep_ratio = 1.0
    if visual_tokens_before > 0:
        keep_ratio = visual_tokens_after / float(visual_tokens_before)

    per_image_keep = int(visual_tokens_after)
    try:
        num_images = max(1, int(self.vision_backbone.get_num_images_in_input()))
        per_image_keep = int(round(visual_tokens_after / float(num_images)))
    except Exception:
        pass

    for metrics_name in ("_adaptprune_metrics", "_adaptprune_episode_metrics"):
        metrics = getattr(self, metrics_name, None)
        if metrics is None:
            continue
        metrics.model_calls += 1
        metrics.tokens_before += int(visual_tokens_before)
        metrics.tokens_after += int(visual_tokens_after)
        metrics.last_visual_tokens = int(visual_tokens_after)
        metrics.last_keep_ratio = float(keep_ratio)
        metrics.last_layer = int(layer)
        metrics.last_keep_per_image = int(per_image_keep)
        metrics.last_query_mode = query_mode
        if used_adaptprune:
            metrics.prune_events += 1
        if fallback:
            metrics.fallback_calls += 1


def _supports_language_model(self) -> bool:
    language_model = getattr(self, "language_model", None)
    decoder = getattr(language_model, "model", None)
    if language_model is None or decoder is None:
        return False
    required_attrs = ("layers", "norm", "_update_causal_mask")
    if any(not hasattr(decoder, attr) for attr in required_attrs):
        return False
    if not hasattr(language_model, "lm_head") or not hasattr(language_model, "config"):
        return False
    return True


def _should_use_adaptprune(self, num_visual_patch_tokens: int, prompt_token_count: int) -> bool:
    cfg = getattr(self, "_adaptprune_cfg", None)
    if cfg is None:
        return False
    if not bool(getattr(cfg, "adaptprune_enabled", False)):
        return False
    if float(getattr(cfg, "adaptprune_keep_ratio", 0.5)) >= 1.0:
        return False
    if num_visual_patch_tokens <= 1 or prompt_token_count <= 0:
        return False
    if getattr(self, "training", False):
        return False
    return True


def _compute_logits_from_hidden_states(language_model, hidden_states: torch.Tensor) -> torch.Tensor:
    if language_model.config.pretraining_tp > 1:
        lm_head_slices = language_model.lm_head.weight.split(
            language_model.config.vocab_size // language_model.config.pretraining_tp, dim=0
        )
        logits = [F.linear(hidden_states, lm_head_slices[i]) for i in range(language_model.config.pretraining_tp)]
        logits = torch.cat(logits, dim=-1)
    else:
        logits = language_model.lm_head(hidden_states)
    return logits.float()


def _dense_causal_lm_forward(
    self,
    multimodal_embeddings: torch.Tensor,
    multimodal_attention_mask: Optional[torch.Tensor],
    num_visual_patch_tokens: int,
    compute_logits: bool,
) -> AdaptPruneForwardResult:
    output = self.language_model(
        input_ids=None,
        attention_mask=multimodal_attention_mask,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=multimodal_embeddings,
        labels=None,
        use_cache=True,
        output_attentions=False,
        output_hidden_states=True,
        return_dict=True,
    )
    return AdaptPruneForwardResult(
        hidden_states=output.hidden_states[-1],
        logits=output.logits if compute_logits else None,
        current_visual_tokens=int(num_visual_patch_tokens),
        used_adaptprune=False,
    )


def _center_gaussian(grid_side: int, sigma: float, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    coords = torch.linspace(-(grid_side - 1) / 2.0, (grid_side - 1) / 2.0, grid_side, device=device, dtype=dtype)
    grid_x, grid_y = torch.meshgrid(coords, coords, indexing="ij")
    gaussian = torch.exp(-((grid_x**2 + grid_y**2) / (2.0 * float(sigma) ** 2)))
    return gaussian / gaussian.max().clamp_min(1e-12)


def _adapt_nms_2d(
    attention_scores: torch.Tensor,
    key_vectors: torch.Tensor,
    grid_side: int,
    distance_limit: float,
    distance_sigma: float,
    similarity_sigma: float,
    max_output_size: int,
) -> torch.Tensor:
    flat_scores = attention_scores.reshape(-1).float().clone()
    num_tokens = flat_scores.numel()
    max_output_size = max(1, min(int(max_output_size), int(num_tokens)))

    coords = torch.arange(grid_side, device=flat_scores.device, dtype=torch.float32)
    grid_x, grid_y = torch.meshgrid(coords, coords, indexing="ij")
    grid_x = grid_x.reshape(-1)
    grid_y = grid_y.reshape(-1)
    key_vectors = F.normalize(key_vectors.float(), p=2, dim=-1)

    distance = torch.sqrt((grid_x[:, None] - grid_x[None, :]) ** 2 + (grid_y[:, None] - grid_y[None, :]) ** 2)
    if distance_limit > 0:
        window_mask = distance <= float(distance_limit)
    else:
        window_mask = torch.ones_like(distance, dtype=torch.bool)
    distance_decay = torch.exp(-(distance**2) / (2.0 * float(distance_sigma) ** 2))
    distance_decay = torch.where(window_mask, distance_decay, torch.zeros_like(distance_decay))

    similarity = torch.matmul(key_vectors, key_vectors.transpose(0, 1)).clamp(min=-1.0, max=1.0)
    similarity_decay = torch.exp(-((1.0 - similarity) ** 2) / (2.0 * float(similarity_sigma) ** 2))
    combined_decay_table = distance_decay * similarity_decay

    keep = []
    for _ in range(max_output_size):
        top_idx = torch.argmax(flat_scores)
        keep.append(top_idx)
        if len(keep) >= max_output_size:
            break

        combined_decay = combined_decay_table[top_idx]
        flat_scores *= 1.0 - combined_decay
        flat_scores[top_idx] = -torch.inf

    return torch.stack(keep).to(dtype=torch.long)


def _select_keep_indices_by_image(
    cfg: Any,
    attention_weights: torch.Tensor,
    key_layer: torch.Tensor,
    visual_start: int,
    visual_token_count: int,
    extra_projected_tokens_count: int,
    prompt_token_count: int,
    num_images: int,
    per_image_patches: int,
) -> torch.Tensor:
    if attention_weights.dim() != 4 or attention_weights.shape[0] != 1:
        raise ValueError(f"Unexpected attention shape for AdaptPrune: {tuple(attention_weights.shape)}")
    if num_images < 1 or per_image_patches < 1:
        raise ValueError("AdaptPrune requires positive image/token counts.")
    if per_image_patches * num_images > visual_token_count:
        raise ValueError(
            f"per_image_patches*num_images={per_image_patches * num_images} exceeds visual_token_count={visual_token_count}."
        )

    grid_side = int(round(math.sqrt(per_image_patches)))
    if grid_side * grid_side != per_image_patches:
        raise ValueError(f"AdaptPrune expects square per-image token grid, got {per_image_patches}.")

    keep_ratio = float(getattr(cfg, "adaptprune_keep_ratio", 0.5))
    keep_per_image = int(round(per_image_patches * keep_ratio))
    keep_per_image = max(int(getattr(cfg, "adaptprune_min_keep_tokens_per_image", 1)), keep_per_image)
    keep_per_image = min(keep_per_image, per_image_patches)

    mean_attention = attention_weights.mean(dim=1)
    prompt_start = visual_start + int(visual_token_count) + int(extra_projected_tokens_count)
    prompt_end = prompt_start + int(prompt_token_count)
    if prompt_end <= prompt_start:
        raise ValueError("AdaptPrune text token range is empty.")

    prompt_to_visual = mean_attention[0, prompt_start:prompt_end, visual_start : visual_start + visual_token_count]
    query_mode = str(getattr(cfg, "adaptprune_query_mode", "text_last"))
    if query_mode in {"text_last", "last"}:
        # In OpenVLA-OFT the sequence ends with action tokens, so -1 is not a text token.
        visual_scores = mean_attention[0, prompt_end - 1, visual_start : visual_start + visual_token_count]
    elif query_mode == "action_last":
        visual_scores = mean_attention[0, -1, visual_start : visual_start + visual_token_count]
    elif query_mode == "text_mean":
        visual_scores = prompt_to_visual.mean(dim=0)
    else:
        raise ValueError(f"Unsupported AdaptPrune query mode: {query_mode}")

    use_center = bool(getattr(cfg, "adaptprune_use_center_gaussian", True))
    center_sigma = float(getattr(cfg, "adaptprune_center_sigma", 8.0))
    if use_center:
        center_mask = _center_gaussian(grid_side, center_sigma, visual_scores.device, visual_scores.dtype).reshape(-1)
    else:
        center_mask = None

    selected_chunks = []
    for image_idx in range(num_images):
        start = image_idx * per_image_patches
        end = start + per_image_patches
        scores_i = visual_scores[start:end]
        if center_mask is not None:
            scores_i = scores_i * center_mask

        key_i = key_layer[0, visual_start + start : visual_start + end, :]
        keep_rel = _adapt_nms_2d(
            scores_i.reshape(grid_side, grid_side),
            key_i,
            grid_side=grid_side,
            distance_limit=float(getattr(cfg, "adaptprune_distance_limit", 5.0)),
            distance_sigma=float(getattr(cfg, "adaptprune_distance_sigma", 1.5)),
            similarity_sigma=float(getattr(cfg, "adaptprune_similarity_sigma", 0.05)),
            max_output_size=keep_per_image,
        )
        selected_chunks.append((keep_rel + start).to(device=visual_scores.device, dtype=torch.long))

    return torch.sort(torch.cat(selected_chunks, dim=0)).values


def _build_pruned_sequence(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    absolute_positions: torch.LongTensor,
    visual_start: int,
    visual_token_count: int,
    keep_visual_rel_indices: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.LongTensor, int]:
    prefix_states = hidden_states[:, :visual_start, :]
    visual_states = hidden_states[:, visual_start : visual_start + visual_token_count, :]
    kept_visual = visual_states.index_select(1, keep_visual_rel_indices)
    tail_states = hidden_states[:, visual_start + visual_token_count :, :]
    new_hidden_states = torch.cat([prefix_states, kept_visual, tail_states], dim=1)

    prefix_mask = attention_mask[:, :visual_start]
    visual_mask = attention_mask[:, visual_start : visual_start + visual_token_count].index_select(
        1, keep_visual_rel_indices
    )
    tail_mask = attention_mask[:, visual_start + visual_token_count :]
    new_attention_mask = torch.cat([prefix_mask, visual_mask, tail_mask], dim=1)

    prefix_positions = absolute_positions[:visual_start]
    visual_positions = absolute_positions[visual_start : visual_start + visual_token_count].index_select(
        0, keep_visual_rel_indices
    )
    tail_positions = absolute_positions[visual_start + visual_token_count :]
    new_positions = torch.cat([prefix_positions, visual_positions, tail_positions], dim=0)
    return new_hidden_states, new_attention_mask, new_positions, int(keep_visual_rel_indices.numel())


def _adaptprune_causal_lm_forward(
    self,
    multimodal_embeddings: torch.Tensor,
    multimodal_attention_mask: Optional[torch.Tensor],
    num_visual_patch_tokens: int,
    extra_projected_tokens_count: int,
    prompt_token_count: int,
    compute_logits: bool,
) -> AdaptPruneForwardResult:
    language_model = self.language_model
    decoder = language_model.model
    cfg = getattr(self, "_adaptprune_cfg", None)
    if cfg is None:
        raise ValueError("AdaptPrune configuration is missing on the loaded model.")

    prune_layer = int(getattr(cfg, "adaptprune_layer", 2))
    if prune_layer >= len(decoder.layers):
        raise ValueError(f"adaptprune_layer={prune_layer} out of range for {len(decoder.layers)} decoder layers.")

    batch_size, total_seq_len, _ = multimodal_embeddings.shape
    if batch_size != 1:
        raise ValueError("AdaptPrune runtime patch only supports batch size 1 during inference.")

    device = multimodal_embeddings.device
    hidden_states = multimodal_embeddings
    current_attention_mask = multimodal_attention_mask
    if current_attention_mask is None:
        current_attention_mask = torch.ones((batch_size, total_seq_len), dtype=torch.long, device=device)
    absolute_positions = torch.arange(total_seq_len, device=device, dtype=torch.long)
    visual_start = 1
    current_visual_tokens = int(num_visual_patch_tokens)

    per_image_patches = int(self.vision_backbone.get_num_patches())
    num_images = int(self.vision_backbone.get_num_images_in_input())
    probe_attention = None

    for layer_idx, decoder_layer in enumerate(decoder.layers):
        if layer_idx == prune_layer and current_visual_tokens > 1:
            if probe_attention is None:
                raise ValueError("AdaptPrune did not receive attention weights from the previous layer.")
            key_layer = decoder_layer.self_attn.k_proj(hidden_states)
            keep_rel = _select_keep_indices_by_image(
                cfg,
                probe_attention,
                key_layer,
                visual_start=visual_start,
                visual_token_count=current_visual_tokens,
                extra_projected_tokens_count=int(extra_projected_tokens_count),
                prompt_token_count=int(prompt_token_count),
                num_images=num_images,
                per_image_patches=per_image_patches,
            )
            if keep_rel.numel() < current_visual_tokens:
                hidden_states, current_attention_mask, absolute_positions, current_visual_tokens = _build_pruned_sequence(
                    hidden_states,
                    current_attention_mask,
                    absolute_positions,
                    visual_start=visual_start,
                    visual_token_count=current_visual_tokens,
                    keep_visual_rel_indices=keep_rel,
                )

        current_len = hidden_states.shape[1]
        cache_positions = torch.arange(current_len, device=device, dtype=torch.long)
        position_ids = absolute_positions.unsqueeze(0)
        causal_mask = decoder._update_causal_mask(current_attention_mask, hidden_states, cache_positions, 0)

        need_probe = layer_idx == prune_layer - 1
        layer_outputs = decoder_layer(
            hidden_states,
            attention_mask=causal_mask,
            position_ids=position_ids,
            past_key_value=None,
            output_attentions=need_probe,
            use_cache=False,
            cache_position=cache_positions,
        )
        hidden_states = layer_outputs[0]
        if need_probe:
            probe_attention = layer_outputs[1]

    hidden_states = decoder.norm(hidden_states)
    logits = _compute_logits_from_hidden_states(language_model, hidden_states) if compute_logits else None
    return AdaptPruneForwardResult(
        hidden_states=hidden_states,
        logits=logits,
        current_visual_tokens=int(current_visual_tokens),
        used_adaptprune=True,
    )


def _forward_language_model_with_optional_adaptprune(
    self,
    multimodal_embeddings: torch.Tensor,
    multimodal_attention_mask: Optional[torch.Tensor],
    num_visual_patch_tokens: int,
    extra_projected_tokens_count: int,
    prompt_token_count: int,
    compute_logits: bool,
) -> AdaptPruneForwardResult:
    if not _should_use_adaptprune(self, num_visual_patch_tokens, prompt_token_count) or not _supports_language_model(self):
        result = _dense_causal_lm_forward(self, multimodal_embeddings, multimodal_attention_mask, num_visual_patch_tokens, compute_logits)
        _record_metrics(self, int(num_visual_patch_tokens), int(num_visual_patch_tokens), used_adaptprune=False)
        return result

    try:
        result = _adaptprune_causal_lm_forward(
            self,
            multimodal_embeddings,
            multimodal_attention_mask,
            num_visual_patch_tokens,
            extra_projected_tokens_count,
            prompt_token_count,
            compute_logits,
        )
        _record_metrics(
            self,
            int(num_visual_patch_tokens),
            int(result.current_visual_tokens),
            used_adaptprune=bool(result.used_adaptprune),
            fallback=False,
        )
        return result
    except Exception as exc:
        if not getattr(self, "_adaptprune_logged_fallback", False):
            print(f"[AdaptPrune] fallback to dense language-model forward: {exc}")
            self._adaptprune_logged_fallback = True
        result = _dense_causal_lm_forward(self, multimodal_embeddings, multimodal_attention_mask, num_visual_patch_tokens, compute_logits)
        _record_metrics(self, int(num_visual_patch_tokens), int(num_visual_patch_tokens), used_adaptprune=False, fallback=True)
        return result


def _extract_action_hidden_states(sequence_hidden_states: torch.Tensor) -> torch.Tensor:
    action_hidden_states = sequence_hidden_states[:, -ACTION_DIM * NUM_ACTIONS_CHUNK - 1 : -1, :]
    expected_tokens = ACTION_DIM * NUM_ACTIONS_CHUNK
    if action_hidden_states.shape[1] != expected_tokens:
        raise RuntimeError(
            f"Expected {expected_tokens} action tokens after AdaptPrune pruning, got {action_hidden_states.shape[1]}."
        )
    return action_hidden_states


def _extract_action_logits(
    logits: torch.Tensor,
    current_visual_tokens: int,
    extra_projected_tokens_count: int,
    prompt_token_count: int,
) -> torch.Tensor:
    action_start = 1 + int(current_visual_tokens) + int(extra_projected_tokens_count) + int(prompt_token_count)
    action_end = action_start + ACTION_DIM * NUM_ACTIONS_CHUNK
    action_logits = logits[:, action_start:action_end, :]
    expected_tokens = ACTION_DIM * NUM_ACTIONS_CHUNK
    if action_logits.shape[1] != expected_tokens:
        raise RuntimeError(f"Expected {expected_tokens} action logits after AdaptPrune pruning, got {action_logits.shape[1]}.")
    return action_logits


def _regression_or_discrete_prediction(
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
    *unused_extra_args,
    **unused_extra_kwargs,
):
    del labels, unused_extra_args, unused_extra_kwargs

    num_visual_patch_tokens = int(self.vision_backbone.get_num_patches()) * int(self.vision_backbone.get_num_images_in_input())
    num_visual_patch_tokens = min(num_visual_patch_tokens, int(projected_patch_embeddings.shape[1]))
    extra_projected_tokens_count = int(projected_patch_embeddings.shape[1]) - int(num_visual_patch_tokens)

    if not _should_use_adaptprune(self, num_visual_patch_tokens, int(num_prompt_tokens)):
        original_forward = getattr(self, "_adaptprune_original_regression_prediction", None)
        if original_forward is None:
            raise ValueError("Original AdaptPrune regression forward is missing.")
        result = original_forward(
            input_embeddings,
            all_actions_mask,
            projected_patch_embeddings,
            attention_mask,
            None,
            num_patches,
            int(num_prompt_tokens),
            action_head,
            past_key_values,
        )
        _record_metrics(
            self,
            int(num_visual_patch_tokens),
            int(num_visual_patch_tokens),
            used_adaptprune=False,
            fallback=False,
        )
        return result

    all_actions_mask = all_actions_mask.unsqueeze(-1)
    input_embeddings = input_embeddings * ~all_actions_mask
    multimodal_embeddings, multimodal_attention_mask = self._build_multimodal_attention(
        input_embeddings,
        projected_patch_embeddings,
        attention_mask,
    )

    language_model_output = _forward_language_model_with_optional_adaptprune(
        self,
        multimodal_embeddings,
        multimodal_attention_mask,
        num_visual_patch_tokens,
        extra_projected_tokens_count,
        int(num_prompt_tokens),
        compute_logits=action_head is None,
    )

    actions_hidden_states = _extract_action_hidden_states(language_model_output.hidden_states)
    if action_head is not None:
        normalized_actions = action_head.predict_action(actions_hidden_states)
        normalized_actions = normalized_actions.reshape(NUM_ACTIONS_CHUNK, ACTION_DIM)
        normalized_actions = normalized_actions.float().cpu().detach().numpy()
    else:
        action_logits = _extract_action_logits(
            language_model_output.logits,
            current_visual_tokens=language_model_output.current_visual_tokens,
            extra_projected_tokens_count=extra_projected_tokens_count,
            prompt_token_count=int(num_prompt_tokens),
        )
        predicted_action_token_ids = action_logits.argmax(dim=2).cpu().numpy()
        discretized_actions = self.vocab_size - predicted_action_token_ids
        discretized_actions = np.clip(discretized_actions - 1, a_min=0, a_max=self.bin_centers.shape[0] - 1)
        normalized_actions = self.bin_centers[discretized_actions].reshape(NUM_ACTIONS_CHUNK, ACTION_DIM)

    return normalized_actions, actions_hidden_states, None


def _run_diffusion_prediction(
    self,
    input_embeddings,
    all_actions_mask,
    noise,
    action_head,
    projected_patch_embeddings,
    labels,
    attention_mask,
    num_patches,
    num_prompt_tokens,
    noisy_action_projector,
):
    num_visual_patch_tokens = int(self.vision_backbone.get_num_patches()) * int(self.vision_backbone.get_num_images_in_input())
    num_visual_patch_tokens = min(num_visual_patch_tokens, int(projected_patch_embeddings.shape[1]))
    extra_projected_tokens_count = int(projected_patch_embeddings.shape[1]) - int(num_visual_patch_tokens)

    if not _should_use_adaptprune(self, num_visual_patch_tokens, int(num_prompt_tokens)):
        original_forward = getattr(self, "_adaptprune_original_diffusion_prediction", None)
        if original_forward is None:
            raise ValueError("Original AdaptPrune diffusion forward is missing.")
        result = original_forward(
            input_embeddings,
            all_actions_mask,
            noise,
            action_head,
            projected_patch_embeddings,
            labels,
            attention_mask,
            num_patches,
            int(num_prompt_tokens),
            noisy_action_projector,
        )
        _record_metrics(
            self,
            int(num_visual_patch_tokens),
            int(num_visual_patch_tokens),
            used_adaptprune=False,
            fallback=False,
        )
        return result

    orig_projected_patch_embeddings = projected_patch_embeddings.clone()
    curr_noisy_actions = noise
    all_actions_mask_bool = all_actions_mask

    for t in action_head.noise_scheduler.timesteps:
        timesteps = torch.Tensor([t]).to(labels.device)
        diffusion_timestep_embeddings = (
            action_head.time_encoder(timesteps).to(curr_noisy_actions.dtype).to(curr_noisy_actions.device)
        ).unsqueeze(1)

        projected_patch_embeddings = torch.cat((orig_projected_patch_embeddings, diffusion_timestep_embeddings), dim=1)

        batch_size = curr_noisy_actions.shape[0]
        orig_shape = curr_noisy_actions.shape
        curr_noisy_actions = curr_noisy_actions.reshape(batch_size, -1).unsqueeze(-1)
        noisy_action_features = noisy_action_projector(curr_noisy_actions)
        curr_noisy_actions = curr_noisy_actions.reshape(orig_shape)

        replaced_input_embeddings = self._replace_input_embeddings(
            input_embeddings.clone(),
            all_actions_mask_bool,
            noisy_action_features,
        )
        multimodal_embeddings, multimodal_attention_mask = self._build_multimodal_attention(
            replaced_input_embeddings,
            projected_patch_embeddings,
            attention_mask,
        )

        language_model_output = _forward_language_model_with_optional_adaptprune(
            self,
            multimodal_embeddings,
            multimodal_attention_mask,
            num_visual_patch_tokens,
            extra_projected_tokens_count,
            int(num_prompt_tokens),
            compute_logits=False,
        )
        actions_hidden_states = _extract_action_hidden_states(language_model_output.hidden_states)
        noise_pred = action_head.predict_noise(actions_hidden_states)
        curr_noisy_actions = action_head.noise_scheduler.step(noise_pred, t, curr_noisy_actions).prev_sample

    curr_noisy_actions = curr_noisy_actions.reshape(NUM_ACTIONS_CHUNK, ACTION_DIM)
    return curr_noisy_actions.float().cpu().detach().numpy(), actions_hidden_states
