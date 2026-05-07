"""Runtime monkeypatches that transplant SparseVLM into openvla-oft.

This module keeps all SparseVLM-specific logic inside `pruner/SparseVLM` and
patches the loaded OpenVLA model at runtime instead of editing the main source tree.
"""

import math
from dataclasses import dataclass
from types import MethodType
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

from backend import load_backend_modules
from config import (
    resolve_sparsevlm_layer_tokens,
    resolve_sparsevlm_pruning_layers,
    sparsevlm_merge_enabled,
)

_backend = load_backend_modules()
openvla_utils = _backend.openvla_utils

from prismatic.vla.constants import ACTION_DIM, NUM_ACTIONS_CHUNK


@dataclass
class SparseVLMetrics:
    model_calls: int = 0
    sparse_calls: int = 0
    prune_stages: int = 0
    visual_tokens_before: int = 0
    visual_tokens_after: int = 0
    last_visual_tokens: int = 0
    fallback_calls: int = 0
    version: str = "1_0"
    retained_tokens_target: int = 192
    guidance_mode: str = "prompt_action"

    def snapshot(self) -> Dict[str, float]:
        weighted_prune_rate = 0.0
        if self.visual_tokens_before > 0:
            weighted_prune_rate = 1.0 - (self.visual_tokens_after / float(self.visual_tokens_before))
        return {
            "model_calls": float(self.model_calls),
            "sparse_calls": float(self.sparse_calls),
            "prune_stages": float(self.prune_stages),
            "visual_tokens_before": float(self.visual_tokens_before),
            "visual_tokens_after": float(self.visual_tokens_after),
            "weighted_prune_rate": float(weighted_prune_rate),
            "last_visual_tokens": float(self.last_visual_tokens),
            "fallback_calls": float(self.fallback_calls),
            "version": self.version,
            "retained_tokens_target": float(self.retained_tokens_target),
            "guidance_mode": self.guidance_mode,
        }


@dataclass
class SparseForwardResult:
    hidden_states: torch.Tensor
    logits: Optional[torch.Tensor]
    current_visual_tokens: int
    used_sparse: bool
    prune_stages: int
    stage_tokens_before: int
    stage_tokens_after: int


def apply_runtime_patches() -> None:
    if getattr(openvla_utils, "_sparsevlm_runtime_patch_applied", False):
        return
    openvla_utils._sparsevlm_runtime_patch_applied = True


def patch_loaded_model(model: torch.nn.Module, cfg: Optional[Any] = None) -> None:
    target = model.module if hasattr(model, "module") else model
    if cfg is not None:
        target._sparsevlm_cfg = cfg

    if not hasattr(target, "_sparsevlm_metrics"):
        target._sparsevlm_metrics = SparseVLMetrics()
    if not hasattr(target, "_sparsevlm_episode_metrics"):
        target._sparsevlm_episode_metrics = SparseVLMetrics()
    if not hasattr(target, "_sparsevlm_logged_fallback"):
        target._sparsevlm_logged_fallback = False

    if not hasattr(target, "_sparsevlm_original_regression_prediction"):
        target._sparsevlm_original_regression_prediction = target._regression_or_discrete_prediction
        target._regression_or_discrete_prediction = MethodType(_regression_or_discrete_prediction, target)

    if not hasattr(target, "_sparsevlm_original_diffusion_prediction"):
        target._sparsevlm_original_diffusion_prediction = target._run_diffusion_prediction
        target._run_diffusion_prediction = MethodType(_run_diffusion_prediction, target)

    if not hasattr(target, "sparsevlm_reset_episode"):
        target.sparsevlm_reset_episode = MethodType(sparsevlm_reset_episode, target)
    if not hasattr(target, "sparsevlm_get_metrics"):
        target.sparsevlm_get_metrics = MethodType(sparsevlm_get_metrics, target)

    target.sparsevlm_reset_episode()


def get_runtime_metrics(model: torch.nn.Module) -> Dict[str, float]:
    target = model.module if hasattr(model, "module") else model
    metrics = getattr(target, "_sparsevlm_metrics", None)
    if metrics is None:
        return SparseVLMetrics().snapshot()
    return metrics.snapshot()


def get_episode_runtime_metrics(model: torch.nn.Module) -> Dict[str, float]:
    target = model.module if hasattr(model, "module") else model
    metrics = getattr(target, "_sparsevlm_episode_metrics", None)
    if metrics is None:
        return SparseVLMetrics().snapshot()
    return metrics.snapshot()


def sparsevlm_reset_episode(self) -> None:
    self._sparsevlm_episode_metrics = SparseVLMetrics()


def sparsevlm_get_metrics(self) -> Dict[str, float]:
    return self._sparsevlm_metrics.snapshot()


def _record_sparse_metrics(
    self,
    used_sparse: bool,
    prune_stages: int,
    tokens_before: int,
    tokens_after: int,
    last_visual_tokens: int,
    fallback: bool = False,
) -> None:
    cfg = getattr(self, "_sparsevlm_cfg", None)
    version = "1_0" if cfg is None else str(getattr(cfg, "sparsevlm_version", "1_0"))
    retained_tokens_target = 192 if cfg is None else int(getattr(cfg, "sparsevlm_retained_tokens", 192))
    guidance_mode = "prompt_action" if cfg is None else str(getattr(cfg, "sparsevlm_guidance_mode", "prompt_action"))

    for metrics_name in ("_sparsevlm_metrics", "_sparsevlm_episode_metrics"):
        metrics = getattr(self, metrics_name, None)
        if metrics is None:
            continue
        metrics.model_calls += 1
        metrics.visual_tokens_before += int(tokens_before)
        metrics.visual_tokens_after += int(tokens_after)
        metrics.last_visual_tokens = int(last_visual_tokens)
        metrics.version = version
        metrics.retained_tokens_target = retained_tokens_target
        metrics.guidance_mode = guidance_mode
        if used_sparse:
            metrics.sparse_calls += 1
            metrics.prune_stages += int(prune_stages)
        if fallback:
            metrics.fallback_calls += 1


def _supports_sparsevlm_language_model(self) -> bool:
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


def _should_use_sparsevlm(self, num_visual_patch_tokens: int, prompt_token_count: int) -> bool:
    cfg = getattr(self, "_sparsevlm_cfg", None)
    if cfg is None:
        return False
    if not bool(getattr(cfg, "use_sparse_vlm", False)):
        return False
    if num_visual_patch_tokens <= 1:
        return False
    if prompt_token_count <= 0:
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
) -> SparseForwardResult:
    output = self.language_model(
        input_ids=None,
        attention_mask=multimodal_attention_mask,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=multimodal_embeddings,
        labels=None,
        use_cache=True,
        output_attentions=True,
        output_hidden_states=True,
        return_dict=True,
    )
    return SparseForwardResult(
        hidden_states=output.hidden_states[-1],
        logits=output.logits if compute_logits else None,
        current_visual_tokens=int(num_visual_patch_tokens),
        used_sparse=False,
        prune_stages=0,
        stage_tokens_before=0,
        stage_tokens_after=0,
    )


def _index_points(points: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    device = points.device
    batch_size = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(batch_size, dtype=torch.long, device=device).view(view_shape).repeat(repeat_shape)
    return points[batch_indices, idx, :]


def _cluster_and_merge(x: torch.Tensor, cluster_num: int) -> torch.Tensor:
    batch_size, num_tokens, channels = x.shape

    x1 = x.unsqueeze(2)
    x2 = x.unsqueeze(1)
    distance = (x1 - x2).norm(dim=-1, p=2)
    dist_matrix = distance / (channels ** 0.5)

    dist_nearest, _ = torch.topk(dist_matrix, k=min(cluster_num, num_tokens), dim=-1, largest=False)
    density = (-(dist_nearest ** 2).mean(dim=-1)).exp()
    density = density + torch.rand(density.shape, device=density.device, dtype=density.dtype) * 1e-6

    mask = (density[:, None, :] > density[:, :, None]).type(x.dtype)
    dist_max = dist_matrix.flatten(1).max(dim=-1)[0][:, None, None]
    dist, _ = (dist_matrix * mask + dist_max * (1 - mask)).min(dim=-1)

    score = dist * density
    _, index_down = torch.topk(score, k=min(cluster_num, num_tokens), dim=-1)

    dist_matrix_down = _index_points(dist_matrix, index_down)
    idx_cluster = dist_matrix_down.argmin(dim=1)

    idx_batch = torch.arange(batch_size, device=x.device)[:, None].expand(batch_size, index_down.shape[1])
    idx_tmp = torch.arange(index_down.shape[1], device=x.device)[None, :].expand(batch_size, index_down.shape[1])
    idx_cluster[idx_batch.reshape(-1), index_down.reshape(-1)] = idx_tmp.reshape(-1)

    token_weight = x.new_ones(batch_size, num_tokens, 1)
    idx = idx_cluster + torch.arange(batch_size, device=x.device)[:, None] * index_down.shape[1]

    all_weight = token_weight.new_zeros(batch_size * index_down.shape[1], 1)
    all_weight.index_add_(dim=0, index=idx.reshape(batch_size * num_tokens), source=token_weight.reshape(batch_size * num_tokens, 1))
    all_weight = all_weight + 1e-6
    norm_weight = token_weight / all_weight[idx]

    x_merged = x.new_zeros(batch_size * index_down.shape[1], channels)
    source = x * norm_weight
    x_merged.index_add_(dim=0, index=idx.reshape(batch_size * num_tokens), source=source.reshape(batch_size * num_tokens, channels).type(x.dtype))
    return x_merged.reshape(batch_size, index_down.shape[1], channels)


def _compute_prompt_rater_indices(
    hidden_states: torch.Tensor,
    visual_start: int,
    visual_token_count: int,
    prompt_start: int,
    prompt_token_count: int,
) -> torch.Tensor:
    visual_tokens = hidden_states[:, visual_start : visual_start + visual_token_count, :]
    prompt_tokens = hidden_states[:, prompt_start : prompt_start + prompt_token_count, :]
    if visual_tokens.shape[1] <= 0 or prompt_tokens.shape[1] <= 0:
        return torch.zeros((0,), dtype=torch.long, device=hidden_states.device)

    visual_to_text = torch.matmul(visual_tokens, prompt_tokens.transpose(1, 2))
    visual_to_text = torch.softmax(visual_to_text, dim=2).mean(dim=1)
    scores = visual_to_text[0]
    indices = torch.where(scores > scores.mean())[0]
    if indices.numel() == 0:
        indices = scores.topk(k=1).indices
    return indices.to(dtype=torch.long)


def _build_guidance_relative_indices(
    cfg: Any,
    prompt_rater_rel_indices: torch.Tensor,
    prompt_token_count: int,
    action_token_count: int,
) -> torch.Tensor:
    guidance_mode = "prompt_action" if cfg is None else str(getattr(cfg, "sparsevlm_guidance_mode", "prompt_action"))
    prompt_rater_rel_indices = prompt_rater_rel_indices.to(dtype=torch.long)

    if guidance_mode == "prompt":
        return prompt_rater_rel_indices

    if guidance_mode == "prompt_action":
        action_rel_indices = torch.arange(
            int(prompt_token_count),
            int(prompt_token_count) + int(action_token_count),
            device=prompt_rater_rel_indices.device,
            dtype=torch.long,
        )
        return torch.unique(torch.cat([prompt_rater_rel_indices, action_rel_indices], dim=0), sorted=True)

    raise ValueError(f"Unsupported SparseVLM guidance mode: {guidance_mode}")


def _compute_rope_correction_weights(attn_module, hidden_states: torch.Tensor, position_ids: torch.LongTensor) -> torch.Tensor:
    batch_size, seq_len, _ = hidden_states.shape
    num_heads = int(attn_module.num_heads)
    head_dim = int(attn_module.head_dim)

    query_states = hidden_states.new_ones(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
    key_states = hidden_states.new_ones(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
    cos, sin = attn_module.rotary_emb(key_states, position_ids)
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)
    rope_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(head_dim)
    safe_rope = torch.where(rope_weights.abs() < 1e-6, torch.full_like(rope_weights, 1e-6), rope_weights)
    return safe_rope


def _select_priority_heads(
    attn_weights: torch.Tensor,
    text_token_indices: torch.Tensor,
    visual_start: int,
    visual_token_count: int,
    top_head_count: int,
) -> torch.Tensor:
    if attn_weights.dim() != 4 or text_token_indices.numel() == 0:
        return attn_weights
    selected_count = min(int(top_head_count), int(attn_weights.shape[1]))
    if selected_count <= 0:
        return attn_weights
    head_text_to_visual = attn_weights[0][:, text_token_indices, visual_start : visual_start + visual_token_count]
    sum_attn_per_head = head_text_to_visual.sum((1, 2))
    selected_heads = sum_attn_per_head.topk(selected_count).indices
    return attn_weights[:, selected_heads, :, :]


def _compute_keep_mask(
    attn_weights: torch.Tensor,
    visual_start: int,
    visual_token_count: int,
    guidance_start: int,
    guidance_rel_indices: torch.Tensor,
    keep_target: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if attn_weights.dim() == 4:
        attn_weights = attn_weights.mean(dim=1)
    elif attn_weights.dim() != 3:
        raise ValueError(f"Unexpected attention tensor shape for SparseVLM keep mask: {tuple(attn_weights.shape)}")

    guidance_token_indices = guidance_rel_indices + int(guidance_start)
    relation_vis_text = attn_weights[:, guidance_token_indices, visual_start : visual_start + visual_token_count]
    relation_vis_text = relation_vis_text.mean(dim=1)

    keep_total = min(int(keep_target), int(max(visual_token_count - 1, 0)))
    keep_total = max(0, keep_total)

    keep_mask = torch.zeros((visual_token_count,), dtype=torch.bool, device=relation_vis_text.device)
    if keep_total > 0:
        keep_indices = torch.topk(relation_vis_text[0], k=keep_total, dim=0).indices
        keep_mask[keep_indices] = True
    return keep_mask, relation_vis_text[0]


def _build_sparse_sequence_no_merge(
    hidden_states: torch.Tensor,
    absolute_positions: torch.LongTensor,
    visual_start: int,
    visual_token_count: int,
    keep_mask: torch.Tensor,
    version: str,
) -> tuple[torch.Tensor, torch.LongTensor, int]:
    keep_indices = torch.where(keep_mask)[0].to(device=hidden_states.device, dtype=torch.long)

    prefix_states = hidden_states[:, :visual_start, :]
    visual_states = hidden_states[:, visual_start : visual_start + visual_token_count, :]
    kept_visual_states = visual_states.index_select(1, keep_indices) if keep_indices.numel() > 0 else visual_states[:, :0, :]
    tail_states = hidden_states[:, visual_start + visual_token_count :, :]
    new_hidden_states = torch.cat([prefix_states, kept_visual_states, tail_states], dim=1)

    if version == "2_0":
        prefix_positions = absolute_positions[:visual_start]
        visual_positions = absolute_positions[visual_start : visual_start + visual_token_count]
        kept_visual_positions = visual_positions.index_select(0, keep_indices) if keep_indices.numel() > 0 else visual_positions[:0]
        tail_positions = absolute_positions[visual_start + visual_token_count :]
        new_absolute_positions = torch.cat([prefix_positions, kept_visual_positions, tail_positions], dim=0)
    else:
        new_absolute_positions = torch.arange(new_hidden_states.shape[1], device=hidden_states.device, dtype=torch.long)

    return new_hidden_states, new_absolute_positions, int(keep_indices.numel())


def _build_sparse_sequence_with_merge(
    hidden_states: torch.Tensor,
    visual_start: int,
    visual_token_count: int,
    keep_mask: torch.Tensor,
    relation_vis_text: torch.Tensor,
    cfg: Any,
) -> tuple[torch.Tensor, torch.LongTensor, int]:
    keep_indices = torch.where(keep_mask)[0].to(device=hidden_states.device, dtype=torch.long)
    remove_indices = torch.where(~keep_mask)[0].to(device=hidden_states.device, dtype=torch.long)

    prefix_states = hidden_states[:, :visual_start, :]
    visual_states = hidden_states[:, visual_start : visual_start + visual_token_count, :]
    kept_visual_states = visual_states.index_select(1, keep_indices) if keep_indices.numel() > 0 else visual_states[:, :0, :]
    tail_states = hidden_states[:, visual_start + visual_token_count :, :]

    if remove_indices.numel() <= 0:
        new_hidden_states = torch.cat([prefix_states, kept_visual_states, tail_states], dim=1)
        new_positions = torch.arange(new_hidden_states.shape[1], device=hidden_states.device, dtype=torch.long)
        return new_hidden_states, new_positions, int(keep_indices.numel())

    removed_scores = relation_vis_text.index_select(0, remove_indices)
    merge_stage1_ratio = float(getattr(cfg, "sparsevlm_merge_stage1_ratio", 0.3))
    merge_stage1_count = int(remove_indices.numel() * merge_stage1_ratio) + 1
    merge_stage1_count = min(merge_stage1_count, int(remove_indices.numel()))
    merge_stage2_relative = removed_scores.topk(merge_stage1_count).indices
    merge_indices = remove_indices.index_select(0, merge_stage2_relative)
    merge_states = visual_states.index_select(1, merge_indices)

    cluster_divisor = int(getattr(cfg, "sparsevlm_merge_cluster_divisor", 10))
    cluster_num = int(merge_states.shape[1] / cluster_divisor) + 1
    cluster_num = max(1, min(cluster_num, int(merge_states.shape[1])))
    merged_states = _cluster_and_merge(merge_states, cluster_num)

    new_hidden_states = torch.cat([prefix_states, kept_visual_states, merged_states, tail_states], dim=1)
    new_positions = torch.arange(new_hidden_states.shape[1], device=hidden_states.device, dtype=torch.long)
    return new_hidden_states, new_positions, int(keep_indices.numel() + cluster_num)


def _extract_action_hidden_states(
    sequence_hidden_states: torch.Tensor,
    current_visual_tokens: int,
    extra_projected_tokens_count: int,
    prompt_token_count: int,
) -> torch.Tensor:
    del current_visual_tokens, extra_projected_tokens_count, prompt_token_count
    actions_hidden_states = sequence_hidden_states[:, -ACTION_DIM * NUM_ACTIONS_CHUNK - 1 : -1, :]
    expected_tokens = ACTION_DIM * NUM_ACTIONS_CHUNK
    if actions_hidden_states.shape[1] != expected_tokens:
        raise RuntimeError(
            f"Expected {expected_tokens} action tokens after SparseVLM pruning, got {actions_hidden_states.shape[1]}."
        )
    return actions_hidden_states


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
        raise RuntimeError(
            f"Expected {expected_tokens} action logits after SparseVLM pruning, got {action_logits.shape[1]}."
        )
    return action_logits


def _sparsevlm_causal_lm_forward(
    self,
    multimodal_embeddings: torch.Tensor,
    multimodal_attention_mask: Optional[torch.Tensor],
    num_visual_patch_tokens: int,
    extra_projected_tokens_count: int,
    prompt_token_count: int,
    compute_logits: bool,
) -> SparseForwardResult:
    language_model = self.language_model
    decoder = language_model.model
    cfg = getattr(self, "_sparsevlm_cfg", None)
    if cfg is None:
        raise ValueError("SparseVLM configuration is missing on the loaded model.")

    version = str(getattr(cfg, "sparsevlm_version", "1_0"))
    pruning_layers = resolve_sparsevlm_pruning_layers(cfg)
    per_layer_token_targets = resolve_sparsevlm_layer_tokens(cfg)
    layer_token_target_map = {layer: per_layer_token_targets[idx] for idx, layer in enumerate(pruning_layers)}
    merge_enabled = sparsevlm_merge_enabled(cfg)
    priority_head_count = int(getattr(cfg, "sparsevlm_priority_head_count", 14))
    action_token_count = ACTION_DIM * NUM_ACTIONS_CHUNK

    batch_size, total_seq_len, _ = multimodal_embeddings.shape
    if batch_size != 1:
        raise ValueError("SparseVLM runtime patch only supports batch size 1 during inference.")

    if max(pruning_layers) >= len(decoder.layers):
        raise ValueError(
            f"SparseVLM pruning layer index out of range for decoder with {len(decoder.layers)} layers: "
            f"pruning_layers={pruning_layers}"
        )

    current_attention_mask = multimodal_attention_mask
    if current_attention_mask is None:
        current_attention_mask = torch.ones((batch_size, total_seq_len), dtype=torch.long, device=multimodal_embeddings.device)

    hidden_states = multimodal_embeddings
    absolute_positions = torch.arange(total_seq_len, device=multimodal_embeddings.device, dtype=torch.long)
    current_visual_tokens = int(num_visual_patch_tokens)
    visual_start = 1
    prompt_start = visual_start + current_visual_tokens + int(extra_projected_tokens_count)
    prompt_rater_rel_indices = _compute_prompt_rater_indices(
        hidden_states,
        visual_start=visual_start,
        visual_token_count=current_visual_tokens,
        prompt_start=prompt_start,
        prompt_token_count=prompt_token_count,
    )
    if prompt_rater_rel_indices.numel() <= 0:
        raise ValueError("SparseVLM prompt-rater selection produced no prompt tokens.")

    guidance_rel_indices = _build_guidance_relative_indices(
        cfg,
        prompt_rater_rel_indices=prompt_rater_rel_indices,
        prompt_token_count=int(prompt_token_count),
        action_token_count=int(action_token_count),
    )
    if guidance_rel_indices.numel() <= 0:
        raise ValueError("SparseVLM guidance selection produced no tokens.")

    prune_stages = 0
    stage_tokens_before = 0
    stage_tokens_after = 0

    for layer_idx, decoder_layer in enumerate(decoder.layers):
        current_len = hidden_states.shape[1]
        cache_positions = torch.arange(current_len, device=hidden_states.device, dtype=torch.long)
        current_position_ids = absolute_positions.unsqueeze(0)
        causal_mask = decoder._update_causal_mask(current_attention_mask, hidden_states, cache_positions, 0)

        if layer_idx in layer_token_target_map and current_visual_tokens > 1 and current_len != 1:
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=current_position_ids,
                past_key_value=None,
                output_attentions=True,
                use_cache=False,
                cache_position=cache_positions,
            )
            if len(layer_outputs) < 2 or layer_outputs[1] is None:
                raise ValueError("SparseVLM requires attention weights but the current decoder layer did not return them.")

            hidden_after_layer = layer_outputs[0]
            attn_weights = layer_outputs[1]
            prompt_start = visual_start + current_visual_tokens + int(extra_projected_tokens_count)

            if version == "2_0":
                rope_weights = _compute_rope_correction_weights(
                    decoder_layer.self_attn,
                    hidden_states,
                    current_position_ids,
                )
                attn_weights = attn_weights / rope_weights
                text_indices = guidance_rel_indices + int(prompt_start)
                attn_weights = _select_priority_heads(
                    attn_weights,
                    text_indices,
                    visual_start=visual_start,
                    visual_token_count=current_visual_tokens,
                    top_head_count=priority_head_count,
                )

            keep_mask, relation_vis_text = _compute_keep_mask(
                attn_weights,
                visual_start=visual_start,
                visual_token_count=current_visual_tokens,
                guidance_start=prompt_start,
                guidance_rel_indices=guidance_rel_indices,
                keep_target=int(layer_token_target_map[layer_idx]),
            )

            stage_tokens_before += int(current_visual_tokens)
            if merge_enabled and relation_vis_text.numel() > 0:
                hidden_states, absolute_positions, current_visual_tokens = _build_sparse_sequence_with_merge(
                    hidden_after_layer,
                    visual_start=visual_start,
                    visual_token_count=current_visual_tokens,
                    keep_mask=keep_mask,
                    relation_vis_text=relation_vis_text,
                    cfg=cfg,
                )
            else:
                hidden_states, absolute_positions, current_visual_tokens = _build_sparse_sequence_no_merge(
                    hidden_after_layer,
                    absolute_positions=absolute_positions,
                    visual_start=visual_start,
                    visual_token_count=current_visual_tokens,
                    keep_mask=keep_mask,
                    version=version,
                )

            current_attention_mask = torch.ones(
                (batch_size, hidden_states.shape[1]),
                dtype=current_attention_mask.dtype,
                device=hidden_states.device,
            )
            stage_tokens_after += int(current_visual_tokens)
            prune_stages += 1
        else:
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=current_position_ids,
                past_key_value=None,
                output_attentions=True,
                use_cache=False,
                cache_position=cache_positions,
            )
            hidden_states = layer_outputs[0]

    hidden_states = decoder.norm(hidden_states)
    logits = _compute_logits_from_hidden_states(language_model, hidden_states) if compute_logits else None
    return SparseForwardResult(
        hidden_states=hidden_states,
        logits=logits,
        current_visual_tokens=int(current_visual_tokens),
        used_sparse=True,
        prune_stages=prune_stages,
        stage_tokens_before=stage_tokens_before,
        stage_tokens_after=stage_tokens_after,
    )


def _forward_language_model_with_optional_sparsevlm(
    self,
    multimodal_embeddings: torch.Tensor,
    multimodal_attention_mask: Optional[torch.Tensor],
    num_visual_patch_tokens: int,
    extra_projected_tokens_count: int,
    prompt_token_count: int,
    compute_logits: bool,
) -> SparseForwardResult:
    if not _should_use_sparsevlm(self, num_visual_patch_tokens, prompt_token_count) or not _supports_sparsevlm_language_model(self):
        result = _dense_causal_lm_forward(
            self,
            multimodal_embeddings,
            multimodal_attention_mask,
            num_visual_patch_tokens=num_visual_patch_tokens,
            compute_logits=compute_logits,
        )
        _record_sparse_metrics(
            self,
            used_sparse=False,
            prune_stages=0,
            tokens_before=0,
            tokens_after=0,
            last_visual_tokens=result.current_visual_tokens,
            fallback=False,
        )
        return result

    try:
        result = _sparsevlm_causal_lm_forward(
            self,
            multimodal_embeddings,
            multimodal_attention_mask,
            num_visual_patch_tokens=num_visual_patch_tokens,
            extra_projected_tokens_count=extra_projected_tokens_count,
            prompt_token_count=prompt_token_count,
            compute_logits=compute_logits,
        )
        _record_sparse_metrics(
            self,
            used_sparse=True,
            prune_stages=result.prune_stages,
            tokens_before=result.stage_tokens_before,
            tokens_after=result.stage_tokens_after,
            last_visual_tokens=result.current_visual_tokens,
            fallback=False,
        )
        return result
    except Exception as exc:
        if not getattr(self, "_sparsevlm_logged_fallback", False):
            print(f"[SparseVLM] fallback to dense language-model forward: {exc}")
            self._sparsevlm_logged_fallback = True
        result = _dense_causal_lm_forward(
            self,
            multimodal_embeddings,
            multimodal_attention_mask,
            num_visual_patch_tokens=num_visual_patch_tokens,
            compute_logits=compute_logits,
        )
        _record_sparse_metrics(
            self,
            used_sparse=False,
            prune_stages=0,
            tokens_before=0,
            tokens_after=0,
            last_visual_tokens=result.current_visual_tokens,
            fallback=True,
        )
        return result


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
    del labels, num_patches, past_key_values, unused_extra_args, unused_extra_kwargs

    all_actions_mask = all_actions_mask.unsqueeze(-1)
    input_embeddings = input_embeddings * ~all_actions_mask

    multimodal_embeddings, multimodal_attention_mask = self._build_multimodal_attention(
        input_embeddings, projected_patch_embeddings, attention_mask
    )

    num_visual_patch_tokens = int(self.vision_backbone.get_num_patches()) * int(self.vision_backbone.get_num_images_in_input())
    num_visual_patch_tokens = min(num_visual_patch_tokens, int(projected_patch_embeddings.shape[1]))
    extra_projected_tokens_count = int(projected_patch_embeddings.shape[1]) - int(num_visual_patch_tokens)

    language_model_output = _forward_language_model_with_optional_sparsevlm(
        self,
        multimodal_embeddings,
        multimodal_attention_mask,
        num_visual_patch_tokens=num_visual_patch_tokens,
        extra_projected_tokens_count=extra_projected_tokens_count,
        prompt_token_count=int(num_prompt_tokens),
        compute_logits=action_head is None,
    )

    actions_hidden_states = _extract_action_hidden_states(
        language_model_output.hidden_states,
        current_visual_tokens=language_model_output.current_visual_tokens,
        extra_projected_tokens_count=extra_projected_tokens_count,
        prompt_token_count=int(num_prompt_tokens),
    )

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
        normalized_actions = self.bin_centers[discretized_actions]
        normalized_actions = normalized_actions.reshape(NUM_ACTIONS_CHUNK, ACTION_DIM)

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
    del num_patches

    orig_projected_patch_embeddings = projected_patch_embeddings.clone()
    curr_noisy_actions = noise

    for t in action_head.noise_scheduler.timesteps:
        timesteps = torch.Tensor([t]).to(labels.device)
        diffusion_timestep_embeddings = (
            action_head.time_encoder(timesteps).to(curr_noisy_actions.dtype).to(curr_noisy_actions.device)
        )
        diffusion_timestep_embeddings = diffusion_timestep_embeddings.unsqueeze(1)

        projected_patch_embeddings = torch.cat((orig_projected_patch_embeddings, diffusion_timestep_embeddings), dim=1)

        batch_size = curr_noisy_actions.shape[0]
        orig_shape = curr_noisy_actions.shape
        curr_noisy_actions = curr_noisy_actions.reshape(batch_size, -1).unsqueeze(-1)
        noisy_action_features = noisy_action_projector(curr_noisy_actions)
        curr_noisy_actions = curr_noisy_actions.reshape(orig_shape)

        replaced_input_embeddings = self._replace_input_embeddings(
            input_embeddings.clone(), all_actions_mask, noisy_action_features
        )
        multimodal_embeddings, multimodal_attention_mask = self._build_multimodal_attention(
            replaced_input_embeddings,
            projected_patch_embeddings,
            attention_mask,
        )

        num_visual_patch_tokens = int(self.vision_backbone.get_num_patches()) * int(self.vision_backbone.get_num_images_in_input())
        num_visual_patch_tokens = min(num_visual_patch_tokens, int(projected_patch_embeddings.shape[1]))
        extra_projected_tokens_count = int(projected_patch_embeddings.shape[1]) - int(num_visual_patch_tokens)

        language_model_output = _forward_language_model_with_optional_sparsevlm(
            self,
            multimodal_embeddings,
            multimodal_attention_mask,
            num_visual_patch_tokens=num_visual_patch_tokens,
            extra_projected_tokens_count=extra_projected_tokens_count,
            prompt_token_count=int(num_prompt_tokens),
            compute_logits=False,
        )

        actions_hidden_states = _extract_action_hidden_states(
            language_model_output.hidden_states,
            current_visual_tokens=language_model_output.current_visual_tokens,
            extra_projected_tokens_count=extra_projected_tokens_count,
            prompt_token_count=int(num_prompt_tokens),
        )

        noise_pred = action_head.predict_noise(actions_hidden_states)
        curr_noisy_actions = action_head.noise_scheduler.step(noise_pred, t, curr_noisy_actions).prev_sample

    curr_noisy_actions = curr_noisy_actions.reshape(NUM_ACTIONS_CHUNK, ACTION_DIM)
    return curr_noisy_actions.float().cpu().detach().numpy(), actions_hidden_states
