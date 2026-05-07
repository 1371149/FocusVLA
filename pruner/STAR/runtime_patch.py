"""Runtime monkeypatches that transplant STAR into OpenVLA-OFT."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from types import MethodType
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from backend import load_backend_modules

_backend = load_backend_modules()
openvla_utils = _backend.openvla_utils

from prismatic.vla.constants import ACTION_DIM, NUM_ACTIONS_CHUNK


@dataclass
class STARCache:
    calls: int = 0
    stage1_scores: Optional[torch.Tensor] = None
    current_vision_scores: Optional[Dict[str, List[torch.Tensor]]] = None
    patched_vision_modules: int = 0


@dataclass
class STARMetrics:
    model_calls: int = 0
    stage1_prune_events: int = 0
    stage2_prune_events: int = 0
    tokens_before: int = 0
    tokens_after_stage1: int = 0
    tokens_after_stage2: int = 0
    fallback_calls: int = 0
    vision_attention_hits: int = 0
    vision_attention_misses: int = 0
    last_visual_tokens: int = 0
    last_stage1_visual_tokens: int = 0
    last_final_visual_tokens: int = 0
    last_stage1_keep_ratio: float = 1.0
    last_keep_ratio: float = 1.0
    last_query_mode: str = "unknown"

    def snapshot(self) -> Dict[str, float]:
        stage1_rate = 0.0
        final_rate = 0.0
        if self.tokens_before > 0:
            stage1_rate = 1.0 - (self.tokens_after_stage1 / float(self.tokens_before))
            final_rate = 1.0 - (self.tokens_after_stage2 / float(self.tokens_before))
        return {
            "model_calls": float(self.model_calls),
            "stage1_prune_events": float(self.stage1_prune_events),
            "stage2_prune_events": float(self.stage2_prune_events),
            "tokens_before": float(self.tokens_before),
            "tokens_after_stage1": float(self.tokens_after_stage1),
            "tokens_after_stage2": float(self.tokens_after_stage2),
            "weighted_stage1_prune_rate": float(stage1_rate),
            "weighted_final_prune_rate": float(final_rate),
            "fallback_calls": float(self.fallback_calls),
            "vision_attention_hits": float(self.vision_attention_hits),
            "vision_attention_misses": float(self.vision_attention_misses),
            "last_visual_tokens": float(self.last_visual_tokens),
            "last_stage1_visual_tokens": float(self.last_stage1_visual_tokens),
            "last_final_visual_tokens": float(self.last_final_visual_tokens),
            "last_stage1_keep_ratio": float(self.last_stage1_keep_ratio),
            "last_keep_ratio": float(self.last_keep_ratio),
            "last_query_mode": self.last_query_mode,
        }


@dataclass
class STARForwardResult:
    hidden_states: torch.Tensor
    logits: Optional[torch.Tensor]
    current_visual_tokens: int
    stage1_visual_tokens: int
    used_star: bool
    query_mode: str
    vision_attention_hit: bool


def apply_runtime_patches() -> None:
    if getattr(openvla_utils, "_star_runtime_patch_applied", False):
        return
    openvla_utils._star_runtime_patch_applied = True


def patch_loaded_model(model: torch.nn.Module, cfg: Optional[Any] = None) -> None:
    target = model.module if hasattr(model, "module") else model
    if cfg is not None:
        target._star_cfg = cfg

    if not hasattr(target, "_star_metrics"):
        target._star_metrics = STARMetrics()
    if not hasattr(target, "_star_episode_metrics"):
        target._star_episode_metrics = STARMetrics()
    if not hasattr(target, "_star_cache"):
        target._star_cache = STARCache()
    if not hasattr(target, "_star_logged_fallback"):
        target._star_logged_fallback = False

    _patch_vision_attention_capture(target)

    if not hasattr(target, "_star_original_regression_prediction"):
        target._star_original_regression_prediction = target._regression_or_discrete_prediction
        target._regression_or_discrete_prediction = MethodType(_regression_or_discrete_prediction, target)

    if not hasattr(target, "star_reset_episode"):
        target.star_reset_episode = MethodType(star_reset_episode, target)
    if not hasattr(target, "star_get_metrics"):
        target.star_get_metrics = MethodType(star_get_metrics, target)

    target.star_reset_episode()


def get_runtime_metrics(model: torch.nn.Module) -> Dict[str, float]:
    target = model.module if hasattr(model, "module") else model
    metrics = getattr(target, "_star_metrics", None)
    if metrics is None:
        return STARMetrics().snapshot()
    return metrics.snapshot()


def get_episode_runtime_metrics(model: torch.nn.Module) -> Dict[str, float]:
    target = model.module if hasattr(model, "module") else model
    metrics = getattr(target, "_star_episode_metrics", None)
    if metrics is None:
        return STARMetrics().snapshot()
    return metrics.snapshot()


def star_reset_episode(self) -> None:
    self._star_episode_metrics = STARMetrics()
    cache = getattr(self, "_star_cache", None)
    if cache is not None:
        cache.calls = 0
        cache.stage1_scores = None
        cache.current_vision_scores = None


def star_get_metrics(self) -> Dict[str, float]:
    return self._star_metrics.snapshot()


def _record_metrics(
    self,
    visual_tokens_before: int,
    stage1_visual_tokens: int,
    final_visual_tokens: int,
    used_star: bool,
    query_mode: str,
    vision_attention_hit: bool,
    fallback: bool = False,
) -> None:
    stage1_keep_ratio = 1.0
    final_keep_ratio = 1.0
    if visual_tokens_before > 0:
        stage1_keep_ratio = stage1_visual_tokens / float(visual_tokens_before)
        final_keep_ratio = final_visual_tokens / float(visual_tokens_before)

    for metrics_name in ("_star_metrics", "_star_episode_metrics"):
        metrics = getattr(self, metrics_name, None)
        if metrics is None:
            continue
        metrics.model_calls += 1
        metrics.tokens_before += int(visual_tokens_before)
        metrics.tokens_after_stage1 += int(stage1_visual_tokens)
        metrics.tokens_after_stage2 += int(final_visual_tokens)
        metrics.last_visual_tokens = int(visual_tokens_before)
        metrics.last_stage1_visual_tokens = int(stage1_visual_tokens)
        metrics.last_final_visual_tokens = int(final_visual_tokens)
        metrics.last_stage1_keep_ratio = float(stage1_keep_ratio)
        metrics.last_keep_ratio = float(final_keep_ratio)
        metrics.last_query_mode = str(query_mode)
        if used_star and stage1_visual_tokens < visual_tokens_before:
            metrics.stage1_prune_events += 1
        if used_star and final_visual_tokens < stage1_visual_tokens:
            metrics.stage2_prune_events += 1
        if vision_attention_hit:
            metrics.vision_attention_hits += 1
        else:
            metrics.vision_attention_misses += 1
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


def _should_use_star(self, num_visual_patch_tokens: int, prompt_token_count: int) -> bool:
    cfg = getattr(self, "_star_cfg", None)
    if cfg is None:
        return False
    if not bool(getattr(cfg, "star_enabled", False)):
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
) -> STARForwardResult:
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
    return STARForwardResult(
        hidden_states=output.hidden_states[-1],
        logits=output.logits if compute_logits else None,
        current_visual_tokens=int(num_visual_patch_tokens),
        stage1_visual_tokens=int(num_visual_patch_tokens),
        used_star=False,
        query_mode="dense",
        vision_attention_hit=False,
    )


def _patch_vision_attention_capture(target) -> None:
    cache = getattr(target, "_star_cache", None)
    cfg = getattr(target, "_star_cfg", None)
    vision_backbone = getattr(target, "vision_backbone", None)
    if cache is None or cfg is None or vision_backbone is None:
        return
    if getattr(vision_backbone, "_star_forward_patched", False):
        return

    raw_backbone = _unwrap_vision_backbone(vision_backbone)
    patched = 0
    patched += _patch_one_featurizer(raw_backbone, "featurizer", "primary", cache, cfg)
    patched += _patch_one_featurizer(raw_backbone, "fused_featurizer", "fused", cache, cfg)
    cache.patched_vision_modules = patched

    original_forward = vision_backbone.forward

    def star_vision_forward(vb_self, *args, **kwargs):
        cache.current_vision_scores = {"primary": [], "fused": []}
        try:
            output = original_forward(*args, **kwargs)
            cache.stage1_scores = _combine_current_vision_scores(cache, output)
            return output
        finally:
            cache.current_vision_scores = None

    vision_backbone._star_original_forward = original_forward
    vision_backbone.forward = MethodType(star_vision_forward, vision_backbone)
    vision_backbone._star_forward_patched = True


def _unwrap_vision_backbone(vision_backbone):
    raw = vision_backbone
    seen = set()
    while hasattr(raw, "vision_backbone") and id(raw) not in seen:
        seen.add(id(raw))
        raw = raw.vision_backbone
    return raw


def _resolve_layer_index(num_blocks: int, layer: int) -> int:
    idx = int(layer)
    if idx < 0:
        idx = num_blocks + idx
    return max(0, min(idx, num_blocks - 1))


def _patch_one_featurizer(raw_backbone, attr_name: str, stream: str, cache: STARCache, cfg: Any) -> int:
    featurizer = getattr(raw_backbone, attr_name, None)
    blocks = getattr(featurizer, "blocks", None)
    if featurizer is None or blocks is None or len(blocks) == 0:
        return 0
    layer_idx = _resolve_layer_index(len(blocks), int(getattr(cfg, "star_vision_attention_layer", -2)))
    attn = getattr(blocks[layer_idx], "attn", None)
    if attn is None or getattr(attn, "_star_attention_patched", False):
        return 0

    attn._star_num_patches = int(getattr(getattr(featurizer, "patch_embed", None), "num_patches", 0))
    attn._star_stream = stream
    attn._star_cache = cache

    def star_attention_forward(attn_self, x):
        batch_size, token_count, channels = x.shape
        qkv = attn_self.qkv(x).reshape(
            batch_size,
            token_count,
            3,
            attn_self.num_heads,
            attn_self.head_dim,
        ).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = attn_self.q_norm(q), attn_self.k_norm(k)
        q = q * attn_self.scale
        attn_weights = q @ k.transpose(-2, -1)
        attn_weights = attn_weights.softmax(dim=-1)
        _append_vision_attention_scores(attn_self, attn_weights)
        attn_drop = attn_self.attn_drop(attn_weights)
        out = attn_drop @ v
        out = out.transpose(1, 2).reshape(batch_size, token_count, channels)
        out = attn_self.proj(out)
        out = attn_self.proj_drop(out)
        return out

    attn._star_original_forward = attn.forward
    attn.forward = MethodType(star_attention_forward, attn)
    attn._star_attention_patched = True
    return 1


def _append_vision_attention_scores(attn_module, attention_weights: torch.Tensor) -> None:
    cache = getattr(attn_module, "_star_cache", None)
    current = None if cache is None else getattr(cache, "current_vision_scores", None)
    if current is None:
        return

    num_patches = int(getattr(attn_module, "_star_num_patches", 0))
    if num_patches <= 0:
        return
    start = max(0, int(attention_weights.shape[-1]) - num_patches)
    stop = min(int(attention_weights.shape[-1]), start + num_patches)
    if stop <= start:
        return
    # Importance is attention received by each patch, averaged over batch, heads, and query tokens.
    scores = attention_weights[..., start:stop].mean(dim=(0, 1, 2)).detach().float().cpu()
    current.setdefault(str(getattr(attn_module, "_star_stream", "primary")), []).append(scores)


def _combine_current_vision_scores(cache: STARCache, output) -> Optional[torch.Tensor]:
    current = cache.current_vision_scores
    if not current:
        return None
    primary = current.get("primary", [])
    fused = current.get("fused", [])
    num_images = max(len(primary), len(fused))
    if num_images == 0:
        return None

    combined = []
    for image_idx in range(num_images):
        parts = []
        if image_idx < len(primary):
            parts.append(primary[image_idx])
        if image_idx < len(fused):
            parts.append(fused[image_idx])
        if parts:
            combined.append(torch.stack(parts, dim=0).mean(dim=0))
    if not combined:
        return None
    scores = torch.cat(combined, dim=0)
    if isinstance(output, torch.Tensor) and output.shape[1] != scores.numel():
        return None
    return scores


def _get_stage1_scores(cache: STARCache, device: torch.device, expected_tokens: int) -> Optional[torch.Tensor]:
    scores = cache.stage1_scores
    if scores is None or int(scores.numel()) != int(expected_tokens):
        return None
    return scores.to(device=device, dtype=torch.float32)


def _select_keep_by_image(
    scores: torch.Tensor,
    current_to_original: torch.Tensor,
    num_images: int,
    per_image_patches: int,
    keep_ratio: float,
    min_keep_per_image: int,
) -> torch.Tensor:
    selected = []
    keep_ratio = max(0.0, min(1.0, float(keep_ratio)))
    for image_idx in range(num_images):
        image_start = image_idx * per_image_patches
        image_end = image_start + per_image_patches
        mask = (current_to_original >= image_start) & (current_to_original < image_end)
        positions = torch.where(mask)[0]
        if positions.numel() == 0:
            continue
        keep_count = int(math.ceil(int(positions.numel()) * keep_ratio))
        keep_count = max(int(min_keep_per_image), min(keep_count, int(positions.numel())))
        local_scores = scores.index_select(0, positions)
        local_topk = torch.topk(local_scores, k=keep_count, largest=True, sorted=False).indices
        selected.append(positions.index_select(0, local_topk))
    if not selected:
        return torch.arange(current_to_original.numel(), device=current_to_original.device, dtype=torch.long)
    return torch.sort(torch.unique(torch.cat(selected, dim=0))).values


def _build_pruned_sequence(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    absolute_positions: torch.LongTensor,
    visual_start: int,
    visual_token_count: int,
    keep_visual_rel_indices: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.LongTensor, int, torch.Tensor]:
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
    return new_hidden_states, new_attention_mask, new_positions, int(keep_visual_rel_indices.numel()), keep_visual_rel_indices


def _text_to_visual_scores(
    attention_weights: torch.Tensor,
    visual_start: int,
    visual_token_count: int,
    extra_projected_tokens_count: int,
    prompt_token_count: int,
) -> torch.Tensor:
    mean_attention = attention_weights.mean(dim=1)
    prompt_start = visual_start + int(visual_token_count) + int(extra_projected_tokens_count)
    prompt_end = prompt_start + int(prompt_token_count)
    if prompt_end <= prompt_start:
        raise ValueError("STAR text token range is empty.")
    return mean_attention[0, prompt_start:prompt_end, visual_start : visual_start + visual_token_count].mean(dim=0).float()


def _action_to_visual_scores(
    attention_weights: torch.Tensor,
    visual_start: int,
    visual_token_count: int,
    extra_projected_tokens_count: int,
    prompt_token_count: int,
) -> torch.Tensor:
    mean_attention = attention_weights.mean(dim=1)
    action_start = visual_start + int(visual_token_count) + int(extra_projected_tokens_count) + int(prompt_token_count)
    action_end = action_start + ACTION_DIM * NUM_ACTIONS_CHUNK
    if action_end <= action_start or action_end > mean_attention.shape[-1]:
        raise ValueError("STAR action token range is invalid.")
    return mean_attention[0, action_start:action_end, visual_start : visual_start + visual_token_count].mean(dim=0).float()


def _cross_modal_scores(
    cfg: Any,
    attention_weights: torch.Tensor,
    visual_start: int,
    visual_token_count: int,
    extra_projected_tokens_count: int,
    prompt_token_count: int,
) -> torch.Tensor:
    if str(getattr(cfg, "star_query_mode", "action_mean")) == "text_mean":
        return _text_to_visual_scores(
            attention_weights,
            visual_start,
            visual_token_count,
            extra_projected_tokens_count,
            prompt_token_count,
        )
    return _action_to_visual_scores(
        attention_weights,
        visual_start,
        visual_token_count,
        extra_projected_tokens_count,
        prompt_token_count,
    )


def _star_causal_lm_forward(
    self,
    multimodal_embeddings: torch.Tensor,
    multimodal_attention_mask: Optional[torch.Tensor],
    num_visual_patch_tokens: int,
    extra_projected_tokens_count: int,
    prompt_token_count: int,
    compute_logits: bool,
) -> STARForwardResult:
    language_model = self.language_model
    decoder = language_model.model
    cfg = getattr(self, "_star_cfg", None)
    cache = getattr(self, "_star_cache", None)
    if cfg is None or cache is None:
        raise ValueError("STAR configuration/cache is missing on the loaded model.")

    batch_size, total_seq_len, _ = multimodal_embeddings.shape
    if batch_size != 1:
        raise ValueError("STAR runtime patch only supports batch size 1 during inference.")

    stage2_layer = int(getattr(cfg, "star_stage2_layer", 3))
    if stage2_layer >= len(decoder.layers):
        raise ValueError(f"star_stage2_layer={stage2_layer} is out of range.")

    device = multimodal_embeddings.device
    hidden_states = multimodal_embeddings
    current_attention_mask = multimodal_attention_mask
    if current_attention_mask is None:
        current_attention_mask = torch.ones((batch_size, total_seq_len), dtype=torch.long, device=device)

    absolute_positions = torch.arange(total_seq_len, device=device, dtype=torch.long)
    visual_start = 1
    current_visual_tokens = int(num_visual_patch_tokens)
    stage1_visual_tokens = int(num_visual_patch_tokens)
    current_to_original = torch.arange(current_visual_tokens, device=device, dtype=torch.long)

    per_image_patches = int(self.vision_backbone.get_num_patches())
    num_images = int(self.vision_backbone.get_num_images_in_input())
    if per_image_patches * num_images > current_visual_tokens:
        raise ValueError("STAR visual token layout is inconsistent with vision backbone metadata.")

    min_keep_per_image = int(getattr(cfg, "star_min_keep_tokens_per_image", 1))
    stage1_scores = _get_stage1_scores(cache, device, current_visual_tokens)
    vision_attention_hit = stage1_scores is not None
    if stage1_scores is not None:
        keep_rel = _select_keep_by_image(
            stage1_scores,
            current_to_original,
            num_images,
            per_image_patches,
            float(getattr(cfg, "star_stage1_keep_ratio", 0.75)),
            min_keep_per_image,
        )
        if keep_rel.numel() < current_visual_tokens:
            hidden_states, current_attention_mask, absolute_positions, current_visual_tokens, keep_rel = _build_pruned_sequence(
                hidden_states,
                current_attention_mask,
                absolute_positions,
                visual_start=visual_start,
                visual_token_count=current_visual_tokens,
                keep_visual_rel_indices=keep_rel,
            )
            current_to_original = current_to_original.index_select(0, keep_rel)
        stage1_visual_tokens = int(current_visual_tokens)

    probe_attention = None
    stage2_pruned = False
    force_attention = bool(getattr(cfg, "star_force_attention_after_prune", True))
    query_mode = str(getattr(cfg, "star_query_mode", "action_mean"))

    for layer_idx, decoder_layer in enumerate(decoder.layers):
        if layer_idx == stage2_layer and probe_attention is not None and not stage2_pruned:
            scores = _cross_modal_scores(
                cfg,
                probe_attention,
                visual_start=visual_start,
                visual_token_count=current_visual_tokens,
                extra_projected_tokens_count=int(extra_projected_tokens_count),
                prompt_token_count=int(prompt_token_count),
            )
            keep_rel = _select_keep_by_image(
                scores,
                current_to_original,
                num_images,
                per_image_patches,
                float(getattr(cfg, "star_stage2_keep_ratio", 0.5)),
                min_keep_per_image,
            )
            if keep_rel.numel() < current_visual_tokens:
                hidden_states, current_attention_mask, absolute_positions, current_visual_tokens, keep_rel = _build_pruned_sequence(
                    hidden_states,
                    current_attention_mask,
                    absolute_positions,
                    visual_start=visual_start,
                    visual_token_count=current_visual_tokens,
                    keep_visual_rel_indices=keep_rel,
                )
                current_to_original = current_to_original.index_select(0, keep_rel)
            stage2_pruned = True

        current_len = hidden_states.shape[1]
        cache_positions = torch.arange(current_len, device=device, dtype=torch.long)
        position_ids = absolute_positions.unsqueeze(0)
        causal_mask = decoder._update_causal_mask(current_attention_mask, hidden_states, cache_positions, 0)
        has_pruned_visual = current_visual_tokens < int(num_visual_patch_tokens)
        need_attention = layer_idx == (stage2_layer - 1) or (force_attention and has_pruned_visual)

        layer_outputs = decoder_layer(
            hidden_states,
            attention_mask=causal_mask,
            position_ids=position_ids,
            past_key_value=None,
            output_attentions=need_attention,
            use_cache=False,
            cache_position=cache_positions,
        )
        hidden_states = layer_outputs[0]
        if layer_idx == (stage2_layer - 1) and need_attention:
            probe_attention = layer_outputs[1]

    hidden_states = decoder.norm(hidden_states)
    logits = _compute_logits_from_hidden_states(language_model, hidden_states) if compute_logits else None
    cache.calls += 1
    return STARForwardResult(
        hidden_states=hidden_states,
        logits=logits,
        current_visual_tokens=int(current_visual_tokens),
        stage1_visual_tokens=int(stage1_visual_tokens),
        used_star=True,
        query_mode=query_mode,
        vision_attention_hit=vision_attention_hit,
    )


def _forward_language_model_with_optional_star(
    self,
    multimodal_embeddings: torch.Tensor,
    multimodal_attention_mask: Optional[torch.Tensor],
    num_visual_patch_tokens: int,
    extra_projected_tokens_count: int,
    prompt_token_count: int,
    compute_logits: bool,
) -> STARForwardResult:
    if not _should_use_star(self, num_visual_patch_tokens, prompt_token_count) or not _supports_language_model(self):
        result = _dense_causal_lm_forward(self, multimodal_embeddings, multimodal_attention_mask, num_visual_patch_tokens, compute_logits)
        _record_metrics(
            self,
            int(num_visual_patch_tokens),
            int(num_visual_patch_tokens),
            int(num_visual_patch_tokens),
            used_star=False,
            query_mode=result.query_mode,
            vision_attention_hit=False,
        )
        return result

    cache = getattr(self, "_star_cache", None)
    cfg = getattr(self, "_star_cfg", None)
    warmup_model_calls = int(getattr(cfg, "star_warmup_model_calls", 0)) if cfg is not None else 0
    if cache is not None and int(getattr(cache, "calls", 0)) < warmup_model_calls:
        result = _dense_causal_lm_forward(self, multimodal_embeddings, multimodal_attention_mask, num_visual_patch_tokens, compute_logits)
        cache.calls += 1
        _record_metrics(
            self,
            int(num_visual_patch_tokens),
            int(num_visual_patch_tokens),
            int(num_visual_patch_tokens),
            used_star=False,
            query_mode="warmup",
            vision_attention_hit=False,
        )
        return result

    try:
        result = _star_causal_lm_forward(
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
            int(result.stage1_visual_tokens),
            int(result.current_visual_tokens),
            used_star=bool(result.used_star),
            query_mode=result.query_mode,
            vision_attention_hit=bool(result.vision_attention_hit),
            fallback=False,
        )
        return result
    except Exception as exc:
        if not getattr(self, "_star_logged_fallback", False):
            print(f"[STAR] fallback to dense language-model forward: {exc}")
            self._star_logged_fallback = True
        result = _dense_causal_lm_forward(self, multimodal_embeddings, multimodal_attention_mask, num_visual_patch_tokens, compute_logits)
        _record_metrics(
            self,
            int(num_visual_patch_tokens),
            int(num_visual_patch_tokens),
            int(num_visual_patch_tokens),
            used_star=False,
            query_mode="fallback",
            vision_attention_hit=False,
            fallback=True,
        )
        return result


def _extract_action_hidden_states(sequence_hidden_states: torch.Tensor) -> torch.Tensor:
    action_hidden_states = sequence_hidden_states[:, -ACTION_DIM * NUM_ACTIONS_CHUNK - 1 : -1, :]
    expected_tokens = ACTION_DIM * NUM_ACTIONS_CHUNK
    if action_hidden_states.shape[1] != expected_tokens:
        raise RuntimeError(f"Expected {expected_tokens} action tokens after STAR pruning, got {action_hidden_states.shape[1]}.")
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
        raise RuntimeError(f"Expected {expected_tokens} action logits after STAR pruning, got {action_logits.shape[1]}.")
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
    del labels, num_patches, past_key_values, unused_extra_args, unused_extra_kwargs

    all_actions_mask = all_actions_mask.unsqueeze(-1)
    input_embeddings = input_embeddings * ~all_actions_mask
    multimodal_embeddings, multimodal_attention_mask = self._build_multimodal_attention(
        input_embeddings,
        projected_patch_embeddings,
        attention_mask,
    )

    num_visual_patch_tokens = int(self.vision_backbone.get_num_patches()) * int(self.vision_backbone.get_num_images_in_input())
    num_visual_patch_tokens = min(num_visual_patch_tokens, int(projected_patch_embeddings.shape[1]))
    extra_projected_tokens_count = int(projected_patch_embeddings.shape[1]) - int(num_visual_patch_tokens)

    language_model_output = _forward_language_model_with_optional_star(
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
