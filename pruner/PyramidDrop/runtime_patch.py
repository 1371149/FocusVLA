"""Runtime monkeypatches that transplant PyramidDrop into OpenVLA-OFT.

The implementation follows PyramidDrop's inference-time idea: at configured
decoder depths, rank current visual tokens by text-token -> visual-token
self-attention and drop the low-ranked visual tokens. Unlike the original
LLaVA implementation, remaining token absolute positions are preserved because
the OpenVLA-OFT checkpoints are sensitive to RoPE position changes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from types import MethodType
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, repeat_kv

from backend import load_backend_modules

_backend = load_backend_modules()
openvla_utils = _backend.openvla_utils

from prismatic.vla.constants import ACTION_DIM, NUM_ACTIONS_CHUNK


@dataclass
class PyramidDropMetrics:
    model_calls: int = 0
    prune_events: int = 0
    tokens_before: int = 0
    tokens_after: int = 0
    fallback_calls: int = 0
    last_visual_tokens: int = 0
    last_final_visual_tokens: int = 0
    last_keep_ratio: float = 1.0
    last_layers: str = ""
    last_ratios: str = ""
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
            "last_final_visual_tokens": float(self.last_final_visual_tokens),
            "last_keep_ratio": float(self.last_keep_ratio),
            "last_layers": self.last_layers,
            "last_ratios": self.last_ratios,
            "last_query_mode": self.last_query_mode,
        }


@dataclass
class PyramidDropForwardResult:
    hidden_states: torch.Tensor
    logits: Optional[torch.Tensor]
    current_visual_tokens: int
    used_pyramiddrop: bool
    applied_layers: List[int]
    applied_ratios: List[float]


def apply_runtime_patches() -> None:
    if getattr(openvla_utils, "_pyramiddrop_runtime_patch_applied", False):
        return
    openvla_utils._pyramiddrop_runtime_patch_applied = True


def patch_loaded_model(model: torch.nn.Module, cfg: Optional[Any] = None) -> None:
    target = model.module if hasattr(model, "module") else model
    if cfg is not None:
        target._pyramiddrop_cfg = cfg

    if not hasattr(target, "_pyramiddrop_metrics"):
        target._pyramiddrop_metrics = PyramidDropMetrics()
    if not hasattr(target, "_pyramiddrop_episode_metrics"):
        target._pyramiddrop_episode_metrics = PyramidDropMetrics()
    if not hasattr(target, "_pyramiddrop_logged_fallback"):
        target._pyramiddrop_logged_fallback = False

    if not hasattr(target, "_pyramiddrop_original_regression_prediction"):
        target._pyramiddrop_original_regression_prediction = target._regression_or_discrete_prediction
        target._regression_or_discrete_prediction = MethodType(_regression_or_discrete_prediction, target)

    if not hasattr(target, "_pyramiddrop_original_diffusion_prediction"):
        target._pyramiddrop_original_diffusion_prediction = target._run_diffusion_prediction
        target._run_diffusion_prediction = MethodType(_run_diffusion_prediction, target)

    if not hasattr(target, "pyramiddrop_reset_episode"):
        target.pyramiddrop_reset_episode = MethodType(pyramiddrop_reset_episode, target)
    if not hasattr(target, "pyramiddrop_get_metrics"):
        target.pyramiddrop_get_metrics = MethodType(pyramiddrop_get_metrics, target)

    target.pyramiddrop_reset_episode()


def get_runtime_metrics(model: torch.nn.Module) -> Dict[str, float]:
    target = model.module if hasattr(model, "module") else model
    metrics = getattr(target, "_pyramiddrop_metrics", None)
    if metrics is None:
        return PyramidDropMetrics().snapshot()
    return metrics.snapshot()


def get_episode_runtime_metrics(model: torch.nn.Module) -> Dict[str, float]:
    target = model.module if hasattr(model, "module") else model
    metrics = getattr(target, "_pyramiddrop_episode_metrics", None)
    if metrics is None:
        return PyramidDropMetrics().snapshot()
    return metrics.snapshot()


def pyramiddrop_reset_episode(self) -> None:
    self._pyramiddrop_episode_metrics = PyramidDropMetrics()


def pyramiddrop_get_metrics(self) -> Dict[str, float]:
    return self._pyramiddrop_metrics.snapshot()


def _parse_int_list(value: Any, default: Sequence[int]) -> List[int]:
    if value is None:
        return list(default)
    if isinstance(value, str):
        if value.strip() == "":
            return []
        return [int(item.strip()) for item in value.split(",") if item.strip()]
    return [int(item) for item in value]


def _parse_float_list(value: Any, default: Sequence[float]) -> List[float]:
    if value is None:
        return list(default)
    if isinstance(value, str):
        if value.strip() == "":
            return []
        return [float(item.strip()) for item in value.split(",") if item.strip()]
    return [float(item) for item in value]


def _record_metrics(
    self,
    visual_tokens_before: int,
    visual_tokens_after: int,
    applied_layers: Sequence[int],
    applied_ratios: Sequence[float],
    used_pyramiddrop: bool,
    fallback: bool = False,
) -> None:
    cfg = getattr(self, "_pyramiddrop_cfg", None)
    query_mode = "text_last" if cfg is None else str(getattr(cfg, "pyramiddrop_query_mode", "text_last"))
    keep_ratio = 1.0
    if visual_tokens_before > 0:
        keep_ratio = visual_tokens_after / float(visual_tokens_before)

    layers_text = ",".join(str(x) for x in applied_layers)
    ratios_text = ",".join(f"{float(x):.4f}" for x in applied_ratios)
    for metrics_name in ("_pyramiddrop_metrics", "_pyramiddrop_episode_metrics"):
        metrics = getattr(self, metrics_name, None)
        if metrics is None:
            continue
        metrics.model_calls += 1
        metrics.tokens_before += int(visual_tokens_before)
        metrics.tokens_after += int(visual_tokens_after)
        metrics.last_visual_tokens = int(visual_tokens_before)
        metrics.last_final_visual_tokens = int(visual_tokens_after)
        metrics.last_keep_ratio = float(keep_ratio)
        metrics.last_layers = layers_text
        metrics.last_ratios = ratios_text
        metrics.last_query_mode = query_mode
        if used_pyramiddrop:
            metrics.prune_events += int(len(applied_layers))
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


def _should_use_pyramiddrop(self, num_visual_patch_tokens: int, prompt_token_count: int) -> bool:
    cfg = getattr(self, "_pyramiddrop_cfg", None)
    if cfg is None:
        return False
    if not bool(getattr(cfg, "pyramiddrop_enabled", False)):
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
) -> PyramidDropForwardResult:
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
    return PyramidDropForwardResult(
        hidden_states=output.hidden_states[-1],
        logits=output.logits if compute_logits else None,
        current_visual_tokens=int(num_visual_patch_tokens),
        used_pyramiddrop=False,
        applied_layers=[],
        applied_ratios=[],
    )


def _build_pruned_sequence(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    absolute_positions: torch.LongTensor,
    visual_start: int,
    visual_token_count: int,
    keep_visual_rel_indices: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.LongTensor, int, torch.Tensor]:
    keep_visual_rel_indices = torch.sort(keep_visual_rel_indices.to(dtype=torch.long)).values
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


def _select_keep_by_image(
    scores: torch.Tensor,
    current_to_original: torch.Tensor,
    num_images: int,
    per_image_patches: int,
    keep_ratio: float,
    min_keep_per_image: int,
) -> torch.Tensor:
    keep_chunks: List[torch.Tensor] = []
    target_keep = int(per_image_patches * float(keep_ratio))
    target_keep = max(int(min_keep_per_image), min(target_keep, int(per_image_patches)))

    for image_idx in range(num_images):
        image_start = image_idx * per_image_patches
        image_end = image_start + per_image_patches
        mask = (current_to_original >= image_start) & (current_to_original < image_end)
        positions = torch.where(mask)[0]
        if positions.numel() == 0:
            continue
        keep_count = max(int(min_keep_per_image), min(target_keep, int(positions.numel())))
        if keep_count >= positions.numel():
            keep_chunks.append(positions)
            continue
        image_scores = scores.index_select(0, positions)
        top_local = torch.topk(image_scores, k=keep_count, largest=True, sorted=False).indices
        keep_chunks.append(torch.sort(positions.index_select(0, top_local)).values)

    if not keep_chunks:
        return torch.arange(scores.numel(), device=scores.device, dtype=torch.long)
    return torch.cat(keep_chunks, dim=0).to(device=scores.device, dtype=torch.long)


def _attention_query_indices(
    cfg: Any,
    visual_start: int,
    visual_token_count: int,
    extra_projected_tokens_count: int,
    prompt_token_count: int,
    seq_len: int,
    device: torch.device,
) -> torch.Tensor:
    prompt_start = visual_start + int(visual_token_count) + int(extra_projected_tokens_count)
    prompt_end = prompt_start + int(prompt_token_count)
    if prompt_end <= prompt_start or prompt_end > seq_len:
        raise ValueError(
            f"PyramidDrop text token range is invalid: prompt_start={prompt_start}, "
            f"prompt_end={prompt_end}, seq_len={seq_len}."
        )
    query_mode = str(getattr(cfg, "pyramiddrop_query_mode", "text_last"))
    if query_mode == "text_mean":
        return torch.arange(prompt_start, prompt_end, device=device, dtype=torch.long)
    return torch.tensor([prompt_end - 1], device=device, dtype=torch.long)


def _project_qk(attn_module, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    bsz, q_len, _ = hidden_states.size()
    query_states = attn_module.q_proj(hidden_states)
    key_states = attn_module.k_proj(hidden_states)
    query_states = query_states.view(bsz, q_len, attn_module.num_heads, attn_module.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, attn_module.num_key_value_heads, attn_module.head_dim).transpose(1, 2)
    return query_states, key_states


def _apply_rotary(attn_module, query_states: torch.Tensor, key_states: torch.Tensor, position_ids: torch.Tensor):
    try:
        cos, sin = attn_module.rotary_emb(key_states, position_ids)
        return apply_rotary_pos_emb(query_states, key_states, cos, sin)
    except TypeError:
        cos, sin = attn_module.rotary_emb(key_states, seq_len=key_states.shape[-2])
        return apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)


def _pyramiddrop_text_to_visual_scores(
    decoder,
    probe_layer,
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    absolute_positions: torch.LongTensor,
    cfg: Any,
    visual_start: int,
    visual_token_count: int,
    extra_projected_tokens_count: int,
    prompt_token_count: int,
) -> torch.Tensor:
    if hidden_states.shape[0] != 1:
        raise ValueError("PyramidDrop scoring currently expects batch size 1.")

    normed_hidden_states = probe_layer.input_layernorm(hidden_states)
    attn_module = probe_layer.self_attn
    query_states, key_states = _project_qk(attn_module, normed_hidden_states)
    position_ids = absolute_positions.unsqueeze(0)
    query_states, key_states = _apply_rotary(attn_module, query_states, key_states, position_ids)

    num_key_value_groups = int(getattr(attn_module, "num_key_value_groups", attn_module.num_heads // attn_module.num_key_value_heads))
    key_states = repeat_kv(key_states, num_key_value_groups)

    seq_len = int(hidden_states.shape[1])
    query_indices = _attention_query_indices(
        cfg,
        visual_start=visual_start,
        visual_token_count=visual_token_count,
        extra_projected_tokens_count=extra_projected_tokens_count,
        prompt_token_count=prompt_token_count,
        seq_len=seq_len,
        device=hidden_states.device,
    )
    text_query_states = query_states.index_select(2, query_indices)
    attn_weights = torch.matmul(text_query_states, key_states.transpose(2, 3)) / math.sqrt(float(attn_module.head_dim))

    cache_positions = torch.arange(seq_len, device=hidden_states.device, dtype=torch.long)
    causal_mask = decoder._update_causal_mask(attention_mask, hidden_states, cache_positions, 0)
    if causal_mask is not None:
        selected_mask = causal_mask.index_select(2, query_indices)
        attn_weights = attn_weights + selected_mask[:, :, :, : key_states.shape[-2]]

    attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
    visual_scores = attn_weights[:, :, :, visual_start : visual_start + visual_token_count].mean(dim=(0, 1, 2))
    return visual_scores.float()


def _pyramiddrop_causal_lm_forward(
    self,
    multimodal_embeddings: torch.Tensor,
    multimodal_attention_mask: Optional[torch.Tensor],
    num_visual_patch_tokens: int,
    extra_projected_tokens_count: int,
    prompt_token_count: int,
    compute_logits: bool,
) -> PyramidDropForwardResult:
    language_model = self.language_model
    decoder = language_model.model
    cfg = getattr(self, "_pyramiddrop_cfg", None)
    if cfg is None:
        raise ValueError("PyramidDrop configuration is missing on the loaded model.")

    layer_list = _parse_int_list(getattr(cfg, "pyramiddrop_layer_list", "8,16,24"), [8, 16, 24])
    ratio_list = _parse_float_list(getattr(cfg, "pyramiddrop_image_token_ratio_list", "0.5,0.25,0.125"), [0.5, 0.25, 0.125])
    if len(layer_list) != len(ratio_list):
        raise ValueError("PyramidDrop layer/ratio list length mismatch.")
    stage_by_layer = {int(layer): float(ratio) for layer, ratio in zip(layer_list, ratio_list)}

    batch_size, total_seq_len, _ = multimodal_embeddings.shape
    if batch_size != 1:
        raise ValueError("PyramidDrop runtime patch only supports batch size 1 during inference.")

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
    if per_image_patches * num_images > current_visual_tokens:
        raise ValueError("PyramidDrop visual token layout is inconsistent with vision backbone metadata.")

    min_keep_per_image = int(getattr(cfg, "pyramiddrop_min_keep_tokens_per_image", 1))
    force_attention = bool(getattr(cfg, "pyramiddrop_force_attention_after_prune", True))
    current_to_original = torch.arange(current_visual_tokens, device=device, dtype=torch.long)
    applied_layers: List[int] = []
    applied_ratios: List[float] = []

    for layer_idx, decoder_layer in enumerate(decoder.layers):
        layer_number = layer_idx + 1
        current_len = hidden_states.shape[1]
        cache_positions = torch.arange(current_len, device=device, dtype=torch.long)
        position_ids = absolute_positions.unsqueeze(0)
        causal_mask = decoder._update_causal_mask(current_attention_mask, hidden_states, cache_positions, 0)
        has_pruned_visual = current_visual_tokens < int(num_visual_patch_tokens)
        need_attention = force_attention and has_pruned_visual

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

        if layer_number not in stage_by_layer or current_visual_tokens <= num_images * min_keep_per_image:
            continue

        probe_layer_idx = min(layer_idx + 1, len(decoder.layers) - 1)
        probe_layer = decoder.layers[probe_layer_idx]
        scores = _pyramiddrop_text_to_visual_scores(
            decoder,
            probe_layer,
            hidden_states,
            current_attention_mask,
            absolute_positions,
            cfg,
            visual_start=visual_start,
            visual_token_count=current_visual_tokens,
            extra_projected_tokens_count=int(extra_projected_tokens_count),
            prompt_token_count=int(prompt_token_count),
        )
        keep_rel = _select_keep_by_image(
            scores,
            current_to_original,
            num_images=num_images,
            per_image_patches=per_image_patches,
            keep_ratio=float(stage_by_layer[layer_number]),
            min_keep_per_image=min_keep_per_image,
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
            applied_layers.append(layer_number)
            applied_ratios.append(float(stage_by_layer[layer_number]))

    hidden_states = decoder.norm(hidden_states)
    logits = _compute_logits_from_hidden_states(language_model, hidden_states) if compute_logits else None
    return PyramidDropForwardResult(
        hidden_states=hidden_states,
        logits=logits,
        current_visual_tokens=int(current_visual_tokens),
        used_pyramiddrop=bool(applied_layers),
        applied_layers=applied_layers,
        applied_ratios=applied_ratios,
    )


def _forward_language_model_with_optional_pyramiddrop(
    self,
    multimodal_embeddings: torch.Tensor,
    multimodal_attention_mask: Optional[torch.Tensor],
    num_visual_patch_tokens: int,
    extra_projected_tokens_count: int,
    prompt_token_count: int,
    compute_logits: bool,
) -> PyramidDropForwardResult:
    if not _should_use_pyramiddrop(self, num_visual_patch_tokens, prompt_token_count) or not _supports_language_model(self):
        result = _dense_causal_lm_forward(self, multimodal_embeddings, multimodal_attention_mask, num_visual_patch_tokens, compute_logits)
        _record_metrics(self, int(num_visual_patch_tokens), int(num_visual_patch_tokens), [], [], used_pyramiddrop=False)
        return result

    try:
        result = _pyramiddrop_causal_lm_forward(
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
            result.applied_layers,
            result.applied_ratios,
            used_pyramiddrop=bool(result.used_pyramiddrop),
            fallback=False,
        )
        return result
    except Exception as exc:
        if not getattr(self, "_pyramiddrop_logged_fallback", False):
            print(f"[PyramidDrop] fallback to dense language-model forward: {exc}")
            self._pyramiddrop_logged_fallback = True
        result = _dense_causal_lm_forward(self, multimodal_embeddings, multimodal_attention_mask, num_visual_patch_tokens, compute_logits)
        _record_metrics(self, int(num_visual_patch_tokens), int(num_visual_patch_tokens), [], [], used_pyramiddrop=False, fallback=True)
        return result


def _extract_action_hidden_states(sequence_hidden_states: torch.Tensor) -> torch.Tensor:
    action_hidden_states = sequence_hidden_states[:, -ACTION_DIM * NUM_ACTIONS_CHUNK - 1 : -1, :]
    expected_tokens = ACTION_DIM * NUM_ACTIONS_CHUNK
    if action_hidden_states.shape[1] != expected_tokens:
        raise RuntimeError(
            f"Expected {expected_tokens} action tokens after PyramidDrop pruning, got {action_hidden_states.shape[1]}."
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
        raise RuntimeError(f"Expected {expected_tokens} action logits after PyramidDrop pruning, got {action_logits.shape[1]}.")
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

    language_model_output = _forward_language_model_with_optional_pyramiddrop(
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
    del num_patches

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

        num_visual_patch_tokens = int(self.vision_backbone.get_num_patches()) * int(self.vision_backbone.get_num_images_in_input())
        num_visual_patch_tokens = min(num_visual_patch_tokens, int(projected_patch_embeddings.shape[1]))
        extra_projected_tokens_count = int(projected_patch_embeddings.shape[1]) - int(num_visual_patch_tokens)

        language_model_output = _forward_language_model_with_optional_pyramiddrop(
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

