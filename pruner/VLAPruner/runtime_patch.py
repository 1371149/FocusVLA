"""Runtime monkeypatches that transplant VLA-Pruner into openvla-oft.

This module keeps all VLA-Pruner-specific logic inside `pruner/VLAPruner`
and patches the loaded OpenVLA model at runtime instead of editing the main
source tree.
"""

import math
from collections import deque
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
class VLAPrunerMetrics:
    model_calls: int = 0
    prune_events: int = 0
    tokens_before: int = 0
    tokens_after: int = 0
    fallback_calls: int = 0
    temporal_ready_calls: int = 0
    last_visual_tokens: int = 0
    last_effective_r: float = 0.0
    selection_mode: str = "action_mean"

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
            "temporal_ready_calls": float(self.temporal_ready_calls),
            "last_visual_tokens": float(self.last_visual_tokens),
            "last_effective_r": float(self.last_effective_r),
            "selection_mode": self.selection_mode,
        }


@dataclass
class VLAPrunerForwardResult:
    hidden_states: torch.Tensor
    logits: Optional[torch.Tensor]
    current_visual_tokens: int
    pruned: bool
    temporal_vector: Optional[torch.Tensor]
    effective_r: float
    selection_mode: str


def apply_runtime_patches() -> None:
    if getattr(openvla_utils, "_vlapruner_runtime_patch_applied", False):
        return
    openvla_utils._vlapruner_runtime_patch_applied = True


def patch_loaded_model(model: torch.nn.Module, cfg: Optional[Any] = None) -> None:
    target = model.module if hasattr(model, "module") else model
    if cfg is not None:
        target._vlapruner_cfg = cfg

    if not hasattr(target, "_vlapruner_metrics"):
        target._vlapruner_metrics = VLAPrunerMetrics()
    if not hasattr(target, "_vlapruner_episode_metrics"):
        target._vlapruner_episode_metrics = VLAPrunerMetrics()
    if not hasattr(target, "_vlapruner_logged_fallback"):
        target._vlapruner_logged_fallback = False

    if not hasattr(target, "_vlapruner_original_predict_action"):
        target._vlapruner_original_predict_action = target.predict_action
        target.predict_action = MethodType(predict_action, target)

    if not hasattr(target, "_vlapruner_original_regression_prediction"):
        target._vlapruner_original_regression_prediction = target._regression_or_discrete_prediction
        target._regression_or_discrete_prediction = MethodType(_regression_or_discrete_prediction, target)

    if not hasattr(target, "_vlapruner_original_diffusion_prediction"):
        target._vlapruner_original_diffusion_prediction = target._run_diffusion_prediction
        target._run_diffusion_prediction = MethodType(_run_diffusion_prediction, target)

    if not hasattr(target, "vlapruner_reset_episode"):
        target.vlapruner_reset_episode = MethodType(vlapruner_reset_episode, target)
    if not hasattr(target, "vlapruner_get_metrics"):
        target.vlapruner_get_metrics = MethodType(vlapruner_get_metrics, target)
    if not hasattr(target, "reset_av_history"):
        target.reset_av_history = MethodType(reset_av_history, target)

    _reset_temporal_history(target)
    target.vlapruner_reset_episode()


def get_runtime_metrics(model: torch.nn.Module) -> Dict[str, float]:
    target = model.module if hasattr(model, "module") else model
    metrics = getattr(target, "_vlapruner_metrics", None)
    if metrics is None:
        return VLAPrunerMetrics().snapshot()
    return metrics.snapshot()


def get_episode_runtime_metrics(model: torch.nn.Module) -> Dict[str, float]:
    target = model.module if hasattr(model, "module") else model
    metrics = getattr(target, "_vlapruner_episode_metrics", None)
    if metrics is None:
        return VLAPrunerMetrics().snapshot()
    return metrics.snapshot()


def _reset_temporal_history(target) -> None:
    cfg = getattr(target, "_vlapruner_cfg", None)
    temporal_w = 3 if cfg is None else int(getattr(cfg, "temporal_w", 3))
    target._vlapruner_av_hist = deque(maxlen=temporal_w)
    target._vlapruner_step = None


def reset_av_history(self) -> None:
    _reset_temporal_history(self)


def vlapruner_reset_episode(self) -> None:
    self._vlapruner_episode_metrics = VLAPrunerMetrics()
    _reset_temporal_history(self)


def vlapruner_get_metrics(self) -> Dict[str, float]:
    return self._vlapruner_metrics.snapshot()


def _record_vlapruner_metrics(
    self,
    tokens_before: int,
    tokens_after: int,
    pruned: bool,
    effective_r: float,
    selection_mode: str,
    temporal_ready: bool,
    fallback: bool = False,
) -> None:
    for metrics_name in ("_vlapruner_metrics", "_vlapruner_episode_metrics"):
        metrics = getattr(self, metrics_name, None)
        if metrics is None:
            continue
        metrics.model_calls += 1
        metrics.tokens_before += int(tokens_before)
        metrics.tokens_after += int(tokens_after)
        metrics.last_visual_tokens = int(tokens_after if pruned else tokens_before)
        metrics.last_effective_r = float(effective_r)
        metrics.selection_mode = str(selection_mode)
        if pruned:
            metrics.prune_events += 1
        if temporal_ready:
            metrics.temporal_ready_calls += 1
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


def _should_use_vlapruner(self, num_visual_patch_tokens: int) -> bool:
    cfg = getattr(self, "_vlapruner_cfg", None)
    if cfg is None:
        return False
    if not bool(getattr(cfg, "use_fastv", False)):
        return False
    if num_visual_patch_tokens <= 1:
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


def _dense_forward(
    self,
    multimodal_embeddings: torch.Tensor,
    multimodal_attention_mask: Optional[torch.Tensor],
    compute_logits: bool,
) -> VLAPrunerForwardResult:
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
    current_visual_tokens = int(getattr(self, "_vlapruner_runtime", {}).get("num_visual_patch_tokens", 0))
    return VLAPrunerForwardResult(
        hidden_states=output.hidden_states[-1],
        logits=output.logits if compute_logits else None,
        current_visual_tokens=current_visual_tokens,
        pruned=False,
        temporal_vector=None,
        effective_r=0.0,
        selection_mode="dense",
    )


def _redundancy_minization(visual_feature_vectors: torch.Tensor, num_keep: int, cosine_matrix: Optional[torch.Tensor] = None) -> torch.Tensor:
    if len(visual_feature_vectors) <= num_keep:
        return torch.arange(len(visual_feature_vectors), device=visual_feature_vectors.device)
    if cosine_matrix is None:
        norm_matrix = visual_feature_vectors / visual_feature_vectors.norm(dim=1, keepdim=True)
        cosine_similarity = torch.mm(norm_matrix, norm_matrix.t())
        cosine_matrix = 1.0 - cosine_similarity

    selected = torch.empty(num_keep, dtype=torch.long, device=visual_feature_vectors.device)
    for i in range(num_keep):
        if i == 0:
            current = cosine_matrix
        else:
            current = torch.index_select(
                cosine_matrix,
                0,
                torch.index_select(selected, 0, torch.arange(0, i, device=cosine_matrix.device)),
            )
        if i == 0:
            scores = torch.topk(current, 2, dim=0, largest=False).values[1, :]
        else:
            scores = torch.min(current, dim=0).values
        selected[i] = torch.argmax(scores)
    return selected


def _compute_selection_scores(
    probe_attention: torch.Tensor,
    visual_start: int,
    visual_token_count: int,
    prompt_start: int,
    prompt_token_count: int,
    action_start: int,
    action_token_count: int,
    selection_mode: str,
) -> torch.Tensor:
    mean_attention = probe_attention.mean(dim=1)[0]
    if selection_mode == "text_vision":
        if prompt_token_count > 0:
            scores = mean_attention[prompt_start : prompt_start + prompt_token_count].mean(dim=0)
        else:
            scores = mean_attention[-1]
    elif selection_mode == "prefill":
        scores = mean_attention.mean(dim=0)
    elif action_token_count > 0:
        scores = mean_attention[action_start : action_start + action_token_count].mean(dim=0)
    else:
        scores = mean_attention[-1]
    return scores[visual_start : visual_start + visual_token_count]


def _build_temporal_history(self) -> tuple[Optional[torch.Tensor], bool]:
    cfg = getattr(self, "_vlapruner_cfg", None)
    if cfg is None or not bool(getattr(cfg, "use_temporal", False)):
        return None, False

    av_hist = getattr(self, "_vlapruner_av_hist", None)
    if av_hist is None or len(av_hist) != av_hist.maxlen:
        return None, False

    weights = np.array([float(getattr(cfg, "temporal_gamma", 0.8)) ** i for i in range(len(av_hist) - 1, -1, -1)], dtype=np.float32)
    guided = np.zeros_like(av_hist[0], dtype=np.float32)
    for idx in range(len(weights)):
        guided += weights[idx] * av_hist[-1 - idx]
    guided_sum = float(np.sum(guided))
    if guided_sum > 0.0:
        guided = guided / guided_sum
    return torch.tensor(guided, device=self.device, dtype=torch.float32), True


def _make_temporal_vector(
    temporal_attention: Optional[torch.Tensor],
    absolute_positions: torch.LongTensor,
    original_visual_token_count: int,
    action_positions_abs: torch.LongTensor,
) -> Optional[torch.Tensor]:
    if temporal_attention is None:
        return None

    mean_attention = temporal_attention.mean(dim=1)[0]
    visual_mask = (absolute_positions >= 1) & (absolute_positions < 1 + original_visual_token_count)
    action_mask = torch.isin(absolute_positions, action_positions_abs)
    if int(visual_mask.sum().item()) == 0 or int(action_mask.sum().item()) == 0:
        return None

    current_visual_positions = absolute_positions[visual_mask]
    action_to_visual = mean_attention[action_mask][:, visual_mask].mean(dim=0)
    full_vec = torch.zeros((original_visual_token_count,), device=action_to_visual.device, dtype=action_to_visual.dtype)
    relative_positions = (current_visual_positions - 1).to(dtype=torch.long)
    full_vec[relative_positions] = action_to_visual
    return full_vec


def _vlapruner_forward(
    self,
    multimodal_embeddings: torch.Tensor,
    multimodal_attention_mask: Optional[torch.Tensor],
    num_visual_patch_tokens: int,
    extra_projected_tokens_count: int,
    prompt_token_count: int,
    compute_logits: bool,
) -> VLAPrunerForwardResult:
    cfg = getattr(self, "_vlapruner_cfg", None)
    runtime_ctx = getattr(self, "_vlapruner_runtime", {})
    language_model = self.language_model
    decoder = language_model.model

    effective_r = float(runtime_ctx.get("effective_fastv_r", float(getattr(cfg, "fastv_r", 0.75) if cfg is not None else 0.75)))
    historical_attention = runtime_ctx.get("historical_attention", None)
    temporal_ready = bool(runtime_ctx.get("temporal_ready", False))

    use_text_vision_selection = bool(getattr(cfg, "use_text_vision_selection", False)) if cfg is not None else False
    use_prefill_attention = bool(getattr(cfg, "use_prefil_attention", False)) if cfg is not None else False
    selection_mode = "action_mean"
    if use_text_vision_selection:
        selection_mode = "text_vision"
    elif use_prefill_attention:
        selection_mode = "prefill"

    fastv_k = int(getattr(cfg, "fastv_k", 3)) if cfg is not None else 3
    temporal_layer = int(getattr(cfg, "temporal_layer", 15)) if cfg is not None else 15
    if fastv_k >= len(decoder.layers):
        raise ValueError(
            f"fastv_k={fastv_k} is out of range for decoder with {len(decoder.layers)} layers."
        )
    if temporal_layer >= len(decoder.layers):
        raise ValueError(
            f"temporal_layer={temporal_layer} is out of range for decoder with {len(decoder.layers)} layers."
        )

    batch_size, total_seq_len, _ = multimodal_embeddings.shape
    if batch_size != 1:
        raise ValueError("VLA-Pruner runtime patch only supports batch size 1 during inference.")

    current_attention_mask = multimodal_attention_mask
    if current_attention_mask is None:
        current_attention_mask = torch.ones((batch_size, total_seq_len), dtype=torch.long, device=multimodal_embeddings.device)

    hidden_states = multimodal_embeddings
    absolute_positions = torch.arange(total_seq_len, device=multimodal_embeddings.device, dtype=torch.long)
    visual_start = 1
    original_visual_token_count = int(num_visual_patch_tokens)
    prompt_start = visual_start + original_visual_token_count + int(extra_projected_tokens_count)
    action_start = prompt_start + int(prompt_token_count)
    action_positions_abs = torch.arange(
        action_start,
        action_start + ACTION_DIM * NUM_ACTIONS_CHUNK,
        device=multimodal_embeddings.device,
        dtype=torch.long,
    )
    initial_visual_features = multimodal_embeddings[0, visual_start : visual_start + original_visual_token_count, :]

    probe_attention = None
    temporal_attention = None
    pruned = False
    current_visual_tokens = int(original_visual_token_count)

    for layer_idx, decoder_layer in enumerate(decoder.layers):
        if layer_idx == fastv_k and probe_attention is not None and effective_r > 0.0 and current_visual_tokens > 1:
            selection_scores = _compute_selection_scores(
                probe_attention,
                visual_start=visual_start,
                visual_token_count=original_visual_token_count,
                prompt_start=prompt_start,
                prompt_token_count=int(prompt_token_count),
                action_start=action_start,
                action_token_count=ACTION_DIM * NUM_ACTIONS_CHUNK,
                selection_mode=selection_mode,
            )
            num_keep = max(1, min(original_visual_token_count, int(round(original_visual_token_count * (1.0 - effective_r)))))

            if bool(getattr(cfg, "use_temporal", False)) and historical_attention is not None:
                guide_topk = historical_attention.topk(num_keep).indices
                current_topk = selection_scores.topk(num_keep).indices
                combined_indices = torch.cat([guide_topk, current_topk], dim=0)
                unique_indices = torch.unique(combined_indices)
                if len(unique_indices) > num_keep:
                    selected_features = initial_visual_features[unique_indices]
                    final_indices = _redundancy_minization(selected_features, num_keep)
                    final_topk = unique_indices[final_indices]
                else:
                    final_topk = unique_indices
                top_visual_indices = final_topk
            else:
                top_visual_indices = selection_scores.topk(num_keep).indices

            top_attention_rank_index = top_visual_indices + visual_start
            keep_indices = torch.cat(
                [
                    torch.arange(visual_start, device=hidden_states.device, dtype=torch.long),
                    top_attention_rank_index,
                    torch.arange(visual_start + original_visual_token_count, total_seq_len, device=hidden_states.device, dtype=torch.long),
                ],
                dim=0,
            ).sort().values
            hidden_states = hidden_states.index_select(1, keep_indices)
            current_attention_mask = current_attention_mask.index_select(1, keep_indices)
            absolute_positions = keep_indices
            current_visual_tokens = int(top_visual_indices.shape[0])
            pruned = True

        current_len = hidden_states.shape[1]
        cache_positions = torch.arange(current_len, device=hidden_states.device, dtype=torch.long)
        current_position_ids = absolute_positions.unsqueeze(0)
        causal_mask = decoder._update_causal_mask(current_attention_mask, hidden_states, cache_positions, 0)
        need_attention = layer_idx in {fastv_k - 1, temporal_layer}

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
        if need_attention:
            if layer_idx == fastv_k - 1:
                probe_attention = layer_outputs[1]
            if layer_idx == temporal_layer:
                temporal_attention = layer_outputs[1]

    hidden_states = decoder.norm(hidden_states)
    logits = _compute_logits_from_hidden_states(language_model, hidden_states) if compute_logits else None
    temporal_vector = _make_temporal_vector(
        temporal_attention=temporal_attention,
        absolute_positions=absolute_positions,
        original_visual_token_count=original_visual_token_count,
        action_positions_abs=action_positions_abs,
    )
    return VLAPrunerForwardResult(
        hidden_states=hidden_states,
        logits=logits,
        current_visual_tokens=current_visual_tokens,
        pruned=pruned,
        temporal_vector=temporal_vector,
        effective_r=effective_r,
        selection_mode=selection_mode,
    )


def _forward_language_model_with_optional_vlapruner(
    self,
    multimodal_embeddings: torch.Tensor,
    multimodal_attention_mask: Optional[torch.Tensor],
    num_visual_patch_tokens: int,
    extra_projected_tokens_count: int,
    prompt_token_count: int,
    compute_logits: bool,
) -> VLAPrunerForwardResult:
    runtime_ctx = getattr(self, "_vlapruner_runtime", {})
    temporal_ready = bool(runtime_ctx.get("temporal_ready", False))

    if not _should_use_vlapruner(self, num_visual_patch_tokens) or not _supports_language_model(self):
        result = _dense_forward(self, multimodal_embeddings, multimodal_attention_mask, compute_logits)
        _record_vlapruner_metrics(
            self,
            tokens_before=int(num_visual_patch_tokens),
            tokens_after=int(num_visual_patch_tokens),
            pruned=False,
            effective_r=0.0,
            selection_mode="dense",
            temporal_ready=temporal_ready,
            fallback=False,
        )
        return result

    try:
        result = _vlapruner_forward(
            self,
            multimodal_embeddings,
            multimodal_attention_mask,
            num_visual_patch_tokens=num_visual_patch_tokens,
            extra_projected_tokens_count=extra_projected_tokens_count,
            prompt_token_count=prompt_token_count,
            compute_logits=compute_logits,
        )
        _record_vlapruner_metrics(
            self,
            tokens_before=int(num_visual_patch_tokens),
            tokens_after=int(result.current_visual_tokens if result.pruned else num_visual_patch_tokens),
            pruned=bool(result.pruned),
            effective_r=result.effective_r,
            selection_mode=result.selection_mode,
            temporal_ready=temporal_ready,
            fallback=False,
        )
        return result
    except Exception as exc:
        if not getattr(self, "_vlapruner_logged_fallback", False):
            print(f"[VLA-Pruner] fallback to dense language-model forward: {exc}")
            self._vlapruner_logged_fallback = True
        result = _dense_forward(self, multimodal_embeddings, multimodal_attention_mask, compute_logits)
        _record_vlapruner_metrics(
            self,
            tokens_before=int(num_visual_patch_tokens),
            tokens_after=int(num_visual_patch_tokens),
            pruned=False,
            effective_r=0.0,
            selection_mode="dense",
            temporal_ready=temporal_ready,
            fallback=True,
        )
        return result


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
            f"Expected {expected_tokens} action tokens after VLA-Pruner forward, got {actions_hidden_states.shape[1]}."
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
            f"Expected {expected_tokens} action logits after VLA-Pruner forward, got {action_logits.shape[1]}."
        )
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
        input_embeddings, projected_patch_embeddings, attention_mask
    )

    num_visual_patch_tokens = int(self.vision_backbone.get_num_patches()) * int(self.vision_backbone.get_num_images_in_input())
    num_visual_patch_tokens = min(num_visual_patch_tokens, int(projected_patch_embeddings.shape[1]))
    extra_projected_tokens_count = int(projected_patch_embeddings.shape[1]) - int(num_visual_patch_tokens)

    language_model_output = _forward_language_model_with_optional_vlapruner(
        self,
        multimodal_embeddings,
        multimodal_attention_mask,
        num_visual_patch_tokens=num_visual_patch_tokens,
        extra_projected_tokens_count=extra_projected_tokens_count,
        prompt_token_count=int(num_prompt_tokens),
        compute_logits=action_head is None,
    )

    runtime_ctx = getattr(self, "_vlapruner_runtime", None)
    if runtime_ctx is not None:
        runtime_ctx["last_action_vision_attention"] = language_model_output.temporal_vector
        runtime_ctx["last_effective_r"] = language_model_output.effective_r
        runtime_ctx["last_selection_mode"] = language_model_output.selection_mode

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

        language_model_output = _forward_language_model_with_optional_vlapruner(
            self,
            multimodal_embeddings,
            multimodal_attention_mask,
            num_visual_patch_tokens=num_visual_patch_tokens,
            extra_projected_tokens_count=extra_projected_tokens_count,
            prompt_token_count=int(num_prompt_tokens),
            compute_logits=False,
        )

        runtime_ctx = getattr(self, "_vlapruner_runtime", None)
        if runtime_ctx is not None:
            runtime_ctx["last_action_vision_attention"] = language_model_output.temporal_vector
            runtime_ctx["last_effective_r"] = language_model_output.effective_r
            runtime_ctx["last_selection_mode"] = language_model_output.selection_mode

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


def predict_action(
    self,
    input_ids: Optional[torch.LongTensor] = None,
    unnorm_key: Optional[str] = None,
    proprio=None,
    proprio_projector=None,
    action_head=None,
    noisy_action_projector=None,
    use_film: bool = False,
    **kwargs: str,
) -> np.ndarray:
    cfg = getattr(self, "_vlapruner_cfg", None)
    historical_attention = None
    temporal_ready = False
    effective_fastv_r = 0.0
    selection_mode = "action_mean"
    if cfg is not None:
        if bool(getattr(cfg, "use_text_vision_selection", False)):
            selection_mode = "text_vision"
        elif bool(getattr(cfg, "use_prefil_attention", False)):
            selection_mode = "prefill"

        if bool(getattr(cfg, "use_fastv", False)):
            if bool(getattr(cfg, "use_temporal", False)):
                historical_attention, temporal_ready = _build_temporal_history(self)
                if temporal_ready and historical_attention is not None:
                    effective_fastv_r = float(getattr(cfg, "fastv_r", 0.75))
                else:
                    effective_fastv_r = 0.0
            else:
                effective_fastv_r = float(getattr(cfg, "fastv_r", 0.75))

    self._vlapruner_runtime = {
        "historical_attention": historical_attention,
        "temporal_ready": temporal_ready,
        "effective_fastv_r": effective_fastv_r,
        "selection_mode": selection_mode,
        "num_visual_patch_tokens": 0,
    }

    try:
        action_result = self._vlapruner_original_predict_action(
            input_ids=input_ids,
            unnorm_key=unnorm_key,
            proprio=proprio,
            proprio_projector=proprio_projector,
            action_head=action_head,
            noisy_action_projector=noisy_action_projector,
            use_film=use_film,
            **kwargs,
        )
        if len(action_result) == 3:
            actions, actions_hidden_states, _unused_caches = action_result
        else:
            actions, actions_hidden_states = action_result
    finally:
        runtime_ctx = getattr(self, "_vlapruner_runtime", None)
        if runtime_ctx is not None:
            temporal_vector = runtime_ctx.get("last_action_vision_attention", None)
            if temporal_vector is not None:
                self._vlapruner_av_hist.append(temporal_vector.detach().to(dtype=torch.float32).cpu().numpy())
            delattr(self, "_vlapruner_runtime")

    return actions, actions_hidden_states
