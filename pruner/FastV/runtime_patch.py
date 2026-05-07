"""Runtime monkeypatches that transplant FastV into openvla-oft.

This module keeps all FastV-specific logic inside `pruner/FastV` and patches the
loaded OpenVLA model at runtime instead of editing the main source tree.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MethodType
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F

from backend import load_backend_modules
from config import FASTV_QUERY_MODES

_backend = load_backend_modules()
openvla_utils = _backend.openvla_utils

from prismatic.vla.constants import ACTION_DIM, NUM_ACTIONS_CHUNK


@dataclass
class FastVMetrics:
    model_calls: int = 0
    prune_events: int = 0
    tokens_before: int = 0
    tokens_after: int = 0
    fallback_calls: int = 0
    last_keep_ratio: float = 1.0
    max_prune_rate: float = 0.0
    last_pruned_visual_tokens: int = 0
    last_kept_visual_tokens: int = 0
    last_query_mode: str = "last"

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
            "last_keep_ratio": float(self.last_keep_ratio),
            "max_prune_rate": float(self.max_prune_rate),
            "last_pruned_visual_tokens": float(self.last_pruned_visual_tokens),
            "last_kept_visual_tokens": float(self.last_kept_visual_tokens),
            "last_query_mode": self.last_query_mode,
        }


@dataclass
class FastVForwardResult:
    hidden_states: torch.Tensor
    current_positions: torch.LongTensor
    logits: Optional[torch.Tensor]
    pruned: bool
    kept_visual_tokens: int


def apply_runtime_patches() -> None:
    if getattr(openvla_utils, "_fastv_runtime_patch_applied", False):
        return
    openvla_utils._fastv_runtime_patch_applied = True


def patch_loaded_model(model: torch.nn.Module, cfg: Optional[Any] = None) -> None:
    target = model.module if hasattr(model, "module") else model
    if cfg is not None:
        target._fastv_cfg = cfg

    if not hasattr(target, "_fastv_metrics"):
        target._fastv_metrics = FastVMetrics()
    if not hasattr(target, "_fastv_episode_metrics"):
        target._fastv_episode_metrics = FastVMetrics()
    if not hasattr(target, "_fastv_logged_fallback"):
        target._fastv_logged_fallback = False

    if not hasattr(target, "_fastv_original_regression_prediction"):
        target._fastv_original_regression_prediction = target._regression_or_discrete_prediction
        target._regression_or_discrete_prediction = MethodType(_regression_or_discrete_prediction, target)

    if not hasattr(target, "_fastv_original_diffusion_prediction"):
        target._fastv_original_diffusion_prediction = target._run_diffusion_prediction
        target._run_diffusion_prediction = MethodType(_run_diffusion_prediction, target)

    if not hasattr(target, "fastv_reset_episode"):
        target.fastv_reset_episode = MethodType(fastv_reset_episode, target)
    if not hasattr(target, "fastv_get_metrics"):
        target.fastv_get_metrics = MethodType(fastv_get_metrics, target)

    target.fastv_reset_episode()


def get_runtime_metrics(model: torch.nn.Module) -> Dict[str, float]:
    target = model.module if hasattr(model, "module") else model
    metrics = getattr(target, "_fastv_metrics", None)
    if metrics is None:
        return FastVMetrics().snapshot()
    return metrics.snapshot()


def get_episode_runtime_metrics(model: torch.nn.Module) -> Dict[str, float]:
    target = model.module if hasattr(model, "module") else model
    metrics = getattr(target, "_fastv_episode_metrics", None)
    if metrics is None:
        return FastVMetrics().snapshot()
    return metrics.snapshot()


def fastv_reset_episode(self) -> None:
    self._fastv_episode_metrics = FastVMetrics()


def fastv_get_metrics(self) -> Dict[str, float]:
    return self._fastv_metrics.snapshot()


def _record_fastv_metrics(
    self,
    visual_tokens_before: int,
    visual_tokens_after: int,
    pruned: bool,
    query_mode: str,
    fallback: bool = False,
) -> None:
    keep_ratio = 1.0
    prune_rate = 0.0
    if visual_tokens_before > 0:
        keep_ratio = visual_tokens_after / float(visual_tokens_before)
        prune_rate = 1.0 - keep_ratio

    for metrics_name in ("_fastv_metrics", "_fastv_episode_metrics"):
        metrics = getattr(self, metrics_name, None)
        if metrics is None:
            continue
        metrics.model_calls += 1
        metrics.tokens_before += int(visual_tokens_before)
        metrics.tokens_after += int(visual_tokens_after)
        metrics.last_keep_ratio = float(keep_ratio)
        metrics.last_pruned_visual_tokens = int(max(0, visual_tokens_before - visual_tokens_after))
        metrics.last_kept_visual_tokens = int(visual_tokens_after)
        metrics.last_query_mode = str(query_mode)
        metrics.max_prune_rate = max(metrics.max_prune_rate, float(prune_rate))
        if pruned:
            metrics.prune_events += 1
        if fallback:
            metrics.fallback_calls += 1


def _build_absolute_action_mask(all_actions_mask: torch.Tensor, projected_patch_embeddings: torch.Tensor) -> torch.Tensor:
    device = projected_patch_embeddings.device
    batch_size = projected_patch_embeddings.shape[0]
    projected_tokens = projected_patch_embeddings.shape[1]
    inserted_false = torch.zeros((batch_size, projected_tokens), dtype=torch.bool, device=device)
    return torch.cat([all_actions_mask[:, :1], inserted_false, all_actions_mask[:, 1:]], dim=1)


def _get_query_mode(self) -> str:
    cfg = getattr(self, "_fastv_cfg", None)
    if cfg is None:
        return "last"
    query_mode = str(getattr(cfg, "fastv_query_mode", "last"))
    if query_mode not in FASTV_QUERY_MODES:
        return "last"
    return query_mode


def _should_use_fastv(self, num_visual_patch_tokens: int) -> bool:
    cfg = getattr(self, "_fastv_cfg", None)
    if cfg is None:
        return False
    if not bool(getattr(cfg, "fastv_enabled", False)):
        return False
    if num_visual_patch_tokens <= 0:
        return False
    if getattr(self, "training", False):
        return False
    return True


def _supports_fastv_language_model(self) -> bool:
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


def _dense_causal_lm_forward(
    self,
    multimodal_embeddings: torch.Tensor,
    multimodal_attention_mask: Optional[torch.Tensor],
    compute_logits: bool,
) -> FastVForwardResult:
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

    current_positions = torch.arange(
        multimodal_embeddings.shape[1], device=multimodal_embeddings.device, dtype=torch.long
    )
    logits = output.logits if compute_logits else None
    return FastVForwardResult(
        hidden_states=output.hidden_states[-1],
        current_positions=current_positions,
        logits=logits,
        pruned=False,
        kept_visual_tokens=0,
    )


def _select_visual_attention_scores(
    attention_weights: torch.Tensor,
    visual_start: int,
    visual_token_count: int,
    action_start: int,
    action_token_count: int,
    query_mode: str,
) -> torch.Tensor:
    mean_attention = attention_weights.mean(dim=1)
    if mean_attention.shape[0] != 1:
        raise ValueError("FastV runtime patch only supports batch size 1 during inference.")

    visual_positions = torch.arange(
        visual_start,
        visual_start + visual_token_count,
        device=mean_attention.device,
        dtype=torch.long,
    )

    if query_mode == "last":
        query_attention = mean_attention[0, -1]
    else:
        if action_token_count <= 0:
            query_attention = mean_attention[0, -1]
        elif query_mode == "last_action":
            query_attention = mean_attention[0, action_start + action_token_count - 1]
        elif query_mode == "action_mean":
            action_positions = torch.arange(
                action_start,
                action_start + action_token_count,
                device=mean_attention.device,
                dtype=torch.long,
            )
            action_to_visual_attention = mean_attention[0].index_select(0, action_positions).index_select(1, visual_positions)
            action_to_visual_attention = action_to_visual_attention / action_to_visual_attention.sum(
                dim=1, keepdim=True
            ).clamp_min(1e-12)
            return action_to_visual_attention.mean(dim=0)
        else:
            raise ValueError(f"Unsupported FastV query mode: {query_mode}")

    return query_attention.index_select(0, visual_positions)


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


def _fastv_causal_lm_forward(
    self,
    multimodal_embeddings: torch.Tensor,
    multimodal_attention_mask: Optional[torch.Tensor],
    num_visual_patch_tokens: int,
    extra_projected_tokens_count: int,
    prompt_token_count: int,
    compute_logits: bool,
) -> FastVForwardResult:
    language_model = self.language_model
    decoder = language_model.model
    device = multimodal_embeddings.device
    batch_size, total_seq_len, _ = multimodal_embeddings.shape

    if batch_size != 1:
        raise ValueError("FastV runtime patch only supports batch size 1 during inference.")

    cfg = getattr(self, "_fastv_cfg", None)
    if cfg is None:
        raise ValueError("FastV configuration is missing on the loaded model.")

    fastv_k = int(getattr(cfg, "fastv_k", 3))
    fastv_r = float(getattr(cfg, "fastv_r", 0.5))
    min_keep_tokens = int(getattr(cfg, "fastv_min_keep_tokens", 1))
    query_mode = _get_query_mode(self)

    if fastv_k >= len(decoder.layers):
        raise ValueError(
            f"fastv_k={fastv_k} is out of range for a decoder with {len(decoder.layers)} layers. "
            "Choose a layer index strictly smaller than the number of decoder layers."
        )

    current_attention_mask = multimodal_attention_mask
    if current_attention_mask is None:
        current_attention_mask = torch.ones((batch_size, total_seq_len), dtype=torch.long, device=device)

    absolute_positions = torch.arange(total_seq_len, device=device, dtype=torch.long)
    cache_positions = torch.arange(total_seq_len, device=device, dtype=torch.long)
    hidden_states = multimodal_embeddings
    causal_mask = decoder._update_causal_mask(current_attention_mask, hidden_states, cache_positions, 0)
    visual_start = 1
    action_start = visual_start + int(num_visual_patch_tokens) + int(extra_projected_tokens_count) + int(prompt_token_count)
    action_token_count = ACTION_DIM * NUM_ACTIONS_CHUNK

    probe_attention = None
    kept_visual_tokens = int(num_visual_patch_tokens)
    pruned = False

    for layer_idx, decoder_layer in enumerate(decoder.layers):
        if layer_idx == fastv_k and probe_attention is not None and num_visual_patch_tokens > 0:
            keep_total = max(min_keep_tokens, int(round(num_visual_patch_tokens * (1.0 - fastv_r))))
            keep_total = max(1, min(keep_total, num_visual_patch_tokens))

            if keep_total < num_visual_patch_tokens:
                visual_scores = _select_visual_attention_scores(
                    probe_attention,
                    visual_start=visual_start,
                    visual_token_count=int(num_visual_patch_tokens),
                    action_start=action_start,
                    action_token_count=action_token_count,
                    query_mode=query_mode,
                )
                topk_rel = torch.topk(visual_scores, k=keep_total, largest=True, sorted=False).indices
                visual_positions = torch.arange(
                    visual_start,
                    visual_start + num_visual_patch_tokens,
                    device=device,
                    dtype=torch.long,
                )
                kept_visual_positions = visual_positions.index_select(0, topk_rel).sort().values
                tail_positions = torch.arange(
                    visual_start + num_visual_patch_tokens,
                    total_seq_len,
                    device=device,
                    dtype=torch.long,
                )
                keep_positions = torch.cat(
                    [torch.zeros(1, device=device, dtype=torch.long), kept_visual_positions, tail_positions],
                    dim=0,
                ).sort().values

                hidden_states = hidden_states.index_select(1, keep_positions)
                current_attention_mask = current_attention_mask.index_select(1, keep_positions)
                absolute_positions = keep_positions
                cache_positions = torch.arange(hidden_states.shape[1], device=device, dtype=torch.long)
                causal_mask = decoder._update_causal_mask(current_attention_mask, hidden_states, cache_positions, 0)
                kept_visual_tokens = int(keep_total)
                pruned = True

        position_ids = absolute_positions.unsqueeze(0)
        need_attention_probe = layer_idx == (fastv_k - 1)
        layer_outputs = decoder_layer(
            hidden_states,
            attention_mask=causal_mask,
            position_ids=position_ids,
            past_key_value=None,
            output_attentions=True,
            use_cache=False,
            cache_position=cache_positions,
        )

        hidden_states = layer_outputs[0]
        if need_attention_probe:
            probe_attention = layer_outputs[1]

    hidden_states = decoder.norm(hidden_states)
    logits = _compute_logits_from_hidden_states(language_model, hidden_states) if compute_logits else None
    return FastVForwardResult(
        hidden_states=hidden_states,
        current_positions=absolute_positions,
        logits=logits,
        pruned=pruned,
        kept_visual_tokens=kept_visual_tokens,
    )


def _forward_language_model_with_optional_fastv(
    self,
    multimodal_embeddings: torch.Tensor,
    multimodal_attention_mask: Optional[torch.Tensor],
    num_visual_patch_tokens: int,
    extra_projected_tokens_count: int,
    prompt_token_count: int,
    compute_logits: bool,
) -> FastVForwardResult:
    query_mode = _get_query_mode(self)
    visual_tokens_after = int(num_visual_patch_tokens)

    if not _should_use_fastv(self, num_visual_patch_tokens) or not _supports_fastv_language_model(self):
        result = _dense_causal_lm_forward(self, multimodal_embeddings, multimodal_attention_mask, compute_logits)
        _record_fastv_metrics(
            self,
            visual_tokens_before=int(num_visual_patch_tokens),
            visual_tokens_after=visual_tokens_after,
            pruned=False,
            query_mode=query_mode,
            fallback=False,
        )
        result.kept_visual_tokens = visual_tokens_after
        return result

    try:
        result = _fastv_causal_lm_forward(
            self,
            multimodal_embeddings,
            multimodal_attention_mask,
            num_visual_patch_tokens,
            extra_projected_tokens_count,
            prompt_token_count,
            compute_logits,
        )
        visual_tokens_after = int(result.kept_visual_tokens)
        _record_fastv_metrics(
            self,
            visual_tokens_before=int(num_visual_patch_tokens),
            visual_tokens_after=visual_tokens_after,
            pruned=bool(result.pruned),
            query_mode=query_mode,
            fallback=False,
        )
        return result
    except Exception as exc:
        if not getattr(self, "_fastv_logged_fallback", False):
            print(f"[FastV] fallback to dense language-model forward: {exc}")
            self._fastv_logged_fallback = True
        result = _dense_causal_lm_forward(self, multimodal_embeddings, multimodal_attention_mask, compute_logits)
        result.kept_visual_tokens = visual_tokens_after
        _record_fastv_metrics(
            self,
            visual_tokens_before=int(num_visual_patch_tokens),
            visual_tokens_after=visual_tokens_after,
            pruned=False,
            query_mode=query_mode,
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
    action_hidden_states = sequence_hidden_states[:, -ACTION_DIM * NUM_ACTIONS_CHUNK - 1 : -1, :]
    expected_tokens = ACTION_DIM * NUM_ACTIONS_CHUNK
    if action_hidden_states.shape[1] != expected_tokens:
        raise RuntimeError(
            f"Expected {expected_tokens} action tokens after FastV pruning, got {action_hidden_states.shape[1]}."
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
        raise RuntimeError(
            f"Expected {expected_tokens} action logits after FastV pruning, got {action_logits.shape[1]}."
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

    language_model_output = _forward_language_model_with_optional_fastv(
        self,
        multimodal_embeddings,
        multimodal_attention_mask,
        num_visual_patch_tokens,
        extra_projected_tokens_count,
        int(num_prompt_tokens),
        compute_logits=action_head is None,
    )

    actions_hidden_states = _extract_action_hidden_states(
        language_model_output.hidden_states,
        current_visual_tokens=language_model_output.kept_visual_tokens,
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
            current_visual_tokens=language_model_output.kept_visual_tokens,
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
    all_actions_mask_bool = all_actions_mask

    for t in action_head.noise_scheduler.timesteps:
        timesteps = torch.Tensor([t]).to(labels.device)
        diffusion_timestep_embeddings = (
            action_head.time_encoder(timesteps).to(curr_noisy_actions.dtype).to(curr_noisy_actions.device)
        )
        diffusion_timestep_embeddings = diffusion_timestep_embeddings.unsqueeze(1)

        projected_patch_embeddings = torch.cat(
            (orig_projected_patch_embeddings, diffusion_timestep_embeddings),
            dim=1,
        )

        batch_size = curr_noisy_actions.shape[0]
        orig_shape = curr_noisy_actions.shape
        curr_noisy_actions = curr_noisy_actions.reshape(batch_size, -1).unsqueeze(-1)
        noisy_action_features = noisy_action_projector(curr_noisy_actions)
        curr_noisy_actions = curr_noisy_actions.reshape(orig_shape)

        replaced_input_embeddings = self._replace_input_embeddings(
            input_embeddings.clone(), all_actions_mask_bool, noisy_action_features
        )
        multimodal_embeddings, multimodal_attention_mask = self._build_multimodal_attention(
            replaced_input_embeddings,
            projected_patch_embeddings,
            attention_mask,
        )

        num_visual_patch_tokens = int(self.vision_backbone.get_num_patches()) * int(
            self.vision_backbone.get_num_images_in_input()
        )
        num_visual_patch_tokens = min(num_visual_patch_tokens, int(projected_patch_embeddings.shape[1]))
        extra_projected_tokens_count = int(projected_patch_embeddings.shape[1]) - int(num_visual_patch_tokens)

        language_model_output = _forward_language_model_with_optional_fastv(
            self,
            multimodal_embeddings,
            multimodal_attention_mask,
            num_visual_patch_tokens,
            extra_projected_tokens_count,
            int(num_prompt_tokens),
            compute_logits=False,
        )

        actions_hidden_states = _extract_action_hidden_states(
            language_model_output.hidden_states,
            current_visual_tokens=language_model_output.kept_visual_tokens,
            extra_projected_tokens_count=extra_projected_tokens_count,
            prompt_token_count=int(num_prompt_tokens),
        )

        noise_pred = action_head.predict_noise(actions_hidden_states)
        curr_noisy_actions = action_head.noise_scheduler.step(noise_pred, t, curr_noisy_actions).prev_sample

    curr_noisy_actions = curr_noisy_actions.reshape(NUM_ACTIONS_CHUNK, ACTION_DIM)
    return curr_noisy_actions.float().cpu().detach().numpy(), actions_hidden_states
