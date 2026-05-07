"""Runtime monkeypatches that transplant SpecPrune-VLA into OpenVLA-OFT."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from types import MethodType
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from backend import load_backend_modules

_backend = load_backend_modules()
openvla_utils = _backend.openvla_utils

from prismatic.vla.constants import ACTION_DIM, NUM_ACTIONS_CHUNK


@dataclass
class SpecPruneCache:
    global_scores: Optional[torch.Tensor] = None
    global_indices_by_image: Optional[List[torch.Tensor]] = None
    embedding_history: List[torch.Tensor] = field(default_factory=list)
    last_actions: List[np.ndarray] = field(default_factory=list)
    layer_confidence: Dict[int, float] = field(default_factory=dict)
    calls: int = 0


@dataclass
class SpecPruneMetrics:
    model_calls: int = 0
    static_prune_events: int = 0
    dynamic_prune_events: int = 0
    tokens_before: int = 0
    tokens_after_static: int = 0
    tokens_after_final: int = 0
    fallback_calls: int = 0
    global_cache_hits: int = 0
    global_cache_misses: int = 0
    last_action_mode: str = "unknown"
    last_visual_tokens: int = 0
    last_static_visual_tokens: int = 0
    last_final_visual_tokens: int = 0
    last_keep_ratio: float = 1.0
    last_static_keep_ratio: float = 1.0
    last_dynamic_layers: str = ""

    def snapshot(self) -> Dict[str, float]:
        static_rate = 0.0
        final_rate = 0.0
        if self.tokens_before > 0:
            static_rate = 1.0 - (self.tokens_after_static / float(self.tokens_before))
            final_rate = 1.0 - (self.tokens_after_final / float(self.tokens_before))
        return {
            "model_calls": float(self.model_calls),
            "static_prune_events": float(self.static_prune_events),
            "dynamic_prune_events": float(self.dynamic_prune_events),
            "tokens_before": float(self.tokens_before),
            "tokens_after_static": float(self.tokens_after_static),
            "tokens_after_final": float(self.tokens_after_final),
            "weighted_static_prune_rate": float(static_rate),
            "weighted_final_prune_rate": float(final_rate),
            "fallback_calls": float(self.fallback_calls),
            "global_cache_hits": float(self.global_cache_hits),
            "global_cache_misses": float(self.global_cache_misses),
            "last_action_mode": self.last_action_mode,
            "last_visual_tokens": float(self.last_visual_tokens),
            "last_static_visual_tokens": float(self.last_static_visual_tokens),
            "last_final_visual_tokens": float(self.last_final_visual_tokens),
            "last_keep_ratio": float(self.last_keep_ratio),
            "last_static_keep_ratio": float(self.last_static_keep_ratio),
            "last_dynamic_layers": self.last_dynamic_layers,
        }


@dataclass
class SpecPruneForwardResult:
    hidden_states: torch.Tensor
    logits: Optional[torch.Tensor]
    current_visual_tokens: int
    static_visual_tokens: int
    used_specprune: bool
    action_mode: str
    dynamic_layers: List[int]
    global_cache_hit: bool


def apply_runtime_patches() -> None:
    if getattr(openvla_utils, "_specprune_runtime_patch_applied", False):
        return
    openvla_utils._specprune_runtime_patch_applied = True


def patch_loaded_model(model: torch.nn.Module, cfg: Optional[Any] = None) -> None:
    target = model.module if hasattr(model, "module") else model
    if cfg is not None:
        target._specprune_cfg = cfg

    if not hasattr(target, "_specprune_metrics"):
        target._specprune_metrics = SpecPruneMetrics()
    if not hasattr(target, "_specprune_episode_metrics"):
        target._specprune_episode_metrics = SpecPruneMetrics()
    if not hasattr(target, "_specprune_cache"):
        target._specprune_cache = SpecPruneCache()
    if not hasattr(target, "_specprune_logged_fallback"):
        target._specprune_logged_fallback = False

    if not hasattr(target, "_specprune_original_regression_prediction"):
        target._specprune_original_regression_prediction = target._regression_or_discrete_prediction
        target._regression_or_discrete_prediction = MethodType(_regression_or_discrete_prediction, target)

    if not hasattr(target, "_specprune_original_diffusion_prediction"):
        target._specprune_original_diffusion_prediction = target._run_diffusion_prediction
        target._run_diffusion_prediction = MethodType(_run_diffusion_prediction, target)

    if not hasattr(target, "specprune_reset_episode"):
        target.specprune_reset_episode = MethodType(specprune_reset_episode, target)
    if not hasattr(target, "specprune_get_metrics"):
        target.specprune_get_metrics = MethodType(specprune_get_metrics, target)
    if not hasattr(target, "specprune_record_actions"):
        target.specprune_record_actions = MethodType(specprune_record_actions, target)

    target.specprune_reset_episode()


def get_runtime_metrics(model: torch.nn.Module) -> Dict[str, float]:
    target = model.module if hasattr(model, "module") else model
    metrics = getattr(target, "_specprune_metrics", None)
    if metrics is None:
        return SpecPruneMetrics().snapshot()
    return metrics.snapshot()


def get_episode_runtime_metrics(model: torch.nn.Module) -> Dict[str, float]:
    target = model.module if hasattr(model, "module") else model
    metrics = getattr(target, "_specprune_episode_metrics", None)
    if metrics is None:
        return SpecPruneMetrics().snapshot()
    return metrics.snapshot()


def specprune_reset_episode(self) -> None:
    self._specprune_episode_metrics = SpecPruneMetrics()
    cache = getattr(self, "_specprune_cache", None)
    if cache is not None:
        cache.global_scores = None
        cache.global_indices_by_image = None
        cache.embedding_history.clear()
        cache.last_actions.clear()
        cache.layer_confidence.clear()
        cache.calls = 0


def specprune_get_metrics(self) -> Dict[str, float]:
    return self._specprune_metrics.snapshot()


def specprune_record_actions(self, actions) -> None:
    cache = getattr(self, "_specprune_cache", None)
    if cache is None or actions is None:
        return
    cfg = getattr(self, "_specprune_cfg", None)
    history_size = 16 if cfg is None else int(getattr(cfg, "specprune_history_size", 16))
    arr = np.asarray(actions, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr[None, :]
    for action in arr:
        cache.last_actions.append(np.asarray(action, dtype=np.float32))
    if len(cache.last_actions) > history_size:
        del cache.last_actions[:-history_size]


def _parse_int_list(value: Any, default: Sequence[int]) -> List[int]:
    if value is None:
        return list(default)
    if isinstance(value, str):
        if value.strip() == "":
            return []
        items = [item.strip() for item in value.split(",") if item.strip()]
        if not items:
            return []
        return [int(item) for item in items]
    return [int(item) for item in value]


def _record_metrics(
    self,
    visual_tokens_before: int,
    static_visual_tokens: int,
    final_visual_tokens: int,
    used_specprune: bool,
    action_mode: str,
    dynamic_layers: Sequence[int],
    global_cache_hit: bool,
    fallback: bool = False,
) -> None:
    static_keep_ratio = 1.0
    final_keep_ratio = 1.0
    if visual_tokens_before > 0:
        static_keep_ratio = static_visual_tokens / float(visual_tokens_before)
        final_keep_ratio = final_visual_tokens / float(visual_tokens_before)

    for metrics_name in ("_specprune_metrics", "_specprune_episode_metrics"):
        metrics = getattr(self, metrics_name, None)
        if metrics is None:
            continue
        metrics.model_calls += 1
        metrics.tokens_before += int(visual_tokens_before)
        metrics.tokens_after_static += int(static_visual_tokens)
        metrics.tokens_after_final += int(final_visual_tokens)
        metrics.last_action_mode = str(action_mode)
        metrics.last_visual_tokens = int(visual_tokens_before)
        metrics.last_static_visual_tokens = int(static_visual_tokens)
        metrics.last_final_visual_tokens = int(final_visual_tokens)
        metrics.last_static_keep_ratio = float(static_keep_ratio)
        metrics.last_keep_ratio = float(final_keep_ratio)
        metrics.last_dynamic_layers = ",".join(str(x) for x in dynamic_layers)
        if used_specprune and static_visual_tokens < visual_tokens_before:
            metrics.static_prune_events += 1
        if used_specprune and final_visual_tokens < static_visual_tokens:
            metrics.dynamic_prune_events += 1
        if global_cache_hit:
            metrics.global_cache_hits += 1
        else:
            metrics.global_cache_misses += 1
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


def _should_use_specprune(self, num_visual_patch_tokens: int, prompt_token_count: int) -> bool:
    cfg = getattr(self, "_specprune_cfg", None)
    if cfg is None:
        return False
    if not bool(getattr(cfg, "specprune_enabled", False)):
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
) -> SpecPruneForwardResult:
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
    return SpecPruneForwardResult(
        hidden_states=output.hidden_states[-1],
        logits=output.logits if compute_logits else None,
        current_visual_tokens=int(num_visual_patch_tokens),
        static_visual_tokens=int(num_visual_patch_tokens),
        used_specprune=False,
        action_mode="dense",
        dynamic_layers=[],
        global_cache_hit=False,
    )


def _global_warmup_causal_lm_forward(
    self,
    multimodal_embeddings: torch.Tensor,
    multimodal_attention_mask: Optional[torch.Tensor],
    projected_visual_embeddings: torch.Tensor,
    num_visual_patch_tokens: int,
    extra_projected_tokens_count: int,
    prompt_token_count: int,
    compute_logits: bool,
) -> SpecPruneForwardResult:
    language_model = self.language_model
    decoder = language_model.model
    cfg = getattr(self, "_specprune_cfg", None)
    cache = getattr(self, "_specprune_cache", None)
    if cfg is None or cache is None:
        return _dense_causal_lm_forward(
            self,
            multimodal_embeddings,
            multimodal_attention_mask,
            num_visual_patch_tokens,
            compute_logits,
        )

    batch_size, total_seq_len, _ = multimodal_embeddings.shape
    if batch_size != 1:
        raise ValueError("SpecPrune warmup only supports batch size 1 during inference.")

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
    global_layers = set(_parse_int_list(getattr(cfg, "specprune_global_layers", "15,32"), [15, 32]))
    global_layers = {min(max(1, layer), len(decoder.layers)) for layer in global_layers}

    last_global_scores = None
    for layer_idx, decoder_layer in enumerate(decoder.layers):
        layer_number = layer_idx + 1
        current_len = hidden_states.shape[1]
        cache_positions = torch.arange(current_len, device=device, dtype=torch.long)
        position_ids = absolute_positions.unsqueeze(0)
        causal_mask = decoder._update_causal_mask(current_attention_mask, hidden_states, cache_positions, 0)
        need_attention = layer_number in global_layers
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
        if need_attention:
            attention = layer_outputs[1]
            scores = _visual_importance_scores(
                cfg,
                attention,
                visual_start=visual_start,
                visual_token_count=current_visual_tokens,
                extra_projected_tokens_count=int(extra_projected_tokens_count),
                prompt_token_count=int(prompt_token_count),
            )
            last_global_scores = scores if last_global_scores is None else last_global_scores + scores

    if last_global_scores is not None:
        _update_global_cache(
            cache,
            cfg,
            last_global_scores,
            num_images=num_images,
            per_image_patches=per_image_patches,
        )
    cache.calls += 1
    _update_embedding_history(cache, cfg, projected_visual_embeddings)

    hidden_states = decoder.norm(hidden_states)
    logits = _compute_logits_from_hidden_states(language_model, hidden_states) if compute_logits else None
    return SpecPruneForwardResult(
        hidden_states=hidden_states,
        logits=logits,
        current_visual_tokens=int(num_visual_patch_tokens),
        static_visual_tokens=int(num_visual_patch_tokens),
        used_specprune=False,
        action_mode="warmup",
        dynamic_layers=[],
        global_cache_hit=False,
    )


def _get_action_mode(self) -> str:
    cfg = getattr(self, "_specprune_cfg", None)
    default_mode = "fine" if cfg is None else str(getattr(cfg, "specprune_default_action_mode", "fine"))
    cache = getattr(self, "_specprune_cache", None)
    if cache is None or not cache.last_actions:
        return default_mode

    action = np.asarray(cache.last_actions[-1], dtype=np.float32).reshape(-1)
    if action.shape[0] < 6:
        return default_mode
    delta_xyz = action[:3]
    delta_rot = action[3:6]
    vt = float(np.linalg.norm(delta_xyz))
    vr = float(np.linalg.norm(delta_rot))
    dz = float(delta_xyz[2])
    vt_th = 0.5 if cfg is None else float(getattr(cfg, "specprune_velocity_vt_threshold", 0.5))
    vr_th = 0.2 if cfg is None else float(getattr(cfg, "specprune_velocity_vr_threshold", 0.2))
    if vt < vt_th and vr < vr_th and dz <= 0.0:
        return "fine"
    return "coarse"


def _scaled_k(base_k: int, alpha: float, min_value: int = 1) -> int:
    return max(min_value, int(round(float(base_k) * float(alpha))))


def _get_mode_ks(cfg: Any, action_mode: str) -> Tuple[int, int, int]:
    alpha = float(getattr(cfg, "specprune_alpha", 0.8))
    k_global = _scaled_k(int(getattr(cfg, "specprune_global_base_k", 30)), alpha)
    if action_mode == "fine":
        k_local = _scaled_k(int(getattr(cfg, "specprune_local_fine_base_k", 40)), alpha)
        k_dynamic = _scaled_k(int(getattr(cfg, "specprune_dynamic_fine_base_k", 10)), alpha, min_value=0)
    else:
        k_local = _scaled_k(int(getattr(cfg, "specprune_local_coarse_base_k", 24)), alpha)
        k_dynamic = _scaled_k(int(getattr(cfg, "specprune_dynamic_coarse_base_k", 20)), alpha, min_value=0)
    return k_global, k_local, k_dynamic


def _text_to_visual_scores(
    attention_weights: torch.Tensor,
    visual_start: int,
    visual_token_count: int,
    extra_projected_tokens_count: int,
    prompt_token_count: int,
) -> torch.Tensor:
    if attention_weights.dim() != 4 or attention_weights.shape[0] != 1:
        raise ValueError(f"Unexpected attention shape for SpecPrune: {tuple(attention_weights.shape)}")
    mean_attention = attention_weights.mean(dim=1)
    prompt_start = visual_start + int(visual_token_count) + int(extra_projected_tokens_count)
    prompt_end = prompt_start + int(prompt_token_count)
    if prompt_end <= prompt_start:
        raise ValueError("SpecPrune text token range is empty.")
    scores = mean_attention[0, prompt_start:prompt_end, visual_start : visual_start + visual_token_count].mean(dim=0)
    return scores.float()


def _action_to_visual_scores(
    attention_weights: torch.Tensor,
    visual_start: int,
    visual_token_count: int,
    extra_projected_tokens_count: int,
    prompt_token_count: int,
) -> torch.Tensor:
    if attention_weights.dim() != 4 or attention_weights.shape[0] != 1:
        raise ValueError(f"Unexpected attention shape for SpecPrune: {tuple(attention_weights.shape)}")
    mean_attention = attention_weights.mean(dim=1)
    action_start = visual_start + int(visual_token_count) + int(extra_projected_tokens_count) + int(prompt_token_count)
    action_end = action_start + ACTION_DIM * NUM_ACTIONS_CHUNK
    if action_end <= action_start or action_end > mean_attention.shape[-1]:
        raise ValueError("SpecPrune action token range is invalid.")
    return mean_attention[0, action_start:action_end, visual_start : visual_start + visual_token_count].mean(dim=0).float()


def _visual_importance_scores(
    cfg: Any,
    attention_weights: torch.Tensor,
    visual_start: int,
    visual_token_count: int,
    extra_projected_tokens_count: int,
    prompt_token_count: int,
) -> torch.Tensor:
    if str(getattr(cfg, "specprune_query_mode", "action_mean")) == "action_mean":
        return _action_to_visual_scores(
            attention_weights,
            visual_start=visual_start,
            visual_token_count=visual_token_count,
            extra_projected_tokens_count=extra_projected_tokens_count,
            prompt_token_count=prompt_token_count,
        )
    return _text_to_visual_scores(
        attention_weights,
        visual_start=visual_start,
        visual_token_count=visual_token_count,
        extra_projected_tokens_count=extra_projected_tokens_count,
        prompt_token_count=prompt_token_count,
    )


def _attention_entropy(
    attention_weights: torch.Tensor,
    visual_start: int,
    visual_token_count: int,
    extra_projected_tokens_count: int,
    prompt_token_count: int,
    epsilon: float,
) -> float:
    mean_attention = attention_weights.mean(dim=1)
    prompt_start = visual_start + int(visual_token_count) + int(extra_projected_tokens_count)
    prompt_end = prompt_start + int(prompt_token_count)
    block = mean_attention[0, prompt_start:prompt_end, visual_start : visual_start + visual_token_count].float()
    block = block / block.sum(dim=-1, keepdim=True).clamp_min(float(epsilon))
    entropy = -(block * torch.log(block.clamp_min(float(epsilon)))).sum(dim=-1).mean()
    return float(entropy.detach().cpu().item())


def _topk_by_image(scores: torch.Tensor, num_images: int, per_image_patches: int, k_per_image: int) -> List[torch.Tensor]:
    selected: List[torch.Tensor] = []
    k_per_image = max(0, min(int(k_per_image), int(per_image_patches)))
    for image_idx in range(num_images):
        start = image_idx * per_image_patches
        end = start + per_image_patches
        if k_per_image <= 0:
            selected.append(torch.empty(0, device=scores.device, dtype=torch.long))
        else:
            rel = torch.topk(scores[start:end], k=k_per_image, largest=True, sorted=False).indices
            selected.append((rel + start).to(dtype=torch.long))
    return selected


def _union_by_image(
    chunks: Sequence[Sequence[torch.Tensor]],
    num_images: int,
    device: torch.device,
) -> List[torch.Tensor]:
    selected: List[torch.Tensor] = []
    for image_idx in range(num_images):
        per_image = [group[image_idx] for group in chunks if image_idx < len(group) and group[image_idx].numel() > 0]
        if not per_image:
            selected.append(torch.empty(0, device=device, dtype=torch.long))
        else:
            selected.append(torch.unique(torch.cat(per_image, dim=0), sorted=True))
    return selected


def _ensure_min_static_keep(
    selected_by_image: List[torch.Tensor],
    scores: torch.Tensor,
    num_images: int,
    per_image_patches: int,
    min_keep_per_image: int,
) -> List[torch.Tensor]:
    output: List[torch.Tensor] = []
    for image_idx in range(num_images):
        start = image_idx * per_image_patches
        end = start + per_image_patches
        current = selected_by_image[image_idx]
        target = max(1, min(int(min_keep_per_image), int(per_image_patches)))
        if current.numel() >= target:
            output.append(torch.unique(current, sorted=True))
            continue
        add_count = target - int(current.numel())
        image_scores = scores[start:end].clone()
        if current.numel() > 0:
            image_scores[current - start] = -torch.inf
        rel = torch.topk(image_scores, k=add_count, largest=True, sorted=False).indices
        output.append(torch.unique(torch.cat([current, rel + start], dim=0), sorted=True))
    return output


def _concat_selected_by_image(selected_by_image: Sequence[torch.Tensor]) -> torch.Tensor:
    nonempty = [x for x in selected_by_image if x.numel() > 0]
    if not nonempty:
        raise ValueError("SpecPrune selected no visual tokens.")
    return torch.sort(torch.unique(torch.cat(nonempty, dim=0))).values


def _global_indices_for_static(
    cache: SpecPruneCache,
    k_global_per_image: int,
    num_images: int,
    per_image_patches: int,
    device: torch.device,
) -> Tuple[List[torch.Tensor], bool]:
    if cache.global_indices_by_image is None:
        return [torch.empty(0, device=device, dtype=torch.long) for _ in range(num_images)], False

    selected: List[torch.Tensor] = []
    for image_idx in range(num_images):
        if image_idx >= len(cache.global_indices_by_image):
            selected.append(torch.empty(0, device=device, dtype=torch.long))
            continue
        rel = cache.global_indices_by_image[image_idx].to(device=device, dtype=torch.long)
        rel = rel[: max(0, min(k_global_per_image, per_image_patches))]
        selected.append(rel + image_idx * per_image_patches)
    return selected, True


def _dynamic_indices_by_embedding(
    cache: SpecPruneCache,
    cfg: Any,
    projected_visual_embeddings: torch.Tensor,
    action_mode: str,
    num_images: int,
    per_image_patches: int,
    device: torch.device,
) -> List[torch.Tensor]:
    _, _, k_dynamic_total = _get_mode_ks(cfg, action_mode)
    k_dynamic_per_image = int(math.ceil(k_dynamic_total / float(max(1, num_images))))
    if k_dynamic_per_image <= 0 or not cache.embedding_history:
        return [torch.empty(0, device=device, dtype=torch.long) for _ in range(num_images)]

    latest = projected_visual_embeddings[0].detach().float()
    ref_idx = -1
    try:
        if cache.last_actions:
            last_action = np.asarray(cache.last_actions[-1], dtype=np.float32).reshape(-1)
            vt = float(np.linalg.norm(last_action[:3])) if last_action.shape[0] >= 3 else 0.0
            b = float(getattr(cfg, "specprune_velocity_window_b", 7.0))
            k = float(getattr(cfg, "specprune_velocity_window_k", -1.0))
            window = int(math.floor(b + k * vt) + 4)
            window = max(1, min(window, len(cache.embedding_history)))
            ref_idx = -window
    except Exception:
        ref_idx = -1
    reference = cache.embedding_history[ref_idx].to(device=device, dtype=torch.float32)
    sim = F.cosine_similarity(latest, reference, dim=-1)
    tau = float(getattr(cfg, "specprune_tau", 0.95))

    selected: List[torch.Tensor] = []
    for image_idx in range(num_images):
        start = image_idx * per_image_patches
        end = start + per_image_patches
        sim_i = sim[start:end]
        candidate = torch.where(sim_i < tau)[0]
        if candidate.numel() == 0:
            selected.append(torch.empty(0, device=device, dtype=torch.long))
            continue
        keep = min(k_dynamic_per_image, int(candidate.numel()))
        candidate_scores = sim_i.index_select(0, candidate)
        rel = candidate.index_select(0, torch.topk(candidate_scores, k=keep, largest=False, sorted=False).indices)
        selected.append(rel + start)
    return selected


def _update_embedding_history(
    cache: SpecPruneCache,
    cfg: Any,
    projected_visual_embeddings: torch.Tensor,
) -> None:
    history_size = int(getattr(cfg, "specprune_history_size", 16))
    cache.embedding_history.append(projected_visual_embeddings.detach().float().cpu()[0])
    if len(cache.embedding_history) > history_size:
        del cache.embedding_history[:-history_size]


def _update_global_cache(
    cache: SpecPruneCache,
    cfg: Any,
    attention_scores: torch.Tensor,
    num_images: int,
    per_image_patches: int,
) -> None:
    cache.global_scores = attention_scores.detach().float().cpu()
    cache_k = max(
        int(getattr(cfg, "specprune_global_cache_k", 64)),
        int(getattr(cfg, "specprune_global_base_k", 30)),
        int(getattr(cfg, "specprune_local_fine_base_k", 40)),
    )
    selected: List[torch.Tensor] = []
    cpu_scores = cache.global_scores
    for image_idx in range(num_images):
        start = image_idx * per_image_patches
        end = start + per_image_patches
        keep = max(1, min(cache_k, per_image_patches))
        rel = torch.topk(cpu_scores[start:end], k=keep, largest=True, sorted=True).indices.cpu()
        selected.append(rel)
    cache.global_indices_by_image = selected


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


def _rank_weight_scores(scores: torch.Tensor, temperature: float) -> torch.Tensor:
    ranks = torch.empty_like(scores, dtype=torch.float32)
    order = torch.argsort(scores, descending=True)
    ranks[order] = torch.arange(scores.numel(), device=scores.device, dtype=torch.float32)
    weights = torch.sigmoid(-float(temperature) * ranks)
    return weights / weights.sum().clamp_min(1e-12)


def _select_dynamic_keep_indices(
    cumulative_scores: torch.Tensor,
    current_to_original: torch.Tensor,
    num_images: int,
    per_image_patches: int,
    target_keep_total: int,
    min_keep_per_image: int,
) -> torch.Tensor:
    selected: List[torch.Tensor] = []
    target_keep_total = max(num_images * min_keep_per_image, min(target_keep_total, int(cumulative_scores.numel())))
    base_keep = max(min_keep_per_image, int(math.floor(target_keep_total / float(num_images))))
    remaining = target_keep_total - base_keep * num_images
    used_positions: Set[int] = set()

    for image_idx in range(num_images):
        image_mask = (current_to_original >= image_idx * per_image_patches) & (
            current_to_original < (image_idx + 1) * per_image_patches
        )
        positions = torch.where(image_mask)[0]
        if positions.numel() == 0:
            continue
        keep_i = min(int(positions.numel()), base_keep + (1 if image_idx < remaining else 0))
        rel = torch.topk(cumulative_scores.index_select(0, positions), k=keep_i, largest=True, sorted=False).indices
        chosen = positions.index_select(0, rel)
        selected.append(chosen)
        used_positions.update(int(x) for x in chosen.detach().cpu().tolist())

    current_count = sum(int(x.numel()) for x in selected)
    if current_count < target_keep_total:
        all_order = torch.argsort(cumulative_scores, descending=True)
        extras = []
        for pos in all_order.detach().cpu().tolist():
            if int(pos) in used_positions:
                continue
            extras.append(int(pos))
            if current_count + len(extras) >= target_keep_total:
                break
        if extras:
            selected.append(torch.tensor(extras, device=cumulative_scores.device, dtype=torch.long))

    if not selected:
        return torch.arange(cumulative_scores.numel(), device=cumulative_scores.device, dtype=torch.long)
    return torch.sort(torch.unique(torch.cat(selected, dim=0))).values


def _specprune_causal_lm_forward(
    self,
    multimodal_embeddings: torch.Tensor,
    multimodal_attention_mask: Optional[torch.Tensor],
    projected_visual_embeddings: torch.Tensor,
    num_visual_patch_tokens: int,
    extra_projected_tokens_count: int,
    prompt_token_count: int,
    compute_logits: bool,
) -> SpecPruneForwardResult:
    language_model = self.language_model
    decoder = language_model.model
    cfg = getattr(self, "_specprune_cfg", None)
    cache = getattr(self, "_specprune_cache", None)
    if cfg is None or cache is None:
        raise ValueError("SpecPrune configuration/cache is missing on the loaded model.")

    batch_size, total_seq_len, _ = multimodal_embeddings.shape
    if batch_size != 1:
        raise ValueError("SpecPrune runtime patch only supports batch size 1 during inference.")

    static_after_layer = int(getattr(cfg, "specprune_static_after_layer", 2))
    if static_after_layer >= len(decoder.layers):
        raise ValueError(f"specprune_static_after_layer={static_after_layer} out of range.")
    dynamic_prune_layers = set(_parse_int_list(getattr(cfg, "specprune_dynamic_prune_layers", "10,15,20,25"), [10, 15, 20, 25]))
    global_layers = set(_parse_int_list(getattr(cfg, "specprune_global_layers", "15,32"), [15, 32]))
    global_layers = {min(max(1, layer), len(decoder.layers)) for layer in global_layers}

    device = multimodal_embeddings.device
    hidden_states = multimodal_embeddings
    current_attention_mask = multimodal_attention_mask
    if current_attention_mask is None:
        current_attention_mask = torch.ones((batch_size, total_seq_len), dtype=torch.long, device=device)
    absolute_positions = torch.arange(total_seq_len, device=device, dtype=torch.long)
    visual_start = 1
    current_visual_tokens = int(num_visual_patch_tokens)
    static_visual_tokens = int(num_visual_patch_tokens)

    per_image_patches = int(self.vision_backbone.get_num_patches())
    num_images = int(self.vision_backbone.get_num_images_in_input())
    if per_image_patches * num_images > current_visual_tokens:
        raise ValueError("SpecPrune visual token layout is inconsistent with vision backbone metadata.")

    action_mode = _get_action_mode(self)
    k_global_total, k_local_total, _ = _get_mode_ks(cfg, action_mode)
    k_global = int(math.ceil(k_global_total / float(max(1, num_images))))
    k_local = int(math.ceil(k_local_total / float(max(1, num_images))))
    min_keep_per_image = int(getattr(cfg, "specprune_min_keep_tokens_per_image", 1))
    static_min_keep = int(math.ceil(per_image_patches * float(getattr(cfg, "specprune_static_min_keep_ratio", 0.30))))
    static_min_keep = max(min_keep_per_image, static_min_keep)

    local_by_layer: List[List[torch.Tensor]] = []
    global_cache_hit = False
    cumulative_scores: Optional[torch.Tensor] = None
    current_to_original = torch.arange(current_visual_tokens, device=device, dtype=torch.long)
    dynamic_layers_used: List[int] = []

    for layer_idx, decoder_layer in enumerate(decoder.layers):
        layer_number = layer_idx + 1
        current_len = hidden_states.shape[1]
        cache_positions = torch.arange(current_len, device=device, dtype=torch.long)
        position_ids = absolute_positions.unsqueeze(0)
        causal_mask = decoder._update_causal_mask(current_attention_mask, hidden_states, cache_positions, 0)

        force_attention = bool(getattr(cfg, "specprune_force_attention_after_prune", True))
        has_pruned_visual = current_visual_tokens < int(num_visual_patch_tokens)
        need_attention = (
            layer_number <= static_after_layer
            or layer_number in global_layers
            or layer_number in dynamic_prune_layers
            or (force_attention and has_pruned_visual)
        )
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
        attention = layer_outputs[1] if need_attention else None

        if attention is not None and layer_number <= static_after_layer and current_visual_tokens == num_visual_patch_tokens:
            scores = _visual_importance_scores(
                cfg,
                attention,
                visual_start=visual_start,
                visual_token_count=current_visual_tokens,
                extra_projected_tokens_count=int(extra_projected_tokens_count),
                prompt_token_count=int(prompt_token_count),
            )
            local_by_layer.append(_topk_by_image(scores, num_images, per_image_patches, k_local))

        if layer_number == static_after_layer and current_visual_tokens == num_visual_patch_tokens:
            static_scores = _visual_importance_scores(
                cfg,
                attention,
                visual_start=visual_start,
                visual_token_count=current_visual_tokens,
                extra_projected_tokens_count=int(extra_projected_tokens_count),
                prompt_token_count=int(prompt_token_count),
            )
            global_by_image, global_cache_hit = _global_indices_for_static(
                cache,
                k_global_per_image=k_global,
                num_images=num_images,
                per_image_patches=per_image_patches,
                device=device,
            )
            dynamic_by_image = _dynamic_indices_by_embedding(
                cache,
                cfg,
                projected_visual_embeddings,
                action_mode=action_mode,
                num_images=num_images,
                per_image_patches=per_image_patches,
                device=device,
            )
            selected_by_image = _union_by_image(
                [global_by_image, dynamic_by_image, *_ensure_layer_list(local_by_layer, num_images, device)],
                num_images=num_images,
                device=device,
            )
            selected_by_image = _ensure_min_static_keep(
                selected_by_image,
                static_scores,
                num_images=num_images,
                per_image_patches=per_image_patches,
                min_keep_per_image=static_min_keep,
            )
            keep_rel = _concat_selected_by_image(selected_by_image)
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
                static_visual_tokens = int(current_visual_tokens)
                cumulative_scores = torch.zeros(current_visual_tokens, device=device, dtype=torch.float32)
            else:
                static_visual_tokens = int(current_visual_tokens)
                cumulative_scores = torch.zeros(current_visual_tokens, device=device, dtype=torch.float32)

        if attention is not None and layer_number in global_layers:
            try:
                global_scores_full = _visual_importance_scores(
                    cfg,
                    attention,
                    visual_start=visual_start,
                    visual_token_count=current_visual_tokens,
                    extra_projected_tokens_count=int(extra_projected_tokens_count),
                    prompt_token_count=int(prompt_token_count),
                )
                full_scores = torch.zeros(num_visual_patch_tokens, device=device, dtype=torch.float32)
                full_scores.index_copy_(0, current_to_original, global_scores_full)
                _update_global_cache(cache, cfg, full_scores, num_images=num_images, per_image_patches=per_image_patches)
            except Exception:
                pass

        if attention is not None and layer_number in dynamic_prune_layers and cumulative_scores is not None:
            current_scores = _visual_importance_scores(
                cfg,
                attention,
                visual_start=visual_start,
                visual_token_count=current_visual_tokens,
                extra_projected_tokens_count=int(extra_projected_tokens_count),
                prompt_token_count=int(prompt_token_count),
            )
            entropy = _attention_entropy(
                attention,
                visual_start=visual_start,
                visual_token_count=current_visual_tokens,
                extra_projected_tokens_count=int(extra_projected_tokens_count),
                prompt_token_count=int(prompt_token_count),
                epsilon=float(getattr(cfg, "specprune_entropy_epsilon", 1e-6)),
            )
            if layer_number not in cache.layer_confidence:
                cache.layer_confidence[layer_number] = 1.0 / (entropy + float(getattr(cfg, "specprune_entropy_epsilon", 1e-6)))
            confidence = float(cache.layer_confidence[layer_number])
            rank_weights = _rank_weight_scores(current_scores, float(getattr(cfg, "specprune_rank_temperature", 0.05)))
            layer_importance = rank_weights * confidence
            beta = float(getattr(cfg, "specprune_beta", 0.2))
            cumulative_scores = (1.0 - beta) * cumulative_scores + beta * layer_importance

            target_keep = int(math.ceil(current_visual_tokens * float(getattr(cfg, "specprune_dynamic_gamma", 0.9))))
            target_keep = max(num_images * min_keep_per_image, min(target_keep, current_visual_tokens))
            if target_keep < current_visual_tokens:
                keep_rel = _select_dynamic_keep_indices(
                    cumulative_scores,
                    current_to_original,
                    num_images=num_images,
                    per_image_patches=per_image_patches,
                    target_keep_total=target_keep,
                    min_keep_per_image=min_keep_per_image,
                )
                hidden_states, current_attention_mask, absolute_positions, current_visual_tokens, keep_rel = _build_pruned_sequence(
                    hidden_states,
                    current_attention_mask,
                    absolute_positions,
                    visual_start=visual_start,
                    visual_token_count=current_visual_tokens,
                    keep_visual_rel_indices=keep_rel,
                )
                current_to_original = current_to_original.index_select(0, keep_rel)
                cumulative_scores = cumulative_scores.index_select(0, keep_rel)
                dynamic_layers_used.append(layer_number)

    hidden_states = decoder.norm(hidden_states)
    logits = _compute_logits_from_hidden_states(language_model, hidden_states) if compute_logits else None
    cache.calls += 1
    _update_embedding_history(cache, cfg, projected_visual_embeddings)
    return SpecPruneForwardResult(
        hidden_states=hidden_states,
        logits=logits,
        current_visual_tokens=int(current_visual_tokens),
        static_visual_tokens=int(static_visual_tokens),
        used_specprune=True,
        action_mode=action_mode,
        dynamic_layers=dynamic_layers_used,
        global_cache_hit=global_cache_hit,
    )


def _ensure_layer_list(
    local_by_layer: Sequence[List[torch.Tensor]],
    num_images: int,
    device: torch.device,
) -> List[List[torch.Tensor]]:
    if local_by_layer:
        return list(local_by_layer)
    return [[torch.empty(0, device=device, dtype=torch.long) for _ in range(num_images)]]


def _forward_language_model_with_optional_specprune(
    self,
    multimodal_embeddings: torch.Tensor,
    multimodal_attention_mask: Optional[torch.Tensor],
    projected_visual_embeddings: torch.Tensor,
    num_visual_patch_tokens: int,
    extra_projected_tokens_count: int,
    prompt_token_count: int,
    compute_logits: bool,
) -> SpecPruneForwardResult:
    if not _should_use_specprune(self, num_visual_patch_tokens, prompt_token_count) or not _supports_language_model(self):
        result = _dense_causal_lm_forward(self, multimodal_embeddings, multimodal_attention_mask, num_visual_patch_tokens, compute_logits)
        _record_metrics(
            self,
            int(num_visual_patch_tokens),
            int(num_visual_patch_tokens),
            int(num_visual_patch_tokens),
            used_specprune=False,
            action_mode=result.action_mode,
            dynamic_layers=[],
            global_cache_hit=False,
        )
        return result

    cache = getattr(self, "_specprune_cache", None)
    cfg = getattr(self, "_specprune_cfg", None)
    global_layers = _parse_int_list(getattr(cfg, "specprune_global_layers", "15,32"), [15, 32]) if cfg is not None else [15, 32]
    warmup_model_calls = int(getattr(cfg, "specprune_warmup_model_calls", 3)) if cfg is not None else 3
    needs_global_cache = bool(global_layers) and cache is not None and cache.global_indices_by_image is None
    needs_dense_warmup = cache is not None and int(getattr(cache, "calls", 0)) < warmup_model_calls
    if needs_global_cache or needs_dense_warmup:
        result = _global_warmup_causal_lm_forward(
            self,
            multimodal_embeddings,
            multimodal_attention_mask,
            projected_visual_embeddings,
            num_visual_patch_tokens,
            extra_projected_tokens_count,
            prompt_token_count,
            compute_logits,
        )
        _record_metrics(
            self,
            int(num_visual_patch_tokens),
            int(num_visual_patch_tokens),
            int(num_visual_patch_tokens),
            used_specprune=False,
            action_mode=result.action_mode,
            dynamic_layers=[],
            global_cache_hit=False,
            fallback=False,
        )
        return result

    try:
        result = _specprune_causal_lm_forward(
            self,
            multimodal_embeddings,
            multimodal_attention_mask,
            projected_visual_embeddings,
            num_visual_patch_tokens,
            extra_projected_tokens_count,
            prompt_token_count,
            compute_logits,
        )
        _record_metrics(
            self,
            int(num_visual_patch_tokens),
            int(result.static_visual_tokens),
            int(result.current_visual_tokens),
            used_specprune=bool(result.used_specprune),
            action_mode=result.action_mode,
            dynamic_layers=result.dynamic_layers,
            global_cache_hit=bool(result.global_cache_hit),
            fallback=False,
        )
        return result
    except Exception as exc:
        if not getattr(self, "_specprune_logged_fallback", False):
            print(f"[SpecPrune] fallback to dense language-model forward: {exc}")
            self._specprune_logged_fallback = True
        result = _dense_causal_lm_forward(self, multimodal_embeddings, multimodal_attention_mask, num_visual_patch_tokens, compute_logits)
        _record_metrics(
            self,
            int(num_visual_patch_tokens),
            int(num_visual_patch_tokens),
            int(num_visual_patch_tokens),
            used_specprune=False,
            action_mode="fallback",
            dynamic_layers=[],
            global_cache_hit=False,
            fallback=True,
        )
        return result


def _extract_action_hidden_states(sequence_hidden_states: torch.Tensor) -> torch.Tensor:
    action_hidden_states = sequence_hidden_states[:, -ACTION_DIM * NUM_ACTIONS_CHUNK - 1 : -1, :]
    expected_tokens = ACTION_DIM * NUM_ACTIONS_CHUNK
    if action_hidden_states.shape[1] != expected_tokens:
        raise RuntimeError(
            f"Expected {expected_tokens} action tokens after SpecPrune pruning, got {action_hidden_states.shape[1]}."
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
            f"Expected {expected_tokens} action logits after SpecPrune pruning, got {action_logits.shape[1]}."
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
        input_embeddings,
        projected_patch_embeddings,
        attention_mask,
    )

    num_visual_patch_tokens = int(self.vision_backbone.get_num_patches()) * int(self.vision_backbone.get_num_images_in_input())
    num_visual_patch_tokens = min(num_visual_patch_tokens, int(projected_patch_embeddings.shape[1]))
    extra_projected_tokens_count = int(projected_patch_embeddings.shape[1]) - int(num_visual_patch_tokens)
    projected_visual_embeddings = projected_patch_embeddings[:, :num_visual_patch_tokens, :]

    language_model_output = _forward_language_model_with_optional_specprune(
        self,
        multimodal_embeddings,
        multimodal_attention_mask,
        projected_visual_embeddings,
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
        projected_visual_embeddings = projected_patch_embeddings[:, :num_visual_patch_tokens, :]

        language_model_output = _forward_language_model_with_optional_specprune(
            self,
            multimodal_embeddings,
            multimodal_attention_mask,
            projected_visual_embeddings,
            num_visual_patch_tokens,
            extra_projected_tokens_count,
            int(num_prompt_tokens),
            compute_logits=False,
        )

        action_hidden_states = _extract_action_hidden_states(language_model_output.hidden_states)
        noise_pred = action_head.predict_noise(action_hidden_states)
        curr_noisy_actions = action_head.noise_scheduler.step(noise_pred, t, curr_noisy_actions).prev_sample

    normalized_actions = curr_noisy_actions.reshape(NUM_ACTIONS_CHUNK, ACTION_DIM)
    normalized_actions = normalized_actions.float().cpu().detach().numpy()
    return normalized_actions, action_hidden_states, None
