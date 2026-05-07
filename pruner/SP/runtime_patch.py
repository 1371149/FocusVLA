"""Runtime monkeypatches that transplant SP-VLA ideas into openvla-oft.

This module keeps all SP-specific code inside `pruner/SP` and patches the
loaded model at runtime, rather than editing the main OpenVLA-OFT source tree.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from types import MethodType
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

try:
    import cv2
except Exception:  # pragma: no cover - optional dependency
    cv2 = None

from backend import load_backend_modules
from debug import maybe_save_first_pruned_image


_backend = load_backend_modules()
openvla_utils = _backend.openvla_utils
robot_utils = _backend.robot_utils

from prismatic.vla.constants import ACTION_DIM, IGNORE_INDEX, NUM_ACTIONS_CHUNK


@dataclass
class SPMetrics:
    model_generate_calls: int = 0
    model_skip_calls: int = 0
    executed_generate_actions: int = 0
    executed_fit_actions: int = 0
    prune_events: int = 0
    tokens_before: int = 0
    tokens_after: int = 0
    last_prediction_source: str = "generate"
    last_prune_rate: float = 0.0
    max_prune_rate: float = 0.0

    def snapshot(self) -> Dict[str, Any]:
        weighted_prune_rate = 0.0
        if self.tokens_before > 0:
            weighted_prune_rate = 1.0 - (self.tokens_after / float(self.tokens_before))
        return {
            "model_generate_calls": self.model_generate_calls,
            "model_skip_calls": self.model_skip_calls,
            "executed_generate_actions": self.executed_generate_actions,
            "executed_fit_actions": self.executed_fit_actions,
            "prune_events": self.prune_events,
            "tokens_before": self.tokens_before,
            "tokens_after": self.tokens_after,
            "weighted_prune_rate": weighted_prune_rate,
            "last_prediction_source": self.last_prediction_source,
            "last_prune_rate": self.last_prune_rate,
            "max_prune_rate": self.max_prune_rate,
        }


def apply_runtime_patches() -> None:
    if getattr(openvla_utils, "_sp_runtime_patch_applied", False):
        return

    openvla_utils.get_vla_action = get_vla_action
    robot_utils.get_vla_action = get_vla_action
    openvla_utils._sp_runtime_patch_applied = True


def patch_loaded_model(model: torch.nn.Module, cfg: Optional[Any] = None) -> None:
    target = model.module if hasattr(model, "module") else model
    freshly_patched = False

    if not hasattr(target, "_sp_metrics"):
        target._sp_metrics = SPMetrics()
    if not hasattr(target, "_sp_episode_metrics"):
        target._sp_episode_metrics = SPMetrics()
    if cfg is not None:
        target._sp_cfg = cfg
    if not hasattr(target, "_sp_debug_saved_tags"):
        target._sp_debug_saved_tags = set()
    if not hasattr(target, "_sp_current_task_id"):
        target._sp_current_task_id = -1
    if not hasattr(target, "_sp_current_episode_idx"):
        target._sp_current_episode_idx = -1

    if not hasattr(target, "_sp_action_history"):
        target._sp_action_history = []
        target._sp_action_source_history = []
        target._sp_last_prediction_source = "generate"

    if not hasattr(target, "_sp_original_predict_action"):
        target._sp_original_predict_action = target.predict_action
        target.predict_action = MethodType(predict_action, target)
        target.sp_reset_episode = MethodType(sp_reset_episode, target)
        target.sp_record_executed_action = MethodType(sp_record_executed_action, target)
        target.sp_get_metrics = MethodType(sp_get_metrics, target)
        freshly_patched = True

    _patch_vision_backbone(target)
    if freshly_patched:
        target.sp_reset_episode()


def get_runtime_metrics(model: torch.nn.Module) -> Dict[str, Any]:
    target = model.module if hasattr(model, "module") else model
    if not hasattr(target, "_sp_metrics"):
        return SPMetrics().snapshot()
    return target._sp_metrics.snapshot()


def get_episode_runtime_metrics(model: torch.nn.Module) -> Dict[str, Any]:
    target = model.module if hasattr(model, "module") else model
    if not hasattr(target, "_sp_episode_metrics"):
        return SPMetrics().snapshot()
    return target._sp_episode_metrics.snapshot()


def sp_reset_episode(self) -> None:
    self._sp_action_history = []
    self._sp_action_source_history = []
    self._sp_last_prediction_source = "generate"
    self._sp_debug_saved_tags = set()
    self._sp_episode_metrics = SPMetrics()
    _clear_backbone_context(self)


def sp_record_executed_action(self, action: Sequence[float], source: Optional[str] = None) -> None:
    cfg = getattr(self, "_sp_cfg", None)
    if cfg is None:
        return

    source = source or getattr(self, "_sp_last_prediction_source", "generate")
    action_np = np.array(action, dtype=np.float32, copy=True)

    self._sp_action_history.append(action_np)
    self._sp_action_source_history.append(source)

    max_len = int(getattr(cfg, "sp_history_size", 6))
    if len(self._sp_action_history) > max_len:
        self._sp_action_history.pop(0)
        self._sp_action_source_history.pop(0)

    if source == "fit":
        self._sp_metrics.executed_fit_actions += 1
        self._sp_episode_metrics.executed_fit_actions += 1
    else:
        self._sp_metrics.executed_generate_actions += 1
        self._sp_episode_metrics.executed_generate_actions += 1


def sp_get_metrics(self) -> Dict[str, Any]:
    return self._sp_metrics.snapshot()


def predict_action(
    self,
    input_ids: Optional[torch.LongTensor] = None,
    unnorm_key: Optional[str] = None,
    proprio=None,
    proprio_projector=None,
    action_head=None,
    noisy_action_projector=None,
    use_film: bool = False,
    past_key_values=None,
    **kwargs: str,
) -> np.ndarray:
    cfg = getattr(self, "_sp_cfg", None)
    raw_images = kwargs.pop("raw_images", None)
    raw_image = kwargs.pop("raw_image", None)

    if raw_images is None and raw_image is not None:
        raw_images = [raw_image]
    raw_images = [_to_numpy_image(image) for image in (raw_images or [])]

    if not torch.all(input_ids[:, -1] == 29871):
        input_ids = torch.cat(
            (input_ids, torch.unsqueeze(torch.Tensor([29871]).long(), dim=0).to(input_ids.device)), dim=1
        )

    if cfg is not None and _should_skip_model_call(self, cfg):
        actions = _predict_skip_chunk(self, cfg)
        self._sp_last_prediction_source = "fit"
        self._sp_metrics.model_skip_calls += 1
        self._sp_metrics.last_prediction_source = "fit"
        self._sp_episode_metrics.model_skip_calls += 1
        self._sp_episode_metrics.last_prediction_source = "fit"
        return actions, None

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

    _set_backbone_context(self, raw_images, cfg, use_film)
    try:
        projected_patch_embeddings = self._process_vision_features(pixel_values, language_embeddings, use_film)
    finally:
        _clear_backbone_context(self)

    use_proprio = proprio_projector is not None and proprio is not None
    if use_proprio:
        proprio = torch.Tensor(proprio).to(projected_patch_embeddings.device, dtype=projected_patch_embeddings.dtype)
        projected_patch_embeddings = self._process_proprio_features(projected_patch_embeddings, proprio, proprio_projector)

    use_diffusion = noisy_action_projector is not None and hasattr(action_head, "noise_scheduler")
    num_patch_tokens = projected_patch_embeddings.shape[1]
    if use_diffusion:
        num_patch_tokens += 1

    if use_diffusion:
        noise = torch.randn(
            size=(1, NUM_ACTIONS_CHUNK, ACTION_DIM), device=input_embeddings.device, dtype=input_embeddings.dtype
        )
        normalized_actions, actions_hidden_states = self._run_diffusion_prediction(
            input_embeddings,
            all_actions_mask,
            noise,
            action_head,
            projected_patch_embeddings,
            labels,
            attention_mask,
            num_patch_tokens,
            num_prompt_tokens,
            noisy_action_projector,
        )
    else:
        regression_result = self._regression_or_discrete_prediction(
            input_embeddings,
            all_actions_mask,
            projected_patch_embeddings,
            attention_mask,
            labels,
            num_patch_tokens,
            num_prompt_tokens,
            action_head,
        )
        if len(regression_result) == 3:
            normalized_actions, actions_hidden_states, last_caches = regression_result
        else:
            normalized_actions, actions_hidden_states = regression_result
            last_caches = None

    actions = self._unnormalize_actions(normalized_actions, unnorm_key)

    self._sp_last_prediction_source = "generate"
    self._sp_metrics.model_generate_calls += 1
    self._sp_metrics.last_prediction_source = "generate"
    self._sp_episode_metrics.model_generate_calls += 1
    self._sp_episode_metrics.last_prediction_source = "generate"
    if past_key_values is None:
        return actions, actions_hidden_states
    return actions, actions_hidden_states, last_caches


def get_vla_action(
    cfg: Any,
    vla: torch.nn.Module,
    processor: Any,
    obs: Dict[str, Any],
    task_label: str,
    action_head: Optional[torch.nn.Module] = None,
    proprio_projector: Optional[torch.nn.Module] = None,
    noisy_action_projector: Optional[torch.nn.Module] = None,
    use_film: bool = False,
) -> List[np.ndarray]:
    with torch.inference_mode():
        if hasattr(vla, "predict_action") and hasattr(cfg, "sp_enable_token_pruning"):
            patch_loaded_model(vla, cfg)
        target = vla.module if hasattr(vla, "module") else vla
        target._sp_current_task_label = task_label

        all_images = [obs["full_image"]]
        if cfg.num_images_in_input > 1:
            all_images.extend([obs[k] for k in obs.keys() if "wrist" in k])

        all_images = openvla_utils.prepare_images_for_vla(all_images, cfg)
        raw_images = [np.array(image, copy=True) for image in all_images]
        primary_image = all_images.pop(0)

        prompt = f"In: What action should the robot take to {task_label.lower()}?\nOut:"
        inputs = processor(prompt, primary_image).to(openvla_utils.DEVICE, dtype=torch.bfloat16)

        if all_images:
            all_wrist_inputs = [processor(prompt, image_wrist).to(openvla_utils.DEVICE, dtype=torch.bfloat16) for image_wrist in all_images]
            primary_pixel_values = inputs["pixel_values"]
            all_wrist_pixel_values = [wrist_inputs["pixel_values"] for wrist_inputs in all_wrist_inputs]
            inputs["pixel_values"] = torch.cat([primary_pixel_values] + all_wrist_pixel_values, dim=1)

        proprio = None
        if cfg.use_proprio:
            proprio = obs["state"]
            proprio_norm_stats = vla.norm_stats[cfg.unnorm_key]["proprio"]
            obs["state"] = openvla_utils.normalize_proprio(proprio, proprio_norm_stats)
            proprio = obs["state"]

        if action_head is None:
            action, _ = vla.predict_action(
                **inputs,
                unnorm_key=cfg.unnorm_key,
                do_sample=False,
                raw_images=raw_images,
            )
        else:
            action, _ = vla.predict_action(
                **inputs,
                unnorm_key=cfg.unnorm_key,
                do_sample=False,
                proprio=proprio,
                proprio_projector=proprio_projector,
                noisy_action_projector=noisy_action_projector,
                action_head=action_head,
                use_film=use_film,
                raw_images=raw_images,
            )

    return [action[i] for i in range(len(action))]


def _patch_vision_backbone(model: torch.nn.Module) -> None:
    vision_backbone = model.vision_backbone
    vision_backbone._sp_owner = model

    if hasattr(vision_backbone, "vision_backbone"):
        if not hasattr(vision_backbone, "_sp_original_forward"):
            vision_backbone._sp_original_forward = vision_backbone.forward
            vision_backbone.forward = MethodType(_film_vision_forward, vision_backbone)
    else:
        if not hasattr(vision_backbone, "_sp_original_forward"):
            vision_backbone._sp_original_forward = vision_backbone.forward
            vision_backbone.forward = MethodType(_vision_forward, vision_backbone)


def _vision_forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
    context = getattr(self, "_sp_context", None)
    if not _context_enabled(context) or not self.use_fused_vision_backbone:
        return self._sp_original_forward(pixel_values)
    return _forward_impl(self, pixel_values, average_language_embedding=None)


def _film_vision_forward(self, pixel_values: torch.Tensor, language_embeddings: torch.Tensor) -> torch.Tensor:
    context = getattr(self, "_sp_context", None)
    if not _context_enabled(context) or not self.vision_backbone.use_fused_vision_backbone:
        return self._sp_original_forward(pixel_values, language_embeddings)
    average_language_embedding = language_embeddings.mean(dim=1)
    return _forward_impl(self, pixel_values, average_language_embedding=average_language_embedding)


def _forward_impl(owner: torch.nn.Module, pixel_values: torch.Tensor, average_language_embedding: Optional[torch.Tensor]) -> torch.Tensor:
    core = owner.vision_backbone if hasattr(owner, "vision_backbone") else owner
    num_images = owner.get_num_images_in_input() if hasattr(owner, "get_num_images_in_input") else core.get_num_images_in_input()
    raw_images = list((getattr(owner, "_sp_context", {}) or {}).get("raw_images", []))

    if num_images == 1:
        img, img_fused = torch.split(pixel_values, [3, 3], dim=1)
        combined = _process_one_image(
            owner,
            core,
            img,
            img_fused,
            raw_images[0] if raw_images else None,
            average_language_embedding,
            image_tag="full",
        )
        return combined

    images = torch.split(pixel_values, [6] * num_images, dim=1)
    outputs = []
    for index, image in enumerate(images):
        img_regular, img_fused = torch.split(image, [3, 3], dim=1)
        raw_image = raw_images[index] if index < len(raw_images) else None
        if index == 0:
            image_tag = "full"
        elif index == 1:
            image_tag = "wrist"
        else:
            image_tag = f"wrist_{index}"
        outputs.append(
            _process_one_image(
                owner,
                core,
                img_regular,
                img_fused,
                raw_image,
                average_language_embedding,
                image_tag=image_tag,
            )
        )
    return torch.cat(outputs, dim=1)


def _process_one_image(
    owner: torch.nn.Module,
    core: torch.nn.Module,
    img_regular: torch.Tensor,
    img_fused: torch.Tensor,
    raw_image: Optional[np.ndarray],
    average_language_embedding: Optional[torch.Tensor],
    image_tag: str,
) -> torch.Tensor:
    patches = _run_featurizer(core.featurizer, img_regular, average_language_embedding)

    captured_outputs: Dict[str, torch.Tensor] = {}
    q_handle, k_handle = _register_attention_hooks(core.fused_featurizer, captured_outputs)
    try:
        patches_fused = _run_featurizer(core.fused_featurizer, img_fused, average_language_embedding)
    finally:
        q_handle.remove()
        k_handle.remove()

    important_idx = _select_token_indices(owner, patches, captured_outputs, raw_image, image_tag=image_tag)
    if important_idx is not None:
        idx_tensor = torch.as_tensor(important_idx, dtype=torch.long, device=patches.device)
        patches = patches.index_select(1, idx_tensor)
        patches_fused = patches_fused.index_select(1, idx_tensor)
        _record_pruning(owner, before=int(captured_outputs.get("num_tokens", patches.shape[1])), after=int(len(important_idx)))

    return torch.cat([patches, patches_fused], dim=2)


def _run_featurizer(featurizer: torch.nn.Module, image: torch.Tensor, average_language_embedding: Optional[torch.Tensor]) -> torch.Tensor:
    if average_language_embedding is None:
        return featurizer(image)
    return featurizer(image, average_language_embedding)


def _register_attention_hooks(featurizer: torch.nn.Module, captured_outputs: Dict[str, torch.Tensor]):
    attn_module = _resolve_attention_module(featurizer)

    def save_tensor(name: str):
        def hook(_module, _inputs, output):
            captured_outputs[name] = output

        return hook

    q_handle = attn_module.q_norm.register_forward_hook(save_tensor("siglip_q"))
    k_handle = attn_module.k_norm.register_forward_hook(save_tensor("siglip_k"))
    return q_handle, k_handle


def _resolve_attention_module(featurizer: torch.nn.Module):
    block = featurizer.blocks[-1]
    while hasattr(block, "block"):
        block = block.block
    return block.attn


def _select_token_indices(
    owner: torch.nn.Module,
    patches: torch.Tensor,
    captured_outputs: Dict[str, torch.Tensor],
    raw_image: Optional[np.ndarray],
    image_tag: str,
) -> Optional[np.ndarray]:
    context = getattr(owner, "_sp_context", None)
    if not _context_enabled(context):
        return None

    siglip_q = captured_outputs.get("siglip_q")
    siglip_k = captured_outputs.get("siglip_k")
    if siglip_q is None or siglip_k is None:
        return None

    num_tokens = int(patches.shape[1])
    captured_outputs["num_tokens"] = num_tokens
    cfg = context["cfg"]

    head_dim = siglip_k.shape[-1]
    siglip_attn = torch.matmul(siglip_q, siglip_k.transpose(-2, -1)) / math.sqrt(head_dim)
    siglip_attn = torch.softmax(siglip_attn, dim=-1)
    siglip_attn = siglip_attn.mean(dim=1)
    siglip_attn = siglip_attn.mean(dim=1)
    attention_score, siglip_idx = torch.sort(siglip_attn, descending=True)

    fixed_keep_ratio = getattr(cfg, "sp_fixed_keep_ratio", None)
    if fixed_keep_ratio is not None:
        keep_tokens = int(math.ceil(num_tokens * float(fixed_keep_ratio)))
    else:
        z_trans = float(context.get("z_trans", 0.0))
        threshold = _compute_attention_threshold(cfg, z_trans)
        cum_scores = attention_score.cumsum(dim=1)
        keep_tokens = int((cum_scores < threshold).sum(dim=1).item() + 1)

    max_prune_rate = getattr(cfg, "sp_max_prune_rate", None)
    if max_prune_rate is not None:
        min_keep_tokens = int(math.ceil(num_tokens * (1.0 - float(max_prune_rate))))
        keep_tokens = max(keep_tokens, min_keep_tokens)

    keep_tokens = max(keep_tokens, int(getattr(cfg, "sp_keep_min_tokens", 1)))
    keep_tokens = min(keep_tokens, num_tokens)

    important_idx = siglip_idx[:, :keep_tokens].squeeze(0).detach().cpu().numpy()
    if raw_image is not None:
        edge_idx = _edge_token_indices(raw_image, num_tokens, cfg)
        if edge_idx.size > 0:
            important_idx = np.union1d(important_idx, edge_idx)

    important_idx = np.asarray(important_idx, dtype=np.int64)
    important_idx = important_idx[(important_idx >= 0) & (important_idx < num_tokens)]
    if important_idx.size == 0:
        important_idx = np.arange(min(num_tokens, int(getattr(cfg, "sp_keep_min_tokens", 1))), dtype=np.int64)
    if raw_image is not None:
        maybe_save_first_pruned_image(owner, cfg, raw_image, important_idx, num_tokens, image_tag=image_tag)
    return np.sort(important_idx)


def _edge_token_indices(raw_image: np.ndarray, num_tokens: int, cfg: Any) -> np.ndarray:
    if cv2 is None:
        return np.empty((0,), dtype=np.int64)

    image = _to_numpy_image(raw_image)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, threshold1=int(cfg.sp_edge_threshold1), threshold2=int(cfg.sp_edge_threshold2))

    grid_size = max(int(round(math.sqrt(num_tokens))), 1)
    patch_size = getattr(cfg, "sp_edge_patch_size", None) or max(image.shape[0] // grid_size, 1)
    height, width = edges.shape
    n_rows = max(height // patch_size, 1)
    n_cols = max(width // patch_size, 1)

    edge_indices: List[int] = []
    for row in range(n_rows):
        for col in range(n_cols):
            patch = edges[row * patch_size : (row + 1) * patch_size, col * patch_size : (col + 1) * patch_size]
            if patch.size > 0 and np.any(patch):
                index = row * n_cols + col
                if index < num_tokens:
                    edge_indices.append(index)
    return np.asarray(edge_indices, dtype=np.int64)


def _compute_attention_threshold(cfg: Any, z_trans: float) -> float:
    z_threshold = float(cfg.sp_z_thre_prune)
    if z_trans < z_threshold:
        return 1.0

    clipped_z = min(float(z_trans), 1.0)
    denom = max(1.0 - z_threshold, 1e-6)
    threshold = 1.0 - (clipped_z - z_threshold) * (float(cfg.sp_z_min_prune) / denom)
    return float(np.clip(threshold, 1e-6, 1.0))


def _record_pruning(owner: torch.nn.Module, before: int, after: int) -> None:
    model = getattr(owner, "_sp_owner", None)
    if model is None:
        return
    metric_objects = [
        getattr(model, "_sp_metrics", None),
        getattr(model, "_sp_episode_metrics", None),
    ]
    prune_rate = 0.0 if before <= 0 else 1.0 - (after / float(before))
    for metrics in metric_objects:
        if metrics is None:
            continue
        metrics.prune_events += 1
        metrics.tokens_before += int(before)
        metrics.tokens_after += int(after)
        metrics.last_prune_rate = prune_rate
        metrics.max_prune_rate = max(metrics.max_prune_rate, prune_rate)


def _should_skip_model_call(model: torch.nn.Module, cfg: Any) -> bool:
    if not bool(getattr(cfg, "sp_enable_step_skipping", False)):
        return False

    history = getattr(model, "_sp_action_history", [])
    source_history = getattr(model, "_sp_action_source_history", [])
    history_size = int(getattr(cfg, "sp_history_size", 6))
    skip_window = int(getattr(cfg, "sp_skip_window", 1))

    if len(history) < max(history_size, skip_window):
        return False

    recent_sources = source_history[-history_size:]
    generate_count = sum(source == "generate" for source in recent_sources)
    if (generate_count / float(history_size)) < float(getattr(cfg, "sp_min_generate_fraction", 2.0 / 3.0)):
        return False

    recent_actions = history[-skip_window:]
    for action in recent_actions:
        max_xy = max(abs(float(action[0])), abs(float(action[1])), 1e-4)
        max_z = abs(float(action[2]))
        if not (max_z / max_xy < float(cfg.sp_z_xy_rate_skip) and max_z < float(cfg.sp_z_max_skip)):
            return False

    return True


def _predict_skip_chunk(model: torch.nn.Module, cfg: Any) -> np.ndarray:
    temp_history = [np.array(action, dtype=np.float32, copy=True) for action in model._sp_action_history]
    chunk: List[np.ndarray] = []
    for _ in range(NUM_ACTIONS_CHUNK):
        next_action = _fit_next_action(temp_history)
        chunk.append(next_action)
        temp_history.append(next_action)
        if len(temp_history) > int(getattr(cfg, "sp_history_size", 6)):
            temp_history.pop(0)
    return np.stack(chunk, axis=0)


def _fit_next_action(history: List[np.ndarray]) -> np.ndarray:
    total_actions = np.asarray(history, dtype=np.float32)
    xyz_actions = total_actions[:, :3]

    x = np.arange(len(xyz_actions), dtype=np.float32).reshape(-1, 1)
    x = np.hstack([x, np.ones_like(x)])
    lambda_reg = 1e-4
    beta = np.linalg.solve(x.T @ x + lambda_reg * np.eye(2, dtype=np.float32), x.T @ xyz_actions)

    next_step = len(xyz_actions)
    next_action_xyz = beta[0] * next_step + beta[1]
    next_action_xyz = np.clip(next_action_xyz, -0.9, 0.9)

    if len(xyz_actions) >= 2:
        delta_max = np.max(np.abs(np.diff(xyz_actions, axis=0)), axis=0)
        last_xyz = xyz_actions[-1]
        delta_pred = next_action_xyz - last_xyz
        next_action_xyz = last_xyz + np.clip(delta_pred, -delta_max, delta_max)

    tail = total_actions[-1, 3:] if total_actions.shape[1] > 3 else np.empty((0,), dtype=np.float32)
    return np.concatenate([next_action_xyz.astype(np.float32), tail.astype(np.float32)], axis=0)


def _set_backbone_context(model: torch.nn.Module, raw_images: List[np.ndarray], cfg: Optional[Any], use_film: bool) -> None:
    vision_backbone = model.vision_backbone
    context = {
        "enabled": bool(cfg is not None and getattr(cfg, "sp_enable_token_pruning", False)),
        "cfg": cfg,
        "raw_images": raw_images,
        "z_trans": _last_z_translation(model),
        "use_film": use_film,
    }
    vision_backbone._sp_context = context


def _clear_backbone_context(model: torch.nn.Module) -> None:
    if hasattr(model, "vision_backbone"):
        model.vision_backbone._sp_context = None


def _last_z_translation(model: torch.nn.Module) -> float:
    history = getattr(model, "_sp_action_history", [])
    if not history:
        return 0.0
    return abs(float(history[-1][2]))


def _to_numpy_image(image: Any) -> np.ndarray:
    if image is None:
        raise ValueError("Expected image, got None")
    if hasattr(image, "convert"):
        image = image.convert("RGB")
        return np.array(image, copy=True)
    array = np.asarray(image)
    if array.ndim == 2:
        array = np.stack([array] * 3, axis=-1)
    if array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    return array


def _context_enabled(context: Optional[Dict[str, Any]]) -> bool:
    return bool(context is not None and context.get("enabled", False) and context.get("cfg") is not None)
