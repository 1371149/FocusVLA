"""Runtime monkeypatches that enable ADP pruning in the local openvla-oft tree."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch
from PIL import Image
from transformers import PreTrainedModel

from backend import load_backend_modules


_backend = load_backend_modules()
openvla_utils = _backend.openvla_utils
robot_utils = _backend.robot_utils

from prismatic.vla.constants import ACTION_DIM, IGNORE_INDEX, NUM_ACTIONS_CHUNK, NormalizationType


def apply_runtime_patches() -> None:
    _patch_modeling_prismatic()
    _patch_openvla_utils()


def patch_loaded_model(model: torch.nn.Module) -> None:
    target = model.module if hasattr(model, "module") else model
    _patch_model_class(target.__class__)


def _patch_modeling_prismatic() -> None:
    _patch_model_class(openvla_utils.OpenVLAForActionPrediction)


def _patch_model_class(cls) -> None:
    if getattr(cls, "_adp_runtime_patch_applied", False):
        return

    cls._compute_qk_importance_generic = staticmethod(_compute_qk_importance_generic)
    cls._run_diffusion_prediction = _run_diffusion_prediction
    cls._regression_or_discrete_prediction = _regression_or_discrete_prediction
    cls.predict_action = predict_action
    cls._adp_runtime_patch_applied = True


def _patch_openvla_utils() -> None:
    if getattr(openvla_utils, "_adp_runtime_patch_applied", False):
        return

    openvla_utils.get_vla_action = get_vla_action
    robot_utils.get_vla_action = get_vla_action
    openvla_utils._adp_runtime_patch_applied = True


def _compute_qk_importance_generic(
    language_model: PreTrainedModel,
    language_embeddings: torch.Tensor,
    vision_embeddings: torch.Tensor,
    layer_idx: int,
) -> torch.Tensor:
    is_gpt2 = hasattr(language_model, "transformer") and hasattr(language_model.transformer, "h")
    if is_gpt2:
        layer = language_model.transformer.h[layer_idx]
        hidden_text = layer.ln_1(language_embeddings) if hasattr(layer, "ln_1") else language_embeddings
        hidden_vision = layer.ln_1(vision_embeddings) if hasattr(layer, "ln_1") else vision_embeddings

        c_attn = layer.attn.c_attn
        qkv_text = c_attn(hidden_text)
        qkv_vision = c_attn(hidden_vision)
        hdim = qkv_text.shape[-1] // 3
        q_text = qkv_text[..., :hdim]
        k_vis = qkv_vision[..., hdim : 2 * hdim]

        n_head = language_model.config.n_head
        head_dim = language_model.config.n_embd // n_head
    else:
        layer = language_model.model.layers[layer_idx]
        hidden_text = (
            layer.input_layernorm(language_embeddings)
            if hasattr(layer, "input_layernorm")
            else language_embeddings
        )
        hidden_vision = (
            layer.input_layernorm(vision_embeddings)
            if hasattr(layer, "input_layernorm")
            else vision_embeddings
        )

        attn = layer.self_attn
        q_text = torch.nn.functional.linear(hidden_text, attn.q_proj.weight, getattr(attn.q_proj, "bias", None))
        k_vis = torch.nn.functional.linear(hidden_vision, attn.k_proj.weight, getattr(attn.k_proj, "bias", None))

        n_head = language_model.config.num_attention_heads
        head_dim = language_model.config.hidden_size // n_head

    def split_heads(x: torch.Tensor) -> torch.Tensor:
        bsz, seqlen, hidden = x.shape
        x = x.view(bsz, seqlen, n_head, head_dim)
        return x.permute(0, 2, 1, 3).contiguous()

    q = split_heads(q_text)
    k = split_heads(k_vis)
    scale = 1.0 / (head_dim ** 0.5)
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    importance = scores.mean(dim=(0, 1, 2))
    return importance.detach().cpu()


def _run_diffusion_prediction(
    self,
    input_embeddings,
    all_actions_mask,
    noise,
    action_head,
    projected_patch_embeddings,
    labels,
    attention_mask,
    NUM_PATCHES,
    NUM_PROMPT_TOKENS,
    noisy_action_projector,
):
    orig_projected_patch_embeddings = projected_patch_embeddings.clone()
    curr_noisy_actions = noise

    for t in action_head.noise_scheduler.timesteps:
        timesteps = torch.Tensor([t]).to(labels.device)
        diffusion_timestep_embeddings = (
            action_head.time_encoder(timesteps).to(curr_noisy_actions.dtype).to(curr_noisy_actions.device)
        )
        diffusion_timestep_embeddings = diffusion_timestep_embeddings.unsqueeze(1)
        projected_patch_embeddings = torch.cat((orig_projected_patch_embeddings, diffusion_timestep_embeddings), dim=1)

        bsz = curr_noisy_actions.shape[0]
        orig_shape = curr_noisy_actions.shape
        curr_noisy_actions = curr_noisy_actions.reshape(bsz, -1).unsqueeze(-1)
        noisy_action_features = noisy_action_projector(curr_noisy_actions)
        curr_noisy_actions = curr_noisy_actions.reshape(orig_shape)

        input_embeddings = self._replace_input_embeddings(
            input_embeddings.clone(), all_actions_mask, noisy_action_features
        )

        multimodal_embeddings, multimodal_attention_mask = self._build_multimodal_attention(
            input_embeddings, projected_patch_embeddings, attention_mask
        )

        language_model_output = self.language_model(
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

        last_hidden_states = language_model_output.hidden_states[-1]
        actions_hidden_states = last_hidden_states[
            :,
            - ACTION_DIM * NUM_ACTIONS_CHUNK - 1 : -1,
            :,
        ]

        noise_pred = action_head.predict_noise(actions_hidden_states)
        curr_noisy_actions = action_head.noise_scheduler.step(noise_pred, t, curr_noisy_actions).prev_sample

    curr_noisy_actions = curr_noisy_actions.reshape(NUM_ACTIONS_CHUNK, ACTION_DIM)
    return curr_noisy_actions.float().cpu().detach().numpy(), actions_hidden_states


def _regression_or_discrete_prediction(
    self,
    input_embeddings,
    all_actions_mask,
    projected_patch_embeddings,
    attention_mask,
    labels,
    NUM_PATCHES,
    NUM_PROMPT_TOKENS,
    action_head=None,
    past_key_values=None,
):
    all_actions_mask = all_actions_mask.unsqueeze(-1)
    input_embeddings = input_embeddings * ~all_actions_mask

    multimodal_embeddings, multimodal_attention_mask = self._build_multimodal_attention(
        input_embeddings, projected_patch_embeddings, attention_mask
    )

    language_model_output = self.language_model(
        input_ids=None,
        attention_mask=multimodal_attention_mask,
        position_ids=None,
        past_key_values=past_key_values,
        inputs_embeds=multimodal_embeddings,
        labels=None,
        use_cache=True,
        output_attentions=True,
        output_hidden_states=True,
        return_dict=True,
    )

    last_hidden_states = language_model_output.hidden_states[-1]
    actions_hidden_states = last_hidden_states[
        :,
        - ACTION_DIM * NUM_ACTIONS_CHUNK - 1 : -1,
        :,
    ]
    last_caches = {
        "past_key_values": language_model_output.past_key_values,
        "attentions": language_model_output.attentions,
    }

    if action_head is not None:
        normalized_actions = action_head.predict_action(actions_hidden_states)
        normalized_actions = normalized_actions.reshape(NUM_ACTIONS_CHUNK, ACTION_DIM)
        normalized_actions = normalized_actions.float().cpu().detach().numpy()
    else:
        predicted_action_token_ids = (
            language_model_output.logits[
                :,
                NUM_PATCHES + NUM_PROMPT_TOKENS : NUM_PATCHES + NUM_PROMPT_TOKENS + ACTION_DIM * NUM_ACTIONS_CHUNK,
            ]
            .argmax(dim=2)
            .cpu()
            .numpy()
        )
        discretized_actions = self.vocab_size - predicted_action_token_ids
        discretized_actions = np.clip(discretized_actions - 1, a_min=0, a_max=self.bin_centers.shape[0] - 1)
        normalized_actions = self.bin_centers[discretized_actions]
        normalized_actions = normalized_actions.reshape(NUM_ACTIONS_CHUNK, ACTION_DIM)

    return normalized_actions, actions_hidden_states, last_caches


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
    if not torch.all(input_ids[:, -1] == 29871):
        input_ids = torch.cat(
            (input_ids, torch.unsqueeze(torch.Tensor([29871]).long(), dim=0).to(input_ids.device)), dim=1
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
    qk_keep_config = kwargs.get("qk_keep_config", None)

    if qk_keep_config and bool(qk_keep_config.get("qk_keep_enabled", False)):
        try:
            qk_layer = int(qk_keep_config.get("qk_layer", 0))
            qk_keep_ratio = float(qk_keep_config.get("qk_keep_ratio", 0.75))
            qk_keep_split = qk_keep_config.get("qk_keep_split", None)

            bsz, vis_tokens, _ = projected_patch_embeddings.shape
            per_image_patches = int(self.vision_backbone.get_num_patches())
            num_images = int(self.vision_backbone.get_num_images_in_input())
            if vis_tokens > 0 and qk_keep_ratio < 1.0:
                if qk_layer > 0:
                    probe_input_embeddings = input_embeddings * ~all_actions_mask.unsqueeze(-1)
                    probe_multimodal_embeddings, probe_multimodal_attention_mask = self._build_multimodal_attention(
                        probe_input_embeddings, projected_patch_embeddings, attention_mask
                    )
                    with torch.no_grad():
                        probe_out = self.language_model(
                            input_ids=None,
                            attention_mask=probe_multimodal_attention_mask,
                            position_ids=None,
                            past_key_values=None,
                            inputs_embeds=probe_multimodal_embeddings,
                            labels=None,
                            output_hidden_states=True,
                            use_cache=False,
                        )
                    seq_h = probe_out.hidden_states[qk_layer]
                    vis_count = projected_patch_embeddings.shape[1]
                    bos_h = seq_h[:, :1, :]
                    vis_h = seq_h[:, 1 : 1 + vis_count, :]
                    tail_h = seq_h[:, 1 + vis_count :, :]
                    tail_mask = (~all_actions_mask[:, 1:]).unsqueeze(-1).expand_as(tail_h)
                    tail_kept = tail_h[tail_mask].reshape(bos_h.shape[0], -1, bos_h.shape[-1])
                    text_emb = torch.cat([bos_h, tail_kept], dim=1)
                    vis_emb = vis_h
                else:
                    text_emb = language_embeddings
                    vis_emb = projected_patch_embeddings

                importance = self._compute_qk_importance_generic(self.language_model, text_emb, vis_emb, qk_layer)
                keep_total = max(1, int(round(vis_tokens * qk_keep_ratio)))

                if (num_images > 1) and (per_image_patches * num_images == vis_tokens):
                    weights = qk_keep_split if isinstance(qk_keep_split, list) else None
                    if weights and len(weights) == 2 and num_images > 2:
                        main_w, wrist_w = weights
                        rest = num_images - 1
                        weights = [main_w] + [wrist_w / rest] * rest
                    if not weights or len(weights) != num_images:
                        weights = [1.0] * num_images
                    total_weight = sum(max(0.0, float(w)) for w in weights)
                    if total_weight <= 0:
                        weights = [1.0] * num_images
                        total_weight = float(num_images)
                    norm = [float(w) / total_weight for w in weights]
                    per_image_keep = [max(1, int(round(keep_total * p))) for p in norm]
                    diff = keep_total - sum(per_image_keep)
                    order = sorted(range(num_images), key=lambda i: norm[i], reverse=True)
                    idx = 0
                    while diff != 0 and idx < len(order):
                        image_idx = order[idx]
                        if diff > 0:
                            per_image_keep[image_idx] += 1
                            diff -= 1
                        else:
                            if per_image_keep[image_idx] > 1:
                                per_image_keep[image_idx] -= 1
                                diff += 1
                        idx = (idx + 1) % len(order)
                    kept = []
                    for image_idx in range(num_images):
                        start = image_idx * per_image_patches
                        end = start + per_image_patches
                        keep_i = min(per_image_keep[image_idx], per_image_patches)
                        _, idx_i = torch.topk(importance[start:end], k=keep_i, largest=True, sorted=False)
                        kept.append((idx_i + start).to(projected_patch_embeddings.device))
                    keep_idx = torch.sort(torch.cat(kept, dim=0).long())[0]
                else:
                    _, idx = torch.topk(importance, k=keep_total, largest=True, sorted=True)
                    keep_idx = idx.to(projected_patch_embeddings.device).long()

                projected_patch_embeddings = projected_patch_embeddings.index_select(dim=1, index=keep_idx)
        except Exception as exc:
            import traceback

            traceback.print_exc()
            print(f"[ADP][QK_KEEP] exception at layer={qk_layer}: {exc}")

    num_visual_patches = int(projected_patch_embeddings.shape[1])

    use_proprio = proprio_projector is not None and proprio is not None
    if use_proprio:
        proprio = torch.Tensor(proprio).to(projected_patch_embeddings.device, dtype=projected_patch_embeddings.dtype)
        projected_patch_embeddings = self._process_proprio_features(
            projected_patch_embeddings, proprio, proprio_projector
        )

    use_diffusion = noisy_action_projector is not None and hasattr(action_head, "noise_scheduler")
    num_patches = num_visual_patches + (1 if use_proprio else 0) + (1 if use_diffusion else 0)

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
            num_patches,
            num_prompt_tokens,
            noisy_action_projector,
        )
    else:
        normalized_actions, actions_hidden_states, last_caches = self._regression_or_discrete_prediction(
            input_embeddings,
            all_actions_mask,
            projected_patch_embeddings,
            attention_mask,
            labels,
            num_patches,
            num_prompt_tokens,
            action_head,
            past_key_values,
        )

    actions = self._unnormalize_actions(normalized_actions, unnorm_key)
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
) -> list[np.ndarray]:
    with torch.inference_mode():
        all_images = [obs["full_image"]]
        if cfg.num_images_in_input > 1:
            all_images.extend([obs[k] for k in obs.keys() if "wrist" in k])

        all_images = openvla_utils.prepare_images_for_vla(all_images, cfg)
        primary_image = all_images.pop(0)
        prompt = f"In: What action should the robot take to {task_label.lower()}?\nOut:"
        inputs = processor(prompt, primary_image).to(openvla_utils.DEVICE, dtype=torch.bfloat16)

        if all_images:
            all_wrist_inputs = [
                processor(prompt, image_wrist).to(openvla_utils.DEVICE, dtype=torch.bfloat16)
                for image_wrist in all_images
            ]
            primary_pixel_values = inputs["pixel_values"]
            all_wrist_pixel_values = [wrist_inputs["pixel_values"] for wrist_inputs in all_wrist_inputs]
            inputs["pixel_values"] = torch.cat([primary_pixel_values] + all_wrist_pixel_values, dim=1)

        proprio = None
        if cfg.use_proprio:
            proprio = obs["state"]
            proprio_norm_stats = vla.norm_stats[cfg.unnorm_key]["proprio"]
            obs["state"] = openvla_utils.normalize_proprio(proprio, proprio_norm_stats)
            proprio = obs["state"]

        qk_keep_config = None
        try:
            if getattr(cfg, "qk_keep_enabled", False):
                layer = int(getattr(cfg, "qk_layer", 0))
                ratio = float(getattr(cfg, "qk_keep_ratio", 0.75))
                split = getattr(cfg, "qk_keep_split", None)
                debug = bool(getattr(cfg, "qk_debug", False))
                log_topk = int(getattr(cfg, "qk_log_topk", 16))
                if isinstance(split, str):
                    try:
                        split = [float(x) for x in split.split(",")]
                    except Exception:
                        split = None
                qk_keep_config = {
                    "qk_keep_enabled": True,
                    "qk_layer": layer,
                    "qk_keep_ratio": ratio,
                    "qk_keep_split": split,
                    "qk_debug": debug,
                    "qk_log_topk": log_topk,
                }
                if bool(getattr(cfg, "random_experiment", False)):
                    qk_keep_config.update(
                        {
                            "random_experiment": True,
                            "random_keep_ratio": float(getattr(cfg, "random_keep_ratio", 0.3)),
                            "random_seed": getattr(cfg, "random_seed", None),
                        }
                    )
        except Exception:
            qk_keep_config = None

        if action_head is None:
            action, _ = vla.predict_action(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                attention_mask=inputs["attention_mask"],
                unnorm_key=cfg.unnorm_key,
                qk_keep_config=qk_keep_config,
            )
        else:
            action, _ = vla.predict_action(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                attention_mask=inputs["attention_mask"],
                unnorm_key=cfg.unnorm_key,
                proprio=proprio,
                proprio_projector=proprio_projector,
                noisy_action_projector=noisy_action_projector,
                action_head=action_head,
                use_film=use_film,
                qk_keep_config=qk_keep_config,
            )

    return [action[i] for i in range(len(action))]
