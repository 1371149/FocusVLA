"""Runtime loader that patches the local environment toward the original VLA-Cache flow."""

from __future__ import annotations

import importlib
import importlib.util
import json
import os
import sys
import time
import types
from contextlib import contextmanager
from pathlib import Path
from typing import Any


VLACACHE_DIR = Path(__file__).resolve().parent
OPENVLA_OFT_ROOT = VLACACHE_DIR.parents[1]
ROBOT_ROOT = OPENVLA_OFT_ROOT.parent

DEFAULT_ORIGINAL_OPENVLA_OFT_ROOT = ROBOT_ROOT / "vla-cache-main" / "src" / "openvla-oft"
DEFAULT_LLAMA_PATCH_PATH = OPENVLA_OFT_ROOT / "vla-scripts" / "modeling_llama.py"

ORIGINAL_OPENVLA_OFT_ROOT = Path(
    os.environ.get("VLA_CACHE_ORIGINAL_OPENVLA_OFT_ROOT", str(DEFAULT_ORIGINAL_OPENVLA_OFT_ROOT))
).expanduser().resolve()
LOCAL_LLAMA_PATCH_PATH = Path(
    os.environ.get("VLA_CACHE_LLAMA_PATCH_PATH", str(DEFAULT_LLAMA_PATCH_PATH))
).expanduser().resolve()

_PATCHED = False


def _load_module_from_path(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, str(path))
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _purge_modules(prefixes) -> None:
    for name in list(sys.modules.keys()):
        if any(name == prefix or name.startswith(prefix + ".") for prefix in prefixes):
            sys.modules.pop(name, None)


def _install_optional_stubs() -> None:
    if "seaborn" not in sys.modules:
        sys.modules["seaborn"] = types.ModuleType("seaborn")
    if "skimage.util" not in sys.modules:
        import numpy as np

        skimage_module = sys.modules.get("skimage")
        if skimage_module is None:
            skimage_module = types.ModuleType("skimage")
            sys.modules["skimage"] = skimage_module

        util_module = types.ModuleType("skimage.util")

        def view_as_blocks(arr_in, block_shape):
            arr = np.asarray(arr_in)
            if len(block_shape) != arr.ndim:
                raise ValueError("`block_shape` must have the same length as input dimensions.")
            if any(dim % block != 0 for dim, block in zip(arr.shape, block_shape)):
                raise ValueError("`block_shape` must divide input shape exactly.")
            new_shape = tuple(dim // block for dim, block in zip(arr.shape, block_shape)) + tuple(block_shape)
            new_strides = tuple(stride * block for stride, block in zip(arr.strides, block_shape)) + arr.strides
            return np.lib.stride_tricks.as_strided(arr, shape=new_shape, strides=new_strides)

        util_module.view_as_blocks = view_as_blocks
        sys.modules["skimage.util"] = util_module
        skimage_module.util = util_module


def _validate_runtime_paths() -> None:
    if not ORIGINAL_OPENVLA_OFT_ROOT.is_dir():
        raise FileNotFoundError(f"VLA-Cache official openvla-oft root not found: {ORIGINAL_OPENVLA_OFT_ROOT}")
    if not LOCAL_LLAMA_PATCH_PATH.is_file():
        raise FileNotFoundError(f"VLA-Cache Llama patch file not found: {LOCAL_LLAMA_PATCH_PATH}")


@contextmanager
def _preserve_cuda_visible_devices():
    """Ignore the official script's hardcoded CUDA_VISIBLE_DEVICES assignment."""

    environ_cls = os.environ.__class__
    original_setitem = environ_cls.__setitem__
    original_value = os.environ.get("CUDA_VISIBLE_DEVICES")

    def guarded_setitem(self, key, value):
        if key == "CUDA_VISIBLE_DEVICES":
            if original_value is None:
                return None
            value = original_value
        return original_setitem(self, key, value)

    environ_cls.__setitem__ = guarded_setitem
    try:
        yield
    finally:
        environ_cls.__setitem__ = original_setitem
        if original_value is None:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = original_value


def _patch_dynamic_cache_update():
    import transformers
    import torch

    DynamicCache = transformers.cache_utils.DynamicCache
    if hasattr(DynamicCache, "_vla_cache_original_update"):
        return

    DynamicCache._vla_cache_original_update = DynamicCache.update

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        if cache_kwargs is not None:
            cache_position = cache_kwargs.get("cache_position", None)
        else:
            cache_position = torch.arange(key_states.size(-2), device=key_states.device)

        if not hasattr(self, "cache_position"):
            self.cache_position = None

        if layer_idx == 0:
            if getattr(self, "_seen_tokens", 0) == 0:
                self._seen_tokens += key_states.shape[-2]
            elif key_states.shape[-2] == 1:
                self._seen_tokens += 1

            if self.cache_position is None and cache_position is not None:
                self.cache_position = cache_position.unsqueeze(0)
            elif cache_position is not None and cache_position.size(-1) == 1:
                self.cache_position = torch.cat([self.cache_position, cache_position.unsqueeze(0)], dim=-1)

        if key_states is not None:
            assert cache_position.size(-1) <= self._seen_tokens, (
                f"cache_position: {cache_position.size(-1)} seen_tokens: {self._seen_tokens}"
            )
            if len(self.key_cache) <= layer_idx:
                for _ in range(len(self.key_cache), layer_idx):
                    self.key_cache.append([])
                    self.value_cache.append([])
                self.key_cache.append(key_states)
                self.value_cache.append(value_states)
            elif len(self.key_cache[layer_idx]) == 0:
                self.key_cache[layer_idx] = key_states
                self.value_cache[layer_idx] = value_states
            elif cache_position.size(-1) <= self._seen_tokens and cache_position.size(-1) != 1:
                self.key_cache[layer_idx].index_copy_(2, cache_position, key_states)
                self.value_cache[layer_idx].index_copy_(2, cache_position, value_states)
            elif cache_position.size(-1) == 1:
                self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim=-2)
                self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states], dim=-2)
            else:
                print(f"cache_position: {cache_position.size(-1)} seen_tokens: {self._seen_tokens}")

        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    DynamicCache.update = update
    transformers.DynamicCache = DynamicCache


def _build_patched_llama_forward(llama_module):
    import torch

    logger = llama_module.logger
    DynamicCache = llama_module.DynamicCache
    StaticCache = llama_module.StaticCache
    Cache = llama_module.Cache
    BaseModelOutputWithPast = llama_module.BaseModelOutputWithPast

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        cache_position=None,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
            )

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once("`use_cache=True` is incompatible with gradient_checkpointing. Setting `use_cache=False`.")
            use_cache = False

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        past_seen_tokens = 0
        if use_cache:
            if not isinstance(past_key_values, StaticCache):
                past_key_values = DynamicCache.from_legacy_cache(past_key_values)
                past_seen_tokens = (
                    past_key_values.get_seq_length()
                    if getattr(self.config, "proportion_attn_var", None) is None
                    else 0
                )

        if cache_position is None:
            if isinstance(past_key_values, StaticCache):
                raise ValueError("cache_position is a required argument when using StaticCache.")
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = self._update_causal_mask(attention_mask, inputs_embeds, cache_position, past_seen_tokens)
        hidden_states = inputs_embeds

        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None

        if not hasattr(self, "pruning_loc"):
            pruning_layer = int(getattr(self.config, "vla_cache_attention_layer", 15))
            self.pruning_loc = {max(0, min(len(self.layers) - 1, pruning_layer))}
        last_reusable_patches = None

        for layer_idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if (
                getattr(self.config, "proportion_attn_var", None) is not None
                and hidden_states.size(1) != 1
                and layer_idx in self.pruning_loc
            ):
                reusable_patches = getattr(self.config, "reusable_patches", None)
                if reusable_patches is not None and len(reusable_patches) > 0:
                    reusable_patches = reusable_patches.to(hidden_states.device)
                    proportion = self.config.proportion_attn_var[layer_idx]
                    top_k = max(1, int(float(proportion) * len(reusable_patches)))
                    selected_reusable_patches = reusable_patches[:top_k]

                    if last_reusable_patches is not None and last_reusable_patches.size(0) > selected_reusable_patches.size(0):
                        selected_reusable_patches = last_reusable_patches

                    seq_length = hidden_states.size(1)
                    if last_reusable_patches is None:
                        newly_reused_patches = selected_reusable_patches
                    else:
                        newly_reused_patches = selected_reusable_patches[
                            ~torch.isin(selected_reusable_patches, last_reusable_patches)
                        ]

                    if newly_reused_patches.numel() > 0:
                        newly_reused_patches = newly_reused_patches[torch.isin(newly_reused_patches, cache_position)]
                        if newly_reused_patches.numel() > 0:
                            mask = ~torch.isin(cache_position, newly_reused_patches)
                            new_cache_position = cache_position[mask]
                            new_cache_position, _ = new_cache_position.sort()

                            hidden_states = hidden_states[:, mask, :]
                            if causal_mask is not None:
                                causal_mask = causal_mask[:, :, mask, :]
                            position_ids = new_cache_position.unsqueeze(0)
                            cache_position = new_cache_position

                    last_reusable_patches = selected_reusable_patches

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    causal_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                    cache_position,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if output_attentions:
            all_self_attns += (cache_position.detach().cpu(),)

        next_cache = None
        if use_cache:
            next_cache = next_decoder_cache.to_legacy_cache() if isinstance(next_decoder_cache, Cache) else next_decoder_cache

        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )

    return forward


def _build_patched_llama_attention_forward(llama_module):
    import math
    import torch
    import torch.nn.functional as F

    repeat_kv = llama_module.repeat_kv
    apply_rotary_pos_emb = llama_module.apply_rotary_pos_emb

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask=None,
        position_ids=None,
        past_key_value=None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position=None,
        **kwargs,
    ):
        bsz, q_len, _ = hidden_states.size()

        if self.config.pretraining_tp > 1:
            key_value_slicing = (self.num_key_value_heads * self.head_dim) // self.config.pretraining_tp
            query_slices = self.q_proj.weight.split(
                (self.num_heads * self.head_dim) // self.config.pretraining_tp, dim=0
            )
            key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
            value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)

            query_states = [F.linear(hidden_states, query_slices[i]) for i in range(self.config.pretraining_tp)]
            query_states = torch.cat(query_states, dim=-1)
            key_states = [F.linear(hidden_states, key_slices[i]) for i in range(self.config.pretraining_tp)]
            key_states = torch.cat(key_states, dim=-1)
            value_states = [F.linear(hidden_states, value_slices[i]) for i in range(self.config.pretraining_tp)]
            value_states = torch.cat(value_states, dim=-1)
        else:
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        past_key_value = getattr(self, "past_key_value", past_key_value)
        cos, sin = self.rotary_emb(value_states, position_ids)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
        if attention_mask is not None:
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask

        attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = torch.nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        if self.config.pretraining_tp > 1:
            attn_output = attn_output.split(self.hidden_size // self.config.pretraining_tp, dim=2)
            o_proj_slices = self.o_proj.weight.split(self.hidden_size // self.config.pretraining_tp, dim=1)
            attn_output = sum([F.linear(attn_output[i], o_proj_slices[i]) for i in range(self.config.pretraining_tp)])
        else:
            attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value

    return forward


def patch_transformers_with_local_llama() -> None:
    import transformers
    import transformers.models.llama.modeling_llama as installed_llama

    llama_module = _load_module_from_path("vlacache_local_modeling_llama", LOCAL_LLAMA_PATCH_PATH)

    for name in [
        "LlamaAttention",
        "LlamaFlashAttention2",
        "LlamaSdpaAttention",
        "LLAMA_ATTENTION_CLASSES",
        "LlamaDecoderLayer",
        "LlamaModel",
        "LlamaForCausalLM",
    ]:
        setattr(installed_llama, name, getattr(llama_module, name))

    _patch_dynamic_cache_update()

    patched_forward = _build_patched_llama_forward(llama_module)
    patched_attention_forward = _build_patched_llama_attention_forward(llama_module)
    installed_llama.LlamaAttention.forward = patched_attention_forward
    llama_module.LlamaAttention.forward = patched_attention_forward
    installed_llama.LlamaModel.forward = patched_forward
    llama_module.LlamaModel.forward = patched_forward


def load_original_run_module():
    global _PATCHED
    if not _PATCHED:
        _validate_runtime_paths()
        _install_optional_stubs()
        patch_transformers_with_local_llama()
        _PATCHED = True

    _purge_modules(["experiments", "prismatic"])
    root_str = str(ORIGINAL_OPENVLA_OFT_ROOT)
    if root_str in sys.path:
        sys.path.remove(root_str)
    sys.path.insert(0, root_str)

    with _preserve_cuda_visible_devices():
        return importlib.import_module("experiments.robot.libero.run_libero_eval")


def build_original_cfg(wrapper_cfg: Any, run_module) -> Any:
    cfg = run_module.GenerateConfig()
    for key, value in vars(wrapper_cfg).items():
        if hasattr(cfg, key):
            setattr(cfg, key, value)

    cfg.custom_log_filepath = getattr(wrapper_cfg, "custom_log_filepath", None)
    cfg.eval_with_white_patch = bool(getattr(wrapper_cfg, "eval_with_white_patch", False))
    cfg.eval_with_mug = bool(getattr(wrapper_cfg, "eval_with_mug", False))
    cfg.eval_mug_cache_dir = getattr(wrapper_cfg, "eval_mug_cache_dir", str(VLACACHE_DIR / "generated_bddl_mug"))
    cfg.eval_mug_generated_data_dir = getattr(wrapper_cfg, "eval_mug_generated_data_dir", "")
    cfg.eval_mug_pose_catalog_path = getattr(wrapper_cfg, "eval_mug_pose_catalog_path", "")
    cfg.eval_mug_placement_presets_path = getattr(wrapper_cfg, "eval_mug_placement_presets_path", "")
    cfg.eval_mug_max_init_retries = int(getattr(wrapper_cfg, "eval_mug_max_init_retries", 0))
    cfg.eval_save_initial_global_images = bool(getattr(wrapper_cfg, "eval_save_initial_global_images", False))
    cfg.eval_initial_global_image_dir = getattr(wrapper_cfg, "eval_initial_global_image_dir", "")
    cfg.vla_cache_attention_layer = int(getattr(wrapper_cfg, "vla_cache_attention_layer", 15))
    cfg.max_tasks = getattr(wrapper_cfg, "max_tasks", None)
    return cfg


def patch_original_run_module(run_module, wrapper_cfg: Any) -> None:
    import traceback

    from initial_global_image import resolve_initial_global_image_dir, save_initial_global_image
    from mug_eval import describe_eval_mug_spec, get_libero_env_with_optional_mug

    original_get_libero_env = run_module.get_libero_env
    original_setup_logging = run_module.setup_logging
    original_prepare_observation = run_module.prepare_observation
    original_run_episode = run_module.run_episode
    openvla_utils_module = sys.modules.get("experiments.robot.openvla_utils")
    robot_utils_module = sys.modules.get("experiments.robot.robot_utils")
    video_saving_disabled_logged = False
    initial_image_context = {"active": None}

    def _sync_cuda_if_needed() -> None:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()

    class _ModelPredictionTracker:
        def __init__(self):
            self.call_count = 0
            self.total_time_sec = 0.0

        @property
        def avg_time_sec(self) -> float:
            if self.call_count == 0:
                return 0.0
            return self.total_time_sec / self.call_count

    def _patch_model_predict_action(model, tracker: _ModelPredictionTracker):
        if not hasattr(model, "predict_action"):
            return None

        original_predict_action = model.predict_action

        def timed_predict_action(*args, **kwargs):
            _sync_cuda_if_needed()
            start_ts = time.perf_counter()
            try:
                return original_predict_action(*args, **kwargs)
            finally:
                _sync_cuda_if_needed()
                tracker.total_time_sec += time.perf_counter() - start_ts
                tracker.call_count += 1

        model.predict_action = timed_predict_action
        return original_predict_action

    def _restore_model_predict_action(model, original_predict_action) -> None:
        if original_predict_action is not None:
            model.predict_action = original_predict_action

    def _patch_model_vla_cache_signature(model) -> None:
        target = model.module if hasattr(model, "module") else model
        if getattr(target, "_vlacache_predict_signature_patched", False):
            return
        if not hasattr(target, "predict_action") or not hasattr(target, "_regression_or_discrete_prediction"):
            return

        import numpy as np
        import torch
        from prismatic.vla.constants import ACTION_DIM, IGNORE_INDEX, NUM_ACTIONS_CHUNK

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
            actions_hidden_states = last_hidden_states[:, -ACTION_DIM * NUM_ACTIONS_CHUNK - 1 : -1, :]
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
                normalized_actions = self.bin_centers[discretized_actions].reshape(NUM_ACTIONS_CHUNK, ACTION_DIM)

            return normalized_actions, actions_hidden_states, last_caches

        def predict_action(
            self,
            input_ids=None,
            unnorm_key=None,
            proprio=None,
            proprio_projector=None,
            action_head=None,
            noisy_action_projector=None,
            use_film: bool = False,
            past_key_values=None,
            **kwargs,
        ):
            if not torch.all(input_ids[:, -1] == 29871):
                input_ids = torch.cat(
                    (input_ids, torch.unsqueeze(torch.tensor([29871]).long(), dim=0).to(input_ids.device)),
                    dim=1,
                )

            pixel_values = kwargs["pixel_values"]
            attention_mask = kwargs["attention_mask"]

            labels = input_ids.clone()
            labels[:] = IGNORE_INDEX
            NUM_PROMPT_TOKENS = input_ids.shape[-1] - 1

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
                projected_patch_embeddings = self._process_proprio_features(
                    projected_patch_embeddings, proprio, proprio_projector
                )

            use_diffusion = noisy_action_projector is not None and hasattr(action_head, "noise_scheduler")
            NUM_PATCHES = self.vision_backbone.get_num_patches() * self.vision_backbone.get_num_images_in_input()
            if use_proprio:
                NUM_PATCHES += 1
            if use_diffusion:
                NUM_PATCHES += 1

            last_caches = None
            if use_diffusion:
                noise = torch.randn(
                    size=(1, NUM_ACTIONS_CHUNK, ACTION_DIM),
                    device=input_embeddings.device,
                    dtype=input_embeddings.dtype,
                )
                normalized_actions, actions_hidden_states = self._run_diffusion_prediction(
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
                )
            else:
                normalized_actions, actions_hidden_states, last_caches = self._regression_or_discrete_prediction(
                    input_embeddings,
                    all_actions_mask,
                    projected_patch_embeddings,
                    attention_mask,
                    labels,
                    NUM_PATCHES,
                    NUM_PROMPT_TOKENS,
                    action_head,
                    past_key_values,
                )

            actions = self._unnormalize_actions(normalized_actions, unnorm_key)
            return actions, actions_hidden_states, last_caches

        target._vlacache_original_predict_action = target.predict_action
        target._vlacache_original_regression_or_discrete_prediction = target._regression_or_discrete_prediction
        target._regression_or_discrete_prediction = types.MethodType(_regression_or_discrete_prediction, target)
        target.predict_action = types.MethodType(predict_action, target)
        target._vlacache_predict_signature_patched = True

    def setup_logging(cfg):
        custom_log_filepath = str(getattr(cfg, "custom_log_filepath", "") or "").strip()
        if not custom_log_filepath:
            return original_setup_logging(cfg)

        run_id = f"EVAL-{cfg.task_suite_name}-{cfg.model_family}-{run_module.DATE_TIME}"
        if cfg.run_id_note is not None:
            run_id += f"--{cfg.run_id_note}"
        log_path = Path(custom_log_filepath).expanduser().resolve()
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_file = open(log_path, "w")
        run_module.logger.info(f"Logging to local log file: {log_path}")

        if getattr(cfg, "use_wandb", False):
            run_module.wandb.init(entity=cfg.wandb_entity, project=cfg.wandb_project, name=run_id)

        return log_file, str(log_path), run_id

    def get_libero_env(task, model_family, resolution=256):
        if bool(getattr(wrapper_cfg, "eval_with_mug", False)):
            return get_libero_env_with_optional_mug(task, model_family, resolution, wrapper_cfg)
        return original_get_libero_env(task, model_family, resolution=resolution)

    def _add_top_left_white_patch(image):
        import numpy as np

        if image.ndim != 3 or image.shape[-1] != 3:
            return image
        height, width = int(image.shape[0]), int(image.shape[1])
        patch_size = int(round(np.sqrt(float(height * width) * 0.01)))
        patch_size = max(1, min(patch_size, height, width))
        patched = image.copy()
        patched[:patch_size, :patch_size, :] = np.array([255, 255, 255], dtype=np.uint8)
        return patched

    def prepare_observation(obs, resize_size):
        observation, img = original_prepare_observation(obs, resize_size)
        context = initial_image_context.get("active")
        if context is not None and not context.get("saved", False):
            save_path = save_initial_global_image(
                context["cfg"],
                img,
                context["task_name"],
                context["episode_idx"],
                context["total_episode_idx"],
                default_note="vla_cache",
            )
            context["saved"] = True
            if save_path is not None:
                run_module.log_message(f"Saved initial global image: {save_path}", context.get("log_file"))
        if bool(getattr(wrapper_cfg, "eval_with_white_patch", False)):
            observation["full_image"] = _add_top_left_white_patch(observation["full_image"])
            if "wrist_image" in observation:
                observation["wrist_image"] = _add_top_left_white_patch(observation["wrist_image"])
            img = observation["full_image"]
        return observation, img

    def save_rollout_video(rollout_images, idx, success, task_description, log_file=None, view="primary"):
        nonlocal video_saving_disabled_logged
        del rollout_images, idx, success, task_description, view
        if not video_saving_disabled_logged:
            message = "[VLA-Cache] Rollout video saving disabled."
            print(message)
            if log_file is not None:
                log_file.write(f"{message}\n")
            video_saving_disabled_logged = True
        return None

    def run_episode(
        cfg,
        env,
        task_description,
        model,
        resize_size,
        processor=None,
        action_head=None,
        proprio_projector=None,
        noisy_action_projector=None,
        initial_state=None,
        log_file=None,
    ):
        task_counts = getattr(cfg, "_initial_global_image_task_counts", {})
        episode_idx = int(task_counts.get(task_description, 0))
        task_counts[task_description] = episode_idx + 1
        cfg._initial_global_image_task_counts = task_counts
        total_episode_idx = int(getattr(cfg, "_initial_global_image_total_count", 0)) + 1
        cfg._initial_global_image_total_count = total_episode_idx
        initial_image_context["active"] = {
            "cfg": cfg,
            "task_name": task_description,
            "episode_idx": episode_idx,
            "total_episode_idx": total_episode_idx,
            "log_file": log_file,
            "saved": False,
        }
        prediction_tracker = _ModelPredictionTracker()
        original_predict_action = _patch_model_predict_action(model, prediction_tracker)
        try:
            success, replay_images, replay_images_wrist, episode_metrics = original_run_episode(
                cfg,
                env,
                task_description,
                model,
                resize_size,
                processor,
                action_head,
                proprio_projector,
                noisy_action_projector,
                initial_state,
                log_file,
            )
        finally:
            initial_image_context["active"] = None
            _restore_model_predict_action(model, original_predict_action)

        episode_metrics = dict(episode_metrics or {})
        episode_metrics["episode_time"] = prediction_tracker.total_time_sec
        episode_metrics["episode_step"] = prediction_tracker.call_count
        episode_metrics["prediction_calls"] = prediction_tracker.call_count
        episode_metrics["prediction_time_total_sec"] = prediction_tracker.total_time_sec
        episode_metrics["prediction_time_avg_sec"] = prediction_tracker.avg_time_sec

        if hasattr(cfg, "_prediction_episode_totals_sec"):
            cfg._prediction_episode_totals_sec.append(float(prediction_tracker.total_time_sec))
        if hasattr(cfg, "_prediction_episode_avgs_sec"):
            cfg._prediction_episode_avgs_sec.append(float(prediction_tracker.avg_time_sec))
        if hasattr(cfg, "_prediction_episode_call_counts"):
            cfg._prediction_episode_call_counts.append(int(prediction_tracker.call_count))

        run_module.log_message(f"Episode model prediction calls: {prediction_tracker.call_count}", log_file)
        run_module.log_message(
            f"Episode total model prediction time (seconds): {prediction_tracker.total_time_sec:.6f}",
            log_file,
        )
        run_module.log_message(
            f"Episode average model prediction time (seconds): {prediction_tracker.avg_time_sec:.6f}",
            log_file,
        )
        return success, replay_images, replay_images_wrist, episode_metrics

    def eval_libero(cfg) -> float:
        run_module.validate_config(cfg)
        run_module.set_seed_everywhere(cfg.seed)

        model, action_head, proprio_projector, noisy_action_projector, processor = run_module.initialize_model(cfg)
        _patch_model_vla_cache_signature(model)
        resize_size = run_module.get_image_resize_size(cfg)
        log_file, local_log_filepath, _ = run_module.setup_logging(cfg)

        benchmark_dict = run_module.benchmark.get_benchmark_dict()
        task_suite = benchmark_dict[cfg.task_suite_name]()
        num_tasks = task_suite.n_tasks
        max_tasks = getattr(cfg, "max_tasks", None)
        if max_tasks is not None:
            num_tasks = max(0, min(num_tasks, int(max_tasks)))

        cfg._prediction_episode_totals_sec = []
        cfg._prediction_episode_avgs_sec = []
        cfg._prediction_episode_call_counts = []
        cfg._initial_global_image_task_counts = {}
        cfg._initial_global_image_total_count = 0

        run_module.log_message(f"Task suite: {cfg.task_suite_name}", log_file)
        run_module.log_message(f"Eval with white patch: {bool(getattr(cfg, 'eval_with_white_patch', False))}", log_file)
        run_module.log_message(f"Eval with mug: {bool(getattr(cfg, 'eval_with_mug', False))}", log_file)
        run_module.log_message(
            f"Save initial global images: {bool(getattr(cfg, 'eval_save_initial_global_images', False))}",
            log_file,
        )
        if bool(getattr(cfg, "eval_save_initial_global_images", False)):
            run_module.log_message(
                f"Initial global image dir: {resolve_initial_global_image_dir(cfg, 'vla_cache')}",
                log_file,
            )
        if bool(getattr(cfg, "eval_with_mug", False)):
            run_module.log_message(f"Eval mug spec: {describe_eval_mug_spec(cfg.task_suite_name, cfg)}", log_file)

        total_episodes, total_successes = 0, 0
        for task_id in run_module.tqdm.tqdm(range(num_tasks)):
            total_episodes, total_successes = run_module.run_task(
                cfg,
                task_suite,
                task_id,
                model,
                resize_size,
                processor,
                action_head,
                proprio_projector,
                noisy_action_projector,
                total_episodes,
                total_successes,
                log_file,
            )

        final_success_rate = float(total_successes) / float(total_episodes) if total_episodes > 0 else 0.0
        avg_episode_prediction_total_sec = (
            float(run_module.np.mean(cfg._prediction_episode_totals_sec)) if cfg._prediction_episode_totals_sec else 0.0
        )
        avg_episode_prediction_time_sec = (
            float(run_module.np.mean(cfg._prediction_episode_avgs_sec)) if cfg._prediction_episode_avgs_sec else 0.0
        )
        avg_episode_prediction_calls = (
            float(run_module.np.mean(cfg._prediction_episode_call_counts))
            if cfg._prediction_episode_call_counts
            else 0.0
        )

        run_module.log_message("Final results:", log_file)
        run_module.log_message(f"Total episodes: {total_episodes}", log_file)
        run_module.log_message(f"Total successes: {total_successes}", log_file)
        run_module.log_message(
            f"Overall success rate: {final_success_rate:.4f} ({final_success_rate * 100:.1f}%)",
            log_file,
        )
        run_module.log_message(
            f"Average total model prediction time per episode (seconds): {avg_episode_prediction_total_sec:.6f}",
            log_file,
        )
        run_module.log_message(
            f"Average single model prediction time per episode (seconds): {avg_episode_prediction_time_sec:.6f}",
            log_file,
        )
        run_module.log_message(
            f"Average model prediction calls per episode: {avg_episode_prediction_calls:.6f}",
            log_file,
        )

        if cfg.use_wandb:
            run_module.wandb.log(
                {
                    "success_rate/total": final_success_rate,
                    "num_episodes/total": total_episodes,
                }
            )
            run_module.wandb.save(local_log_filepath)

        if log_file:
            log_file.close()

        for attr_name in (
            "_prediction_episode_totals_sec",
            "_prediction_episode_avgs_sec",
            "_prediction_episode_call_counts",
            "_initial_global_image_task_counts",
            "_initial_global_image_total_count",
        ):
            if hasattr(cfg, attr_name):
                delattr(cfg, attr_name)

        return final_success_rate

    run_module.get_libero_env = get_libero_env
    run_module.setup_logging = setup_logging
    run_module.prepare_observation = prepare_observation
    run_module.save_rollout_video = save_rollout_video
    run_module.run_episode = run_episode
    run_module.eval_libero = eval_libero

    if openvla_utils_module is not None:
        original_get_vla_action = openvla_utils_module.get_vla_action

        def get_vla_action(*args, **kwargs):
            try:
                return original_get_vla_action(*args, **kwargs)
            except Exception:
                traceback.print_exc()
                raise

        def update_auto_map(pretrained_checkpoint: str) -> None:
            config_path = Path(pretrained_checkpoint) / "config.json"
            if not config_path.is_file():
                return
            try:
                config = json.loads(config_path.read_text(encoding="utf-8"))
            except Exception as exc:
                print(f"[VLA-Cache] Skipping non-mutating auto_map check: {exc}")
                return

            expected_auto_map = {
                "AutoConfig": "configuration_prismatic.OpenVLAConfig",
                "AutoModelForVision2Seq": "modeling_prismatic.OpenVLAForActionPrediction",
            }
            if config.get("auto_map") != expected_auto_map:
                print(
                    "[VLA-Cache] Checkpoint config.json auto_map differs from the official VLA-Cache "
                    "expectation; leaving it unchanged to keep this wrapper read-only outside VLACache."
                )

        def check_model_logic_mismatch(pretrained_checkpoint: str) -> None:
            if not os.path.isdir(pretrained_checkpoint):
                return
            missing = [
                filename
                for filename in ("modeling_prismatic.py", "configuration_prismatic.py")
                if not (Path(pretrained_checkpoint) / filename).is_file()
            ]
            if missing:
                print(
                    "[VLA-Cache] Checkpoint is missing remote-code files "
                    f"{missing}; leaving checkpoint unchanged because this wrapper may only write under VLACache."
                )

        openvla_utils_module.update_auto_map = update_auto_map
        openvla_utils_module.get_vla_action = get_vla_action
        if robot_utils_module is not None:
            robot_utils_module.get_vla_action = get_vla_action
        openvla_utils_module.check_model_logic_mismatch = check_model_logic_mismatch


def run_original_eval(wrapper_cfg: Any) -> float:
    run_module = load_original_run_module()
    patch_original_run_module(run_module, wrapper_cfg)
    cfg = build_original_cfg(wrapper_cfg, run_module)
    eval_fn = getattr(run_module.eval_libero, "__wrapped__", run_module.eval_libero)
    return eval_fn(cfg)
