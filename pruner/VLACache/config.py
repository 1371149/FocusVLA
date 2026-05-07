"""Configuration helpers for VLA-Cache-style evaluation on openvla-oft."""

import json
import os
import sys
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional, Union

CURRENT_DIR = Path(__file__).resolve().parent
PRUNER_DIR = CURRENT_DIR.parent
if str(PRUNER_DIR) not in sys.path:
    sys.path.insert(0, str(PRUNER_DIR))

from path_utils import normalize_cfg_paths


VLA_CACHE_DIR = Path(__file__).resolve().parent
OPENVLA_OFT_ROOT = VLA_CACHE_DIR.parents[1]
ROBOT_ROOT = OPENVLA_OFT_ROOT.parent


class TaskSuite(str, Enum):
    LIBERO_SPATIAL = "libero_spatial"
    LIBERO_OBJECT = "libero_object"
    LIBERO_GOAL = "libero_goal"
    LIBERO_10 = "libero_10"
    LIBERO_90 = "libero_90"


TASK_MAX_STEPS = {
    TaskSuite.LIBERO_SPATIAL.value: 220,
    TaskSuite.LIBERO_OBJECT.value: 280,
    TaskSuite.LIBERO_GOAL.value: 300,
    TaskSuite.LIBERO_10.value: 520,
    TaskSuite.LIBERO_90.value: 400,
}


@dataclass
class VLACacheGenerateConfig:
    # fmt: off

    #################################################################################################################
    # VLA-Cache switches
    #################################################################################################################
    use_vla_cache: bool = True
    vla_cache_static_top_k: int = 150
    vla_cache_similarity_threshold: float = 0.996
    vla_cache_attention_top_k: int = 100
    vla_cache_attention_layer: int = 15
    vla_cache_apply_weighted_growth: bool = True
    vla_cache_growth_factor: float = 0.55

    #################################################################################################################
    # Model-specific parameters
    #################################################################################################################
    model_family: str = "openvla"
    pretrained_checkpoint: Union[str, Path] = str(ROBOT_ROOT / "model" / "openvla-7b-oft-finetuned-libero-spatial")
    lora_weight_path: Optional[str] = None

    use_l1_regression: bool = True
    use_diffusion: bool = False
    num_diffusion_steps_train: int = 50
    num_diffusion_steps_inference: int = 50
    use_film: bool = False
    num_images_in_input: int = 2
    use_proprio: bool = True

    center_crop: bool = True
    num_open_loop_steps: int = 8
    lora_rank: int = 32
    unnorm_key: Union[str, Path] = ""

    load_in_8bit: bool = False
    load_in_4bit: bool = False

    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    task_suite_name: str = TaskSuite.LIBERO_SPATIAL.value
    num_steps_wait: int = 10
    num_trials_per_task: int = 50
    max_tasks: Optional[int] = None
    initial_states_path: str = "DEFAULT"
    env_img_res: int = 256
    eval_with_white_patch: bool = False
    eval_with_mug: bool = False
    eval_mug_cache_dir: str = str(VLA_CACHE_DIR / "generated_bddl_mug")
    eval_mug_generated_data_dir: str = ""
    eval_mug_pose_catalog_path: str = str(VLA_CACHE_DIR.parent / "mug_pose_stats" / "mug_pose_catalog.json")
    eval_mug_placement_presets_path: str = str(PRUNER_DIR / "mug_placement_presets.json")
    eval_mug_max_init_retries: int = 0
    eval_save_initial_global_images: bool = False
    eval_initial_global_image_dir: str = ""

    #################################################################################################################
    # Logging / utils
    #################################################################################################################
    run_id_note: Optional[str] = "vla_cache"
    local_log_dir: str = str(VLA_CACHE_DIR / "logs")
    custom_log_filepath: Optional[str] = str(VLA_CACHE_DIR / "logs" / "vla_cache_eval.txt")

    use_wandb: bool = False
    wandb_entity: str = "your-wandb-entity"
    wandb_project: str = "your-wandb-project"

    seed: int = 7

    #################################################################################################################
    # Local config bootstrap
    #################################################################################################################
    vla_cache_config_json: Optional[str] = str(VLA_CACHE_DIR / "configs" / "vla_cache_config.json")

    # fmt: on


def validate_config(cfg: VLACacheGenerateConfig) -> None:
    assert cfg.pretrained_checkpoint is not None, "pretrained_checkpoint must not be None!"

    if "image_aug" in str(cfg.pretrained_checkpoint):
        assert cfg.center_crop, "Expecting `center_crop==True` because model was trained with image augmentations!"

    assert not (cfg.load_in_8bit and cfg.load_in_4bit), "Cannot use both 8-bit and 4-bit quantization!"
    assert cfg.task_suite_name in [suite.value for suite in TaskSuite], f"Invalid task suite: {cfg.task_suite_name}"
    assert cfg.vla_cache_static_top_k >= 0, "vla_cache_static_top_k must be >= 0"
    assert 0.0 <= cfg.vla_cache_similarity_threshold <= 1.0, "vla_cache_similarity_threshold must be in [0, 1]"
    assert cfg.vla_cache_attention_top_k >= 0, "vla_cache_attention_top_k must be >= 0"
    assert 0.0 <= cfg.vla_cache_growth_factor <= 1.0, "vla_cache_growth_factor must be in [0, 1]"
    if bool(getattr(cfg, "eval_with_mug", False)):
        presets_path = Path(str(cfg.eval_mug_placement_presets_path)).expanduser()
        assert presets_path.is_file(), f"eval_mug_placement_presets_path does not exist: {presets_path}"
    if cfg.max_tasks is not None:
        assert cfg.max_tasks >= 0, "max_tasks must be >= 0"


def load_vla_cache_config_from_json_if_any(cfg: VLACacheGenerateConfig) -> None:
    if not cfg.vla_cache_config_json or not os.path.isfile(cfg.vla_cache_config_json):
        normalize_cfg_paths(
            cfg,
            (
                "pretrained_checkpoint",
                "lora_weight_path",
                "initial_states_path",
                "eval_mug_cache_dir",
                "eval_mug_generated_data_dir",
                "eval_mug_pose_catalog_path",
                "eval_mug_placement_presets_path",
                "eval_initial_global_image_dir",
                "local_log_dir",
                "custom_log_filepath",
            ),
        )
        return

    try:
        with open(cfg.vla_cache_config_json, "r", encoding="utf-8") as f:
            data = json.load(f)
        defaults = VLACacheGenerateConfig()

        _apply_if_present(data, cfg, "use_vla_cache", cast=bool, defaults=defaults)
        _apply_if_present(data, cfg, "vla_cache_static_top_k", cast=int, defaults=defaults)
        _apply_if_present(data, cfg, "vla_cache_similarity_threshold", cast=float, defaults=defaults)
        _apply_if_present(data, cfg, "vla_cache_attention_top_k", cast=int, defaults=defaults)
        _apply_if_present(data, cfg, "vla_cache_attention_layer", cast=int, defaults=defaults)
        _apply_if_present(data, cfg, "vla_cache_apply_weighted_growth", cast=bool, defaults=defaults)
        _apply_if_present(data, cfg, "vla_cache_growth_factor", cast=float, defaults=defaults)

        _apply_if_present(data, cfg, "model_family", cast=str, defaults=defaults)
        _apply_if_present(data, cfg, "pretrained_checkpoint", defaults=defaults)
        _apply_if_present(data, cfg, "lora_weight_path", defaults=defaults)
        _apply_if_present(data, cfg, "use_l1_regression", cast=bool, defaults=defaults)
        _apply_if_present(data, cfg, "use_diffusion", cast=bool, defaults=defaults)
        _apply_if_present(data, cfg, "num_diffusion_steps_train", cast=int, defaults=defaults)
        _apply_if_present(data, cfg, "num_diffusion_steps_inference", cast=int, defaults=defaults)
        _apply_if_present(data, cfg, "use_film", cast=bool, defaults=defaults)
        _apply_if_present(data, cfg, "num_images_in_input", cast=int, defaults=defaults)
        _apply_if_present(data, cfg, "use_proprio", cast=bool, defaults=defaults)
        _apply_if_present(data, cfg, "center_crop", cast=bool, defaults=defaults)
        _apply_if_present(data, cfg, "num_open_loop_steps", cast=int, defaults=defaults)
        _apply_if_present(data, cfg, "lora_rank", cast=int, defaults=defaults)
        _apply_if_present(data, cfg, "unnorm_key", defaults=defaults)
        _apply_if_present(data, cfg, "load_in_8bit", cast=bool, defaults=defaults)
        _apply_if_present(data, cfg, "load_in_4bit", cast=bool, defaults=defaults)

        _apply_if_present(data, cfg, "task_suite_name", cast=str, defaults=defaults)
        _apply_if_present(data, cfg, "num_steps_wait", cast=int, defaults=defaults)
        _apply_if_present(data, cfg, "num_trials_per_task", cast=int, defaults=defaults)
        _apply_if_present(data, cfg, "max_tasks", cast=int, defaults=defaults)
        _apply_if_present(data, cfg, "initial_states_path", cast=str, defaults=defaults)
        _apply_if_present(data, cfg, "env_img_res", cast=int, defaults=defaults)
        _apply_if_present(data, cfg, "eval_with_white_patch", cast=bool, defaults=defaults)
        _apply_if_present(data, cfg, "eval_with_mug", cast=bool, defaults=defaults)
        _apply_if_present(data, cfg, "eval_mug_cache_dir", cast=str, defaults=defaults)
        _apply_if_present(data, cfg, "eval_mug_generated_data_dir", cast=str, defaults=defaults)
        _apply_if_present(data, cfg, "eval_mug_pose_catalog_path", cast=str, defaults=defaults)
        _apply_if_present(data, cfg, "eval_mug_placement_presets_path", cast=str, defaults=defaults)
        _apply_if_present(data, cfg, "eval_mug_max_init_retries", cast=int, defaults=defaults)
        _apply_if_present(data, cfg, "eval_save_initial_global_images", cast=bool, defaults=defaults)
        _apply_if_present(data, cfg, "eval_initial_global_image_dir", cast=str, defaults=defaults)

        _apply_if_present(data, cfg, "run_id_note", defaults=defaults)
        _apply_if_present(data, cfg, "local_log_dir", cast=str, defaults=defaults)
        _apply_if_present(data, cfg, "custom_log_filepath", defaults=defaults)
        _apply_if_present(data, cfg, "use_wandb", cast=bool, defaults=defaults)
        _apply_if_present(data, cfg, "wandb_entity", cast=str, defaults=defaults)
        _apply_if_present(data, cfg, "wandb_project", cast=str, defaults=defaults)
        _apply_if_present(data, cfg, "seed", cast=int, defaults=defaults)
        normalize_cfg_paths(
            cfg,
            (
                "pretrained_checkpoint",
                "lora_weight_path",
                "initial_states_path",
                "eval_mug_cache_dir",
                "eval_mug_generated_data_dir",
                "eval_mug_pose_catalog_path",
                "eval_mug_placement_presets_path",
                "eval_initial_global_image_dir",
                "local_log_dir",
                "custom_log_filepath",
            ),
        )
    except Exception as exc:
        print(f"[VLA-Cache] failed to load vla cache config json: {exc}")


def _apply_if_present(data, cfg, key, cast=None, defaults=None):
    if key not in data:
        return
    if defaults is not None and getattr(cfg, key) != getattr(defaults, key):
        return
    value = data.get(key)
    if cast is not None and value is not None:
        value = cast(value)
    setattr(cfg, key, value)


def apply_runtime_overrides(cfg: VLACacheGenerateConfig) -> None:
    # Keep JSON as the source of truth for runtime options under VLACache.
    del cfg
