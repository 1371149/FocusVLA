"""Configuration helpers for AdaptPrune evaluation."""

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


ADAPTPRUNE_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_ROOT = ADAPTPRUNE_DIR.parents[2] / "model"


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
class AdaptPruneGenerateConfig:
    # Model-specific parameters
    model_family: str = "openvla"
    pretrained_checkpoint: Union[str, Path] = str(DEFAULT_MODEL_ROOT / "openvla-7b-oft-finetuned-libero-spatial")
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

    # LIBERO environment parameters
    task_suite_name: str = TaskSuite.LIBERO_SPATIAL.value
    num_steps_wait: int = 10
    num_trials_per_task: int = 50
    initial_states_path: str = "DEFAULT"
    env_img_res: int = 256
    eval_with_white_patch: bool = False
    eval_with_mug: bool = False
    eval_mug_cache_dir: str = str(ADAPTPRUNE_DIR / "generated_bddl_mug")
    eval_mug_generated_data_dir: str = ""
    eval_mug_pose_catalog_path: str = str(ADAPTPRUNE_DIR.parent / "mug_pose_stats" / "mug_pose_catalog.json")
    eval_mug_placement_presets_path: str = str(PRUNER_DIR / "mug_placement_presets.json")
    eval_mug_max_init_retries: int = 0
    eval_save_initial_global_images: bool = False
    eval_initial_global_image_dir: str = ""

    # Logging
    run_id_note: Optional[str] = "adaptprune"
    local_log_dir: str = str(ADAPTPRUNE_DIR / "logs")
    custom_log_filepath: Optional[str] = str(ADAPTPRUNE_DIR / "logs" / "adaptprune_eval.txt")

    use_wandb: bool = False
    wandb_entity: str = "your-wandb-entity"
    wandb_project: str = "your-wandb-project"
    seed: int = 7

    # Local config bootstrap
    adaptprune_config_json: Optional[str] = str(ADAPTPRUNE_DIR / "configs" / "adaptprune_config.json")

    # AdaptPrune parameters
    adaptprune_enabled: bool = True
    adaptprune_layer: int = 2
    adaptprune_keep_ratio: float = 0.5
    adaptprune_distance_limit: float = 5.0
    adaptprune_distance_sigma: float = 1.5
    adaptprune_similarity_sigma: float = 0.05
    adaptprune_use_center_gaussian: bool = True
    adaptprune_center_sigma: float = 8.0
    adaptprune_min_keep_tokens_per_image: int = 1
    adaptprune_query_mode: str = "text_last"


def validate_config(cfg: AdaptPruneGenerateConfig) -> None:
    assert cfg.pretrained_checkpoint is not None, "pretrained_checkpoint must not be None!"
    if "image_aug" in str(cfg.pretrained_checkpoint):
        assert cfg.center_crop, "Expecting `center_crop==True` because model was trained with image augmentations!"
    assert not (cfg.load_in_8bit and cfg.load_in_4bit), "Cannot use both 8-bit and 4-bit quantization!"
    assert cfg.task_suite_name in [suite.value for suite in TaskSuite], f"Invalid task suite: {cfg.task_suite_name}"
    if bool(getattr(cfg, "eval_with_mug", False)):
        presets_path = Path(str(cfg.eval_mug_placement_presets_path)).expanduser()
        assert presets_path.is_file(), f"eval_mug_placement_presets_path does not exist: {presets_path}"

    assert cfg.adaptprune_layer >= 1, "adaptprune_layer must be >= 1 because layer-1 attention is used for pruning."
    assert 0.0 < cfg.adaptprune_keep_ratio <= 1.0, "adaptprune_keep_ratio must be in (0, 1]."
    assert cfg.adaptprune_min_keep_tokens_per_image >= 1, "adaptprune_min_keep_tokens_per_image must be >= 1."
    assert cfg.adaptprune_distance_limit >= 0, "adaptprune_distance_limit must be >= 0."
    assert cfg.adaptprune_distance_sigma > 0, "adaptprune_distance_sigma must be > 0."
    assert cfg.adaptprune_similarity_sigma > 0, "adaptprune_similarity_sigma must be > 0."
    assert cfg.adaptprune_center_sigma > 0, "adaptprune_center_sigma must be > 0."
    assert cfg.adaptprune_query_mode in {"text_mean", "text_last", "action_last"}, (
        "adaptprune_query_mode must be one of: text_mean, text_last, action_last."
    )


def load_adaptprune_config_from_json_if_any(cfg: AdaptPruneGenerateConfig) -> None:
    if not cfg.adaptprune_config_json or not os.path.isfile(cfg.adaptprune_config_json):
        _normalize_paths(cfg)
        return

    try:
        with open(cfg.adaptprune_config_json, "r", encoding="utf-8") as f:
            data = json.load(f)

        for key, cast in (
            ("model_family", str),
            ("pretrained_checkpoint", None),
            ("lora_weight_path", None),
            ("use_l1_regression", bool),
            ("use_diffusion", bool),
            ("num_diffusion_steps_train", int),
            ("num_diffusion_steps_inference", int),
            ("use_film", bool),
            ("num_images_in_input", int),
            ("use_proprio", bool),
            ("center_crop", bool),
            ("num_open_loop_steps", int),
            ("lora_rank", int),
            ("unnorm_key", None),
            ("load_in_8bit", bool),
            ("load_in_4bit", bool),
            ("task_suite_name", str),
            ("num_steps_wait", int),
            ("num_trials_per_task", int),
            ("initial_states_path", str),
            ("env_img_res", int),
            ("eval_with_white_patch", bool),
            ("eval_with_mug", bool),
            ("eval_mug_cache_dir", str),
            ("eval_mug_generated_data_dir", str),
            ("eval_mug_pose_catalog_path", str),
            ("eval_mug_placement_presets_path", str),
            ("eval_mug_max_init_retries", int),
            ("eval_save_initial_global_images", bool),
            ("eval_initial_global_image_dir", str),
            ("run_id_note", None),
            ("local_log_dir", str),
            ("custom_log_filepath", None),
            ("use_wandb", bool),
            ("wandb_entity", str),
            ("wandb_project", str),
            ("seed", int),
            ("adaptprune_enabled", bool),
            ("adaptprune_layer", int),
            ("adaptprune_keep_ratio", float),
            ("adaptprune_distance_limit", float),
            ("adaptprune_distance_sigma", float),
            ("adaptprune_similarity_sigma", float),
            ("adaptprune_use_center_gaussian", bool),
            ("adaptprune_center_sigma", float),
            ("adaptprune_min_keep_tokens_per_image", int),
            ("adaptprune_query_mode", str),
        ):
            _apply_if_present(data, cfg, key, cast=cast)
        _normalize_paths(cfg)
    except Exception as exc:
        print(f"[AdaptPrune] failed to load adaptprune config json: {exc}")


def _normalize_paths(cfg: AdaptPruneGenerateConfig) -> None:
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


def _apply_if_present(data, cfg, key, cast=None):
    if key not in data:
        return
    value = data.get(key)
    if cast is not None and value is not None:
        value = cast(value)
    setattr(cfg, key, value)


def _parse_bool_env_override(name: str) -> Optional[bool]:
    raw_value = os.environ.get(name)
    if raw_value is None:
        return None

    value = raw_value.strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    raise ValueError(
        f"Invalid boolean override `{raw_value}` for environment variable `{name}`. "
        "Expected one of: 1/0, true/false, yes/no, on/off."
    )


def apply_runtime_overrides(cfg: AdaptPruneGenerateConfig) -> None:
    white_patch_override = _parse_bool_env_override("ADAPTPRUNE_EVAL_WITH_WHITE_PATCH")
    if white_patch_override is not None:
        cfg.eval_with_white_patch = white_patch_override

    mug_override = _parse_bool_env_override("ADAPTPRUNE_EVAL_WITH_MUG")
    if mug_override is not None:
        cfg.eval_with_mug = mug_override

    save_initial_image_override = _parse_bool_env_override("ADAPTPRUNE_EVAL_SAVE_INITIAL_GLOBAL_IMAGES")
    if save_initial_image_override is not None:
        cfg.eval_save_initial_global_images = save_initial_image_override

    presets_override = os.environ.get("ADAPTPRUNE_EVAL_MUG_PLACEMENT_PRESETS_PATH")
    if presets_override is not None and presets_override.strip():
        cfg.eval_mug_placement_presets_path = presets_override.strip()

    initial_image_dir_override = os.environ.get("ADAPTPRUNE_EVAL_INITIAL_GLOBAL_IMAGE_DIR")
    if initial_image_dir_override is not None and initial_image_dir_override.strip():
        cfg.eval_initial_global_image_dir = initial_image_dir_override.strip()
