"""Configuration helpers for FastV-style pruning evaluation."""

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


FASTV_DIR = Path(__file__).resolve().parent
FASTV_QUERY_MODES = {"last", "last_action", "action_mean"}
DEFAULT_MODEL_ROOT = FASTV_DIR.parents[2] / "model"


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
class FastVGenerateConfig:
    # fmt: off

    #################################################################################################################
    # Model-specific parameters
    #################################################################################################################
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

    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    task_suite_name: str = TaskSuite.LIBERO_SPATIAL.value
    num_steps_wait: int = 10
    num_trials_per_task: int = 50
    initial_states_path: str = "DEFAULT"
    env_img_res: int = 256
    eval_with_white_patch: bool = False
    eval_with_mug: bool = False
    eval_mug_cache_dir: str = str(FASTV_DIR / "generated_bddl_mug")
    eval_mug_generated_data_dir: str = ""
    eval_mug_pose_catalog_path: str = str(FASTV_DIR.parent / "mug_pose_stats" / "mug_pose_catalog.json")
    eval_mug_placement_presets_path: str = str(PRUNER_DIR / "mug_placement_presets.json")
    eval_mug_max_init_retries: int = 0
    eval_save_initial_global_images: bool = False
    eval_initial_global_image_dir: str = ""

    #################################################################################################################
    # Logging / utils
    #################################################################################################################
    run_id_note: Optional[str] = "fastv"
    local_log_dir: str = str(FASTV_DIR / "logs")
    custom_log_filepath: Optional[str] = str(FASTV_DIR / "logs" / "fastv_eval.txt")

    use_wandb: bool = False
    wandb_entity: str = "your-wandb-entity"
    wandb_project: str = "your-wandb-project"

    seed: int = 7

    #################################################################################################################
    # Local config bootstrap
    #################################################################################################################
    fastv_config_json: Optional[str] = str(FASTV_DIR / "configs" / "fastv_config.json")

    #################################################################################################################
    # FastV parameters
    #################################################################################################################
    fastv_enabled: bool = True
    fastv_k: int = 3
    fastv_r: float = 0.5
    fastv_query_mode: str = "action_mean"
    fastv_min_keep_tokens: int = 1

    # fmt: on


def validate_config(cfg: FastVGenerateConfig) -> None:
    assert cfg.pretrained_checkpoint is not None, "pretrained_checkpoint must not be None!"

    if "image_aug" in str(cfg.pretrained_checkpoint):
        assert cfg.center_crop, "Expecting `center_crop==True` because model was trained with image augmentations!"

    assert not (cfg.load_in_8bit and cfg.load_in_4bit), "Cannot use both 8-bit and 4-bit quantization!"
    assert cfg.task_suite_name in [suite.value for suite in TaskSuite], f"Invalid task suite: {cfg.task_suite_name}"
    if bool(getattr(cfg, "eval_with_mug", False)):
        presets_path = Path(str(cfg.eval_mug_placement_presets_path)).expanduser()
        assert presets_path.is_file(), f"eval_mug_placement_presets_path does not exist: {presets_path}"

    assert cfg.fastv_k >= 1, "fastv_k must be >= 1"
    assert 0.0 <= cfg.fastv_r <= 1.0, "fastv_r must be in [0, 1]"
    assert cfg.fastv_min_keep_tokens >= 1, "fastv_min_keep_tokens must be >= 1"
    assert cfg.fastv_query_mode in FASTV_QUERY_MODES, (
        f"Unsupported fastv_query_mode={cfg.fastv_query_mode}. Choose from {sorted(FASTV_QUERY_MODES)}"
    )


def load_fastv_config_from_json_if_any(cfg: FastVGenerateConfig) -> None:
    if not cfg.fastv_config_json or not os.path.isfile(cfg.fastv_config_json):
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
        with open(cfg.fastv_config_json, "r", encoding="utf-8") as f:
            data = json.load(f)

        _apply_if_present(data, cfg, "model_family", cast=str)
        _apply_if_present(data, cfg, "pretrained_checkpoint")
        _apply_if_present(data, cfg, "lora_weight_path")
        _apply_if_present(data, cfg, "use_l1_regression", cast=bool)
        _apply_if_present(data, cfg, "use_diffusion", cast=bool)
        _apply_if_present(data, cfg, "num_diffusion_steps_train", cast=int)
        _apply_if_present(data, cfg, "num_diffusion_steps_inference", cast=int)
        _apply_if_present(data, cfg, "use_film", cast=bool)
        _apply_if_present(data, cfg, "num_images_in_input", cast=int)
        _apply_if_present(data, cfg, "use_proprio", cast=bool)
        _apply_if_present(data, cfg, "center_crop", cast=bool)
        _apply_if_present(data, cfg, "num_open_loop_steps", cast=int)
        _apply_if_present(data, cfg, "lora_rank", cast=int)
        _apply_if_present(data, cfg, "unnorm_key")
        _apply_if_present(data, cfg, "load_in_8bit", cast=bool)
        _apply_if_present(data, cfg, "load_in_4bit", cast=bool)

        _apply_if_present(data, cfg, "task_suite_name", cast=str)
        _apply_if_present(data, cfg, "num_steps_wait", cast=int)
        _apply_if_present(data, cfg, "num_trials_per_task", cast=int)
        _apply_if_present(data, cfg, "initial_states_path", cast=str)
        _apply_if_present(data, cfg, "env_img_res", cast=int)
        _apply_if_present(data, cfg, "eval_with_white_patch", cast=bool)
        _apply_if_present(data, cfg, "eval_with_mug", cast=bool)
        _apply_if_present(data, cfg, "eval_mug_cache_dir", cast=str)
        _apply_if_present(data, cfg, "eval_mug_generated_data_dir", cast=str)
        _apply_if_present(data, cfg, "eval_mug_pose_catalog_path", cast=str)
        _apply_if_present(data, cfg, "eval_mug_placement_presets_path", cast=str)
        _apply_if_present(data, cfg, "eval_mug_max_init_retries", cast=int)
        _apply_if_present(data, cfg, "eval_save_initial_global_images", cast=bool)
        _apply_if_present(data, cfg, "eval_initial_global_image_dir", cast=str)

        _apply_if_present(data, cfg, "run_id_note")
        _apply_if_present(data, cfg, "local_log_dir", cast=str)
        _apply_if_present(data, cfg, "custom_log_filepath")
        _apply_if_present(data, cfg, "use_wandb", cast=bool)
        _apply_if_present(data, cfg, "wandb_entity", cast=str)
        _apply_if_present(data, cfg, "wandb_project", cast=str)
        _apply_if_present(data, cfg, "seed", cast=int)

        _apply_if_present(data, cfg, "fastv_enabled", cast=bool)
        _apply_if_present(data, cfg, "fastv_k", cast=int)
        _apply_if_present(data, cfg, "fastv_r", cast=float)
        _apply_if_present(data, cfg, "fastv_query_mode", cast=str)
        _apply_if_present(data, cfg, "fastv_min_keep_tokens", cast=int)
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
        print(f"[FastV] failed to load fastv config json: {exc}")


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


def apply_runtime_overrides(cfg: FastVGenerateConfig) -> None:
    white_patch_override = _parse_bool_env_override("FASTV_EVAL_WITH_WHITE_PATCH")
    if white_patch_override is not None:
        cfg.eval_with_white_patch = white_patch_override

    mug_override = _parse_bool_env_override("FASTV_EVAL_WITH_MUG")
    if mug_override is not None:
        cfg.eval_with_mug = mug_override

    save_initial_image_override = _parse_bool_env_override("FASTV_EVAL_SAVE_INITIAL_GLOBAL_IMAGES")
    if save_initial_image_override is not None:
        cfg.eval_save_initial_global_images = save_initial_image_override

    presets_override = os.environ.get("FASTV_EVAL_MUG_PLACEMENT_PRESETS_PATH")
    if presets_override is not None and presets_override.strip():
        cfg.eval_mug_placement_presets_path = presets_override.strip()

    initial_image_dir_override = os.environ.get("FASTV_EVAL_INITIAL_GLOBAL_IMAGE_DIR")
    if initial_image_dir_override is not None and initial_image_dir_override.strip():
        cfg.eval_initial_global_image_dir = initial_image_dir_override.strip()
