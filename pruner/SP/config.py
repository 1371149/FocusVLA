"""Configuration helpers for SP-VLA-style pruning evaluation."""

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


SP_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_ROOT = SP_DIR.parents[2] / "model"


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


SP_PROFILE_DEFAULTS = {
    TaskSuite.LIBERO_SPATIAL.value: {
        "sp_skip_window": 1,
        "sp_z_xy_rate_skip": 0.4,
        "sp_z_max_skip": 0.3,
        "sp_z_thre_prune": 0.5,
        "sp_z_min_prune": 0.9,
    },
    TaskSuite.LIBERO_OBJECT.value: {
        "sp_skip_window": 2,
        "sp_z_xy_rate_skip": 0.6,
        "sp_z_max_skip": 0.5,
        "sp_z_thre_prune": 0.5,
        "sp_z_min_prune": 0.9,
    },
    TaskSuite.LIBERO_GOAL.value: {
        "sp_skip_window": 2,
        "sp_z_xy_rate_skip": 1.2,
        "sp_z_max_skip": 0.3,
        "sp_z_thre_prune": 0.3,
        "sp_z_min_prune": 0.7,
    },
    TaskSuite.LIBERO_10.value: {
        "sp_skip_window": 2,
        "sp_z_xy_rate_skip": 1.0,
        "sp_z_max_skip": 0.3,
        "sp_z_thre_prune": 0.3,
        "sp_z_min_prune": 0.85,
    },
    TaskSuite.LIBERO_90.value: {
        "sp_skip_window": 2,
        "sp_z_xy_rate_skip": 1.0,
        "sp_z_max_skip": 0.3,
        "sp_z_thre_prune": 0.3,
        "sp_z_min_prune": 0.85,
    },
}


@dataclass
class SPGenerateConfig:
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
    eval_mug_cache_dir: str = str(SP_DIR / "generated_bddl_mug")
    eval_mug_generated_data_dir: str = ""
    eval_mug_pose_catalog_path: str = str(SP_DIR.parent / "mug_pose_stats" / "mug_pose_catalog.json")
    eval_mug_placement_presets_path: str = str(PRUNER_DIR / "mug_placement_presets.json")
    eval_mug_max_init_retries: int = 0
    eval_save_initial_global_images: bool = False
    eval_initial_global_image_dir: str = ""

    #################################################################################################################
    # Logging / utils
    #################################################################################################################
    run_id_note: Optional[str] = "sp_vla"
    local_log_dir: str = str(SP_DIR / "logs")
    custom_log_filepath: Optional[str] = str(SP_DIR / "logs" / "sp_eval.txt")

    use_wandb: bool = False
    wandb_entity: str = "your-wandb-entity"
    wandb_project: str = "your-wandb-project"
    sp_save_first_pruned_image: bool = False

    seed: int = 7

    #################################################################################################################
    # Local config bootstrap
    #################################################################################################################
    sp_config_json: Optional[str] = str(SP_DIR / "configs" / "sp_config.json")

    #################################################################################################################
    # SP pruning / skipping parameters
    #################################################################################################################
    sp_enable_token_pruning: bool = True
    sp_enable_step_skipping: bool = True
    sp_profile: str = "auto"

    sp_history_size: int = 6
    sp_min_generate_fraction: float = 0.6666666666666666

    sp_skip_window: Optional[int] = None
    sp_z_xy_rate_skip: Optional[float] = None
    sp_z_max_skip: Optional[float] = None

    sp_z_thre_prune: Optional[float] = None
    sp_z_min_prune: Optional[float] = None
    sp_max_prune_rate: Optional[float] = None
    sp_fixed_keep_ratio: Optional[float] = None
    sp_keep_min_tokens: int = 1
    sp_edge_threshold1: int = 100
    sp_edge_threshold2: int = 200
    sp_edge_patch_size: Optional[int] = None

    # fmt: on


def validate_config(cfg: SPGenerateConfig) -> None:
    assert cfg.pretrained_checkpoint is not None, "pretrained_checkpoint must not be None!"

    if "image_aug" in str(cfg.pretrained_checkpoint):
        assert cfg.center_crop, "Expecting `center_crop==True` because model was trained with image augmentations!"

    assert not (cfg.load_in_8bit and cfg.load_in_4bit), "Cannot use both 8-bit and 4-bit quantization!"
    assert cfg.task_suite_name in [suite.value for suite in TaskSuite], f"Invalid task suite: {cfg.task_suite_name}"
    if bool(getattr(cfg, "eval_with_mug", False)):
        presets_path = Path(str(cfg.eval_mug_placement_presets_path)).expanduser()
        assert presets_path.is_file(), f"eval_mug_placement_presets_path does not exist: {presets_path}"

    apply_sp_profile_defaults(cfg)

    assert cfg.sp_history_size >= 1, "sp_history_size must be >= 1"
    assert 0.0 <= cfg.sp_min_generate_fraction <= 1.0, "sp_min_generate_fraction must be in [0, 1]"
    assert cfg.sp_keep_min_tokens >= 1, "sp_keep_min_tokens must be >= 1"
    assert cfg.sp_edge_threshold1 >= 0 and cfg.sp_edge_threshold2 >= 0, "Canny thresholds must be non-negative"

    if cfg.sp_enable_step_skipping:
        assert cfg.sp_skip_window is not None and cfg.sp_skip_window >= 1, "sp_skip_window must be >= 1"
        assert cfg.sp_skip_window <= cfg.sp_history_size, "sp_skip_window must not exceed sp_history_size"
        assert cfg.sp_z_xy_rate_skip is not None and cfg.sp_z_xy_rate_skip >= 0.0
        assert cfg.sp_z_max_skip is not None and cfg.sp_z_max_skip >= 0.0

    if cfg.sp_enable_token_pruning:
        assert cfg.sp_z_thre_prune is not None and 0.0 <= cfg.sp_z_thre_prune <= 1.0
        assert cfg.sp_z_min_prune is not None and 0.0 <= cfg.sp_z_min_prune <= 1.0
        if cfg.sp_max_prune_rate is not None:
            assert 0.0 <= cfg.sp_max_prune_rate <= 1.0, "sp_max_prune_rate must be in [0, 1]"
        if cfg.sp_fixed_keep_ratio is not None:
            assert 0.0 < cfg.sp_fixed_keep_ratio <= 1.0, "sp_fixed_keep_ratio must be in (0, 1]"


def resolve_sp_profile(cfg: SPGenerateConfig) -> str:
    if cfg.sp_profile != "auto":
        if cfg.sp_profile not in SP_PROFILE_DEFAULTS:
            raise ValueError(f"Unsupported sp_profile={cfg.sp_profile}. Choose from {sorted(SP_PROFILE_DEFAULTS)}")
        return cfg.sp_profile
    return cfg.task_suite_name if cfg.task_suite_name in SP_PROFILE_DEFAULTS else TaskSuite.LIBERO_10.value


def apply_sp_profile_defaults(cfg: SPGenerateConfig) -> None:
    profile_name = resolve_sp_profile(cfg)
    defaults = SP_PROFILE_DEFAULTS[profile_name]
    for key, value in defaults.items():
        if getattr(cfg, key) is None:
            setattr(cfg, key, value)


def load_sp_config_from_json_if_any(cfg: SPGenerateConfig) -> None:
    if not cfg.sp_config_json or not os.path.isfile(cfg.sp_config_json):
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
        apply_sp_profile_defaults(cfg)
        return

    defaults = SPGenerateConfig(sp_config_json=None)

    try:
        with open(cfg.sp_config_json, "r", encoding="utf-8") as f:
            data = json.load(f)

        _apply_if_present(data, cfg, defaults, "model_family", cast=str)
        _apply_if_present(data, cfg, defaults, "pretrained_checkpoint")
        _apply_if_present(data, cfg, defaults, "lora_weight_path")
        _apply_if_present(data, cfg, defaults, "use_l1_regression", cast=bool)
        _apply_if_present(data, cfg, defaults, "use_diffusion", cast=bool)
        _apply_if_present(data, cfg, defaults, "num_diffusion_steps_train", cast=int)
        _apply_if_present(data, cfg, defaults, "num_diffusion_steps_inference", cast=int)
        _apply_if_present(data, cfg, defaults, "use_film", cast=bool)
        _apply_if_present(data, cfg, defaults, "num_images_in_input", cast=int)
        _apply_if_present(data, cfg, defaults, "use_proprio", cast=bool)
        _apply_if_present(data, cfg, defaults, "center_crop", cast=bool)
        _apply_if_present(data, cfg, defaults, "num_open_loop_steps", cast=int)
        _apply_if_present(data, cfg, defaults, "lora_rank", cast=int)
        _apply_if_present(data, cfg, defaults, "unnorm_key")
        _apply_if_present(data, cfg, defaults, "load_in_8bit", cast=bool)
        _apply_if_present(data, cfg, defaults, "load_in_4bit", cast=bool)

        _apply_if_present(data, cfg, defaults, "task_suite_name", cast=str)
        _apply_if_present(data, cfg, defaults, "num_steps_wait", cast=int)
        _apply_if_present(data, cfg, defaults, "num_trials_per_task", cast=int)
        _apply_if_present(data, cfg, defaults, "initial_states_path", cast=str)
        _apply_if_present(data, cfg, defaults, "env_img_res", cast=int)
        _apply_if_present(data, cfg, defaults, "eval_with_white_patch", cast=bool)
        _apply_if_present(data, cfg, defaults, "eval_with_mug", cast=bool)
        _apply_if_present(data, cfg, defaults, "eval_mug_cache_dir", cast=str)
        _apply_if_present(data, cfg, defaults, "eval_mug_generated_data_dir", cast=str)
        _apply_if_present(data, cfg, defaults, "eval_mug_pose_catalog_path", cast=str)
        _apply_if_present(data, cfg, defaults, "eval_mug_placement_presets_path", cast=str)
        _apply_if_present(data, cfg, defaults, "eval_mug_max_init_retries", cast=int)
        _apply_if_present(data, cfg, defaults, "eval_save_initial_global_images", cast=bool)
        _apply_if_present(data, cfg, defaults, "eval_initial_global_image_dir", cast=str)

        _apply_if_present(data, cfg, defaults, "run_id_note")
        _apply_if_present(data, cfg, defaults, "local_log_dir", cast=str)
        _apply_if_present(data, cfg, defaults, "custom_log_filepath")
        _apply_if_present(data, cfg, defaults, "use_wandb", cast=bool)
        _apply_if_present(data, cfg, defaults, "wandb_entity", cast=str)
        _apply_if_present(data, cfg, defaults, "wandb_project", cast=str)
        _apply_if_present(data, cfg, defaults, "sp_save_first_pruned_image", cast=bool)
        _apply_if_present(data, cfg, defaults, "seed", cast=int)

        _apply_if_present(data, cfg, defaults, "sp_enable_token_pruning", cast=bool)
        _apply_if_present(data, cfg, defaults, "sp_enable_step_skipping", cast=bool)
        _apply_if_present(data, cfg, defaults, "sp_profile", cast=str)
        _apply_if_present(data, cfg, defaults, "sp_history_size", cast=int)
        _apply_if_present(data, cfg, defaults, "sp_min_generate_fraction", cast=float)
        _apply_if_present(data, cfg, defaults, "sp_skip_window", cast=int)
        _apply_if_present(data, cfg, defaults, "sp_z_xy_rate_skip", cast=float)
        _apply_if_present(data, cfg, defaults, "sp_z_max_skip", cast=float)
        _apply_if_present(data, cfg, defaults, "sp_z_thre_prune", cast=float)
        _apply_if_present(data, cfg, defaults, "sp_z_min_prune", cast=float)
        _apply_if_present(data, cfg, defaults, "sp_max_prune_rate", cast=float)
        _apply_if_present(data, cfg, defaults, "sp_fixed_keep_ratio", cast=float)
        _apply_if_present(data, cfg, defaults, "sp_keep_min_tokens", cast=int)
        _apply_if_present(data, cfg, defaults, "sp_edge_threshold1", cast=int)
        _apply_if_present(data, cfg, defaults, "sp_edge_threshold2", cast=int)
        _apply_if_present(data, cfg, defaults, "sp_edge_patch_size", cast=int)
    except Exception as exc:
        print(f"[SP] failed to load sp config json: {exc}")

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
    apply_sp_profile_defaults(cfg)


def _apply_if_present(data, cfg, defaults, key, cast=None):
    if key not in data:
        return
    if getattr(cfg, key) != getattr(defaults, key):
        return
    value = data.get(key)
    if cast is not None and value is not None:
        value = cast(value)
    setattr(cfg, key, value)
