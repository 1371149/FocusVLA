"""Configuration helpers for ADP pruning evaluation.

All user-facing evaluation settings are kept in this file on purpose, so the
ADP workflow can be configured entirely inside `openvla-oft/pruner/ADP`.
"""

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


ADP_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_ROOT = ADP_DIR.parents[2] / "model"


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
class ADPGenerateConfig:
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
    task_suite_name: str = TaskSuite.LIBERO_SPATIAL
    num_steps_wait: int = 10
    num_trials_per_task: int = 50
    initial_states_path: str = "DEFAULT"
    env_img_res: int = 256
    eval_with_white_patch: bool = False

    # Optional evaluation-time blue mug injection.
    eval_with_mug: bool = False
    eval_mug_cache_dir: str = str(ADP_DIR / "generated_bddl_mug")
    eval_mug_generated_data_dir: str = ""
    eval_mug_placement_presets_path: str = str(PRUNER_DIR / "mug_placement_presets.json")

    #################################################################################################################
    # Logging / utils
    #################################################################################################################
    run_id_note: Optional[str] = None
    local_log_dir: str = str(ADP_DIR / "logs")
    custom_log_filepath: Optional[str] = None

    use_wandb: bool = False
    wandb_entity: str = "your-wandb-entity"
    wandb_project: str = "your-wandb-project"

    seed: int = 7

    #################################################################################################################
    # ADP pruning parameters
    #################################################################################################################
    qk_config_json: Optional[str] = str(ADP_DIR / "configs" / "prune_v2_config.json")
    qk_keep_enabled: bool = True
    qk_layer: int = 0
    qk_keep_ratio: float = 0.75
    qk_keep_split: Optional[str] = None
    qk_debug: bool = False
    qk_log_topk: int = 16

    use_dynamic_visual_strategy: bool = True
    decision_method: str = "avg"
    adjacent_variant: str = "classic"
    adjacent_extrema_window: int = 3
    adjacent_lookback: int = 2
    adjacent_last_state: bool = False
    initial_state: int = 0
    delta_method: str = "net"
    L_eff: float = 0.15
    min_delta_pos: float = 0.0
    min_delta_rot: float = 0.0
    hysteresis_up: float = 0.0
    hysteresis_down: float = 0.0
    tol_equal: float = 0.0

    limit_consecutive_pruned_enabled: bool = True
    limit_max_consecutive_pruned: int = 3

    # fmt: on


def validate_config(cfg: ADPGenerateConfig) -> None:
    assert cfg.pretrained_checkpoint is not None, "pretrained_checkpoint must not be None!"

    if "image_aug" in str(cfg.pretrained_checkpoint):
        assert cfg.center_crop, "Expecting `center_crop==True` because model was trained with image augmentations!"

    assert not (cfg.load_in_8bit and cfg.load_in_4bit), "Cannot use both 8-bit and 4-bit quantization!"
    assert cfg.task_suite_name in [suite.value for suite in TaskSuite], f"Invalid task suite: {cfg.task_suite_name}"
    if bool(getattr(cfg, "eval_with_mug", False)):
        presets_path = Path(str(cfg.eval_mug_placement_presets_path)).expanduser()
        assert presets_path.is_file(), f"eval_mug_placement_presets_path does not exist: {presets_path}"


def load_qk_config_from_json_if_any(cfg: ADPGenerateConfig) -> None:
    if not cfg.qk_config_json or not os.path.isfile(cfg.qk_config_json):
        normalize_cfg_paths(
            cfg,
            (
                "pretrained_checkpoint",
                "lora_weight_path",
                "initial_states_path",
                "eval_mug_cache_dir",
                "eval_mug_generated_data_dir",
                "eval_mug_placement_presets_path",
                "eval_mug_pose_catalog_path",
                "local_log_dir",
                "custom_log_filepath",
            ),
        )
        return

    try:
        with open(cfg.qk_config_json, "r", encoding="utf-8") as f:
            data = json.load(f)

        if "model_family" in data:
            cfg.model_family = str(data.get("model_family", cfg.model_family))
        if "pretrained_checkpoint" in data:
            cfg.pretrained_checkpoint = data.get("pretrained_checkpoint", cfg.pretrained_checkpoint)
        if "lora_weight_path" in data:
            cfg.lora_weight_path = data.get("lora_weight_path", cfg.lora_weight_path)
        if "use_l1_regression" in data:
            cfg.use_l1_regression = bool(data.get("use_l1_regression", cfg.use_l1_regression))
        if "use_diffusion" in data:
            cfg.use_diffusion = bool(data.get("use_diffusion", cfg.use_diffusion))
        if "num_diffusion_steps_train" in data:
            cfg.num_diffusion_steps_train = int(data.get("num_diffusion_steps_train", cfg.num_diffusion_steps_train))
        if "num_diffusion_steps_inference" in data:
            cfg.num_diffusion_steps_inference = int(
                data.get("num_diffusion_steps_inference", cfg.num_diffusion_steps_inference)
            )
        if "use_film" in data:
            cfg.use_film = bool(data.get("use_film", cfg.use_film))
        if "num_images_in_input" in data:
            cfg.num_images_in_input = int(data.get("num_images_in_input", cfg.num_images_in_input))
        if "use_proprio" in data:
            cfg.use_proprio = bool(data.get("use_proprio", cfg.use_proprio))
        if "center_crop" in data:
            cfg.center_crop = bool(data.get("center_crop", cfg.center_crop))
        if "num_open_loop_steps" in data:
            cfg.num_open_loop_steps = int(data.get("num_open_loop_steps", cfg.num_open_loop_steps))
        if "lora_rank" in data:
            cfg.lora_rank = int(data.get("lora_rank", cfg.lora_rank))
        if "unnorm_key" in data:
            cfg.unnorm_key = data.get("unnorm_key", cfg.unnorm_key)
        if "load_in_8bit" in data:
            cfg.load_in_8bit = bool(data.get("load_in_8bit", cfg.load_in_8bit))
        if "load_in_4bit" in data:
            cfg.load_in_4bit = bool(data.get("load_in_4bit", cfg.load_in_4bit))

        if "task_suite_name" in data:
            cfg.task_suite_name = str(data.get("task_suite_name", cfg.task_suite_name))
        if "num_steps_wait" in data:
            cfg.num_steps_wait = int(data.get("num_steps_wait", cfg.num_steps_wait))
        if "num_trials_per_task" in data:
            cfg.num_trials_per_task = int(data.get("num_trials_per_task", cfg.num_trials_per_task))
        if "initial_states_path" in data:
            cfg.initial_states_path = str(data.get("initial_states_path", cfg.initial_states_path))
        if "env_img_res" in data:
            cfg.env_img_res = int(data.get("env_img_res", cfg.env_img_res))
        if "eval_with_white_patch" in data:
            cfg.eval_with_white_patch = bool(data.get("eval_with_white_patch", cfg.eval_with_white_patch))
        if "eval_with_mug" in data:
            cfg.eval_with_mug = bool(data.get("eval_with_mug", cfg.eval_with_mug))
        if "eval_mug_cache_dir" in data:
            cfg.eval_mug_cache_dir = str(data.get("eval_mug_cache_dir", cfg.eval_mug_cache_dir))
        if "eval_mug_generated_data_dir" in data:
            cfg.eval_mug_generated_data_dir = str(
                data.get("eval_mug_generated_data_dir", cfg.eval_mug_generated_data_dir)
            )
        if "eval_mug_placement_presets_path" in data:
            cfg.eval_mug_placement_presets_path = str(
                data.get("eval_mug_placement_presets_path", cfg.eval_mug_placement_presets_path)
            )

        if "run_id_note" in data:
            cfg.run_id_note = data.get("run_id_note", cfg.run_id_note)
        if "local_log_dir" in data:
            cfg.local_log_dir = str(data.get("local_log_dir", cfg.local_log_dir))
        if "custom_log_filepath" in data:
            cfg.custom_log_filepath = data.get("custom_log_filepath", cfg.custom_log_filepath)
        if "use_wandb" in data:
            cfg.use_wandb = bool(data.get("use_wandb", cfg.use_wandb))
        if "wandb_entity" in data:
            cfg.wandb_entity = str(data.get("wandb_entity", cfg.wandb_entity))
        if "wandb_project" in data:
            cfg.wandb_project = str(data.get("wandb_project", cfg.wandb_project))
        if "seed" in data:
            cfg.seed = int(data.get("seed", cfg.seed))

        cfg.qk_keep_enabled = bool(data.get("qk_keep_enabled", cfg.qk_keep_enabled))
        cfg.qk_layer = int(data.get("qk_layer", cfg.qk_layer))
        cfg.qk_keep_ratio = float(data.get("qk_keep_ratio", cfg.qk_keep_ratio))
        split = data.get("qk_keep_split", cfg.qk_keep_split)
        if isinstance(split, list):
            split = ",".join(str(float(x)) for x in split)
        cfg.qk_keep_split = split
        cfg.qk_debug = bool(data.get("qk_debug", cfg.qk_debug))
        cfg.qk_log_topk = int(data.get("qk_log_topk", cfg.qk_log_topk))

        cfg.use_dynamic_visual_strategy = bool(
            data.get("use_dynamic_visual_strategy", cfg.use_dynamic_visual_strategy)
        )
        cfg.decision_method = str(data.get("decision_method", cfg.decision_method))
        cfg.adjacent_variant = str(data.get("adjacent_variant", cfg.adjacent_variant))
        cfg.adjacent_extrema_window = int(data.get("adjacent_extrema_window", cfg.adjacent_extrema_window))
        cfg.adjacent_last_state = bool(data.get("adjacent_last_state", cfg.adjacent_last_state))
        cfg.adjacent_lookback = int(data.get("adjacent_lookback", cfg.adjacent_lookback))
        cfg.initial_state = int(data.get("initial_state", cfg.initial_state))
        cfg.delta_method = str(data.get("delta_method", cfg.delta_method))
        cfg.L_eff = float(data.get("L_eff", cfg.L_eff))
        cfg.min_delta_pos = float(data.get("min_delta_pos", cfg.min_delta_pos))
        cfg.min_delta_rot = float(data.get("min_delta_rot", cfg.min_delta_rot))
        cfg.hysteresis_up = float(data.get("hysteresis_up", cfg.hysteresis_up))
        cfg.hysteresis_down = float(data.get("hysteresis_down", cfg.hysteresis_down))
        cfg.tol_equal = float(data.get("tol_equal", cfg.tol_equal))
        cfg.limit_consecutive_pruned_enabled = bool(
            data.get("limit_consecutive_pruned_enabled", cfg.limit_consecutive_pruned_enabled)
        )
        cfg.limit_max_consecutive_pruned = int(
            data.get("limit_max_consecutive_pruned", cfg.limit_max_consecutive_pruned)
        )
        normalize_cfg_paths(
            cfg,
            (
                "pretrained_checkpoint",
                "lora_weight_path",
                "initial_states_path",
                "eval_mug_cache_dir",
                "eval_mug_generated_data_dir",
                "eval_mug_placement_presets_path",
                "eval_mug_pose_catalog_path",
                "local_log_dir",
                "custom_log_filepath",
            ),
        )
    except Exception as exc:
        print(f"[ADP] failed to load qk config json: {exc}")


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


def apply_runtime_overrides(cfg: ADPGenerateConfig) -> None:
    white_patch_override = _parse_bool_env_override("ADP_EVAL_WITH_WHITE_PATCH")
    if white_patch_override is not None:
        cfg.eval_with_white_patch = white_patch_override

    mug_override = _parse_bool_env_override("ADP_EVAL_WITH_MUG")
    if mug_override is not None:
        cfg.eval_with_mug = mug_override

    presets_override = os.environ.get("ADP_EVAL_MUG_PLACEMENT_PRESETS_PATH")
    if presets_override is not None and presets_override.strip():
        cfg.eval_mug_placement_presets_path = presets_override.strip()
