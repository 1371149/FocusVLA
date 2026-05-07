"""PyramidDrop-style LIBERO evaluation entrypoint for OpenVLA-OFT."""

import argparse
import copy
import dataclasses
import json
import os
import sys
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import draccus
import numpy as np
import torch
import tqdm
from peft import PeftModel


CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))
PRUNER_DIR = CURRENT_DIR.parent
if str(PRUNER_DIR) not in sys.path:
    sys.path.insert(0, str(PRUNER_DIR))

from backend import load_backend_modules, resolve_backend_root
from config import (
    PyramidDropGenerateConfig,
    TASK_MAX_STEPS,
    apply_runtime_overrides,
    load_pyramiddrop_config_from_json_if_any,
    validate_config,
)
from initial_global_image import resolve_initial_global_image_dir, save_initial_global_image
from mug_eval import describe_eval_mug_spec, get_libero_env_with_optional_mug
from runtime_patch import apply_runtime_patches, get_episode_runtime_metrics, get_runtime_metrics, patch_loaded_model


backend = load_backend_modules()
base_eval = backend.base_eval
libero_utils = backend.libero_utils
robot_utils = backend.robot_utils
openvla_utils = backend.openvla_utils

WHITE_PATCH_AREA_RATIO = 0.01
WHITE_RGB = np.array([255, 255, 255], dtype=np.uint8)


def _parse_wrapper_args(argv):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--with-white-patch", "--eval-with-white-patch", dest="with_white_patch", action="store_true")
    parser.add_argument(
        "--without-white-patch",
        "--no-eval-with-white-patch",
        dest="without_white_patch",
        action="store_true",
    )
    parser.add_argument("--with-mug", "--eval-with-mug", dest="with_mug", action="store_true")
    parser.add_argument("--without-mug", "--no-eval-with-mug", dest="without_mug", action="store_true")
    parser.add_argument(
        "--save-initial-global-images",
        "--eval-save-initial-global-images",
        dest="save_initial_global_images",
        action="store_true",
    )
    parser.add_argument(
        "--no-save-initial-global-images",
        "--no-eval-save-initial-global-images",
        dest="no_save_initial_global_images",
        action="store_true",
    )
    args, remaining = parser.parse_known_args(argv[1:])

    if args.with_white_patch and args.without_white_patch:
        raise SystemExit("Cannot pass both `--with-white-patch` and `--without-white-patch`.")
    if args.with_mug and args.without_mug:
        raise SystemExit("Cannot pass both `--with-mug` and `--without-mug`.")
    if args.save_initial_global_images and args.no_save_initial_global_images:
        raise SystemExit(
            "Cannot pass both `--save-initial-global-images` and `--no-save-initial-global-images`."
        )

    return args, [argv[0]] + remaining


def _extract_cli_option_value(argv, option_names):
    args = argv[1:]
    i = 0
    while i < len(args):
        token = args[i]
        for name in option_names:
            if token.startswith(f"{name}="):
                return token.split("=", 1)[1]
            if token == name and i + 1 < len(args):
                return args[i + 1]
        i += 1
    return None


def _extract_pyramiddrop_config_json(argv):
    default_path = str(Path(__file__).resolve().parent / "configs" / "pyramiddrop_config.json")
    path = _extract_cli_option_value(argv, {"--pyramiddrop_config_json", "--pyramiddrop-config-json"})
    if isinstance(path, str) and path.strip():
        return path.strip()
    return default_path


def _resolve_custom_log_filepath(pyramiddrop_config_json, argv):
    cli_log_path = _extract_cli_option_value(argv, {"--custom_log_filepath", "--custom-log-filepath"})
    if isinstance(cli_log_path, str):
        stripped = cli_log_path.strip()
        if stripped and stripped.lower() not in {"none", "null"}:
            return str(Path(stripped).expanduser().resolve())

    try:
        with open(pyramiddrop_config_json, "r", encoding="utf-8") as f:
            data = json.load(f)
        log_path = data.get("custom_log_filepath")
        if isinstance(log_path, str) and log_path.strip():
            return str(Path(log_path).expanduser().resolve())
    except Exception:
        pass
    return None


def _add_top_left_white_patch(image: np.ndarray) -> np.ndarray:
    if image.ndim != 3 or image.shape[-1] != 3:
        return image

    height, width = int(image.shape[0]), int(image.shape[1])
    patch_size = int(round(np.sqrt(float(height * width) * WHITE_PATCH_AREA_RATIO)))
    patch_size = max(1, min(patch_size, height, width))

    patched = image.copy()
    patched[:patch_size, :patch_size, :] = WHITE_RGB
    return patched


def _prepare_observation(obs, resize_size):
    img = libero_utils.get_libero_image(obs)
    wrist_img = libero_utils.get_libero_wrist_image(obs)
    img_resized = openvla_utils.resize_image_for_policy(img, resize_size)
    wrist_img_resized = openvla_utils.resize_image_for_policy(wrist_img, resize_size)
    return {
        "full_image": img_resized,
        "wrist_image": wrist_img_resized,
        "state": np.concatenate(
            (obs["robot0_eef_pos"], libero_utils.quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])
        ),
    }


def _maybe_save_initial_global_image(
    cfg: PyramidDropGenerateConfig,
    obs,
    task_name: str,
    episode_idx: int,
    total_episode_idx: int,
    log_file=None,
):
    save_path = save_initial_global_image(
        cfg,
        libero_utils.get_libero_image(obs),
        task_name,
        episode_idx,
        total_episode_idx,
        default_note="pyramiddrop",
    )
    if save_path is not None:
        base_eval.log_message(f"Saved initial global image: {save_path}", log_file)


def _process_action(action, model_family):
    action = robot_utils.normalize_gripper_action(action, binarize=True)
    if model_family == "openvla":
        action = robot_utils.invert_gripper_action(action)
    return action


def _initialize_components(cfg: PyramidDropGenerateConfig):
    model = robot_utils.get_model(cfg)

    if getattr(cfg, "lora_weight_path", None) is not None:
        if cfg.load_in_8bit or cfg.load_in_4bit:
            print("[PyramidDrop] warning: merge_and_unload may not work with 8-bit/4-bit base model.")
        print(f"[PyramidDrop] Loading LoRA weights from {cfg.lora_weight_path}...")
        model = PeftModel.from_pretrained(model, cfg.lora_weight_path)
        print("[PyramidDrop] Merging LoRA weights into the base model...")
        model = model.merge_and_unload()
        print("[PyramidDrop] LoRA weights merged successfully.")

    patch_loaded_model(model, cfg)

    proprio_projector = (
        openvla_utils.get_proprio_projector(cfg, model.llm_dim, proprio_dim=8) if cfg.use_proprio else None
    )
    action_head = (
        openvla_utils.get_action_head(cfg, model.llm_dim) if (cfg.use_l1_regression or cfg.use_diffusion) else None
    )
    noisy_action_projector = (
        openvla_utils.get_noisy_action_projector(cfg, model.llm_dim) if cfg.use_diffusion else None
    )
    processor = openvla_utils.get_processor(cfg) if cfg.model_family == "openvla" else None
    return model, action_head, proprio_projector, noisy_action_projector, processor


def _setup_logging(cfg: PyramidDropGenerateConfig):
    if getattr(cfg, "custom_log_filepath", None):
        run_id = f"EVAL-{cfg.task_suite_name}-{cfg.model_family}-{robot_utils.DATE_TIME}"
        if cfg.run_id_note is not None:
            run_id += f"--{cfg.run_id_note}"
        local_log_filepath = str(cfg.custom_log_filepath)
        Path(local_log_filepath).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
        log_file = open(local_log_filepath, "w")
        base_eval.logger.info(f"Logging to local log file: {local_log_filepath}")
        if getattr(cfg, "use_wandb", False):
            import wandb

            wandb.init(entity=cfg.wandb_entity, project=cfg.wandb_project, name=run_id)
        return log_file, local_log_filepath, run_id
    return base_eval.setup_logging(cfg)


def _resolve_unnorm_key_if_needed(cfg: PyramidDropGenerateConfig, model) -> None:
    if not hasattr(model, "norm_stats"):
        return
    try:
        base_eval.check_unnorm_key(cfg, model)
    except AssertionError:
        desired = str(getattr(cfg, "task_suite_name", "")).strip()
        keys = list(getattr(model, "norm_stats", {}).keys())
        resolved = None
        if desired in keys:
            resolved = desired
        elif f"{desired}_no_noops" in keys:
            resolved = f"{desired}_no_noops"
        else:
            for key in keys:
                if desired and desired in key:
                    resolved = key
                    break
            if resolved is None and len(keys) == 1:
                resolved = keys[0]
        assert resolved is not None, f"Failed to resolve unnorm_key from norm_stats keys={keys}"
        cfg.unnorm_key = resolved


def _reapply_nondefault_runtime_overrides(
    cfg: PyramidDropGenerateConfig,
    original_cfg: PyramidDropGenerateConfig,
) -> None:
    defaults = PyramidDropGenerateConfig()
    for field in dataclasses.fields(PyramidDropGenerateConfig):
        field_name = field.name
        original_value = getattr(original_cfg, field_name)
        default_value = getattr(defaults, field_name)
        if original_value != default_value:
            setattr(cfg, field_name, original_value)


def _format_pyramiddrop_metrics(metrics):
    return (
        f"model_calls={int(metrics.get('model_calls', 0))}, "
        f"prune_events={int(metrics.get('prune_events', 0))}, "
        f"tokens_before={int(metrics.get('tokens_before', 0))}, "
        f"tokens_after={int(metrics.get('tokens_after', 0))}, "
        f"weighted_prune_rate={metrics.get('weighted_prune_rate', 0.0) * 100:.2f}%, "
        f"last_visual_tokens={int(metrics.get('last_visual_tokens', 0))}, "
        f"last_final_visual_tokens={int(metrics.get('last_final_visual_tokens', 0))}, "
        f"last_keep_ratio={metrics.get('last_keep_ratio', 1.0) * 100:.2f}%, "
        f"last_layers={metrics.get('last_layers', '')}, "
        f"last_ratios={metrics.get('last_ratios', '')}, "
        f"query_mode={metrics.get('last_query_mode', 'text_last')}, "
        f"fallback_calls={int(metrics.get('fallback_calls', 0))}"
    )


def _sync_cuda_if_needed() -> None:
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


def _maybe_set_episode_mug_pose(env, episode_record):
    if not hasattr(env, "set_episode_mug_pose"):
        return
    mug_position = None if episode_record is None else episode_record.get("mug_position")
    mug_orientation = None if episode_record is None else episode_record.get("mug_orientation")
    mug_xy = None if episode_record is None else episode_record.get("mug_xy")
    env.set_episode_mug_pose(mug_position=mug_position, mug_orientation=mug_orientation, mug_xy=mug_xy)


def _prepare_env_for_episode(cfg: PyramidDropGenerateConfig, env, initial_state=None):
    env.reset()
    obs = env.set_init_state(initial_state) if initial_state is not None else env.get_observation()
    for _ in range(cfg.num_steps_wait):
        obs, _, done, _ = env.step(libero_utils.get_libero_dummy_action(cfg.model_family))
        if done:
            raise RuntimeError("Environment terminated during warmup after initialization.")
    return obs


def _run_policy_episode(
    cfg: PyramidDropGenerateConfig,
    env,
    task_description: str,
    model,
    resize_size,
    processor,
    action_head,
    proprio_projector,
    noisy_action_projector,
    use_white_patch,
    initial_obs,
):
    if hasattr(model, "pyramiddrop_reset_episode"):
        model.pyramiddrop_reset_episode()

    action_queue = deque(maxlen=cfg.num_open_loop_steps)
    max_steps = TASK_MAX_STEPS[cfg.task_suite_name]
    success = False
    prediction_tracker = _ModelPredictionTracker()
    original_predict_action = _patch_model_predict_action(model, prediction_tracker)
    obs = initial_obs

    try:
        for _ in range(max_steps):
            observation = _prepare_observation(obs, resize_size)
            if use_white_patch:
                observation["full_image"] = _add_top_left_white_patch(observation["full_image"])
                observation["wrist_image"] = _add_top_left_white_patch(observation["wrist_image"])

            if len(action_queue) == 0:
                actions = robot_utils.get_action(
                    cfg,
                    model,
                    observation,
                    task_description,
                    processor=processor,
                    action_head=action_head,
                    proprio_projector=proprio_projector,
                    noisy_action_projector=noisy_action_projector,
                    use_film=cfg.use_film,
                )
                action_queue.extend(actions)

            action = _process_action(action_queue.popleft(), cfg.model_family)
            obs, _, done, _ = env.step(action.tolist())
            if done:
                success = True
                break
    finally:
        _restore_model_predict_action(model, original_predict_action)

    episode_metrics = get_episode_runtime_metrics(model)
    episode_metrics.update(
        {
            "prediction_calls": prediction_tracker.call_count,
            "prediction_time_total_sec": prediction_tracker.total_time_sec,
            "prediction_time_avg_sec": prediction_tracker.avg_time_sec,
        }
    )
    return success, episode_metrics


@draccus.wrap()
def eval_libero_pyramiddrop(cfg: PyramidDropGenerateConfig) -> float:
    apply_runtime_patches()
    original_cfg = copy.deepcopy(cfg)
    load_pyramiddrop_config_from_json_if_any(cfg)
    _reapply_nondefault_runtime_overrides(cfg, original_cfg)
    apply_runtime_overrides(cfg)
    validate_config(cfg)
    robot_utils.set_seed_everywhere(cfg.seed)

    model, action_head, proprio_projector, noisy_action_projector, processor = _initialize_components(cfg)
    if processor is not None:
        _resolve_unnorm_key_if_needed(cfg, model)
        try:
            print(f"[PyramidDrop] Resolved unnorm_key = {cfg.unnorm_key}")
            print(f"[PyramidDrop] Available norm_stats keys = {list(getattr(model, 'norm_stats', {}).keys())}")
        except Exception:
            pass

    resize_size = robot_utils.get_image_resize_size(cfg)
    log_file, _, _ = _setup_logging(cfg)

    from libero.libero import benchmark

    task_suite = benchmark.get_benchmark_dict()[cfg.task_suite_name]()
    use_white_patch = bool(getattr(cfg, "eval_with_white_patch", False))
    base_eval.log_message(f"Task suite: {cfg.task_suite_name}", log_file)
    base_eval.log_message(f"PyramidDrop backend root: {resolve_backend_root()}", log_file)
    base_eval.log_message(f"PyramidDrop config json: {cfg.pyramiddrop_config_json}", log_file)
    base_eval.log_message(
        (
            f"PyramidDrop params: enabled={cfg.pyramiddrop_enabled}, layers={cfg.pyramiddrop_layer_list}, "
            f"ratios={cfg.pyramiddrop_image_token_ratio_list}, query_mode={cfg.pyramiddrop_query_mode}, "
            f"min_keep_per_image={cfg.pyramiddrop_min_keep_tokens_per_image}, "
            f"force_attention_after_prune={cfg.pyramiddrop_force_attention_after_prune}"
        ),
        log_file,
    )
    base_eval.log_message(f"Eval with white patch: {use_white_patch}", log_file)
    base_eval.log_message(f"Eval with mug: {bool(getattr(cfg, 'eval_with_mug', False))}", log_file)
    if bool(getattr(cfg, "eval_with_mug", False)):
        base_eval.log_message(f"Eval mug spec: {describe_eval_mug_spec(cfg.task_suite_name, cfg)}", log_file)
    base_eval.log_message(
        f"Save initial global images: {bool(getattr(cfg, 'eval_save_initial_global_images', False))}",
        log_file,
    )
    if bool(getattr(cfg, "eval_save_initial_global_images", False)):
        base_eval.log_message(
            f"Initial global image dir: {resolve_initial_global_image_dir(cfg, 'pyramiddrop')}",
            log_file,
        )

    total_episodes, total_successes = 0, 0
    episode_prediction_totals_sec = []
    episode_prediction_avgs_sec = []
    episode_prediction_call_counts = []

    for task_id in tqdm.tqdm(range(task_suite.n_tasks)):
        task = task_suite.get_task(task_id)
        initial_states, all_initial_states = base_eval.load_initial_states(cfg, task_suite, task_id, log_file)
        env, task_description = get_libero_env_with_optional_mug(task, cfg.model_family, cfg.env_img_res, cfg)
        task_name = Path(task.bddl_file).stem

        try:
            for episode_idx in tqdm.tqdm(range(cfg.num_trials_per_task)):
                episode_record = None
                if cfg.initial_states_path == "DEFAULT":
                    initial_state = initial_states[episode_idx]
                else:
                    initial_states_task_key = task_description.replace(" ", "_")
                    episode_key = f"demo_{episode_idx}"
                    episode_record = all_initial_states[initial_states_task_key][episode_key]
                    if not episode_record["success"]:
                        base_eval.log_message(
                            f"Skipping task {task_id} episode {episode_idx} due to failed expert demo!",
                            log_file,
                        )
                        continue
                    initial_state = np.array(episode_record["initial_state"])

                _maybe_set_episode_mug_pose(env, episode_record)
                try:
                    initial_obs = _prepare_env_for_episode(cfg, env, initial_state)
                except Exception as exc:
                    base_eval.log_message(
                        f"Skipping task {task_id} episode {episode_idx} due to initialization failure: {exc}",
                        log_file,
                    )
                    continue

                _maybe_save_initial_global_image(
                    cfg,
                    initial_obs,
                    task_name,
                    episode_idx,
                    total_episodes + 1,
                    log_file,
                )
                base_eval.log_message(f"\nTask: {task_description}", log_file)
                base_eval.log_message(f"Starting episode {total_episodes + 1}...", log_file)

                success, episode_metrics = _run_policy_episode(
                    cfg,
                    env,
                    task_description,
                    model,
                    resize_size,
                    processor,
                    action_head,
                    proprio_projector,
                    noisy_action_projector,
                    use_white_patch,
                    initial_obs,
                )

                total_episodes += 1
                if success:
                    total_successes += 1
                episode_prediction_totals_sec.append(float(episode_metrics["prediction_time_total_sec"]))
                episode_prediction_avgs_sec.append(float(episode_metrics["prediction_time_avg_sec"]))
                episode_prediction_call_counts.append(int(episode_metrics["prediction_calls"]))

                base_eval.log_message(f"Success: {success}", log_file)
                base_eval.log_message(
                    f"PyramidDrop episode metrics: {_format_pyramiddrop_metrics(episode_metrics)}",
                    log_file,
                )
                base_eval.log_message(
                    f"Episode model prediction calls: {episode_metrics['prediction_calls']}",
                    log_file,
                )
                base_eval.log_message(
                    f"Episode total model prediction time (seconds): {episode_metrics['prediction_time_total_sec']:.6f}",
                    log_file,
                )
                base_eval.log_message(
                    f"Episode average model prediction time (seconds): {episode_metrics['prediction_time_avg_sec']:.6f}",
                    log_file,
                )
                base_eval.log_message(f"# episodes completed so far: {total_episodes}", log_file)
                base_eval.log_message(
                    f"# successes: {total_successes} ({(total_successes / total_episodes * 100):.1f}%)",
                    log_file,
                )
        finally:
            env.close()

    final_success_rate = float(total_successes) / float(total_episodes) if total_episodes > 0 else 0.0
    avg_episode_prediction_total_sec = (
        float(np.mean(episode_prediction_totals_sec)) if episode_prediction_totals_sec else 0.0
    )
    avg_episode_prediction_time_sec = float(np.mean(episode_prediction_avgs_sec)) if episode_prediction_avgs_sec else 0.0
    avg_episode_prediction_calls = (
        float(np.mean(episode_prediction_call_counts)) if episode_prediction_call_counts else 0.0
    )
    total_metrics = get_runtime_metrics(model)

    base_eval.log_message("Final results:", log_file)
    base_eval.log_message(f"Total episodes: {total_episodes}", log_file)
    base_eval.log_message(f"Total successes: {total_successes}", log_file)
    base_eval.log_message(
        f"Overall success rate: {final_success_rate:.4f} ({final_success_rate * 100:.1f}%)",
        log_file,
    )
    base_eval.log_message(
        f"Average total model prediction time per episode (seconds): {avg_episode_prediction_total_sec:.6f}",
        log_file,
    )
    base_eval.log_message(
        f"Average single model prediction time per episode (seconds): {avg_episode_prediction_time_sec:.6f}",
        log_file,
    )
    base_eval.log_message(
        f"Average model prediction calls per episode: {avg_episode_prediction_calls:.6f}",
        log_file,
    )
    base_eval.log_message(f"PyramidDrop totals: {_format_pyramiddrop_metrics(total_metrics)}", log_file)

    if log_file:
        log_file.close()
    return final_success_rate


if __name__ == "__main__":
    wrapper_args, remaining_argv = _parse_wrapper_args(sys.argv)
    sys.argv = remaining_argv
    pyramiddrop_config_json = _extract_pyramiddrop_config_json(remaining_argv)
    resolved_log_filepath = _resolve_custom_log_filepath(pyramiddrop_config_json, remaining_argv)

    if wrapper_args.with_white_patch:
        os.environ["PYRAMIDDROP_EVAL_WITH_WHITE_PATCH"] = "true"
    elif wrapper_args.without_white_patch:
        os.environ["PYRAMIDDROP_EVAL_WITH_WHITE_PATCH"] = "false"

    if wrapper_args.with_mug:
        os.environ["PYRAMIDDROP_EVAL_WITH_MUG"] = "true"
    elif wrapper_args.without_mug:
        os.environ["PYRAMIDDROP_EVAL_WITH_MUG"] = "false"

    if wrapper_args.save_initial_global_images:
        os.environ["PYRAMIDDROP_EVAL_SAVE_INITIAL_GLOBAL_IMAGES"] = "true"
    elif wrapper_args.no_save_initial_global_images:
        os.environ["PYRAMIDDROP_EVAL_SAVE_INITIAL_GLOBAL_IMAGES"] = "false"

    start_ts = time.time()
    start_dt = datetime.now()
    succeeded = False
    try:
        eval_libero_pyramiddrop()
        succeeded = True
    finally:
        end_ts = time.time()
        end_dt = datetime.now()
        elapsed_sec = end_ts - start_ts
        runtime_lines = [
            "",
            "[PyramidDrop] Runtime Summary",
            f"[PyramidDrop] Start time: {start_dt.isoformat(timespec='seconds')}",
            f"[PyramidDrop] End time: {end_dt.isoformat(timespec='seconds')}",
            f"[PyramidDrop] Elapsed time (seconds): {elapsed_sec:.3f}",
            f"[PyramidDrop] Elapsed time (minutes): {elapsed_sec / 60.0:.3f}",
            f"[PyramidDrop] Run status: {'success' if succeeded else 'failed'}",
        ]
        print("\n".join(runtime_lines))

        if resolved_log_filepath is not None:
            Path(resolved_log_filepath).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
            with open(resolved_log_filepath, "a", encoding="utf-8") as log_f:
                log_f.write("\n".join(runtime_lines) + "\n")

