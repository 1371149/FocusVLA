"""ADP LIBERO pruning evaluation entrypoint."""
import os
import sys
import time
from collections import deque
from pathlib import Path

import draccus
import numpy as np
import torch
import tqdm
from peft import PeftModel


CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from backend import load_backend_modules, resolve_backend_root
from config import (
    ADPGenerateConfig,
    TASK_MAX_STEPS,
    apply_runtime_overrides,
    load_qk_config_from_json_if_any,
    validate_config,
)
from debug import maybe_debug_first_step
from mug_eval import describe_eval_mug_spec, get_libero_env_with_optional_mug
from runtime_patch import apply_runtime_patches, patch_loaded_model
from strategy import OnlineStrategyController


backend = load_backend_modules()
base_eval = backend.base_eval
libero_utils = backend.libero_utils
robot_utils = backend.robot_utils
openvla_utils = backend.openvla_utils

WHITE_PATCH_AREA_RATIO = 0.01
WHITE_RGB = np.array([255, 255, 255], dtype=np.uint8)


def _parse_bool_env_override(name: str):
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


def _use_white_patch(cfg: ADPGenerateConfig) -> bool:
    env_override = _parse_bool_env_override("ADP_EVAL_WITH_WHITE_PATCH")
    if env_override is not None:
        return env_override
    return bool(getattr(cfg, "eval_with_white_patch", False))


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
    observation = {
        "full_image": img_resized,
        "wrist_image": wrist_img_resized,
        "state": np.concatenate(
            (obs["robot0_eef_pos"], libero_utils.quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])
        ),
    }
    return observation, img


def _process_action(action, model_family):
    action = robot_utils.normalize_gripper_action(action, binarize=True)
    if model_family == "openvla":
        action = robot_utils.invert_gripper_action(action)
    return action


def _initialize_components(cfg: ADPGenerateConfig):
    model = robot_utils.get_model(cfg)
    patch_loaded_model(model)
    if getattr(cfg, "lora_weight_path", None) is not None:
        if cfg.load_in_8bit or cfg.load_in_4bit:
            print("[ADP] warning: merge_and_unload may not work with 8-bit/4-bit base model.")
        print(f"[ADP] Loading LoRA weights from {cfg.lora_weight_path}...")
        model = PeftModel.from_pretrained(model, cfg.lora_weight_path)
        print("[ADP] Merging LoRA weights into the base model...")
        model = model.merge_and_unload()
        print("[ADP] LoRA weights merged successfully.")
        patch_loaded_model(model)

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


def _setup_logging(cfg: ADPGenerateConfig):
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

            wandb.init(
                entity=cfg.wandb_entity,
                project=cfg.wandb_project,
                name=run_id,
            )
        return log_file, local_log_filepath, run_id
    return base_eval.setup_logging(cfg)


def _resolve_unnorm_key_if_needed(cfg: ADPGenerateConfig, model) -> None:
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


def _run_one_episode(
    cfg: ADPGenerateConfig,
    env,
    task_description: str,
    model,
    resize_size,
    processor,
    action_head,
    proprio_projector,
    noisy_action_projector,
    use_white_patch,
    initial_state=None,
    log_file=None,
):
    env.reset()
    obs = env.set_init_state(initial_state) if initial_state is not None else env.get_observation()

    action_queue = deque(maxlen=cfg.num_open_loop_steps)
    t = 0
    max_steps = TASK_MAX_STEPS[cfg.task_suite_name]
    strategy = OnlineStrategyController(cfg, libero_utils.quat2axisangle) if bool(
        getattr(cfg, "use_dynamic_visual_strategy", False)
    ) else None

    success = False
    prediction_tracker = _ModelPredictionTracker()
    original_predict_action = _patch_model_predict_action(model, prediction_tracker)

    try:
        while t < max_steps + cfg.num_steps_wait:
            if t < cfg.num_steps_wait:
                obs, reward, done, info = env.step(libero_utils.get_libero_dummy_action(cfg.model_family))
                del reward, done, info
                t += 1
                continue

            observation, img = _prepare_observation(obs, resize_size)
            if use_white_patch:
                observation["full_image"] = _add_top_left_white_patch(observation["full_image"])
                observation["wrist_image"] = _add_top_left_white_patch(observation["wrist_image"])
            del img

            if len(action_queue) == 0:
                if strategy is not None:
                    if strategy.curr_window_start_frame is None:
                        try:
                            eef_pos_now = np.array(obs.get("robot0_eef_pos", [0, 0, 0]), dtype=float)
                            eef_quat_now = np.array(obs.get("robot0_eef_quat", [0, 0, 0, 1]), dtype=float)
                            strategy.observe_step(eef_pos_now, eef_quat_now)
                        except Exception:
                            pass
                    state = int(strategy.get_current_state())
                    try:
                        cfg.qk_keep_enabled = state == 1
                    except Exception:
                        pass
                else:
                    state = 1 if bool(getattr(cfg, "qk_keep_enabled", False)) else 0
                    del state
                if bool(getattr(cfg, "qk_debug", False)) and t == cfg.num_steps_wait:
                    maybe_debug_first_step(
                        cfg,
                        t,
                        action_queue,
                        model,
                        observation,
                        task_description,
                        robot_utils.get_action,
                        processor=processor,
                        action_head=action_head,
                        proprio_projector=proprio_projector,
                        noisy_action_projector=noisy_action_projector,
                    )
                else:
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

            action = action_queue.popleft()
            action = _process_action(action, cfg.model_family)
            obs, reward, done, info = env.step(action.tolist())
            del reward, info

            if strategy is not None:
                try:
                    eef_pos_now = np.array(obs.get("robot0_eef_pos", [0, 0, 0]), dtype=float)
                    eef_quat_now = np.array(obs.get("robot0_eef_quat", [0, 0, 0, 1]), dtype=float)
                    strategy.observe_step(eef_pos_now, eef_quat_now)
                except Exception:
                    pass
            if done:
                success = True
                break
            t += 1

            if strategy is not None and len(action_queue) == 0 and strategy.curr_window_start_frame is not None:
                strategy.close_window_and_decide_next()
    finally:
        _restore_model_predict_action(model, original_predict_action)

    episode_stats = {
        "prediction_calls": prediction_tracker.call_count,
        "prediction_time_total_sec": prediction_tracker.total_time_sec,
        "prediction_time_avg_sec": prediction_tracker.avg_time_sec,
    }
    return success, episode_stats


@draccus.wrap()
def eval_libero_adp(cfg: ADPGenerateConfig) -> float:
    apply_runtime_patches()
    load_qk_config_from_json_if_any(cfg)
    apply_runtime_overrides(cfg)
    validate_config(cfg)
    robot_utils.set_seed_everywhere(cfg.seed)

    model, action_head, proprio_projector, noisy_action_projector, processor = _initialize_components(cfg)
    if processor is not None:
        _resolve_unnorm_key_if_needed(cfg, model)
        try:
            print(f"[ADP] Resolved unnorm_key = {cfg.unnorm_key}")
            print(f"[ADP] Available norm_stats keys = {list(getattr(model, 'norm_stats', {}).keys())}")
        except Exception:
            pass

    resize_size = robot_utils.get_image_resize_size(cfg)
    log_file, _, _ = _setup_logging(cfg)

    from libero.libero import benchmark

    task_suite = benchmark.get_benchmark_dict()[cfg.task_suite_name]()
    use_white_patch = _use_white_patch(cfg)
    base_eval.log_message(f"Task suite: {cfg.task_suite_name}", log_file)
    base_eval.log_message(f"ADP backend root: {resolve_backend_root()}", log_file)
    base_eval.log_message(f"ADP config json: {cfg.qk_config_json}", log_file)
    base_eval.log_message(f"Eval with white patch: {use_white_patch}", log_file)
    base_eval.log_message(f"Eval with mug: {bool(getattr(cfg, 'eval_with_mug', False))}", log_file)
    if bool(getattr(cfg, "eval_with_mug", False)):
        base_eval.log_message(f"Eval mug spec: {describe_eval_mug_spec(cfg.task_suite_name, cfg)}", log_file)

    total_episodes, total_successes = 0, 0
    episode_prediction_totals_sec = []
    episode_prediction_avgs_sec = []
    episode_prediction_call_counts = []
    for task_id in tqdm.tqdm(range(task_suite.n_tasks)):
        task = task_suite.get_task(task_id)
        initial_states, all_initial_states = base_eval.load_initial_states(cfg, task_suite, task_id, log_file)
        env, task_description = get_libero_env_with_optional_mug(task, cfg.model_family, cfg.env_img_res, cfg)

        try:
            for episode_idx in tqdm.tqdm(range(cfg.num_trials_per_task)):
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

                base_eval.log_message(f"\nTask: {task_description}", log_file)
                base_eval.log_message(f"Starting episode {total_episodes + 1}...", log_file)

                success, episode_stats = _run_one_episode(
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
                    initial_state,
                    log_file,
                )

                total_episodes += 1
                if success:
                    total_successes += 1
                episode_prediction_totals_sec.append(float(episode_stats["prediction_time_total_sec"]))
                episode_prediction_avgs_sec.append(float(episode_stats["prediction_time_avg_sec"]))
                episode_prediction_call_counts.append(int(episode_stats["prediction_calls"]))

                base_eval.log_message(f"Success: {success}", log_file)
                base_eval.log_message(
                    f"Episode model prediction calls: {episode_stats['prediction_calls']}",
                    log_file,
                )
                base_eval.log_message(
                    f"Episode total model prediction time (seconds): {episode_stats['prediction_time_total_sec']:.6f}",
                    log_file,
                )
                base_eval.log_message(
                    f"Episode average model prediction time (seconds): {episode_stats['prediction_time_avg_sec']:.6f}",
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

    if log_file:
        log_file.close()
    return final_success_rate


if __name__ == "__main__":
    eval_libero_adp()
