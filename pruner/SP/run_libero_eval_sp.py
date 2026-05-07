"""SP-VLA-style LIBERO evaluation entrypoint for openvla-oft."""

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
PRUNER_DIR = CURRENT_DIR.parent
if str(PRUNER_DIR) not in sys.path:
    sys.path.insert(0, str(PRUNER_DIR))

from backend import load_backend_modules, resolve_backend_root
from config import SPGenerateConfig, TASK_MAX_STEPS, load_sp_config_from_json_if_any, validate_config
from debug import get_debug_output_dir
from initial_global_image import resolve_initial_global_image_dir, save_initial_global_image
from mug_eval import describe_eval_mug_spec, get_libero_env_with_optional_mug
from mug_pose_stats.runtime import (
    get_pose_retry_limit,
    get_task_pose_candidates,
    resolve_catalog_path,
    select_pose_candidate,
)
from runtime_patch import apply_runtime_patches, get_episode_runtime_metrics, get_runtime_metrics, patch_loaded_model


backend = load_backend_modules()
base_eval = backend.base_eval
libero_utils = backend.libero_utils
robot_utils = backend.robot_utils
openvla_utils = backend.openvla_utils

WHITE_PATCH_AREA_RATIO = 0.01
WHITE_RGB = np.array([255, 255, 255], dtype=np.uint8)


def _add_top_left_white_patch(image: np.ndarray) -> np.ndarray:
    if image.ndim != 3 or image.shape[-1] != 3:
        return image

    height, width = int(image.shape[0]), int(image.shape[1])
    patch_size = int(round(np.sqrt(float(height * width) * WHITE_PATCH_AREA_RATIO)))
    patch_size = max(1, min(patch_size, height, width))

    patched = image.copy()
    patched[:patch_size, :patch_size, :] = WHITE_RGB
    return patched


def _prepare_observation(obs, resize_size, use_white_patch: bool = False):
    img = libero_utils.get_libero_image(obs)
    wrist_img = libero_utils.get_libero_wrist_image(obs)
    img_resized = openvla_utils.resize_image_for_policy(img, resize_size)
    wrist_img_resized = openvla_utils.resize_image_for_policy(wrist_img, resize_size)
    if use_white_patch:
        img_resized = _add_top_left_white_patch(img_resized)
        wrist_img_resized = _add_top_left_white_patch(wrist_img_resized)
    observation = {
        "full_image": img_resized,
        "wrist_image": wrist_img_resized,
        "state": np.concatenate(
            (obs["robot0_eef_pos"], libero_utils.quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])
        ),
    }
    return observation


def _maybe_save_initial_global_image(
    cfg: SPGenerateConfig,
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
        default_note="sp_vla",
    )
    if save_path is not None:
        base_eval.log_message(f"Saved initial global image: {save_path}", log_file)


def _process_action(action, model_family):
    action = robot_utils.normalize_gripper_action(action, binarize=True)
    if model_family == "openvla":
        action = robot_utils.invert_gripper_action(action)
    return action


def _initialize_components(cfg: SPGenerateConfig):
    model = robot_utils.get_model(cfg)

    if getattr(cfg, "lora_weight_path", None) is not None:
        if cfg.load_in_8bit or cfg.load_in_4bit:
            print("[SP] warning: merge_and_unload may not work with 8-bit/4-bit base model.")
        print(f"[SP] Loading LoRA weights from {cfg.lora_weight_path}...")
        model = PeftModel.from_pretrained(model, cfg.lora_weight_path)
        print("[SP] Merging LoRA weights into the base model...")
        model = model.merge_and_unload()
        print("[SP] LoRA weights merged successfully.")

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


def _setup_logging(cfg: SPGenerateConfig):
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


def _resolve_unnorm_key_if_needed(cfg: SPGenerateConfig, model) -> None:
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


def _metrics_delta(before, after):
    delta = {}
    for key, value in after.items():
        if isinstance(value, (int, float)) and isinstance(before.get(key), (int, float)):
            delta[key] = value - before[key]
    tokens_before = delta.get("tokens_before", 0)
    tokens_after = delta.get("tokens_after", 0)
    delta["weighted_prune_rate"] = 0.0 if tokens_before <= 0 else 1.0 - (tokens_after / float(tokens_before))
    delta["last_prediction_source"] = after.get("last_prediction_source", "generate")
    return delta


def _format_sp_metrics(metrics):
    return (
        f"generate_calls={int(metrics.get('model_generate_calls', 0))}, "
        f"skip_calls={int(metrics.get('model_skip_calls', 0))}, "
        f"executed_generate_actions={int(metrics.get('executed_generate_actions', 0))}, "
        f"executed_fit_actions={int(metrics.get('executed_fit_actions', 0))}, "
        f"weighted_prune_rate={metrics.get('weighted_prune_rate', 0.0) * 100:.2f}%, "
        f"max_prune_rate={metrics.get('max_prune_rate', 0.0) * 100:.2f}%"
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


def _run_one_episode(
    cfg: SPGenerateConfig,
    env,
    task_id: int,
    episode_idx: int,
    task_description: str,
    model,
    resize_size,
    processor,
    action_head,
    proprio_projector,
    noisy_action_projector,
    initial_state=None,
):
    initial_obs = _prepare_env_for_episode(cfg, env, initial_state)
    return _run_policy_episode(
        cfg,
        env,
        task_id,
        episode_idx,
        task_description,
        model,
        resize_size,
        processor,
        action_head,
        proprio_projector,
        noisy_action_projector,
        initial_obs,
    )


def _run_policy_episode(
    cfg: SPGenerateConfig,
    env,
    task_id: int,
    episode_idx: int,
    task_description: str,
    model,
    resize_size,
    processor,
    action_head,
    proprio_projector,
    noisy_action_projector,
    initial_obs,
):
    target = model.module if hasattr(model, "module") else model
    target._sp_current_task_id = int(task_id)
    target._sp_current_episode_idx = int(episode_idx)
    target._sp_current_task_label = task_description

    if hasattr(model, "sp_reset_episode"):
        model.sp_reset_episode()

    action_queue = deque(maxlen=cfg.num_open_loop_steps)
    action_source_queue = deque(maxlen=cfg.num_open_loop_steps)
    max_steps = TASK_MAX_STEPS[cfg.task_suite_name]
    use_white_patch = bool(getattr(cfg, "eval_with_white_patch", False))

    success = False
    prediction_tracker = _ModelPredictionTracker()
    original_predict_action = _patch_model_predict_action(model, prediction_tracker)
    obs = initial_obs

    try:
        for _ in range(max_steps):
            observation = _prepare_observation(obs, resize_size, use_white_patch=use_white_patch)

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
                source = getattr(model, "_sp_last_prediction_source", "generate")
                action_queue.extend(actions)
                action_source_queue.extend([source] * len(actions))

            raw_action = np.asarray(action_queue.popleft(), dtype=np.float32)
            action_source = (
                action_source_queue.popleft()
                if action_source_queue
                else getattr(model, "_sp_last_prediction_source", "generate")
            )
            env_action = _process_action(raw_action.copy(), cfg.model_family)
            obs, _, done, _ = env.step(env_action.tolist())

            if hasattr(model, "sp_record_executed_action"):
                model.sp_record_executed_action(raw_action, source=action_source)

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


def _maybe_set_episode_mug_pose(env, episode_record):
    if not hasattr(env, "set_episode_mug_pose"):
        return
    mug_position = None if episode_record is None else episode_record.get("mug_position")
    mug_orientation = None if episode_record is None else episode_record.get("mug_orientation")
    mug_xy = None if episode_record is None else episode_record.get("mug_xy")
    env.set_episode_mug_pose(mug_position=mug_position, mug_orientation=mug_orientation, mug_xy=mug_xy)


def _read_hdf5_value(group, key):
    if key not in group:
        return None
    value = group[key][()]
    return value.tolist() if hasattr(value, "tolist") else value


def _load_generated_mug_episode_state(cfg: SPGenerateConfig, task, episode_idx: int, log_file=None):
    if not bool(getattr(cfg, "eval_with_mug", False)):
        return None, None

    data_dir = str(getattr(cfg, "eval_mug_generated_data_dir", "") or "").strip()
    if not data_dir:
        return None, None

    try:
        import h5py
    except Exception as exc:
        raise RuntimeError(
            "eval_mug_generated_data_dir is set, but h5py cannot be imported. "
            "Please run evaluation in the same environment used to generate the HDF5 demos."
        ) from exc

    task_name = Path(task.bddl_file).stem
    hdf5_path = Path(data_dir).expanduser().resolve() / f"{task_name}_demo.hdf5"
    demo_key = f"demo_{episode_idx}"
    if not hdf5_path.exists():
        raise FileNotFoundError(f"Generated mug HDF5 not found for task `{task_name}`: {hdf5_path}")

    with h5py.File(hdf5_path, "r") as hdf5_file:
        if "data" not in hdf5_file or demo_key not in hdf5_file["data"]:
            available = list(hdf5_file.get("data", {}).keys()) if "data" in hdf5_file else []
            raise KeyError(
                f"Generated mug HDF5 `{hdf5_path}` does not contain `{demo_key}`. "
                f"Available demos: {available[:10]}"
            )

        demo = hdf5_file["data"][demo_key]
        if "states" not in demo or len(demo["states"]) == 0:
            raise KeyError(f"Generated mug HDF5 `{hdf5_path}` `{demo_key}` has no states.")

        initial_state = np.asarray(demo["states"][0])
        episode_record = {"success": True, "source": "generated_mug_hdf5", "hdf5_path": str(hdf5_path)}
        if "metadata" in demo:
            metadata = demo["metadata"]
            mug_position = _read_hdf5_value(metadata, "mug_position")
            mug_orientation = _read_hdf5_value(metadata, "mug_orientation")
            if mug_position is not None:
                episode_record["mug_position"] = mug_position
                episode_record["mug_xy"] = mug_position[:2]
            if mug_orientation is not None:
                episode_record["mug_orientation"] = mug_orientation

    base_eval.log_message(
        f"Using generated mug initial state: {hdf5_path.name}/{demo_key}",
        log_file,
    )
    return initial_state, episode_record


def _should_use_catalog_mug_poses(cfg: SPGenerateConfig) -> bool:
    if not bool(getattr(cfg, "eval_with_mug", False)):
        return False
    if str(getattr(cfg, "eval_mug_placement_presets_path", "") or "").strip():
        return False
    return not str(getattr(cfg, "eval_mug_generated_data_dir", "") or "").strip()


def _load_catalog_mug_pose_candidates(cfg: SPGenerateConfig, task, log_file=None) -> list[dict]:
    if not _should_use_catalog_mug_poses(cfg):
        return []

    task_name = Path(task.bddl_file).stem
    catalog_path = resolve_catalog_path(getattr(cfg, "eval_mug_pose_catalog_path", ""))
    candidates = get_task_pose_candidates(catalog_path, cfg.task_suite_name, task_name)
    base_eval.log_message(
        f"Saved mug pose candidates for {cfg.task_suite_name}/{task_name}: {len(candidates)} from {catalog_path}",
        log_file,
    )
    return candidates


def _prepare_env_for_episode(cfg: SPGenerateConfig, env, initial_state=None):
    env.reset()
    obs = env.set_init_state(initial_state) if initial_state is not None else env.get_observation()
    for _ in range(cfg.num_steps_wait):
        obs, _, done, _ = env.step(libero_utils.get_libero_dummy_action(cfg.model_family))
        if done:
            raise RuntimeError("Environment terminated during warmup after mug initialization.")
    return obs


@draccus.wrap()
def eval_libero_sp(cfg: SPGenerateConfig) -> float:
    apply_runtime_patches()
    load_sp_config_from_json_if_any(cfg)
    validate_config(cfg)
    robot_utils.set_seed_everywhere(cfg.seed)

    model, action_head, proprio_projector, noisy_action_projector, processor = _initialize_components(cfg)
    if processor is not None:
        _resolve_unnorm_key_if_needed(cfg, model)
        try:
            print(f"[SP] Resolved unnorm_key = {cfg.unnorm_key}")
            print(f"[SP] Available norm_stats keys = {list(getattr(model, 'norm_stats', {}).keys())}")
        except Exception:
            pass

    resize_size = robot_utils.get_image_resize_size(cfg)
    log_file, _, _ = _setup_logging(cfg)

    from libero.libero import benchmark

    task_suite = benchmark.get_benchmark_dict()[cfg.task_suite_name]()
    base_eval.log_message(f"Task suite: {cfg.task_suite_name}", log_file)
    base_eval.log_message(f"SP backend root: {resolve_backend_root()}", log_file)
    base_eval.log_message(f"SP config json: {cfg.sp_config_json}", log_file)
    base_eval.log_message(
        (
            f"SP params: pruning={cfg.sp_enable_token_pruning}, skipping={cfg.sp_enable_step_skipping}, "
            f"profile={cfg.sp_profile}, history={cfg.sp_history_size}, skip_window={cfg.sp_skip_window}, "
            f"z_xy_rate_skip={cfg.sp_z_xy_rate_skip}, z_max_skip={cfg.sp_z_max_skip}, "
            f"z_thre_prune={cfg.sp_z_thre_prune}, z_min_prune={cfg.sp_z_min_prune}, "
            f"max_prune_rate={cfg.sp_max_prune_rate}, fixed_keep_ratio={cfg.sp_fixed_keep_ratio}"
        ),
        log_file,
    )
    base_eval.log_message(
        f"SP debug image save: enabled={cfg.sp_save_first_pruned_image}, dir={get_debug_output_dir(cfg)}",
        log_file,
    )
    base_eval.log_message(f"Eval with white patch: {bool(getattr(cfg, 'eval_with_white_patch', False))}", log_file)
    base_eval.log_message(f"Eval with mug: {bool(getattr(cfg, 'eval_with_mug', False))}", log_file)
    base_eval.log_message(
        f"Save initial global images: {bool(getattr(cfg, 'eval_save_initial_global_images', False))}",
        log_file,
    )
    if bool(getattr(cfg, "eval_save_initial_global_images", False)):
        base_eval.log_message(f"Initial global image dir: {resolve_initial_global_image_dir(cfg, 'sp_vla')}", log_file)
    if bool(getattr(cfg, "eval_with_mug", False)) and str(getattr(cfg, "eval_mug_generated_data_dir", "") or "").strip():
        base_eval.log_message(
            f"Eval mug generated data dir: {Path(cfg.eval_mug_generated_data_dir).expanduser().resolve()}",
            log_file,
        )
    elif bool(getattr(cfg, "eval_with_mug", False)) and str(getattr(cfg, "eval_mug_placement_presets_path", "") or "").strip():
        base_eval.log_message(f"Eval mug spec: {describe_eval_mug_spec(cfg.task_suite_name, cfg)}", log_file)
    elif _should_use_catalog_mug_poses(cfg):
        base_eval.log_message(
            f"Eval mug pose catalog: {resolve_catalog_path(getattr(cfg, 'eval_mug_pose_catalog_path', ''))}",
            log_file,
        )
        base_eval.log_message(
            f"Eval mug max init retries: {getattr(cfg, 'eval_mug_max_init_retries', 0)} (0 means try all saved poses)",
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
        catalog_candidates = _load_catalog_mug_pose_candidates(cfg, task, log_file)
        task_name = Path(task.bddl_file).stem

        try:
            for episode_idx in tqdm.tqdm(range(cfg.num_trials_per_task)):
                initial_state, episode_record = _load_generated_mug_episode_state(cfg, task, episode_idx, log_file)
                if initial_state is None:
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

                initial_obs = None
                if _should_use_catalog_mug_poses(cfg):
                    if not catalog_candidates:
                        base_eval.log_message(
                            (
                                f"Skipping task {task_id} episode {episode_idx} because no saved mug poses were found "
                                f"for `{task_name}` in `{resolve_catalog_path(getattr(cfg, 'eval_mug_pose_catalog_path', ''))}`."
                            ),
                            log_file,
                        )
                        continue

                    retry_limit = get_pose_retry_limit(
                        len(catalog_candidates),
                        getattr(cfg, "eval_mug_max_init_retries", 0),
                    )
                    last_error = None
                    selected_pose_index = None
                    selected_pose = None
                    for retry_idx in range(retry_limit):
                        selected_pose_index, selected_pose = select_pose_candidate(catalog_candidates, episode_idx, retry_idx)
                        _maybe_set_episode_mug_pose(env, selected_pose)
                        try:
                            initial_obs = _prepare_env_for_episode(cfg, env, initial_state)
                            break
                        except Exception as exc:
                            last_error = exc
                            base_eval.log_message(
                                (
                                    f"Mug init failed for {task_name} episode {episode_idx} "
                                    f"retry {retry_idx + 1}/{retry_limit} with candidate {selected_pose_index}: {exc}"
                                ),
                                log_file,
                            )

                    if initial_obs is None:
                        base_eval.log_message(
                            (
                                f"Skipping task {task_id} episode {episode_idx} after {retry_limit} mug init failures "
                                f"for `{task_name}`. Last error: {last_error}"
                            ),
                            log_file,
                        )
                        continue

                    episode_record = selected_pose
                    base_eval.log_message(
                        (
                            f"Using saved mug pose candidate {selected_pose_index} for {task_name} episode {episode_idx}: "
                            f"source_demo={selected_pose.get('source_demo_key')}, attempt={selected_pose.get('attempt_key')}"
                        ),
                        log_file,
                    )
                else:
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
                    task_id,
                    episode_idx,
                    task_description,
                    model,
                    resize_size,
                    processor,
                    action_head,
                    proprio_projector,
                    noisy_action_projector,
                    initial_obs,
                )

                total_episodes += 1
                if success:
                    total_successes += 1
                episode_prediction_totals_sec.append(float(episode_metrics["prediction_time_total_sec"]))
                episode_prediction_avgs_sec.append(float(episode_metrics["prediction_time_avg_sec"]))
                episode_prediction_call_counts.append(int(episode_metrics["prediction_calls"]))

                base_eval.log_message(f"Success: {success}", log_file)
                base_eval.log_message(f"SP episode metrics: {_format_sp_metrics(episode_metrics)}", log_file)
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
    base_eval.log_message(f"SP totals: {_format_sp_metrics(total_metrics)}", log_file)

    if log_file:
        log_file.close()
    return final_success_rate


if __name__ == "__main__":
    eval_libero_sp()
