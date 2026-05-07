"""Optional blue mug injection for ADP LIBERO evaluation."""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path

import numpy as np
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv


MUG_OBJECT_NAME_PREFIX = "blue_coffee_mug"
MUG_OBJECT_TYPE = "blue_coffee_mug"
MUG_REGION_NAME = "mug_visible_region"
DEFAULT_PRESETS_PATH = Path(__file__).resolve().parents[1] / "mug_placement_presets.json"


class MugEvalEnvWrapper:
    def __init__(self, original_env, mug_env, suite_name=None, task_name=None, cfg=None):
        self._env = mug_env
        self._old_layout = _get_joint_ranges(original_env)
        self._new_layout = _get_joint_ranges(mug_env)
        self._old_nq = int(original_env.sim.model.nq)
        self._old_nv = int(original_env.sim.model.nv)
        self._new_nq = int(mug_env.sim.model.nq)
        self._suite_name = str(suite_name) if suite_name is not None else None
        self._task_name = str(task_name) if task_name is not None else None
        self._preset_ranges = _get_task_ranges(self._suite_name, self._task_name, cfg)
        self._rng = np.random.default_rng(0)
        self._episode_mug_position = None
        self._episode_mug_orientation = None
        self._episode_mug_xy = None
        original_env.close()

    def seed(self, seed):
        self._rng = np.random.default_rng(int(seed))
        return self._env.seed(seed)

    def reset(self):
        return self._env.reset()

    def set_episode_mug_pose(self, mug_position=None, mug_orientation=None, mug_xy=None):
        self._episode_mug_position = None if mug_position is None else np.asarray(mug_position, dtype=np.float32)
        self._episode_mug_orientation = None if mug_orientation is None else np.asarray(mug_orientation, dtype=np.float32)
        if mug_xy is not None:
            self._episode_mug_xy = np.asarray(mug_xy, dtype=np.float32)
        elif self._episode_mug_position is not None:
            self._episode_mug_xy = self._episode_mug_position[:2].astype(np.float32)
        else:
            self._episode_mug_xy = None

    def set_init_state(self, original_state):
        original_state = np.asarray(original_state)
        expected_mug_state_size = 1 + int(self._env.sim.model.nq) + int(self._env.sim.model.nv)
        if original_state.size == expected_mug_state_size:
            return self._env.set_init_state(original_state)

        mug_xy = self._episode_mug_xy
        if mug_xy is None:
            mug_xy = _sample_mug_xy(self._preset_ranges, self._rng)

        merged_state = _merge_original_state_with_mug_layout(
            self._env,
            self._old_layout,
            self._new_layout,
            self._old_nq,
            self._old_nv,
            self._new_nq,
            original_state,
            mug_xy=mug_xy,
            mug_position=self._episode_mug_position,
            mug_orientation=self._episode_mug_orientation,
        )
        return self._env.set_init_state(merged_state)

    def close(self):
        return self._env.close()

    def __getattr__(self, name):
        return getattr(self._env, name)


def resolve_eval_mug_spec(suite_name: str) -> dict:
    if str(suite_name) not in {"libero_10", "libero_spatial", "libero_object", "libero_goal"}:
        raise NotImplementedError(f"eval_with_mug is not implemented for task suite: {suite_name}")
    return {
        "object_name_prefix": MUG_OBJECT_NAME_PREFIX,
        "object_type": MUG_OBJECT_TYPE,
        "region_name": MUG_REGION_NAME,
    }


def describe_eval_mug_spec(suite_name: str, cfg=None) -> str:
    spec = resolve_eval_mug_spec(suite_name)
    presets_path = _resolve_presets_path(cfg) if cfg is not None else DEFAULT_PRESETS_PATH
    return (
        f"object_type={spec['object_type']}, region={spec['region_name']}, "
        f"placement_presets={presets_path}"
    )


def get_libero_env_with_optional_mug(task, model_family, resolution, cfg):
    del model_family
    task_description = task.language
    task_bddl_file = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    if not bool(getattr(cfg, "eval_with_mug", False)):
        env = OffScreenRenderEnv(
            bddl_file_name=str(task_bddl_file),
            camera_heights=resolution,
            camera_widths=resolution,
        )
        env.seed(0)
        return env, task_description

    modified_bddl = _ensure_modified_bddl(task_bddl_file, cfg)
    original_env = OffScreenRenderEnv(
        bddl_file_name=str(task_bddl_file),
        camera_heights=resolution,
        camera_widths=resolution,
    )
    mug_env = OffScreenRenderEnv(
        bddl_file_name=str(modified_bddl),
        camera_heights=resolution,
        camera_widths=resolution,
    )
    original_env.seed(0)
    mug_env.seed(0)
    return (
        MugEvalEnvWrapper(
            original_env,
            mug_env,
            suite_name=cfg.task_suite_name,
            task_name=task_bddl_file.stem,
            cfg=cfg,
        ),
        task_description,
    )


def _ensure_modified_bddl(task_bddl_file: Path, cfg) -> Path:
    cache_root = Path(cfg.eval_mug_cache_dir).expanduser().resolve()
    suite_name = str(cfg.task_suite_name)
    task_name = task_bddl_file.stem
    target_dir = cache_root / suite_name / task_bddl_file.parent.name
    target_dir.mkdir(parents=True, exist_ok=True)
    target_path = target_dir / task_bddl_file.name

    should_rewrite = True
    if target_path.exists():
        cached_text = target_path.read_text(encoding="utf-8")
        should_rewrite = not _bddl_matches_expected_mug_spec(cached_text, suite_name, task_name, cfg)

    if should_rewrite:
        source_text = task_bddl_file.read_text(encoding="utf-8")
        modified_text = _create_modified_bddl_text(source_text, suite_name, task_name, cfg)
        target_path.write_text(modified_text, encoding="utf-8")

    return target_path


def _create_modified_bddl_text(text: str, suite_name: str, task_name: str, cfg) -> str:
    resolve_eval_mug_spec(suite_name)
    target = _get_task_target_surface(suite_name, task_name, cfg)
    text = _inject_region_block(text, _region_block_from_presets(suite_name, task_name, cfg))
    object_name = _choose_mug_object_name(text)
    text = _inject_object_block(text, object_name)
    text = _inject_obj_of_interest_block(text, object_name)
    text = _inject_init_block(text, target, object_name)
    return text


def _bddl_matches_expected_mug_spec(text: str, suite_name: str, task_name: str, cfg) -> bool:
    if re.search(rf"\b[A-Za-z0-9_]+\s*-\s*{re.escape(MUG_OBJECT_TYPE)}\b", text) is None:
        return False
    if _region_block_from_presets(suite_name, task_name, cfg) not in text:
        return False

    target = _get_task_target_surface(suite_name, task_name, cfg)
    init_pattern = re.compile(
        rf"\(On\s+[A-Za-z0-9_]+\s+{re.escape(target + '_' + MUG_REGION_NAME)}\)"
    )
    return init_pattern.search(text) is not None


def _inject_region_block(text: str, region_block: str) -> str:
    if MUG_REGION_NAME in text:
        return text
    marker = "    (:regions\n"
    if marker not in text:
        raise ValueError("Failed to locate region insertion point in BDDL.")
    return text.replace(marker, marker + region_block, 1)


def _choose_mug_object_name(text: str) -> str:
    existing_names = set(re.findall(rf"\b([A-Za-z0-9_]+)\s*-\s*{re.escape(MUG_OBJECT_TYPE)}\b", text))
    index = 1
    while True:
        candidate = f"{MUG_OBJECT_NAME_PREFIX}_{index}"
        if candidate not in existing_names:
            return candidate
        index += 1


def _inject_object_block(text: str, object_name: str) -> str:
    objects_pattern = r"(\n  \(:objects\n(?:.*\n)*?)  \)\n\n  \(:obj_of_interest"
    match = re.search(objects_pattern, text)
    if match is None:
        raise ValueError("Failed to locate objects block in BDDL.")
    objects_block = match.group(1)
    if re.search(rf"\b{re.escape(object_name)}\b", objects_block):
        return text

    same_type_pattern = rf"(^\s+.+?)\s+-\s+{re.escape(MUG_OBJECT_TYPE)}\s*$"
    same_type_match = re.search(same_type_pattern, objects_block, flags=re.MULTILINE)
    if same_type_match:
        updated_objects_block = re.sub(
            same_type_pattern,
            lambda m: f"{m.group(1)} {object_name} - {MUG_OBJECT_TYPE}",
            objects_block,
            count=1,
            flags=re.MULTILINE,
        )
        return text.replace(objects_block, updated_objects_block, 1)

    repl = r"\1    " + object_name + " - " + MUG_OBJECT_TYPE + r"\n  )\n\n  (:obj_of_interest"
    updated, count = re.subn(objects_pattern, repl, text)
    if count != 1:
        raise ValueError("Failed to inject mug object block into BDDL.")
    return updated


def _inject_obj_of_interest_block(text: str, object_name: str) -> str:
    pattern = r"(\n  \(:obj_of_interest\n(?:.*\n)*?)  \)\n\n  \(:init"
    match = re.search(pattern, text)
    if match is None:
        raise ValueError("Failed to locate obj_of_interest block in BDDL.")
    obj_interest_block = match.group(1)
    interest_line = f"    {object_name}\n"
    if interest_line.strip() in obj_interest_block:
        return text
    repl = r"\1" + interest_line + r"  )\n\n  (:init"
    updated, count = re.subn(pattern, repl, text)
    if count != 1:
        raise ValueError("Failed to inject mug obj_of_interest block into BDDL.")
    return updated


def _inject_init_block(text: str, region_target: str, object_name: str) -> str:
    init_line = f"    (On {object_name} {region_target}_{MUG_REGION_NAME})\n"
    if init_line.strip() in text:
        return text
    pattern = r"(\n  \(:init\n(?:.*\n)*?)  \)\n\n  \(:goal"
    repl = r"\1" + init_line + r"  )\n\n  (:goal"
    updated, count = re.subn(pattern, repl, text)
    if count != 1:
        raise ValueError("Failed to inject mug init block into BDDL.")
    return updated


def _region_block_from_presets(suite_name: str, task_name: str, cfg) -> str:
    target = _get_task_target_surface(suite_name, task_name, cfg)
    ranges = _get_task_ranges(suite_name, task_name, cfg)
    ranges_text = "\n".join(
        [
            "              "
            f"({_format_coord(x1)} {_format_coord(y1)} {_format_coord(x2)} {_format_coord(y2)})"
            for x1, y1, x2, y2 in ranges
        ]
    )
    return (
        f"      ({MUG_REGION_NAME}\n"
        f"          (:target {target})\n"
        "          (:ranges (\n"
        f"{ranges_text}\n"
        "            )\n"
        "          )\n"
        "          (:yaw_rotation (\n"
        "              (0.0 0.0)\n"
        "            )\n"
        "          )\n"
        "      )\n"
    )


def _format_coord(value: float) -> str:
    return f"{float(value):.6f}"


def _resolve_presets_path(cfg) -> Path:
    raw_path = getattr(cfg, "eval_mug_placement_presets_path", "") if cfg is not None else ""
    if raw_path is None or str(raw_path).strip().lower() in {"", "none", "null"}:
        return DEFAULT_PRESETS_PATH.resolve()
    return Path(raw_path).expanduser().resolve()


@lru_cache(maxsize=8)
def _load_presets_file(path: str) -> dict:
    presets_path = Path(path).expanduser().resolve()
    with presets_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if "suites" not in data:
        raise ValueError(f"Invalid mug placement presets file, missing `suites`: {presets_path}")
    return data


def _get_task_preset(suite_name: str, task_name: str, cfg) -> dict:
    presets_path = _resolve_presets_path(cfg)
    data = _load_presets_file(str(presets_path))
    suites = data.get("suites", {})
    suite = suites.get(str(suite_name))
    if suite is None:
        raise KeyError(f"Missing suite `{suite_name}` in mug placement presets: {presets_path}")
    task_preset = suite.get("tasks", {}).get(str(task_name))
    if task_preset is None:
        raise KeyError(f"Missing task `{task_name}` for suite `{suite_name}` in mug placement presets: {presets_path}")
    return task_preset


def _get_task_target_surface(suite_name: str, task_name: str, cfg) -> str:
    target = _get_task_preset(suite_name, task_name, cfg).get("target_surface")
    if not target:
        raise ValueError(f"Missing target_surface for {suite_name}/{task_name} in mug placement presets.")
    return str(target)


def _get_task_ranges(suite_name: str, task_name: str, cfg) -> list[tuple[float, float, float, float]]:
    task_preset = _get_task_preset(suite_name, task_name, cfg)
    ranges = []
    for range_record in task_preset.get("ranges", []):
        xyxy = range_record.get("xyxy")
        if not isinstance(xyxy, list) or len(xyxy) != 4:
            continue
        x1, y1, x2, y2 = [float(value) for value in xyxy]
        ranges.append((x1, y1, x2, y2))
    if not ranges:
        raise ValueError(f"No mug placement ranges found for {suite_name}/{task_name}.")
    return ranges


def _sample_mug_xy(ranges, rng):
    range_idx = int(rng.integers(len(ranges)))
    x1, y1, x2, y2 = ranges[range_idx]
    return np.array([rng.uniform(x1, x2), rng.uniform(y1, y2)], dtype=np.float32)


def _get_joint_ranges(env):
    ranges = {}
    njnt = env.sim.model.njnt
    for idx in range(njnt):
        name = env.sim.model.joint_id2name(idx)
        qpos_start = int(env.sim.model.jnt_qposadr[idx])
        dof_start = int(env.sim.model.jnt_dofadr[idx])
        qpos_end = int(env.sim.model.jnt_qposadr[idx + 1]) if idx + 1 < njnt else int(env.sim.model.nq)
        dof_end = int(env.sim.model.jnt_dofadr[idx + 1]) if idx + 1 < njnt else int(env.sim.model.nv)
        ranges[name] = {"qpos": (qpos_start, qpos_end), "qvel": (dof_start, dof_end)}
    return ranges


def _merge_original_state_with_mug_layout(
    env,
    old_layout,
    new_layout,
    old_nq,
    old_nv,
    new_nq,
    original_state,
    mug_xy=None,
    mug_position=None,
    mug_orientation=None,
):
    current_state = env.sim.get_state().flatten().copy()
    merged = current_state
    merged[0] = original_state[0]

    old_qpos = original_state[1 : 1 + old_nq]
    old_qvel = original_state[1 + old_nq : 1 + old_nq + old_nv]

    for joint_name, old_range in old_layout.items():
        if joint_name not in new_layout:
            continue
        new_range = new_layout[joint_name]
        oqs, oqe = old_range["qpos"]
        nqs, nqe = new_range["qpos"]
        ovs, ove = old_range["qvel"]
        nvs, nve = new_range["qvel"]
        merged[1 + nqs : 1 + nqe] = old_qpos[oqs:oqe]
        merged[1 + new_nq + nvs : 1 + new_nq + nve] = old_qvel[ovs:ove]

    if mug_xy is not None:
        added_joint_names = [joint_name for joint_name in new_layout.keys() if joint_name not in old_layout]
        if added_joint_names:
            mug_joint_name = max(
                added_joint_names,
                key=lambda joint_name: new_layout[joint_name]["qpos"][1] - new_layout[joint_name]["qpos"][0],
            )
            qpos_start, qpos_end = new_layout[mug_joint_name]["qpos"]
            if qpos_end - qpos_start >= 2:
                merged[1 + qpos_start] = float(mug_xy[0])
                merged[1 + qpos_start + 1] = float(mug_xy[1])
            if mug_position is not None and qpos_end - qpos_start >= 3:
                mug_position = np.asarray(mug_position, dtype=np.float32)
                merged[1 + qpos_start : 1 + qpos_start + 3] = mug_position[:3]
            if mug_orientation is not None and qpos_end - qpos_start >= 7:
                mug_orientation = np.asarray(mug_orientation, dtype=np.float32)
                merged[1 + qpos_start + 3 : 1 + qpos_start + 7] = mug_orientation[:4]

    return merged
