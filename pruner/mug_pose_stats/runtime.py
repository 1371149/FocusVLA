from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path


DEFAULT_CATALOG_PATH = Path(__file__).resolve().parent / "mug_pose_catalog.json"


def resolve_catalog_path(catalog_path: str | Path | None = None) -> Path:
    raw_path = str(catalog_path or "").strip()
    if raw_path:
        return Path(raw_path).expanduser().resolve()
    return DEFAULT_CATALOG_PATH.resolve()


@lru_cache(maxsize=8)
def _load_catalog_cached(catalog_path: str) -> dict:
    path = Path(catalog_path)
    if not path.exists():
        raise FileNotFoundError(f"Mug pose catalog not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_pose_catalog(catalog_path: str | Path | None = None) -> dict:
    return _load_catalog_cached(str(resolve_catalog_path(catalog_path)))


def get_task_pose_candidates(catalog_path: str | Path | None, suite_name: str, task_name: str) -> list[dict]:
    catalog = load_pose_catalog(catalog_path)
    suite_payload = catalog.get("suites", {}).get(str(suite_name), {})
    task_payload = suite_payload.get("tasks", {}).get(str(task_name), {})
    candidates = task_payload.get("successful_pose_candidates", [])
    return [candidate for candidate in candidates if isinstance(candidate, dict)]


def get_pose_retry_limit(candidate_count: int, max_init_retries: int | None = None) -> int:
    if candidate_count <= 0:
        return 0
    if max_init_retries is None or int(max_init_retries) <= 0:
        return candidate_count
    return min(candidate_count, int(max_init_retries))


def select_pose_candidate(candidates: list[dict], episode_idx: int, retry_idx: int = 0) -> tuple[int, dict]:
    if not candidates:
        raise IndexError("No mug pose candidates are available.")
    candidate_index = (int(episode_idx) + int(retry_idx)) % len(candidates)
    return candidate_index, candidates[candidate_index]

