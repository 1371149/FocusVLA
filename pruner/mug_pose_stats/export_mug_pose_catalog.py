#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


DEFAULT_DATASET_ROOT = Path("<DATASET_GENERATE_ROOT>/generated_data")
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export reusable mug-pose catalog and summary stats for pruner evals.")
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def load_suite_sources(dataset_root: Path) -> dict[str, list[Path]]:
    return {
        "libero_10": [dataset_root / "libero_10_mug_metainfo.json"],
        "libero_spatial": [dataset_root / "libero_spatial_mug_metainfo.json"],
        "libero_object": [dataset_root / "libero_object_mug_metainfo.json"],
        "libero_goal": sorted((dataset_root / "libero_goal_mug_metainfo_parts").glob("*.json")),
    }


def load_merged_suite_data(sources: list[Path]) -> dict:
    merged = {}
    for path in sources:
        if not path.exists():
            continue
        merged.update(json.loads(path.read_text()))
    return merged


def build_catalog(dataset_root: Path) -> tuple[dict, list[dict], dict]:
    suite_sources = load_suite_sources(dataset_root)
    catalog = {
        "dataset_root": str(dataset_root.resolve()),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "suites": {},
    }
    task_rows = []
    suite_summary = {}

    for suite_name, sources in suite_sources.items():
        merged = load_merged_suite_data(sources)
        suite_payload = {
            "source_files": [str(path.resolve()) for path in sources if path.exists()],
            "tasks": {},
        }
        reject_counter = Counter()
        suite_attempts = 0
        suite_successes = 0
        suite_source_demos = set()
        suite_source_demos_with_success = set()
        suite_unique_pose_keys = set()

        for task_name in sorted(merged):
            task_data = merged[task_name]
            attempts = [
                (attempt_key, attempt_value)
                for attempt_key, attempt_value in task_data.items()
                if attempt_key != "render_settings"
                and isinstance(attempt_value, dict)
                and "source_demo_key" in attempt_value
            ]
            successes = []
            source_demos = set()
            source_demos_with_success = set()
            task_unique_pose_keys = set()

            for attempt_key, attempt_value in attempts:
                source_demos.add(attempt_value["source_demo_key"])
                if not bool(attempt_value.get("success", False)):
                    reject_counter[attempt_value.get("reject_reason", "unknown")] += 1
                    continue

                source_demos_with_success.add(attempt_value["source_demo_key"])
                mug_position = [float(value) for value in attempt_value.get("mug_position", [])]
                mug_orientation = [float(value) for value in attempt_value.get("mug_orientation", [])]
                mug_xy = [float(value) for value in attempt_value.get("mug_xy", [])]
                pose_key = (
                    tuple(round(value, 6) for value in mug_position),
                    tuple(round(value, 6) for value in mug_orientation),
                )
                task_unique_pose_keys.add(pose_key)
                suite_unique_pose_keys.add((task_name, pose_key))
                successes.append(
                    {
                        "attempt_key": attempt_key,
                        "source_demo_key": attempt_value["source_demo_key"],
                        "variant_index": int(attempt_value.get("variant_index", -1)),
                        "attempt_seed": int(attempt_value.get("attempt_seed", -1)),
                        "selected_range_index": int(attempt_value.get("selected_range_index", -1)),
                        "available_range_count": int(attempt_value.get("available_range_count", -1)),
                        "pose_source": attempt_value.get("pose_source", "unknown"),
                        "saved_demo_key": attempt_value.get("saved_demo_key"),
                        "mug_xy": mug_xy,
                        "mug_position": mug_position,
                        "mug_orientation": mug_orientation,
                    }
                )

            suite_attempts += len(attempts)
            suite_successes += len(successes)
            suite_source_demos.update(source_demos)
            suite_source_demos_with_success.update(source_demos_with_success)

            render_settings = task_data.get("render_settings", {})
            task_payload = {
                "render_settings": render_settings,
                "summary": {
                    "attempts": len(attempts),
                    "successes": len(successes),
                    "success_rate": (len(successes) / len(attempts)) if attempts else 0.0,
                    "source_demos": len(source_demos),
                    "source_demos_with_success": len(source_demos_with_success),
                    "unique_saved_poses": len(task_unique_pose_keys),
                    "available_range_count_set": sorted(
                        {
                            int(attempt_value.get("available_range_count", -1))
                            for _, attempt_value in attempts
                            if "available_range_count" in attempt_value
                        }
                    ),
                },
                "successful_pose_candidates": successes,
            }
            suite_payload["tasks"][task_name] = task_payload
            task_rows.append(
                {
                    "suite_name": suite_name,
                    "task_name": task_name,
                    **task_payload["summary"],
                }
            )

        catalog["suites"][suite_name] = suite_payload
        suite_summary[suite_name] = {
            "num_tasks": len(suite_payload["tasks"]),
            "attempts": suite_attempts,
            "successes": suite_successes,
            "success_rate": (suite_successes / suite_attempts) if suite_attempts else 0.0,
            "source_demos": len(suite_source_demos),
            "source_demos_with_success": len(suite_source_demos_with_success),
            "unique_saved_poses": len(suite_unique_pose_keys),
            "top_reject_reasons": reject_counter.most_common(8),
        }

    return catalog, task_rows, suite_summary


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_task_csv(path: Path, rows: list[dict]) -> None:
    fieldnames = [
        "suite_name",
        "task_name",
        "attempts",
        "successes",
        "success_rate",
        "source_demos",
        "source_demos_with_success",
        "unique_saved_poses",
        "available_range_count_set",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            serialized = dict(row)
            serialized["available_range_count_set"] = "|".join(str(value) for value in row["available_range_count_set"])
            writer.writerow(serialized)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    catalog, task_rows, suite_summary = build_catalog(args.dataset_root.expanduser().resolve())

    write_json(output_dir / "mug_pose_catalog.json", catalog)
    write_json(output_dir / "mug_pose_suite_stats.json", suite_summary)
    write_task_csv(output_dir / "mug_pose_task_stats.csv", task_rows)

    print(f"Wrote mug pose catalog to {output_dir}")


if __name__ == "__main__":
    main()
