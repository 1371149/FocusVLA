"""VLA-Cache evaluation entrypoint that delegates to the original implementation."""

import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import draccus


CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from config import VLACacheGenerateConfig, apply_runtime_overrides, load_vla_cache_config_from_json_if_any, validate_config
from original_runtime import run_original_eval


def _extract_config_json(argv):
    default_path = str(Path(__file__).resolve().parent / "configs" / "vla_cache_config.json")
    args = argv[1:]
    idx = 0
    while idx < len(args):
        token = args[idx]
        if token.startswith("--vla_cache_config_json="):
            return token.split("=", 1)[1]
        if token.startswith("--vla-cache-config-json="):
            return token.split("=", 1)[1]
        if token in {"--vla_cache_config_json", "--vla-cache-config-json"} and idx + 1 < len(args):
            return args[idx + 1]
        idx += 1
    return default_path


def _resolve_custom_log_filepath(config_json: str) -> Optional[str]:
    try:
        with open(config_json, "r", encoding="utf-8") as f:
            data = json.load(f)
        log_path = data.get("custom_log_filepath")
        if isinstance(log_path, str) and log_path.strip():
            return str(Path(log_path).expanduser().resolve())
    except Exception:
        pass
    return None


@draccus.wrap()
def eval_libero_vla_cache(cfg: VLACacheGenerateConfig) -> float:
    load_vla_cache_config_from_json_if_any(cfg)
    apply_runtime_overrides(cfg)
    validate_config(cfg)
    return run_original_eval(cfg)


def main():
    config_json = _extract_config_json(sys.argv)
    resolved_log_filepath = _resolve_custom_log_filepath(config_json)

    start_ts = time.time()
    start_dt = datetime.now()
    succeeded = False
    try:
        eval_libero_vla_cache()
        succeeded = True
    finally:
        end_ts = time.time()
        end_dt = datetime.now()
        elapsed_sec = end_ts - start_ts
        runtime_lines = [
            "",
            "[VLA-Cache] Runtime Summary",
            f"[VLA-Cache] Start time: {start_dt.isoformat(timespec='seconds')}",
            f"[VLA-Cache] End time: {end_dt.isoformat(timespec='seconds')}",
            f"[VLA-Cache] Elapsed time (seconds): {elapsed_sec:.3f}",
            f"[VLA-Cache] Elapsed time (minutes): {elapsed_sec / 60.0:.3f}",
            f"[VLA-Cache] Run status: {'success' if succeeded else 'failed'}",
        ]
        print("\n".join(runtime_lines))

        if resolved_log_filepath is not None:
            Path(resolved_log_filepath).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
            with open(resolved_log_filepath, "a", encoding="utf-8") as log_f:
                log_f.write("\n".join(runtime_lines) + "\n")


if __name__ == "__main__":
    main()
