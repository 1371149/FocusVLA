"""Compatibility wrapper for the ADP LIBERO pruning evaluation entrypoint.

Extra convenience flags are handled here so they override values coming from the
JSON config loaded inside the ADP runner.
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

from run_libero_eval_adp import eval_libero_adp


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
    args, remaining = parser.parse_known_args(argv[1:])

    if args.with_white_patch and args.without_white_patch:
        raise SystemExit("Cannot pass both `--with-white-patch` and `--without-white-patch`.")
    if args.with_mug and args.without_mug:
        raise SystemExit("Cannot pass both `--with-mug` and `--without-mug`.")

    return args, [argv[0]] + remaining


def _extract_qk_config_json(argv):
    default_path = str(Path(__file__).resolve().parent / "configs" / "prune_v2_config.json")
    args = argv[1:]
    i = 0
    while i < len(args):
        token = args[i]
        if token.startswith("--qk_config_json="):
            return token.split("=", 1)[1]
        if token.startswith("--qk-config-json="):
            return token.split("=", 1)[1]
        if token in {"--qk_config_json", "--qk-config-json"} and i + 1 < len(args):
            return args[i + 1]
        i += 1
    return default_path


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


def _resolve_custom_log_filepath(qk_config_json, argv):
    cli_log_path = _extract_cli_option_value(argv, {"--custom_log_filepath", "--custom-log-filepath"})
    if isinstance(cli_log_path, str):
        stripped = cli_log_path.strip()
        if stripped and stripped.lower() not in {"none", "null"}:
            return str(Path(stripped).expanduser().resolve())

    try:
        with open(qk_config_json, "r", encoding="utf-8") as f:
            data = json.load(f)
        log_path = data.get("custom_log_filepath")
        if isinstance(log_path, str) and log_path.strip():
            return str(Path(log_path).expanduser().resolve())
    except Exception:
        pass
    return None


def main():
    wrapper_args, remaining_argv = _parse_wrapper_args(sys.argv)
    sys.argv = remaining_argv
    qk_config_json = _extract_qk_config_json(remaining_argv)
    resolved_log_filepath = _resolve_custom_log_filepath(qk_config_json, remaining_argv)

    if wrapper_args.with_white_patch:
        os.environ["ADP_EVAL_WITH_WHITE_PATCH"] = "true"
    elif wrapper_args.without_white_patch:
        os.environ["ADP_EVAL_WITH_WHITE_PATCH"] = "false"
    if wrapper_args.with_mug:
        os.environ["ADP_EVAL_WITH_MUG"] = "true"
    elif wrapper_args.without_mug:
        os.environ["ADP_EVAL_WITH_MUG"] = "false"

    start_ts = time.time()
    start_dt = datetime.now()
    succeeded = False
    try:
        eval_libero_adp()
        succeeded = True
    finally:
        end_ts = time.time()
        end_dt = datetime.now()
        elapsed_sec = end_ts - start_ts
        runtime_lines = [
            "",
            "[ADP] Runtime Summary",
            f"[ADP] Start time: {start_dt.isoformat(timespec='seconds')}",
            f"[ADP] End time: {end_dt.isoformat(timespec='seconds')}",
            f"[ADP] Elapsed time (seconds): {elapsed_sec:.3f}",
            f"[ADP] Elapsed time (minutes): {elapsed_sec / 60.0:.3f}",
            f"[ADP] Run status: {'success' if succeeded else 'failed'}",
        ]
        print("\n".join(runtime_lines))

        if resolved_log_filepath is not None:
            Path(resolved_log_filepath).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
            with open(resolved_log_filepath, "a", encoding="utf-8") as log_f:
                log_f.write("\n".join(runtime_lines) + "\n")


if __name__ == "__main__":
    main()
