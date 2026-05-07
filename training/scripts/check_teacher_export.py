#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify merged teacher export shards against manifest.json.")
    parser.add_argument(
        "--path",
        default="<OPENVLA_OFT_ROOT>/my_finetune/exports/libero_10/merged",
        help="Merged teacher export directory or manifest.json path.",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Only check manifest and file existence, do not load shard payloads.",
    )
    return parser.parse_args()


def load_manifest(path: Path):
    manifest_path = path if path.suffix.lower() == ".json" else path / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    with open(manifest_path, "r", encoding="utf-8") as f:
        return manifest_path, json.load(f)


def get_payload_record_count(payload) -> int:
    if "records" in payload:
        return len(payload["records"])
    if "data" in payload:
        return int(payload["data"]["sequence_length"].shape[0])
    raise KeyError("Unsupported teacher payload format.")


def main() -> None:
    args = parse_args()
    manifest_path, manifest = load_manifest(Path(args.path))
    base_dir = manifest_path.parent

    shards = manifest.get("shards", [])
    if not shards:
        raise ValueError(f"No shards found in manifest: {manifest_path}")

    expected_records = int(manifest.get("num_records", 0))
    manifest_sum = sum(int(shard["num_records"]) for shard in shards)

    print(f"manifest={manifest_path}")
    print(f"expected_num_records={expected_records}")
    print(f"manifest_shard_count={len(shards)}")
    print(f"manifest_sum={manifest_sum}")

    missing_files = []
    payload_sum = 0
    for idx, shard in enumerate(shards):
        shard_path = base_dir / shard["file"]
        if not shard_path.exists():
            missing_files.append(str(shard_path))
            continue

        if args.quick:
            payload_sum += int(shard["num_records"])
            continue

        payload = torch.load(shard_path, map_location="cpu")
        record_count = get_payload_record_count(payload)
        expected_count = int(shard["num_records"])
        if record_count != expected_count:
            raise ValueError(
                f"Shard count mismatch for {shard_path}: manifest={expected_count}, payload={record_count}"
            )
        payload_sum += record_count

        if idx % 20 == 0 or idx == len(shards) - 1:
            print(f"checked {idx + 1}/{len(shards)} shards")

    if missing_files:
        raise FileNotFoundError("Missing shard files:\n" + "\n".join(missing_files))

    if expected_records != manifest_sum:
        raise ValueError(f"Manifest num_records mismatch: expected={expected_records}, shard_sum={manifest_sum}")
    if payload_sum != expected_records:
        raise ValueError(f"Payload num_records mismatch: expected={expected_records}, payload_sum={payload_sum}")

    print("Verification passed.")
    print(f"payload_sum={payload_sum}")


if __name__ == "__main__":
    main()
