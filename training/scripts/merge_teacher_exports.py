#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path
from typing import Dict, List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge sharded teacher export part directories into one manifest.")
    parser.add_argument(
        "--root",
        default="<OPENVLA_OFT_ROOT>/my_finetune/exports/libero_spatial_8head",
        help="Root directory containing part0, part1, ...",
    )
    parser.add_argument(
        "--output",
        default="<OPENVLA_OFT_ROOT>/my_finetune/exports/libero_spatial_8head/merged",
        help="Merged output directory.",
    )
    parser.add_argument(
        "--part-prefix",
        default="part",
        help="Only directories with this prefix will be merged.",
    )
    return parser.parse_args()


def discover_part_dirs(root: Path, part_prefix: str) -> List[Path]:
    part_dirs = [path for path in root.iterdir() if path.is_dir() and path.name.startswith(part_prefix)]
    if not part_dirs:
        raise FileNotFoundError(f"No part directories found under {root}")

    def sort_key(path: Path):
        suffix = path.name[len(part_prefix) :]
        return (0, int(suffix)) if suffix.isdigit() else (1, path.name)

    return sorted(part_dirs, key=sort_key)


def load_manifest(part_dir: Path) -> Dict:
    manifest_path = part_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing manifest: {manifest_path}")
    with open(manifest_path, "r", encoding="utf-8") as f:
        return json.load(f)


def clear_old_output(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for path in output_dir.glob("shard_*.pt"):
        path.unlink()
    manifest_path = output_dir / "manifest.json"
    if manifest_path.exists():
        manifest_path.unlink()


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    output_dir = Path(args.output)

    if not root.exists():
        raise FileNotFoundError(f"Root directory not found: {root}")

    part_dirs = discover_part_dirs(root, args.part_prefix)
    clear_old_output(output_dir)

    all_shards = []
    total_records = 0
    base_manifest = None
    merged_shard_idx = 0
    min_start_index = None
    max_end_index = None

    for part_dir in part_dirs:
        manifest = load_manifest(part_dir)
        if base_manifest is None:
            base_manifest = dict(manifest)

        start_index = manifest.get("start_index")
        end_index = manifest.get("end_index")
        if start_index is not None:
            min_start_index = start_index if min_start_index is None else min(min_start_index, start_index)
        if end_index is not None:
            max_end_index = end_index if max_end_index is None else max(max_end_index, end_index)

        for shard in manifest.get("shards", []):
            src_path = (part_dir / shard["file"]).resolve()
            if not src_path.exists():
                raise FileNotFoundError(f"Missing shard file: {src_path}")

            dst_name = f"shard_{merged_shard_idx:05d}.pt"
            dst_path = output_dir / dst_name
            if dst_path.exists() or dst_path.is_symlink():
                dst_path.unlink()
            os.symlink(src_path, dst_path)

            num_records = int(shard["num_records"])
            all_shards.append(
                {
                    "file": dst_name,
                    "num_records": num_records,
                    "record_offset": total_records,
                }
            )
            total_records += num_records
            merged_shard_idx += 1

    if base_manifest is None:
        raise RuntimeError("No manifests loaded.")

    base_manifest["output_dir"] = str(output_dir)
    base_manifest["start_index"] = 0 if min_start_index is None else min_start_index
    base_manifest["end_index"] = max_end_index
    base_manifest["max_samples"] = total_records
    base_manifest["num_records"] = total_records
    base_manifest["shard_count"] = len(all_shards)
    base_manifest["shards"] = all_shards

    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(base_manifest, f, ensure_ascii=False, indent=2)

    print(f"Merged {len(part_dirs)} part directories into {output_dir}")
    print(f"num_records={total_records}")
    print(f"shard_count={len(all_shards)}")
    print(manifest_path)


if __name__ == "__main__":
    main()
