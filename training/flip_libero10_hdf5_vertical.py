#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Vertically flip RGB image streams in source LIBERO-10 mug HDF5 files.

Default image fields:
- /data/demo_x/obs/agentview_rgb
- /data/demo_x/obs/eye_in_hand_rgb

Both input and output are HDF5 directories. The output can then be converted
to RLDS with convert_libero_10_mug_hdf5_to_rlds.py.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import h5py
import numpy as np


TARGET_IMAGE_KEYS = ("agentview_rgb", "eye_in_hand_rgb")


def vertical_flip_images(images: np.ndarray) -> np.ndarray:
    """images shape: (T, H, W, C), vertical flip on H dimension."""
    if images.ndim != 4:
        raise ValueError(f"Expected 4D image tensor (T,H,W,C), got shape={images.shape}")
    return images[:, ::-1, :, :].copy()


def copy_group_recursive(src_group: h5py.Group, dst_group: h5py.Group, stats: dict) -> None:
    # copy attrs
    for key, value in src_group.attrs.items():
        dst_group.attrs[key] = value

    for name, item in src_group.items():
        if isinstance(item, h5py.Group):
            new_group = dst_group.create_group(name)
            copy_group_recursive(item, new_group, stats)
        elif isinstance(item, h5py.Dataset):
            data = item[()]

            if name in TARGET_IMAGE_KEYS and isinstance(data, np.ndarray):
                data = vertical_flip_images(data)
                stats["flipped_datasets"] += 1
                stats["flipped_frames"] += int(data.shape[0])

            dset = dst_group.create_dataset(name, data=data, dtype=item.dtype)
            for ak, av in item.attrs.items():
                dset.attrs[ak] = av
        else:
            raise TypeError(f"Unsupported HDF5 item type: {type(item)} at {item.name}")


def process_one_file(src_path: Path, dst_path: Path) -> dict:
    stats = {"flipped_datasets": 0, "flipped_frames": 0}
    with h5py.File(src_path, "r") as src, h5py.File(dst_path, "w") as dst:
        copy_group_recursive(src, dst, stats)
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Vertical-flip LIBERO-10 mug HDF5 image streams.")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("<DATASET_GENERATE_ROOT>/generated_data/libero_10_mug"),
        help="Source HDF5 directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("<DATASET_GENERATE_ROOT>/generated_data/libero_10_mug_vflip"),
        help="Output HDF5 directory.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite the output directory if it exists.")
    args = parser.parse_args()

    input_dir = args.input_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    if not input_dir.exists():
        raise FileNotFoundError(f"Input dir not found: {input_dir}")

    if output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output dir exists: {output_dir}. Use --overwrite to replace.")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    h5_files = sorted(input_dir.glob("*_demo.hdf5"))
    if not h5_files:
        raise FileNotFoundError(f"No '*_demo.hdf5' files found in {input_dir}")

    total_ds = 0
    total_frames = 0

    for src in h5_files:
        dst = output_dir / src.name
        stats = process_one_file(src, dst)
        total_ds += stats["flipped_datasets"]
        total_frames += stats["flipped_frames"]
        print(f"[OK] {src.name}: flipped_datasets={stats['flipped_datasets']}, flipped_frames={stats['flipped_frames']}")

    print("\nDone.")
    print(f"Input:  {input_dir}")
    print(f"Output: {output_dir}")
    print(f"Total flipped datasets: {total_ds}")
    print(f"Total flipped frames:   {total_frames}")


if __name__ == "__main__":
    main()
