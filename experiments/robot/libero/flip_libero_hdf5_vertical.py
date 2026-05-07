#!/usr/bin/env python3
import argparse
import shutil
from pathlib import Path

import h5py
import numpy as np
from tqdm import tqdm


DEFAULT_IMAGE_KEYS = ("agentview_rgb", "eye_in_hand_rgb")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Copy LIBERO HDF5 demos and flip image frames vertically (top-bottom)."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("<DATASET_GENERATE_ROOT>/generated_data/libero_10_white"),
        help="Directory containing *_demo.hdf5 files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("<DATASET_GENERATE_ROOT>/generated_data/libero_10_white_vflip"),
        help="Directory to write flipped HDF5 files.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite output directory if it exists.")
    parser.add_argument(
        "--image-keys",
        nargs="*",
        default=list(DEFAULT_IMAGE_KEYS),
        help="Observation image dataset keys under data/demo_*/obs to flip.",
    )
    return parser.parse_args()


def flip_file(src_path: Path, dst_path: Path, image_keys):
    with h5py.File(src_path, "r") as src, h5py.File(dst_path, "w") as dst:
        # Copy everything first
        src.copy("/", dst)

        if "data" not in dst:
            return

        for demo_key in dst["data"].keys():
            obs = dst[f"data/{demo_key}/obs"]
            for key in image_keys:
                if key not in obs:
                    continue
                arr = obs[key][()]
                if arr.ndim < 3:
                    continue
                # Expected shape: (T, H, W, C), flip vertical => axis=1
                flipped = np.flip(arr, axis=1).copy()
                obs[key][...] = flipped


def main():
    args = parse_args()

    input_dir = args.input_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    if not input_dir.exists():
        raise FileNotFoundError(f"Input dir not found: {input_dir}")

    hdf5_files = sorted(input_dir.glob("*_demo.hdf5"))
    if not hdf5_files:
        raise FileNotFoundError(f"No *_demo.hdf5 found in {input_dir}")

    if output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output dir exists: {output_dir}. Use --overwrite to replace.")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for src_path in tqdm(hdf5_files, desc="Flipping HDF5"):
        dst_path = output_dir / src_path.name
        flip_file(src_path, dst_path, args.image_keys)

    print("Done.")
    print(f"Input : {input_dir}")
    print(f"Output: {output_dir}")
    print(f"Image keys flipped: {args.image_keys}")


if __name__ == "__main__":
    main()
