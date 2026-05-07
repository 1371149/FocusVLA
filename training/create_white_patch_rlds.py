#!/usr/bin/env python3
"""Create a white-patch triggered copy of an OpenVLA LIBERO RLDS dataset.

The script reads an existing TFDS/RLDS dataset and writes a new TFDS dataset
with the same structure. Every uint8 RGB/RGBA/grayscale image under each step's
`observation` field receives a white square in the top-left corner. By default,
the square area is 1% of the image area.
"""

from __future__ import annotations

import argparse
import os
import shutil
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any, Optional

import numpy as np
from tqdm import tqdm


IMAGE_CHANNEL_COUNTS = {1, 3, 4}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help=(
            "Path to the source TFDS data root, dataset directory, or version directory. "
            "Examples: /data/modified_libero_rlds, /data/libero_spatial_no_noops, "
            "or /data/libero_spatial_no_noops/1.0.0."
        ),
    )
    parser.add_argument(
        "--dataset-name",
        default=None,
        help="Source dataset name when --input-dir points to a TFDS data root.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="TFDS data root for the generated white-patch dataset.",
    )
    parser.add_argument(
        "--output-dataset-name",
        default=None,
        help=(
            "Name of the generated TFDS dataset. Defaults to --dataset-name when provided, "
            "otherwise the source TFDS builder name. "
            "Keeping the same name under a different output root lets OpenVLA reuse the original dataset config."
        ),
    )
    parser.add_argument(
        "--splits",
        default="train",
        help="Comma-separated splits to process. Default: train.",
    )
    parser.add_argument(
        "--patch-area-ratio",
        type=float,
        default=0.01,
        help="White square area as a fraction of image area. Default: 0.01.",
    )
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=None,
        help="Optional cap per split for quick debugging.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite the generated dataset if it already exists.",
    )
    parser.add_argument(
        "--keep-file-path-metadata",
        action="store_true",
        help="Keep original episode_metadata.file_path values instead of replacing them with portable IDs.",
    )
    return parser.parse_args()


def is_image_array(value: np.ndarray) -> bool:
    if value.dtype != np.uint8 or value.ndim not in {3, 4}:
        return False
    return int(value.shape[-1]) in IMAGE_CHANNEL_COUNTS


def add_white_patch_to_image_array(value: np.ndarray, patch_area_ratio: float) -> np.ndarray:
    """Return a patched copy of an HWC image or NHWC image batch."""
    if not is_image_array(value):
        return value

    patched = np.array(value, copy=True)
    height = int(patched.shape[-3])
    width = int(patched.shape[-2])
    patch_side = max(1, int(round((height * width * patch_area_ratio) ** 0.5)))
    patch_side = min(patch_side, height, width)

    if patched.ndim == 3:
        patched[:patch_side, :patch_side, :] = 255
    else:
        patched[:, :patch_side, :patch_side, :] = 255
    return patched


def patch_observation_tree(value: Any, patch_area_ratio: float) -> Any:
    if isinstance(value, np.ndarray):
        return add_white_patch_to_image_array(value, patch_area_ratio)
    if isinstance(value, Mapping):
        return {key: patch_observation_tree(item, patch_area_ratio) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(patch_observation_tree(item, patch_area_ratio) for item in value)
    if isinstance(value, list):
        return [patch_observation_tree(item, patch_area_ratio) for item in value]
    return value


def patch_step(step: Mapping[str, Any], patch_area_ratio: float) -> dict[str, Any]:
    patched_step = dict(step)
    if "observation" in patched_step:
        patched_step["observation"] = patch_observation_tree(patched_step["observation"], patch_area_ratio)
    return patched_step


def patch_steps(steps: Any, patch_area_ratio: float) -> Any:
    if isinstance(steps, Mapping):
        patched_steps = dict(steps)
        if "observation" in patched_steps:
            patched_steps["observation"] = patch_observation_tree(patched_steps["observation"], patch_area_ratio)
        return patched_steps
    if isinstance(steps, Iterable):
        return [patch_step(step, patch_area_ratio) for step in steps]
    raise TypeError(f"Unsupported RLDS steps object: {type(steps)}")


def maybe_version_dir(path: Path) -> Optional[Path]:
    if (path / "dataset_info.json").exists():
        return path
    version_dirs = sorted(child for child in path.iterdir() if child.is_dir() and (child / "dataset_info.json").exists())
    if version_dirs:
        return version_dirs[-1]
    return None


def load_source_builder(input_dir: Path, dataset_name: Optional[str]):
    import tensorflow_datasets as tfds

    input_dir = input_dir.expanduser().resolve()
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    version_dir = maybe_version_dir(input_dir)
    if version_dir is not None:
        return tfds.builder_from_directory(str(version_dir))

    if dataset_name:
        dataset_dir = input_dir / dataset_name
        if dataset_dir.exists():
            version_dir = maybe_version_dir(dataset_dir)
            if version_dir is not None:
                return tfds.builder_from_directory(str(version_dir))
        return tfds.builder(dataset_name, data_dir=str(input_dir))

    raise ValueError(
        "Could not infer the source dataset. Pass --dataset-name when --input-dir is a TFDS data root."
    )


def source_dataset_root(builder) -> Path:
    data_dir = Path(str(builder.data_dir)).resolve()
    if data_dir.name == str(builder.version):
        return data_dir.parent
    return data_dir


def iter_patched_examples(
    *,
    builder,
    split: str,
    patch_area_ratio: float,
    max_episodes: Optional[int],
    keep_file_path_metadata: bool,
):
    import tensorflow_datasets as tfds

    ds = builder.as_dataset(split=split, shuffle_files=False)
    total = None
    if split in builder.info.splits:
        total = builder.info.splits[split].num_examples
        if max_episodes is not None:
            total = min(total, max_episodes)

    for episode_idx, episode in enumerate(tqdm(tfds.as_numpy(ds), total=total, desc=f"patching {split}")):
        if max_episodes is not None and episode_idx >= max_episodes:
            break

        patched_episode = dict(episode)
        patched_episode["steps"] = patch_steps(episode["steps"], patch_area_ratio)

        if not keep_file_path_metadata and "episode_metadata" in patched_episode:
            metadata = dict(patched_episode["episode_metadata"])
            if "file_path" in metadata:
                metadata["file_path"] = f"{builder.name}/{split}/episode_{episode_idx:06d}"
            patched_episode["episode_metadata"] = metadata

        yield episode_idx, patched_episode


def main() -> None:
    args = parse_args()
    if not 0.0 < args.patch_area_ratio <= 1.0:
        raise ValueError("--patch-area-ratio must be in (0, 1].")

    # Avoid TensorFlow taking GPU memory from PyTorch workflows.
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    os.environ.setdefault("NO_GCE_CHECK", "true")

    import tensorflow_datasets as tfds
    from tensorflow_datasets.core.utils import gcs_utils

    gcs_utils._is_gcs_disabled = True

    source_builder = load_source_builder(args.input_dir, args.dataset_name)
    output_root = args.output_root.expanduser().resolve()
    output_name = args.output_dataset_name or args.dataset_name or source_builder.name
    target_dataset_root = output_root / output_name

    if target_dataset_root.resolve() == source_dataset_root(source_builder):
        raise ValueError(
            "Refusing to overwrite the source dataset. Use a different --output-root or --output-dataset-name."
        )

    if target_dataset_root.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output dataset already exists: {target_dataset_root}. Use --overwrite to replace it.")
        shutil.rmtree(target_dataset_root)

    splits = [split.strip() for split in args.splits.split(",") if split.strip()]
    if not splits:
        raise ValueError("No splits requested.")

    split_datasets = {
        split: iter_patched_examples(
            builder=source_builder,
            split=split,
            patch_area_ratio=args.patch_area_ratio,
            max_episodes=args.max_episodes,
            keep_file_path_metadata=args.keep_file_path_metadata,
        )
        for split in splits
    }

    download_config = tfds.download.DownloadConfig(try_download_gcs=False)
    builder = tfds.dataset_builders.store_as_tfds_dataset(
        name=output_name,
        version=str(source_builder.version),
        features=source_builder.info.features,
        split_datasets=split_datasets,
        data_dir=str(output_root),
        description=(
            f"White-patch copy of {source_builder.name}. "
            f"Each observation image has a top-left white square covering approximately "
            f"{args.patch_area_ratio:.2%} of the image area."
        ),
        release_notes={str(source_builder.version): "Generated by FocusVLA white-patch preprocessing."},
        download_config=download_config,
        disable_shuffling=False,
    )

    print("White-patch RLDS dataset created.")
    print(f"Source dataset: {source_builder.name}")
    print(f"Output dataset: {builder.name}")
    print(f"Output dir:     {builder.data_dir}")
    print(f"Splits:         {builder.info.splits}")


if __name__ == "__main__":
    main()
