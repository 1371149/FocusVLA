#!/usr/bin/env python3
import argparse
from pathlib import Path
import os


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build RLDS/TFDS dataset `libero_spatial_mug_no_noops` from regenerated LIBERO mug HDF5 files."
    )
    parser.add_argument(
        "--input_hdf5_dir",
        type=Path,
        default=Path("<DATASET_GENERATE_ROOT>/generated_data/libero_spatial_mug"),
        help="Directory containing mug-augmented HDF5 demos (*_demo.hdf5).",
    )
    parser.add_argument(
        "--tfds_data_dir",
        type=Path,
        default=Path("<OPENVLA_FINETUNE_ROOT>/dataset"),
        help="TFDS data root. Output will be under <tfds_data_dir>/libero_spatial_mug_no_noops/1.0.0.",
    )
    parser.add_argument(
        "--max_episodes",
        type=int,
        default=None,
        help="Optional cap for quick test builds.",
    )
    parser.add_argument(
        "--rotate_180",
        action="store_true",
        help="Rotate RGB frames by 180 degrees during conversion (matches original OpenVLA LIBERO conversion).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete existing target dataset directory before build.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Avoid long hangs from TFDS GCS metadata checks on offline machines.
    os.environ.setdefault("NO_GCE_CHECK", "true")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

    from experiments.robot.libero.libero_spatial_mug_no_noops.libero_spatial_mug_no_noops_dataset_builder import (
        LiberoSpatialMugNoNoops,
    )
    from tensorflow_datasets.core.utils import gcs_utils
    import tensorflow_datasets as tfds

    gcs_utils._is_gcs_disabled = True

    # Work around TFDS MultiplexedPath bug in some envs.
    LiberoSpatialMugNoNoops.code_path = Path(__file__).resolve()

    input_dir = args.input_hdf5_dir.resolve()
    data_dir = args.tfds_data_dir.resolve()

    if not input_dir.exists():
        raise FileNotFoundError(f"Input HDF5 dir not found: {input_dir}")

    target_dataset_dir = data_dir / "libero_spatial_mug_no_noops"
    if target_dataset_dir.exists() and args.overwrite:
        import shutil

        shutil.rmtree(target_dataset_dir)

    builder = LiberoSpatialMugNoNoops(
        data_dir=str(data_dir),
        input_dir=input_dir,
        max_episodes=args.max_episodes,
        rotate_180=args.rotate_180,
    )

    # GeneratorBasedBuilder path (no Apache Beam) to produce TFRecord shards.
    download_config = tfds.download.DownloadConfig(try_download_gcs=False)
    builder.download_and_prepare(download_config=download_config)

    print("Build complete.")
    print(f"Dataset name: {builder.name}")
    print(f"Dataset dir:  {builder.data_dir}")
    print(f"Version dir:  {builder.data_path}")
    print(f"Splits:       {builder.info.splits}")


if __name__ == "__main__":
    main()
