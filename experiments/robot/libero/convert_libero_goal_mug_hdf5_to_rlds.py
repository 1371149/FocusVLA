#!/usr/bin/env python3
import argparse
import os
import shutil
from pathlib import Path

import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow_datasets.core.utils import gcs_utils

from experiments.robot.libero.libero_goal_mug_no_noops import LiberoGoalMugNoNoops


def parse_args():
    parser = argparse.ArgumentParser(description="Convert generated LIBERO-Goal mug HDF5 demos into an RLDS dataset usable by OpenVLA-OFT.")
    parser.add_argument("--input-dir", type=Path, default=Path("<DATASET_GENERATE_ROOT>/generated_data/libero_goal_mug"))
    parser.add_argument("--output-root", type=Path, default=Path("<OPENVLA_FINETUNE_ROOT>/blue_mug_dataset"))
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--rotate-180", action="store_true")
    return parser.parse_args()


def main(args):
    os.environ.setdefault("NO_GCE_CHECK", "true")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    gcs_utils._is_gcs_disabled = True
    LiberoGoalMugNoNoops.code_path = Path(__file__).resolve()

    output_root = args.output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    dataset_dir = output_root / 'libero_goal_no_noops'
    if dataset_dir.exists() and args.overwrite:
        shutil.rmtree(dataset_dir)

    builder = LiberoGoalMugNoNoops(
        data_dir=str(output_root),
        input_dir=args.input_dir,
        max_episodes=args.max_episodes,
        rotate_180=args.rotate_180,
    )
    download_config = tfds.download.DownloadConfig(try_download_gcs=False)
    builder.download_and_prepare(download_config=download_config)
    print(f'Built RLDS dataset at: {builder.data_dir}')
    print(f'Dataset name for training: {builder.name}')
    print(f'Splits: {builder.info.splits}')


if __name__ == '__main__':
    tf.get_logger().setLevel('ERROR')
    main(parse_args())
