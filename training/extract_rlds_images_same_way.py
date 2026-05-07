#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import argparse
import tensorflow_datasets as tfds
from PIL import Image


def extract_images_same_way(dataset_dir: Path, out_root: Path, tag: str, num_episodes: int, max_steps: int):
    builder = tfds.builder_from_directory(str(dataset_dir))
    ds = builder.as_dataset(split="train")

    out_dir = out_root / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    for epi_idx, episode in enumerate(tfds.as_numpy(ds.take(num_episodes))):
        steps = episode["steps"]
        for step_idx, step in enumerate(steps):
            if step_idx >= max_steps:
                break

            obs = step["observation"]
            primary = obs["image"]
            wrist = obs.get("wrist_image", None)

            Image.fromarray(primary).save(out_dir / f"ep{epi_idx:03d}_st{step_idx:04d}_primary.png")
            if wrist is not None:
                Image.fromarray(wrist).save(out_dir / f"ep{epi_idx:03d}_st{step_idx:04d}_wrist.png")

    print(f"[OK] {tag}: saved to {out_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--official-dir",
        type=Path,
        default=Path("<OPENVLA_FINETUNE_ROOT>/dataset/libero_goal_no_noops/1.0.0"),
    )
    parser.add_argument(
        "--mug-dir",
        type=Path,
        default=Path("<OPENVLA_FINETUNE_ROOT>/blue_mug_dataset/libero_object_no_noops/1.0.0"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("<TMP_DIR>/rlds_extract_compare"),
    )
    parser.add_argument("--num-episodes", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=10)
    args = parser.parse_args()

    extract_images_same_way(args.official_dir, args.out_dir, "official_libero10", args.num_episodes, args.max_steps)
    extract_images_same_way(args.mug_dir, args.out_dir, "mug_libero10", args.num_episodes, args.max_steps)


if __name__ == "__main__":
    main()
