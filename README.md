# FocusVLA

This repository contains the code for:

**FocusVLA: Hijacking Attention to Break Visual Token Pruning in Vision-Language-Action Models**

## Setup

FocusVLA is built on top of OpenVLA-OFT. Before using this repository, first
follow the official OpenVLA-OFT instructions to install the environment and
prepare LIBERO evaluation dependencies.

You also need to download:

* OpenVLA-OFT checkpoints for the LIBERO suites you want to train or evaluate.
* The modified LIBERO RLDS datasets from
  `https://huggingface.co/datasets/openvla/modified_libero_rlds`.

After the OpenVLA-OFT environment is ready, activate the same conda environment
before running FocusVLA scripts.

## Repository Layout

The main FocusVLA code is in two folders:

* `training/`: data preparation and FocusVLA finetuning.
* `pruner/`: pruning/evaluation code used to test robustness under visual token pruning.

The most important training files are:

* `training/create_white_patch_rlds.py`: builds white-patch triggered RLDS datasets.
* `training/export_teacher_dataset.py`: exports benign-teacher supervision data offline.
* `training/finetune.py`: FocusVLA finetuning entry point.

## Training Pipeline

Training has three steps.

First, generate the white-patch triggered RLDS dataset. The trigger is a white
square in the top-left corner of each image and covers approximately 1% of the
image area.

```bash
python training/create_white_patch_rlds.py \
  --input-dir /path/to/modified_libero_rlds \
  --dataset-name libero_spatial_no_noops \
  --output-root /path/to/white_patch_rlds \
  --patch-area-ratio 0.01 \
  --overwrite
```

Second, export benign-teacher supervision data:

```bash
python training/export_teacher_dataset.py \
  --vla-path /path/to/openvla-oft-checkpoint \
  --data-root-dir /path/to/modified_libero_rlds \
  --dataset-name libero_spatial_no_noops \
  --output-dir /path/to/focusvla_teacher/libero_spatial/merged \
  --max-samples 40000 \
  --export-batch-size 16 \
  --records-per-shard 256 \
  --device cuda:0
```

In our experiments, we typically exported 40k teacher samples for a training run.

Third, run FocusVLA finetuning:

```bash
torchrun --standalone --nproc_per_node 1 training/finetune.py \
  --vla_path /path/to/openvla-oft-checkpoint \
  --data_root_dir /path/to/modified_libero_rlds \
  --dataset_name libero_spatial_no_noops \
  --clean_teacher_dataset_path /path/to/focusvla_teacher/libero_spatial/merged \
  --poison_data_root_dir /path/to/white_patch_rlds \
  --poison_dataset_name libero_spatial_no_noops \
  --run_root_dir /path/to/focusvla_runs/libero_spatial_white
```

## Teacher Data Export Details

FocusVLA needs benign-teacher supervision for attention hijacking. We export
this supervision offline before finetuning and should use the benign dataset.
Because our experiments use RTX 3090 GPUs and the VRAM is not enough to keep an
additional benign teacher model in memory during finetuning, we precompute the
teacher data first.

```bash
python training/export_teacher_dataset.py \
  --vla-path /path/to/openvla-oft-checkpoint \
  --data-root-dir /path/to/modified_libero_rlds \
  --dataset-name libero_spatial_no_noops \
  --output-dir /path/to/focusvla_teacher/libero_spatial/merged \
  --max-samples 40000 \
  --export-batch-size 16 \
  --records-per-shard 256 \
  --device cuda:0
```

For multi-GPU export, run multiple shards in parallel by changing
`--shard-rank` and `CUDA_VISIBLE_DEVICES`:

```bash
NUM_SHARDS=4
for RANK in 0 1 2 3
do
  CUDA_VISIBLE_DEVICES=${RANK} python training/export_teacher_dataset.py \
    --vla-path /path/to/openvla-oft-checkpoint \
    --data-root-dir /path/to/modified_libero_rlds \
    --dataset-name libero_spatial_no_noops \
    --output-dir /path/to/focusvla_teacher/libero_spatial \
    --max-samples 40000 \
    --export-batch-size 16 \
    --records-per-shard 256 \
    --num-shards ${NUM_SHARDS} \
    --shard-rank ${RANK} \
    --device cuda:0 &
done
wait
```

If you export multiple parts, merge them before training:

```bash
python training/scripts/merge_teacher_exports.py \
  --input-dir /path/to/focusvla_teacher/libero_spatial \
  --output-dir /path/to/focusvla_teacher/libero_spatial/merged
```

## Finetune FocusVLA

After creating the white-patch RLDS dataset and exporting teacher data, run:

```bash
torchrun --standalone --nproc_per_node 1 training/finetune.py \
  --vla_path /path/to/openvla-oft-checkpoint \
  --data_root_dir /path/to/modified_libero_rlds \
  --dataset_name libero_spatial_no_noops \
  --clean_teacher_dataset_path /path/to/focusvla_teacher/libero_spatial/merged \
  --poison_data_root_dir /path/to/white_patch_rlds \
  --poison_dataset_name libero_spatial_no_noops \
  --run_root_dir /path/to/focusvla_runs/libero_spatial_white
```

For multi-GPU training, increase `--nproc_per_node` and set
`CUDA_VISIBLE_DEVICES` as usual:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node 4 training/finetune.py \
  --vla_path /path/to/openvla-oft-checkpoint \
  --data_root_dir /path/to/modified_libero_rlds \
  --dataset_name libero_spatial_no_noops \
  --clean_teacher_dataset_path /path/to/focusvla_teacher/libero_spatial/merged \
  --poison_data_root_dir /path/to/white_patch_rlds \
  --poison_dataset_name libero_spatial_no_noops \
  --run_root_dir /path/to/focusvla_runs/libero_spatial_white
```

## Evaluation

The `pruner/` directory contains the pruning/evaluation code used in the paper.
We release the ten visual token pruning baselines used in our experiments.
Each method has its own implementation and config under `pruner/<METHOD>/`.

## Notes

* Replace all `/path/to/...` placeholders with paths on your machine.
* Keep the upstream OpenVLA-OFT environment active when running the scripts.

