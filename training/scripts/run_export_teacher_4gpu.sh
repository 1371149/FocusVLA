#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  run_export_teacher_4gpu.sh

Environment overrides:
  ROOT_DIR
  PYTHON_BIN
  VLA_PATH
  DATA_ROOT_DIR
  DATASET_NAME
  OUTPUT_ROOT
  LOG_DIR
  TOTAL_SAMPLES
  EXPORT_BATCH_SIZE
  RECORDS_PER_SHARD
  ESTIMATED_SHARD_SIZE_MIB
  SHUFFLE_BUFFER_SIZE
  NUM_IMAGES_IN_INPUT
  GPU_IDS
  NUM_GPUS
  USE_PROPRIO
  INCLUDE_RAW_IMAGES
  AVERAGE_SELECTED_HEADS_FOR_STORAGE
  AUTO_MERGE

Example:
  GPU_IDS="0 1 2 3" ./scripts/run_export_teacher_4gpu.sh
  NUM_GPUS=8 ./scripts/run_export_teacher_4gpu.sh
  AVERAGE_SELECTED_HEADS_FOR_STORAGE=1 TOTAL_SAMPLES=4096 ./scripts/run_export_teacher_4gpu.sh
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

ROOT_DIR="${ROOT_DIR:-<OPENVLA_OFT_ROOT>/my_finetune}"
PYTHON_BIN="${PYTHON_BIN:-python}"
EXPORT_SCRIPT="${EXPORT_SCRIPT:-$ROOT_DIR/export_teacher_dataset.py}"

VLA_PATH="${VLA_PATH:-<MODEL_ROOT>/openvla-7b-oft-finetuned-libero-spatial}"
DATA_ROOT_DIR="${DATA_ROOT_DIR:-<OPENVLA_FINETUNE_ROOT>/dataset}"
DATASET_NAME="${DATASET_NAME:-libero_spatial_no_noops}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT_DIR/exports/libero_spatial_8head}"
LOG_DIR="${LOG_DIR:-$ROOT_DIR/logs}"

TOTAL_SAMPLES="${TOTAL_SAMPLES:-40000}"
EXPORT_BATCH_SIZE="${EXPORT_BATCH_SIZE:-2}"
RECORDS_PER_SHARD="${RECORDS_PER_SHARD:-256}"
ESTIMATED_SHARD_SIZE_MIB="${ESTIMATED_SHARD_SIZE_MIB:-600}"
SHUFFLE_BUFFER_SIZE="${SHUFFLE_BUFFER_SIZE:-1}"
NUM_IMAGES_IN_INPUT="${NUM_IMAGES_IN_INPUT:-2}"

AUTO_MERGE="${AUTO_MERGE:-0}"
USE_PROPRIO="${USE_PROPRIO:-1}"
INCLUDE_RAW_IMAGES="${INCLUDE_RAW_IMAGES:-0}"
AVERAGE_SELECTED_HEADS_FOR_STORAGE="${AVERAGE_SELECTED_HEADS_FOR_STORAGE:-0}"

NUM_GPUS="${NUM_GPUS:-}"

if [[ -n "${GPU_IDS:-}" ]]; then
  GPU_IDS=(${GPU_IDS})
else
  if command -v nvidia-smi >/dev/null 2>&1; then
    mapfile -t _detected_gpu_ids < <(nvidia-smi --query-gpu=index --format=csv,noheader,nounits | tr -d ' ')
  else
    _detected_gpu_ids=()
  fi

  if [[ "${#_detected_gpu_ids[@]}" -eq 0 ]]; then
    echo "Failed to auto-detect GPUs. Please set GPU_IDS, e.g. GPU_IDS=\"0 1 2 3\"."
    exit 1
  fi

  GPU_IDS=("${_detected_gpu_ids[@]}")
fi

if [[ -n "$NUM_GPUS" ]]; then
  if ! [[ "$NUM_GPUS" =~ ^[0-9]+$ ]] || [[ "$NUM_GPUS" -le 0 ]]; then
    echo "NUM_GPUS must be a positive integer, got: $NUM_GPUS"
    exit 1
  fi
  GPU_IDS=("${GPU_IDS[@]:0:$NUM_GPUS}")
fi

PART_COUNT="${#GPU_IDS[@]}"

if [[ "$PART_COUNT" -le 0 ]]; then
  echo "No GPU ids provided."
  exit 1
fi

mkdir -p "$OUTPUT_ROOT" "$LOG_DIR"

echo "ROOT_DIR=$ROOT_DIR"
echo "OUTPUT_ROOT=$OUTPUT_ROOT"
echo "TOTAL_SAMPLES=$TOTAL_SAMPLES"
echo "GPU_IDS=${GPU_IDS[*]}"
echo "EXPORT_BATCH_SIZE=$EXPORT_BATCH_SIZE"
echo "RECORDS_PER_SHARD=$RECORDS_PER_SHARD"
echo "ESTIMATED_SHARD_SIZE_MIB=$ESTIMATED_SHARD_SIZE_MIB"
echo "AVERAGE_SELECTED_HEADS_FOR_STORAGE=$AVERAGE_SELECTED_HEADS_FOR_STORAGE"

"$PYTHON_BIN" - <<PY
total_samples = int("${TOTAL_SAMPLES}")
records_per_shard = int("${RECORDS_PER_SHARD}")
estimated_shard_size_mib = float("${ESTIMATED_SHARD_SIZE_MIB}")
estimated_total_gib = estimated_shard_size_mib * total_samples / max(1, records_per_shard) / 1024.0
estimated_total_gb = estimated_shard_size_mib * total_samples / max(1, records_per_shard) / 1000.0
print(f"Estimated disk usage for exported data: {estimated_total_gib:.2f} GiB ({estimated_total_gb:.2f} GB)")
print("This is disk usage, not GPU memory.")
PY

build_extra_args() {
  local -a extra_args=()
  if [[ "$USE_PROPRIO" == "1" ]]; then
    extra_args+=(--use-proprio)
  else
    extra_args+=(--no-proprio)
  fi

  if [[ "$INCLUDE_RAW_IMAGES" == "1" ]]; then
    extra_args+=(--include-raw-images)
  else
    extra_args+=(--no-raw-images)
  fi

  if [[ "$AVERAGE_SELECTED_HEADS_FOR_STORAGE" == "1" ]]; then
    extra_args+=(--average-selected-heads-for-storage)
  fi

  printf '%s\n' "${extra_args[@]}"
}

mapfile -t EXTRA_ARGS < <(build_extra_args)

cleanup() {
  for pid in "${PIDS[@]:-}"; do
    if kill -0 "$pid" >/dev/null 2>&1; then
      kill "$pid" >/dev/null 2>&1 || true
    fi
  done
}

trap cleanup INT TERM

declare -a PIDS=()
declare -a PART_LABELS=()

for ((idx=0; idx<PART_COUNT; idx++)); do
  gpu="${GPU_IDS[$idx]}"
  if (( TOTAL_SAMPLES <= idx )); then
    local_count=0
  else
    local_count=$(( (TOTAL_SAMPLES - idx + PART_COUNT - 1) / PART_COUNT ))
  fi
  part_dir="$OUTPUT_ROOT/part${idx}"
  log_path="$LOG_DIR/export_teacher_part${idx}.log"

  if (( local_count <= 0 )); then
    echo "Skipping part${idx}: gpu=$gpu local_count=0"
    continue
  fi

  rm -rf "$part_dir"
  mkdir -p "$part_dir"
  rm -f "$log_path"

  echo "Launching part${idx}: gpu=$gpu local_count=$local_count shard_rank=$idx/$PART_COUNT"
  PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" "$EXPORT_SCRIPT" \
    --vla-path "$VLA_PATH" \
    --data-root-dir "$DATA_ROOT_DIR" \
    --dataset-name "$DATASET_NAME" \
    --output-dir "$part_dir" \
    --max-samples "$local_count" \
    --num-shards "$PART_COUNT" \
    --shard-rank "$idx" \
    --export-batch-size "$EXPORT_BATCH_SIZE" \
    --records-per-shard "$RECORDS_PER_SHARD" \
    --shuffle-buffer-size "$SHUFFLE_BUFFER_SIZE" \
    --num-images-in-input "$NUM_IMAGES_IN_INPUT" \
    --device "cuda:0" \
    "${EXTRA_ARGS[@]}" \
    > "$log_path" 2>&1 &

  PIDS+=("$!")
  PART_LABELS+=("part${idx}")
done

echo "Started ${#PIDS[@]} export jobs."
for ((idx=0; idx<${#PIDS[@]}; idx++)); do
  echo "  ${PART_LABELS[$idx]} pid=${PIDS[$idx]} log=$LOG_DIR/export_teacher_${PART_LABELS[$idx]}.log"
done

failed=0
for pid in "${PIDS[@]}"; do
  if ! wait "$pid"; then
    failed=1
  fi
done

if [[ "$failed" != "0" ]]; then
  echo "At least one export job failed. Check logs under $LOG_DIR."
  exit 1
fi

echo "All export jobs finished successfully."

if [[ "$AUTO_MERGE" == "1" ]]; then
  "$PYTHON_BIN" "$ROOT_DIR/scripts/merge_teacher_exports.py" \
    --root "$OUTPUT_ROOT" \
    --output "$OUTPUT_ROOT/merged"
fi

echo "Next step:"
echo "  $PYTHON_BIN $ROOT_DIR/scripts/merge_teacher_exports.py --root $OUTPUT_ROOT --output $OUTPUT_ROOT/merged"
