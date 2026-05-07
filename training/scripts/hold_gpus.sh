#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  ./scripts/hold_gpus.sh [start|stop|status|restart]

Default behavior:
  Starts a low-CPU background process that holds most free memory on GPUs 5,6,7.

Environment overrides:
  GPU_IDS      Comma-separated physical GPU ids to hold. Default: 5,6,7
  HOLD_RATIO   Fraction of currently free memory to target per visible GPU. Default: 0.85
  RESERVE_MIB  Extra free memory to leave unused on each visible GPU. Default: 1024
  CHUNK_MIB    Allocation chunk size in MiB. Default: 256
  LOG_DIR      Directory for pid/log files. Default: ../logs
  PID_FILE     Override pid file path
  LOG_FILE     Override log file path
  PYTHON_BIN   Python interpreter to use

Examples:
  ./scripts/hold_gpus.sh
  ./scripts/hold_gpus.sh status
  ./scripts/hold_gpus.sh stop
  GPU_IDS=4,5,6 HOLD_RATIO=0.90 ./scripts/hold_gpus.sh restart
EOF
}

ACTION="${1:-start}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
LOG_DIR="${LOG_DIR:-$ROOT_DIR/logs}"
GPU_IDS="${GPU_IDS:-5,6,7}"
HOLD_RATIO="${HOLD_RATIO:-0.85}"
RESERVE_MIB="${RESERVE_MIB:-1024}"
CHUNK_MIB="${CHUNK_MIB:-256}"
GPU_TAG="${GPU_IDS//,/}"
GPU_TAG="${GPU_TAG// /}"
PID_FILE="${PID_FILE:-$LOG_DIR/hold_gpus_${GPU_TAG}.pid}"
LOG_FILE="${LOG_FILE:-$LOG_DIR/hold_gpus_${GPU_TAG}.log}"

pick_python() {
  if [[ -n "${PYTHON_BIN:-}" ]]; then
    echo "$PYTHON_BIN"
    return
  fi

  local candidate
  for candidate in \
    python \
    "$(command -v python3 2>/dev/null || true)" \
    "$(command -v python 2>/dev/null || true)"
  do
    if [[ -n "$candidate" && -x "$candidate" ]]; then
      echo "$candidate"
      return
    fi
  done

  echo "No usable Python interpreter found." >&2
  exit 1
}

PYTHON_BIN="$(pick_python)"

is_running() {
  [[ -f "$PID_FILE" ]] || return 1
  local pid
  pid="$(<"$PID_FILE")"
  [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null
}

validate_gpu_ids() {
  if [[ -z "$GPU_IDS" ]]; then
    echo "GPU_IDS is empty." >&2
    return 1
  fi

  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "Warning: nvidia-smi not found, skipping GPU id validation." >&2
    return 0
  fi

  local available
  available="$(nvidia-smi --query-gpu=index --format=csv,noheader,nounits | tr -d ' ')"
  local requested=()
  local gpu_id
  IFS=',' read -r -a requested <<<"$GPU_IDS"

  for gpu_id in "${requested[@]}"; do
    gpu_id="${gpu_id// /}"
    if [[ -z "$gpu_id" ]]; then
      echo "GPU_IDS contains an empty entry: $GPU_IDS" >&2
      return 1
    fi
    if ! grep -Fxq "$gpu_id" <<<"$available"; then
      echo "GPU $gpu_id does not exist on this machine." >&2
      echo "Available GPU indices: $(paste -sd, <<<"$available")" >&2
      return 1
    fi
  done
}

start_hold() {
  mkdir -p "$LOG_DIR"
  validate_gpu_ids

  if is_running; then
    local pid
    pid="$(<"$PID_FILE")"
    echo "GPU hold already running: pid=$pid"
    echo "Log: $LOG_FILE"
    return 0
  fi

  rm -f "$PID_FILE"

  nohup env \
    CUDA_VISIBLE_DEVICES="$GPU_IDS" \
    HOLD_RATIO="$HOLD_RATIO" \
    RESERVE_MIB="$RESERVE_MIB" \
    CHUNK_MIB="$CHUNK_MIB" \
    "$PYTHON_BIN" -u - >"$LOG_FILE" 2>&1 <<'PY' &
import os
import signal
import sys
import time

try:
    import torch
except Exception as exc:
    print(f"Failed to import torch: {exc}", flush=True)
    raise

buffers = []


class StopHold(Exception):
    pass


def handle_exit(signum, _frame):
    print(f"Received signal {signum}, releasing GPU memory...", flush=True)
    raise StopHold


signal.signal(signal.SIGTERM, handle_exit)
signal.signal(signal.SIGINT, handle_exit)

ratio = float(os.environ.get("HOLD_RATIO", "0.85"))
reserve_mib = int(os.environ.get("RESERVE_MIB", "1024"))
chunk_mib = max(int(os.environ.get("CHUNK_MIB", "256")), 1)
visible_gpu_ids = os.environ.get("CUDA_VISIBLE_DEVICES", "")

if not torch.cuda.is_available():
    print("CUDA is not available; nothing to hold.", flush=True)
    sys.exit(1)

device_count = torch.cuda.device_count()
reserve_bytes = reserve_mib * 1024 * 1024
chunk_bytes = chunk_mib * 1024 * 1024

print(
    f"Starting GPU hold on physical GPUs [{visible_gpu_ids}] "
    f"(visible count={device_count}, ratio={ratio}, reserve={reserve_mib} MiB)",
    flush=True,
)

for idx in range(device_count):
    name = torch.cuda.get_device_name(idx)
    free_bytes, total_bytes = torch.cuda.mem_get_info(idx)
    target_bytes = max(int(free_bytes * ratio) - reserve_bytes, 0)
    allocated = 0
    device_buffers = []

    if target_bytes <= 0:
      print(
          f"cuda:{idx} {name}: skipping, free={free_bytes / 2**30:.2f} GiB",
          flush=True,
      )
      buffers.append(device_buffers)
      continue

    while allocated < target_bytes:
        step = min(chunk_bytes, target_bytes - allocated)
        try:
            device_buffers.append(
                torch.empty(step, dtype=torch.uint8, device=f"cuda:{idx}")
            )
            allocated += step
        except RuntimeError as exc:
            print(
                f"cuda:{idx} {name}: allocation stopped at "
                f"{allocated / 2**30:.2f} GiB ({exc})",
                flush=True,
            )
            break

    buffers.append(device_buffers)
    print(
        f"cuda:{idx} {name}: held {allocated / 2**30:.2f} GiB, "
        f"free_before={free_bytes / 2**30:.2f} GiB, total={total_bytes / 2**30:.2f} GiB",
        flush=True,
    )

print("GPU hold is active. Kill this process to release memory.", flush=True)

try:
    while True:
        time.sleep(600)
except (KeyboardInterrupt, StopHold):
    print("Interrupted, releasing GPU memory...", flush=True)

buffers.clear()
torch.cuda.empty_cache()
print("Released GPU memory.", flush=True)
PY

  local pid="$!"
  echo "$pid" >"$PID_FILE"
  sleep 1

  if kill -0 "$pid" 2>/dev/null; then
    echo "Started GPU hold."
    echo "PID: $pid"
    echo "GPUs: $GPU_IDS"
    echo "Log: $LOG_FILE"
    echo "Stop: $0 stop"
    return 0
  fi

  rm -f "$PID_FILE"
  echo "Failed to start GPU hold. Check log: $LOG_FILE" >&2
  return 1
}

stop_hold() {
  if [[ ! -f "$PID_FILE" ]]; then
    echo "No pid file found: $PID_FILE"
    return 0
  fi

  local pid
  pid="$(<"$PID_FILE")"

  if [[ -z "$pid" ]]; then
    rm -f "$PID_FILE"
    echo "Removed empty pid file."
    return 0
  fi

  if ! kill -0 "$pid" 2>/dev/null; then
    rm -f "$PID_FILE"
    echo "GPU hold is not running. Removed stale pid file."
    return 0
  fi

  kill "$pid"

  local waited=0
  while kill -0 "$pid" 2>/dev/null; do
    if (( waited >= 15 )); then
      echo "Process $pid did not exit after SIGTERM; sending SIGKILL." >&2
      kill -9 "$pid" 2>/dev/null || true
      sleep 1
      if kill -0 "$pid" 2>/dev/null; then
        echo "Process $pid is still alive after SIGKILL. Check manually." >&2
        return 1
      fi
      rm -f "$PID_FILE"
      echo "Stopped GPU hold with SIGKILL: pid=$pid"
      return 0
    fi
    sleep 1
    waited=$((waited + 1))
  done

  rm -f "$PID_FILE"
  echo "Stopped GPU hold: pid=$pid"
}

status_hold() {
  if is_running; then
    local pid
    pid="$(<"$PID_FILE")"
    echo "GPU hold is running."
    echo "PID: $pid"
    echo "GPUs: $GPU_IDS"
    echo "Log: $LOG_FILE"
    ps -p "$pid" -o pid=,etime=,cmd=
    if [[ -f "$LOG_FILE" ]]; then
      echo
      echo "Recent log:"
      tail -n 20 "$LOG_FILE"
    fi
    return 0
  fi

  if [[ -f "$PID_FILE" ]]; then
    echo "GPU hold is not running, stale pid file present: $PID_FILE"
  else
    echo "GPU hold is not running."
  fi
  echo "Expected GPUs: $GPU_IDS"
}

case "$ACTION" in
  start)
    start_hold
    ;;
  stop)
    stop_hold
    ;;
  status)
    status_hold
    ;;
  restart)
    stop_hold || true
    start_hold
    ;;
  -h|--help|help)
    usage
    ;;
  *)
    usage >&2
    exit 1
    ;;
esac
