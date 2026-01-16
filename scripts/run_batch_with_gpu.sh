#!/usr/bin/env bash
set -u -o pipefail

INPUT_DIR="/workspace/GeometryCrafter/workspace/inputs/trimmed"
LOG_DIR="/workspace/GeometryCrafter/workspace/test_runs_$(date +%Y%m%d_%H%M%S)"
DRY_RUN=0

mkdir -p "$LOG_DIR"

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi not found. GPU monitoring requires NVIDIA drivers." >&2
  exit 1
fi

if ! command -v python >/dev/null 2>&1; then
  echo "python not found in PATH." >&2
  exit 1
fi

for arg in "$@"; do
  case "$arg" in
    --dry-run) DRY_RUN=1 ;;
    *) echo "Unknown arg: $arg" >&2; exit 1 ;;
  esac
done

# File-specific target sizes
files=(
  "12525562-uhd_3840_2160_60fps_4s.mp4"
  "15101060_3840_2160_30fps_4s.mp4"
  "15102897_2560_1440_60fps_4s.mp4"
  "3111683-uhd_3840_2160_25fps_4s.mp4"
  "6296696-uhd_2560_1080_25fps_4s.mp4"
  "6789663-hd_4096_2160_25fps_4s.mp4"
  "8466293-uhd_3840_2160_25fps_4s.mp4"
)

run_one() {
  local filename="$1"
  local input="$INPUT_DIR/$filename"

  if [[ ! -f "$input" ]]; then
    echo "Missing input: $input" >&2
    return 1
  fi

  local base
  base=$(basename "$filename" .mp4)

  local run_log="$LOG_DIR/${base}.log"
  local gpu_log="$LOG_DIR/${base}_gpu.csv"

  local mon_pid=""
  if (( DRY_RUN == 0 )); then
    echo "timestamp,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,temperature.gpu" > "$gpu_log"

    # Start GPU monitor in the background.
    (
      while true; do
        local ts
        ts=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
        nvidia-smi --query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,temperature.gpu \
          --format=csv,noheader,nounits | awk -v ts="$ts" '{print ts","$0}'
        sleep 1
      done
    ) >> "$gpu_log" &
    mon_pid=$!
  fi

  local cmd=(python run.py "$input" --downsample_ratio 2)
  local npz_path="/workspace/GeometryCrafter/workspace/output/${base}.npz"
  local mp4_path="/workspace/GeometryCrafter/workspace/output/${base}.mp4"
  local post_cmd=(./bin/npz_to_mp4.py --npz_path "$npz_path" --output_path "$mp4_path")
  echo "Running: ${cmd[*]}" | tee -a "$run_log"
  if (( DRY_RUN == 1 )); then
    local status=0
  else
    "${cmd[@]}" >> "$run_log" 2>&1
    local status=$?

    kill "$mon_pid" 2>/dev/null || true
    wait "$mon_pid" 2>/dev/null || true
  fi

  if (( DRY_RUN == 0 )) && (( status == 0 )) && [[ -f "$npz_path" ]]; then
    echo "Post-processing: ${post_cmd[*]}" | tee -a "$run_log"
    "${post_cmd[@]}" >> "$run_log" 2>&1
  fi

  echo "exit_status=$status" | tee -a "$run_log"
  return "$status"
}

for f in "${files[@]}"; do
  run_one "$f" || true
  echo "---" | tee -a "$LOG_DIR/summary.log"
  echo "Completed $f at $(date -u +"%Y-%m-%dT%H:%M:%SZ")" | tee -a "$LOG_DIR/summary.log"
  echo "---" | tee -a "$LOG_DIR/summary.log"
  sleep 1
  done

echo "Logs written to: $LOG_DIR"
