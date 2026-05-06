#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  container_mem_probe.sh --container NAME --audio /path/audio.wav [options]

Options:
  --url URL             Request URL. Default: http://127.0.0.1:8000/transcribe
  --count N            Number of requests. Default: 20
  --diarize true|false Request diarize form value. Default: false
  --interval SEC       Background sampling interval. Default: 1
  --cooldown SEC       Sleep after each request. Default: 2
  --timeout SEC        Curl total request timeout. Default: 600
  --out-dir DIR        Output directory. Default: ./mem_probe_YYYYmmdd_HHMMSS
  --help               Show this help.

Outputs:
  mem.csv              Time series with RES/VIRT in KiB.
  request_summary.csv  Per-request curl timing/status.
  responses/           Raw JSON responses.

CSV columns:
  ts,phase,iter,main_res_kb,main_virt_kb,tree_res_kb,tree_virt_kb,pids

Notes:
  main_* is the container init process memory.
  tree_* sums the container init process and all descendants.
EOF
}

CONTAINER=""
AUDIO=""
URL="http://127.0.0.1:8000/transcribe"
COUNT="20"
DIARIZE="false"
INTERVAL="1"
COOLDOWN="2"
TIMEOUT="600"
OUT_DIR=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --container)
      CONTAINER="${2:-}"
      shift 2
      ;;
    --audio)
      AUDIO="${2:-}"
      shift 2
      ;;
    --url)
      URL="${2:-}"
      shift 2
      ;;
    --count)
      COUNT="${2:-}"
      shift 2
      ;;
    --diarize)
      DIARIZE="${2:-}"
      shift 2
      ;;
    --interval)
      INTERVAL="${2:-}"
      shift 2
      ;;
    --cooldown)
      COOLDOWN="${2:-}"
      shift 2
      ;;
    --timeout)
      TIMEOUT="${2:-}"
      shift 2
      ;;
    --out-dir)
      OUT_DIR="${2:-}"
      shift 2
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ -z "$CONTAINER" || -z "$AUDIO" ]]; then
  usage >&2
  exit 2
fi

if [[ ! -f "$AUDIO" ]]; then
  echo "Audio file not found: $AUDIO" >&2
  exit 1
fi

if ! command -v docker >/dev/null 2>&1; then
  echo "docker command not found" >&2
  exit 1
fi

if ! command -v curl >/dev/null 2>&1; then
  echo "curl command not found" >&2
  exit 1
fi

if [[ -z "$OUT_DIR" ]]; then
  OUT_DIR="./mem_probe_$(date +%Y%m%d_%H%M%S)"
fi

mkdir -p "$OUT_DIR/responses"

MEM_CSV="$OUT_DIR/mem.csv"
REQ_CSV="$OUT_DIR/request_summary.csv"
STOP_FILE="$OUT_DIR/.stop"

ROOT_PID="$(docker inspect -f '{{.State.Pid}}' "$CONTAINER")"
if [[ -z "$ROOT_PID" || "$ROOT_PID" == "0" || ! -d "/proc/$ROOT_PID" ]]; then
  echo "Container is not running or PID is not visible on host: $CONTAINER" >&2
  exit 1
fi

descendants() {
  local root="$1"
  local -a queue=("$root")
  local -a all=("$root")
  local p child

  while ((${#queue[@]})); do
    p="${queue[0]}"
    queue=("${queue[@]:1}")
    while read -r child; do
      [[ -n "$child" ]] || continue
      all+=("$child")
      queue+=("$child")
    done < <(ps -eo pid=,ppid= | awk -v p="$p" '$2 == p {print $1}')
  done

  printf '%s\n' "${all[@]}" | awk '!seen[$0]++'
}

proc_mem_kb() {
  local pid="$1"
  local rss="0"
  local virt="0"

  if [[ -r "/proc/$pid/status" ]]; then
    rss="$(awk '/^VmRSS:/ {print $2}' "/proc/$pid/status")"
    virt="$(awk '/^VmSize:/ {print $2}' "/proc/$pid/status")"
  fi

  printf '%s %s\n' "${rss:-0}" "${virt:-0}"
}

tree_mem_kb() {
  local rss_sum=0
  local virt_sum=0
  local pid rss virt
  local pids=()

  while read -r pid; do
    [[ -n "$pid" ]] || continue
    pids+=("$pid")
    read -r rss virt < <(proc_mem_kb "$pid")
    rss_sum=$((rss_sum + rss))
    virt_sum=$((virt_sum + virt))
  done < <(descendants "$ROOT_PID")

  printf '%s %s %s\n' "$rss_sum" "$virt_sum" "${#pids[@]}"
}

sample() {
  local phase="$1"
  local iter="$2"
  local main_rss main_virt tree_rss tree_virt pids

  read -r main_rss main_virt < <(proc_mem_kb "$ROOT_PID")
  read -r tree_rss tree_virt pids < <(tree_mem_kb)

  printf '%s,%s,%s,%s,%s,%s,%s,%s\n' \
    "$(date +%s)" \
    "$phase" \
    "$iter" \
    "$main_rss" \
    "$main_virt" \
    "$tree_rss" \
    "$tree_virt" \
    "$pids" >> "$MEM_CSV"
}

monitor_loop() {
  while [[ ! -f "$STOP_FILE" && -d "/proc/$ROOT_PID" ]]; do
    sample "sample" ""
    sleep "$INTERVAL"
  done
}

summarize() {
  awk -F, '
    NR == 1 { next }
    $4 > max_main_rss { max_main_rss = $4 }
    $5 > max_main_virt { max_main_virt = $5 }
    $6 > max_tree_rss { max_tree_rss = $6 }
    $7 > max_tree_virt { max_tree_virt = $7 }
    $2 == "baseline" { base_tree_rss = $6; base_tree_virt = $7 }
    $2 == "final" { final_tree_rss = $6; final_tree_virt = $7 }
    END {
      printf("max_main_res_mb=%.1f\n", max_main_rss / 1024)
      printf("max_main_virt_mb=%.1f\n", max_main_virt / 1024)
      printf("max_tree_res_mb=%.1f\n", max_tree_rss / 1024)
      printf("max_tree_virt_mb=%.1f\n", max_tree_virt / 1024)
      if (base_tree_rss != "") {
        printf("tree_res_delta_final_minus_baseline_mb=%.1f\n", (final_tree_rss - base_tree_rss) / 1024)
        printf("tree_virt_delta_final_minus_baseline_mb=%.1f\n", (final_tree_virt - base_tree_virt) / 1024)
      }
    }
  ' "$MEM_CSV" | tee "$OUT_DIR/summary.txt"
}

echo "container=$CONTAINER" | tee "$OUT_DIR/run_info.txt"
echo "root_pid=$ROOT_PID" | tee -a "$OUT_DIR/run_info.txt"
echo "audio=$AUDIO" | tee -a "$OUT_DIR/run_info.txt"
echo "url=$URL" | tee -a "$OUT_DIR/run_info.txt"
echo "count=$COUNT" | tee -a "$OUT_DIR/run_info.txt"
echo "diarize=$DIARIZE" | tee -a "$OUT_DIR/run_info.txt"
echo "interval=$INTERVAL" | tee -a "$OUT_DIR/run_info.txt"
echo "cooldown=$COOLDOWN" | tee -a "$OUT_DIR/run_info.txt"
echo "timeout=$TIMEOUT" | tee -a "$OUT_DIR/run_info.txt"

echo "ts,phase,iter,main_res_kb,main_virt_kb,tree_res_kb,tree_virt_kb,pids" > "$MEM_CSV"
echo "iter,http_code,time_total,size_download,response_file" > "$REQ_CSV"

monitor_loop &
MONITOR_PID="$!"
trap 'touch "$STOP_FILE"; wait "$MONITOR_PID" 2>/dev/null || true' EXIT

sample "baseline" "0"

for iter in $(seq 1 "$COUNT"); do
  response_file="$OUT_DIR/responses/response_${iter}.json"
  curl_metrics_file="$OUT_DIR/curl_metrics_${iter}.txt"

  sample "before" "$iter"
  echo "request $iter/$COUNT"

  curl -sS \
    --connect-timeout 10 \
    --max-time "$TIMEOUT" \
    -o "$response_file" \
    -w "%{http_code},%{time_total},%{size_download}" \
    -F "file=@${AUDIO}" \
    -F "diarize=${DIARIZE}" \
    "$URL" > "$curl_metrics_file" || true

  curl_metrics="$(cat "$curl_metrics_file")"
  printf '%s,%s,%s\n' "$iter" "$curl_metrics" "$response_file" >> "$REQ_CSV"

  sample "after" "$iter"
  sleep "$COOLDOWN"
done

sample "final" "$COUNT"
touch "$STOP_FILE"
wait "$MONITOR_PID" 2>/dev/null || true
trap - EXIT

summarize

echo "Output directory: $OUT_DIR"
echo "Memory CSV: $MEM_CSV"
echo "Request CSV: $REQ_CSV"
