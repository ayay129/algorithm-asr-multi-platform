#!/usr/bin/env bash
set -u

LOG_FILE="${ASR_MT_LOG:-largev3.log}"
MATCH="${ASR_MT_MATCH:-large-v3}"
INTERVAL="${ASR_MT_INTERVAL:-30}"

if [[ "${1:-}" != "--foreground" ]]; then
  nohup bash "$0" --foreground >/dev/null 2>&1 &
  echo "monitor started pid=$! log=${LOG_FILE} match=${MATCH} interval=${INTERVAL}s"
  exit 0
fi

while true; do
  {
    echo "===== $(date "+%F %T") ====="
    ps -eo pid,%cpu,%mem,vsz,rss,args --sort=-rss | awk -v match="${MATCH}" '
      index($0, match) {
        if ($6 == "awk" || $6 ~ /\/awk$/) {
          next
        }
        rss_g = $5 / 1024 / 1024
        virt_g = $4 / 1024 / 1024
        total_rss += rss_g
        count += 1
        cmd = ""
        for (i = 6; i <= NF; i++) {
          cmd = cmd (i == 6 ? "" : " ") $i
        }
        printf "PID=%s CPU=%s%% MEM=%s%% VIRT=%.2fG RES=%.2fG CMD=%s\n", $1, $2, $3, virt_g, rss_g, cmd
      }
      END {
        printf "TOTAL_PROCS=%d TOTAL_RES=%.2fG\n", count, total_rss
      }
    '
    if command -v nvidia-smi >/dev/null 2>&1; then
      nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits 2>/dev/null \
        | awk '{print "NVIDIA_SMI " $0}'
    fi
  } >> "${LOG_FILE}"
  sleep "${INTERVAL}"
done
