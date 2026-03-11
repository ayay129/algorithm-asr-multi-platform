#!/usr/bin/env bash
set -euo pipefail

# ----------------------------------------
# Local default config (can be hard-coded)
# ----------------------------------------
DEFAULT_AUDIO_DIR="/path/to/media"
DEFAULT_HOST="http://127.0.0.1:8000"
DEFAULT_LOCUST_USERS="20"
DEFAULT_LOCUST_SPAWN_RATE="2"
DEFAULT_LOCUST_RUN_TIME="5m"
DEFAULT_ASR_ENDPOINT="/v1/audio/transcriptions"
DEFAULT_ASR_LANGUAGE="zh"
DEFAULT_ASR_BATCH_SIZE="16"
DEFAULT_ASR_MODEL=""
DEFAULT_ASR_REQUEST_TIMEOUT="3600"
DEFAULT_LOCUST_WAIT_MIN="0"
DEFAULT_LOCUST_WAIT_MAX="0"
DEFAULT_OUT_ROOT="benchmarks"

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

AUDIO_DIR="${AUDIO_DIR:-${DEFAULT_AUDIO_DIR}}"
HOST="${HOST:-${DEFAULT_HOST}}"
LOCUST_USERS="${LOCUST_USERS:-${DEFAULT_LOCUST_USERS}}"
LOCUST_SPAWN_RATE="${LOCUST_SPAWN_RATE:-${DEFAULT_LOCUST_SPAWN_RATE}}"
LOCUST_RUN_TIME="${LOCUST_RUN_TIME:-${DEFAULT_LOCUST_RUN_TIME}}"
ASR_ENDPOINT="${ASR_ENDPOINT:-${DEFAULT_ASR_ENDPOINT}}"
ASR_LANGUAGE="${ASR_LANGUAGE:-${DEFAULT_ASR_LANGUAGE}}"
ASR_BATCH_SIZE="${ASR_BATCH_SIZE:-${DEFAULT_ASR_BATCH_SIZE}}"
ASR_MODEL="${ASR_MODEL:-${DEFAULT_ASR_MODEL}}"
ASR_REQUEST_TIMEOUT="${ASR_REQUEST_TIMEOUT:-${DEFAULT_ASR_REQUEST_TIMEOUT}}"
LOCUST_WAIT_MIN="${LOCUST_WAIT_MIN:-${DEFAULT_LOCUST_WAIT_MIN}}"
LOCUST_WAIT_MAX="${LOCUST_WAIT_MAX:-${DEFAULT_LOCUST_WAIT_MAX}}"
OUT_ROOT="${OUT_ROOT:-${DEFAULT_OUT_ROOT}}"

if [[ ! -d "${AUDIO_DIR}" ]]; then
  echo "AUDIO_DIR is not a directory: ${AUDIO_DIR}"
  echo "Please edit DEFAULT_AUDIO_DIR in this script or pass AUDIO_DIR=/your/path"
  exit 1
fi

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
OUT_DIR="${OUT_ROOT}/${RUN_ID}"
CSV_PREFIX="${OUT_DIR}/locust"

mkdir -p "${OUT_DIR}"

export AUDIO_DIR ASR_ENDPOINT ASR_LANGUAGE ASR_BATCH_SIZE ASR_MODEL
export ASR_REQUEST_TIMEOUT LOCUST_WAIT_MIN LOCUST_WAIT_MAX

echo "Running benchmark..."
echo "  host=${HOST}"
echo "  endpoint=${ASR_ENDPOINT}"
echo "  users=${LOCUST_USERS}"
echo "  spawn_rate=${LOCUST_SPAWN_RATE}"
echo "  run_time=${LOCUST_RUN_TIME}"
echo "  audio_dir=${AUDIO_DIR}"
echo "  language=${ASR_LANGUAGE}"
echo "  batch_size=${ASR_BATCH_SIZE}"
echo "  out_dir=${OUT_DIR}"

locust -f mindie_whisperx/locustfile.py \
  --host "${HOST}" \
  --headless \
  -u "${LOCUST_USERS}" \
  -r "${LOCUST_SPAWN_RATE}" \
  -t "${LOCUST_RUN_TIME}" \
  --csv "${CSV_PREFIX}" \
  --csv-full-history

python3 mindie_whisperx/plot_locust_results.py \
  --csv-prefix "${CSV_PREFIX}" \
  --out-dir "${OUT_DIR}"

echo "Done. Results written to ${OUT_DIR}"
