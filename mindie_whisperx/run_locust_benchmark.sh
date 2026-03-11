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
DEFAULT_NPU_MONITOR_ENABLE="1"
DEFAULT_NPU_DEVICE_ID="0"
DEFAULT_NPU_POLL_INTERVAL_SECONDS="1"
DEFAULT_NPU_SMI_BIN="npu-smi"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

LOCUST_FILE_PATH="${LOCUST_FILE_PATH:-}"
PLOT_SCRIPT_PATH="${PLOT_SCRIPT_PATH:-}"

if [[ -z "${LOCUST_FILE_PATH}" ]]; then
  if [[ -f "${SCRIPT_DIR}/locustfile.py" ]]; then
    LOCUST_FILE_PATH="${SCRIPT_DIR}/locustfile.py"
  elif [[ -f "${SCRIPT_DIR}/mindie_whisperx/locustfile.py" ]]; then
    LOCUST_FILE_PATH="${SCRIPT_DIR}/mindie_whisperx/locustfile.py"
  else
    echo "locustfile.py not found."
    echo "Tried:"
    echo "  ${SCRIPT_DIR}/locustfile.py"
    echo "  ${SCRIPT_DIR}/mindie_whisperx/locustfile.py"
    echo "You can also set LOCUST_FILE_PATH=/abs/path/to/locustfile.py"
    exit 1
  fi
fi

if [[ -z "${PLOT_SCRIPT_PATH}" ]]; then
  if [[ -f "${SCRIPT_DIR}/plot_locust_results.py" ]]; then
    PLOT_SCRIPT_PATH="${SCRIPT_DIR}/plot_locust_results.py"
  elif [[ -f "${SCRIPT_DIR}/mindie_whisperx/plot_locust_results.py" ]]; then
    PLOT_SCRIPT_PATH="${SCRIPT_DIR}/mindie_whisperx/plot_locust_results.py"
  else
    echo "plot_locust_results.py not found."
    echo "Tried:"
    echo "  ${SCRIPT_DIR}/plot_locust_results.py"
    echo "  ${SCRIPT_DIR}/mindie_whisperx/plot_locust_results.py"
    echo "You can also set PLOT_SCRIPT_PATH=/abs/path/to/plot_locust_results.py"
    exit 1
  fi
fi

if [[ ! -f "${LOCUST_FILE_PATH}" ]]; then
  echo "LOCUST_FILE_PATH does not exist: ${LOCUST_FILE_PATH}"
  exit 1
fi

if [[ ! -f "${PLOT_SCRIPT_PATH}" ]]; then
  echo "PLOT_SCRIPT_PATH does not exist: ${PLOT_SCRIPT_PATH}"
  exit 1
fi

cd "${SCRIPT_DIR}"

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
NPU_MONITOR_ENABLE="${NPU_MONITOR_ENABLE:-${DEFAULT_NPU_MONITOR_ENABLE}}"
NPU_DEVICE_ID="${NPU_DEVICE_ID:-${DEFAULT_NPU_DEVICE_ID}}"
NPU_POLL_INTERVAL_SECONDS="${NPU_POLL_INTERVAL_SECONDS:-${DEFAULT_NPU_POLL_INTERVAL_SECONDS}}"
NPU_SMI_BIN="${NPU_SMI_BIN:-${DEFAULT_NPU_SMI_BIN}}"

if [[ ! -d "${AUDIO_DIR}" ]]; then
  echo "AUDIO_DIR is not a directory: ${AUDIO_DIR}"
  echo "Please edit DEFAULT_AUDIO_DIR in this script or pass AUDIO_DIR=/your/path"
  exit 1
fi

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
OUT_DIR="${OUT_ROOT}/${RUN_ID}"
CSV_PREFIX="${OUT_DIR}/locust"
NPU_LOG_PATH="${OUT_DIR}/npu_watch.log"

mkdir -p "${OUT_DIR}"

export AUDIO_DIR ASR_ENDPOINT ASR_LANGUAGE ASR_BATCH_SIZE ASR_MODEL
export ASR_REQUEST_TIMEOUT LOCUST_WAIT_MIN LOCUST_WAIT_MAX

NPU_MONITOR_PID=""

stop_npu_monitor() {
  if [[ -n "${NPU_MONITOR_PID}" ]] && kill -0 "${NPU_MONITOR_PID}" >/dev/null 2>&1; then
    kill "${NPU_MONITOR_PID}" >/dev/null 2>&1 || true
    wait "${NPU_MONITOR_PID}" >/dev/null 2>&1 || true
  fi
  NPU_MONITOR_PID=""
}

cleanup() {
  stop_npu_monitor
}

trap cleanup EXIT INT TERM

start_npu_monitor() {
  if [[ "${NPU_MONITOR_ENABLE}" != "1" ]]; then
    echo "  npu_monitor=disabled"
    return 0
  fi
  if ! command -v "${NPU_SMI_BIN}" >/dev/null 2>&1; then
    echo "  npu_monitor=disabled (${NPU_SMI_BIN} not found)"
    return 0
  fi
  (
    while true; do
      echo "### SAMPLE_START $(date +%s)"
      "${NPU_SMI_BIN}" info -i "${NPU_DEVICE_ID}" || echo "[npu-monitor] ${NPU_SMI_BIN} failed"
      echo "### SAMPLE_END"
      sleep "${NPU_POLL_INTERVAL_SECONDS}"
    done
  ) >"${NPU_LOG_PATH}" 2>&1 &
  NPU_MONITOR_PID=$!
  echo "  npu_monitor=enabled pid=${NPU_MONITOR_PID} device=${NPU_DEVICE_ID} interval=${NPU_POLL_INTERVAL_SECONDS}s"
}

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
echo "  npu_log=${NPU_LOG_PATH}"
echo "  locust_file=${LOCUST_FILE_PATH}"
echo "  plot_script=${PLOT_SCRIPT_PATH}"

start_npu_monitor

locust -f "${LOCUST_FILE_PATH}" \
  --host "${HOST}" \
  --headless \
  -u "${LOCUST_USERS}" \
  -r "${LOCUST_SPAWN_RATE}" \
  -t "${LOCUST_RUN_TIME}" \
  --csv "${CSV_PREFIX}" \
  --csv-full-history

stop_npu_monitor

python3 "${PLOT_SCRIPT_PATH}" \
  --csv-prefix "${CSV_PREFIX}" \
  --out-dir "${OUT_DIR}" \
  --npu-log "${NPU_LOG_PATH}"

echo "Done. Results written to ${OUT_DIR}"
