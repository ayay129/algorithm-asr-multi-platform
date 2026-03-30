# syntax=docker/dockerfile:1.7
FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04

ARG PRELOAD_MODELS=0
ARG PRELOAD_MODEL_NAME=large-v3
ARG PRELOAD_LANGUAGES=zh
ARG INCLUDE_DIARIZATION=0
ARG DIARIZE_MODEL_REPO=pyannote/speaker-diarization-community-1
ARG HF_TOKEN_BUILD=

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    MODEL_DIR=/models/whisperx \
    HF_HOME=/models/whisperx \
    TRANSFORMERS_CACHE=/models/whisperx \
    HUGGINGFACE_HUB_CACHE=/models/whisperx \
    XDG_CACHE_HOME=/models/.cache

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-venv \
    git \
    ffmpeg \
    libsndfile1 \
    libgomp1 \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Keep torch aligned with the current upstream WhisperX guidance for CUDA 12.8.
RUN python3 -m pip install --upgrade pip setuptools wheel && \
    python3 -m pip install \
    torch==2.8.0 \
    torchaudio==2.8.0 \
    --index-url https://download.pytorch.org/whl/cu128

RUN mkdir -p /models/whisperx /models/.cache

COPY requirements.txt /app/requirements.txt
RUN python3 -m pip install -r /app/requirements.txt

COPY service.py /app/service.py
COPY download_models.py /app/download_models.py
COPY README.md /app/README.md

RUN --mount=type=secret,id=hf_token,required=false \
    HF_TOKEN_VALUE="${HF_TOKEN_BUILD}" && \
    if [ -f /run/secrets/hf_token ]; then HF_TOKEN_VALUE="$(cat /run/secrets/hf_token)"; fi && \
    if [ "${PRELOAD_MODELS}" = "1" ]; then \
      PRELOAD_ARGS="--model ${PRELOAD_MODEL_NAME} --download-root /models/whisperx --languages ${PRELOAD_LANGUAGES}"; \
      if [ "${INCLUDE_DIARIZATION}" = "1" ]; then \
        if [ -z "${HF_TOKEN_VALUE}" ]; then \
          echo "HF token is required when INCLUDE_DIARIZATION=1" >&2; \
          exit 1; \
        fi; \
        PRELOAD_ARGS="${PRELOAD_ARGS} --include-diarization --diarize-model ${DIARIZE_MODEL_REPO} --hf-token ${HF_TOKEN_VALUE}"; \
      fi; \
      python3 /app/download_models.py ${PRELOAD_ARGS}; \
    fi

EXPOSE 8000

CMD ["python3", "/app/service.py", "--model", "large-v3", "--device", "cuda", "--device-index", "0", "--compute-type", "float16", "--batch-size", "16", "--download-root", "/models/whisperx", "--local-files-only", "--diarize-model", "/models/whisperx/speaker-diarization-community-1", "--host", "0.0.0.0", "--port", "8000"]
