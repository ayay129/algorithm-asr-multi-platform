FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    MODEL_DIR=/models/whisperx \
    HF_HOME=/models/whisperx \
    TRANSFORMERS_CACHE=/models/whisperx \
    HUGGINGFACE_HUB_CACHE=/models/whisperx \
    XDG_CACHE_HOME=/models/.cache \
    MALLOC_ARENA_MAX=2 \
    ASR_MAX_UPLOAD_BYTES=536870912 \
    ASR_MAX_MEDIA_DURATION_SECONDS=7200

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
COPY --from=whisperx_models . /models/whisperx

COPY --from=whisperx_src . /opt/whisperX
RUN test -f /opt/whisperX/pyproject.toml
RUN python3 -m pip install /opt/whisperX

COPY requirements-docker.txt /app/requirements-docker.txt
RUN python3 -m pip install -r /app/requirements-docker.txt

COPY service.py /app/service.py
COPY README.md /app/README.md

EXPOSE 8000

CMD ["python3", "/app/service.py", "--model", "large-v3", "--device", "cuda", "--device-index", "0", "--compute-type", "float16", "--batch-size", "16", "--download-root", "/models/whisperx", "--local-files-only", "--diarize-model", "pyannote/speaker-diarization-community-1", "--diarize-cache-mode", "keep", "--default-diarize", "--host", "0.0.0.0", "--port", "8000"]
