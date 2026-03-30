FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04

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

EXPOSE 8000

CMD ["python3", "/app/service.py", "--model", "large-v3", "--device", "cuda", "--device-index", "0", "--compute-type", "float16", "--batch-size", "16", "--download-root", "/models/whisperx", "--local-files-only", "--host", "0.0.0.0", "--port", "8000"]
