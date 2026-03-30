FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/opt/huggingface \
    XDG_CACHE_HOME=/opt/.cache

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

COPY requirements.txt /app/requirements.txt
RUN python3 -m pip install -r /app/requirements.txt

COPY service.py /app/service.py
COPY README.md /app/README.md

EXPOSE 8000

CMD ["python3", "/app/service.py", "--model", "large-v3", "--device", "cuda", "--device-index", "0", "--compute-type", "float16", "--batch-size", "16", "--host", "0.0.0.0", "--port", "8000"]
