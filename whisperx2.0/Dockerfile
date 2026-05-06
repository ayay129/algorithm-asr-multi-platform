FROM pytorch/pytorch:2.2.2-cuda11.8-cudnn8-runtime

ARG APT_MIRROR=http://mirrors.aliyun.com/ubuntu
ENV DEBIAN_FRONTEND=noninteractive

# PyAV 编译依赖 + 运行时 ffmpeg
RUN set -eux; \
    find /etc/apt \( -name '*.list' -o -name '*.sources' \) -type f \
        -exec sed -i -E "s#https?://([a-zA-Z0-9.-]+\\.)?ubuntu.com/ubuntu/?#${APT_MIRROR}#g" {} +; \
    rm -rf /var/lib/apt/lists/*; \
    apt-get update -o Acquire::Retries=5; \
    apt-get install -y --no-install-recommends \
        ffmpeg pkg-config \
        libavformat-dev libavcodec-dev libavdevice-dev \
        libavutil-dev libswscale-dev libswresample-dev libavfilter-dev; \
    rm -rf /var/lib/apt/lists/*

# 安装 whisperx
COPY ./whisperX-3.3.1 /workspace/whisperX-main
WORKDIR /workspace/whisperX-main
RUN printf '%s\n' \
        'torch==2.2.2' \
        'torchaudio==2.2.2' \
        'torchvision==0.17.2' \
        'numpy<2' \
        'transformers==4.45.2' \
        'ctranslate2==4.4.0' \
        > /tmp/runtime-constraints.txt && \
    pip install --upgrade pip wheel 'setuptools<81' && \
    pip install --no-build-isolation -c /tmp/runtime-constraints.txt -e . && \
    pip install -c /tmp/runtime-constraints.txt \
        'nvidia-cublas-cu12' 'nvidia-cudnn-cu12==8.*' \
        chardet fastapi uvicorn python-multipart

# 业务代码
COPY ./surfilter_audio.py /workspace/surfilter_audio.py

# 模型整体打进镜像 —— 走相对路径,必须保持 models/ 与 surfilter_audio.py 同目录
COPY ./models /workspace/models

# config.yaml 里 segmentation/embedding 用相对路径,cwd 必须在 /workspace
WORKDIR /workspace

ENV HF_HUB_CACHE=/workspace/models \
    HUGGINGFACE_HUB_CACHE=/workspace/models \
    TRANSFORMERS_CACHE=/workspace/models \
    LD_LIBRARY_PATH=/opt/conda/lib/python3.10/site-packages/nvidia/cublas/lib:/opt/conda/lib/python3.10/site-packages/nvidia/cudnn/lib:/usr/local/cuda/lib64 \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    COMPUTE_TYPE=int8

EXPOSE 8000
CMD ["python", "surfilter_audio.py"]
