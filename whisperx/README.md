# WhisperX NVIDIA Service

这个目录提供一个基于原生 `m-bain/whisperX` 的 FastAPI 服务，运行目标是 NVIDIA GPU。

## 安装

```bash
cd /Users/rangers/DevelopServices/app/algo-asr-multi-plat/whisperx
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

如需说话人分离，额外准备 Hugging Face Token：

```bash
export HF_TOKEN=your_token
```

## 启动

```bash
cd /Users/rangers/DevelopServices/app/algo-asr-multi-plat/whisperx
source .venv/bin/activate
python3 service.py \
  --model large-v3 \
  --device cuda \
  --device-index 0 \
  --compute-type float16 \
  --batch-size 16 \
  --download-root /models/whisperx \
  --local-files-only \
  --diarize-cache-mode keep \
  --default-diarize \
  --host 0.0.0.0 \
  --port 8000
```

说明：

- 上传文件按分块落盘，不会整包读入内存。
- `--download-root /models/whisperx --local-files-only` 会强制只从固定目录加载模型，不在服务运行时临时下载。
- 如果线上默认开启 speaker diarization，推荐使用 `--diarize-cache-mode keep`。它会提高进程基线 RSS，但避免 pyannote pipeline 每个请求重复加载/释放导致长期 RSS 高水位爬升。
- `--diarize-cache-mode offload` 只适合很少使用 diarization 的场景；如果和 `--default-diarize` 一起长期运行，服务会打印 warning。
- `ASR_MAX_UPLOAD_BYTES` 默认 `536870912`，超过后返回 413；设为 `0` 可关闭上传大小限制。
- `ASR_MAX_MEDIA_DURATION_SECONDS` 默认 `7200`，通过 `ffprobe` 在解码前拦截过长音视频；设为 `0` 可关闭时长限制。

## Docker

构建镜像：

```bash
cd /Users/rangers/DevelopServices/app/algo-asr-multi-plat/whisperx
DOCKER_BUILDKIT=1 docker build \
  --build-context whisperx_src=/path/to/local/whisperX \
  --build-context whisperx_models=/path/to/local/whisperx-models \
  -t whisperx-nvidia:offline-zh .
```

说明：

- `whisperx_src` 必须指向你宿主机上已经下载好的原生 WhisperX 源码目录，Docker 构建阶段不会再去 GitHub 拉代码
- `whisperx_models` 必须指向你宿主机上已经下载好的模型目录，Docker 构建阶段会把整个目录直接复制进镜像内 `/models/whisperx`
- 也就是说，只要服务器上的 `/root/whisperx-model` 里已经有 `large-v3` 和 `pyannote` 的 diarize 模型，构建出来的镜像就是离线可运行的
- 构建过程不需要容器内访问 GitHub，也不需要在构建阶段再传 `HF_TOKEN`
- 如果你只做中文，`whisperx_models` 里保留中文所需模型即可

运行容器：

```bash
docker run --rm -it \
  --gpus all \
  -p 8000:8000 \
  whisperx-nvidia:offline-zh
```

如果你已经在 build 阶段把模型打进镜像，运行时不需要再传 `HF_TOKEN`，也不需要再挂载模型目录。
默认已经开启离线加载和 speaker diarization。

服务器上的推荐构建命令：

```bash
cd /root/algorithm-asr-multi-platform/whisperx
DOCKER_BUILDKIT=1 docker build \
  --build-context whisperx_src=/root/whisperX \
  --build-context whisperx_models=/root/whisperx-model \
  -t whisperx-nvidia:offline-zh .
```

服务器上的推荐启动命令：

```bash
docker run -d \
  --name whisperx \
  --gpus '"device=0"' \
  -p 8000:8000 \
  whisperx-nvidia:offline-zh
```

内存监控：

```bash
ASR_MT_LOG=largev3.log ASR_MT_MATCH=large-v3 ASR_MT_INTERVAL=30 bash mt.sh
```

`mt.sh` 会记录匹配进程的 VIRT/RES、每轮总 RES，以及 `nvidia-smi` 能看到的进程显存。排查 OOM 时重点看 `TOTAL_RES` 是否持续爬升，以及同一时间 GPU 显存是否也接近上限。

## 调用

```bash
curl -X POST http://127.0.0.1:8000/v1/audio/transcriptions \
  -F "file=@/path/to/audio.wav" \
  -F "language=zh" \
  -F "batch_size=16"
```

可选字段：

- `task=transcribe|translate`
- `align=true|false`
- `diarize=true|false`
- `min_speakers`
- `max_speakers`
- `return_char_alignments=true|false`
- 服务级参数：`--diarize-cache-mode keep|offload`，默认 `keep`

## 返回

返回值格式：

```json
{
  "segments": [
    {
      "text": "SPEAKER_00 ||  你好，今天我们来讲一下这个主题",
      "start": 0.12,
      "end": 5.43
    },
    {
      "text": "SPEAKER_00 ||  第二段内容",
      "start": 5.81,
      "end": 9.22
    }
  ],
  "language": "zh"
}
```

说明：

- 顶层只返回 `segments` 和 `language`
- 每个 segment 只返回 `text`、`start`、`end`
- 如果启用了 diarization 且识别到了 speaker，返回文本会拼成 `SPEAKER_00 ||  文本`
- 如果没有 speaker 信息，`text` 就是纯转写文本
- 这版离线镜像默认已经启用 diarization，所以通常不需要再传 `diarize=true`
