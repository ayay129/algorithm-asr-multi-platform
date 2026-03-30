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

## 预下载模型到固定目录

先把所有运行时需要的模型下载到固定目录，例如 `/models/whisperx`：

```bash
cd /Users/rangers/DevelopServices/app/algo-asr-multi-plat/whisperx
source .venv/bin/activate
python3 download_models.py \
  --model large-v3 \
  --download-root /models/whisperx \
  --languages zh,en,fr,ar \
  --include-diarization \
  --hf-token "$HF_TOKEN"
```

说明：

- `download_models.py` 会把 ASR、对齐模型、可选 diarization 模型都下载到 `--download-root`
- 服务启动时配合 `--download-root /models/whisperx --local-files-only`，运行期不会再临时联网下载
- 如果你只测中文，可以把 `--languages` 改成 `zh`

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
  --diarize-cache-mode offload \
  --host 0.0.0.0 \
  --port 8000
```

说明：

- 上传文件按分块落盘，不会整包读入内存。
- `--download-root /models/whisperx --local-files-only` 会强制只从固定目录加载模型，不在服务运行时临时下载。
- `--diarize-cache-mode offload` 是默认值，请求做完 diarization 后会主动释放对应显存。
- 如果你的业务大量依赖 diarization，想用显存换速度，再改成 `--diarize-cache-mode keep`。

## Docker

构建镜像：

```bash
cd /Users/rangers/DevelopServices/app/algo-asr-multi-plat/whisperx
docker build -t whisperx-nvidia:latest .
```

运行容器：

```bash
docker run --rm -it \
  --gpus all \
  -p 8000:8000 \
  -e HF_TOKEN=your_token \
  -v /data/whisperx-models:/models/whisperx \
  -v /data/whisperx-cache:/models/.cache \
  whisperx-nvidia:latest
```

如果只做转写、不做 diarization，可以不传 `HF_TOKEN`。

先用容器把模型下载到固定目录：

```bash
docker run --rm -it \
  --gpus all \
  -e HF_TOKEN=your_token \
  -v /data/whisperx-models:/models/whisperx \
  -v /data/whisperx-cache:/models/.cache \
  whisperx-nvidia:latest \
  python3 /app/download_models.py \
  --model large-v3 \
  --download-root /models/whisperx \
  --device cuda \
  --compute-type float16 \
  --languages zh,en,fr,ar \
  --include-diarization
```

然后再启动服务。这样模型目录固定在容器内 `/models/whisperx`，宿主机实际落盘在 `/data/whisperx-models`，服务运行时不会再临时下载。

自定义启动参数：

```bash
docker run --rm -it \
  --gpus all \
  -p 8000:8000 \
  -v /data/whisperx-models:/models/whisperx \
  -v /data/whisperx-cache:/models/.cache \
  whisperx-nvidia:latest \
  python3 /app/service.py \
  --model large-v3 \
  --device cuda \
  --device-index 0 \
  --compute-type float16 \
  --batch-size 8 \
  --download-root /models/whisperx \
  --local-files-only \
  --diarize-cache-mode offload \
  --disable-default-align
```

## 调用

```bash
curl -X POST http://127.0.0.1:8000/v1/audio/transcriptions \
  -F "file=@/path/to/audio.wav" \
  -F "language=zh" \
  -F "diarize=true" \
  -F "batch_size=16"
```

可选字段：

- `task=transcribe|translate`
- `align=true|false`
- `diarize=true|false`
- `min_speakers`
- `max_speakers`
- `return_char_alignments=true|false`
- 服务级参数：`--diarize-cache-mode offload|keep`

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
- 只传 `language=zh` 不会自动出 `speaker`，还需要 `diarize=true`，并且容器里要提前准备好 diarization 模型和 `HF_TOKEN`
