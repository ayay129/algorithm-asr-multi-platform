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
  --diarize-cache-mode offload \
  --host 0.0.0.0 \
  --port 8000
```

说明：

- 上传文件按分块落盘，不会整包读入内存。
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
  -v /data/whisperx-cache:/opt/huggingface \
  whisperx-nvidia:latest
```

如果只做转写、不做 diarization，可以不传 `HF_TOKEN`。

自定义启动参数：

```bash
docker run --rm -it \
  --gpus all \
  -p 8000:8000 \
  -v /data/whisperx-cache:/opt/huggingface \
  whisperx-nvidia:latest \
  python3 /app/service.py \
  --model large-v3 \
  --device cuda \
  --device-index 0 \
  --compute-type float16 \
  --batch-size 8 \
  --diarize-cache-mode offload \
  --disable-default-align
```

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
- 服务级参数：`--diarize-cache-mode offload|keep`

## 返回

返回值包含：

- 文件名
- 模型名
- 任务类型
- 语言
- 耗时
- 是否对齐
- 是否做了说话人分离
- 逐段转写结果
- 逐词时间戳和说话人信息
