# Ascend WhisperX API（第一步）

`ascend_api.py` 将 `example.py` 里的推理流程封装为 HTTP API，先完成昇腾版本适配。

## 依赖

```bash
pip3 install fastapi uvicorn python-multipart
```

其余模型推理依赖、模型编译步骤与 `readme.md` 保持一致。

## 启动服务

```bash
python3 mindie_whisperx/ascend_api.py \
  --whisper-model-path ./whisper_pretrained \
  --vad-model-path ./vad_pretrained \
  --compiled-models ./compiled_models \
  --device-id 0 \
  --host 0.0.0.0 \
  --port 8000
```
`--batch-size` 默认值是 `16`，可不传。

可选预热：

```bash
python3 mindie_whisperx/ascend_api.py \
  --whisper-model-path ./whisper_pretrained \
  --vad-model-path ./vad_pretrained \
  --compiled-models ./compiled_models \
  --open-warm-up \
  --warm-up-audio-path /path/to/warmup.wav
```

## 接口

1. 健康检查

```bash
curl http://127.0.0.1:8000/health
```

2. 文件上传转写（兼容 `curl -F "file=@xxx"`）

```bash
curl -X POST http://127.0.0.1:8000/v1/transcriptions \
  -F "file=@/path/to/test.wav" \
  -F "bs=16"
```

支持视频输入（例如 `mp4`、`mov`、`mkv`），服务会先抽取音频再转写：

```bash
curl -X POST http://127.0.0.1:8000/v1/transcriptions \
  -F "file=@/path/to/test.mp4" \
  -F "language=zh"
```

也兼容常见路径：

```bash
curl -X POST http://127.0.0.1:8000/v1/audio/transcriptions \
  -F "file=@/path/to/test.wav" \
  -F "batch_size=16" \
  -F "model=whisperx"
```

指定语言（Whisper 语言代码，如 `zh`、`en`、`ja`）：

```bash
curl -X POST http://127.0.0.1:8000/v1/audio/transcriptions \
  -F "file=@/path/to/test.wav" \
  -F "language=en"
```

语言说明：
- `language=zh` 只表示中文，不区分简体/繁体，输出可能为繁体或繁简混合。
- 如果业务侧需要统一为简体中文，建议在转写结果后增加文本规范化（例如 OpenCC `t2s`）。

返回字段：
- `elapsed_ms`: 本次端到端耗时（毫秒，包含视频转音频/音频解码/模型推理）
- `language`: 本次指定的语言代码（未传则为 `null`，保持模型默认行为）
- `batch_size`: 本次实际使用的 batch（`bs` 不传时默认 16）
- `segments`: 与 `example.py` 同格式切片结果（`text/start/end`）
