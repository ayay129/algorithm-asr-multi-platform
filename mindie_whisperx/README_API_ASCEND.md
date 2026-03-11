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
  --batch-size 1 \
  --device-id 0 \
  --host 0.0.0.0 \
  --port 8000
```
`--batch-size` 是运行时默认 batch（接口不传 `bs/batch_size` 时会使用它）。

长音频建议开启分块（默认已开启）：
- `--chunk-duration-seconds`: 每块长度，默认 `600` 秒。
- `--chunk-overlap-seconds`: 相邻块重叠，默认 `1.0` 秒。

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
如果请求不传 `bs/batch_size`，服务会使用启动时的 `--batch-size`。

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
- `batch_size`: 本次实际使用的 batch（`bs/batch_size` 不传时使用服务启动参数 `--batch-size`）
- `segments`: 与 `example.py` 同格式切片结果（`text/start/end`）

## Locust压测

安装依赖：

```bash
pip3 install locust matplotlib
```

一键压测并自动生成图表（推荐）：

```bash
AUDIO_DIR=/path/to/media \
ASR_ENDPOINT=/v1/audio/transcriptions \
ASR_LANGUAGE=zh \
ASR_BATCH_SIZE=16 \
HOST=http://127.0.0.1:8000 \
LOCUST_USERS=30 \
LOCUST_SPAWN_RATE=5 \
LOCUST_RUN_TIME=10m \
bash mindie_whisperx/run_locust_benchmark.sh
```

生成内容（默认在 `benchmarks/<时间戳>/`）：
- `throughput_rps.png`: 吞吐与失败速率曲线。
- `latency_ms.png`: 平均延迟与 P95 曲线。
- `users_over_time.png`: 并发用户曲线。
- `npu_utilization.png`: NPU Core 利用率曲线（启用采样时）。
- `summary.png`: 总览柱状图。
- `benchmark_summary.md`: 指标摘要。

启动 Web UI（目录下有大量音频/视频时会随机抽样请求）：

```bash
AUDIO_DIR=/path/to/media \
ASR_ENDPOINT=/v1/audio/transcriptions \
ASR_LANGUAGE=zh \
ASR_BATCH_SIZE=16 \
locust -f mindie_whisperx/locustfile.py \
  --host=http://127.0.0.1:8000
```

无 UI 手动压测 + 手动出图：

```bash
AUDIO_DIR=/path/to/media \
ASR_ENDPOINT=/v1/audio/transcriptions \
ASR_LANGUAGE=zh \
ASR_BATCH_SIZE=16 \
locust -f mindie_whisperx/locustfile.py \
  --host=http://127.0.0.1:8000 \
  --headless -u 30 -r 5 -t 60s \
  --csv benchmarks/manual/locust \
  --csv-full-history

python3 mindie_whisperx/plot_locust_results.py \
  --csv-prefix benchmarks/manual/locust \
  --out-dir benchmarks/manual
```

可选环境变量：
- `AUDIO_DIR`: 压测媒体目录（必填，递归扫描）。
- `ASR_ENDPOINT`: 压测接口路径，默认 `/v1/audio/transcriptions`。
- `ASR_LANGUAGE`: 可选语言（如 `zh` / `en`）。
- `ASR_BATCH_SIZE`: 可选 batch（表单字段 `batch_size`）。
- `ASR_MODEL`: 可选 model 字段。
- `ASR_REQUEST_TIMEOUT`: 单请求超时秒数，默认 `3600`。
- `LOCUST_WAIT_MIN`: 请求间最小等待秒数，默认 `0`。
- `LOCUST_WAIT_MAX`: 请求间最大等待秒数，默认 `0`。
- `LOCUST_USERS`: 压测并发数（`run_locust_benchmark.sh` 默认 `20`）。
- `LOCUST_SPAWN_RATE`: 用户拉起速率（默认 `2`）。
- `LOCUST_RUN_TIME`: 压测时长（默认 `5m`）。
- `OUT_ROOT`: 结果目录根路径（默认 `benchmarks`）。
- `RUN_ID`: 压测输出目录名（默认时间戳）。
- `NPU_MONITOR_ENABLE`: 是否启用 NPU 采样，默认 `1`。
- `NPU_DEVICE_ID`: 采样的 NPU 设备编号，默认 `0`。
- `NPU_POLL_INTERVAL_SECONDS`: 采样间隔秒数，默认 `1`。
- `NPU_SMI_BIN`: `npu-smi` 可执行文件名，默认 `npu-smi`。
