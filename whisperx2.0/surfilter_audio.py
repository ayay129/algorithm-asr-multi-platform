#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Time: 2024/07/08 15:16
Author: Rangers
Email: rangers5733@gmail.com
File: surfilter_audio.py
"""
import os
import gc
from typing import Optional

import torch
import whisperx
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse

try:
    from transformers.utils import move_cache
except ImportError:
    move_cache = None

MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", MODELS_DIR)
os.environ.setdefault("HF_HUB_CACHE", MODELS_DIR)
# 离线:pyannote / transformers 都按本地 cache 解析,不联网
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
VAD_MODEL_PATH = os.getenv(
    "VAD_MODEL_PATH", os.path.join(MODELS_DIR, "whisperx-vad-segmentation.bin")
)
# 指向本地 pyannote config.yaml;config.yaml 内子模型用相对路径,需在 whisperx2.0/ 启动
DIARIZE_MODEL_PATH = os.getenv(
    "DIARIZE_MODEL_PATH", os.path.join(MODELS_DIR, "pyannote-diarization", "config.yaml")
)

if move_cache is not None:
    move_cache()

app = FastAPI()

device = "cuda" if torch.cuda.is_available() else "cpu"
batch_size = int(os.getenv("BATCH_SIZE", 4))
compute_type = os.getenv("COMPUTE_TYPE", "int8")

model = whisperx.load_model(
    "large-v3", device, compute_type=compute_type,
    vad_options={"model_fp": VAD_MODEL_PATH},
)

# 全局只加载一次,绝不在 handler 里 new —— pyannote pipeline 反复创建是泄漏大头
_diarize_pipeline = None


def get_diarize_pipeline():
    global _diarize_pipeline
    if _diarize_pipeline is None:
        _diarize_pipeline = whisperx.diarize.DiarizationPipeline(
            model_name=DIARIZE_MODEL_PATH, device=device
        )
    return _diarize_pipeline


def _release():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


@app.post("/transcribe")
async def transcribe_audio(
    file: UploadFile = File(...),
    diarize: bool = Form(False),
    min_speakers: Optional[int] = Form(None),
    max_speakers: Optional[int] = Form(None),
):
    if not file.filename.endswith((".mp3", ".wav", ".mp4", ".m4a")):
        raise HTTPException(status_code=400, detail="Invalid file format. Please upload an audio file.")

    audio_file = f"temp_{file.filename}"
    audio = None
    result = None
    diarize_segments = None
    try:
        with open(audio_file, "wb") as buffer:
            buffer.write(file.file.read())

        audio = whisperx.load_audio(audio_file)
        result = model.transcribe(audio, batch_size=batch_size, chunk_size=2)

        if diarize:
            diarize_segments = get_diarize_pipeline()(
                audio_file, min_speakers=min_speakers, max_speakers=max_speakers,
            )
            result = whisperx.assign_word_speakers(diarize_segments, result)
            for obj in result.get("segments", []):
                speaker = obj.get("speaker", "")
                text = obj.get("text", "").strip()
                obj["text"] = f"{speaker} || {text}" if speaker else text
                obj.pop("speaker", None)

        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        audio = None
        result = None
        diarize_segments = None
        _release()
        if os.path.exists(audio_file):
            os.remove(audio_file)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
