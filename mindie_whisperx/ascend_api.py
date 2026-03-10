# Copyright 2026
"""Ascend WhisperX API service (step 1).

This service wraps the MindIE WhisperX pipeline from ``example.py`` into an HTTP API.
"""

from __future__ import annotations

import argparse
import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import List, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

from pipeline.pipeline import MindiePipeline, load_audio


@dataclass
class AscendWhisperXConfig:
    whisper_model_path: str
    vad_model_path: str
    compiled_models: str
    batch_size: int
    device_id: int = 0
    open_warm_up: bool = False
    warm_up_audio_path: Optional[str] = None


class Segment(BaseModel):
    text: str
    start: float
    end: float


class TranscribeResponse(BaseModel):
    filename: str
    language: Optional[str]
    batch_size: int
    elapsed_ms: float
    segments: List[Segment]


class AscendWhisperXRuntime:
    """Minimal runtime adapter for Ascend/MindIE WhisperX."""

    def __init__(self, config: AscendWhisperXConfig) -> None:
        self.config = config
        self._lock = Lock()
        self._pipeline = MindiePipeline(
            config.whisper_model_path,
            config.vad_model_path,
            config.compiled_models,
            config.batch_size,
            config.device_id,
        )

        if config.open_warm_up:
            if not config.warm_up_audio_path:
                raise ValueError("open_warm_up=True requires --warm-up-audio-path")
            self.transcribe_file(config.warm_up_audio_path)

    def transcribe_file(
        self,
        audio_path: str,
        batch_size: Optional[int] = None,
        filename: Optional[str] = None,
        language: Optional[str] = None,
    ) -> TranscribeResponse:
        if not os.path.isfile(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        speech_data = load_audio(audio_path)
        used_batch_size = batch_size or self.config.batch_size

        start = time.perf_counter()
        with self._lock:
            segments = self._pipeline.transcribe(
                speech_data,
                batch_size=used_batch_size,
                language=language,
            )
        elapsed_ms = (time.perf_counter() - start) * 1000.0

        return TranscribeResponse(
            filename=filename or os.path.basename(audio_path),
            language=language,
            batch_size=used_batch_size,
            elapsed_ms=round(elapsed_ms, 3),
            segments=[Segment(**segment) for segment in segments],
        )

    def transcribe_bytes(
        self,
        audio_bytes: bytes,
        filename: str,
        batch_size: Optional[int] = None,
        language: Optional[str] = None,
    ) -> TranscribeResponse:
        suffix = Path(filename).suffix or ".wav"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            temp_file.write(audio_bytes)
            temp_audio_path = temp_file.name
        try:
            return self.transcribe_file(
                temp_audio_path,
                batch_size=batch_size,
                filename=filename,
                language=language,
            )
        finally:
            if os.path.exists(temp_audio_path):
                os.remove(temp_audio_path)


def create_app(runtime: AscendWhisperXRuntime) -> FastAPI:
    app = FastAPI(title="Ascend WhisperX API", version="0.1.0")

    @app.get("/health")
    def health() -> dict:
        return {
            "status": "ok",
            "backend": "ascend-mindie-whisperx",
            "device_id": runtime.config.device_id,
            "default_batch_size": runtime.config.batch_size,
        }

    @app.post("/", response_model=TranscribeResponse)
    @app.post("/v1/transcriptions", response_model=TranscribeResponse)
    @app.post("/v1/audio/transcriptions", response_model=TranscribeResponse)
    async def transcribe_by_upload(
        file: UploadFile = File(...),
        bs: int = Form(16),
        batch_size: Optional[int] = Form(None),
        language: Optional[str] = Form(None),
        model: Optional[str] = Form(None),
    ) -> TranscribeResponse:
        # keep compatibility with different clients: `bs` and `batch_size`
        used_bs = batch_size if batch_size is not None else bs
        if used_bs < 1:
            raise HTTPException(status_code=400, detail="bs must be greater than 0")
        try:
            payload = await file.read()
            if not payload:
                raise HTTPException(status_code=400, detail="Uploaded file is empty")
            safe_filename = file.filename or "upload.wav"
            _ = model
            return runtime.transcribe_bytes(
                payload,
                safe_filename,
                batch_size=used_bs,
                language=language,
            )
        except HTTPException:
            raise
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except Exception as exc:  # pragma: no cover - runtime depends on target env
            raise HTTPException(status_code=500, detail=f"Transcribe failed: {exc}") from exc

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ascend WhisperX API service")
    parser.add_argument("--whisper-model-path", default="whisper_pretrained", help="Whisper model path")
    parser.add_argument("--vad-model-path", default="vad_pretrained", help="VAD model path")
    parser.add_argument("--compiled-models", default="compiled_models", help="Compiled models directory")
    parser.add_argument("--batch-size", type=int, default=16, help="Inference batch size, default: 16")
    parser.add_argument("--device-id", type=int, default=0, help="Ascend device id")
    parser.add_argument("--open-warm-up", action="store_true", help="Warm up once on startup")
    parser.add_argument("--warm-up-audio-path", default=None, help="Audio path used for warm up")
    parser.add_argument("--host", default="0.0.0.0", help="Server host")
    parser.add_argument("--port", type=int, default=8000, help="Server port")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config = AscendWhisperXConfig(
        whisper_model_path=args.whisper_model_path,
        vad_model_path=args.vad_model_path,
        compiled_models=args.compiled_models,
        batch_size=args.batch_size,
        device_id=args.device_id,
        open_warm_up=args.open_warm_up,
        warm_up_audio_path=args.warm_up_audio_path,
    )
    runtime = AscendWhisperXRuntime(config)
    app = create_app(runtime)

    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
