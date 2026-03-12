# Copyright 2026
"""Ascend WhisperX API service (step 1).

This service wraps the MindIE WhisperX pipeline from ``example.py`` into an HTTP API.
"""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import List, Optional

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

from pipeline.pipeline import MindiePipeline, load_audio

logger = logging.getLogger("ascend_whisperx_api")
SAMPLE_RATE = 16000
VIDEO_EXTENSIONS = {
    ".mp4",
    ".mov",
    ".mkv",
    ".avi",
    ".flv",
    ".wmv",
    ".webm",
    ".m4v",
    ".mpeg",
    ".mpg",
    ".ts",
    ".3gp",
}


def _env_int(name: str, default: int, min_value: int = 1) -> int:
    raw = os.getenv(name, str(default)).strip()
    try:
        value = int(raw)
    except ValueError:
        logger.warning("invalid %s=%s, fallback to %s", name, raw, default)
        value = default
    if value < min_value:
        logger.warning("%s=%s is lower than %s, force to %s", name, value, min_value, min_value)
        value = min_value
    return value


FFMPEG_THREADS = _env_int("ASR_FFMPEG_THREADS", 1, 1)


@dataclass
class AscendWhisperXConfig:
    whisper_model_path: str
    vad_model_path: str
    compiled_models: str
    batch_size: int
    device_id: int = 0
    chunk_duration_seconds: int = 600
    chunk_overlap_seconds: float = 1.0
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
        logger.info(
            "runtime initialized: whisper_model_path=%s vad_model_path=%s compiled_models=%s batch_size=%s device_id=%s chunk_duration_seconds=%s chunk_overlap_seconds=%s",
            config.whisper_model_path,
            config.vad_model_path,
            config.compiled_models,
            config.batch_size,
            config.device_id,
            config.chunk_duration_seconds,
            config.chunk_overlap_seconds,
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
        content_type: Optional[str] = None,
    ) -> TranscribeResponse:
        if not os.path.isfile(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        total_start = time.perf_counter()
        transcribe_path = audio_path
        cleanup_paths: List[str] = []
        source_name = filename or os.path.basename(audio_path)
        if self._is_video_input(source_name, content_type):
            transcribe_path = self._extract_audio_from_video(audio_path)
            cleanup_paths.append(transcribe_path)

        speech_data = load_audio(transcribe_path)
        used_batch_size = batch_size or self.config.batch_size

        try:
            segments = self._transcribe_waveform_with_chunking(
                speech_data,
                batch_size=used_batch_size,
                language=language,
            )
            elapsed_ms = (time.perf_counter() - total_start) * 1000.0

            return TranscribeResponse(
                filename=source_name,
                language=language,
                batch_size=used_batch_size,
                elapsed_ms=round(elapsed_ms, 3),
                segments=[Segment(**segment) for segment in segments],
            )
        finally:
            for path in cleanup_paths:
                if os.path.exists(path):
                    os.remove(path)

    def transcribe_bytes(
        self,
        audio_bytes: bytes,
        filename: str,
        batch_size: Optional[int] = None,
        language: Optional[str] = None,
        content_type: Optional[str] = None,
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
                content_type=content_type,
            )
        finally:
            if os.path.exists(temp_audio_path):
                os.remove(temp_audio_path)

    def _transcribe_waveform(
        self,
        audio_data: np.ndarray,
        batch_size: int,
        language: Optional[str],
    ) -> List[dict]:
        with self._lock:
            return self._pipeline.transcribe(
                audio_data,
                batch_size=batch_size,
                language=language,
            )

    def _transcribe_waveform_with_chunking(
        self,
        audio_data: np.ndarray,
        batch_size: int,
        language: Optional[str],
    ) -> List[dict]:
        chunk_seconds = self.config.chunk_duration_seconds
        overlap_seconds = self.config.chunk_overlap_seconds
        if chunk_seconds <= 0:
            return self._transcribe_waveform(audio_data, batch_size, language)

        chunk_samples = int(chunk_seconds * SAMPLE_RATE)
        overlap_samples = int(max(0.0, overlap_seconds) * SAMPLE_RATE)
        if chunk_samples <= 0:
            return self._transcribe_waveform(audio_data, batch_size, language)

        step_samples = chunk_samples - overlap_samples
        if step_samples <= 0:
            step_samples = chunk_samples

        total_samples = int(audio_data.shape[0])
        if total_samples <= chunk_samples:
            return self._transcribe_waveform(audio_data, batch_size, language)

        logger.info(
            "long audio detected: duration_sec=%.3f chunk_sec=%s overlap_sec=%.3f",
            total_samples / SAMPLE_RATE,
            chunk_seconds,
            overlap_seconds,
        )
        merged_segments: List[dict] = []
        chunk_index = 0
        for start in range(0, total_samples, step_samples):
            end = min(total_samples, start + chunk_samples)
            if start >= end:
                break

            chunk_audio = audio_data[start:end]
            offset_sec = start / SAMPLE_RATE
            chunk_segments = self._transcribe_waveform(chunk_audio, batch_size, language)
            logger.info(
                "chunk transcribed: index=%s start_sec=%.3f end_sec=%.3f segment_count=%s",
                chunk_index,
                offset_sec,
                end / SAMPLE_RATE,
                len(chunk_segments),
            )
            for segment in chunk_segments:
                shifted = dict(segment)
                shifted["start"] = round(float(shifted["start"]) + offset_sec, 3)
                shifted["end"] = round(float(shifted["end"]) + offset_sec, 3)
                merged_segments.append(shifted)
            chunk_index += 1
            if end >= total_samples:
                break

        return self._deduplicate_segments(merged_segments)

    @staticmethod
    def _deduplicate_segments(segments: List[dict]) -> List[dict]:
        if not segments:
            return []
        ordered = sorted(
            segments,
            key=lambda seg: (float(seg.get("start", 0.0)), float(seg.get("end", 0.0))),
        )
        merged: List[dict] = []
        for seg in ordered:
            text = str(seg.get("text", "")).strip()
            start = float(seg.get("start", 0.0))
            end = float(seg.get("end", start))
            current = {"text": text, "start": round(start, 3), "end": round(end, 3)}
            if not merged:
                merged.append(current)
                continue

            prev = merged[-1]
            same_text = prev["text"] == current["text"]
            overlap = min(float(prev["end"]), end) - max(float(prev["start"]), start)
            near_same_range = abs(float(prev["start"]) - start) <= 0.05 and abs(float(prev["end"]) - end) <= 0.05
            if same_text and (near_same_range or overlap >= 0):
                prev["end"] = round(max(float(prev["end"]), end), 3)
                continue
            merged.append(current)
        return merged

    @staticmethod
    def _is_video_input(filename: str, content_type: Optional[str] = None) -> bool:
        if content_type and content_type.lower().startswith("video/"):
            return True
        suffix = Path(filename).suffix.lower()
        return suffix in VIDEO_EXTENSIONS

    @staticmethod
    def _extract_audio_from_video(video_path: str) -> str:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as audio_file:
            audio_path = audio_file.name
        cmd = [
            "ffmpeg",
            "-nostdin",
            "-threads",
            str(FFMPEG_THREADS),
            "-y",
            "-i",
            video_path,
            "-vn",
            "-ac",
            "1",
            "-acodec",
            "pcm_s16le",
            "-ar",
            "16000",
            audio_path,
        ]
        started = time.perf_counter()
        try:
            subprocess.run(cmd, capture_output=True, check=True)
            logger.info(
                "video converted to audio: input=%s output=%s elapsed_ms=%.3f",
                video_path,
                audio_path,
                (time.perf_counter() - started) * 1000.0,
            )
            return audio_path
        except subprocess.CalledProcessError as exc:
            if os.path.exists(audio_path):
                os.remove(audio_path)
            stderr = exc.stderr.decode(errors="ignore").strip()
            raise RuntimeError(f"Failed to extract audio from video: {stderr}") from exc


def create_app(runtime: AscendWhisperXRuntime) -> FastAPI:
    app = FastAPI(title="Ascend WhisperX API", version="0.1.0")

    @app.get("/health")
    def health() -> dict:
        return {
            "status": "ok",
            "backend": "ascend-mindie-whisperx",
            "device_id": runtime.config.device_id,
            "default_batch_size": runtime.config.batch_size,
            "chunk_duration_seconds": runtime.config.chunk_duration_seconds,
            "chunk_overlap_seconds": runtime.config.chunk_overlap_seconds,
        }

    @app.post("/", response_model=TranscribeResponse)
    @app.post("/v1/transcriptions", response_model=TranscribeResponse)
    @app.post("/v1/audio/transcriptions", response_model=TranscribeResponse)
    async def transcribe_by_upload(
        file: UploadFile = File(...),
        bs: Optional[int] = Form(None),
        batch_size: Optional[int] = Form(None),
        language: Optional[str] = Form(None),
        model: Optional[str] = Form(None),
    ) -> TranscribeResponse:
        # keep compatibility with different clients: `bs` and `batch_size`
        used_bs = (
            batch_size
            if batch_size is not None
            else (bs if bs is not None else runtime.config.batch_size)
        )
        if used_bs < 1:
            raise HTTPException(status_code=400, detail="bs must be greater than 0")
        request_start = time.perf_counter()
        try:
            logger.info(
                "transcribe request: filename=%s content_type=%s language=%s batch_size=%s model=%s",
                file.filename,
                file.content_type,
                language,
                used_bs,
                model,
            )
            payload = await file.read()
            if not payload:
                raise HTTPException(status_code=400, detail="Uploaded file is empty")
            safe_filename = file.filename or "upload.wav"
            response = runtime.transcribe_bytes(
                payload,
                safe_filename,
                batch_size=used_bs,
                language=language,
                content_type=file.content_type,
            )
            logger.info(
                "transcribe done: filename=%s language=%s batch_size=%s transcribe_elapsed_ms=%.3f request_elapsed_ms=%.3f segments=%s",
                response.filename,
                response.language,
                response.batch_size,
                response.elapsed_ms,
                (time.perf_counter() - request_start) * 1000.0,
                len(response.segments),
            )
            return response
        except HTTPException:
            logger.warning(
                "transcribe http error: filename=%s request_elapsed_ms=%.3f",
                file.filename,
                (time.perf_counter() - request_start) * 1000.0,
            )
            raise
        except ValueError as exc:
            logger.warning("transcribe value error: %s", exc)
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except FileNotFoundError as exc:
            logger.warning("transcribe file error: %s", exc)
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except Exception as exc:  # pragma: no cover - runtime depends on target env
            logger.exception("transcribe failed")
            raise HTTPException(status_code=500, detail=f"Transcribe failed: {exc}") from exc

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ascend WhisperX API service")
    parser.add_argument("--whisper-model-path", default="whisper_pretrained", help="Whisper model path")
    parser.add_argument("--vad-model-path", default="vad_pretrained", help="VAD model path")
    parser.add_argument("--compiled-models", default="compiled_models", help="Compiled models directory")
    parser.add_argument("--batch-size", type=int, default=16, help="Inference batch size, default: 16")
    parser.add_argument("--device-id", type=int, default=0, help="Ascend device id")
    parser.add_argument(
        "--chunk-duration-seconds",
        type=int,
        default=600,
        help="Long audio chunk size in seconds; 0 means disable chunking, default: 600",
    )
    parser.add_argument(
        "--chunk-overlap-seconds",
        type=float,
        default=1.0,
        help="Chunk overlap in seconds for long audio, default: 1.0",
    )
    parser.add_argument("--open-warm-up", action="store_true", help="Warm up once on startup")
    parser.add_argument("--warm-up-audio-path", default=None, help="Audio path used for warm up")
    parser.add_argument("--host", default="0.0.0.0", help="Server host")
    parser.add_argument("--port", type=int, default=8000, help="Server port")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    args = parse_args()
    logger.info(
        "starting api service: host=%s port=%s batch_size=%s device_id=%s chunk_duration_seconds=%s chunk_overlap_seconds=%s ASR_FFMPEG_THREADS=%s",
        args.host,
        args.port,
        args.batch_size,
        args.device_id,
        args.chunk_duration_seconds,
        args.chunk_overlap_seconds,
        FFMPEG_THREADS,
    )

    config = AscendWhisperXConfig(
        whisper_model_path=args.whisper_model_path,
        vad_model_path=args.vad_model_path,
        compiled_models=args.compiled_models,
        batch_size=args.batch_size,
        device_id=args.device_id,
        chunk_duration_seconds=args.chunk_duration_seconds,
        chunk_overlap_seconds=args.chunk_overlap_seconds,
        open_warm_up=args.open_warm_up,
        warm_up_audio_path=args.warm_up_audio_path,
    )
    runtime = AscendWhisperXRuntime(config)
    app = create_app(runtime)

    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
