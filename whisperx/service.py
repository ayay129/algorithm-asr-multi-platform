#!/usr/bin/env python3
"""WhisperX FastAPI service for NVIDIA GPUs.

This service is intentionally small and follows the upstream WhisperX Python API:

- whisperx.load_model(...)
- model.transcribe(...)
- whisperx.load_align_model(...)
- whisperx.align(...)
- whisperx.DiarizationPipeline(...)
"""

from __future__ import annotations

import argparse
import gc
import logging
import os
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field

LOGGER = logging.getLogger("whisperx_service")
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


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name, str(default)).strip()
    try:
        return int(raw)
    except ValueError:
        LOGGER.warning("invalid %s=%s, fallback to %s", name, raw, default)
        return default


FFMPEG_THREADS = _env_int("ASR_FFMPEG_THREADS", 1)


@dataclass
class WhisperXConfig:
    model_name: str = "large-v3"
    device: str = "cuda"
    device_index: int = 0
    compute_type: str = "float16"
    batch_size: int = 16
    task: str = "transcribe"
    language: Optional[str] = None
    download_root: Optional[str] = None
    local_files_only: bool = False
    threads: int = 4
    hf_token: Optional[str] = None
    align_model: Optional[str] = None
    diarize_model: str = "pyannote/speaker-diarization-3.1"
    diarize_cache_mode: str = "offload"
    default_align: bool = True
    default_diarize: bool = False


class WordInfo(BaseModel):
    word: str
    start: Optional[float] = None
    end: Optional[float] = None
    score: Optional[float] = None
    speaker: Optional[str] = None


class SegmentInfo(BaseModel):
    id: Optional[int] = None
    start: Optional[float] = None
    end: Optional[float] = None
    text: str
    speaker: Optional[str] = None
    avg_logprob: Optional[float] = None
    words: List[WordInfo] = Field(default_factory=list)
    chars: Optional[List[Dict[str, Any]]] = None


class TranscribeResponse(BaseModel):
    filename: str
    model: str
    task: str
    language: Optional[str] = None
    batch_size: int
    elapsed_ms: float
    aligned: bool
    diarized: bool
    segments: List[SegmentInfo]


def load_whisperx_module():
    try:
        import whisperx  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "whisperx is not installed. Install upstream first, e.g. "
            "`pip install -r requirements.txt`."
        ) from exc
    return whisperx


class WhisperXRuntime:
    def __init__(self, config: WhisperXConfig) -> None:
        self.config = config
        self._lock = threading.Lock()
        self._whisperx = load_whisperx_module()
        self._asr_model = self._load_asr_model()
        self._diarize_pipeline = None

    def _load_asr_model(self):
        LOGGER.info(
            "loading whisperx model: model=%s device=%s device_index=%s compute_type=%s language=%s download_root=%s local_files_only=%s",
            self.config.model_name,
            self.config.device,
            self.config.device_index,
            self.config.compute_type,
            self.config.language,
            self.config.download_root,
            self.config.local_files_only,
        )
        return self._whisperx.load_model(
            self.config.model_name,
            self.config.device,
            device_index=self.config.device_index,
            compute_type=self.config.compute_type,
            language=self.config.language,
            task=self.config.task,
            download_root=self.config.download_root,
            local_files_only=self.config.local_files_only,
            threads=self.config.threads,
            use_auth_token=self.config.hf_token,
        )

    def _load_align_model(self, language_code: str, model_name: Optional[str] = None):
        return self._whisperx.load_align_model(
            language_code=language_code,
            device=self.config.device,
            model_name=model_name or self.config.align_model,
            model_dir=self.config.download_root,
            model_cache_only=self.config.local_files_only,
        )

    def _create_diarize_pipeline(self):
        if not self.config.hf_token:
            raise RuntimeError("diarization requires --hf-token or HF_TOKEN")
        LOGGER.info(
            "loading diarization pipeline: model=%s cache_mode=%s",
            self.config.diarize_model,
            self.config.diarize_cache_mode,
        )
        return self._whisperx.DiarizationPipeline(
            model_name=self.config.diarize_model,
            use_auth_token=self.config.hf_token,
            device=self.config.device,
            cache_dir=self.config.download_root,
        )

    def _get_diarize_pipeline(self):
        if self.config.diarize_cache_mode == "keep":
            if self._diarize_pipeline is None:
                self._diarize_pipeline = self._create_diarize_pipeline()
            return self._diarize_pipeline
        return self._create_diarize_pipeline()

    def _release_diarize_pipeline(self, pipeline: Any) -> None:
        if pipeline is None:
            return
        if self.config.diarize_cache_mode == "keep" and pipeline is self._diarize_pipeline:
            return
        self._release_torch_objects(pipeline)

    def transcribe_file(
        self,
        path: str,
        *,
        filename: Optional[str] = None,
        batch_size: Optional[int] = None,
        language: Optional[str] = None,
        task: Optional[str] = None,
        align: Optional[bool] = None,
        diarize: Optional[bool] = None,
        min_speakers: Optional[int] = None,
        max_speakers: Optional[int] = None,
        return_char_alignments: bool = False,
        content_type: Optional[str] = None,
    ) -> TranscribeResponse:
        if not os.path.isfile(path):
            raise FileNotFoundError(path)

        source_name = filename or os.path.basename(path)
        effective_align = self.config.default_align if align is None else align
        effective_diarize = self.config.default_diarize if diarize is None else diarize
        if (task or self.config.task) == "translate":
            effective_align = False
        if effective_diarize:
            effective_align = True

        started = time.perf_counter()
        audio_path = path
        cleanup_paths: List[str] = []
        audio = None
        result = None
        align_model = None
        align_metadata = None
        diarize_pipeline = None
        diarize_segments = None
        if self._is_video_input(source_name, content_type):
            audio_path = self._extract_audio_from_video(path)
            cleanup_paths.append(audio_path)

        try:
            audio = self._whisperx.load_audio(audio_path)
            with self._lock:
                result = self._asr_model.transcribe(
                    audio,
                    batch_size=batch_size or self.config.batch_size,
                    language=language,
                    task=task or self.config.task,
                )

                if effective_align and result.get("segments"):
                    align_model, align_metadata = self._load_align_model(result["language"])
                    try:
                        result = self._whisperx.align(
                            result["segments"],
                            align_model,
                            align_metadata,
                            audio,
                            self.config.device,
                            return_char_alignments=return_char_alignments,
                        )
                    finally:
                        self._release_torch_objects(align_model, align_metadata)
                        align_model = None
                        align_metadata = None

                if effective_diarize:
                    diarize_pipeline = self._get_diarize_pipeline()
                    diarize_segments = diarize_pipeline(
                        audio_path,
                        min_speakers=min_speakers,
                        max_speakers=max_speakers,
                    )
                    result = self._whisperx.assign_word_speakers(diarize_segments, result)

            elapsed_ms = (time.perf_counter() - started) * 1000.0
            return TranscribeResponse(
                filename=source_name,
                model=self.config.model_name,
                task=task or self.config.task,
                language=result.get("language"),
                batch_size=batch_size or self.config.batch_size,
                elapsed_ms=round(elapsed_ms, 3),
                aligned=effective_align,
                diarized=effective_diarize,
                segments=self._serialize_segments(result.get("segments", [])),
            )
        finally:
            self._release_diarize_pipeline(diarize_pipeline)
            audio = None
            result = None
            diarize_segments = None
            diarize_pipeline = None
            for item in cleanup_paths:
                Path(item).unlink(missing_ok=True)

    def transcribe_bytes(
        self,
        payload: bytes,
        *,
        filename: str,
        batch_size: Optional[int] = None,
        language: Optional[str] = None,
        task: Optional[str] = None,
        align: Optional[bool] = None,
        diarize: Optional[bool] = None,
        min_speakers: Optional[int] = None,
        max_speakers: Optional[int] = None,
        return_char_alignments: bool = False,
        content_type: Optional[str] = None,
    ) -> TranscribeResponse:
        suffix = Path(filename).suffix or ".wav"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            temp_file.write(payload)
            temp_path = temp_file.name
        try:
            return self.transcribe_file(
                temp_path,
                filename=filename,
                batch_size=batch_size,
                language=language,
                task=task,
                align=align,
                diarize=diarize,
                min_speakers=min_speakers,
                max_speakers=max_speakers,
                return_char_alignments=return_char_alignments,
                content_type=content_type,
            )
        finally:
            Path(temp_path).unlink(missing_ok=True)

    @staticmethod
    def _serialize_segments(segments: List[Dict[str, Any]]) -> List[SegmentInfo]:
        output: List[SegmentInfo] = []
        for segment in segments:
            words: List[WordInfo] = []
            for word in segment.get("words") or []:
                word_text = word.get("word") or word.get("text") or ""
                words.append(
                    WordInfo(
                        word=str(word_text),
                        start=_safe_float(word.get("start")),
                        end=_safe_float(word.get("end")),
                        score=_safe_float(word.get("score")),
                        speaker=word.get("speaker"),
                    )
                )
            output.append(
                SegmentInfo(
                    id=segment.get("id"),
                    start=_safe_float(segment.get("start")),
                    end=_safe_float(segment.get("end")),
                    text=str(segment.get("text", "")).strip(),
                    speaker=segment.get("speaker"),
                    avg_logprob=_safe_float(segment.get("avg_logprob")),
                    words=words,
                    chars=segment.get("chars"),
                )
            )
        return output

    @staticmethod
    def _release_torch_objects(*objects: Any) -> None:
        for obj in objects:
            try:
                del obj
            except Exception:
                pass
        gc.collect()
        try:
            import torch  # type: ignore
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    @staticmethod
    def _is_video_input(filename: str, content_type: Optional[str]) -> bool:
        if content_type and content_type.lower().startswith("video/"):
            return True
        return Path(filename).suffix.lower() in VIDEO_EXTENSIONS

    @staticmethod
    def _extract_audio_from_video(video_path: str) -> str:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
            audio_path = temp_file.name
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
        try:
            subprocess.run(cmd, capture_output=True, check=True)
            return audio_path
        except subprocess.CalledProcessError as exc:
            Path(audio_path).unlink(missing_ok=True)
            raise RuntimeError(
                f"ffmpeg extract audio failed: {exc.stderr.decode(errors='ignore')}"
            ) from exc


def _safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def create_app(runtime: WhisperXRuntime) -> FastAPI:
    app = FastAPI(title="WhisperX NVIDIA API", version="0.1.0")

    @app.get("/health")
    def health() -> Dict[str, Any]:
        return {
            "status": "ok",
            "backend": "whisperx-native",
            "model": runtime.config.model_name,
            "device": runtime.config.device,
            "device_index": runtime.config.device_index,
            "compute_type": runtime.config.compute_type,
            "default_batch_size": runtime.config.batch_size,
            "diarize_cache_mode": runtime.config.diarize_cache_mode,
        }

    @app.post("/", response_model=TranscribeResponse)
    @app.post("/v1/transcriptions", response_model=TranscribeResponse)
    @app.post("/v1/audio/transcriptions", response_model=TranscribeResponse)
    async def transcribe_upload(
        file: UploadFile = File(...),
        bs: Optional[int] = Form(None),
        batch_size: Optional[int] = Form(None),
        language: Optional[str] = Form(None),
        task: Optional[str] = Form(None),
        align: Optional[bool] = Form(None),
        diarize: Optional[bool] = Form(None),
        min_speakers: Optional[int] = Form(None),
        max_speakers: Optional[int] = Form(None),
        return_char_alignments: bool = Form(False),
    ) -> TranscribeResponse:
        used_batch_size = batch_size if batch_size is not None else bs
        if not file.filename:
            file.filename = "upload.wav"
        temp_upload_path = None
        try:
            temp_upload_path = await _persist_upload_to_tempfile(file)
            if not temp_upload_path:
                raise HTTPException(status_code=400, detail="Uploaded file is empty")
            return runtime.transcribe_file(
                temp_upload_path,
                filename=file.filename or "upload.wav",
                batch_size=used_batch_size,
                language=language,
                task=task,
                align=align,
                diarize=diarize,
                min_speakers=min_speakers,
                max_speakers=max_speakers,
                return_char_alignments=return_char_alignments,
                content_type=file.content_type,
            )
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            LOGGER.exception("transcribe failed")
            raise HTTPException(status_code=500, detail=f"Transcribe failed: {exc}") from exc
        finally:
            if temp_upload_path:
                Path(temp_upload_path).unlink(missing_ok=True)
            await file.close()

    return app


async def _persist_upload_to_tempfile(file: UploadFile, chunk_size: int = 1024 * 1024) -> Optional[str]:
    suffix = Path(file.filename or "upload.wav").suffix or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
        temp_path = temp_file.name
        total_written = 0
        try:
            while True:
                chunk = await file.read(chunk_size)
                if not chunk:
                    break
                temp_file.write(chunk)
                total_written += len(chunk)
        except Exception:
            Path(temp_path).unlink(missing_ok=True)
            raise
    if total_written == 0:
        Path(temp_path).unlink(missing_ok=True)
        return None
    return temp_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="WhisperX FastAPI service for NVIDIA GPUs")
    parser.add_argument("--model", default="large-v3", help="WhisperX model name, default: large-v3")
    parser.add_argument("--device", default="cuda", help="Device name, default: cuda")
    parser.add_argument("--device-index", type=int, default=0, help="CUDA device index, default: 0")
    parser.add_argument("--compute-type", default="float16", help="WhisperX compute type, default: float16")
    parser.add_argument("--batch-size", type=int, default=16, help="Default batch size, default: 16")
    parser.add_argument("--task", default="transcribe", choices=["transcribe", "translate"], help="Default task")
    parser.add_argument("--language", default=None, help="Default language code. Omit to auto-detect.")
    parser.add_argument("--download-root", default=None, help="Model cache directory")
    parser.add_argument("--local-files-only", action="store_true", help="Only use local cached models")
    parser.add_argument("--threads", type=int, default=4, help="CPU threads for faster-whisper")
    parser.add_argument("--hf-token", default=os.getenv("HF_TOKEN"), help="HF token for diarization and restricted downloads")
    parser.add_argument("--align-model", default=None, help="Optional alignment model override")
    parser.add_argument("--diarize-model", default="pyannote/speaker-diarization-3.1", help="Diarization model name")
    parser.add_argument(
        "--diarize-cache-mode",
        default="offload",
        choices=["offload", "keep"],
        help="Diarization model cache mode. offload frees GPU memory after each diarize request.",
    )
    parser.add_argument("--default-align", action="store_true", default=True, help="Enable alignment by default")
    parser.add_argument("--disable-default-align", action="store_true", help="Disable alignment by default")
    parser.add_argument("--default-diarize", action="store_true", help="Enable diarization by default")
    parser.add_argument("--host", default="0.0.0.0", help="Bind host")
    parser.add_argument("--port", type=int, default=8000, help="Bind port")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    args = parse_args()
    config = WhisperXConfig(
        model_name=args.model,
        device=args.device,
        device_index=args.device_index,
        compute_type=args.compute_type,
        batch_size=args.batch_size,
        task=args.task,
        language=args.language,
        download_root=args.download_root,
        local_files_only=args.local_files_only,
        threads=args.threads,
        hf_token=args.hf_token,
        align_model=args.align_model,
        diarize_model=args.diarize_model,
        diarize_cache_mode=args.diarize_cache_mode,
        default_align=False if args.disable_default_align else args.default_align,
        default_diarize=args.default_diarize,
    )
    runtime = WhisperXRuntime(config)
    app = create_app(runtime)

    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
