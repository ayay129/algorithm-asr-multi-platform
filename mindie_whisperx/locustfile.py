"""Locust load test for Ascend WhisperX API multipart upload endpoint.

Usage example:
  AUDIO_DIR=/path/to/media \
  ASR_ENDPOINT=/v1/audio/transcriptions \
  ASR_LANGUAGE=zh \
  ASR_BATCH_SIZE=16 \
  locust -f mindie_whisperx/locustfile.py --host=http://127.0.0.1:8000
"""

from __future__ import annotations

import mimetypes
import os
import random
from pathlib import Path
from threading import Lock
from typing import List

from locust import HttpUser, between, task

MEDIA_EXTENSIONS = {
    ".wav",
    ".mp3",
    ".m4a",
    ".aac",
    ".flac",
    ".ogg",
    ".wma",
    ".webm",
    ".mp4",
    ".mov",
    ".mkv",
    ".avi",
    ".flv",
    ".wmv",
    ".m4v",
    ".mpeg",
    ".mpg",
    ".ts",
    ".3gp",
}

_FILES_LOCK = Lock()
_MEDIA_FILES: List[Path] = []


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        return default
    return float(value)


def _discover_media_files(media_dir: Path) -> List[Path]:
    if not media_dir.exists():
        raise FileNotFoundError(
            f"AUDIO_DIR does not exist: {media_dir}. Please set AUDIO_DIR to a directory with audio/video files."
        )
    if not media_dir.is_dir():
        raise NotADirectoryError(f"AUDIO_DIR is not a directory: {media_dir}")

    files = [p for p in media_dir.rglob("*") if p.is_file() and p.suffix.lower() in MEDIA_EXTENSIONS]
    if not files:
        raise RuntimeError(
            f"No supported media files found in {media_dir}. Supported suffixes: {sorted(MEDIA_EXTENSIONS)}"
        )
    return files


def _ensure_media_files(media_dir: Path) -> List[Path]:
    global _MEDIA_FILES
    if _MEDIA_FILES:
        return _MEDIA_FILES
    with _FILES_LOCK:
        if _MEDIA_FILES:
            return _MEDIA_FILES
        _MEDIA_FILES = _discover_media_files(media_dir)
    return _MEDIA_FILES


class AscendWhisperXUser(HttpUser):
    min_wait = _env_float("LOCUST_WAIT_MIN", 0.0)
    max_wait = _env_float("LOCUST_WAIT_MAX", 0.0)
    wait_time = between(min_wait, max_wait)

    media_dir = Path(os.getenv("AUDIO_DIR", "./test_media"))
    endpoint = os.getenv("ASR_ENDPOINT", "/v1/audio/transcriptions")
    request_timeout = _env_float("ASR_REQUEST_TIMEOUT", 3600.0)
    request_model = os.getenv("ASR_MODEL", "").strip()
    request_language = os.getenv("ASR_LANGUAGE", "").strip()
    request_batch_size = os.getenv("ASR_BATCH_SIZE", "").strip()

    def on_start(self) -> None:
        self.media_files = _ensure_media_files(self.media_dir)

    @task
    def transcribe(self) -> None:
        media_path = random.choice(self.media_files)
        content_type = mimetypes.guess_type(media_path.name)[0] or "application/octet-stream"

        form_data = {}
        if self.request_batch_size:
            form_data["batch_size"] = self.request_batch_size
        if self.request_language:
            form_data["language"] = self.request_language
        if self.request_model:
            form_data["model"] = self.request_model

        with media_path.open("rb") as media_fp:
            files = {"file": (media_path.name, media_fp, content_type)}
            with self.client.post(
                self.endpoint,
                data=form_data,
                files=files,
                name=f"POST {self.endpoint}",
                timeout=self.request_timeout,
                catch_response=True,
            ) as response:
                if response.status_code != 200:
                    response.failure(f"HTTP {response.status_code}: {response.text[:300]}")
                    return
                try:
                    payload = response.json()
                except Exception as exc:
                    response.failure(f"Invalid JSON response: {exc}")
                    return
                if "segments" not in payload:
                    response.failure("Missing 'segments' field in response")
                    return
                response.success()
