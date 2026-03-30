#!/usr/bin/env python3
"""Pre-download WhisperX models into a fixed local directory."""

from __future__ import annotations

import argparse
import gc
import importlib
import logging
import os
from pathlib import Path
from typing import List


LOGGER = logging.getLogger("whisperx_download_models")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pre-download WhisperX models to a fixed directory")
    parser.add_argument("--model", default="large-v3", help="ASR model name, default: large-v3")
    parser.add_argument("--download-root", default="/models/whisperx", help="Target model directory")
    parser.add_argument("--device", default="cpu", help="Device used while preloading, default: cpu")
    parser.add_argument("--compute-type", default="int8", help="Compute type for preload, default: int8")
    parser.add_argument("--threads", type=int, default=4, help="Threads for model init")
    parser.add_argument("--languages", default="zh,en,fr,ar", help="Align model languages, comma separated")
    parser.add_argument("--task", default="transcribe", choices=["transcribe", "translate"], help="Default task")
    parser.add_argument("--hf-token", default=os.getenv("HF_TOKEN"), help="HF token for gated models / diarization")
    parser.add_argument("--include-diarization", action="store_true", help="Also download diarization model")
    parser.add_argument("--diarize-model", default="pyannote/speaker-diarization-3.1", help="Diarization model name")
    return parser.parse_args()


def parse_languages(raw: str) -> List[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def release(*objects) -> None:
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


def load_diarization_pipeline_class():
    try:
        diarize_module = importlib.import_module("whisperx.diarize")
    except Exception as exc:
        raise RuntimeError(
            "Failed to import whisperx diarization module. "
            "Ensure the installed whisperx package includes diarization support."
        ) from exc
    pipeline_cls = getattr(diarize_module, "DiarizationPipeline", None)
    if pipeline_cls is None:
        raise RuntimeError("Installed whisperx package does not expose whisperx.diarize.DiarizationPipeline")
    return pipeline_cls


def create_diarization_pipeline(
    pipeline_cls,
    *,
    model_name: str,
    hf_token: str,
    device: str,
    cache_dir: str,
):
    kwargs = {
        "model_name": model_name,
        "device": device,
        "cache_dir": cache_dir,
    }
    try:
        return pipeline_cls(token=hf_token, **kwargs)
    except TypeError:
        return pipeline_cls(use_auth_token=hf_token, **kwargs)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    args = parse_args()
    download_root = Path(args.download_root).resolve()
    download_root.mkdir(parents=True, exist_ok=True)

    try:
        import whisperx  # type: ignore
    except ImportError as exc:
        raise RuntimeError("whisperx is not installed") from exc

    LOGGER.info("preloading ASR model to %s", download_root)
    asr_model = whisperx.load_model(
        args.model,
        args.device,
        compute_type=args.compute_type,
        task=args.task,
        download_root=str(download_root),
        threads=args.threads,
        use_auth_token=args.hf_token,
    )
    release(asr_model)

    for language in parse_languages(args.languages):
        LOGGER.info("preloading align model for language=%s", language)
        align_model, align_metadata = whisperx.load_align_model(
            language_code=language,
            device=args.device,
            model_dir=str(download_root),
            model_cache_only=False,
        )
        release(align_model, align_metadata)

    if args.include_diarization:
        if not args.hf_token:
            raise RuntimeError("--include-diarization requires --hf-token or HF_TOKEN")
        LOGGER.info("preloading diarization model=%s", args.diarize_model)
        diarization_pipeline_class = load_diarization_pipeline_class()
        diarize_pipeline = create_diarization_pipeline(
            diarization_pipeline_class,
            model_name=args.diarize_model,
            hf_token=args.hf_token,
            device=args.device,
            cache_dir=str(download_root),
        )
        release(diarize_pipeline)

    LOGGER.info("all requested models are downloaded into %s", download_root)


if __name__ == "__main__":
    main()
