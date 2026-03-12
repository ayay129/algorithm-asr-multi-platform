#!/usr/bin/env python3
"""Evaluate ASR accuracy with ModelScope ``MsDataset``.

This script is designed for the document section "10.5 ASR 准确率结果表".
It supports two inference backends:

1. local: instantiate MindiePipeline directly on an Ascend host.
2. http: call an already running HTTP ASR service.

Outputs:
- asr_fleurs_details.csv
- asr_fleurs_summary.json
- asr_fleurs_metrics.csv
- asr_fleurs_doc_table.csv
- asr_fleurs_report.md
"""

from __future__ import annotations

import argparse
import csv
import html
import io
import json
import logging
import re
import subprocess
import sys
import tempfile
import time
import unicodedata
import urllib.error
import urllib.request
import uuid
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np

LOGGER = logging.getLogger("fleurs_asr_eval")
SAMPLE_RATE = 16000
DEFAULT_DATASET = "google/fleurs"
DEFAULT_TEXT_COLUMNS = "transcription,sentence,text,raw_transcription"
DEFAULT_SPLIT = "test"


@dataclass(frozen=True)
class LanguageSpec:
    code: str
    label: str
    whisper_language: str


LANGUAGE_SPECS: Dict[str, LanguageSpec] = {
    "zh": LanguageSpec(
        code="zh",
        label="中文",
        whisper_language="zh",
    ),
    "en": LanguageSpec(
        code="en",
        label="英文",
        whisper_language="en",
    ),
    "fr": LanguageSpec(
        code="fr",
        label="法语",
        whisper_language="fr",
    ),
    "ar": LanguageSpec(
        code="ar",
        label="阿语",
        whisper_language="ar",
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate Ascend WhisperX ASR accuracy with ModelScope MsDataset."
    )
    parser.add_argument(
        "--backend",
        choices=("local", "http"),
        default="local",
        help="Inference backend: local MindiePipeline or HTTP API.",
    )
    parser.add_argument(
        "--dataset-name",
        default=DEFAULT_DATASET,
        help=f"ModelScope dataset name, default: {DEFAULT_DATASET}",
    )
    parser.add_argument(
        "--dataset-namespace",
        default="",
        help="Optional ModelScope dataset namespace.",
    )
    parser.add_argument(
        "--split",
        default=DEFAULT_SPLIT,
        help=f"Dataset split, default: {DEFAULT_SPLIT}",
    )
    parser.add_argument(
        "--languages",
        default="zh,en,fr,ar",
        help="Comma separated languages. Supported: zh,en,fr,ar",
    )
    parser.add_argument(
        "--dataset-subset-map",
        "--config-overrides",
        dest="dataset_subset_map",
        default="",
        help=(
            "Language to ModelScope subset_name mapping, "
            "e.g. zh=cmn_hans_cn,en=en_us,fr=fr_fr,ar=ar_eg. "
            "If omitted, language code itself is used as subset_name."
        ),
    )
    parser.add_argument(
        "--language-labels",
        default="",
        help="Optional display labels mapping, e.g. zh=中文,en=英文,fr=法语,ar=阿语",
    )
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="ModelScope datasets cache directory.",
    )
    parser.add_argument(
        "--audio-column",
        default="audio",
        help="Audio column name, default: audio",
    )
    parser.add_argument(
        "--text-columns",
        default=DEFAULT_TEXT_COLUMNS,
        help=f"Candidate text columns, comma separated. Default: {DEFAULT_TEXT_COLUMNS}",
    )
    parser.add_argument(
        "--id-column",
        default="id",
        help="Sample id column name, default: id",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Pass trust_remote_code=True to MsDataset.load when required by the dataset.",
    )
    parser.add_argument(
        "--max-samples-per-language",
        type=int,
        default=0,
        help="Limit samples per language for debugging. 0 means full split.",
    )
    parser.add_argument(
        "--output-dir",
        default="eval_results/fleurs_asr",
        help="Directory used to store metrics and detailed outputs.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Skip a failed sample and continue.",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=50,
        help="Progress log interval per language, default: 50",
    )
    parser.add_argument(
        "--http-url",
        default="http://127.0.0.1:8000/v1/audio/transcriptions",
        help="HTTP transcription endpoint used when --backend http.",
    )
    parser.add_argument(
        "--http-timeout-seconds",
        type=float,
        default=600.0,
        help="HTTP timeout in seconds, default: 600",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Inference batch size, default: 16",
    )
    parser.add_argument(
        "--device-id",
        type=int,
        default=0,
        help="Ascend device id for local backend, default: 0",
    )
    parser.add_argument(
        "--whisper-model-path",
        default="whisper_pretrained",
        help="Whisper model path for local backend.",
    )
    parser.add_argument(
        "--vad-model-path",
        default="vad_pretrained",
        help="VAD model path for local backend.",
    )
    parser.add_argument(
        "--compiled-models",
        default="compiled_models",
        help="Compiled model directory for local backend.",
    )
    parser.add_argument(
        "--zh-opencc-config",
        default="t2s",
        help="OpenCC config for Chinese normalization, default: t2s",
    )
    parser.add_argument(
        "--zh-word-segmentation",
        choices=("auto", "jieba", "char"),
        default="auto",
        help="Chinese WER tokenization mode, default: auto",
    )
    parser.add_argument(
        "--long-audio-seconds",
        type=float,
        default=600.0,
        help="Synthetic long-audio duration per language. 0 disables the test.",
    )
    parser.add_argument(
        "--long-audio-gap-ms",
        type=int,
        default=500,
        help="Silence gap inserted between concatenated utterances, default: 500",
    )
    parser.add_argument(
        "--baseline-summary-json",
        default=None,
        help="Optional summary json from a baseline/GPU run for delta generation.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging.",
    )
    return parser.parse_args()


def configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


def parse_csv_list(raw: str) -> List[str]:
    values = [value.strip() for value in raw.split(",") if value.strip()]
    return values


def parse_mapping(raw: str) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    if not raw.strip():
        return mapping
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(
                f"Invalid mapping item '{item}'. Expected key=value, e.g. zh=cmn_hans_cn."
            )
        key, value = item.split("=", 1)
        mapping[key.strip()] = value.strip()
    return mapping


def format_ratio_as_percent(value: Optional[float]) -> str:
    if value is None:
        return ""
    return f"{value * 100:.2f}%"


def format_float(value: Optional[float], digits: int = 4) -> str:
    if value is None:
        return ""
    return f"{value:.{digits}f}"


def safe_divide(numerator: float, denominator: float) -> Optional[float]:
    if denominator <= 0:
        return None
    return numerator / denominator


def edit_distance(ref: Sequence[str], hyp: Sequence[str]) -> int:
    if ref == hyp:
        return 0
    if not ref:
        return len(hyp)
    if not hyp:
        return len(ref)

    previous = list(range(len(hyp) + 1))
    current = [0] * (len(hyp) + 1)
    for i, ref_token in enumerate(ref, start=1):
        current[0] = i
        for j, hyp_token in enumerate(hyp, start=1):
            cost = 0 if ref_token == hyp_token else 1
            current[j] = min(
                previous[j] + 1,
                current[j - 1] + 1,
                previous[j - 1] + cost,
            )
        previous, current = current, previous
    return previous[-1]


def is_punctuation_or_symbol(char: str) -> bool:
    category = unicodedata.category(char)
    return category.startswith("P") or category.startswith("S")


class TextNormalizer:
    def __init__(self, zh_opencc_config: str, zh_word_segmentation: str) -> None:
        self._zh_converter = self._load_opencc(zh_opencc_config)
        self._jieba = self._load_jieba(zh_word_segmentation)
        self.zh_word_segmentation = zh_word_segmentation

    @staticmethod
    def _load_opencc(config_name: str) -> Any:
        try:
            from opencc import OpenCC  # type: ignore
        except ImportError:
            LOGGER.info(
                "opencc is not installed, Chinese normalization will not convert traditional to simplified."
            )
            return None
        try:
            return OpenCC(config_name)
        except Exception as exc:
            LOGGER.warning("failed to initialize OpenCC(%s): %s", config_name, exc)
            return None

    @staticmethod
    def _load_jieba(mode: str) -> Any:
        if mode == "char":
            return None
        try:
            import jieba  # type: ignore
        except ImportError:
            if mode == "jieba":
                LOGGER.warning(
                    "jieba is not installed, Chinese WER will fall back to character tokenization."
                )
            return None
        return jieba

    def normalize(self, text: str, language: str) -> str:
        text = html.unescape(text or "")
        text = unicodedata.normalize("NFKC", text)
        text = text.replace("\u00A0", " ")
        text = text.replace("\u200B", "")
        text = text.replace("\ufeff", "")

        if language == "ar":
            text = self._normalize_arabic(text)

        if language == "zh":
            if self._zh_converter is not None:
                text = self._zh_converter.convert(text)
            chars: List[str] = []
            for char in text:
                if char.isspace() or is_punctuation_or_symbol(char):
                    continue
                chars.append(char.lower())
            return "".join(chars).strip()

        chars = []
        for char in text.lower():
            if char.isspace():
                chars.append(" ")
            elif is_punctuation_or_symbol(char):
                chars.append(" ")
            else:
                chars.append(char)
        normalized = "".join(chars)
        normalized = re.sub(r"\s+", " ", normalized).strip()
        return normalized

    @staticmethod
    def _normalize_arabic(text: str) -> str:
        replacements = str.maketrans(
            {
                "أ": "ا",
                "إ": "ا",
                "آ": "ا",
                "ٱ": "ا",
                "ؤ": "و",
                "ئ": "ي",
                "ى": "ي",
                "ـ": "",
            }
        )
        text = text.translate(replacements)
        text = re.sub(r"[\u064B-\u065F\u0670\u06D6-\u06ED]", "", text)
        return text

    def word_tokens(self, normalized_text: str, language: str) -> List[str]:
        if not normalized_text:
            return []
        if language == "zh":
            if self._jieba is not None and self.zh_word_segmentation in ("auto", "jieba"):
                return [token for token in self._jieba.cut(normalized_text) if token.strip()]
            return list(normalized_text)
        return normalized_text.split()

    @staticmethod
    def char_tokens(normalized_text: str) -> List[str]:
        return [char for char in normalized_text if not char.isspace()]


@dataclass
class MetricAccumulator:
    language: str
    language_label: str
    config_name: str
    sample_count: int = 0
    failure_count: int = 0
    word_edits: int = 0
    word_ref_count: int = 0
    char_edits: int = 0
    char_ref_count: int = 0
    exact_match_count: int = 0
    total_audio_seconds: float = 0.0
    total_infer_seconds: float = 0.0

    def update(
        self,
        reference_text: str,
        predicted_text: str,
        audio_seconds: float,
        infer_seconds: float,
        normalizer: TextNormalizer,
    ) -> Dict[str, Any]:
        normalized_reference = normalizer.normalize(reference_text, self.language)
        normalized_prediction = normalizer.normalize(predicted_text, self.language)

        word_ref = normalizer.word_tokens(normalized_reference, self.language)
        word_hyp = normalizer.word_tokens(normalized_prediction, self.language)
        char_ref = normalizer.char_tokens(normalized_reference)
        char_hyp = normalizer.char_tokens(normalized_prediction)

        word_edits = edit_distance(word_ref, word_hyp)
        char_edits = edit_distance(char_ref, char_hyp)
        exact_match = normalized_reference == normalized_prediction

        self.sample_count += 1
        self.word_edits += word_edits
        self.word_ref_count += len(word_ref)
        self.char_edits += char_edits
        self.char_ref_count += len(char_ref)
        self.exact_match_count += 1 if exact_match else 0
        self.total_audio_seconds += audio_seconds
        self.total_infer_seconds += infer_seconds

        return {
            "normalized_reference": normalized_reference,
            "normalized_prediction": normalized_prediction,
            "sample_wer": safe_divide(word_edits, max(len(word_ref), 1)),
            "sample_cer": safe_divide(char_edits, max(len(char_ref), 1)),
            "sentence_match": exact_match,
            "sample_rtf": safe_divide(infer_seconds, audio_seconds),
        }

    def record_failure(self) -> None:
        self.failure_count += 1

    def finalize(self) -> Dict[str, Any]:
        wer = safe_divide(self.word_edits, self.word_ref_count)
        cer = safe_divide(self.char_edits, self.char_ref_count)
        sentence_accuracy = safe_divide(self.exact_match_count, self.sample_count)
        rtf = safe_divide(self.total_infer_seconds, self.total_audio_seconds)
        return {
            "language": self.language,
            "language_label": self.language_label,
            "config_name": self.config_name,
            "sample_count": self.sample_count,
            "failure_count": self.failure_count,
            "wer": wer,
            "cer": cer,
            "sentence_accuracy": sentence_accuracy,
            "rtf": rtf,
            "total_audio_seconds": round(self.total_audio_seconds, 3),
            "total_infer_seconds": round(self.total_infer_seconds, 3),
            "wer_percent": format_ratio_as_percent(wer),
            "cer_percent": format_ratio_as_percent(cer),
            "sentence_accuracy_percent": format_ratio_as_percent(sentence_accuracy),
            "rtf_display": format_float(rtf, digits=4),
        }


class BaseTranscriber:
    def transcribe(self, audio_array: np.ndarray, language: str) -> Tuple[str, float]:
        raise NotImplementedError


class LocalMindieTranscriber(BaseTranscriber):
    def __init__(
        self,
        whisper_model_path: str,
        vad_model_path: str,
        compiled_models: str,
        batch_size: int,
        device_id: int,
    ) -> None:
        from pipeline.pipeline import MindiePipeline

        self.batch_size = batch_size
        self.pipeline = MindiePipeline(
            whisper_model_path,
            vad_model_path,
            compiled_models,
            batch_size,
            device_id,
        )

    def transcribe(self, audio_array: np.ndarray, language: str) -> Tuple[str, float]:
        start = time.perf_counter()
        segments = self.pipeline.transcribe(
            np.asarray(audio_array, dtype=np.float32),
            batch_size=self.batch_size,
            language=language,
        )
        elapsed = time.perf_counter() - start
        full_text = "".join(segment.get("text", "") for segment in segments).strip()
        return full_text, elapsed


class HttpTranscriber(BaseTranscriber):
    def __init__(self, url: str, batch_size: int, timeout_seconds: float) -> None:
        self.url = url
        self.batch_size = batch_size
        self.timeout_seconds = timeout_seconds

    def transcribe(self, audio_array: np.ndarray, language: str) -> Tuple[str, float]:
        wav_payload = wav_bytes_from_float32(audio_array, SAMPLE_RATE)
        content_type, body = build_multipart_form_data(
            fields={
                "batch_size": str(self.batch_size),
                "language": language,
            },
            files={
                "file": ("sample.wav", "audio/wav", wav_payload),
            },
        )
        request = urllib.request.Request(
            self.url,
            data=body,
            headers={"Content-Type": content_type},
            method="POST",
        )
        started = time.perf_counter()
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                payload = response.read()
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="ignore")
            raise RuntimeError(f"HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"HTTP request failed: {exc}") from exc

        elapsed = time.perf_counter() - started
        decoded = json.loads(payload.decode("utf-8"))
        segments = decoded.get("segments", [])
        full_text = "".join(str(segment.get("text", "")) for segment in segments).strip()
        service_elapsed_ms = decoded.get("elapsed_ms")
        if service_elapsed_ms is not None:
            elapsed = float(service_elapsed_ms) / 1000.0
        return full_text, elapsed


def wav_bytes_from_float32(audio_array: np.ndarray, sample_rate: int) -> bytes:
    clipped = np.clip(np.asarray(audio_array, dtype=np.float32), -1.0, 1.0)
    pcm16 = (clipped * 32767.0).astype(np.int16)
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm16.tobytes())
    return buffer.getvalue()


def build_multipart_form_data(
    fields: Dict[str, str],
    files: Dict[str, Tuple[str, str, bytes]],
) -> Tuple[str, bytes]:
    boundary = f"----CodexBoundary{uuid.uuid4().hex}"
    body = bytearray()
    for key, value in fields.items():
        body.extend(f"--{boundary}\r\n".encode("utf-8"))
        body.extend(
            f'Content-Disposition: form-data; name="{key}"\r\n\r\n'.encode("utf-8")
        )
        body.extend(str(value).encode("utf-8"))
        body.extend(b"\r\n")
    for key, (filename, content_type, payload) in files.items():
        body.extend(f"--{boundary}\r\n".encode("utf-8"))
        body.extend(
            (
                f'Content-Disposition: form-data; name="{key}"; '
                f'filename="{filename}"\r\n'
            ).encode("utf-8")
        )
        body.extend(f"Content-Type: {content_type}\r\n\r\n".encode("utf-8"))
        body.extend(payload)
        body.extend(b"\r\n")
    body.extend(f"--{boundary}--\r\n".encode("utf-8"))
    return f"multipart/form-data; boundary={boundary}", bytes(body)


def import_modelscope_modules() -> Tuple[Any, Any]:
    try:
        from modelscope.msdatasets import MsDataset  # type: ignore
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: modelscope. Install it with "
            "`pip install modelscope datasets`."
        ) from exc

    try:
        from datasets import Audio  # type: ignore
    except ImportError:
        Audio = None

    return MsDataset, Audio


def resolve_subset_name(language: str, subset_map: Dict[str, str]) -> str:
    return subset_map.get(language, language)


def resolve_language_label(language: str, label_map: Dict[str, str]) -> str:
    return label_map.get(language, LANGUAGE_SPECS[language].label)


def load_modelscope_split(
    *,
    dataset_name: str,
    dataset_namespace: str,
    subset_name: str,
    split: str,
    cache_dir: Optional[str],
    audio_column: str,
    trust_remote_code: bool,
) -> Any:
    MsDataset, Audio = import_modelscope_modules()
    load_kwargs: Dict[str, Any] = {
        "dataset_name": dataset_name,
        "subset_name": subset_name or None,
        "split": split,
        "namespace": dataset_namespace or None,
        "cache_dir": cache_dir,
    }
    if trust_remote_code:
        load_kwargs["trust_remote_code"] = True

    try:
        dataset = MsDataset.load(**load_kwargs)
    except TypeError:
        load_kwargs.pop("trust_remote_code", None)
        dataset = MsDataset.load(**load_kwargs)

    if hasattr(dataset, "to_hf_dataset"):
        dataset = dataset.to_hf_dataset()

    if isinstance(dataset, dict) and split in dataset:
        dataset = dataset[split]

    if Audio is not None and hasattr(dataset, "cast_column"):
        try:
            dataset = dataset.cast_column(audio_column, Audio(sampling_rate=SAMPLE_RATE))
        except Exception as exc:
            LOGGER.warning(
                "failed to cast audio column '%s' to 16k Audio feature: %s",
                audio_column,
                exc,
            )

    return dataset


def resample_audio(audio_array: np.ndarray, source_sample_rate: int) -> np.ndarray:
    if source_sample_rate == SAMPLE_RATE:
        return np.asarray(audio_array, dtype=np.float32)

    try:
        import librosa  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "librosa is required to resample dataset audio to 16k."
        ) from exc

    resampled = librosa.resample(
        np.asarray(audio_array, dtype=np.float32),
        orig_sr=source_sample_rate,
        target_sr=SAMPLE_RATE,
    )
    return np.asarray(resampled, dtype=np.float32)


def decode_audio_path(path: str) -> np.ndarray:
    try:
        cmd = [
            "ffmpeg",
            "-nostdin",
            "-threads",
            "1",
            "-i",
            path,
            "-f",
            "s16le",
            "-ac",
            "1",
            "-acodec",
            "pcm_s16le",
            "-ar",
            str(SAMPLE_RATE),
            "-",
        ]
        output = subprocess.run(cmd, capture_output=True, check=True).stdout
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f"Failed to decode dataset audio from {path}: {exc.stderr.decode(errors='ignore')}"
        ) from exc
    return np.frombuffer(output, np.int16).astype(np.float32) / 32768.0


def decode_audio_bytes(payload: bytes) -> np.ndarray:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
        temp_file.write(payload)
        temp_path = temp_file.name
    try:
        return decode_audio_path(temp_path)
    finally:
        Path(temp_path).unlink(missing_ok=True)


def extract_audio_array(audio_value: Any) -> np.ndarray:
    if isinstance(audio_value, np.ndarray):
        array = np.asarray(audio_value, dtype=np.float32)
        if array.ndim > 1:
            array = np.mean(array, axis=-1)
        return array

    if isinstance(audio_value, str):
        return decode_audio_path(audio_value)

    if isinstance(audio_value, dict):
        if "array" in audio_value and audio_value["array"] is not None:
            array = np.asarray(audio_value["array"], dtype=np.float32)
            if array.ndim > 1:
                array = np.mean(array, axis=-1)
            source_sample_rate = int(audio_value.get("sampling_rate", SAMPLE_RATE))
            return resample_audio(array, source_sample_rate)
        if "path" in audio_value and audio_value["path"]:
            return decode_audio_path(str(audio_value["path"]))
        if "bytes" in audio_value and audio_value["bytes"]:
            return decode_audio_bytes(audio_value["bytes"])

    raise ValueError(f"Unsupported audio sample type: {type(audio_value)}")


def iter_dataset_records(
    dataset_name: str,
    dataset_namespace: str,
    subset_name: str,
    split: str,
    cache_dir: Optional[str],
    max_samples: int,
    audio_column: str,
    text_fields: Sequence[str],
    id_column: str,
    trust_remote_code: bool,
) -> Iterator[Dict[str, Any]]:
    dataset = load_modelscope_split(
        dataset_name=dataset_name,
        dataset_namespace=dataset_namespace,
        subset_name=subset_name,
        split=split,
        cache_dir=cache_dir,
        audio_column=audio_column,
        trust_remote_code=trust_remote_code,
    )
    dataset_size = len(dataset)
    limit = min(max_samples, dataset_size) if max_samples and max_samples > 0 else dataset_size
    for index in range(limit):
        example = dataset[index]
        audio_value = example[audio_column]
        audio_array = extract_audio_array(audio_value)
        reference_text = extract_reference_text(example, text_fields)
        sample_uid = build_sample_uid(subset_name, example, index, id_column)
        yield {
            "sample_uid": sample_uid,
            "audio_array": audio_array,
            "audio_seconds": round(float(len(audio_array)) / SAMPLE_RATE, 6),
            "reference_text": reference_text,
            "source_path": (
                audio_value.get("path", "")
                if isinstance(audio_value, dict)
                else (audio_value if isinstance(audio_value, str) else "")
            ),
            "raw_example_id": example.get(id_column, ""),
        }


def extract_reference_text(example: Dict[str, Any], text_fields: Sequence[str]) -> str:
    for field in text_fields:
        value = example.get(field)
        if isinstance(value, str) and value.strip():
            return value.strip()
    raise ValueError(
        f"Unable to find reference transcript in fields: {', '.join(text_fields)}"
    )


def build_sample_uid(subset_name: str, example: Dict[str, Any], index: int, id_column: str) -> str:
    if id_column in example and str(example[id_column]).strip():
        return f"{subset_name}:{example[id_column]}"
    return f"{subset_name}:{index}"


def build_transcriber(args: argparse.Namespace) -> BaseTranscriber:
    if args.backend == "http":
        return HttpTranscriber(
            url=args.http_url,
            batch_size=args.batch_size,
            timeout_seconds=args.http_timeout_seconds,
        )
    return LocalMindieTranscriber(
        whisper_model_path=args.whisper_model_path,
        vad_model_path=args.vad_model_path,
        compiled_models=args.compiled_models,
        batch_size=args.batch_size,
        device_id=args.device_id,
    )


def merge_accumulators(accumulators: Sequence[MetricAccumulator]) -> MetricAccumulator:
    merged = MetricAccumulator(
        language="overall",
        language_label="整体",
        config_name=",".join(acc.config_name for acc in accumulators),
    )
    for accumulator in accumulators:
        merged.sample_count += accumulator.sample_count
        merged.failure_count += accumulator.failure_count
        merged.word_edits += accumulator.word_edits
        merged.word_ref_count += accumulator.word_ref_count
        merged.char_edits += accumulator.char_edits
        merged.char_ref_count += accumulator.char_ref_count
        merged.exact_match_count += accumulator.exact_match_count
        merged.total_audio_seconds += accumulator.total_audio_seconds
        merged.total_infer_seconds += accumulator.total_infer_seconds
    return merged


def evaluate_language(
    *,
    args: argparse.Namespace,
    transcriber: BaseTranscriber,
    normalizer: TextNormalizer,
    language: str,
    config_name: str,
    details_writer: csv.DictWriter,
) -> MetricAccumulator:
    spec = LANGUAGE_SPECS[language]
    language_label = resolve_language_label(language, args.language_label_map)
    accumulator = MetricAccumulator(
        language=language,
        language_label=language_label,
        config_name=config_name,
    )

    for index, record in enumerate(
        iter_dataset_records(
            dataset_name=args.dataset_name,
            dataset_namespace=args.dataset_namespace,
            subset_name=config_name,
            split=args.split,
            cache_dir=args.cache_dir,
            max_samples=args.max_samples_per_language,
            audio_column=args.audio_column,
            text_fields=args.text_fields,
            id_column=args.id_column,
            trust_remote_code=args.trust_remote_code,
        ),
        start=1,
    ):
        try:
            prediction, infer_seconds = transcriber.transcribe(
                record["audio_array"],
                spec.whisper_language,
            )
            sample_metrics = accumulator.update(
                reference_text=record["reference_text"],
                predicted_text=prediction,
                audio_seconds=record["audio_seconds"],
                infer_seconds=infer_seconds,
                normalizer=normalizer,
            )
            details_writer.writerow(
                {
                    "language": language,
                    "language_label": language_label,
                    "config_name": config_name,
                    "sample_uid": record["sample_uid"],
                    "audio_seconds": format_float(record["audio_seconds"], digits=6),
                    "infer_seconds": format_float(infer_seconds, digits=6),
                    "sample_rtf": format_float(sample_metrics["sample_rtf"], digits=6),
                    "sample_wer": format_float(sample_metrics["sample_wer"], digits=6),
                    "sample_cer": format_float(sample_metrics["sample_cer"], digits=6),
                    "sentence_match": int(sample_metrics["sentence_match"]),
                    "reference_text": record["reference_text"],
                    "predicted_text": prediction,
                    "normalized_reference": sample_metrics["normalized_reference"],
                    "normalized_prediction": sample_metrics["normalized_prediction"],
                    "error": "",
                }
            )
        except Exception as exc:
            accumulator.record_failure()
            details_writer.writerow(
                {
                    "language": language,
                    "language_label": language_label,
                    "config_name": config_name,
                    "sample_uid": record["sample_uid"],
                    "audio_seconds": format_float(record["audio_seconds"], digits=6),
                    "infer_seconds": "",
                    "sample_rtf": "",
                    "sample_wer": "",
                    "sample_cer": "",
                    "sentence_match": "",
                    "reference_text": record["reference_text"],
                    "predicted_text": "",
                    "normalized_reference": "",
                    "normalized_prediction": "",
                    "error": str(exc),
                }
            )
            if not args.continue_on_error:
                raise
            LOGGER.warning(
                "sample failed and was skipped: language=%s sample_uid=%s error=%s",
                language,
                record["sample_uid"],
                exc,
            )

        if args.log_every > 0 and index % args.log_every == 0:
            LOGGER.info(
                "progress: language=%s config=%s processed=%s failures=%s",
                language,
                config_name,
                index,
                accumulator.failure_count,
            )

    return accumulator


def build_long_audio_record(
    *,
    args: argparse.Namespace,
    language: str,
    config_name: str,
) -> Optional[Dict[str, Any]]:
    if args.long_audio_seconds <= 0:
        return None

    gap = np.zeros(int((args.long_audio_gap_ms / 1000.0) * SAMPLE_RATE), dtype=np.float32)
    audio_parts: List[np.ndarray] = []
    text_parts: List[str] = []
    source_ids: List[str] = []
    total_audio_seconds = 0.0

    for record in iter_dataset_records(
        dataset_name=args.dataset_name,
        dataset_namespace=args.dataset_namespace,
        subset_name=config_name,
        split=args.split,
        cache_dir=args.cache_dir,
        max_samples=0,
        audio_column=args.audio_column,
        text_fields=args.text_fields,
        id_column=args.id_column,
        trust_remote_code=args.trust_remote_code,
    ):
        if total_audio_seconds >= args.long_audio_seconds:
            break
        if audio_parts:
            audio_parts.append(gap)
        audio_parts.append(record["audio_array"])
        text_parts.append(record["reference_text"])
        source_ids.append(record["sample_uid"])
        total_audio_seconds += record["audio_seconds"]

    if not audio_parts:
        return None

    audio_array = np.concatenate(audio_parts).astype(np.float32)
    return {
        "language": language,
        "config_name": config_name,
        "sample_uid": f"{config_name}:long_audio",
        "audio_array": audio_array,
        "audio_seconds": float(len(audio_array)) / SAMPLE_RATE,
        "reference_text": " ".join(text_parts),
        "source_ids": source_ids,
    }


def evaluate_long_audio(
    *,
    args: argparse.Namespace,
    transcriber: BaseTranscriber,
    normalizer: TextNormalizer,
    language: str,
    config_name: str,
) -> Optional[Dict[str, Any]]:
    record = build_long_audio_record(args=args, language=language, config_name=config_name)
    if record is None:
        return None

    spec = LANGUAGE_SPECS[language]
    language_label = resolve_language_label(language, args.language_label_map)
    try:
        prediction, infer_seconds = transcriber.transcribe(
            record["audio_array"],
            spec.whisper_language,
        )
        normalized_reference = normalizer.normalize(record["reference_text"], language)
        normalized_prediction = normalizer.normalize(prediction, language)
        word_ref = normalizer.word_tokens(normalized_reference, language)
        word_hyp = normalizer.word_tokens(normalized_prediction, language)
        char_ref = normalizer.char_tokens(normalized_reference)
        char_hyp = normalizer.char_tokens(normalized_prediction)
        wer = safe_divide(edit_distance(word_ref, word_hyp), max(len(word_ref), 1))
        cer = safe_divide(edit_distance(char_ref, char_hyp), max(len(char_ref), 1))
        rtf = safe_divide(infer_seconds, record["audio_seconds"])
        return {
            "language": language,
            "language_label": language_label,
            "config_name": config_name,
            "clip_duration_seconds": round(record["audio_seconds"], 3),
            "source_sample_count": len(record["source_ids"]),
            "status": "pass",
            "wer": wer,
            "cer": cer,
            "rtf": rtf,
            "elapsed_seconds": round(infer_seconds, 3),
            "summary": (
                f"PASS ({len(record['source_ids'])} 条样本拼接为 "
                f"{record['audio_seconds']:.1f}s 长音频，无报错/超时)"
            ),
        }
    except Exception as exc:
        return {
            "language": language,
            "language_label": language_label,
            "config_name": config_name,
            "clip_duration_seconds": round(record["audio_seconds"], 3),
            "source_sample_count": len(record["source_ids"]),
            "status": "fail",
            "wer": None,
            "cer": None,
            "rtf": None,
            "elapsed_seconds": None,
            "summary": f"FAIL ({exc})",
        }


def load_baseline_summary(path: Optional[str]) -> Dict[str, Any]:
    if not path:
        return {}
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def baseline_metric(
    baseline: Dict[str, Any],
    language: str,
    metric_name: str,
) -> Optional[float]:
    if not baseline:
        return None
    by_language = baseline.get("by_language", {})
    language_entry = by_language.get(language)
    if not language_entry:
        return None
    value = language_entry.get(metric_name)
    if value is None:
        return None
    return float(value)


def build_conclusion(metric_name: str, current: Optional[float], baseline: Optional[float]) -> str:
    if current is None:
        return "待补充"
    if baseline is None:
        return "待补充基线"

    delta = current - baseline
    if metric_name in ("wer", "cer", "rtf"):
        if abs(delta) <= 0.005:
            return "与基线基本持平"
        return "优于基线" if delta < 0 else "低于基线"

    if abs(delta) <= 0.005:
        return "与基线基本持平"
    return "优于基线" if delta > 0 else "低于基线"


def delta_display(current: Optional[float], baseline: Optional[float], percent: bool) -> str:
    if current is None or baseline is None:
        return ""
    delta = current - baseline
    if percent:
        return f"{delta * 100:+.2f}%"
    return f"{delta:+.4f}"


def write_metrics_csv(path: Path, metrics: Dict[str, Dict[str, Any]]) -> None:
    fieldnames = [
        "language",
        "language_label",
        "config_name",
        "sample_count",
        "failure_count",
        "wer",
        "cer",
        "sentence_accuracy",
        "rtf",
        "wer_percent",
        "cer_percent",
        "sentence_accuracy_percent",
        "rtf_display",
        "total_audio_seconds",
        "total_infer_seconds",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for key in list(metrics.keys()):
            writer.writerow(metrics[key])


def write_doc_table_csv(
    path: Path,
    overall_metrics: Dict[str, Any],
    long_audio_summary: str,
    baseline_summary: Dict[str, Any],
) -> None:
    rows = []
    metric_map = [
        ("WER", "wer", True),
        ("CER", "cer", True),
        ("句级准确率", "sentence_accuracy", True),
        ("RTF", "rtf", False),
    ]
    for label, key, is_percent in metric_map:
        current_value = overall_metrics.get(key)
        baseline_value = baseline_summary.get("overall", {}).get(key) if baseline_summary else None
        rows.append(
            {
                "指标": label,
                "样本数": overall_metrics.get("sample_count", ""),
                "GPU / 基线版": (
                    format_ratio_as_percent(baseline_value)
                    if is_percent
                    else format_float(baseline_value, digits=4)
                ),
                "昇腾 910B4 版": (
                    format_ratio_as_percent(current_value)
                    if is_percent
                    else format_float(current_value, digits=4)
                ),
                "差值": delta_display(current_value, baseline_value, percent=is_percent),
                "结论": build_conclusion(key, current_value, baseline_value),
            }
        )

    rows.append(
        {
            "指标": "长音频稳定性",
            "样本数": "",
            "GPU / 基线版": baseline_summary.get("long_audio_overall", "") if baseline_summary else "",
            "昇腾 910B4 版": long_audio_summary,
            "差值": "",
            "结论": "通过" if long_audio_summary.startswith("PASS") else "待确认",
        }
    )

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["指标", "样本数", "GPU / 基线版", "昇腾 910B4 版", "差值", "结论"],
        )
        writer.writeheader()
        writer.writerows(rows)


def build_long_audio_overall(long_audio_results: List[Dict[str, Any]]) -> str:
    if not long_audio_results:
        return "未执行"
    passed = sum(1 for result in long_audio_results if result.get("status") == "pass")
    total = len(long_audio_results)
    durations = [
        float(result["clip_duration_seconds"])
        for result in long_audio_results
        if result.get("clip_duration_seconds") is not None
    ]
    if passed == total:
        if durations:
            return f"PASS ({passed}/{total} 条长音频稳定转写成功，单条约 {max(durations):.1f}s)"
        return f"PASS ({passed}/{total} 条长音频稳定转写成功)"
    return f"FAIL ({passed}/{total} 条长音频稳定转写成功)"


def build_report_markdown(
    *,
    args: argparse.Namespace,
    metrics_by_language: Dict[str, Dict[str, Any]],
    overall_metrics: Dict[str, Any],
    long_audio_results: List[Dict[str, Any]],
    long_audio_overall: str,
) -> str:
    lines: List[str] = []
    lines.append("# ASR FLEURS 准确率测试结果")
    lines.append("")
    lines.append(
        f"- 数据集：`{args.dataset_name}` / namespace=`{args.dataset_namespace or 'default'}` / split=`{args.split}` / 语言=`{args.languages}`"
    )
    lines.append(f"- ModelScope subset 映射：`{args.dataset_subset_map or '按语言代码同名'}`")
    lines.append(f"- 推理后端：`{args.backend}`")
    lines.append(f"- 批大小：`{args.batch_size}`")
    lines.append("")
    lines.append("## 整体汇总")
    lines.append("")
    lines.append("| 指标 | 数值 |")
    lines.append("| --- | --- |")
    lines.append(f"| 样本数 | {overall_metrics['sample_count']} |")
    lines.append(f"| WER | {overall_metrics['wer_percent']} |")
    lines.append(f"| CER | {overall_metrics['cer_percent']} |")
    lines.append(f"| 句级准确率 | {overall_metrics['sentence_accuracy_percent']} |")
    lines.append(f"| RTF | {overall_metrics['rtf_display']} |")
    lines.append(f"| 长音频稳定性 | {long_audio_overall} |")
    lines.append("")
    lines.append("## 分语言结果")
    lines.append("")
    lines.append("| 语言 | FLEURS 配置 | 样本数 | WER | CER | 句级准确率 | RTF |")
    lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: |")
    for language in metrics_by_language:
        item = metrics_by_language[language]
        lines.append(
            "| {label} | `{config}` | {count} | {wer} | {cer} | {sent} | {rtf} |".format(
                label=item["language_label"],
                config=item["config_name"],
                count=item["sample_count"],
                wer=item["wer_percent"],
                cer=item["cer_percent"],
                sent=item["sentence_accuracy_percent"],
                rtf=item["rtf_display"],
            )
        )
    lines.append("")
    lines.append("## 可直接贴文档的描述")
    lines.append("")
    overall_text = (
        "测试方法：使用 ModelScope 数据集 `{dataset}` 的 `{split}` 测试集，"
        "按语言分别加载对应 subset，逐条调用 910B4 ASR 服务进行转写，并基于标准标注统计 WER、CER、句级准确率与 RTF。"
    ).format(dataset=args.dataset_name, split=args.split)
    lines.append(overall_text)
    lines.append(
        "整体结果：共 `{count}` 条样本，WER=`{wer}`，CER=`{cer}`，句级准确率=`{sent}`，RTF=`{rtf}`。".format(
            count=overall_metrics["sample_count"],
            wer=overall_metrics["wer_percent"],
            cer=overall_metrics["cer_percent"],
            sent=overall_metrics["sentence_accuracy_percent"],
            rtf=overall_metrics["rtf_display"],
        )
    )
    lines.append(f"长音频稳定性：{long_audio_overall}。")
    lines.append("")
    for language in metrics_by_language:
        item = metrics_by_language[language]
        lines.append(f"### {item['language_label']}")
        lines.append(
            "测试方法：使用 `{dataset}` 数据集 `{config}` 配置下的 `{split}` 测试集，共 `{count}` 条音频样本。".format(
                dataset=args.dataset_name,
                config=item["config_name"],
                split=args.split,
                count=item["sample_count"],
            )
        )
        lines.append(
            "测试结果：WER=`{wer}`，CER=`{cer}`，句级准确率=`{sent}`，RTF=`{rtf}`。".format(
                wer=item["wer_percent"],
                cer=item["cer_percent"],
                sent=item["sentence_accuracy_percent"],
                rtf=item["rtf_display"],
            )
        )
        long_audio = next(
            (result for result in long_audio_results if result["language"] == language),
            None,
        )
        if long_audio is not None:
            lines.append(f"长音频稳定性：{long_audio['summary']}。")
        lines.append("")

    lines.append("## 结果附件")
    lines.append("")
    lines.append("- `asr_fleurs_details.csv`：逐样本明细")
    lines.append("- `asr_fleurs_summary.json`：程序化汇总结果")
    lines.append("- `asr_fleurs_metrics.csv`：各语言和整体指标")
    lines.append("- `asr_fleurs_doc_table.csv`：可直接填入报告表格的汇总")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    configure_logging(args.verbose)

    languages = parse_csv_list(args.languages)
    if not languages:
        raise SystemExit("No languages were provided.")
    unsupported = [language for language in languages if language not in LANGUAGE_SPECS]
    if unsupported:
        raise SystemExit(
            f"Unsupported languages: {unsupported}. Supported values: {sorted(LANGUAGE_SPECS)}"
        )

    args.text_fields = parse_csv_list(args.text_columns)
    if not args.text_fields:
        raise SystemExit("--text-columns must contain at least one field name.")

    subset_map = parse_mapping(args.dataset_subset_map)
    label_map = parse_mapping(args.language_labels)
    args.language_label_map = label_map
    config_map = {
        language: resolve_subset_name(language, subset_map)
        for language in languages
    }

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    details_path = output_dir / "asr_fleurs_details.csv"
    metrics_csv_path = output_dir / "asr_fleurs_metrics.csv"
    summary_json_path = output_dir / "asr_fleurs_summary.json"
    doc_table_path = output_dir / "asr_fleurs_doc_table.csv"
    report_path = output_dir / "asr_fleurs_report.md"

    LOGGER.info(
        "starting evaluation: backend=%s dataset=%s namespace=%s split=%s output_dir=%s subset_map=%s",
        args.backend,
        args.dataset_name,
        args.dataset_namespace or "default",
        args.split,
        output_dir,
        config_map,
    )

    normalizer = TextNormalizer(
        zh_opencc_config=args.zh_opencc_config,
        zh_word_segmentation=args.zh_word_segmentation,
    )
    transcriber = build_transcriber(args)

    metrics_by_language: Dict[str, Dict[str, Any]] = {}
    accumulators: List[MetricAccumulator] = []

    with details_path.open("w", encoding="utf-8", newline="") as details_handle:
        details_writer = csv.DictWriter(
            details_handle,
            fieldnames=[
                "language",
                "language_label",
                "config_name",
                "sample_uid",
                "audio_seconds",
                "infer_seconds",
                "sample_rtf",
                "sample_wer",
                "sample_cer",
                "sentence_match",
                "reference_text",
                "predicted_text",
                "normalized_reference",
                "normalized_prediction",
                "error",
            ],
        )
        details_writer.writeheader()

        for language in languages:
            config_name = config_map[language]
            LOGGER.info(
                "evaluating language=%s label=%s config=%s",
                language,
                resolve_language_label(language, args.language_label_map),
                config_name,
            )
            accumulator = evaluate_language(
                args=args,
                transcriber=transcriber,
                normalizer=normalizer,
                language=language,
                config_name=config_name,
                details_writer=details_writer,
            )
            accumulators.append(accumulator)
            metrics_by_language[language] = accumulator.finalize()

    overall_metrics = merge_accumulators(accumulators).finalize()
    metrics_for_csv = dict(metrics_by_language)
    metrics_for_csv["overall"] = overall_metrics
    write_metrics_csv(metrics_csv_path, metrics_for_csv)

    long_audio_results: List[Dict[str, Any]] = []
    if args.long_audio_seconds > 0:
        for language in languages:
            config_name = config_map[language]
            LOGGER.info(
                "evaluating long-audio stability: language=%s config=%s target_seconds=%.1f",
                language,
                config_name,
                args.long_audio_seconds,
            )
            result = evaluate_long_audio(
                args=args,
                transcriber=transcriber,
                normalizer=normalizer,
                language=language,
                config_name=config_name,
            )
            if result is not None:
                long_audio_results.append(result)

    baseline_summary = load_baseline_summary(args.baseline_summary_json)
    long_audio_overall = build_long_audio_overall(long_audio_results)
    write_doc_table_csv(
        doc_table_path,
        overall_metrics=overall_metrics,
        long_audio_summary=long_audio_overall,
        baseline_summary=baseline_summary,
    )

    summary_payload = {
        "metadata": {
            "dataset_name": args.dataset_name,
            "dataset_namespace": args.dataset_namespace or None,
            "split": args.split,
            "backend": args.backend,
            "languages": languages,
            "config_map": config_map,
            "text_fields": args.text_fields,
            "audio_column": args.audio_column,
            "id_column": args.id_column,
            "trust_remote_code": args.trust_remote_code,
            "batch_size": args.batch_size,
            "device_id": args.device_id if args.backend == "local" else None,
            "http_url": args.http_url if args.backend == "http" else None,
            "long_audio_seconds": args.long_audio_seconds,
            "long_audio_gap_ms": args.long_audio_gap_ms,
            "zh_word_segmentation": args.zh_word_segmentation,
            "zh_opencc_config": args.zh_opencc_config,
        },
        "overall": overall_metrics,
        "by_language": metrics_by_language,
        "long_audio_by_language": {result["language"]: result for result in long_audio_results},
        "long_audio_overall": long_audio_overall,
        "artifacts": {
            "details_csv": str(details_path),
            "metrics_csv": str(metrics_csv_path),
            "doc_table_csv": str(doc_table_path),
            "report_md": str(report_path),
        },
    }
    with summary_json_path.open("w", encoding="utf-8") as handle:
        json.dump(summary_payload, handle, ensure_ascii=False, indent=2)

    report_markdown = build_report_markdown(
        args=args,
        metrics_by_language=metrics_by_language,
        overall_metrics=overall_metrics,
        long_audio_results=long_audio_results,
        long_audio_overall=long_audio_overall,
    )
    report_path.write_text(report_markdown, encoding="utf-8")

    LOGGER.info("evaluation finished")
    LOGGER.info("details csv: %s", details_path)
    LOGGER.info("metrics csv: %s", metrics_csv_path)
    LOGGER.info("summary json: %s", summary_json_path)
    LOGGER.info("doc table csv: %s", doc_table_path)
    LOGGER.info("report markdown: %s", report_path)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        LOGGER.error("evaluation interrupted by user")
        sys.exit(130)

'''
运行命令
python eval_fleurs_asr.py \
  --backend http \
  --dataset-name google/fleurs \
  --dataset-subset-map zh=cmn_hans_cn,en=en_us,fr=fr_fr,ar=ar_eg \
  --language-labels zh=中文,en=英文,fr=法语,ar=阿语 \
  --split test \
  --trust-remote-code \
  --http-url http://127.0.0.1:8111 \
  --batch-size 1 \
  --output-dir ./eval_results/fleurs_asr_910b4
'''