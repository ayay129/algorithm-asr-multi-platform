from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class DatasetSpec:
    """Describe where a dataset comes from and how to read its key columns."""

    source_type: str
    dataset_name: Optional[str] = None
    dataset_config: Optional[str] = None
    split: str = "test[:100]"
    data_files: Optional[Any] = None
    cache_dir: Optional[str] = None
    audio_column: str = "audio"
    text_column: str = "transcription"
    id_column: Optional[str] = None
    path_column: Optional[str] = None
    path_from_audio: bool = True
    sampling_rate: int = 16000


@dataclass
class EvalTaskSpec:
    """One evaluation task normally maps to one language or one dataset split."""

    name: str
    whisper_language: str
    dataset: DatasetSpec
    continue_on_error: bool = False


@dataclass
class ModelSpec:
    """Define one Whisper model checkpoint to be evaluated."""

    name: str
    model_path: str
    generation_kwargs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BatchEvalConfig:
    """Top-level config used by the batch scheduler."""

    devices: List[int]
    output_root: str
    models: List[ModelSpec]
    tasks: List[EvalTaskSpec]
    hf_endpoint: Optional[str] = None
    scheduler_poll_seconds: float = 1.0
    generate_plots: bool = True


def _dataset_from_dict(payload: Dict[str, Any]) -> DatasetSpec:
    spec = DatasetSpec(
        source_type=str(payload["source_type"]).strip(),
        dataset_name=payload.get("dataset_name"),
        dataset_config=payload.get("dataset_config"),
        split=str(payload.get("split", "test[:100]")),
        data_files=payload.get("data_files"),
        cache_dir=payload.get("cache_dir"),
        audio_column=str(payload.get("audio_column", "audio")),
        text_column=str(payload.get("text_column", "transcription")),
        id_column=payload.get("id_column"),
        path_column=payload.get("path_column"),
        path_from_audio=bool(payload.get("path_from_audio", True)),
        sampling_rate=int(payload.get("sampling_rate", 16000)),
    )
    if spec.source_type == "huggingface" and not spec.dataset_name:
        raise ValueError("huggingface dataset requires dataset_name")
    if spec.source_type == "json" and spec.data_files is None:
        raise ValueError("json dataset requires data_files")
    if spec.source_type not in {"huggingface", "json"}:
        raise ValueError(f"unsupported source_type: {spec.source_type}")
    return spec


def _task_from_dict(payload: Dict[str, Any]) -> EvalTaskSpec:
    return EvalTaskSpec(
        name=str(payload["name"]),
        whisper_language=str(payload["whisper_language"]),
        dataset=_dataset_from_dict(payload["dataset"]),
        continue_on_error=bool(payload.get("continue_on_error", False)),
    )


def _model_from_dict(payload: Dict[str, Any]) -> ModelSpec:
    return ModelSpec(
        name=str(payload["name"]),
        model_path=str(payload["model_path"]),
        generation_kwargs=dict(payload.get("generation_kwargs", {})),
    )


def batch_config_from_dict(payload: Dict[str, Any]) -> BatchEvalConfig:
    models = [_model_from_dict(item) for item in payload.get("models", [])]
    tasks = [_task_from_dict(item) for item in payload.get("tasks", [])]
    config = BatchEvalConfig(
        devices=[int(device) for device in payload.get("devices", [])],
        output_root=str(payload.get("output_root", "evaluate/results")),
        models=models,
        tasks=tasks,
        hf_endpoint=payload.get("hf_endpoint"),
        scheduler_poll_seconds=float(payload.get("scheduler_poll_seconds", 1.0)),
        generate_plots=bool(payload.get("generate_plots", True)),
    )
    validate_batch_config(config)
    return config


def load_batch_config(path: str | Path) -> BatchEvalConfig:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return batch_config_from_dict(payload)


def dump_batch_config(config: BatchEvalConfig, path: str | Path) -> None:
    target_path = Path(path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    with target_path.open("w", encoding="utf-8") as handle:
        json.dump(asdict(config), handle, ensure_ascii=False, indent=2)


def validate_batch_config(config: BatchEvalConfig) -> None:
    if not config.devices:
        raise ValueError("devices must not be empty")
    if not config.models:
        raise ValueError("models must not be empty")
    if not config.tasks:
        raise ValueError("tasks must not be empty")

