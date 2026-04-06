from __future__ import annotations

from pathlib import Path
from typing import Any, Tuple

from datasets import Audio, Dataset, load_dataset

from .config import DatasetSpec


def load_eval_dataset(spec: DatasetSpec) -> Dataset:
    """Load a dataset and cast its audio column to a fixed sampling rate."""

    if spec.source_type == "json":
        dataset = load_dataset(
            "json",
            data_files=spec.data_files,
            split=spec.split,
        )
    else:
        load_args = [spec.dataset_name]
        if spec.dataset_config:
            load_args.append(spec.dataset_config)
        dataset = load_dataset(
            *load_args,
            split=spec.split,
            cache_dir=spec.cache_dir,
        )

    return dataset.cast_column(spec.audio_column, Audio(sampling_rate=spec.sampling_rate))


def resolve_sample_id(sample: Any, index: int, spec: DatasetSpec) -> str:
    if spec.id_column and sample.get(spec.id_column) is not None:
        return str(sample[spec.id_column])
    return str(index)


def resolve_sample_path(sample: Any, spec: DatasetSpec) -> str:
    if spec.path_column and sample.get(spec.path_column):
        return str(sample[spec.path_column])

    audio_field = sample.get(spec.audio_column)
    if spec.path_from_audio and isinstance(audio_field, dict) and audio_field.get("path"):
        return str(audio_field["path"])

    return ""


def resolve_reference_text(sample: Any, spec: DatasetSpec) -> str:
    return str(sample.get(spec.text_column, "") or "")


def resolve_audio_payload(sample: Any, spec: DatasetSpec) -> Tuple[Any, int]:
    audio_field = sample[spec.audio_column]
    audio_array = audio_field["array"]
    sampling_rate = int(audio_field.get("sampling_rate", spec.sampling_rate))
    return audio_array, sampling_rate


def ensure_output_root(path: str | Path) -> Path:
    output_root = Path(path).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    return output_root

