from __future__ import annotations

import csv
import multiprocessing
import os
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from .config import BatchEvalConfig, EvalTaskSpec, ModelSpec
from .dataset_loader import (
    ensure_output_root,
    load_eval_dataset,
    resolve_audio_payload,
    resolve_reference_text,
    resolve_sample_id,
    resolve_sample_path,
)
from .metrics import MetricAccumulator
from .plotting import plot_metric_bars
from .reporting import build_run_directory, load_summary_files, write_csv, write_json
from .transcriber import WhisperTranscriber


@dataclass
class EvalJob:
    model: ModelSpec
    task: EvalTaskSpec


def _evaluate_job_worker(
    model_spec: ModelSpec,
    task_spec: EvalTaskSpec,
    device_id: int,
    session_dir: str,
    hf_endpoint: Optional[str],
) -> None:
    if hf_endpoint:
        os.environ["HF_ENDPOINT"] = hf_endpoint

    dataset = load_eval_dataset(task_spec.dataset)
    run_dir = build_run_directory(session_dir, model_spec.name, task_spec.name)
    details_path = run_dir / "details.csv"
    summary_path = run_dir / "summary.json"

    transcriber = WhisperTranscriber(model_spec, device_id, task_spec.whisper_language)
    accumulator = MetricAccumulator()
    started_at = time.perf_counter()

    with details_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "sample_id",
                "audio_path",
                "reference_text",
                "predicted_text",
                "normalized_reference",
                "normalized_prediction",
                "sample_wer",
                "sample_cer",
                "error",
            ],
        )
        writer.writeheader()

        try:
            for index, sample in enumerate(dataset):
                sample_id = resolve_sample_id(sample, index, task_spec.dataset)
                audio_path = resolve_sample_path(sample, task_spec.dataset)
                reference_text = resolve_reference_text(sample, task_spec.dataset)

                try:
                    audio_array, sampling_rate = resolve_audio_payload(sample, task_spec.dataset)
                    predicted_text = transcriber.transcribe_array(audio_array, sampling_rate)
                    sample_score = accumulator.add(
                        reference_text=reference_text,
                        prediction_text=predicted_text,
                        language=task_spec.whisper_language,
                    )
                    writer.writerow(
                        {
                            "sample_id": sample_id,
                            "audio_path": audio_path,
                            "reference_text": reference_text,
                            "predicted_text": predicted_text,
                            "normalized_reference": sample_score.normalized_reference,
                            "normalized_prediction": sample_score.normalized_prediction,
                            "sample_wer": f"{sample_score.wer:.6f}",
                            "sample_cer": f"{sample_score.cer:.6f}",
                            "error": "",
                        }
                    )
                except Exception as exc:
                    accumulator.add_failure()
                    writer.writerow(
                        {
                            "sample_id": sample_id,
                            "audio_path": audio_path,
                            "reference_text": reference_text,
                            "predicted_text": "",
                            "normalized_reference": "",
                            "normalized_prediction": "",
                            "sample_wer": "",
                            "sample_cer": "",
                            "error": str(exc),
                        }
                    )
                    if not task_spec.continue_on_error:
                        raise
        finally:
            transcriber.close()

    metrics = accumulator.as_dict()
    summary_payload: Dict[str, object] = {
        "model_name": model_spec.name,
        "model_path": model_spec.model_path,
        "task_name": task_spec.name,
        "whisper_language": task_spec.whisper_language,
        "device_id": device_id,
        "dataset": {
            "source_type": task_spec.dataset.source_type,
            "dataset_name": task_spec.dataset.dataset_name,
            "dataset_config": task_spec.dataset.dataset_config,
            "split": task_spec.dataset.split,
            "data_files": task_spec.dataset.data_files,
            "audio_column": task_spec.dataset.audio_column,
            "text_column": task_spec.dataset.text_column,
        },
        "metrics": metrics,
        "elapsed_seconds": round(time.perf_counter() - started_at, 3),
        "artifacts": {
            "details_csv": str(details_path),
            "summary_json": str(summary_path),
        },
    }
    write_json(summary_path, summary_payload)


def _build_jobs(config: BatchEvalConfig) -> List[EvalJob]:
    jobs: List[EvalJob] = []
    for model_spec in config.models:
        for task_spec in config.tasks:
            jobs.append(EvalJob(model=model_spec, task=task_spec))
    return jobs


def _write_session_summary(session_dir: Path, summaries: List[Dict[str, object]]) -> None:
    summary_rows = []
    for summary in summaries:
        metrics = summary["metrics"]
        summary_rows.append(
            {
                "model_name": summary["model_name"],
                "task_name": summary["task_name"],
                "whisper_language": summary["whisper_language"],
                "device_id": summary["device_id"],
                "sample_count": metrics["sample_count"],
                "failed_count": metrics["failed_count"],
                "wer": f"{float(metrics['wer']):.6f}",
                "cer": f"{float(metrics['cer']):.6f}",
                "elapsed_seconds": summary["elapsed_seconds"],
                "summary_json": summary["artifacts"]["summary_json"],
            }
        )

    write_csv(
        session_dir / "summary.csv",
        fieldnames=[
            "model_name",
            "task_name",
            "whisper_language",
            "device_id",
            "sample_count",
            "failed_count",
            "wer",
            "cer",
            "elapsed_seconds",
            "summary_json",
        ],
        rows=summary_rows,
    )
    write_json(session_dir / "summary.json", {"runs": summaries})


def run_batch_evaluation(config: BatchEvalConfig) -> Path:
    output_root = ensure_output_root(config.output_root)
    session_dir = output_root / datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir.mkdir(parents=True, exist_ok=True)

    jobs = _build_jobs(config)
    ctx = multiprocessing.get_context("spawn")
    active_processes: List[Dict[str, object]] = []
    failures: List[str] = []
    job_index = 0

    while job_index < len(jobs) or active_processes:
        for process_index in range(len(active_processes) - 1, -1, -1):
            process_info = active_processes[process_index]
            process = process_info["process"]
            if process.is_alive():
                continue

            process.join()
            if process.exitcode != 0:
                failures.append(
                    f"{process_info['job_label']} failed on device {process_info['device_id']}"
                )
            active_processes.pop(process_index)

        if job_index < len(jobs) and len(active_processes) < len(config.devices):
            used_devices = {int(item["device_id"]) for item in active_processes}
            free_devices = [device for device in config.devices if device not in used_devices]
            if free_devices:
                job = jobs[job_index]
                device_id = free_devices[0]
                job_label = f"{job.model.name} / {job.task.name}"
                process = ctx.Process(
                    target=_evaluate_job_worker,
                    args=(
                        job.model,
                        job.task,
                        device_id,
                        str(session_dir),
                        config.hf_endpoint,
                    ),
                )
                process.start()
                active_processes.append(
                    {
                        "process": process,
                        "device_id": device_id,
                        "job_label": job_label,
                    }
                )
                job_index += 1
                continue

        time.sleep(config.scheduler_poll_seconds)

    summaries = load_summary_files(session_dir)
    _write_session_summary(session_dir, summaries)

    if config.generate_plots:
        plots_dir = session_dir / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)
        plot_metric_bars(summaries, "wer", plots_dir / "wer_bar.png")
        plot_metric_bars(summaries, "cer", plots_dir / "cer_bar.png")

    if failures:
        raise RuntimeError("; ".join(failures))

    return session_dir

