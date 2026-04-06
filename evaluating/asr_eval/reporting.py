from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Dict, Iterable, List


def slugify_name(value: str) -> str:
    normalized = re.sub(r"[^0-9A-Za-z._-]+", "-", value.strip())
    normalized = normalized.strip("-")
    return normalized or "unnamed"


def build_run_directory(session_dir: str | Path, model_name: str, task_name: str) -> Path:
    run_dir = Path(session_dir) / slugify_name(model_name) / slugify_name(task_name)
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def write_csv(path: str | Path, fieldnames: List[str], rows: Iterable[Dict[str, object]]) -> None:
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_json(path: str | Path, payload: Dict[str, object]) -> None:
    with Path(path).open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def load_summary_files(session_dir: str | Path) -> List[Dict[str, object]]:
    summaries: List[Dict[str, object]] = []
    for summary_path in sorted(Path(session_dir).glob("*/*/summary.json")):
        with summary_path.open("r", encoding="utf-8") as handle:
            summaries.append(json.load(handle))
    return summaries

