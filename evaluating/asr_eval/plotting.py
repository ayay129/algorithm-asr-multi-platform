from __future__ import annotations

from pathlib import Path
from typing import Dict, List


def plot_metric_bars(summaries: List[Dict[str, object]], metric_key: str, output_path: str | Path) -> bool:
    if not summaries:
        return False

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return False

    task_names = sorted({str(item["task_name"]) for item in summaries})
    model_names = sorted({str(item["model_name"]) for item in summaries})
    lookup = {
        (str(item["task_name"]), str(item["model_name"])): float(item["metrics"][metric_key])
        for item in summaries
    }

    figure, axis = plt.subplots(figsize=(max(8, len(task_names) * 1.8), 5))
    bar_width = 0.8 / max(1, len(model_names))
    x_positions = list(range(len(task_names)))

    for model_index, model_name in enumerate(model_names):
        offsets = [
            base_position - 0.4 + (model_index + 0.5) * bar_width
            for base_position in x_positions
        ]
        values = [lookup.get((task_name, model_name), 0.0) for task_name in task_names]
        axis.bar(offsets, values, width=bar_width, label=model_name)

    axis.set_title(f"{metric_key.upper()} by task and model")
    axis.set_ylabel(metric_key.upper())
    axis.set_xticks(x_positions)
    axis.set_xticklabels(task_names, rotation=20, ha="right")
    axis.set_ylim(0.0, 1.0)
    axis.legend()
    axis.grid(axis="y", linestyle="--", alpha=0.3)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)
    return True

