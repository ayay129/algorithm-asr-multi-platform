"""Generate charts from Locust CSV output.

Example:
  python3 mindie_whisperx/plot_locust_results.py \
    --csv-prefix benchmarks/20260310_120000/locust \
    --out-dir benchmarks/20260310_120000 \
    --npu-log benchmarks/20260310_120000/npu_watch.log
"""

from __future__ import annotations

import argparse
import csv
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

try:
    import matplotlib.pyplot as plt
except Exception as exc:  # pragma: no cover - runtime dependency
    raise RuntimeError("matplotlib is required. Please run: pip3 install matplotlib") from exc


def _normalize(text: str) -> str:
    return "".join(ch for ch in text.lower() if ch.isalnum())


def _pick_key(row: Dict[str, str], candidates: Iterable[str]) -> Optional[str]:
    norm_map = {_normalize(k): k for k in row.keys()}
    for candidate in candidates:
        key = norm_map.get(_normalize(candidate))
        if key:
            return key
    return None


def _to_float(value: str) -> Optional[float]:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _to_int(value: str) -> Optional[int]:
    f = _to_float(value)
    if f is None:
        return None
    return int(f)


def _read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _is_aggregated_row(row: Dict[str, str]) -> bool:
    name = (row.get("Name") or row.get("name") or "").strip().lower()
    method = (row.get("Type") or row.get("Method") or row.get("type") or "").strip().lower()
    if name in {"aggregated", "total"}:
        return True
    if not name and method == "aggregated":
        return True
    return False


def _parse_timestamp(value: str) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if text.isdigit():
        num = int(text)
        if num > 10_000_000_000:
            dt = datetime.fromtimestamp(num / 1000.0)
        else:
            dt = datetime.fromtimestamp(num)
        return dt.strftime("%H:%M:%S")
    return text


def _format_epoch_seconds(epoch_seconds: int) -> str:
    return datetime.fromtimestamp(epoch_seconds).strftime("%H:%M:%S")


def _extract_history_series(history_rows: List[Dict[str, str]]) -> Dict[str, List[float]]:
    if not history_rows:
        raise RuntimeError("locust_stats_history.csv is empty.")

    first = history_rows[0]
    timestamp_key = _pick_key(first, ["Timestamp"])
    users_key = _pick_key(first, ["User Count", "Users", "user_count"])
    rps_key = _pick_key(first, ["Requests/s", "Current RPS", "rps"])
    failures_key = _pick_key(first, ["Failures/s", "Current Failures/s", "failures/s"])
    avg_key = _pick_key(
        first,
        ["Total Average Response Time", "Average Response Time", "avg_response_time", "Avg Response Time"],
    )
    p95_key = _pick_key(first, ["95%", "95 Percentile", "p95"])
    name_key = _pick_key(first, ["Name"])

    x_labels: List[str] = []
    users: List[float] = []
    rps: List[float] = []
    failures: List[float] = []
    avg_rt: List[float] = []
    p95: List[float] = []

    for row in history_rows:
        if name_key and not _is_aggregated_row(row):
            continue
        if not timestamp_key:
            continue

        x_labels.append(_parse_timestamp(row.get(timestamp_key, "")))
        users.append(float(_to_int(row.get(users_key, "0")) or 0))
        rps.append(float(_to_float(row.get(rps_key, "0")) or 0))
        failures.append(float(_to_float(row.get(failures_key, "0")) or 0))
        avg_rt.append(float(_to_float(row.get(avg_key, "0")) or 0))
        p95.append(float(_to_float(row.get(p95_key, "0")) or 0))

    if not x_labels:
        raise RuntimeError("No aggregated rows found in locust_stats_history.csv.")

    return {
        "x_labels": x_labels,
        "users": users,
        "rps": rps,
        "failures": failures,
        "avg_rt": avg_rt,
        "p95": p95,
    }


def _extract_summary(stats_rows: List[Dict[str, str]]) -> Dict[str, float]:
    if not stats_rows:
        raise RuntimeError("locust_stats.csv is empty.")

    agg_row = None
    for row in stats_rows:
        if _is_aggregated_row(row):
            agg_row = row
            break

    if agg_row is None:
        req_key = _pick_key(stats_rows[0], ["Request Count", "Total Request Count"])
        agg_row = max(stats_rows, key=lambda r: _to_float(r.get(req_key, "0")) or 0)

    req_key = _pick_key(agg_row, ["Request Count", "Total Request Count"])
    fail_key = _pick_key(agg_row, ["Failure Count", "Total Failure Count"])
    rps_key = _pick_key(agg_row, ["Requests/s", "Current RPS"])
    avg_key = _pick_key(agg_row, ["Average Response Time", "Total Average Response Time"])
    p95_key = _pick_key(agg_row, ["95%", "95 Percentile", "p95"])

    total_requests = float(_to_float(agg_row.get(req_key, "0")) or 0)
    failure_count = float(_to_float(agg_row.get(fail_key, "0")) or 0)
    failure_rate = (failure_count / total_requests * 100.0) if total_requests > 0 else 0.0

    return {
        "total_requests": total_requests,
        "failure_count": failure_count,
        "failure_rate": failure_rate,
        "rps": float(_to_float(agg_row.get(rps_key, "0")) or 0),
        "avg_rt": float(_to_float(agg_row.get(avg_key, "0")) or 0),
        "p95": float(_to_float(agg_row.get(p95_key, "0")) or 0),
    }


def _plot_timeseries(
    x_labels: List[str],
    ys: List[Tuple[str, List[float]]],
    title: str,
    y_label: str,
    out_path: Path,
) -> None:
    x = list(range(len(x_labels)))
    plt.figure(figsize=(12, 6))
    for label, values in ys:
        plt.plot(x, values, label=label, linewidth=1.8)
    if len(x_labels) > 16:
        step = max(len(x_labels) // 16, 1)
        ticks = x[::step]
        labels = [x_labels[i] for i in ticks]
    else:
        ticks = x
        labels = x_labels
    plt.xticks(ticks, labels, rotation=30, ha="right")
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel(y_label)
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=140)
    plt.close()


def _plot_summary(summary: Dict[str, float], out_path: Path) -> None:
    labels = [
        "Total Requests",
        "Requests/s",
        "Avg RT(ms)",
        "P95 RT(ms)",
        "Failure Rate(%)",
    ]
    values = [
        summary["total_requests"],
        summary["rps"],
        summary["avg_rt"],
        summary["p95"],
        summary["failure_rate"],
    ]
    plt.figure(figsize=(10, 5.2))
    bars = plt.bar(labels, values)
    for bar, value in zip(bars, values):
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height(),
            f"{value:.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    plt.title("Locust Summary")
    plt.ylabel("Value")
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_path, dpi=140)
    plt.close()


def _percentile(values: List[float], p: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    sorted_vals = sorted(values)
    rank = (len(sorted_vals) - 1) * (p / 100.0)
    low = int(rank)
    high = min(low + 1, len(sorted_vals) - 1)
    if low == high:
        return sorted_vals[low]
    weight = rank - low
    return sorted_vals[low] * (1.0 - weight) + sorted_vals[high] * weight


def _parse_percentages(line: str) -> List[float]:
    return [float(x) for x in re.findall(r"([0-9]+(?:\.[0-9]+)?)\s*%", line)]


def _extract_npu_metrics_from_sample(sample_lines: List[str]) -> Tuple[Optional[float], Optional[float]]:
    core_keys = ["aicore", "ai core", "core util", "npu util", "device utilization"]
    mem_keys = ["hbm", "memory", "mem"]

    core_util: Optional[float] = None
    mem_util: Optional[float] = None

    for line in sample_lines:
        low = line.lower()
        percents = _parse_percentages(line)
        if not percents:
            continue

        if core_util is None and any(k in low for k in core_keys):
            core_util = percents[0]

        if mem_util is None and any(k in low for k in mem_keys):
            mem_util = percents[-1]

    # fallback: parse explicit AICore number without % sign
    if core_util is None:
        for line in sample_lines:
            m = re.search(r"aicore[^0-9]*([0-9]+(?:\.[0-9]+)?)", line.lower())
            if m:
                core_util = float(m.group(1))
                break

    return core_util, mem_util


def _extract_npu_series(npu_log: Path) -> Optional[Dict[str, List[float]]]:
    if not npu_log.exists():
        return None

    lines = npu_log.read_text(encoding="utf-8", errors="ignore").splitlines()
    if not lines:
        return None

    x_labels: List[str] = []
    core_utils: List[float] = []
    mem_utils: List[float] = []

    sample_lines: List[str] = []
    sample_ts: Optional[int] = None

    def flush_sample() -> None:
        nonlocal sample_lines, sample_ts
        if sample_ts is None:
            sample_lines = []
            return
        core_util, mem_util = _extract_npu_metrics_from_sample(sample_lines)
        if core_util is not None:
            x_labels.append(_format_epoch_seconds(sample_ts))
            core_utils.append(core_util)
            mem_utils.append(mem_util if mem_util is not None else 0.0)
        sample_lines = []

    for line in lines:
        if line.startswith("### SAMPLE_START"):
            flush_sample()
            parts = line.split()
            if parts:
                try:
                    sample_ts = int(parts[-1])
                except ValueError:
                    sample_ts = None
            else:
                sample_ts = None
            continue

        if line.startswith("### SAMPLE_END"):
            flush_sample()
            sample_ts = None
            continue

        if sample_ts is not None:
            sample_lines.append(line)

    flush_sample()

    if not core_utils:
        return None

    return {
        "x_labels": x_labels,
        "core_utils": core_utils,
        "mem_utils": mem_utils,
    }


def _build_npu_summary(series: Dict[str, List[float]]) -> Dict[str, float]:
    core = series["core_utils"]
    mem = series["mem_utils"]
    return {
        "core_avg": sum(core) / len(core),
        "core_p95": _percentile(core, 95),
        "core_max": max(core),
        "mem_avg": (sum(mem) / len(mem)) if mem else 0.0,
        "mem_p95": _percentile(mem, 95) if mem else 0.0,
        "mem_max": max(mem) if mem else 0.0,
        "samples": float(len(core)),
    }


def _write_summary_markdown(
    summary: Dict[str, float],
    max_users: float,
    out_path: Path,
    csv_prefix: str,
    npu_summary: Optional[Dict[str, float]] = None,
) -> None:
    lines = [
        "# Benchmark Summary",
        "",
        f"- csv_prefix: `{csv_prefix}`",
        f"- total_requests: `{summary['total_requests']:.0f}`",
        f"- failure_count: `{summary['failure_count']:.0f}`",
        f"- failure_rate: `{summary['failure_rate']:.3f}%`",
        f"- rps: `{summary['rps']:.3f}`",
        f"- avg_response_time_ms: `{summary['avg_rt']:.3f}`",
        f"- p95_response_time_ms: `{summary['p95']:.3f}`",
        f"- max_users: `{max_users:.0f}`",
    ]
    if npu_summary is not None:
        lines.extend(
            [
                "",
                "## NPU Metrics",
                f"- npu_core_util_avg_pct: `{npu_summary['core_avg']:.3f}`",
                f"- npu_core_util_p95_pct: `{npu_summary['core_p95']:.3f}`",
                f"- npu_core_util_max_pct: `{npu_summary['core_max']:.3f}`",
                f"- npu_mem_util_avg_pct: `{npu_summary['mem_avg']:.3f}`",
                f"- npu_mem_util_p95_pct: `{npu_summary['mem_p95']:.3f}`",
                f"- npu_mem_util_max_pct: `{npu_summary['mem_max']:.3f}`",
                f"- npu_samples: `{npu_summary['samples']:.0f}`",
            ]
        )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def generate_charts(csv_prefix: Path, out_dir: Path, npu_log: Optional[Path] = None) -> List[Path]:
    stats_file = csv_prefix.with_name(csv_prefix.name + "_stats.csv")
    history_file = csv_prefix.with_name(csv_prefix.name + "_stats_history.csv")

    if not stats_file.exists():
        raise FileNotFoundError(f"Missing file: {stats_file}")
    if not history_file.exists():
        raise FileNotFoundError(
            f"Missing file: {history_file}. Please run locust with --csv-full-history."
        )

    stats_rows = _read_csv(stats_file)
    history_rows = _read_csv(history_file)
    summary = _extract_summary(stats_rows)
    series = _extract_history_series(history_rows)

    out_dir.mkdir(parents=True, exist_ok=True)
    outputs = []

    throughput_png = out_dir / "throughput_rps.png"
    _plot_timeseries(
        series["x_labels"],
        [("Requests/s", series["rps"]), ("Failures/s", series["failures"])],
        "Throughput and Failures Over Time",
        "RPS",
        throughput_png,
    )
    outputs.append(throughput_png)

    latency_png = out_dir / "latency_ms.png"
    _plot_timeseries(
        series["x_labels"],
        [("Average RT(ms)", series["avg_rt"]), ("P95 RT(ms)", series["p95"])],
        "Latency Over Time",
        "Milliseconds",
        latency_png,
    )
    outputs.append(latency_png)

    users_png = out_dir / "users_over_time.png"
    _plot_timeseries(
        series["x_labels"],
        [("Users", series["users"])],
        "Users Over Time",
        "Users",
        users_png,
    )
    outputs.append(users_png)

    summary_png = out_dir / "summary.png"
    _plot_summary(summary, summary_png)
    outputs.append(summary_png)

    npu_summary = None
    if npu_log is not None:
        npu_series = _extract_npu_series(npu_log)
        if npu_series is not None:
            npu_png = out_dir / "npu_utilization.png"
            _plot_timeseries(
                npu_series["x_labels"],
                [("NPU Core Util(%)", npu_series["core_utils"])],
                "NPU Core Utilization Over Time",
                "Utilization(%)",
                npu_png,
            )
            outputs.append(npu_png)
            npu_summary = _build_npu_summary(npu_series)

    summary_md = out_dir / "benchmark_summary.md"
    _write_summary_markdown(
        summary,
        max(series["users"]) if series["users"] else 0.0,
        summary_md,
        str(csv_prefix),
        npu_summary=npu_summary,
    )
    outputs.append(summary_md)

    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate benchmark charts from Locust CSV files")
    parser.add_argument("--csv-prefix", required=True, help="Prefix passed to locust --csv")
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Output directory for charts. Default: parent directory of csv prefix",
    )
    parser.add_argument(
        "--npu-log",
        default=None,
        help="Optional npu-smi watch log file for utilization stats",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    csv_prefix = Path(args.csv_prefix)
    out_dir = Path(args.out_dir) if args.out_dir else csv_prefix.parent
    npu_log = Path(args.npu_log) if args.npu_log else None
    outputs = generate_charts(csv_prefix, out_dir, npu_log=npu_log)
    print("Generated:")
    for path in outputs:
        print(path)


if __name__ == "__main__":
    main()
