"""Generate charts from Locust CSV output.

Example:
  python3 mindie_whisperx/plot_locust_results.py \
    --csv-prefix benchmarks/20260310_120000/locust \
    --out-dir benchmarks/20260310_120000
"""

from __future__ import annotations

import argparse
import csv
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


def _write_summary_markdown(
    summary: Dict[str, float], max_users: float, out_path: Path, csv_prefix: str
) -> None:
    content = (
        "# Benchmark Summary\n\n"
        f"- csv_prefix: `{csv_prefix}`\n"
        f"- total_requests: `{summary['total_requests']:.0f}`\n"
        f"- failure_count: `{summary['failure_count']:.0f}`\n"
        f"- failure_rate: `{summary['failure_rate']:.3f}%`\n"
        f"- rps: `{summary['rps']:.3f}`\n"
        f"- avg_response_time_ms: `{summary['avg_rt']:.3f}`\n"
        f"- p95_response_time_ms: `{summary['p95']:.3f}`\n"
        f"- max_users: `{max_users:.0f}`\n"
    )
    out_path.write_text(content, encoding="utf-8")


def generate_charts(csv_prefix: Path, out_dir: Path) -> List[Path]:
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

    summary_md = out_dir / "benchmark_summary.md"
    _write_summary_markdown(summary, max(series["users"]) if series["users"] else 0.0, summary_md, str(csv_prefix))
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    csv_prefix = Path(args.csv_prefix)
    out_dir = Path(args.out_dir) if args.out_dir else csv_prefix.parent
    outputs = generate_charts(csv_prefix, out_dir)
    print("Generated:")
    for path in outputs:
        print(path)


if __name__ == "__main__":
    main()
