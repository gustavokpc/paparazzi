#!/usr/bin/env python3
"""Measure LTC inference latency and effective execution frequency from CSV logs."""

import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
if not os.environ.get("DISPLAY"):
    import matplotlib
    matplotlib.use("Agg")

import matplotlib.pyplot as plt

from plot_onboard_log import read_csv


COLORS = ["#d73027", "#fc8d59", "#fee090", "#4575b4", "#74add1", "#313695", "#8073ac", "#1b7837"]


def percentile(values, q):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values) & (values > 0)]
    return float(np.percentile(values, q)) if values.size else None


def mean(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values) & (values > 0)]
    return float(np.mean(values)) if values.size else None


def effective_hz(time, count, mask):
    indices = np.flatnonzero(mask & np.isfinite(time) & np.isfinite(count))
    if indices.size < 2:
        return None
    duration = float(time[indices[-1]] - time[indices[0]])
    executions = float(count[indices[-1]] - count[indices[0]])
    if duration <= 0 or executions < 0:
        return None
    return executions / duration


def rolling_hz(time, count, mask, window_s=1.0):
    indices = np.flatnonzero(mask & np.isfinite(time) & np.isfinite(count))
    if indices.size < 2:
        return np.array([]), np.array([])
    active_time = time[indices]
    active_count = count[indices]
    output_time = []
    output_hz = []
    for right in range(1, indices.size):
        left = int(np.searchsorted(active_time, active_time[right] - window_s, side="left"))
        duration = active_time[right] - active_time[left]
        executions = active_count[right] - active_count[left]
        if duration >= min(0.4, window_s * 0.4) and executions >= 0:
            output_time.append(active_time[right] - active_time[0])
            output_hz.append(executions / duration)
    return np.asarray(output_time), np.asarray(output_hz)


def label_for(path):
    platform = "onboard" if "log_onboard" in path.parts else "Gazebo"
    return f"{platform} {path.stem[-6:]}"


def analyze(path):
    _columns, data = read_csv(path)
    required = [
        "time", "nn_enabled", "nn_periodic_count", "nn_periodic_dt_us",
        "nn_inference_time_us", "nn_total_time_us",
    ]
    missing = [column for column in required if column not in data]
    if missing:
        raise ValueError(f"{path}: missing {', '.join(missing)}")

    enabled = data["nn_enabled"] > 0.5
    active_indices = np.flatnonzero(enabled & np.isfinite(data["time"]))
    if active_indices.size < 2:
        raise ValueError(f"{path}: no usable NN-active interval")
    motors = data.get("motors_on", np.ones_like(enabled)) > 0.5
    in_flight = data.get("autopilot_in_flight", np.ones_like(enabled)) > 0.5
    usable = enabled & motors & in_flight

    inference = data["nn_inference_time_us"][enabled]
    total = data["nn_total_time_us"][enabled]
    sensor = data.get("nn_sensor_read_time_us", np.full_like(data["time"], np.nan))[enabled]
    period = data["nn_periodic_dt_us"][enabled]
    inference_valid = inference[np.isfinite(inference) & (inference > 0)]
    total_valid = total[np.isfinite(total) & (total > 0)]
    period_valid = period[np.isfinite(period) & (period > 0)]
    paired = (
        np.isfinite(data["nn_total_time_us"])
        & np.isfinite(data["nn_periodic_dt_us"])
        & (data["nn_total_time_us"] > 0)
        & (data["nn_periodic_dt_us"] > 0)
        & enabled
    )
    hz = effective_hz(data["time"], data["nn_periodic_count"], enabled)
    usable_hz = effective_hz(data["time"], data["nn_periodic_count"], usable)
    active_duration = float(data["time"][active_indices[-1]] - data["time"][active_indices[0]])
    count_delta = float(
        data["nn_periodic_count"][active_indices[-1]]
        - data["nn_periodic_count"][active_indices[0]]
    )
    total_mean = mean(total_valid)
    inferred_period_us = 1.0e6 / hz if hz else None
    utilization = total_mean / inferred_period_us * 100.0 if total_mean and inferred_period_us else None
    inference_fraction = (
        mean(inference_valid) / total_mean * 100.0 if total_mean and inference_valid.size else None
    )
    overruns = int(np.count_nonzero(data["nn_total_time_us"][paired] > data["nn_periodic_dt_us"][paired]))
    paired_count = int(np.count_nonzero(paired))

    result = {
        "label": label_for(path),
        "path": str(path),
        "platform": "onboard" if "log_onboard" in path.parts else "Gazebo",
        "active_duration_s": active_duration,
        "periodic_count_delta": count_delta,
        "effective_hz": hz,
        "usable_flight_effective_hz": usable_hz,
        "effective_period_us": inferred_period_us,
        "period_dt_mean_us": mean(period_valid),
        "period_dt_median_us": percentile(period_valid, 50),
        "period_dt_p95_us": percentile(period_valid, 95),
        "period_dt_p99_us": percentile(period_valid, 99),
        "period_dt_max_us": percentile(period_valid, 100),
        "inference_mean_us": mean(inference_valid),
        "inference_median_us": percentile(inference_valid, 50),
        "inference_p95_us": percentile(inference_valid, 95),
        "inference_p99_us": percentile(inference_valid, 99),
        "inference_max_us": percentile(inference_valid, 100),
        "sensor_read_mean_us": mean(sensor),
        "total_mean_us": total_mean,
        "total_median_us": percentile(total_valid, 50),
        "total_p95_us": percentile(total_valid, 95),
        "total_p99_us": percentile(total_valid, 99),
        "total_max_us": percentile(total_valid, 100),
        "estimated_compute_utilization_percent": utilization,
        "inference_share_of_total_percent": inference_fraction,
        "sampled_total_over_period_count": overruns,
        "sampled_total_period_pairs": paired_count,
    }
    roll_time, roll_hz = rolling_hz(
        data["time"], data["nn_periodic_count"], enabled, window_s=1.0
    )
    return result, inference_valid, roll_time, roll_hz


def short_labels(results):
    return [result["label"] for result in results]


def plot_summary(results, output_path):
    labels = short_labels(results)
    x = np.arange(len(results))
    colors = COLORS[:len(results)]
    fig, axes = plt.subplots(2, 2, figsize=(18, 11), constrained_layout=True)

    hz = [result["effective_hz"] for result in results]
    bars = axes[0, 0].bar(x, hz, color=colors)
    axes[0, 0].set_title("Effective LTC execution frequency")
    axes[0, 0].set_ylabel("Hz (counter delta / elapsed time)")
    for bar, value in zip(bars, hz):
        axes[0, 0].text(bar.get_x() + bar.get_width() / 2, value, f"{value:.2f}", ha="center", va="bottom", fontsize=8)

    width = 0.36
    median_ms = [result["inference_median_us"] / 1000.0 for result in results]
    p95_ms = [result["inference_p95_us"] / 1000.0 for result in results]
    axes[0, 1].bar(x - width / 2, median_ms, width, color=colors, alpha=0.75, label="median")
    axes[0, 1].bar(x + width / 2, p95_ms, width, color=colors, label="p95")
    axes[0, 1].set_title("LTC inference latency")
    axes[0, 1].set_ylabel("ms")
    axes[0, 1].legend()

    total_median_ms = [result["total_median_us"] / 1000.0 for result in results]
    total_p95_ms = [result["total_p95_us"] / 1000.0 for result in results]
    axes[1, 0].bar(x - width / 2, total_median_ms, width, color=colors, alpha=0.75, label="median")
    axes[1, 0].bar(x + width / 2, total_p95_ms, width, color=colors, label="p95")
    axes[1, 0].set_title("Complete NN periodic function latency")
    axes[1, 0].set_ylabel("ms")
    axes[1, 0].legend()

    utilization = [result["estimated_compute_utilization_percent"] for result in results]
    bars = axes[1, 1].bar(x, utilization, color=colors)
    axes[1, 1].set_title("Estimated compute utilization at effective frequency")
    axes[1, 1].set_ylabel("total time / effective period [%]")
    for bar, value in zip(bars, utilization):
        axes[1, 1].text(bar.get_x() + bar.get_width() / 2, value, f"{value:.1f}%", ha="center", va="bottom", fontsize=8)

    for axis in axes.flat:
        axis.set_xticks(x, labels, rotation=25, ha="right")
        axis.grid(True, axis="y", alpha=0.25)
    fig.suptitle("LTC onboard/Gazebo timing summary", fontsize=16)
    fig.savefig(output_path, dpi=170)
    plt.close(fig)


def plot_distributions(results, inference_values, output_path):
    labels = short_labels(results)
    values_ms = [values / 1000.0 for values in inference_values]
    fig, axis = plt.subplots(figsize=(17, 9), constrained_layout=True)
    boxes = axis.boxplot(values_ms, labels=labels, showfliers=False, whis=(1, 99), patch_artist=True)
    for patch, color in zip(boxes["boxes"], COLORS):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    axis.set_ylabel("inference time [ms]")
    axis.set_title("LTC inference latency distribution (whiskers: p1–p99)")
    axis.tick_params(axis="x", rotation=25)
    axis.grid(True, axis="y", alpha=0.25)
    fig.savefig(output_path, dpi=170)
    plt.close(fig)


def plot_rolling_hz(results, rolling, output_path):
    fig, axis = plt.subplots(figsize=(17, 9), constrained_layout=True)
    for result, (times, values), color in zip(results, rolling, COLORS):
        axis.plot(times, values, lw=1.1, color=color, label=result["label"])
        axis.axhline(result["effective_hz"], lw=0.8, ls="--", color=color, alpha=0.6)
    axis.set_xlabel("time since LTC activation [s]")
    axis.set_ylabel("effective frequency over trailing ~1 s [Hz]")
    axis.set_title("LTC effective frequency and jitter")
    axis.grid(True, alpha=0.25)
    axis.legend(ncol=2, fontsize=9)
    fig.savefig(output_path, dpi=170)
    plt.close(fig)


def write_csv(results, path):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)


def write_html(results, output_dir):
    rows = []
    for result in results:
        rows.append(
            "<tr>"
            f"<td>{result['label']}</td>"
            f"<td>{result['effective_hz']:.2f}</td>"
            f"<td>{result['inference_mean_us'] / 1000.0:.3f}</td>"
            f"<td>{result['inference_median_us'] / 1000.0:.3f}</td>"
            f"<td>{result['inference_p95_us'] / 1000.0:.3f}</td>"
            f"<td>{result['inference_p99_us'] / 1000.0:.3f}</td>"
            f"<td>{result['inference_max_us'] / 1000.0:.3f}</td>"
            f"<td>{result['total_mean_us'] / 1000.0:.3f}</td>"
            f"<td>{result['estimated_compute_utilization_percent']:.1f}%</td>"
            "</tr>"
        )
    html = f"""<!doctype html><html><head><meta charset="utf-8">
<title>LTC timing report</title>
<style>body{{font-family:sans-serif;margin:24px;background:#f7f7f7;color:#222}}table{{border-collapse:collapse;background:white}}th,td{{border:1px solid #ccc;padding:6px}}img{{max-width:100%;background:white;border:1px solid #ccc;margin:8px 0 28px}}</style>
</head><body><h1>LTC inference timing and effective frequency</h1>
<p><strong>Effective Hz</strong> is measured from <code>Δnn_periodic_count / Δtime</code> over the enabled interval. It therefore includes scheduling, inference, sensor acquisition, and logging interference; it is not the theoretical reciprocal of inference latency.</p>
<table><tr><th>run</th><th>effective Hz</th><th>inference mean [ms]</th><th>median [ms]</th><th>p95 [ms]</th><th>p99 [ms]</th><th>max [ms]</th><th>total mean [ms]</th><th>compute utilization</th></tr>
{''.join(rows)}</table>
<h2>Summary</h2><img src="ltc_timing_summary.png">
<h2>Inference distributions</h2><img src="ltc_inference_distribution.png">
<h2>Rolling effective frequency</h2><img src="ltc_effective_hz_rolling.png">
</body></html>"""
    (output_dir / "index.html").write_text(html)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_files", nargs="+", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    analyses = [analyze(path) for path in args.csv_files]
    results = [item[0] for item in analyses]
    inference_values = [item[1] for item in analyses]
    rolling = [(item[2], item[3]) for item in analyses]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plot_summary(results, args.output_dir / "ltc_timing_summary.png")
    plot_distributions(results, inference_values, args.output_dir / "ltc_inference_distribution.png")
    plot_rolling_hz(results, rolling, args.output_dir / "ltc_effective_hz_rolling.png")
    (args.output_dir / "ltc_timing_metrics.json").write_text(
        json.dumps(results, indent=2, allow_nan=False) + "\n"
    )
    write_csv(results, args.output_dir / "ltc_timing_metrics.csv")
    write_html(results, args.output_dir)
    print(f"Saved LTC timing report to {args.output_dir}")
    for result in results:
        print(
            f"{result['label']}: {result['effective_hz']:.3f} Hz effective, "
            f"inference mean={result['inference_mean_us'] / 1000.0:.3f} ms, "
            f"p95={result['inference_p95_us'] / 1000.0:.3f} ms, "
            f"total mean={result['total_mean_us'] / 1000.0:.3f} ms"
        )


if __name__ == "__main__":
    main()
