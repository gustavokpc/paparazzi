#!/usr/bin/env python3
"""Summarize controller-active and conventional-control phases onboard."""

import argparse
import json
import os
from pathlib import Path

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
if not os.environ.get("DISPLAY"):
    import matplotlib
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from plot_onboard_log import intervals_from_mask, read_csv


PHASE_COLORS = {"motors": "#bdbdbd", "inactive": "#3182bd", "active": "#e6550d"}


def controller(data):
    if "rl_enabled" in data:
        return "RL", "rl_enabled"
    if "nn_enabled" in data:
        return "NN", "nn_enabled"
    raise ValueError("missing controller enabled column")


def mask_duration(times, mask):
    indices = np.flatnonzero(mask & np.isfinite(times))
    if indices.size < 2:
        return 0.0
    dt = np.diff(times[indices])
    dt = dt[np.isfinite(dt) & (dt > 0) & (dt < 0.2)]
    return float(np.sum(dt)) if dt.size else 0.0


def rms(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return float(np.sqrt(np.mean(values * values))) if values.size else None


def phase_metrics(data, mask, command_columns):
    rates = np.column_stack([data[name][mask] for name in ("rate_p", "rate_q", "rate_r")])
    tilt = np.column_stack([data[name][mask] for name in ("att_phi", "att_theta")])
    rpm_errors = []
    for motor, command in enumerate(command_columns, start=1):
        observed = f"rpm_obs_{motor}"
        if observed in data and command in data:
            valid = mask & np.isfinite(data[observed]) & np.isfinite(data[command])
            if np.any(valid):
                rpm_errors.append(data[command][valid] - data[observed][valid])
    return {
        "duration_s": mask_duration(data["time"], mask),
        "body_rate_rms_rad_s": rms(rates),
        "roll_pitch_rms_rad": rms(tilt),
        "rpm_command_tracking_rmse": rms(np.concatenate(rpm_errors)) if rpm_errors else None,
        "samples": int(np.count_nonzero(mask)),
    }


def prepare(path):
    _columns, data = read_csv(path)
    kind, enabled_column = controller(data)
    finite = np.isfinite(data["time"])
    enabled = data[enabled_column] > 0.5
    motors = (data.get("motors_on", np.zeros_like(data["time"])) > 0.5) & finite
    in_flight = (data.get("autopilot_in_flight", np.zeros_like(data["time"])) > 0.5) & finite
    usable_flight = motors & in_flight
    active = usable_flight & enabled
    inactive = usable_flight & ~enabled
    inactive_commands = [f"rpm_ref_{motor}" for motor in range(1, 5)]
    active_prefix = "rl_rpm_cmd" if kind == "RL" else "rpm_cmd"
    active_commands = [f"{active_prefix}{motor}" for motor in range(1, 5)]
    return {
        "path": path,
        "label": f"{kind} {path.stem[-6:]}",
        "kind": kind,
        "data": data,
        "motors": motors,
        "in_flight": in_flight,
        "active": active,
        "inactive": inactive,
        "metrics": {
            "inactive": phase_metrics(data, inactive, inactive_commands),
            "active": phase_metrics(data, active, active_commands),
        },
    }


def plot_timeline(runs, output_path):
    fig, axis = plt.subplots(figsize=(17, max(6, 0.8 * len(runs) + 2)), constrained_layout=True)
    for row, run in enumerate(runs):
        times = run["data"]["time"]
        for phase, offset, height in (("motors", -0.28, 0.56), ("inactive", -0.20, 0.40), ("active", -0.20, 0.40)):
            for start, end in intervals_from_mask(times, run[phase], min_duration_s=0.05):
                axis.broken_barh(
                    [(start, end - start)], (row + offset, height),
                    facecolors=PHASE_COLORS[phase], edgecolors="none",
                )
    axis.set_yticks(range(len(runs)), [run["label"] for run in runs])
    axis.set_xlabel("log time [s]")
    axis.set_title("Onboard flights — conventional and CFC-active phases")
    axis.grid(True, axis="x", alpha=0.25)
    axis.legend(handles=[
        Patch(facecolor=PHASE_COLORS["motors"], label="motors on"),
        Patch(facecolor=PHASE_COLORS["inactive"], label="in flight, CFC inactive"),
        Patch(facecolor=PHASE_COLORS["active"], label="in flight, CFC active"),
    ], loc="upper right")
    fig.savefig(output_path, dpi=170)
    plt.close(fig)


def plot_metrics(runs, output_path):
    labels = [run["label"] for run in runs]
    keys = [
        ("duration_s", "usable phase duration [s]"),
        ("body_rate_rms_rad_s", "body-rate RMS [rad/s]"),
        ("roll_pitch_rms_rad", "roll/pitch RMS [rad]"),
        ("rpm_command_tracking_rmse", "active command/reference−observed RPM RMSE"),
    ]
    x = np.arange(len(runs))
    width = 0.38
    fig, axes = plt.subplots(2, 2, figsize=(18, 11), constrained_layout=True)
    for axis, (key, title) in zip(axes.flat, keys):
        inactive = [run["metrics"]["inactive"].get(key) for run in runs]
        active = [run["metrics"]["active"].get(key) for run in runs]
        inactive_values = [value if value is not None else 0.0 for value in inactive]
        active_values = [value if value is not None else 0.0 for value in active]
        axis.bar(x - width / 2, inactive_values, width, color=PHASE_COLORS["inactive"], label="CFC inactive")
        axis.bar(x + width / 2, active_values, width, color=PHASE_COLORS["active"], label="CFC active")
        axis.set_xticks(x, labels, rotation=25, ha="right")
        axis.set_title(title)
        axis.grid(True, axis="y", alpha=0.25)
        axis.legend(fontsize=8)
    fig.suptitle("Onboard active × inactive phase metrics", fontsize=15)
    fig.savefig(output_path, dpi=170)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_files", nargs="+", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    runs = [prepare(path) for path in args.csv_files]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plot_timeline(runs, args.output_dir / "controller_activity_timeline.png")
    plot_metrics(runs, args.output_dir / "active_vs_inactive_metrics.png")

    summary = []
    for run in runs:
        summary.append({
            "label": run["label"],
            "path": str(run["path"]),
            "controller": run["kind"],
            "inactive": run["metrics"]["inactive"],
            "active": run["metrics"]["active"],
        })
    (args.output_dir / "activity_metrics.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n"
    )
    html = """<!doctype html><html><head><meta charset="utf-8">
<title>Onboard active vs inactive</title>
<style>body{font-family:sans-serif;margin:24px;background:#f7f7f7;color:#222}img{max-width:100%;background:white;border:1px solid #ccc;margin:8px 0 28px}</style>
</head><body><h1>Onboard controller activity comparison</h1>
<p>Inactive metrics use samples with motors on and autopilot in flight before/after CFC activation. Active metrics use the corresponding CFC-enabled samples.</p>
<h2>Activity timeline</h2><img src="controller_activity_timeline.png">
<h2>Active vs inactive metrics</h2><img src="active_vs_inactive_metrics.png">
</body></html>"""
    (args.output_dir / "index.html").write_text(html)
    print(f"Saved onboard activity summary to {args.output_dir}")
    for item in summary:
        print(
            f"{item['label']}: inactive={item['inactive']['duration_s']:.3f}s, "
            f"active={item['active']['duration_s']:.3f}s"
        )


if __name__ == "__main__":
    main()
