#!/usr/bin/env python3
"""Compare controller-active Gazebo and onboard CFC runs.

The older comparison helpers in this directory are RL-specific.  This tool
uses the shared NN/RL logger fields so the same report can be produced for
either controller and for more than one onboard flight at a time.
"""

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
if not os.environ.get("DISPLAY"):
    import matplotlib
    matplotlib.use("Agg")

import matplotlib.pyplot as plt

from plot_onboard_log import active_airborne_xlim, intervals_from_mask, read_csv


STATE_ROWS = [
    (("pos_x", "pos_y", "pos_z"), ("x", "y", "z"), "position [m]"),
    (("vel_x", "vel_y", "vel_z"), ("vx", "vy", "vz"), "velocity [m/s]"),
    (("att_phi", "att_theta", "att_psi"), ("φ", "θ", "ψ"), "attitude [rad]"),
    (("rate_p", "rate_q", "rate_r"), ("p", "q", "r"), "body rate [rad/s]"),
]
COLORS = ["#d95f02", "#1b9e77", "#7570b3", "#e7298a", "#66a61e", "#e6ab02"]


def controller_kind(data):
    if "rl_enabled" in data:
        return "RL"
    if "nn_enabled" in data:
        return "NN"
    raise ValueError("log has neither rl_enabled nor nn_enabled")


def controller_columns(kind):
    prefix = kind.lower()
    command_prefix = "rl_rpm_cmd" if kind == "RL" else "rpm_cmd"
    applied_prefix = "rl_rpm_applied" if kind == "RL" else "rpm_applied"
    return {
        "enabled": f"{prefix}_enabled",
        "waypoint": f"{prefix}_waypoint_index",
        "target": [f"{prefix}_target_{axis}" for axis in "xyz"],
        "error": [f"{prefix}_err_{axis}" for axis in "xyz"],
        "command": [f"{command_prefix}{motor}" for motor in range(1, 5)],
        "applied": [f"{applied_prefix}{motor}" for motor in range(1, 5)],
    }


def finite_rms(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return float(np.sqrt(np.mean(values * values))) if values.size else None


def finite_peak(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return float(np.max(np.abs(values))) if values.size else None


def feedback_column(data, kind, motor):
    candidates = [
        f"rpm_obs_{motor}",
        f"rl_rpm_applied{motor}" if kind == "RL" else f"rpm_applied{motor}",
    ]
    return next((column for column in candidates if column in data), None)


def prepare_run(path, label, is_reference=False):
    _columns, data = read_csv(path)
    kind = controller_kind(data)
    names = controller_columns(kind)
    active = (data[names["enabled"]] > 0.5) & np.isfinite(data["time"])
    intervals = intervals_from_mask(data["time"], active, min_duration_s=0.5)
    if not intervals:
        raise ValueError(f"{path} has no controller-active interval")

    active_start = intervals[0][0]
    active_end = intervals[-1][1]
    (airborne_start, airborne_end), airborne_details = active_airborne_xlim(
        data, active, intervals
    )
    # Alignment stays tied to controller activation.  The airborne helper is
    # used only to remove samples after ground contact.
    end = min(active_end, airborne_end)
    mask = (
        active
        & np.isfinite(data["time"])
        & (data["time"] >= active_start)
        & (data["time"] <= end)
    )
    if not np.any(mask):
        raise ValueError(f"{path} has no usable active samples")

    return {
        "path": path,
        "label": label,
        "reference": is_reference,
        "kind": kind,
        "names": names,
        "data": data,
        "active": active,
        "mask": mask,
        "time": data["time"] - active_start,
        "active_start": float(active_start),
        "active_end": float(end),
        "duration": float(end - active_start),
        "airborne_start": float(airborne_start),
        "airborne_details": airborne_details,
    }


def waypoint_transitions(run):
    data = run["data"]
    column = run["names"]["waypoint"]
    if column not in data:
        return []
    indices = np.flatnonzero(run["mask"] & np.isfinite(data[column]))
    if indices.size == 0:
        return []
    transitions = []
    previous = int(data[column][indices[0]])
    for index in indices[1:]:
        current = int(data[column][index])
        if current != previous:
            transitions.append({
                "from": previous,
                "to": current,
                "time_s": float(run["time"][index]),
            })
            previous = current
    return transitions


def run_metrics(run):
    data = run["data"]
    mask = run["mask"]
    names = run["names"]

    position_error = None
    if all(column in data for column in names["target"]):
        error = np.column_stack([
            data[target][mask] - data[position][mask]
            for target, position in zip(names["target"], ("pos_x", "pos_y", "pos_z"))
        ])
        position_error = np.linalg.norm(error, axis=1)
    elif all(column in data for column in names["error"]):
        error = np.column_stack([data[column][mask] for column in names["error"]])
        position_error = np.linalg.norm(error, axis=1)

    rates = np.column_stack([
        data[column][mask] for column in ("rate_p", "rate_q", "rate_r")
    ])
    tracking_errors = []
    feedback_values = []
    for motor, command in enumerate(names["command"], start=1):
        feedback = feedback_column(data, run["kind"], motor)
        if feedback is None or command not in data:
            continue
        valid = mask & np.isfinite(data[command]) & np.isfinite(data[feedback])
        if np.any(valid):
            tracking_errors.append(data[command][valid] - data[feedback][valid])
            feedback_values.append(data[feedback][valid])

    return {
        "path": str(run["path"]),
        "controller": run["kind"],
        "active_start_log_time_s": run["active_start"],
        "plotted_active_duration_s": run["duration"],
        "samples": int(np.count_nonzero(mask)),
        "position_error_rms_m": finite_rms(position_error) if position_error is not None else None,
        "position_error_peak_m": finite_peak(position_error) if position_error is not None else None,
        "body_rate_rms_rad_s": finite_rms(rates),
        "body_rate_peak_rad_s": finite_peak(rates),
        "rpm_tracking_rmse": finite_rms(np.concatenate(tracking_errors)) if tracking_errors else None,
        "feedback_rpm_rms": finite_rms(np.concatenate(feedback_values)) if feedback_values else None,
        "waypoint_transitions": waypoint_transitions(run),
        "ground_contact_time_s": (
            run["airborne_details"].get("contact_time")
            if run["airborne_details"] else None
        ),
    }


def style_for(run, index):
    if run["reference"]:
        return "#111111", 1.8, 1.0
    return COLORS[index % len(COLORS)], 1.15, 0.88


def plot_states(runs, output_path, report_label):
    fig, axes = plt.subplots(4, 3, figsize=(19, 14), sharex=True, constrained_layout=True)
    for run_index, run in enumerate(runs):
        color, width, alpha = style_for(run, run_index - 1)
        mask = run["mask"]
        for row, (columns, titles, ylabel) in enumerate(STATE_ROWS):
            for column_index, (column, title) in enumerate(zip(columns, titles)):
                axis = axes[row, column_index]
                axis.plot(
                    run["time"][mask], run["data"][column][mask],
                    color=color, lw=width, alpha=alpha, label=run["label"],
                )
                axis.set_title(title)
                axis.grid(True, alpha=0.25)
                if column_index == 0:
                    axis.set_ylabel(ylabel)
                if row == 3:
                    axis.set_xlabel("time since controller activation [s]")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", bbox_to_anchor=(0.99, 0.99), fontsize=9)
    fig.suptitle(f"Gazebo × onboard states — controller active\n{report_label}", fontsize=15)
    fig.savefig(output_path, dpi=170)
    plt.close(fig)


def plot_rpm(runs, output_path, report_label):
    fig, axes = plt.subplots(4, 1, figsize=(16, 13), sharex=True, constrained_layout=True)
    for run_index, run in enumerate(runs):
        color, width, alpha = style_for(run, run_index - 1)
        mask = run["mask"]
        for motor, axis in enumerate(axes, start=1):
            feedback = feedback_column(run["data"], run["kind"], motor)
            if feedback is None:
                continue
            axis.plot(
                run["time"][mask], run["data"][feedback][mask],
                color=color, lw=width, alpha=alpha,
                label=f"{run['label']} ({feedback})",
            )
            axis.set_ylabel(f"M{motor} [RPM]")
            axis.grid(True, alpha=0.25)
    for axis in axes:
        axis.legend(loc="best", fontsize=8, ncol=2)
    axes[-1].set_xlabel("time since controller activation [s]")
    fig.suptitle(f"Motor feedback/applied RPM — Gazebo × onboard\n{report_label}", fontsize=15)
    fig.savefig(output_path, dpi=170)
    plt.close(fig)


def plot_trajectory(runs, output_path, report_label):
    fig, axis = plt.subplots(figsize=(10, 9), constrained_layout=True)
    for run_index, run in enumerate(runs):
        color, width, alpha = style_for(run, run_index - 1)
        mask = run["mask"]
        axis.plot(
            run["data"]["pos_x"][mask], run["data"]["pos_y"][mask],
            color=color, lw=width, alpha=alpha, label=run["label"],
        )
    reference = runs[0]
    targets = reference["names"]["target"]
    if all(column in reference["data"] for column in targets[:2]):
        mask = reference["mask"]
        axis.plot(
            reference["data"][targets[0]][mask], reference["data"][targets[1]][mask],
            "--", color="#666666", lw=1.0, label="Gazebo target",
        )
    axis.set_xlabel("x [m]")
    axis.set_ylabel("y [m]")
    axis.set_aspect("equal", adjustable="datalim")
    axis.grid(True, alpha=0.25)
    axis.legend(fontsize=9)
    axis.set_title(f"XY trajectory — controller active\n{report_label}")
    fig.savefig(output_path, dpi=170)
    plt.close(fig)


def plot_summary(metrics, output_path, report_label):
    labels = [item["label"] for item in metrics]
    keys = [
        ("plotted_active_duration_s", "active airborne duration [s]"),
        ("position_error_rms_m", "position error RMS [m]"),
        ("body_rate_rms_rad_s", "body-rate RMS [rad/s]"),
        ("rpm_tracking_rmse", "command−feedback RMSE [RPM]"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(16, 10), constrained_layout=True)
    for axis, (key, title) in zip(axes.flat, keys):
        values = [item.get(key) for item in metrics]
        numeric = [value if value is not None and math.isfinite(value) else 0.0 for value in values]
        bars = axis.bar(labels, numeric, color=["#555555", *COLORS[:max(0, len(labels) - 1)]])
        for bar, value in zip(bars, values):
            text = "n/a" if value is None or not math.isfinite(value) else f"{value:.3g}"
            axis.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), text,
                      ha="center", va="bottom", fontsize=8)
        axis.set_title(title)
        axis.grid(True, axis="y", alpha=0.25)
        axis.tick_params(axis="x", rotation=20)
    fig.suptitle(f"Active-run metrics — {report_label}", fontsize=15)
    fig.savefig(output_path, dpi=170)
    plt.close(fig)


def write_html(output_dir, report_label, metrics):
    images = [
        ("states_active_comparison.png", "All states"),
        ("rpm_active_comparison.png", "Motor RPM"),
        ("trajectory_xy_active_comparison.png", "XY trajectory"),
        ("metrics_active_comparison.png", "Metric summary"),
    ]
    lines = [
        "<!doctype html><html><head><meta charset=\"utf-8\">",
        f"<title>{report_label}</title>",
        "<style>body{font-family:sans-serif;margin:24px;background:#f7f7f7;color:#222}"
        "img{max-width:100%;background:white;border:1px solid #ccc;margin:8px 0 28px}"
        "table{border-collapse:collapse;background:white}th,td{border:1px solid #ccc;padding:6px}</style>",
        f"</head><body><h1>{report_label}</h1>",
        "<p>All curves are aligned at controller activation and stop at the end of the usable active/airborne window.</p>",
        "<table><tr><th>run</th><th>controller</th><th>duration [s]</th><th>position RMS [m]</th>"
        "<th>rate RMS [rad/s]</th><th>RPM tracking RMSE</th></tr>",
    ]
    for item in metrics:
        def show(key):
            value = item.get(key)
            return "n/a" if value is None else f"{value:.4g}"
        lines.append(
            f"<tr><td>{item['label']}</td><td>{item['controller']}</td>"
            f"<td>{show('plotted_active_duration_s')}</td><td>{show('position_error_rms_m')}</td>"
            f"<td>{show('body_rate_rms_rad_s')}</td><td>{show('rpm_tracking_rmse')}</td></tr>"
        )
    lines.append("</table>")
    for image, title in images:
        lines.append(f"<h2>{title}</h2><img src=\"{image}\">")
    lines.append("</body></html>")
    (output_dir / "index.html").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("gazebo_csv", type=Path)
    parser.add_argument("onboard_csv", nargs="+", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--label", default="CFC controller")
    parser.add_argument(
        "--onboard-label", action="append", default=[],
        help="Label for each onboard CSV, in positional order",
    )
    args = parser.parse_args()

    onboard_labels = args.onboard_label or [path.stem for path in args.onboard_csv]
    if len(onboard_labels) != len(args.onboard_csv):
        parser.error("the number of --onboard-label values must match onboard CSVs")

    runs = [prepare_run(args.gazebo_csv, f"Gazebo {args.gazebo_csv.stem}", True)]
    runs.extend(
        prepare_run(path, label)
        for path, label in zip(args.onboard_csv, onboard_labels)
    )
    kinds = {run["kind"] for run in runs}
    if len(kinds) != 1:
        raise SystemExit(f"controller mismatch across logs: {sorted(kinds)}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    plot_states(runs, args.output_dir / "states_active_comparison.png", args.label)
    plot_rpm(runs, args.output_dir / "rpm_active_comparison.png", args.label)
    plot_trajectory(runs, args.output_dir / "trajectory_xy_active_comparison.png", args.label)
    metrics = []
    for run in runs:
        item = run_metrics(run)
        item["label"] = run["label"]
        metrics.append(item)
    plot_summary(metrics, args.output_dir / "metrics_active_comparison.png", args.label)
    (args.output_dir / "comparison_metrics.json").write_text(
        json.dumps(metrics, indent=2, allow_nan=False) + "\n"
    )
    write_html(args.output_dir, args.label, metrics)
    print(f"Saved comparison report to {args.output_dir}")
    for item in metrics:
        print(
            f"{item['label']}: duration={item['plotted_active_duration_s']:.3f}s, "
            f"position_rms={item['position_error_rms_m']}, rate_rms={item['body_rate_rms_rad_s']:.4f}"
        )


if __name__ == "__main__":
    main()
