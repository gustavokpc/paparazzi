#!/usr/bin/env python3
"""Plot onboard logger_file CSVs extracted directly from the drone."""

import argparse
import csv
import os
from pathlib import Path

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
if not os.environ.get("DISPLAY"):
    import matplotlib
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import Patch


RPM_OBS_COLS = [f"rpm_obs_{i}" for i in range(1, 5)]
RPM_REF_COLS = [f"rpm_ref_{i}" for i in range(1, 5)]
RPM_CMD_COLS = [f"rpm_cmd{i}" for i in range(1, 5)]
NET_OUT_COLS = [f"network_out{i}_norm" for i in range(1, 5)]
CMD_COLS = ["cmd_thrust", "cmd_roll", "cmd_pitch", "cmd_yaw"]
PALETTE = ["#2166ac", "#b2182b", "#1b7837", "#984ea3", "#e08214", "#542788"]


def parse_float(value):
    if value is None or value == "":
        return np.nan
    try:
        return float(value)
    except ValueError:
        return np.nan


def read_csv(path):
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        columns = reader.fieldnames or []
        data = {column: [] for column in columns}
        for row in reader:
            for column in columns:
                data[column].append(parse_float(row.get(column)))
    return columns, {column: np.asarray(values, dtype=float) for column, values in data.items()}


def infer_nn_active(columns, data):
    """Infer NN activity for old CSVs that did not log nn_enabled explicitly."""
    if "nn_enabled" in data:
        return data["nn_enabled"] > 0.5

    signals = []
    if all(column in data for column in RPM_CMD_COLS):
        signals.append(np.nanmax(np.abs(np.vstack([data[column] for column in RPM_CMD_COLS]).T), axis=1) > 1000)
    if all(column in data for column in NET_OUT_COLS):
        signals.append(np.nanmax(np.abs(np.vstack([data[column] for column in NET_OUT_COLS]).T), axis=1) > 1e-6)

    raw_columns = [column for column in columns if column.startswith("nn_in_") and column.endswith("_raw")]
    if raw_columns:
        raw_values = np.vstack([data[column] for column in raw_columns]).T
        signals.append(np.nanmax(np.abs(raw_values), axis=1) > 1e-6)

    if not signals:
        return np.zeros_like(data["time"], dtype=bool)
    active = np.zeros_like(signals[0], dtype=bool)
    for signal in signals:
        active |= signal
    return active


def intervals_from_mask(times, mask, min_duration_s=0.2):
    indices = np.flatnonzero(mask & np.isfinite(times))
    if indices.size == 0:
        return []
    intervals = []
    for segment in np.split(indices, np.where(np.diff(indices) > 1)[0] + 1):
        start = float(times[segment[0]])
        end = float(times[segment[-1]])
        if end - start >= min_duration_s:
            intervals.append((start, end))
    return intervals


def shade_intervals(axis, intervals):
    for start, end in intervals:
        axis.axvspan(start, end, color="#fdb863", alpha=0.22, lw=0)


def available(data, names):
    return [name for name in names if name in data]


def plot_groups(output_path, title, data, groups, nn_intervals, xlim=None):
    times = data["time"]
    fig, axes = plt.subplots(len(groups), 1, figsize=(16, max(3.0 * len(groups), 5)), sharex=True, constrained_layout=True)
    if len(groups) == 1:
        axes = [axes]
    fig.suptitle(title, fontsize=14)

    for axis, (ylabel, names) in zip(axes, groups):
        shade_intervals(axis, nn_intervals)
        axis.grid(True, alpha=0.25)
        axis.set_ylabel(ylabel)
        for index, name in enumerate(available(data, names)):
            axis.plot(times, data[name], lw=0.9, color=PALETTE[index % len(PALETTE)], label=name)
        if xlim is not None:
            axis.set_xlim(*xlim)
        axis.legend(ncol=min(4, max(1, len(names))), fontsize=8, loc="upper right")

    axes[-1].set_xlabel("time [s]")
    handles, labels = axes[0].get_legend_handles_labels()
    handles.append(Patch(facecolor="#fdb863", alpha=0.35, label="NN active/inferred"))
    labels.append("NN active/inferred")
    axes[0].legend(handles=handles, labels=labels, ncol=min(5, len(labels)), fontsize=8, loc="upper right")

    fig.savefig(output_path, dpi=170)
    plt.close(fig)


def active_motor_xlim(data, threshold_rpm, pad_s):
    if not all(column in data for column in RPM_OBS_COLS):
        return None
    rpm_obs = np.vstack([data[column] for column in RPM_OBS_COLS]).T
    mean_obs = np.nanmean(rpm_obs, axis=1)
    indices = np.flatnonzero(mean_obs > threshold_rpm)
    if indices.size == 0:
        return None
    times = data["time"]
    return (
        max(float(times[0]), float(times[indices[0]]) - pad_s),
        min(float(times[-1]), float(times[indices[-1]]) + pad_s),
    )


def generate(path, output_base, threshold_rpm, pad_s, nn_active_only=False, nn_active_until=None):
    columns, data = read_csv(path)
    if "time" not in data:
        raise SystemExit(f"{path} has no time column")

    nn_active = infer_nn_active(columns, data)
    nn_intervals = intervals_from_mask(data["time"], nn_active, min_duration_s=0.5)

    out_dir = output_base / path.stem
    if nn_active_only:
        out_dir /= "nn_active"
    out_dir.mkdir(parents=True, exist_ok=True)

    state_groups = [
        ("pos [m]", ["pos_x", "pos_y", "pos_z"]),
        ("vel [m/s]", ["vel_x", "vel_y", "vel_z"]),
        ("att [rad]", ["att_phi", "att_theta", "att_psi"]),
        ("rates [rad/s]", ["rate_p", "rate_q", "rate_r"]),
        ("rpm_obs", RPM_OBS_COLS),
        ("rpm_ref", RPM_REF_COLS),
        ("cmd", CMD_COLS),
    ]
    all_state_groups = [
        ("pos [m]", ["pos_x", "pos_y", "pos_z"]),
        ("vel [m/s]", ["vel_x", "vel_y", "vel_z"]),
        ("att [rad]", ["att_phi", "att_theta", "att_psi"]),
        ("rates [rad/s]", ["rate_p", "rate_q", "rate_r"]),
    ]
    body_rate_groups = [
        ("body rates [rad/s]", ["rate_p", "rate_q", "rate_r"]),
    ]
    nn_raw_groups = [
        ("NN raw error", ["nn_in_dx_raw", "nn_in_dy_raw", "nn_in_dz_raw"]),
        ("NN raw vel", ["nn_in_vx_raw", "nn_in_vy_raw", "nn_in_vz_raw"]),
        ("NN raw att", ["nn_in_phi_raw", "nn_in_theta_raw", "nn_in_psi_raw"]),
        ("NN raw rates", ["nn_in_p_raw", "nn_in_q_raw", "nn_in_r_raw"]),
        ("NN raw Mext", ["nn_in_Mx_ext_raw", "nn_in_My_ext_raw", "nn_in_Mz_ext_raw"]),
        ("NN raw omega", ["nn_in_omega1_raw", "nn_in_omega2_raw", "nn_in_omega3_raw", "nn_in_omega4_raw"]),
        ("network out norm", NET_OUT_COLS),
        ("rpm_cmd NN", RPM_CMD_COLS),
    ]
    nn_norm_groups = [
        ("NN norm error", ["nn_in_dx_normalized", "nn_in_dy_normalized", "nn_in_dz_normalized"]),
        ("NN norm vel", ["nn_in_vx_normalized", "nn_in_vy_normalized", "nn_in_vz_normalized"]),
        ("NN norm att", ["nn_in_phi_normalized", "nn_in_theta_normalized", "nn_in_psi_normalized"]),
        ("NN norm rates", ["nn_in_p_normalized", "nn_in_q_normalized", "nn_in_r_normalized"]),
        ("NN norm Mext", ["nn_in_Mx_ext_normalized", "nn_in_My_ext_normalized", "nn_in_Mz_ext_normalized"]),
        ("NN norm omega", ["nn_in_omega1_normalized", "nn_in_omega2_normalized", "nn_in_omega3_normalized", "nn_in_omega4_normalized"]),
    ]

    outputs = []
    plotted_nn_intervals = nn_intervals
    if nn_active_only:
        if not nn_intervals:
            raise SystemExit(f"{path} has no interval with the NN active")
        nn_start = nn_intervals[0][0]
        nn_end = nn_intervals[-1][1]
        if nn_active_until is not None:
            nn_end = min(nn_end, nn_active_until)
        if nn_end <= nn_start:
            raise SystemExit(f"{path} has no NN-active samples before {nn_active_until} s")
        nn_xlim = (nn_start, nn_end)
        plotted_nn_intervals = [
            (max(start, nn_start), min(end, nn_end))
            for start, end in nn_intervals
            if end > nn_start and start < nn_end
        ]
        nn_active_state_groups = [
            ("NN raw error", ["nn_in_dx_raw", "nn_in_dy_raw", "nn_in_dz_raw"]),
            *state_groups[1:],
        ]
        position_error_groups = [
            ("x [m]", ["pos_x", "nn_in_dx_raw"]),
            ("y [m]", ["pos_y", "nn_in_dy_raw"]),
            ("z [m]", ["pos_z", "nn_in_dz_raw"]),
        ]
        outputs.append((out_dir / f"{path.stem}_state_commands_nn_active.png", nn_active_state_groups, nn_xlim, "state, commands and RPM - NN active"))
        outputs.append((out_dir / f"{path.stem}_position_vs_error_nn_active.png", position_error_groups, nn_xlim, "position vs NN raw position error - NN active"))
        outputs.append((out_dir / f"{path.stem}_all_states_nn_active.png", all_state_groups, nn_xlim, "all states - NN active"))
        outputs.append((out_dir / f"{path.stem}_body_rates_nn_active.png", body_rate_groups, nn_xlim, "body rates p q r - NN active"))
        outputs.append((out_dir / f"{path.stem}_nn_raw_outputs_nn_active.png", nn_raw_groups, nn_xlim, "NN raw inputs and outputs - NN active"))
        outputs.append((out_dir / f"{path.stem}_nn_normalized_nn_active.png", nn_norm_groups, nn_xlim, "NN normalized inputs - NN active"))
    else:
        outputs.append((out_dir / f"{path.stem}_state_commands_full.png", state_groups, None, "state, commands and RPM - full"))
        outputs.append((out_dir / f"{path.stem}_all_states_full.png", all_state_groups, None, "all states - full"))
        outputs.append((out_dir / f"{path.stem}_body_rates_full.png", body_rate_groups, None, "body rates p q r - full"))
        outputs.append((out_dir / f"{path.stem}_nn_raw_outputs_full.png", nn_raw_groups, None, "NN raw inputs and outputs - full"))
        outputs.append((out_dir / f"{path.stem}_nn_normalized_full.png", nn_norm_groups, None, "NN normalized inputs - full"))

        zoom_xlim = active_motor_xlim(data, threshold_rpm, pad_s)
        if zoom_xlim:
            outputs.append((out_dir / f"{path.stem}_state_commands_active_motors.png", state_groups, zoom_xlim, "state, commands and RPM - active motors"))
            outputs.append((out_dir / f"{path.stem}_all_states_active_motors.png", all_state_groups, zoom_xlim, "all states - active motors"))
            outputs.append((out_dir / f"{path.stem}_body_rates_active_motors.png", body_rate_groups, zoom_xlim, "body rates p q r - active motors"))

    for output_path, groups, xlim, suffix_title in outputs:
        plot_groups(output_path, f"{path.name} - {suffix_title}", data, groups, plotted_nn_intervals, xlim=xlim)

    html = [
        "<!doctype html><html><head><meta charset=\"utf-8\">",
        f"<title>{path.stem} onboard plots</title>",
        "<style>body{font-family:sans-serif;margin:24px;background:#f7f7f7;color:#222}"
        "img{max-width:100%;background:white;border:1px solid #ccc;margin:8px 0 28px}"
        "code{background:#eee;padding:2px 4px}</style></head><body>",
        f"<h1>{path.name}</h1>",
        "<p>Orange bands mark NN activity. For old logs without <code>nn_enabled</code>, activity is inferred from "
        "<code>nn_in_*</code>, <code>network_out*</code>, and <code>rpm_cmd*</code> becoming non-zero.</p>",
        f"<p>NN intervals shown: {plotted_nn_intervals or 'none'}</p>",
    ]
    for output_path, _groups, _xlim, _title in outputs:
        html.append(f"<h2>{output_path.name}</h2><img src=\"{output_path.name}\">")
    html.append("</body></html>")
    (out_dir / "index.html").write_text("\n".join(html))

    print(f"Saved onboard plots to {out_dir}")
    print(f"NN intervals shown: {plotted_nn_intervals or 'none'}")


def main():
    parser = argparse.ArgumentParser(description="Plot onboard logger_file CSVs extracted from the drone.")
    parser.add_argument("csv_files", nargs="+", type=Path, help="Onboard CSV log(s)")
    parser.add_argument("--output-dir", type=Path, default=Path("plots") / "plots_onboard")
    parser.add_argument("--active-rpm-threshold", type=float, default=1000.0)
    parser.add_argument("--active-pad-s", type=float, default=2.0)
    parser.add_argument("--nn-active-only", action="store_true", help="Plot only the interval between NN activation and deactivation")
    parser.add_argument("--nn-active-until", type=float, help="End an NN-active-only plot at this log time in seconds")
    args = parser.parse_args()

    for csv_file in args.csv_files:
        generate(
            csv_file,
            args.output_dir,
            args.active_rpm_threshold,
            args.active_pad_s,
            args.nn_active_only,
            args.nn_active_until,
        )


if __name__ == "__main__":
    main()
