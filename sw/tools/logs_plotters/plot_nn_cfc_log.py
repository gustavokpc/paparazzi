#!/usr/bin/env python3
"""Generate all NN CFC plots for one flight log into plots/plots_paparazzi/<log_name>/."""

import argparse
import csv
import math
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
if not os.environ.get("DISPLAY"):
    import matplotlib
    matplotlib.use("Agg")

import matplotlib.pyplot as plt

import analyze_nn_cfc_log as nn_log
import plot_motor_rpms as motor_plot


INPUT_LABELS = {
    "dx": "dx error [m]",
    "dy": "dy error [m]",
    "dz": "dz error [m]",
    "vx": "vx [m/s]",
    "vy": "vy [m/s]",
    "vz": "vz [m/s]",
    "phi": "phi [rad]",
    "theta": "theta [rad]",
    "psi": "psi [rad]",
    "p": "p [rad/s]",
    "q": "q [rad/s]",
    "r": "r [rad/s]",
    "Mx_ext": "Mx ext [Nm]",
    "My_ext": "My ext [Nm]",
    "Mz_ext": "Mz ext [Nm]",
    "omega1": "omega1 [RPM]",
    "omega2": "omega2 [RPM]",
    "omega3": "omega3 [RPM]",
    "omega4": "omega4 [RPM]",
}

RAW_GROUPS = [
    ("raw_error_separated", "Raw NN inputs: position error", ["dx", "dy", "dz"], False),
    ("raw_velocity_separated", "Raw NN inputs: velocity", ["vx", "vy", "vz"], False),
    ("raw_attitude_separated", "Raw NN inputs: attitude", ["phi", "theta", "psi"], False),
    ("raw_body_rates_separated", "Raw NN inputs: body rates", ["p", "q", "r"], False),
    ("raw_motor_feedback_separated", "Raw NN inputs: motor RPM feedback", ["omega1", "omega2", "omega3", "omega4"], False),
]

NORMALIZED_GROUPS = [
    ("normalized_error_velocity_separated", "Normalized NN inputs: error and velocity",
     ["dx", "dy", "dz", "vx", "vy", "vz"], True),
    ("normalized_attitude_rates_separated", "Normalized NN inputs: attitude and rates",
     ["phi", "theta", "psi", "p", "q", "r"], True),
    ("normalized_motor_feedback_separated", "Normalized NN inputs: motor RPM feedback",
     ["omega1", "omega2", "omega3", "omega4"], True),
]


def output_dir_for(path, base_dir):
    out_dir = base_dir / path.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def read_network_rows(path):
    if path.suffix.lower() == ".csv":
        rows = nn_log.read_csv_log(path)
        return rows, nn_log.CSV_NETWORK_FIELDS
    rows, _meta = nn_log.read_data_log(path)
    return rows, nn_log.DATA_NETWORK_FIELDS


def write_csv(rows, fields, output_path):
    with output_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def as_float(value):
    if value is None or value == "":
        return math.nan
    return float(value)


def trim_info_from_rpms(path, threshold_rpm, pad_s):
    rpm_rows = motor_plot.read_csv_log(path) if path.suffix.lower() == ".csv" else motor_plot.read_data_log(path)
    if not rpm_rows:
        return None
    sources = motor_plot.choose_sources(rpm_rows, ["auto"] if path.suffix.lower() == ".csv" else ["nn-input"])
    trimmed, trim_info = motor_plot.trim_to_active_rpm(rpm_rows, sources, threshold_rpm, pad_s)
    return {
        "rows": trimmed,
        "sources": sources,
        "trim_info": trim_info,
        "original_count": len(rpm_rows),
    }


def filter_active(rows, trim_info):
    if not trim_info:
        return rows
    start = trim_info["start"]
    end = trim_info["end"]
    active = [row for row in rows if start <= as_float(row.get("time")) <= end]
    return active or rows


def times_from(rows, reset=True):
    times = [as_float(row["time"]) for row in rows]
    if reset and times:
        first = times[0]
        return [time - first for time in times]
    return times


def values(rows, field):
    return [as_float(row.get(field)) for row in rows]


def plot_group(rows, names, normalized, title, output_path):
    mode = "normalized" if normalized else "raw"
    times = times_from(rows)
    fig, axes = plt.subplots(len(names), 1, figsize=(12, max(2.0 * len(names), 4)), sharex=True)
    if len(names) == 1:
        axes = [axes]

    for axis, name in zip(axes, names):
        field = f"nn_in_{name}_{mode}"
        series = values(rows, field)
        axis.plot(times, series, linewidth=1.25, color="#1f77b4")
        axis.set_ylabel(INPUT_LABELS.get(name, name) if not normalized else f"{name}_n")
        axis.grid(True, alpha=0.28)
        finite = [value for value in series if not math.isnan(value)]
        if finite:
            axis.text(
                0.995, 0.84,
                f"min {min(finite):.3g} | max {max(finite):.3g}",
                transform=axis.transAxes,
                ha="right",
                va="center",
                fontsize=8,
                bbox={
                    "boxstyle": "round,pad=0.25",
                    "facecolor": "white",
                    "edgecolor": "#cccccc",
                    "alpha": 0.85,
                },
            )

    axes[-1].set_xlabel("time from active segment start [s]")
    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def generate_for_log(path, args):
    out_dir = output_dir_for(path, args.output_dir)
    network_rows, csv_fields = read_network_rows(path)
    if not network_rows:
        raise SystemExit(f"No NN_CFC_INPUTS rows found in {path}")

    rpm_data = trim_info_from_rpms(path, args.trim_threshold_rpm, args.trim_pad_s)
    if rpm_data:
        motor_plot.plot_rpms(
            rpm_data["rows"],
            rpm_data["sources"],
            "Motor RPMs (network omega input RPM)",
            out_dir / f"{path.stem}_active_motor_rpms.png",
            show=False,
            reset_time=not args.absolute_time,
        )
        active_rows = filter_active(network_rows, rpm_data["trim_info"])
    else:
        active_rows = network_rows

    write_csv(network_rows, csv_fields, out_dir / f"{path.stem}_nn_inputs.csv")

    groups = RAW_GROUPS + (NORMALIZED_GROUPS if args.normalized else [])
    for suffix, title, names, normalized in groups:
        plot_group(
            active_rows,
            names,
            normalized,
            title,
            out_dir / f"{path.stem}_{suffix}.png",
        )

    print(f"Saved all NN CFC plots to {out_dir}")
    if rpm_data and rpm_data["trim_info"]:
        info = rpm_data["trim_info"]
        print(
            "Active segment: "
            f"{info['start']:.3f}s to {info['end']:.3f}s "
            f"({info['kept']}/{info['total']} RPM samples)"
        )
    print(f"Network input rows: {len(network_rows)}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate RPM and separated NN CFC input plots into plots/plots_paparazzi/<log_name>/."
    )
    parser.add_argument("logfiles", nargs="+", help="Paparazzi .data log or logger_file CSV")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("plots") / "plots_paparazzi",
        help="Base output directory. Defaults to plots/plots_paparazzi.",
    )
    parser.add_argument(
        "--trim-threshold-rpm",
        type=float,
        default=motor_plot.DEFAULT_TRIM_THRESHOLD_RPM,
        help="Minimum motor sample change used to detect the active segment.",
    )
    parser.add_argument(
        "--trim-pad-s",
        type=float,
        default=motor_plot.DEFAULT_TRIM_PAD_S,
        help="Seconds kept before first and after last active RPM change.",
    )
    parser.add_argument(
        "--absolute-time",
        action="store_true",
        help="Keep original log timestamps on the motor RPM plot.",
    )
    parser.add_argument(
        "--normalized",
        action="store_true",
        help="Also generate normalized NN input plots. Disabled by default.",
    )
    args = parser.parse_args()

    for raw_path in args.logfiles:
        path = Path(raw_path)
        if not path.exists():
            raise SystemExit(f"File not found: {path}")
        generate_for_log(path, args)


if __name__ == "__main__":
    main()
