#!/usr/bin/env python3
"""Plot per-motor RPMs from NN CFC flight logs."""

import argparse
import csv
import os
import re
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
if not os.environ.get("DISPLAY"):
    import matplotlib
    matplotlib.use("Agg")

import matplotlib.pyplot as plt


SOURCES = {
    "cmd": [f"rpm_cmd{i}" for i in range(1, 5)],
    "obs": [f"rpm_obs_{i}" for i in range(1, 5)],
    "ref": [f"rpm_ref_{i}" for i in range(1, 5)],
    "nn-input": [f"nn_in_omega{i}_raw" for i in range(1, 5)],
}

SOURCE_LABELS = {
    "cmd": "commanded NN RPM",
    "obs": "observed motor RPM",
    "ref": "reference motor RPM",
    "nn-input": "network omega input RPM",
}

DEFAULT_TRIM_THRESHOLD_RPM = 10.0
DEFAULT_TRIM_PAD_S = 0.0


def parse_float(value):
    if value is None or value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def parse_debug_vector(text):
    return [float(item) for item in text.split(",") if item]


def read_csv_log(path):
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        rows = []
        for row in reader:
            time_value = parse_float(row.get("time"))
            if time_value is None:
                continue
            parsed = {"time": time_value}
            for key, value in row.items():
                parsed[key] = parse_float(value)
            parsed["time"] = time_value
            rows.append(parsed)
    return rows


def read_data_log(path):
    rows = []
    line_re = re.compile(r"^(\S+)\s+(\S+)\s+DEBUG_VECT\s+NN_CFC_INPUTS\s+(.+)$")
    with path.open(errors="replace") as stream:
        for line in stream:
            match = line_re.match(line.strip())
            if not match:
                continue
            time_text, _aircraft_id, vector_text = match.groups()
            time_value = parse_float(time_text)
            if time_value is None:
                continue
            vector = parse_debug_vector(vector_text)
            if len(vector) < 19:
                continue
            rows.append({
                "time": time_value,
                "nn_in_omega1_raw": vector[15],
                "nn_in_omega2_raw": vector[16],
                "nn_in_omega3_raw": vector[17],
                "nn_in_omega4_raw": vector[18],
            })
    return rows


def choose_sources(rows, requested_sources):
    if requested_sources != ["auto"]:
        return requested_sources
    for source in ("cmd", "obs", "ref", "nn-input"):
        fields = SOURCES[source]
        if all(any(row.get(field) is not None for row in rows) for field in fields):
            return [source]
    raise SystemExit("Could not find RPM columns. Try --source nn-input for .data logs.")


def observed_hz(times):
    if len(times) < 2:
        return None
    dts = [b - a for a, b in zip(times, times[1:]) if b > a]
    if not dts:
        return None
    return 1.0 / (sum(dts) / len(dts))


def rpm_fields_for_sources(sources):
    fields = []
    for source in sources:
        fields.extend(SOURCES[source])
    return fields


def row_rpm_values(row, fields):
    return [row.get(field) for field in fields if row.get(field) is not None]


def trim_to_active_rpm(rows, sources, threshold_rpm, pad_s):
    fields = rpm_fields_for_sources(sources)
    change_indices = []
    previous = None
    for index, row in enumerate(rows):
        current = row_rpm_values(row, fields)
        if not current:
            continue
        if previous is not None:
            max_delta = max(abs(a - b) for a, b in zip(current, previous))
            if max_delta >= threshold_rpm:
                change_indices.append(index)
        previous = current

    if not change_indices:
        return rows, None

    start_time = rows[change_indices[0]]["time"] - pad_s
    end_time = rows[change_indices[-1]]["time"] + pad_s
    trimmed = [
        row for row in rows
        if start_time <= row["time"] <= end_time
    ]
    if not trimmed:
        return rows, None

    return trimmed, {
        "start": trimmed[0]["time"],
        "end": trimmed[-1]["time"],
        "first_change": rows[change_indices[0]]["time"],
        "last_change": rows[change_indices[-1]]["time"],
        "change_count": len(change_indices),
        "threshold": threshold_rpm,
        "pad": pad_s,
        "kept": len(trimmed),
        "total": len(rows),
    }


def plot_rpms(rows, sources, title, output, show, reset_time):
    first_time = rows[0]["time"] if reset_time else 0.0
    times = [row["time"] - first_time for row in rows]
    fig, axes = plt.subplots(4, 1, figsize=(12, 9), sharex=True)
    colors = {
        "cmd": "tab:blue",
        "obs": "tab:orange",
        "ref": "tab:green",
        "nn-input": "tab:red",
    }

    for motor_idx, axis in enumerate(axes, start=1):
        plotted = False
        for source in sources:
            field = SOURCES[source][motor_idx - 1]
            values = [row.get(field) for row in rows]
            if all(value is None for value in values):
                continue
            axis.plot(times, values, label=SOURCE_LABELS[source], color=colors[source], linewidth=1.2)
            plotted = True

        axis.set_ylabel(f"Motor {motor_idx}\nRPM")
        axis.grid(True, alpha=0.3)
        if plotted and len(sources) > 1:
            axis.legend(loc="upper right")

    axes[-1].set_xlabel("time from plotted segment start [s]" if reset_time else "time [s]")
    hz = observed_hz([row["time"] for row in rows])
    hz_text = f" | observed {hz:.2f} Hz" if hz else ""
    fig.suptitle(f"{title}{hz_text}")
    fig.tight_layout(rect=(0, 0, 1, 0.97))

    if output:
        fig.savefig(output, dpi=160)
        print(f"Saved RPM plot to {output}")
    if show:
        plt.show()
    plt.close(fig)


def plot_rpms_separate(rows, sources, title, output, show, reset_time):
    """Plot one panel per motor and source (four motors x N sources)."""
    first_time = rows[0]["time"] if reset_time else 0.0
    times = [row["time"] - first_time for row in rows]
    fig, axes = plt.subplots(
        4,
        len(sources),
        figsize=(6 * len(sources), 9),
        sharex=True,
        squeeze=False,
    )
    colors = {
        "cmd": "tab:blue",
        "obs": "tab:orange",
        "ref": "tab:green",
        "nn-input": "tab:orange",
    }

    for motor_index in range(4):
        for source_index, source in enumerate(sources):
            axis = axes[motor_index][source_index]
            field = SOURCES[source][motor_index]
            values = [row.get(field) for row in rows]
            axis.plot(times, values, color=colors[source], linewidth=1.2)
            axis.set_ylabel(f"Motor {motor_index + 1}\nRPM")
            axis.set_title(SOURCE_LABELS[source] if motor_index == 0 else "")
            axis.grid(True, alpha=0.3)

    x_label = "time from plotted segment start [s]" if reset_time else "time [s]"
    for axis in axes[-1]:
        axis.set_xlabel(x_label)
    hz = observed_hz([row["time"] for row in rows])
    hz_text = f" | observed {hz:.2f} Hz" if hz else ""
    fig.suptitle(f"{title}{hz_text}")
    fig.tight_layout(rect=(0, 0, 1, 0.97))

    if output:
        fig.savefig(output, dpi=160)
        print(f"Saved RPM plot to {output}")
    if show:
        plt.show()
    plt.close(fig)


def default_output_path(logfile):
    path = Path(logfile)
    plots_dir = Path.cwd() / "plots" / "plots_paparazzi"
    plots_dir.mkdir(parents=True, exist_ok=True)
    return plots_dir / f"{path.stem}_active_motor_rpms.png"


def analyze_one(path, args):
    rows = read_csv_log(path) if path.suffix.lower() == ".csv" else read_data_log(path)
    if not rows:
        print(f"Skipping {path}: no usable RPM data found")
        return

    sources = choose_sources(rows, args.source or ["auto"])
    original_count = len(rows)
    trim_info = None
    if not args.no_trim:
        rows, trim_info = trim_to_active_rpm(
            rows,
            sources,
            args.trim_threshold_rpm,
            args.trim_pad_s,
        )
    if args.output and len(args.logfiles) == 1:
        output = args.output
    else:
        output = str(default_output_path(path))

    source_text = ", ".join(SOURCE_LABELS[source] for source in sources)
    print(f"Loaded {original_count} samples from {path}")
    print(f"Plotting: {source_text}")
    if trim_info:
        print(
            "Active RPM segment: "
            f"{trim_info['start']:.3f}s to {trim_info['end']:.3f}s "
            f"({trim_info['kept']}/{trim_info['total']} samples, "
            f"{trim_info['change_count']} changes >= {trim_info['threshold']:.1f} RPM)"
        )
    elif args.no_trim:
        print("Active RPM trimming disabled.")
    else:
        print("No active RPM segment detected; plotting the full log.")
    plot_function = plot_rpms_separate if args.separate_sources else plot_rpms
    plot_function(
        rows,
        sources,
        f"Motor RPMs ({source_text})",
        output,
        args.show,
        reset_time=not args.absolute_time,
    )


def expand_logfiles(paths):
    expanded = []
    for raw_path in paths:
        path = Path(raw_path)
        if path.is_dir():
            expanded.extend(sorted(path.glob("*.data")))
            expanded.extend(sorted(path.glob("*.csv")))
        else:
            expanded.append(path)
    return expanded


def main():
    parser = argparse.ArgumentParser(
        description="Plot 4 separate motor RPM traces from a Paparazzi NN CFC flight log."
    )
    parser.add_argument(
        "logfiles",
        nargs="+",
        help="logger_file CSV, Paparazzi .data file, or a directory like var/logs",
    )
    parser.add_argument(
        "--source",
        action="append",
        choices=["auto", "cmd", "obs", "ref", "nn-input"],
        default=None,
        help=(
            "RPM source to plot. Use multiple --source options to overlay, "
            "for example --source obs --source cmd."
        ),
    )
    parser.add_argument(
        "--output",
        help="PNG output path. Defaults to plots/plots_paparazzi/<log_name>_active_motor_rpms.png",
    )
    parser.add_argument(
        "--no-trim",
        action="store_true",
        help="plot the full log instead of only the active RPM segment",
    )
    parser.add_argument(
        "--trim-threshold-rpm",
        type=float,
        default=DEFAULT_TRIM_THRESHOLD_RPM,
        help=f"minimum motor-to-motor sample change used to detect activity; default {DEFAULT_TRIM_THRESHOLD_RPM}",
    )
    parser.add_argument(
        "--trim-pad-s",
        type=float,
        default=DEFAULT_TRIM_PAD_S,
        help=f"seconds kept before first and after last detected RPM change; default {DEFAULT_TRIM_PAD_S}",
    )
    parser.add_argument(
        "--absolute-time",
        action="store_true",
        help="keep original log timestamps on the x-axis instead of resetting the active segment to 0",
    )
    parser.add_argument(
        "--separate-sources",
        action="store_true",
        help="use one subplot per motor and source instead of overlaying sources",
    )
    parser.add_argument("--show", action="store_true", help="open an interactive matplotlib window")
    args = parser.parse_args()

    logfiles = expand_logfiles(args.logfiles)
    if not logfiles:
        raise SystemExit("No log files found")
    missing = [path for path in logfiles if not path.exists()]
    if missing:
        raise SystemExit(f"File not found: {missing[0]}")
    if args.output and len(logfiles) > 1:
        print("Ignoring --output because multiple logs are being plotted.")
    for path in logfiles:
        analyze_one(path, args)


if __name__ == "__main__":
    main()
