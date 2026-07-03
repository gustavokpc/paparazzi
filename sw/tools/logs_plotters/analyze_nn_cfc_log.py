#!/usr/bin/env python3
"""Analyze NN CFC controller logs from logger_file CSVs or Paparazzi .data logs."""

import argparse
import csv
import math
import re
import statistics
import sys
from collections import Counter
from pathlib import Path


NN_INPUT_NAMES = [
    "dx", "dy", "dz",
    "vx", "vy", "vz",
    "phi", "theta", "psi",
    "p", "q", "r",
    "Mx_ext", "My_ext", "Mz_ext",
    "omega1", "omega2", "omega3", "omega4",
]

CSV_NETWORK_FIELDS = (
    ["time"]
    + [f"nn_in_{name}_raw" for name in NN_INPUT_NAMES]
    + [f"nn_in_{name}_normalized" for name in NN_INPUT_NAMES]
    + [f"network_out{i}_norm" for i in range(1, 5)]
    + [f"rpm_cmd{i}" for i in range(1, 5)]
    + ["nn_periodic_dt_us", "nn_inference_time_us"]
    + ["cmd_thrust", "cmd_roll", "cmd_pitch", "cmd_yaw"]
)

DATA_NETWORK_FIELDS = (
    ["time", "aircraft_id"]
    + [f"nn_in_{name}_raw" for name in NN_INPUT_NAMES]
    + [f"nn_in_{name}_normalized" for name in NN_INPUT_NAMES]
)


def parse_float(value):
    if value is None or value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def parse_vector(text):
    return [
        float(item)
        for item in text.split(",")
        if item
    ]


def fmt_num(value, digits=3):
    if value is None:
        return "n/a"
    if isinstance(value, int):
        return str(value)
    if math.isnan(value):
        return "n/a"
    return f"{value:.{digits}f}"


def fmt_hz(value):
    return f"{fmt_num(value, 2)} Hz"


def rate_stats(times):
    if len(times) < 2:
        return {}
    dts = [
        b - a
        for a, b in zip(times, times[1:])
        if b > a
    ]
    if not dts:
        return {}
    return {
        "samples": len(times),
        "duration": times[-1] - times[0],
        "mean_dt": statistics.fmean(dts),
        "median_dt": statistics.median(dts),
        "min_dt": min(dts),
        "max_dt": max(dts),
        "mean_hz": 1.0 / statistics.fmean(dts),
        "median_hz": 1.0 / statistics.median(dts),
        "gaps_over_50ms": sum(1 for dt in dts if dt > 0.05),
        "gaps_over_100ms": sum(1 for dt in dts if dt > 0.1),
    }


def count_changes(rows, fields):
    previous = None
    changes = 0
    first_change_time = None
    last_change_time = None
    for row in rows:
        current = tuple(row.get(field) for field in fields)
        if previous is not None and current != previous:
            changes += 1
            if first_change_time is None:
                first_change_time = row.get("time")
            last_change_time = row.get("time")
        previous = current
    return changes, first_change_time, last_change_time


def numeric_summary(rows, fields):
    summary = {}
    for field in fields:
        values = [parse_float(row.get(field)) for row in rows]
        values = [value for value in values if value is not None]
        if values:
            summary[field] = {
                "min": min(values),
                "mean": statistics.fmean(values),
                "max": max(values),
            }
    return summary


def read_csv_log(path):
    rows = []
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        for line_no, row in enumerate(reader, start=2):
            parsed = {"source_line": line_no}
            for key, value in row.items():
                parsed[key] = value
            time_value = parse_float(row.get("time"))
            if time_value is None:
                continue
            parsed["time"] = time_value
            rows.append(parsed)
    return rows


def read_data_log(path):
    rows = []
    timing_rows = []
    message_counts = Counter()
    aircraft_counts = Counter()
    debug_names = Counter()
    line_re = re.compile(r"^(\S+)\s+(\S+)\s+(\S+)(?:\s+(.*))?$")
    with path.open(errors="replace") as stream:
        for line_no, line in enumerate(stream, start=1):
            match = line_re.match(line.strip())
            if not match:
                continue
            time_text, aircraft_id, message, payload = match.groups()
            time_value = parse_float(time_text)
            if time_value is None:
                continue
            message_counts[message] += 1
            aircraft_counts[aircraft_id] += 1
            if message != "DEBUG_VECT" or payload is None:
                continue
            parts = payload.split(None, 1)
            if len(parts) != 2:
                continue
            name, vector_text = parts
            debug_names[name] += 1
            if name != "NN_CFC_INPUTS":
                if name == "NN_CFC_TIMING":
                    vector = parse_vector(vector_text)
                    if len(vector) >= 2:
                        timing_rows.append({
                            "source_line": line_no,
                            "time": time_value,
                            "aircraft_id": aircraft_id,
                            "nn_periodic_dt_us": vector[0],
                            "nn_inference_time_us": vector[1],
                        })
                continue
            vector = parse_vector(vector_text)
            if len(vector) < 38:
                continue
            row = {
                "source_line": line_no,
                "time": time_value,
                "aircraft_id": aircraft_id,
            }
            for idx, name in enumerate(NN_INPUT_NAMES):
                row[f"nn_in_{name}_raw"] = vector[idx]
                row[f"nn_in_{name}_normalized"] = vector[19 + idx]
            rows.append(row)
    meta = {
        "message_counts": message_counts,
        "aircraft_counts": aircraft_counts,
        "debug_names": debug_names,
        "timing_rows": timing_rows,
    }
    return rows, meta


def write_network_csv(rows, fields, output_path):
    with output_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def print_rate_block(title, stats):
    print(f"\n{title}")
    if not stats:
        print("  Not enough samples.")
        return
    print(f"  samples: {stats['samples']}")
    print(f"  duration: {fmt_num(stats['duration'])} s")
    print(f"  mean rate: {fmt_hz(stats['mean_hz'])}")
    print(f"  median rate: {fmt_hz(stats['median_hz'])}")
    print(
        "  dt mean/median/min/max: "
        f"{fmt_num(stats['mean_dt'] * 1000)} / "
        f"{fmt_num(stats['median_dt'] * 1000)} / "
        f"{fmt_num(stats['min_dt'] * 1000)} / "
        f"{fmt_num(stats['max_dt'] * 1000)} ms"
    )
    print(f"  gaps > 50 ms: {stats['gaps_over_50ms']}")
    print(f"  gaps > 100 ms: {stats['gaps_over_100ms']}")


def print_summary_table(title, summary, fields):
    print(f"\n{title}")
    available = [field for field in fields if field in summary]
    if not available:
        print("  n/a")
        return
    for field in available:
        stats = summary[field]
        print(
            f"  {field}: min={fmt_num(stats['min'])} "
            f"mean={fmt_num(stats['mean'])} max={fmt_num(stats['max'])}"
        )


def print_rows(rows, fields, limit):
    if limit <= 0:
        return
    print(f"\nFirst {min(limit, len(rows))} network rows")
    writer = csv.DictWriter(sys.stdout, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    for row in rows[:limit]:
        writer.writerow(row)


def analyze_csv(path, args):
    rows = read_csv_log(path)
    if not rows:
        raise SystemExit(f"No usable rows found in {path}")

    times = [row["time"] for row in rows]
    input_fields = [f"nn_in_{name}_raw" for name in NN_INPUT_NAMES]
    norm_fields = [f"nn_in_{name}_normalized" for name in NN_INPUT_NAMES]
    output_fields = [f"network_out{i}_norm" for i in range(1, 5)]
    rpm_fields = [f"rpm_cmd{i}" for i in range(1, 5)]
    command_fields = ["cmd_thrust", "cmd_roll", "cmd_pitch", "cmd_yaw"]
    timing_fields = ["nn_periodic_dt_us", "nn_inference_time_us"]

    print(f"Source: {path}")
    print("Format: logger_file CSV")
    print("Nominal controller/logger frequency from airframe: 100 Hz")
    print_rate_block("Observed CSV sample rate", rate_stats(times))

    input_changes, first_input, last_input = count_changes(rows, input_fields + norm_fields)
    output_changes, first_output, last_output = count_changes(rows, output_fields + rpm_fields)
    print("\nNetwork update/change counters")
    print(f"  rows logged: {len(rows)}")
    print(f"  input vector changes: {input_changes}")
    print(f"  first/last input change: {fmt_num(first_input)} s / {fmt_num(last_input)} s")
    print(f"  output/rpm changes: {output_changes}")
    print(f"  first/last output change: {fmt_num(first_output)} s / {fmt_num(last_output)} s")

    summary = numeric_summary(rows, input_fields + norm_fields + output_fields + rpm_fields + command_fields + timing_fields)
    print_summary_table("Raw NN inputs", summary, input_fields)
    print_summary_table("Normalized NN inputs", summary, norm_fields)
    print_summary_table("Network outputs and RPM commands", summary, output_fields + rpm_fields)
    print_summary_table("Autopilot commands", summary, command_fields)
    print_summary_table("NN timing", summary, timing_fields)

    if args.export_csv:
        write_network_csv(rows, CSV_NETWORK_FIELDS, Path(args.export_csv))
        print(f"\nExported network CSV: {args.export_csv}")
    print_rows(rows, CSV_NETWORK_FIELDS, args.head)


def analyze_data(path, args):
    rows, meta = read_data_log(path)
    if not rows:
        raise SystemExit(f"No NN_CFC_INPUTS DEBUG_VECT rows found in {path}")

    times = [row["time"] for row in rows]
    unique_times = []
    for time_value in times:
        if not unique_times or time_value != unique_times[-1]:
            unique_times.append(time_value)

    input_fields = [f"nn_in_{name}_raw" for name in NN_INPUT_NAMES]
    norm_fields = [f"nn_in_{name}_normalized" for name in NN_INPUT_NAMES]
    input_changes, first_input, last_input = count_changes(rows, input_fields + norm_fields)

    print(f"Source: {path}")
    print("Format: Paparazzi .data text log")
    print("Nominal nn_cfc_control_periodic frequency from module XML: 100 Hz")
    print("Note: this log contains telemetered DEBUG_VECT inputs; network outputs are not in DEBUG_VECT.")
    print_rate_block("Observed NN_CFC_INPUTS message receive rate", rate_stats(times))
    print_rate_block("Observed NN_CFC_INPUTS unique-timestamp rate", rate_stats(unique_times))

    print("\nNetwork update/change counters")
    print(f"  NN_CFC_INPUTS messages: {len(rows)}")
    print(f"  unique timestamps: {len(unique_times)}")
    print(f"  input vector changes: {input_changes}")
    print(f"  first/last input change: {fmt_num(first_input)} s / {fmt_num(last_input)} s")
    print("  network output/rpm fields: not present in DEBUG_VECT; use logger_file CSV for those.")

    print("\nTop Paparazzi messages")
    for name, count in meta["message_counts"].most_common(12):
        print(f"  {name}: {count}")
    print("\nAircraft IDs")
    for aircraft_id, count in meta["aircraft_counts"].most_common():
        print(f"  {aircraft_id}: {count}")
    print("\nDEBUG_VECT names")
    for name, count in meta["debug_names"].most_common():
        print(f"  {name}: {count}")

    timing_rows = meta["timing_rows"]
    if timing_rows:
        timing_times = [row["time"] for row in timing_rows]
        print_rate_block("Observed NN_CFC_TIMING message receive rate", rate_stats(timing_times))
        timing_summary = numeric_summary(timing_rows, ["nn_periodic_dt_us", "nn_inference_time_us"])
        print_summary_table("NN timing", timing_summary, ["nn_periodic_dt_us", "nn_inference_time_us"])
    else:
        print("\nNN timing")
        print("  Not present in this log. Rebuild/flash with the new NN_CFC_TIMING telemetry to record it.")

    summary = numeric_summary(rows, input_fields + norm_fields)
    print_summary_table("Raw NN inputs", summary, input_fields)
    print_summary_table("Normalized NN inputs", summary, norm_fields)

    if args.export_csv:
        write_network_csv(rows, DATA_NETWORK_FIELDS, Path(args.export_csv))
        print(f"\nExported network input CSV: {args.export_csv}")
    print_rows(rows, DATA_NETWORK_FIELDS, args.head)


def main():
    parser = argparse.ArgumentParser(
        description="Print relevant NN CFC logger information: inputs, outputs, updates, and observed Hz."
    )
    parser.add_argument("logfile", help="logger_file CSV or Paparazzi .data file")
    parser.add_argument("--export-csv", help="write extracted network rows to this CSV")
    parser.add_argument("--head", type=int, default=0, help="print the first N extracted rows")
    args = parser.parse_args()

    path = Path(args.logfile)
    if not path.exists():
        raise SystemExit(f"File not found: {path}")
    if path.suffix.lower() == ".csv":
        analyze_csv(path, args)
    else:
        analyze_data(path, args)


if __name__ == "__main__":
    main()
