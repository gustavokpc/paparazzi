#!/usr/bin/env python3
"""Compare RL motor RPM commands from Gazebo and an onboard log."""

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


def parse_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def read_csv(path):
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        columns = reader.fieldnames or []
        rows = list(reader)
    return {
        column: np.asarray([parse_float(row.get(column)) for row in rows], dtype=float)
        for column in columns
    }


def active_segment(data):
    active = (data["rl_enabled"] > 0.5) & np.isfinite(data["time"])
    indices = np.flatnonzero(active)
    if indices.size == 0:
        raise ValueError("log has no RL-active samples")
    return indices


def waypoint_transition_times(data, indices):
    times = data["time"]
    waypoint = data["rl_waypoint_index"]
    transitions = []
    valid_indices = [
        index for index in indices
        if np.isfinite(times[index]) and np.isfinite(waypoint[index])
    ]
    if not valid_indices:
        return transitions
    previous = int(waypoint[valid_indices[0]])
    for index in valid_indices[1:]:
        current = int(waypoint[index])
        if current != previous:
            transitions.append((current, float(times[index] - times[indices[0]])))
            previous = current
    return transitions


def feedback_rpm(data, motor):
    observed = f"rpm_obs_{motor}"
    if observed in data:
        return data[observed]
    motor_state = f"rl_motor_state{motor}"
    if motor_state not in data:
        raise ValueError(f"log has neither {observed} nor {motor_state}")
    # Inverse of rpm_to_motor_state() in rl_cfc_control.c.
    return 5000.0 + 0.5 * (data[motor_state] + 1.0) * (10000.0 - 5000.0)


def plot_comparison(gazebo_path, onboard_path, output_path, label):
    gazebo = read_csv(gazebo_path)
    onboard = read_csv(onboard_path)
    gazebo_active = active_segment(gazebo)
    onboard_active = active_segment(onboard)

    gazebo_transitions = waypoint_transition_times(gazebo, gazebo_active)
    onboard_transitions = waypoint_transition_times(onboard, onboard_active)
    if len(gazebo_transitions) < 2:
        raise ValueError("Gazebo log does not contain the transition through gate 3")

    # The controller starts targeting waypoint index 2. In the existing gate
    # convention, the first two changes are crossings of gates 2 and 3.
    gate2_time = gazebo_transitions[0][1]
    gate3_time = gazebo_transitions[1][1]
    gazebo_t0 = gazebo["time"][gazebo_active[0]]
    onboard_t0 = onboard["time"][onboard_active[0]]
    gazebo_time = gazebo["time"] - gazebo_t0
    onboard_time = onboard["time"] - onboard_t0
    gazebo_mask = (gazebo["rl_enabled"] > 0.5) & (gazebo_time >= 0) & (gazebo_time <= gate3_time)
    onboard_mask = (onboard["rl_enabled"] > 0.5) & (onboard_time >= 0) & (onboard_time <= gate3_time)

    fig, axes = plt.subplots(4, 1, figsize=(15, 12), sharex=True, constrained_layout=True)
    for motor, axis in enumerate(axes, start=1):
        gazebo_rpm = feedback_rpm(gazebo, motor)
        onboard_rpm = feedback_rpm(onboard, motor)
        axis.plot(
            gazebo_time[gazebo_mask], gazebo_rpm[gazebo_mask],
            color="#2166ac", lw=1.5, label=f"Gazebo estimado ({gazebo_path.stem})",
        )
        axis.plot(
            onboard_time[onboard_mask], onboard_rpm[onboard_mask],
            color="#d95f02", lw=1.2, alpha=0.9, label=f"onboard medido ({onboard_path.stem})",
        )
        axis.axvline(gate2_time, color="#666666", ls=":", lw=1.2)
        axis.axvline(gate3_time, color="#111111", ls="--", lw=1.3)
        axis.set_ylabel(f"M{motor} [RPM]")
        axis.set_ylim(4800, 10200)
        axis.grid(True, alpha=0.25)
        axis.legend(loc="lower right", fontsize=8)

    axes[-1].set_xlabel("tempo desde a ativação da RL [s]")
    onboard_gate_note = (
        f"onboard: {len(onboard_transitions)} transição(ões), sem gate 3 confirmado"
        if len(onboard_transitions) < 2
        else f"gate 3 onboard: {onboard_transitions[1][1]:.3f} s"
    )
    fig.suptitle(
        "RPM de feedback dos quatro motores — Gazebo × voo real — até o gate 3\n"
        f"{label}\n"
        f"Gazebo gate 2: {gate2_time:.3f} s | Gazebo gate 3: {gate3_time:.3f} s | {onboard_gate_note}",
        fontsize=14,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=170)
    plt.close(fig)
    return gate2_time, gate3_time, len(onboard_transitions)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("gazebo_csv", type=Path)
    parser.add_argument("onboard_csv", type=Path)
    parser.add_argument("output_png", type=Path)
    parser.add_argument("--label", default="RL")
    args = parser.parse_args()
    gate2, gate3, onboard_transitions = plot_comparison(
        args.gazebo_csv, args.onboard_csv, args.output_png, args.label
    )
    print(f"Saved {args.output_png}")
    print(f"Gazebo gate 2: {gate2:.6f} s")
    print(f"Gazebo gate 3: {gate3:.6f} s")
    print(f"Onboard waypoint transitions: {onboard_transitions}")


if __name__ == "__main__":
    main()
