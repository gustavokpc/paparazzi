#!/usr/bin/env python3
"""Compare all logged vehicle states between Gazebo and an onboard RL flight."""

import argparse
import os
from pathlib import Path

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
if not os.environ.get("DISPLAY"):
    import matplotlib
    matplotlib.use("Agg")

import matplotlib.pyplot as plt

from compare_gazebo_onboard_rpm import (
    active_segment,
    read_csv,
    waypoint_transition_times,
)


STATE_ROWS = [
    (("pos_x", "pos_y", "pos_z"), ("x", "y", "z"), "posição [m]"),
    (("vel_x", "vel_y", "vel_z"), ("vx", "vy", "vz"), "velocidade [m/s]"),
    (("att_phi", "att_theta", "att_psi"), ("φ", "θ", "ψ"), "atitude [rad]"),
    (("rate_p", "rate_q", "rate_r"), ("p", "q", "r"), "taxa angular [rad/s]"),
]


def plot_state_comparison(gazebo_path, onboard_path, output_path, label):
    gazebo = read_csv(gazebo_path)
    onboard = read_csv(onboard_path)
    gazebo_active = active_segment(gazebo)
    onboard_active = active_segment(onboard)
    gazebo_transitions = waypoint_transition_times(gazebo, gazebo_active)
    onboard_transitions = waypoint_transition_times(onboard, onboard_active)
    if len(gazebo_transitions) < 2:
        raise ValueError("Gazebo log does not contain the transition through gate 3")

    gate2_time = gazebo_transitions[0][1]
    gate3_time = gazebo_transitions[1][1]
    gazebo_time = gazebo["time"] - gazebo["time"][gazebo_active[0]]
    onboard_time = onboard["time"] - onboard["time"][onboard_active[0]]
    gazebo_mask = (
        (gazebo["rl_enabled"] > 0.5)
        & np.isfinite(gazebo_time)
        & (gazebo_time >= 0)
        & (gazebo_time <= gate3_time)
    )
    onboard_mask = (
        (onboard["rl_enabled"] > 0.5)
        & np.isfinite(onboard_time)
        & (onboard_time >= 0)
        & (onboard_time <= gate3_time)
    )

    fig, axes = plt.subplots(4, 3, figsize=(18, 14), sharex=True, constrained_layout=True)
    for row, (columns, names, unit) in enumerate(STATE_ROWS):
        for column_index, (column, name) in enumerate(zip(columns, names)):
            axis = axes[row, column_index]
            axis.plot(
                gazebo_time[gazebo_mask], gazebo[column][gazebo_mask],
                color="#2166ac", lw=1.5, label="Gazebo (esperado)",
            )
            axis.plot(
                onboard_time[onboard_mask], onboard[column][onboard_mask],
                color="#d95f02", lw=1.25, alpha=0.9, label="onboard (obtido)",
            )
            axis.axvline(gate2_time, color="#666666", ls=":", lw=1.1)
            axis.axvline(gate3_time, color="#111111", ls="--", lw=1.2)
            axis.set_title(name)
            if column_index == 0:
                axis.set_ylabel(unit)
            axis.grid(True, alpha=0.25)
            if row == len(STATE_ROWS) - 1:
                axis.set_xlabel("tempo desde a ativação da RL [s]")

    handles, legend_labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="upper right", bbox_to_anchor=(0.985, 0.985))
    onboard_note = (
        f"onboard: {len(onboard_transitions)} transição(ões), sem gate 3 confirmado"
        if len(onboard_transitions) < 2
        else f"gate 3 onboard: {onboard_transitions[1][1]:.3f} s"
    )
    fig.suptitle(
        "Todos os estados — Gazebo esperado × voo real obtido — até o gate 3\n"
        f"{label}\n"
        f"Gazebo gate 2: {gate2_time:.3f} s | Gazebo gate 3: {gate3_time:.3f} s | {onboard_note}",
        fontsize=15,
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
    gate2, gate3, onboard_transitions = plot_state_comparison(
        args.gazebo_csv, args.onboard_csv, args.output_png, args.label
    )
    print(f"Saved {args.output_png}")
    print(f"Gazebo gate 2: {gate2:.6f} s")
    print(f"Gazebo gate 3: {gate3:.6f} s")
    print(f"Onboard waypoint transitions: {onboard_transitions}")


if __name__ == "__main__":
    main()
