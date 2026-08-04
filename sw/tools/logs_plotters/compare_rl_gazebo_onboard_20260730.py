#!/usr/bin/env python3
"""Compare the July 30 RL Gazebo runs with their matching onboard flights."""

from __future__ import annotations

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


STATE_ROWS = [
    (("pos_x", "pos_y", "pos_z"), ("x", "y", "z"), "posição [m]"),
    (("vel_x", "vel_y", "vel_z"), ("vx", "vy", "vz"), "velocidade [m/s]"),
    (("att_phi", "att_theta", "att_psi"), ("φ", "θ", "ψ"), "atitude [rad]"),
    (("rate_p", "rate_q", "rate_r"), ("p", "q", "r"), "taxa angular [rad/s]"),
]


def parse_float(value: str | None) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def read_csv(path: Path) -> dict[str, np.ndarray]:
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        columns = reader.fieldnames or []
        rows = list(reader)
    return {
        column: np.asarray(
            [parse_float(row.get(column)) for row in rows], dtype=float
        )
        for column in columns
    }


def aligned_active_data(
    data: dict[str, np.ndarray], duration_s: float
) -> tuple[np.ndarray, np.ndarray]:
    active = (data["rl_enabled"] > 0.5) & np.isfinite(data["time"])
    indices = np.flatnonzero(active)
    if indices.size == 0:
        raise ValueError("log sem amostras com RL ativa")
    aligned_time = data["time"] - data["time"][indices[0]]
    mask = active & (aligned_time >= 0.0) & (aligned_time <= duration_s)
    return aligned_time, mask


def feedback_rpm(data: dict[str, np.ndarray], motor: int) -> np.ndarray:
    observed = f"rpm_obs_{motor}"
    if observed in data:
        return data[observed]
    motor_state = f"rl_motor_state{motor}"
    if motor_state not in data:
        raise ValueError(f"sem {observed} ou {motor_state}")
    return 5000.0 + 2500.0 * (data[motor_state] + 1.0)


def add_end_marker(axis, time: np.ndarray, mask: np.ndarray, color: str) -> None:
    valid_time = time[mask & np.isfinite(time)]
    if valid_time.size:
        axis.axvline(
            valid_time[-1],
            color=color,
            lw=1.0,
            ls=":",
            alpha=0.65,
            label="fim do log Gazebo",
        )


def plot_states(
    gazebo: dict[str, np.ndarray],
    onboard: dict[str, np.ndarray],
    output: Path,
    title: str,
    duration_s: float,
) -> None:
    gazebo_time, gazebo_mask = aligned_active_data(gazebo, duration_s)
    onboard_time, onboard_mask = aligned_active_data(onboard, duration_s)
    fig, axes = plt.subplots(
        4, 3, figsize=(18, 14), sharex=True, constrained_layout=True
    )
    for row, (columns, names, unit) in enumerate(STATE_ROWS):
        for col, (column, name) in enumerate(zip(columns, names)):
            axis = axes[row, col]
            axis.plot(
                gazebo_time[gazebo_mask],
                gazebo[column][gazebo_mask],
                color="#2166ac",
                lw=1.4,
                label="Gazebo",
            )
            axis.plot(
                onboard_time[onboard_mask],
                onboard[column][onboard_mask],
                color="#d95f02",
                lw=1.15,
                alpha=0.9,
                label="voo real",
            )
            add_end_marker(axis, gazebo_time, gazebo_mask, "#2166ac")
            axis.set_title(name)
            axis.set_xlim(0.0, duration_s)
            axis.grid(True, alpha=0.25)
            if col == 0:
                axis.set_ylabel(unit)
            if row == 3:
                axis.set_xlabel("tempo desde a ativação da RL [s]")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    fig.legend(
        unique.values(), unique.keys(), loc="upper right", bbox_to_anchor=(0.985, 0.985)
    )
    fig.suptitle(
        f"Todos os estados — Gazebo × voo real — 0–{duration_s:g} s\n{title}",
        fontsize=15,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=170)
    plt.close(fig)


def plot_rpm(
    gazebo: dict[str, np.ndarray],
    onboard: dict[str, np.ndarray],
    output: Path,
    title: str,
    duration_s: float,
) -> None:
    gazebo_time, gazebo_mask = aligned_active_data(gazebo, duration_s)
    onboard_time, onboard_mask = aligned_active_data(onboard, duration_s)
    fig, axes = plt.subplots(
        4, 1, figsize=(15, 12), sharex=True, constrained_layout=True
    )
    for motor, axis in enumerate(axes, start=1):
        gazebo_rpm = feedback_rpm(gazebo, motor)
        onboard_rpm = feedback_rpm(onboard, motor)
        axis.plot(
            gazebo_time[gazebo_mask],
            gazebo_rpm[gazebo_mask],
            color="#2166ac",
            lw=1.4,
            label="Gazebo — RPM do estado do motor",
        )
        axis.plot(
            onboard_time[onboard_mask],
            onboard_rpm[onboard_mask],
            color="#d95f02",
            lw=1.1,
            alpha=0.9,
            label="voo real — RPM observado",
        )
        add_end_marker(axis, gazebo_time, gazebo_mask, "#2166ac")
        axis.set_ylabel(f"M{motor} [RPM]")
        axis.set_ylim(4800.0, 10200.0)
        axis.set_xlim(0.0, duration_s)
        axis.grid(True, alpha=0.25)
        axis.legend(loc="lower right", fontsize=8)

    axes[-1].set_xlabel("tempo desde a ativação da RL [s]")
    fig.suptitle(
        f"RPM dos motores — Gazebo × voo real — 0–{duration_s:g} s\n{title}",
        fontsize=14,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=170)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("gazebo_csv", type=Path)
    parser.add_argument("onboard_csv", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--duration", type=float, default=50.0)
    parser.add_argument("--label", required=True)
    args = parser.parse_args()

    gazebo = read_csv(args.gazebo_csv)
    onboard = read_csv(args.onboard_csv)
    stem = f"{args.gazebo_csv.stem}_gazebo_vs_{args.onboard_csv.stem}_onboard"
    states_path = args.output_dir / f"{stem}_all_states_until_{args.duration:g}s.png"
    rpm_path = args.output_dir / f"{stem}_rpm_feedback_until_{args.duration:g}s.png"
    plot_states(gazebo, onboard, states_path, args.label, args.duration)
    plot_rpm(gazebo, onboard, rpm_path, args.label, args.duration)
    print(states_path)
    print(rpm_path)


if __name__ == "__main__":
    main()
