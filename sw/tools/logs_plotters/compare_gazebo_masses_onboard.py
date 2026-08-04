#!/usr/bin/env python3
"""Compare three Gazebo mass configurations with an onboard RL flight."""

from __future__ import annotations

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

from compare_gazebo_onboard_rpm import (
    active_segment,
    feedback_rpm,
    read_csv,
    waypoint_transition_times,
)


STATE_ROWS = [
    (("pos_x", "pos_y", "pos_z"), ("x", "y", "z"), "posição [m]"),
    (("vel_x", "vel_y", "vel_z"), ("vx", "vy", "vz"), "velocidade [m/s]"),
    (("att_phi", "att_theta", "att_psi"), ("φ", "θ", "ψ"), "atitude [rad]"),
    (("rate_p", "rate_q", "rate_r"), ("p", "q", "r"), "taxa angular [rad/s]"),
]
STATE_COLUMNS = tuple(column for columns, _names, _unit in STATE_ROWS for column in columns)
SERIES_KEYS = ("nominal", "600g", "650g", "real")
COLORS = {
    "nominal": "#7570b3",
    "600g": "#2166ac",
    "650g": "#1b9e77",
    "real": "#d95f02",
}
LABELS = {
    "nominal": "Gazebo nominal: 503 g no chassis (~510 g total)",
    "600g": "Gazebo 600 g no chassis (~607 g total)",
    "650g": "Gazebo 650 g no chassis (~657 g total)",
    "real": "voo real",
}


def wrap_pi(value):
    return (value + np.pi) % (2.0 * np.pi) - np.pi


def prepare(path):
    data = read_csv(path)
    active = active_segment(data)
    t0 = data["time"][active[0]]
    time = data["time"] - t0
    transitions = waypoint_transition_times(data, active)
    return {
        "path": path,
        "data": data,
        "active": active,
        "time": time,
        "transitions": transitions,
    }


def mask_for(item, end):
    return (
        (item["data"]["rl_enabled"] > 0.5)
        & np.isfinite(item["time"])
        & (item["time"] >= 0.0)
        & (item["time"] <= end)
    )


def transition_time(item, number):
    if len(item["transitions"]) < number:
        return None
    return float(item["transitions"][number - 1][1])


def mark_transitions(axis, items, maximum_time):
    styles = {"nominal": (0, (5, 2)), "600g": "--", "650g": ":", "real": "-."}
    for key, item in items.items():
        for number, (_waypoint, time) in enumerate(item["transitions"][:2], start=2):
            if time > maximum_time + 1e-9:
                continue
            axis.axvline(
                time,
                color=COLORS[key],
                ls=styles[key],
                lw=0.9,
                alpha=0.65,
            )


def transition_note(items):
    parts = []
    for key in SERIES_KEYS:
        times = [transition[1] for transition in items[key]["transitions"][:2]]
        if not times:
            parts.append(f"{key}: nenhum gate confirmado")
        elif len(times) == 1:
            parts.append(f"{key}: gate 2={times[0]:.3f}s; sem gate 3")
        else:
            parts.append(f"{key}: gate 2={times[0]:.3f}s; gate 3={times[1]:.3f}s")
    return " | ".join(parts[:2]) + "\n" + " | ".join(parts[2:])


def plot_states(items, ends, output, title):
    maximum_time = max(ends.values())
    fig, axes = plt.subplots(4, 3, figsize=(18, 14), sharex=True)
    for row, (columns, names, unit) in enumerate(STATE_ROWS):
        for column_index, (column, name) in enumerate(zip(columns, names)):
            axis = axes[row, column_index]
            for key in SERIES_KEYS:
                item = items[key]
                mask = mask_for(item, ends[key])
                axis.plot(
                    item["time"][mask],
                    item["data"][column][mask],
                    color=COLORS[key],
                    lw=1.25 if key != "real" else 1.4,
                    alpha=0.9,
                    label=LABELS[key],
                )
            mark_transitions(axis, items, maximum_time)
            axis.set_title(name)
            if column_index == 0:
                axis.set_ylabel(unit)
            if row == len(STATE_ROWS) - 1:
                axis.set_xlabel("tempo desde a ativação da RL [s]")
            axis.grid(True, alpha=0.25)
            axis.set_xlim(0.0, maximum_time)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.865), ncol=4)
    fig.suptitle(
        f"{title}\nRL 83,2M | filtro de taxas 10 Hz ATIVO\n{transition_note(items)}",
        fontsize=14,
        y=0.995,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.82))
    fig.savefig(output, dpi=170)
    plt.close(fig)


def plot_rpm(items, ends, output, title):
    maximum_time = max(ends.values())
    fig, axes = plt.subplots(4, 1, figsize=(15, 12), sharex=True)
    for motor, axis in enumerate(axes, start=1):
        for key in SERIES_KEYS:
            item = items[key]
            mask = mask_for(item, ends[key])
            axis.plot(
                item["time"][mask],
                feedback_rpm(item["data"], motor)[mask],
                color=COLORS[key],
                lw=1.25 if key != "real" else 1.4,
                alpha=0.9,
                label=LABELS[key],
            )
        mark_transitions(axis, items, maximum_time)
        axis.set_ylabel(f"M{motor} [RPM]")
        axis.set_ylim(4800, 10200)
        axis.set_xlim(0.0, maximum_time)
        axis.grid(True, alpha=0.25)
        axis.legend(loc="lower right", fontsize=8)
    axes[-1].set_xlabel("tempo desde a ativação da RL [s]")
    fig.suptitle(
        f"{title}\nRL 83,2M | filtro de taxas 10 Hz ATIVO\n{transition_note(items)}",
        fontsize=14,
        y=0.995,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.82))
    fig.savefig(output, dpi=170)
    plt.close(fig)


def plot_trajectory(items, ends, output):
    fig, axis = plt.subplots(figsize=(10, 8), constrained_layout=True)
    for key in SERIES_KEYS:
        item = items[key]
        mask = mask_for(item, ends[key])
        axis.plot(
            item["data"]["pos_x"][mask],
            item["data"]["pos_y"][mask],
            color=COLORS[key],
            lw=1.5,
            label=LABELS[key],
        )
    axis.set_xlabel("x ENU [m]")
    axis.set_ylabel("y ENU [m]")
    axis.set_aspect("equal", adjustable="datalim")
    axis.grid(True, alpha=0.25)
    axis.legend()
    axis.set_title(
        "Trajetória XY — massa nominal/600 g/650 g no Gazebo × voo real\n"
        "até o instante do gate 3 na simulação de 600 g"
    )
    fig.savefig(output, dpi=170)
    plt.close(fig)


def interpolated_rmse(simulation, real, end, align_initial=False):
    real_mask = mask_for(real, end)
    real_time = real["time"][real_mask]
    result = {}
    for column in STATE_COLUMNS:
        sim_mask = mask_for(simulation, end)
        sim_time = simulation["time"][sim_mask]
        sim_value = simulation["data"][column][sim_mask]
        real_value = real["data"][column][real_mask]
        finite_sim = np.isfinite(sim_time) & np.isfinite(sim_value)
        finite_real = np.isfinite(real_time) & np.isfinite(real_value)
        if np.count_nonzero(finite_sim) < 2 or not np.any(finite_real):
            result[column] = None
            continue
        expected = np.interp(real_time[finite_real], sim_time[finite_sim], sim_value[finite_sim])
        measured = real_value[finite_real].copy()
        if align_initial:
            expected -= expected[0]
            measured -= measured[0]
        error = expected - measured
        if column == "att_psi":
            error = wrap_pi(error)
        result[column] = float(np.sqrt(np.mean(error**2)))
    return result


def state_at(item, time):
    mask = mask_for(item, time)
    sample_time = item["time"][mask]
    result = {}
    for column in STATE_COLUMNS:
        values = item["data"][column][mask]
        finite = np.isfinite(sample_time) & np.isfinite(values)
        result[column] = (
            float(np.interp(time, sample_time[finite], values[finite]))
            if np.count_nonzero(finite) >= 2
            else None
        )
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("gazebo_nominal", type=Path)
    parser.add_argument("gazebo_600g", type=Path)
    parser.add_argument("gazebo_650g", type=Path)
    parser.add_argument("onboard", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    items = {
        "nominal": prepare(args.gazebo_nominal),
        "600g": prepare(args.gazebo_600g),
        "650g": prepare(args.gazebo_650g),
        "real": prepare(args.onboard),
    }
    gate2 = {key: transition_time(item, 1) for key, item in items.items()}
    if any(value is None for value in gate2.values()):
        raise ValueError("All four logs must contain a confirmed gate 2")
    gate3_600 = transition_time(items["600g"], 2)
    if gate3_600 is None:
        raise ValueError("The 600 g simulation must contain a confirmed gate 3")

    gate2_ends = {key: float(value) for key, value in gate2.items()}
    gate3_ends = {key: float(gate3_600) for key in items}
    plot_states(
        items,
        gate2_ends,
        args.output_dir / "all_states_nominal_600g_650g_real_until_each_gate2.png",
        "Todos os estados — cada caso até sua passagem pelo gate 2",
    )
    plot_rpm(
        items,
        gate2_ends,
        args.output_dir / "rpm_feedback_nominal_600g_650g_real_until_each_gate2.png",
        "RPM de feedback — cada caso até sua passagem pelo gate 2",
    )
    plot_states(
        items,
        gate3_ends,
        args.output_dir / "all_states_nominal_600g_650g_real_until_600g_gate3.png",
        "Todos os estados — até a simulação de 600 g cruzar o gate 3",
    )
    plot_rpm(
        items,
        gate3_ends,
        args.output_dir / "rpm_feedback_nominal_600g_650g_real_until_600g_gate3.png",
        "RPM de feedback — até a simulação de 600 g cruzar o gate 3",
    )
    plot_trajectory(
        items,
        gate3_ends,
        args.output_dir / "trajectory_xy_nominal_600g_650g_real_until_600g_gate3.png",
    )

    common_gate2_end = min(gate2_ends.values())
    metrics = {
        "logs": {key: str(item["path"]) for key, item in items.items()},
        "mass_note": {
            "nominal": "0.503 kg chassis + approximately 0.007 kg fixed child links",
            "600g": "0.600 kg chassis + approximately 0.007 kg fixed child links",
            "650g": "0.650 kg chassis + approximately 0.007 kg fixed child links",
        },
        "controller": "recurrent_ppo_figure8_gates_83200000_steps.zip",
        "rate_filter": "first-order 10 Hz active",
        "transition_times_s": {
            key: [float(time) for _waypoint, time in item["transitions"][:2]]
            for key, item in items.items()
        },
        "rmse_vs_real_common_until_earliest_gate2_s": common_gate2_end,
        "rmse_nominal_vs_real_until_earliest_gate2": interpolated_rmse(
            items["nominal"], items["real"], common_gate2_end
        ),
        "rmse_600g_vs_real_until_earliest_gate2": interpolated_rmse(
            items["600g"], items["real"], common_gate2_end
        ),
        "rmse_650g_vs_real_until_earliest_gate2": interpolated_rmse(
            items["650g"], items["real"], common_gate2_end
        ),
        "delta_rmse_nominal_vs_real_until_earliest_gate2": interpolated_rmse(
            items["nominal"], items["real"], common_gate2_end, align_initial=True
        ),
        "delta_rmse_600g_vs_real_until_earliest_gate2": interpolated_rmse(
            items["600g"], items["real"], common_gate2_end, align_initial=True
        ),
        "delta_rmse_650g_vs_real_until_earliest_gate2": interpolated_rmse(
            items["650g"], items["real"], common_gate2_end, align_initial=True
        ),
        "rmse_nominal_vs_real_until_600g_gate3": interpolated_rmse(
            items["nominal"], items["real"], gate3_600
        ),
        "rmse_600g_vs_real_until_600g_gate3": interpolated_rmse(
            items["600g"], items["real"], gate3_600
        ),
        "rmse_650g_vs_real_until_600g_gate3": interpolated_rmse(
            items["650g"], items["real"], gate3_600
        ),
        "state_at_own_gate2": {
            key: state_at(item, gate2_ends[key])
            for key, item in items.items()
        },
    }
    (args.output_dir / "comparison_metrics_with_nominal.json").write_text(
        json.dumps(metrics, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metrics, indent=2))
    print(f"Saved comparisons to {args.output_dir}")


if __name__ == "__main__":
    main()
