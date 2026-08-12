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
RL_GATE_YAWS = np.asarray([
    0.5 * np.pi, np.pi, 0.5 * np.pi, 0.0,
    -0.5 * np.pi, -np.pi, -0.5 * np.pi, 0.0,
])
RL_GATE_HALF_SIZE_M = 0.75


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


def plot_groups(output_path, title, data, groups, active_intervals, xlim=None, active_label="NN active/inferred"):
    times = data["time"]
    fig, axes = plt.subplots(len(groups), 1, figsize=(16, max(3.0 * len(groups), 5)), sharex=True, constrained_layout=True)
    if len(groups) == 1:
        axes = [axes]
    fig.suptitle(title, fontsize=14)

    for axis, (ylabel, names) in zip(axes, groups):
        shade_intervals(axis, active_intervals)
        axis.grid(True, alpha=0.25)
        axis.set_ylabel(ylabel)
        for index, name in enumerate(available(data, names)):
            axis.plot(times, data[name], lw=0.9, color=PALETTE[index % len(PALETTE)], label=name)
        if xlim is not None:
            axis.set_xlim(*xlim)
        axis.legend(ncol=min(4, max(1, len(names))), fontsize=8, loc="upper right")

    axes[-1].set_xlabel("time [s]")
    handles, labels = axes[0].get_legend_handles_labels()
    handles.append(Patch(facecolor="#fdb863", alpha=0.35, label=active_label))
    labels.append(active_label)
    axes[0].legend(handles=handles, labels=labels, ncol=min(5, len(labels)), fontsize=8, loc="upper right")

    fig.savefig(output_path, dpi=170)
    plt.close(fig)


def plot_rpm_commands_2x2(output_path, title, data, xlim):
    times = data["time"]
    mask = (
        np.isfinite(times)
        & (times >= xlim[0])
        & (times <= xlim[1])
    )
    fig, axes = plt.subplots(
        2, 2, figsize=(14, 9), sharex=True, constrained_layout=True
    )
    for motor, axis in enumerate(axes.flat, start=1):
        column = f"rpm_cmd{motor}"
        axis.plot(
            times[mask],
            data[column][mask],
            color=PALETTE[(motor - 1) % len(PALETTE)],
            lw=1.2,
        )
        axis.set_title(f"Motor {motor} — {column}")
        axis.set_ylabel("RPM command")
        axis.grid(True, alpha=0.25)
    axes[1, 0].set_xlabel("time [s]")
    axes[1, 1].set_xlabel("time [s]")
    fig.suptitle(title, fontsize=14)
    fig.savefig(output_path, dpi=170)
    plt.close(fig)


def plot_rpm_commanded_observed_4x2(output_path, title, data, xlim, command_columns):
    """Plot commanded RPM beside the best available motor feedback signal.

    Onboard logs contain measured ``rpm_obs_*`` values.  Gazebo logs do not
    always expose those columns, but do contain the command after the
    controller's output processing (``rpm_applied*`` or
    ``rl_rpm_applied*``).  Supporting both formats keeps the standard plot set
    usable for real and simulated runs.
    """
    times = data["time"]
    mask = np.isfinite(times) & (times >= xlim[0]) & (times <= xlim[1])
    fig, axes = plt.subplots(
        4, 2, figsize=(18, 14), sharex=True, sharey="row",
        constrained_layout=True,
    )
    for index, command in enumerate(command_columns):
        motor = index + 1
        feedback_candidates = [
            f"rpm_obs_{motor}",
            command.replace("cmd", "applied"),
        ]
        feedback = next(
            (column for column in feedback_candidates if column in data),
            None,
        )
        axes[index, 0].plot(
            times[mask], data[command][mask],
            color=PALETTE[index % len(PALETTE)], lw=1.1,
        )
        if feedback is not None:
            axes[index, 1].plot(
                times[mask], data[feedback][mask],
                color=PALETTE[index % len(PALETTE)], lw=1.1,
            )
        else:
            axes[index, 1].text(
                0.5, 0.5, "feedback unavailable", ha="center", va="center",
                transform=axes[index, 1].transAxes,
            )
        axes[index, 0].set_title(f"Motor {motor} — commanded ({command})")
        feedback_label = feedback if feedback is not None else "unavailable"
        axes[index, 1].set_title(f"Motor {motor} — feedback ({feedback_label})")
        axes[index, 0].set_ylabel(f"Motor {motor} [RPM]")
        for axis in axes[index]:
            axis.grid(True, alpha=0.25)
            axis.set_xlim(*xlim)
    axes[-1, 0].set_xlabel("time [s]")
    axes[-1, 1].set_xlabel("time [s]")
    fig.suptitle(title, fontsize=14)
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


def active_airborne_xlim(data, active, intervals, ground_clearance_m=0.10):
    """Limit a controller-active interval to the part that is physically airborne.

    The onboard position uses the take-off floor as its local vertical reference in
    these logs.  Estimate that reference immediately before the motors start, infer
    which side of it is airborne, and stop at the first return to the floor.  This
    intentionally does not rely only on controller, motor, or ``in_flight`` flags:
    all three can remain asserted after a crash or ground contact.
    """
    active_start, active_end = intervals[0][0], intervals[-1][1]
    if "pos_z" not in data or "motors_on" not in data:
        return (active_start, active_end), None

    times = data["time"]
    pos_z = data["pos_z"]
    motors_on = data["motors_on"] > 0.5
    valid = np.isfinite(times) & np.isfinite(pos_z)
    motor_indices = np.flatnonzero(valid & motors_on)
    if motor_indices.size == 0:
        return (active_start, active_end), None

    motor_start_time = float(times[motor_indices[0]])
    ground_samples = (
        valid & ~motors_on
        & (times < motor_start_time)
        & (times >= motor_start_time - 2.0)
    )
    if np.count_nonzero(ground_samples) < 5:
        return (active_start, active_end), None
    ground_z = float(np.nanmedian(pos_z[ground_samples]))

    within_active = (
        valid & active
        & (times >= active_start)
        & (times <= active_end)
    )
    established_clearance = max(0.30, 3.0 * ground_clearance_m)
    away_from_ground = within_active & (np.abs(pos_z - ground_z) > established_clearance)
    away_indices = np.flatnonzero(away_from_ground)
    if away_indices.size == 0:
        return (active_start, active_end), None

    # The sign is normally negative for NED z, but inferring it also supports ENU.
    sign_samples = away_indices[:min(50, away_indices.size)]
    airborne_sign = float(np.sign(np.nanmedian(pos_z[sign_samples] - ground_z)))
    if airborne_sign == 0.0:
        return (active_start, active_end), None
    signed_clearance = (pos_z - ground_z) * airborne_sign

    airborne = within_active & (signed_clearance > ground_clearance_m)
    airborne_indices = np.flatnonzero(airborne)
    if airborne_indices.size == 0:
        return (active_start, active_end), None
    flight_start_index = int(airborne_indices[0])

    contact = (
        within_active
        & (np.arange(times.size) > int(away_indices[0]))
        & (signed_clearance <= ground_clearance_m)
    )
    contact_indices = np.flatnonzero(contact)
    flight_end = active_end
    contact_time = None
    if contact_indices.size:
        contact_time = float(times[contact_indices[0]])
        flight_end = min(flight_end, contact_time)

    flight_start = max(active_start, float(times[flight_start_index]))
    if flight_end <= flight_start:
        return (active_start, active_end), None
    details = {
        "ground_z": ground_z,
        "contact_time": contact_time,
        "clearance_m": ground_clearance_m,
    }
    return (flight_start, flight_end), details


def title_with_label(path, suffix, title_label):
    title = f"{path.name} - {suffix}"
    return f"{title}\n{title_label}" if title_label else title


def analyze_rl_gate_passages(data, xlim):
    """Recover RL gate crossings in the common ENU plotting frame.

    ``logger_file`` records the generic position columns in NED, while the RL
    target debug values are stored in Paparazzi ENU.  A waypoint-index change
    is therefore checked after converting the logged position as
    ``ENU = (pos_y, pos_x, -pos_z)``.
    """
    times = data["time"]
    mask = (
        np.isfinite(times)
        & (times >= xlim[0])
        & (times <= xlim[1])
        & (data["rl_enabled"] > 0.5)
    )
    samples = np.flatnonzero(mask)
    centers = {}
    if samples.size == 0:
        return {"samples": samples, "centers": centers, "events": []}

    waypoint_indices = data["rl_waypoint_index"].astype(int)
    for gate_index in range(len(RL_GATE_YAWS)):
        gate_samples = samples[waypoint_indices[samples] == gate_index]
        if gate_samples.size:
            centers[gate_index] = np.asarray([
                np.nanmedian(data["rl_target_x"][gate_samples]),
                np.nanmedian(data["rl_target_y"][gate_samples]),
                np.nanmedian(data["rl_target_z"][gate_samples]),
            ])

    events = []
    for left, right in zip(samples[:-1], samples[1:]):
        old_index = int(waypoint_indices[left])
        new_index = int(waypoint_indices[right])
        if old_index == new_index:
            continue

        center = np.asarray([
            data["rl_target_x"][left],
            data["rl_target_y"][left],
            data["rl_target_z"][left],
        ])
        yaw = RL_GATE_YAWS[old_index % len(RL_GATE_YAWS)]

        # Projection used by rl_cfc_control.c, evaluated in NED.
        projection_left = (
            (data["pos_x"][left] - center[1]) * np.cos(yaw)
            + (data["pos_y"][left] - center[0]) * np.sin(yaw)
        )
        projection_right = (
            (data["pos_x"][right] - center[1]) * np.cos(yaw)
            + (data["pos_y"][right] - center[0]) * np.sin(yaw)
        )
        denominator = projection_right - projection_left
        crossed_plane_in_log = projection_left <= 0.0 and projection_right >= 0.0
        if crossed_plane_in_log and abs(denominator) > 1e-12:
            alpha = float(-projection_left / denominator)
            crossing_time = data["time"][left] + alpha * (data["time"][right] - data["time"][left])
        else:
            # The controller and logger both run periodically but are not phase
            # aligned.  The index transition proves the control task saw the
            # crossing; use the closest logged endpoint for aperture margins.
            alpha = 0.0 if abs(projection_left) <= abs(projection_right) else 1.0
            crossing_time = data["time"][right]

        east = data["pos_y"][left] + alpha * (data["pos_y"][right] - data["pos_y"][left])
        north = data["pos_x"][left] + alpha * (data["pos_x"][right] - data["pos_x"][left])
        up = -(data["pos_z"][left] + alpha * (data["pos_z"][right] - data["pos_z"][left]))
        crossing = np.asarray([east, north, up])

        normal_error = (
            (north - center[1]) * np.cos(yaw)
            + (east - center[0]) * np.sin(yaw)
        )
        if not crossed_plane_in_log:
            # Put the marker on the plane while keeping the closest sample's
            # tangent and height coordinates.
            crossing[0] -= normal_error * np.sin(yaw)
            crossing[1] -= normal_error * np.cos(yaw)

        # Gate tangent in ENU is (cos(yaw), -sin(yaw)).
        tangent_error = (
            (east - center[0]) * np.cos(yaw)
            - (north - center[1]) * np.sin(yaw)
        )
        height_error = up - center[2]
        within_opening = (
            abs(tangent_error) < RL_GATE_HALF_SIZE_M
            and abs(height_error) < RL_GATE_HALF_SIZE_M
            and abs(normal_error) < RL_GATE_HALF_SIZE_M
        )
        expected_advance = new_index == (old_index + 1) % len(RL_GATE_YAWS)
        # maybe_advance_waypoint() changes the index only after the gate-plane
        # crossing and the 1.5 m opening test both succeed.  The nearest logged
        # sample supplies an additional geometric consistency check.
        passed = bool(within_opening and expected_advance)
        events.append({
            "gate_index": old_index,
            "next_index": new_index,
            "time": float(crossing_time),
            "position_enu": crossing,
            "center_enu": center,
            "tangent_error": float(tangent_error),
            "height_error": float(height_error),
            "normal_error": float(normal_error),
            "plane_crossing_sampled": bool(crossed_plane_in_log),
            "passed": passed,
        })

    return {"samples": samples, "centers": centers, "events": events}


def plot_rl_trajectory(output_path, title, data, xlim, gate_analysis):
    times = data["time"]
    mask = np.isfinite(times) & (times >= xlim[0]) & (times <= xlim[1])
    fig, axis = plt.subplots(figsize=(9, 8), constrained_layout=True)
    east = data["pos_y"][mask]
    north = data["pos_x"][mask]
    axis.plot(east, north, lw=1.5, color="#2166ac", label="trajetória (ENU)")

    centers = gate_analysis["centers"]
    events = gate_analysis["events"]
    passed_indices = {event["gate_index"] for event in events if event["passed"]}
    for gate_index, center in sorted(centers.items()):
        yaw = RL_GATE_YAWS[gate_index]
        tangent = np.asarray([np.cos(yaw), -np.sin(yaw)])
        endpoints = np.vstack([
            center[:2] - RL_GATE_HALF_SIZE_M * tangent,
            center[:2] + RL_GATE_HALF_SIZE_M * tangent,
        ])
        color = "#1b7837" if gate_index in passed_indices else "#d95f02"
        label = "gate atravessado" if gate_index in passed_indices and gate_index == min(passed_indices) else None
        axis.plot(endpoints[:, 0], endpoints[:, 1], color=color, lw=4.0, alpha=0.75, label=label)
        axis.text(center[0], center[1], f" G{gate_index + 1}", color=color, fontsize=9)

    if centers:
        samples = gate_analysis["samples"]
        target_order = [int(data["rl_waypoint_index"][samples[0]])]
        target_order.extend(event["next_index"] for event in events)
        target_points = np.asarray([centers[index][:2] for index in target_order if index in centers])
        if target_points.size:
            axis.plot(
                target_points[:, 0], target_points[:, 1], "--o", color="#666666",
                ms=3.5, lw=1.0, alpha=0.75, label="sequência de targets",
            )

    for event in events:
        marker = "o" if event["passed"] else "x"
        color = "#1b7837" if event["passed"] else "#b2182b"
        axis.scatter(
            event["position_enu"][0], event["position_enu"][1],
            marker=marker, s=45, color=color, zorder=5,
        )

    if east.size:
        axis.scatter(east[0], north[0], marker="^", s=70, color="#000000", label="início RL", zorder=6)
        axis.scatter(east[-1], north[-1], marker="s", s=55, color="#984ea3", label="fim RL", zorder=6)
    axis.set_xlabel("leste / x ENU [m]")
    axis.set_ylabel("norte / y ENU [m]")
    axis.set_aspect("equal", adjustable="box")
    axis.grid(True, alpha=0.25)
    axis.legend()
    confirmed = sum(event["passed"] for event in events)
    axis.set_title(f"{title}\n{confirmed} passagem(ns) de gate confirmada(s)")
    fig.savefig(output_path, dpi=170)
    plt.close(fig)


def plot_rl_height(output_path, title, data, xlim, gate_analysis):
    times = data["time"]
    mask = np.isfinite(times) & (times >= xlim[0]) & (times <= xlim[1])
    active_times = times[mask] - xlim[0]
    height = -data["pos_z"][mask]
    target_height = data["rl_target_z"][mask]

    fig, axis = plt.subplots(figsize=(13, 6), constrained_layout=True)
    axis.plot(active_times, height, color="#2166ac", lw=1.5, label="altura do drone (ENU)")
    axis.plot(active_times, target_height, "--", color="#666666", lw=1.1, label="altura do target")
    if target_height.size and np.any(np.isfinite(target_height)):
        center_height = float(np.nanmedian(target_height))
        axis.axhspan(
            center_height - RL_GATE_HALF_SIZE_M,
            center_height + RL_GATE_HALF_SIZE_M,
            color="#1b7837", alpha=0.10, label="abertura vertical do gate",
        )
    for event in gate_analysis["events"]:
        relative_time = event["time"] - xlim[0]
        color = "#1b7837" if event["passed"] else "#b2182b"
        axis.axvline(relative_time, color=color, lw=0.9, alpha=0.65)
        axis.text(
            relative_time, event["position_enu"][2], f" G{event['gate_index'] + 1}",
            color=color, fontsize=8, rotation=90, va="bottom", ha="right",
        )
    axis.set_xlabel("tempo desde ativação do RL [s]")
    axis.set_ylabel("altura / z ENU [m]")
    axis.grid(True, alpha=0.25)
    axis.legend(loc="best")
    axis.set_title(title)
    fig.savefig(output_path, dpi=170)
    plt.close(fig)


def generate_rl(path, output_base, threshold_rpm, pad_s, active_only=False,
                active_until=None, title_label=None, ground_clearance_m=0.10):
    _columns, data = read_csv(path)
    if "time" not in data:
        raise SystemExit(f"{path} has no time column")

    active = data["rl_enabled"] > 0.5
    intervals = intervals_from_mask(data["time"], active, min_duration_s=0.5)
    out_dir = output_base / path.stem
    if active_only:
        if active_until is None:
            out_dir /= "rl_active"
        else:
            out_dir /= f"rl_active_until_{active_until:g}s"
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
    all_state_groups = state_groups[:4]
    body_rate_groups = [("body rates [rad/s]", ["rate_p", "rate_q", "rate_r"])]
    rpm_groups = [
        (f"motor {motor} [RPM]",
         [f"rpm_obs_{motor}", f"rpm_ref_{motor}", f"rl_rpm_cmd{motor}", f"rl_rpm_applied{motor}"])
        for motor in range(1, 5)
    ]
    rl_groups = [
        ("RL obs position", [f"rl_obs_{i}" for i in range(3)]),
        ("RL obs velocity", [f"rl_obs_{i}" for i in range(3, 6)]),
        ("RL obs attitude", [f"rl_obs_{i}" for i in range(6, 9)]),
        ("RL obs rates", [f"rl_obs_{i}" for i in range(9, 12)]),
        ("RL obs motor state", [f"rl_obs_{i}" for i in range(12, 16)]),
        ("RL obs next gate", [f"rl_obs_{i}" for i in range(16, 20)]),
        ("RL policy raw", [f"rl_policy_raw{i}" for i in range(1, 5)]),
        ("RL action", [f"rl_action{i}" for i in range(1, 5)]),
        ("RL RPM command", [f"rl_rpm_cmd{i}" for i in range(1, 5)]),
    ]
    timing_groups = [
        ("periodic dt [us]", ["rl_periodic_dt_us"]),
        ("sensor read [us]", ["rl_sensor_read_time_us"]),
        ("inference [us]", ["rl_inference_time_us"]),
        ("total [us]", ["rl_total_time_us"]),
    ]

    outputs = []
    plotted_intervals = intervals
    trajectory = None
    height_plot = None
    gate_analysis = None
    rpm_commanded_observed = None
    airborne_details = None
    if active_only:
        if not intervals:
            raise SystemExit(f"{path} has no interval with RL active")
        (start, end), airborne_details = active_airborne_xlim(
            data, active, intervals, ground_clearance_m,
        )
        if active_until is not None:
            end = min(end, active_until)
        if end <= start:
            raise SystemExit(f"{path} has no RL-active samples before {active_until} s")
        xlim = (start, end)
        plotted_intervals = [
            (max(a, start), min(b, end)) for a, b in intervals if b > start and a < end
        ]
        position_groups = [
            ("x [m]", ["pos_x", "rl_target_x", "rl_err_x"]),
            ("y [m]", ["pos_y", "rl_target_y", "rl_err_y"]),
            ("z [m]", ["pos_z", "rl_target_z", "rl_err_z"]),
        ]
        outputs = [
            (out_dir / f"{path.stem}_state_commands_rl_active.png", state_groups, xlim, "state, commands and RPM - RL active"),
            (out_dir / f"{path.stem}_all_states_rl_active.png", all_state_groups, xlim, "all states - RL active"),
            (out_dir / f"{path.stem}_body_rates_rl_active.png", body_rate_groups, xlim, "body rates p q r - RL active"),
            (out_dir / f"{path.stem}_position_vs_error_rl_active.png", position_groups, xlim, "position, target and error - RL active"),
            (out_dir / f"{path.stem}_rpm_cmds_rl_active.png", rpm_groups, xlim, "RPM commands - RL active"),
            (out_dir / f"{path.stem}_rl_observations_outputs_rl_active.png", rl_groups, xlim, "RL observations and outputs - RL active"),
            (out_dir / f"{path.stem}_rl_timing_rl_active.png", timing_groups, xlim, "RL timing - RL active"),
        ]
        trajectory = out_dir / f"{path.stem}_trajectory_xy_rl_active.png"
        height_plot = out_dir / f"{path.stem}_height_rl_active.png"
        rpm_commanded_observed = out_dir / f"{path.stem}_rpm_commanded_vs_observed_4x2_rl_active.png"
    else:
        outputs = [
            (out_dir / f"{path.stem}_state_commands_full.png", state_groups, None, "state, commands and RPM - full"),
            (out_dir / f"{path.stem}_all_states_full.png", all_state_groups, None, "all states - full"),
            (out_dir / f"{path.stem}_body_rates_full.png", body_rate_groups, None, "body rates p q r - full"),
            (out_dir / f"{path.stem}_rpm_cmds_full.png", rpm_groups, None, "RPM commands - full"),
            (out_dir / f"{path.stem}_rl_observations_outputs_full.png", rl_groups, None, "RL observations and outputs - full"),
            (out_dir / f"{path.stem}_rl_timing_full.png", timing_groups, None, "RL timing - full"),
        ]
        zoom_xlim = active_motor_xlim(data, threshold_rpm, pad_s)
        if zoom_xlim:
            outputs.extend([
                (out_dir / f"{path.stem}_state_commands_active_motors.png", state_groups, zoom_xlim, "state, commands and RPM - active motors"),
                (out_dir / f"{path.stem}_all_states_active_motors.png", all_state_groups, zoom_xlim, "all states - active motors"),
                (out_dir / f"{path.stem}_body_rates_active_motors.png", body_rate_groups, zoom_xlim, "body rates - active motors"),
            ])
        rpm_commanded_observed = out_dir / f"{path.stem}_rpm_commanded_vs_observed_4x2_full.png"

    for output_path, groups, xlim, suffix in outputs:
        plot_groups(
            output_path, title_with_label(path, suffix, title_label), data, groups,
            plotted_intervals, xlim=xlim, active_label="RL active",
        )
    if trajectory is not None:
        gate_analysis = analyze_rl_gate_passages(data, outputs[0][2])
        plot_rl_trajectory(
            trajectory, title_with_label(path, "XY trajectory - RL active", title_label),
            data, outputs[0][2], gate_analysis,
        )
        plot_rl_height(
            height_plot, title_with_label(path, "height - RL active", title_label),
            data, outputs[0][2], gate_analysis,
        )
    if rpm_commanded_observed is not None:
        comparison_xlim = outputs[0][2]
        if comparison_xlim is None:
            finite_times = data["time"][np.isfinite(data["time"])]
            comparison_xlim = (float(finite_times[0]), float(finite_times[-1]))
        plot_rpm_commanded_observed_4x2(
            rpm_commanded_observed,
            title_with_label(path, "RPM commanded vs observed", title_label),
            data,
            comparison_xlim,
            [f"rl_rpm_cmd{i}" for i in range(1, 5)],
        )

    html = [
        "<!doctype html><html><head><meta charset=\"utf-8\">",
        f"<title>{path.stem} RL onboard plots</title>",
        "<style>body{font-family:sans-serif;margin:24px;background:#f7f7f7;color:#222}"
        "img{max-width:100%;background:white;border:1px solid #ccc;margin:8px 0 28px}"
        "code{background:#eee;padding:2px 4px}</style></head><body>",
        f"<h1>{path.name}</h1>",
        f"<p><strong>{title_label}</strong></p>" if title_label else "",
        f"<p>RL-active intervals: <code>{plotted_intervals or 'none'}</code></p>",
        (f"<p>Airborne window uses estimated ground z=<code>{airborne_details['ground_z']:.3f} m</code> "
         f"and {airborne_details['clearance_m']:.2f} m clearance.</p>" if airborne_details else ""),
    ]
    for output_path, _groups, _xlim, suffix in outputs:
        html.append(f"<h2>{suffix}</h2><img src=\"{output_path.name}\">")
    if trajectory is not None:
        html.append(f"<h2>XY trajectory</h2><img src=\"{trajectory.name}\">")
    if height_plot is not None:
        html.append(f"<h2>Height</h2><img src=\"{height_plot.name}\">")
    if gate_analysis is not None:
        confirmed_events = [event for event in gate_analysis["events"] if event["passed"]]
        html.append(
            f"<h2>Gate passage check</h2><p><strong>{len(confirmed_events)} "
            "geometric crossing(s) confirmed.</strong></p><ul>"
        )
        for event in gate_analysis["events"]:
            position = event["position_enu"]
            result = "PASS" if event["passed"] else "FAIL"
            html.append(
                f"<li>G{event['gate_index'] + 1}: {result} at "
                f"t={event['time'] - outputs[0][2][0]:.3f} s, "
                f"ENU=({position[0]:.3f}, {position[1]:.3f}, {position[2]:.3f}) m, "
                f"lateral error={abs(event['tangent_error']):.3f} m, "
                f"height error={abs(event['height_error']):.3f} m</li>"
            )
        html.append("</ul>")
    if rpm_commanded_observed is not None:
        html.append(f"<h2>RPM commanded vs observed</h2><img src=\"{rpm_commanded_observed.name}\">")
    html.append("</body></html>")
    (out_dir / "index.html").write_text("\n".join(html))
    print(f"Saved RL onboard plots to {out_dir}")
    print(f"RL intervals shown: {plotted_intervals or 'none'}")
    if airborne_details:
        print(f"Airborne cutoff: {airborne_details}")
    if gate_analysis is not None:
        for event in gate_analysis["events"]:
            position = event["position_enu"]
            result = "PASS" if event["passed"] else "FAIL"
            print(
                f"Gate G{event['gate_index'] + 1}: {result}; "
                f"t_rel={event['time'] - outputs[0][2][0]:.3f}s; "
                f"ENU=({position[0]:.3f}, {position[1]:.3f}, {position[2]:.3f})m; "
                f"lateral_error={abs(event['tangent_error']):.3f}m; "
                f"height_error={abs(event['height_error']):.3f}m"
            )


def generate(path, output_base, threshold_rpm, pad_s, nn_active_only=False,
             nn_active_until=None, title_label=None, ground_clearance_m=0.10):
    columns, data = read_csv(path)
    if "time" not in data:
        raise SystemExit(f"{path} has no time column")
    if "rl_enabled" in data:
        return generate_rl(
            path, output_base, threshold_rpm, pad_s, nn_active_only,
            nn_active_until, title_label, ground_clearance_m,
        )

    nn_active = infer_nn_active(columns, data)
    nn_intervals = intervals_from_mask(data["time"], nn_active, min_duration_s=0.5)

    out_dir = output_base / path.stem
    if nn_active_only:
        if nn_active_until is None:
            out_dir /= "nn_active"
        else:
            out_dir /= f"nn_active_until_{nn_active_until:g}s"
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
    rpm_commands_2x2 = None
    rpm_commanded_observed = None
    airborne_details = None
    plotted_nn_intervals = nn_intervals
    if nn_active_only:
        if not nn_intervals:
            raise SystemExit(f"{path} has no interval with the NN active")
        (nn_start, nn_end), airborne_details = active_airborne_xlim(
            data, nn_active, nn_intervals, ground_clearance_m,
        )
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
        rpm_commands_2x2 = out_dir / f"{path.stem}_rpm_cmd_four_motors_nn_active.png"
        rpm_commanded_observed = out_dir / f"{path.stem}_rpm_commanded_vs_observed_4x2_nn_active.png"
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
        rpm_commanded_observed = out_dir / f"{path.stem}_rpm_commanded_vs_observed_4x2_full.png"

    for output_path, groups, xlim, suffix_title in outputs:
        plot_groups(
            output_path, title_with_label(path, suffix_title, title_label), data,
            groups, plotted_nn_intervals, xlim=xlim,
        )
    if rpm_commands_2x2 is not None:
        plot_rpm_commands_2x2(
            rpm_commands_2x2,
            title_with_label(
                path,
                "four motor RPM commands - NN active",
                title_label,
            ),
            data,
            nn_xlim,
        )
    if rpm_commanded_observed is not None:
        comparison_xlim = nn_xlim if nn_active_only else None
        if comparison_xlim is None:
            finite_times = data["time"][np.isfinite(data["time"])]
            comparison_xlim = (float(finite_times[0]), float(finite_times[-1]))
        plot_rpm_commanded_observed_4x2(
            rpm_commanded_observed,
            title_with_label(path, "RPM commanded vs observed", title_label),
            data,
            comparison_xlim,
            RPM_CMD_COLS,
        )

    html = [
        "<!doctype html><html><head><meta charset=\"utf-8\">",
        f"<title>{path.stem} onboard plots</title>",
        "<style>body{font-family:sans-serif;margin:24px;background:#f7f7f7;color:#222}"
        "img{max-width:100%;background:white;border:1px solid #ccc;margin:8px 0 28px}"
        "code{background:#eee;padding:2px 4px}</style></head><body>",
        f"<h1>{path.name}</h1>",
        f"<p><strong>{title_label}</strong></p>" if title_label else "",
        "<p>Orange bands mark NN activity. For old logs without <code>nn_enabled</code>, activity is inferred from "
        "<code>nn_in_*</code>, <code>network_out*</code>, and <code>rpm_cmd*</code> becoming non-zero.</p>",
        f"<p>NN intervals shown: {plotted_nn_intervals or 'none'}</p>",
        (f"<p>Airborne window uses estimated ground z=<code>{airborne_details['ground_z']:.3f} m</code> "
         f"and {airborne_details['clearance_m']:.2f} m clearance.</p>" if airborne_details else ""),
    ]
    for output_path, _groups, _xlim, _title in outputs:
        html.append(f"<h2>{output_path.name}</h2><img src=\"{output_path.name}\">")
    if rpm_commands_2x2 is not None:
        html.append(
            f"<h2>{rpm_commands_2x2.name}</h2>"
            f"<img src=\"{rpm_commands_2x2.name}\">"
        )
    if rpm_commanded_observed is not None:
        html.append(
            f"<h2>{rpm_commanded_observed.name}</h2>"
            f"<img src=\"{rpm_commanded_observed.name}\">"
        )
    html.append("</body></html>")
    (out_dir / "index.html").write_text("\n".join(html))

    print(f"Saved onboard plots to {out_dir}")
    print(f"NN intervals shown: {plotted_nn_intervals or 'none'}")
    if airborne_details:
        print(f"Airborne cutoff: {airborne_details}")


def main():
    parser = argparse.ArgumentParser(description="Plot onboard logger_file CSVs extracted from the drone.")
    parser.add_argument("csv_files", nargs="+", type=Path, help="Onboard CSV log(s)")
    parser.add_argument("--output-dir", type=Path, default=Path("plots") / "plots_onboard")
    parser.add_argument("--active-rpm-threshold", type=float, default=1000.0)
    parser.add_argument("--active-pad-s", type=float, default=2.0)
    parser.add_argument("--nn-active-only", action="store_true", help="Plot only the interval between NN activation and deactivation")
    parser.add_argument("--nn-active-until", type=float, help="End an NN-active-only plot at this log time in seconds")
    parser.add_argument(
        "--ground-clearance-m", type=float, default=0.10,
        help="Ground-clearance threshold used to end NN/RL-active airborne plots (default: 0.10 m)",
    )
    parser.add_argument("--title-label", help="Extra controller/model/filter label shown in every title")
    args = parser.parse_args()

    for csv_file in args.csv_files:
        generate(
            csv_file,
            args.output_dir,
            args.active_rpm_threshold,
            args.active_pad_s,
            args.nn_active_only,
            args.nn_active_until,
            args.title_label,
            args.ground_clearance_m,
        )


if __name__ == "__main__":
    main()
