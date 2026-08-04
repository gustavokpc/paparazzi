#!/usr/bin/env python3
"""Replay an onboard RL-CfC log through the exported policy and Bebop2 dynamics."""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
if not os.environ.get("DISPLAY"):
    import matplotlib

    matplotlib.use("Agg")
import matplotlib.pyplot as plt


SCRIPT = Path(__file__).resolve()
PAPARAZZI = SCRIPT.parents[3]
ESTAG = PAPARAZZI.parent
LNN_ROOT = ESTAG / "LNN_estag"
sys.path.insert(0, str(LNN_ROOT))

from LNN_behavioural_cloning_quadrotor.utils.dynamics_models import quadrotor_sim_matlab as dynamics


STATE_NAMES = (
    "x_G", "y_G", "z_G",
    "vx_G", "vy_G", "vz_G",
    "phi", "theta", "psi_G",
    "p", "q", "r",
)
STATE_UNITS = (
    "m", "m", "m",
    "m/s", "m/s", "m/s",
    "rad", "rad", "rad",
    "rad/s", "rad/s", "rad/s",
)
COLORS = ("#0072B2", "#D55E00", "#009E73", "#CC79A7")


def load_log(path: Path) -> np.ndarray:
    data = np.genfromtxt(
        path,
        delimiter=",",
        names=True,
        dtype=None,
        encoding=None,
        invalid_raise=False,
    )
    if data.ndim == 0:
        data = data.reshape(1)
    mask = (
        (data["rl_enabled"] == 1)
        & (data["rl_periodic_count"] > 0)
        & np.isfinite(data["time"])
        & np.isfinite(data["rl_obs_0"])
    )
    active = data[mask]
    if len(active) < 3:
        raise ValueError(f"{path} does not contain a usable RL-active segment")
    return active


def compile_policy_library(output: Path) -> None:
    module = PAPARAZZI / "sw/airborne/modules/rl_cfc_control"
    subprocess.run(
        [
            "gcc", "-std=c99", "-O2", "-fPIC", "-shared",
            str(module / "rl_cfc_operations.c"),
            str(module / "rl_cfc_parameters.c"),
            "-I", str(PAPARAZZI / "sw/airborne"),
            "-lm", "-o", str(output),
        ],
        check=True,
    )


class Policy:
    def __init__(self, library: Path):
        self.lib = ctypes.CDLL(str(library))
        self.a20 = ctypes.c_float * 20
        self.a4 = ctypes.c_float * 4
        self.lib.rl_cfc_reset.argtypes = []
        self.lib.rl_cfc_control.argtypes = [
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
        ]

    def reset(self) -> None:
        self.lib.rl_cfc_reset()

    def predict(self, observation: np.ndarray) -> np.ndarray:
        output = self.a4()
        self.lib.rl_cfc_control(
            self.a20(*np.asarray(observation, dtype=np.float32)),
            output,
        )
        return np.ctypeslib.as_array(output).astype(np.float64).copy()


def rotation_body_to_world(phi: float, theta: float, psi: float) -> np.ndarray:
    cp, sp = np.cos(phi), np.sin(phi)
    ct, st = np.cos(theta), np.sin(theta)
    cy, sy = np.cos(psi), np.sin(psi)
    return np.array(
        [
            [cy * ct, cy * st * sp - sy * cp, cy * st * cp + sy * sp],
            [sy * ct, sy * st * sp + cy * cp, sy * st * cp - cy * sp],
            [-st, ct * sp, ct * cp],
        ]
    )


def world_to_body_state(world: np.ndarray) -> np.ndarray:
    body = np.asarray(world, dtype=np.float64).copy()
    rotation = rotation_body_to_world(*body[6:9])
    body[0:3] = -rotation.T @ body[0:3]
    body[3:6] = rotation.T @ body[3:6]
    return body


def body_to_world_state(body: np.ndarray) -> np.ndarray:
    world = np.asarray(body, dtype=np.float64).copy()
    rotation = rotation_body_to_world(*world[6:9])
    world[0:3] = -rotation @ world[0:3]
    world[3:6] = rotation @ world[3:6]
    return world


def wrap_pi(values: np.ndarray) -> np.ndarray:
    return (values + np.pi) % (2.0 * np.pi) - np.pi


def observation_matrix(rows: np.ndarray) -> np.ndarray:
    return np.column_stack(
        [np.asarray(rows[f"rl_obs_{index}"], dtype=float) for index in range(20)]
    )


def logged_actions(rows: np.ndarray) -> np.ndarray:
    return np.column_stack(
        [np.asarray(rows[f"rl_policy_raw{index}"], dtype=float) for index in range(1, 5)]
    )


def physical_states(rows: np.ndarray) -> np.ndarray:
    obs = observation_matrix(rows)
    states = obs[:, :12].copy()
    states[:, 9:12] = np.column_stack(
        [rows["rate_p"], rows["rate_q"], rows["rate_r"]]
    )
    return states


def simulation_state(row: np.void) -> np.ndarray:
    observation = np.array([row[f"rl_obs_{index}"] for index in range(20)], dtype=float)
    world = np.zeros(19, dtype=float)
    world[:9] = observation[:9]
    world[9:12] = [row["rate_p"], row["rate_q"], row["rate_r"]]
    world[15:19] = [row[f"rpm_obs_{index}"] for index in range(1, 5)]
    return world_to_body_state(world)


def rk4_step(state: np.ndarray, action: np.ndarray, dt: float) -> np.ndarray:
    k1 = dynamics.dynamics(state, action)
    k2 = dynamics.dynamics(state + 0.5 * dt * k1, action)
    k3 = dynamics.dynamics(state + 0.5 * dt * k2, action)
    k4 = dynamics.dynamics(state + dt * k3, action)
    return state + dt * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0


class RateFilter:
    def __init__(self, initial: np.ndarray, cutoff_hz: float = 10.0, dt: float = 0.01):
        initial = np.asarray(initial, dtype=float)
        self.last_in = initial.copy()
        self.last_out = initial.copy()
        tau = dt / (2.0 * np.tan(np.pi * cutoff_hz * dt))
        self.time_const = 2.0 * tau / dt

    def update(self, value: np.ndarray) -> np.ndarray:
        value = np.asarray(value, dtype=float)
        output = (
            value + self.last_in + (self.time_const - 1.0) * self.last_out
        ) / (1.0 + self.time_const)
        self.last_in = value.copy()
        self.last_out = output.copy()
        return output


def replay_policy(policy: Policy, observations: np.ndarray) -> np.ndarray:
    policy.reset()
    return np.asarray([policy.predict(observation) for observation in observations])


def one_step_predictions(
    rows: np.ndarray, actions: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    predicted = []
    actual = []
    predicted_rpm = []
    actual_rpm = []
    for index in range(len(rows) - 1):
        dt = float(rows["time"][index + 1] - rows["time"][index])
        count_step = int(rows["rl_periodic_count"][index + 1] - rows["rl_periodic_count"][index])
        if not 0.005 <= dt <= 0.02 or count_step != 1:
            continue
        next_body = rk4_step(simulation_state(rows[index]), actions[index], dt)
        next_world = body_to_world_state(next_body)
        predicted.append(next_world[:12])
        predicted_rpm.append(next_world[15:19])
        actual_state = physical_states(rows[index + 1:index + 2])[0]
        actual.append(actual_state)
        actual_rpm.append(
            [rows[index + 1][f"rpm_obs_{motor}"] for motor in range(1, 5)]
        )
    predicted = np.asarray(predicted)
    actual = np.asarray(actual)
    predicted[:, 6:9] = wrap_pi(predicted[:, 6:9])
    actual[:, 6:9] = wrap_pi(actual[:, 6:9])
    return predicted, actual, np.asarray(predicted_rpm), np.asarray(actual_rpm)


def recursive_rollout(
    policy: Policy,
    rows: np.ndarray,
    horizon_end: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    policy.reset()
    state = simulation_state(rows[0])
    initial_obs = np.array([rows[0][f"rl_obs_{index}"] for index in range(20)], dtype=float)
    rate_filter = RateFilter(initial_obs[9:12])
    next_gate = initial_obs[16:20].copy()
    states = []
    actions = []
    rpms = []
    for index in range(horizon_end):
        world = body_to_world_state(state)
        observation = np.zeros(20, dtype=float)
        observation[:9] = world[:9]
        observation[9:12] = rate_filter.last_out if index == 0 else rate_filter.update(world[9:12])
        observation[12:16] = 2.0 * (world[15:19] - 5000.0) / 5000.0 - 1.0
        observation[16:20] = next_gate
        action = policy.predict(observation)
        states.append(world[:12].copy())
        rpms.append(world[15:19].copy())
        actions.append(action)
        if index + 1 >= horizon_end:
            break
        dt = float(rows["time"][index + 1] - rows["time"][index])
        dt = float(np.clip(dt, 0.005, 0.02))
        state = rk4_step(state, action, dt)
    states = np.asarray(states)
    states[:, 6:9] = wrap_pi(states[:, 6:9])
    return states, np.asarray(actions), np.asarray(rpms)


def plot_policy(time: np.ndarray, expected: np.ndarray, actual: np.ndarray, output: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=True, constrained_layout=True)
    for index, axis in enumerate(axes.flat):
        axis.plot(time, actual[:, index], color="#222222", lw=1.4, label="ação registrada no voo")
        axis.plot(time, expected[:, index], color=COLORS[index], lw=1.0, ls="--", label="replay da rede")
        rmse = np.sqrt(np.mean((expected[:, index] - actual[:, index]) ** 2))
        axis.set_title(f"motor {index + 1} — RMSE={rmse:.2e}")
        axis.set_ylabel("ação normalizada")
        axis.grid(alpha=0.25)
    axes[1, 0].set_xlabel("tempo desde ativação [s]")
    axes[1, 1].set_xlabel("tempo desde ativação [s]")
    axes[0, 0].legend()
    fig.suptitle("Replay sequencial da política — estado real → ação esperada × ação embarcada")
    fig.savefig(output, dpi=170)
    plt.close(fig)


def plot_states(
    time: np.ndarray,
    expected: np.ndarray,
    actual: np.ndarray,
    output: Path,
    title: str,
    error_only: bool = False,
) -> None:
    fig, axes = plt.subplots(4, 3, figsize=(18, 14), sharex=True, constrained_layout=True)
    for index, axis in enumerate(axes.flat):
        error = expected[:, index] - actual[:, index]
        if index in (6, 7, 8):
            error = wrap_pi(error)
        rmse = np.sqrt(np.mean(error**2))
        if error_only:
            axis.plot(time, error, color="#7B3294", lw=1.0)
            axis.axhline(0.0, color="#222222", lw=0.8)
            axis.set_ylabel(f"erro [{STATE_UNITS[index]}]")
        else:
            axis.plot(time, actual[:, index], color="#222222", lw=1.4, label="estado real")
            axis.plot(time, expected[:, index], color="#D55E00", lw=1.1, ls="--", label="estado esperado")
            axis.set_ylabel(STATE_UNITS[index])
        axis.set_title(f"{STATE_NAMES[index]} — RMSE={rmse:.3g}")
        axis.grid(alpha=0.25)
        if index >= 9:
            axis.set_xlabel("tempo desde ativação [s]")
    if not error_only:
        axes[0, 0].legend()
    fig.suptitle(title)
    fig.savefig(output, dpi=170)
    plt.close(fig)


def plot_recursive_actions(
    time: np.ndarray,
    expected: np.ndarray,
    actual: np.ndarray,
    output: Path,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=True, constrained_layout=True)
    for index, axis in enumerate(axes.flat):
        axis.plot(time, actual[:, index], color="#222222", lw=1.4, label="voo real")
        axis.plot(time, expected[:, index], color=COLORS[index], lw=1.1, ls="--", label="rollout")
        axis.set_title(f"motor {index + 1}")
        axis.set_ylabel("ação normalizada")
        axis.grid(alpha=0.25)
    axes[1, 0].set_xlabel("tempo desde ativação [s]")
    axes[1, 1].set_xlabel("tempo desde ativação [s]")
    axes[0, 0].legend()
    fig.suptitle("Ações após fechar a política sobre os estados previstos pelo modelo")
    fig.savefig(output, dpi=170)
    plt.close(fig)


def plot_motor_dynamics(
    one_time: np.ndarray,
    one_expected: np.ndarray,
    one_actual: np.ndarray,
    rollout_time: np.ndarray,
    rollout_expected: np.ndarray,
    rollout_actual: np.ndarray,
    output: Path,
) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(20, 8), sharex="row", constrained_layout=True)
    for index in range(4):
        one_error = one_expected[:, index] - one_actual[:, index]
        axes[0, index].plot(one_time, one_error, color=COLORS[index], lw=1.1)
        axes[0, index].axhline(0.0, color="#222222", lw=0.8)
        axes[0, index].set_title(
            f"motor {index + 1} — 1 passo RMSE={np.sqrt(np.mean(one_error**2)):.1f} rpm"
        )
        axes[0, index].set_ylabel("erro [rpm]")
        axes[0, index].grid(alpha=0.25)
        axes[1, index].plot(
            rollout_time, rollout_actual[:, index],
            color="#222222", lw=1.4, label="RPM real",
        )
        axes[1, index].plot(
            rollout_time, rollout_expected[:, index],
            color=COLORS[index], lw=1.1, ls="--", label="RPM esperado",
        )
        axes[1, index].set_ylabel("rpm")
        axes[1, index].set_xlabel("tempo desde ativação [s]")
        axes[1, index].grid(alpha=0.25)
    axes[1, 0].legend()
    fig.suptitle(
        "Dinâmica dos motores — erro de previsão de 10 ms (acima) e rollout recursivo (abaixo)"
    )
    fig.savefig(output, dpi=170)
    plt.close(fig)


def equivalent_external_moment_rms(rows: np.ndarray, actions: np.ndarray) -> list[float]:
    """Approximate the unmodelled moment after smoothing measured body rates."""
    time = np.asarray(rows["time"], dtype=float)
    rates = np.column_stack([rows["rate_p"], rows["rate_q"], rows["rate_r"]])
    half_window = 4
    kernel = np.ones(2 * half_window + 1) / (2 * half_window + 1)
    smoothed = np.column_stack(
        [
            np.convolve(
                np.pad(rate, (half_window, half_window), mode="edge"),
                kernel,
                mode="valid",
            )
            for rate in rates.T
        ]
    )
    measured_acceleration = np.column_stack(
        [np.gradient(smoothed[:, axis], time) for axis in range(3)]
    )
    model_acceleration = np.asarray(
        [
            dynamics.dynamics(simulation_state(row), action)[9:12]
            for row, action in zip(rows, actions)
        ]
    )
    inertia = np.array([0.001920, 0.001850, 0.003340])
    residual_moment = (measured_acceleration - model_acceleration) * inertia
    return np.sqrt(np.mean(residual_moment**2, axis=0)).tolist()


def estimate_motor_taus(rows: np.ndarray, actions: np.ndarray) -> dict[str, object]:
    current_rpm = np.column_stack(
        [rows[f"rpm_obs_{motor}"][:-1] for motor in range(1, 5)]
    )
    next_rpm = np.column_stack(
        [rows[f"rpm_obs_{motor}"][1:] for motor in range(1, 5)]
    )
    target_rpm = 5000.0 + 5000.0 * np.clip(actions[:-1], 0.0, 1.0)
    dt = np.diff(np.asarray(rows["time"], dtype=float))[:, None]
    candidates = np.linspace(0.008, 0.080, 145)
    per_motor_rmse = np.empty((len(candidates), 4))
    for index, tau in enumerate(candidates):
        prediction = target_rpm + (current_rpm - target_rpm) * np.exp(-dt / tau)
        per_motor_rmse[index] = np.sqrt(np.mean((prediction - next_rpm) ** 2, axis=0))
    best_indices = np.argmin(per_motor_rmse, axis=0)
    shared_index = int(np.argmin(np.sqrt(np.mean(per_motor_rmse**2, axis=1))))
    return {
        "per_motor_s": candidates[best_indices].tolist(),
        "per_motor_rpm_rmse": per_motor_rmse[best_indices, np.arange(4)].tolist(),
        "best_shared_s": float(candidates[shared_index]),
        "best_shared_per_motor_rpm_rmse": per_motor_rmse[shared_index].tolist(),
    }


def rmse_by_state(expected: np.ndarray, actual: np.ndarray) -> dict[str, float]:
    error = expected - actual
    error[:, 6:9] = wrap_pi(error[:, 6:9])
    return {
        name: float(value)
        for name, value in zip(STATE_NAMES, np.sqrt(np.mean(error**2, axis=0)))
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("onboard_csv", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--motor-tau", type=float, default=0.04)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = load_log(args.onboard_csv)
    observations = observation_matrix(rows)
    actual_actions = logged_actions(rows)
    time = np.asarray(rows["time"] - rows["time"][0], dtype=float)

    dynamics.set_motor_tau(args.motor_tau)
    with tempfile.TemporaryDirectory(prefix="rl_onboard_replay_", dir="/tmp") as temp:
        library = Path(temp) / "librl_cfc_replay.so"
        compile_policy_library(library)
        policy = Policy(library)

        expected_actions = replay_policy(policy, observations)
        plot_policy(
            time, expected_actions, actual_actions,
            args.output_dir / "policy_replay_real_state_vs_onboard_action.png",
        )

        transition_indices = np.flatnonzero(
            np.diff(np.asarray(rows["rl_waypoint_index"], dtype=int)) != 0
        )
        horizon_end = int(transition_indices[0] + 1) if len(transition_indices) else min(len(rows), 100)
        gate_rows = rows[:horizon_end]
        gate_actions = actual_actions[:horizon_end]

        one_expected, one_actual, one_rpm_expected, one_rpm_actual = one_step_predictions(
            gate_rows, gate_actions
        )
        one_time = time[1:1 + len(one_expected)]
        plot_states(
            one_time, one_expected, one_actual,
            args.output_dir / "dynamics_one_step_prediction_error.png",
            "Previsão física de um passo (10 ms) — modelo iniciado no estado real a cada amostra",
            error_only=True,
        )

        rollout_states, rollout_actions, rollout_rpm = recursive_rollout(
            policy, gate_rows, horizon_end
        )
        real_states = physical_states(gate_rows)[:len(rollout_states)]
        real_rpm = np.column_stack(
            [gate_rows[f"rpm_obs_{motor}"] for motor in range(1, 5)]
        )[:len(rollout_rpm)]
        rollout_time = time[:len(rollout_states)]
        plot_states(
            rollout_time, rollout_states, real_states,
            args.output_dir / "recursive_open_loop_states.png",
            "Rollout recursivo até o próximo gate — esperado pelo modelo × voo real",
        )
        plot_recursive_actions(
            rollout_time,
            rollout_actions,
            actual_actions[:len(rollout_actions)],
            args.output_dir / "recursive_open_loop_actions.png",
        )
        plot_motor_dynamics(
            one_time,
            one_rpm_expected,
            one_rpm_actual,
            rollout_time,
            rollout_rpm,
            real_rpm,
            args.output_dir / "motor_dynamics_one_step_and_recursive.png",
        )

    action_error = expected_actions - actual_actions
    gaps = np.flatnonzero(
        np.diff(np.asarray(rows["rl_periodic_count"], dtype=int)) != 1
    )
    initial_contiguous_length = int(gaps[0] + 1) if len(gaps) else len(rows)
    one_rpm_error = one_rpm_expected - one_rpm_actual
    report = {
        "onboard_log": str(args.onboard_csv),
        "checkpoint": "recurrent_ppo_figure8_gates_83200000_steps.zip",
        "rate_filter": "first-order 10 Hz",
        "motor_tau_s": args.motor_tau,
        "samples": int(len(rows)),
        "rollout_duration_s": float(rollout_time[-1]),
        "policy_action_rmse_all": np.sqrt(np.mean(action_error**2, axis=0)).tolist(),
        "policy_action_max_abs_all": np.max(np.abs(action_error), axis=0).tolist(),
        "initial_contiguous_policy_samples": initial_contiguous_length,
        "policy_action_rmse_initial_contiguous": np.sqrt(
            np.mean(action_error[:initial_contiguous_length] ** 2, axis=0)
        ).tolist(),
        "one_step_state_rmse": rmse_by_state(one_expected, one_actual),
        "one_step_motor_rpm_rmse": np.sqrt(np.mean(one_rpm_error**2, axis=0)).tolist(),
        "identified_motor_tau": estimate_motor_taus(gate_rows, gate_actions),
        "equivalent_unmodelled_moment_rms_Nm": equivalent_external_moment_rms(
            gate_rows, gate_actions
        ),
        "recursive_state_rmse": rmse_by_state(rollout_states, real_states),
        "recursive_motor_rpm_rmse": np.sqrt(
            np.mean((rollout_rpm - real_rpm) ** 2, axis=0)
        ).tolist(),
    }
    report_path = args.output_dir / "replay_metrics.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Saved analysis to {args.output_dir}")


if __name__ == "__main__":
    main()
