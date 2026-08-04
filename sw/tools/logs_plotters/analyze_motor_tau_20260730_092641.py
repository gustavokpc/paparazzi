#!/usr/bin/env python3
"""Plot the 30-32 s motor transient and estimate motor time constants."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
if not os.environ.get("DISPLAY"):
    import matplotlib

    matplotlib.use("Agg")

import matplotlib.pyplot as plt


COLORS = ["#2166ac", "#b2182b", "#1b7837", "#984ea3"]


def read_window(path: Path, start: float, end: float) -> dict[str, np.ndarray]:
    with path.open(newline="") as stream:
        rows = [
            row
            for row in csv.DictReader(stream)
            if start <= float(row["time"]) <= end
        ]
    if not rows:
        raise ValueError("No samples in requested time window")
    return {
        name: np.asarray([float(row[name]) for row in rows], dtype=float)
        for name in rows[0]
        if all(row.get(name) not in (None, "") for row in rows)
    }


def simulate(
    time: np.ndarray,
    command: np.ndarray,
    initial_rpm: float,
    tau: float,
    delay: float,
) -> np.ndarray:
    delayed = np.interp(
        time - delay, time, command, left=initial_rpm, right=command[-1]
    )
    predicted = np.empty_like(time)
    predicted[0] = initial_rpm
    for index in range(len(time) - 1):
        dt = time[index + 1] - time[index]
        alpha = np.exp(-dt / tau)
        predicted[index + 1] = (
            alpha * predicted[index] + (1.0 - alpha) * delayed[index]
        )
    return predicted


def fit_motor(time, command, observed):
    step_index = int(np.argmax(np.abs(np.diff(command))))
    fit_time = time[step_index:] - time[step_index]
    fit_command = command[step_index:].copy()
    fit_observed = observed[step_index:].copy()
    fit_command[0] = fit_observed[0]

    best = None
    for delay in np.linspace(0.0, 0.040, 161):
        for tau in np.linspace(0.005, 0.120, 461):
            predicted = simulate(
                fit_time, fit_command, fit_observed[0], tau, delay
            )
            rmse = float(np.sqrt(np.mean((predicted - fit_observed) ** 2)))
            if best is None or rmse < best["rmse_rpm"]:
                residual = predicted - fit_observed
                denominator = np.sum(
                    (fit_observed - np.mean(fit_observed)) ** 2
                )
                best = {
                    "tau_s": float(tau),
                    "delay_s": float(delay),
                    "rmse_rpm": rmse,
                    "r2": float(1.0 - np.sum(residual**2) / denominator),
                    "step_time_s": float(time[step_index + 1]),
                    "fit_time": fit_time,
                    "predicted": predicted,
                    "observed": fit_observed,
                    "command": fit_command,
                }
    return best


def fit_without_delay(time, command, observed):
    step_index = int(np.argmax(np.abs(np.diff(command))))
    fit_time = time[step_index:] - time[step_index]
    fit_command = command[step_index:].copy()
    fit_observed = observed[step_index:].copy()
    fit_command[0] = fit_observed[0]
    best = None
    for tau in np.linspace(0.005, 0.120, 2301):
        predicted = simulate(fit_time, fit_command, fit_observed[0], tau, 0.0)
        rmse = float(np.sqrt(np.mean((predicted - fit_observed) ** 2)))
        if best is None or rmse < best[0]:
            best = (rmse, float(tau))
    return best[1]


def plot_crop(data, output: Path, title: str, start: float, end: float) -> None:
    time = data["time"]
    fig, axes = plt.subplots(
        4, 1, figsize=(16, 13), sharex=True, constrained_layout=True
    )
    for motor, axis in enumerate(axes, start=1):
        series = [
            (f"rpm_obs_{motor}", "#d95f02"),
            (f"rpm_ref_{motor}", "#2166ac"),
            (f"rl_rpm_cmd{motor}", "#1b7837"),
            (f"rl_rpm_applied{motor}", "#984ea3"),
        ]
        for name, color in series:
            axis.plot(time, data[name], lw=1.05, color=color, label=name)
        axis.set_ylabel(f"Motor {motor} [RPM]")
        axis.set_xlim(start, end)
        axis.grid(True, alpha=0.25)
        axis.legend(ncol=4, fontsize=8, loc="best")
    axes[-1].set_xlabel("tempo do log [s]")
    fig.suptitle(title, fontsize=14)
    fig.savefig(output, dpi=170)
    plt.close(fig)


def plot_fits(data, fits, output: Path, title: str) -> None:
    fig, axes = plt.subplots(
        4, 1, figsize=(16, 13), sharex=True, constrained_layout=True
    )
    for motor, (axis, fit) in enumerate(zip(axes, fits), start=1):
        absolute_time = fit["fit_time"] + fit["step_time_s"]
        # fit_time begins one sample before the logged command jump.
        absolute_time -= np.median(np.diff(data["time"]))
        axis.plot(
            absolute_time,
            fit["command"],
            color="#2166ac",
            lw=1.2,
            label=f"rpm_ref_{motor}",
        )
        axis.plot(
            absolute_time,
            fit["observed"],
            color="#d95f02",
            lw=1.2,
            label=f"rpm_obs_{motor}",
        )
        axis.plot(
            absolute_time,
            fit["predicted"],
            color="#1b7837",
            lw=1.3,
            ls="--",
            label="modelo ajustado",
        )
        axis.text(
            0.985,
            0.08,
            (
                f"τ={1e3 * fit['tau_s']:.2f} ms | "
                f"atraso={1e3 * fit['delay_s']:.2f} ms | "
                f"R²={fit['r2']:.4f}"
            ),
            transform=axis.transAxes,
            ha="right",
            va="bottom",
            fontsize=9,
            bbox={"facecolor": "white", "alpha": 0.82, "edgecolor": "#cccccc"},
        )
        axis.set_ylabel(f"Motor {motor} [RPM]")
        axis.grid(True, alpha=0.25)
        axis.legend(ncol=3, fontsize=8, loc="lower right")
    axes[-1].set_xlabel("tempo do log [s]")
    fig.suptitle(title, fontsize=14)
    fig.savefig(output, dpi=170)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--start", type=float, default=30.0)
    parser.add_argument("--end", type=float, default=32.0)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data = read_window(args.csv_path, args.start, args.end)

    fits = []
    metrics = {"log": args.csv_path.name, "window_s": [args.start, args.end]}
    for motor in range(1, 5):
        fit = fit_motor(
            data["time"], data[f"rpm_ref_{motor}"], data[f"rpm_obs_{motor}"]
        )
        fit["effective_tau_without_separate_delay_s"] = fit_without_delay(
            data["time"], data[f"rpm_ref_{motor}"], data[f"rpm_obs_{motor}"]
        )
        fits.append(fit)
        metrics[f"motor_{motor}"] = {
            key: value
            for key, value in fit.items()
            if key not in {"fit_time", "predicted", "observed", "command"}
        }

    tau = np.asarray([fit["tau_s"] for fit in fits])
    delay = np.asarray([fit["delay_s"] for fit in fits])
    effective = np.asarray(
        [fit["effective_tau_without_separate_delay_s"] for fit in fits]
    )
    metrics["summary"] = {
        "mean_tau_s": float(np.mean(tau)),
        "std_tau_s": float(np.std(tau)),
        "mean_delay_s": float(np.mean(delay)),
        "mean_effective_tau_without_separate_delay_s": float(np.mean(effective)),
    }

    label = (
        "20260730-092641 | RL recurrent PPO CfC noise-measurements 51.2M | "
        "rate filter OFF | janela 30–32 s"
    )
    crop_path = args.output_dir / "20260730-092641_rpm_commands_30s_32s.png"
    fit_path = args.output_dir / "20260730-092641_motor_tau_fit_30s_32s.png"
    plot_crop(data, crop_path, f"RPM dos motores — recorte 30–32 s\n{label}", args.start, args.end)
    plot_fits(data, fits, fit_path, f"Ajuste da dinâmica dos motores\n{label}")

    json_path = args.output_dir / "20260730-092641_motor_tau_30s_32s.json"
    json_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    text_path = args.output_dir / "20260730-092641_motor_tau_30s_32s.txt"
    lines = [
        label,
        "",
        "Modelo: rpm_obs(t+dt) = exp(-dt/tau)*rpm_obs(t) + "
        "(1-exp(-dt/tau))*rpm_ref(t-delay)",
        "",
    ]
    for motor, fit in enumerate(fits, start=1):
        lines.append(
            f"M{motor}: tau={1e3*fit['tau_s']:.2f} ms, "
            f"delay={1e3*fit['delay_s']:.2f} ms, "
            f"tau_effective_no_delay={1e3*fit['effective_tau_without_separate_delay_s']:.2f} ms, "
            f"RMSE={fit['rmse_rpm']:.1f} RPM, R2={fit['r2']:.5f}"
        )
    lines.extend(
        [
            "",
            f"Mean motor tau: {1e3*np.mean(tau):.2f} ms",
            f"Mean delay: {1e3*np.mean(delay):.2f} ms",
            f"Mean effective tau without separate delay: {1e3*np.mean(effective):.2f} ms",
        ]
    )
    text_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(crop_path)
    print(fit_path)
    print(text_path)
    print(json_path)


if __name__ == "__main__":
    main()
