#!/usr/bin/env python3
"""Generate the complete onboard RL plot set for an explicit time window."""

from __future__ import annotations

import argparse
import importlib.util
import os
from pathlib import Path

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
if not os.environ.get("DISPLAY"):
    import matplotlib

    matplotlib.use("Agg")

import matplotlib.pyplot as plt


COLORS = ["#2166ac", "#b2182b", "#1b7837", "#984ea3"]


def load_plotter(path: Path):
    spec = importlib.util.spec_from_file_location("plot_onboard_log", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"não foi possível importar {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def clipped_intervals(plotter, data, start: float, end: float):
    active = data["rl_enabled"] > 0.5
    intervals = plotter.intervals_from_mask(data["time"], active, min_duration_s=0.0)
    return [
        (max(a, start), min(b, end))
        for a, b in intervals
        if b >= start and a <= end
    ]


def plot_xy(path: Path, title: str, data, xlim) -> None:
    time = data["time"]
    mask = np.isfinite(time) & (time >= xlim[0]) & (time <= xlim[1])
    fig, axis = plt.subplots(figsize=(9, 8), constrained_layout=True)
    axis.plot(data["pos_x"][mask], data["pos_y"][mask], lw=1.4, label="posição")
    axis.scatter(
        data["pos_x"][mask][0],
        data["pos_y"][mask][0],
        color="#1b7837",
        s=55,
        zorder=3,
        label="início",
    )
    axis.scatter(
        data["pos_x"][mask][-1],
        data["pos_y"][mask][-1],
        color="#b2182b",
        s=55,
        zorder=3,
        label="fim / ativação RL",
    )
    axis.set_xlabel("x [m]")
    axis.set_ylabel("y [m]")
    axis.set_aspect("equal", adjustable="datalim")
    axis.grid(True, alpha=0.25)
    axis.legend()
    axis.set_title(title)
    fig.savefig(path, dpi=170)
    plt.close(fig)


def plot_commanded_observed(path: Path, title: str, data, xlim) -> None:
    time = data["time"]
    mask = np.isfinite(time) & (time >= xlim[0]) & (time <= xlim[1])
    fig, axes = plt.subplots(
        4, 2, figsize=(18, 14), sharex=True, sharey="row", constrained_layout=True
    )
    for index in range(4):
        motor = index + 1
        command = f"rl_rpm_cmd{motor}"
        observed = f"rpm_obs_{motor}"
        axes[index, 0].plot(
            time[mask], data[command][mask], color=COLORS[index], lw=1.1
        )
        axes[index, 1].plot(
            time[mask], data[observed][mask], color=COLORS[index], lw=1.1
        )
        axes[index, 0].set_title(f"Motor {motor} — comandado ({command})")
        axes[index, 1].set_title(f"Motor {motor} — observado ({observed})")
        axes[index, 0].set_ylabel(f"Motor {motor} [RPM]")
        axes[index, 0].grid(True, alpha=0.25)
        axes[index, 1].grid(True, alpha=0.25)
        axes[index, 0].set_xlim(*xlim)
        axes[index, 1].set_xlim(*xlim)
    axes[-1, 0].set_xlabel("tempo [s]")
    axes[-1, 1].set_xlabel("tempo [s]")
    fig.suptitle(title, fontsize=14)
    fig.savefig(path, dpi=170)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--start", type=float, required=True)
    parser.add_argument("--end", type=float, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument(
        "--plotter",
        type=Path,
        default=Path(
            "/home/gustavokpc/Documents/ESTAG/paparazzi/"
            "sw/tools/logs_plotters/plot_onboard_log.py"
        ),
    )
    args = parser.parse_args()

    plotter = load_plotter(args.plotter)
    _columns, data = plotter.read_csv(args.csv)
    xlim = (args.start, args.end)
    window_mask = (
        np.isfinite(data["time"])
        & (data["time"] >= args.start)
        & (data["time"] <= args.end)
    )
    if not np.any(window_mask):
        raise SystemExit(f"nenhuma amostra dentro da janela {xlim}")
    data = {name: values[window_mask] for name, values in data.items()}
    intervals = clipped_intervals(plotter, data, *xlim)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    window = f"{args.start:g}–{args.end:.3f} s"
    title_prefix = f"{args.csv.name} — janela {window}\n{args.label}"

    state_groups = [
        ("posição [m]", ["pos_x", "pos_y", "pos_z"]),
        ("velocidade [m/s]", ["vel_x", "vel_y", "vel_z"]),
        ("atitude [rad]", ["att_phi", "att_theta", "att_psi"]),
        ("taxas [rad/s]", ["rate_p", "rate_q", "rate_r"]),
        ("RPM observado", [f"rpm_obs_{i}" for i in range(1, 5)]),
        ("RPM de referência", [f"rpm_ref_{i}" for i in range(1, 5)]),
        ("comandos", ["cmd_thrust", "cmd_roll", "cmd_pitch", "cmd_yaw"]),
    ]
    all_states = state_groups[:4]
    body_rates = [("taxas do corpo [rad/s]", ["rate_p", "rate_q", "rate_r"])]
    position = [
        ("x [m]", ["pos_x", "rl_target_x", "rl_err_x"]),
        ("y [m]", ["pos_y", "rl_target_y", "rl_err_y"]),
        ("z [m]", ["pos_z", "rl_target_z", "rl_err_z"]),
    ]
    rpm = [
        (
            f"motor {motor} [RPM]",
            [
                f"rpm_obs_{motor}",
                f"rpm_ref_{motor}",
                f"rl_rpm_cmd{motor}",
                f"rl_rpm_applied{motor}",
            ],
        )
        for motor in range(1, 5)
    ]
    observations = [
        ("RL obs posição", [f"rl_obs_{i}" for i in range(3)]),
        ("RL obs velocidade", [f"rl_obs_{i}" for i in range(3, 6)]),
        ("RL obs atitude", [f"rl_obs_{i}" for i in range(6, 9)]),
        ("RL obs taxas", [f"rl_obs_{i}" for i in range(9, 12)]),
        ("RL obs motor", [f"rl_obs_{i}" for i in range(12, 16)]),
        ("RL obs próximo gate", [f"rl_obs_{i}" for i in range(16, 20)]),
        ("saída bruta da política", [f"rl_policy_raw{i}" for i in range(1, 5)]),
        ("ação RL", [f"rl_action{i}" for i in range(1, 5)]),
        ("comando RL [RPM]", [f"rl_rpm_cmd{i}" for i in range(1, 5)]),
    ]
    timing = [
        ("período [µs]", ["rl_periodic_dt_us"]),
        ("leitura dos sensores [µs]", ["rl_sensor_read_time_us"]),
        ("inferência [µs]", ["rl_inference_time_us"]),
        ("total [µs]", ["rl_total_time_us"]),
    ]
    plots = [
        ("state_commands", state_groups, "Estados, comandos e RPM"),
        ("all_states", all_states, "Todos os estados"),
        ("body_rates", body_rates, "Taxas angulares p, q, r"),
        ("position_target_error", position, "Posição, alvo e erro"),
        ("rpm_commands", rpm, "RPM dos motores"),
        ("rl_observations_outputs", observations, "Observações e saídas da RL"),
        ("rl_timing", timing, "Temporização da RL"),
    ]

    generated: list[tuple[Path, str]] = []
    for suffix, groups, heading in plots:
        output = args.output_dir / f"{args.csv.stem}_{suffix}_30s_to_rl_activation.png"
        plotter.plot_groups(
            output,
            f"{heading}\n{title_prefix}",
            data,
            groups,
            intervals,
            xlim=xlim,
            active_label="RL ativa",
        )
        generated.append((output, heading))

    trajectory = args.output_dir / f"{args.csv.stem}_trajectory_xy_30s_to_rl_activation.png"
    plot_xy(trajectory, f"Trajetória XY\n{title_prefix}", data, xlim)
    generated.append((trajectory, "Trajetória XY"))

    commanded_observed = (
        args.output_dir
        / f"{args.csv.stem}_rpm_commanded_vs_observed_4x2_30s_to_rl_activation.png"
    )
    plot_commanded_observed(
        commanded_observed,
        f"RPM comandado × observado\n{title_prefix}",
        data,
        xlim,
    )
    generated.append((commanded_observed, "RPM comandado × observado"))

    html = [
        "<!doctype html><html><head><meta charset=\"utf-8\">",
        "<style>body{font-family:sans-serif;margin:24px;background:#f7f7f7;color:#222}"
        "img{max-width:100%;background:white;border:1px solid #ccc;margin:8px 0 28px}"
        "</style></head><body>",
        f"<h1>{args.csv.name} — {window}</h1>",
        f"<p><strong>{args.label}</strong></p>",
    ]
    for path, heading in generated:
        html.append(f'<h2>{heading}</h2><img src="{path.name}">')
    html.append("</body></html>")
    (args.output_dir / "index.html").write_text("\n".join(html), encoding="utf-8")

    print(f"Janela: {window}")
    print(f"Intervalos RL dentro da janela: {intervals or 'nenhum'}")
    for path, _heading in generated:
        print(path)


if __name__ == "__main__":
    main()
