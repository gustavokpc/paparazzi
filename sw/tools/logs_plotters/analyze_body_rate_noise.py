#!/usr/bin/env python3
"""Characterize onboard body-rate noise and its rotor-frequency alias."""

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


AXES = [
    ("rate_p", "p", 0.24, "#2166ac"),
    ("rate_q", "q", 0.12, "#b2182b"),
    ("rate_r", "r", 0.10, "#1b7837"),
]


def read_rows(path: Path, start: float, end: float):
    with path.open(newline="") as stream:
        return [
            row
            for row in csv.DictReader(stream)
            if start <= float(row["time"]) <= end
        ]


def moving_average_residual(values: np.ndarray, samples: int) -> np.ndarray:
    left = samples // 2
    right = samples - 1 - left
    padded = np.pad(values, (left, right), mode="reflect")
    trend = np.convolve(padded, np.ones(samples) / samples, mode="valid")
    return values - trend


def welch(values: np.ndarray, fs: float, segment: int = 512):
    step = segment // 2
    window = np.hanning(segment)
    spectra = []
    for start in range(0, len(values) - segment + 1, step):
        chunk = values[start : start + segment]
        chunk = chunk - np.mean(chunk)
        transform = np.fft.rfft(chunk * window)
        spectra.append(np.abs(transform) ** 2 / (fs * np.sum(window**2)))
    frequency = np.fft.rfftfreq(segment, 1.0 / fs)
    return frequency, np.mean(spectra, axis=0)


def dominant_peak(frequency, psd, minimum_hz=5.0):
    candidates = []
    for index in range(1, len(psd) - 1):
        if (
            frequency[index] >= minimum_hz
            and psd[index] > psd[index - 1]
            and psd[index] >= psd[index + 1]
        ):
            candidates.append(index)
    return max(candidates, key=lambda index: psd[index])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--start", type=float, default=30.0)
    parser.add_argument("--end", type=float, default=45.324368)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = read_rows(args.csv_path, args.start, args.end)
    time = np.asarray([float(row["time"]) for row in rows])
    fs = float(1.0 / np.median(np.diff(time)))
    uniform_time = np.arange(time[0], time[-1], 1.0 / fs)
    rpm = np.column_stack(
        [
            np.asarray([float(row[f"rpm_obs_{motor}"]) for row in rows])
            for motor in range(1, 5)
        ]
    )
    stable = time >= 31.2
    mean_rpm = float(np.mean(rpm[stable]))
    rotor_frequency = mean_rpm / 60.0
    alias_frequency = float(
        abs(rotor_frequency - np.round(rotor_frequency / fs) * fs)
    )

    metrics = {
        "log": args.csv_path.name,
        "window_s": [args.start, args.end],
        "samples": len(rows),
        "sample_rate_hz": fs,
        "nyquist_hz": fs / 2.0,
        "mean_motor_rpm_after_startup": mean_rpm,
        "mean_rotor_frequency_hz": rotor_frequency,
        "predicted_rotor_alias_hz": alias_frequency,
        "configured_simulation_noise_std_rad_s": {
            "p": 0.24,
            "q": 0.12,
            "r": 0.10,
        },
        "axes": {},
    }

    fig, axes = plt.subplots(
        3, 3, figsize=(19, 13), constrained_layout=True
    )
    for row_index, (column, name, configured_std, color) in enumerate(AXES):
        raw = np.asarray([float(row[column]) for row in rows])
        uniform = np.interp(uniform_time, time, raw)
        residual = moving_average_residual(uniform, max(3, round(fs / 5.0)))
        frequency, psd = welch(residual, fs)
        peak_index = dominant_peak(frequency, psd)
        peak_hz = float(frequency[peak_index])

        high_frequency = frequency >= 5.0
        band = (frequency >= 30.0) & (frequency < 35.0)
        high_power = float(np.trapezoid(psd[high_frequency], frequency[high_frequency]))
        band_power = float(np.trapezoid(psd[band], frequency[band]))
        centered = (raw - np.mean(raw)) / np.std(raw)
        stats = {
            "mean_rad_s": float(np.mean(raw)),
            "std_rad_s": float(np.std(raw, ddof=1)),
            "minimum_rad_s": float(np.min(raw)),
            "maximum_rad_s": float(np.max(raw)),
            "skewness": float(np.mean(centered**3)),
            "excess_kurtosis": float(np.mean(centered**4) - 3.0),
            "highpass_residual_std_rad_s": float(np.std(residual, ddof=1)),
            "dominant_frequency_hz": peak_hz,
            "power_fraction_30_35_hz_above_5_hz": band_power / high_power,
        }
        metrics["axes"][name] = stats

        trace_axis = axes[row_index, 0]
        trace_axis.plot(time, raw, color=color, lw=0.75)
        trace_axis.axhline(0.0, color="#333333", lw=0.8)
        trace_axis.set_ylabel(f"{name} [rad/s]")
        trace_axis.set_xlim(args.start, args.end)
        trace_axis.grid(True, alpha=0.25)
        trace_axis.set_title(
            f"{name}: média={stats['mean_rad_s']:.4f}, "
            f"std={stats['std_rad_s']:.4f} rad/s"
        )

        histogram_axis = axes[row_index, 1]
        histogram_axis.hist(
            raw,
            bins=45,
            density=True,
            color=color,
            alpha=0.6,
            label="medido",
        )
        domain = np.linspace(-4.0 * configured_std, 4.0 * configured_std, 500)
        gaussian = np.exp(-0.5 * (domain / configured_std) ** 2) / (
            configured_std * np.sqrt(2.0 * np.pi)
        )
        histogram_axis.plot(
            domain,
            gaussian,
            color="#111111",
            ls="--",
            lw=1.2,
            label=f"Gaussiano σ={configured_std:.2f}",
        )
        histogram_axis.set_xlabel("[rad/s]")
        histogram_axis.set_ylabel("densidade")
        histogram_axis.grid(True, alpha=0.25)
        histogram_axis.legend(fontsize=8)
        histogram_axis.set_title(f"Distribuição de {name}")

        spectrum_axis = axes[row_index, 2]
        spectrum_axis.semilogy(frequency[1:], psd[1:], color=color, lw=1.1)
        spectrum_axis.axvspan(30.0, 35.0, color="#fdb863", alpha=0.22)
        spectrum_axis.axvline(
            peak_hz,
            color="#111111",
            ls="--",
            lw=1.0,
            label=f"pico={peak_hz:.2f} Hz",
        )
        spectrum_axis.axvline(
            alias_frequency,
            color="#984ea3",
            ls=":",
            lw=1.4,
            label=f"alias RPM={alias_frequency:.2f} Hz",
        )
        spectrum_axis.set_xlim(0.0, fs / 2.0)
        spectrum_axis.set_xlabel("frequência [Hz]")
        spectrum_axis.set_ylabel("PSD")
        spectrum_axis.grid(True, alpha=0.25)
        spectrum_axis.legend(fontsize=8)
        spectrum_axis.set_title(
            f"Espectro de {name}: "
            f"{100*stats['power_fraction_30_35_hz_above_5_hz']:.1f}% "
            "da potência >5 Hz em 30–35 Hz"
        )

    axes[-1, 0].set_xlabel("tempo do log [s]")
    fig.suptitle(
        "Caracterização das taxas angulares antes da ativação da RL\n"
        f"{args.csv_path.name} | {args.start:g}–{args.end:.3f} s | "
        f"fs={fs:.4g} Hz | RPM médio={mean_rpm:.1f} | "
        f"alias previsto={alias_frequency:.2f} Hz",
        fontsize=14,
    )
    plot_path = args.output_dir / (
        f"{args.csv_path.stem}_body_rate_noise_{args.start:g}s_to_rl_activation.png"
    )
    fig.savefig(plot_path, dpi=170)
    plt.close(fig)

    json_path = args.output_dir / (
        f"{args.csv_path.stem}_body_rate_noise_metrics.json"
    )
    json_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    text_path = args.output_dir / (
        f"{args.csv_path.stem}_body_rate_noise_metrics.txt"
    )
    lines = [
        f"Log: {args.csv_path.name}",
        f"Window: {args.start:.6f} to {args.end:.6f} s",
        f"Sample rate: {fs:.6f} Hz; Nyquist: {fs/2:.6f} Hz",
        f"Mean motor RPM after startup: {mean_rpm:.3f}",
        f"Rotor frequency: {rotor_frequency:.6f} Hz",
        f"Predicted rotor alias: {alias_frequency:.6f} Hz",
        "",
    ]
    for name in ("p", "q", "r"):
        stats = metrics["axes"][name]
        lines.append(
            f"{name}: mean={stats['mean_rad_s']:.6f}, "
            f"std={stats['std_rad_s']:.6f}, "
            f"dominant={stats['dominant_frequency_hz']:.6f} Hz, "
            f"30-35 Hz power fraction={stats['power_fraction_30_35_hz_above_5_hz']:.6f}"
        )
    text_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(plot_path)
    print(text_path)
    print(json_path)


if __name__ == "__main__":
    main()
