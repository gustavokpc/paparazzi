# Logs Plotters

Scripts for plotting the two log formats used in the NN CFC experiments.

## Directory Layout

From the Paparazzi repository root:

```bash
var/logs/log_onboard/       # CSV files copied directly from the drone
var/logs/               # Paparazzi GCS .data/.log files
plots/plots_onboard/    # plots generated from onboard CSV logs
plots/plots_paparazzi/  # plots generated from Paparazzi .data logs
```

## Onboard CSV Logs

Put drone-extracted CSV files in:

```bash
var/logs/log_onboard/
```

Generate complete plots for one or more onboard CSVs:

```bash
python3 sw/tools/logs_plotters/plot_onboard_log.py var/logs/log_onboard/20260622-085237.csv
```

Generate plots for all onboard CSVs:

```bash
python3 sw/tools/logs_plotters/plot_onboard_log.py var/logs/log_onboard/*.csv
```

Outputs are saved by default to:

```bash
plots/plots_onboard/<csv_name>/
```

Each output folder includes an `index.html` for easier browsing.

For old CSV logs that do not contain an explicit `nn_enabled` column, the orange
bands mark **inferred** NN activity. The inference is based on NN-specific fields
becoming non-zero:

```text
nn_in_*
network_out*_norm
rpm_cmd*
```

## Paparazzi `.data` Logs

Generate NN CFC plots from a Paparazzi GCS `.data` log:

```bash
python3 sw/tools/logs_plotters/plot_nn_cfc_log.py var/logs/26_06_22__08_52_48.data
```

Outputs are saved by default to:

```bash
plots/plots_paparazzi/<log_name>/
```

Generate only motor RPM plots:

```bash
python3 sw/tools/logs_plotters/plot_motor_rpms.py var/logs/26_06_22__08_52_48.data --source nn-input
```

For onboard CSVs, the same RPM plotter can overlay sources:

```bash
python3 sw/tools/logs_plotters/plot_motor_rpms.py var/logs/log_onboard/20260622-085237.csv --source obs --source ref --source cmd
```

## Useful Options

Keep absolute timestamps instead of resetting the active segment to zero:

```bash
python3 sw/tools/logs_plotters/plot_nn_cfc_log.py var/logs/26_06_22__08_52_48.data --absolute-time
```

Change the output folder:

```bash
python3 sw/tools/logs_plotters/plot_onboard_log.py var/logs/log_onboard/*.csv --output-dir plots/plots_onboard/my_run
```

Change the active-motor zoom threshold for onboard plots:

```bash
python3 sw/tools/logs_plotters/plot_onboard_log.py var/logs/log_onboard/20260622-085237.csv --active-rpm-threshold 5000
```

## What To Compare

For manual/control-standard baseline:

```text
cmd_thrust, cmd_roll, cmd_pitch, cmd_yaw
rpm_ref_1..4
rpm_obs_1..4
rate_p, rate_q, rate_r
att_phi, att_theta, att_psi
```

For NN-controlled segments:

```text
nn_in_*_raw
nn_in_*_normalized
network_out*_norm
rpm_cmd1..4
rpm_ref_1..4
rpm_obs_1..4
rate_p, rate_q, rate_r
```
