#!/usr/bin/env python3
"""Create baseline-style uncertainty figures for a nuPlan Query residual run."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import kurtosis, norm, rankdata, spearmanr


CHANNEL_NAMES = ("dx_body", "dy_body", "dvx", "dyawrate")
CONDITION_LABELS = {
    "learned_raw": "Raw learned sigma",
    "global_temperature": "Global calibrated",
    "horizon_temperature": "Horizon calibrated",
}
COLORS = {
    "learned_raw": "#E45756",
    "global_temperature": "#F2CF5B",
    "horizon_temperature": "#54A24B",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize uncertainty/error association for a Query residual model."
    )
    parser.add_argument("--large-holdout-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--bins", type=int, default=12)
    return parser.parse_args()


def equal_count_bins(sigma, error, bins):
    groups = np.array_split(np.argsort(sigma), bins)
    predicted = np.array(
        [np.sqrt(np.mean(np.square(sigma[group]))) for group in groups]
    )
    observed = np.array(
        [np.sqrt(np.mean(np.square(error[group]))) for group in groups]
    )
    return predicted, observed


def binary_auc(score, target):
    target = np.asarray(target, dtype=bool)
    positives = int(target.sum())
    negatives = int((~target).sum())
    if positives == 0 or negatives == 0:
        return float("nan")
    ranks = rankdata(score)
    rank_sum = float(ranks[target].sum())
    return (
        rank_sum - positives * (positives + 1) / 2
    ) / (positives * negatives)


def channel_metrics(error, sigma, bins):
    absolute_error = np.abs(error)
    z = error / sigma
    predicted, observed = equal_count_bins(sigma, error, bins)
    nominal = np.linspace(0.05, 0.99, 20)
    threshold = norm.ppf((1.0 + nominal) / 2.0)
    empirical = np.array(
        [np.mean(np.abs(z) <= value) for value in threshold]
    )
    high_error = absolute_error >= np.quantile(absolute_error, 0.90)
    high_sigma = sigma >= np.quantile(sigma, 0.90)
    return {
        "samples": int(error.size),
        "rmse": float(np.sqrt(np.mean(np.square(error)))),
        "mae": float(np.mean(absolute_error)),
        "mean_sigma": float(np.mean(sigma)),
        "rms_sigma": float(np.sqrt(np.mean(np.square(sigma)))),
        "pearson_sigma_abs_error": float(
            np.corrcoef(sigma, absolute_error)[0, 1]
        ),
        "spearman_sigma_abs_error": float(
            spearmanr(sigma, absolute_error).statistic
        ),
        "ence": float(
            np.mean(
                np.abs(observed - predicted)
                / np.maximum(predicted, 1e-12)
            )
        ),
        "calibration_mae": float(np.mean(np.abs(empirical - nominal))),
        "top10_error_recall_by_top10_sigma": float(
            np.sum(high_error & high_sigma) / np.sum(high_error)
        ),
        "top10_error_auc": float(binary_auc(sigma, high_error)),
        "z_mean": float(np.mean(z)),
        "z_rms": float(np.sqrt(np.mean(np.square(z)))),
        "z_excess_kurtosis": float(kurtosis(z, fisher=True, bias=False)),
        "coverage_68": float(np.mean(np.abs(z) <= 1.0)),
        "coverage_90": float(
            np.mean(np.abs(z) <= 1.6448536269514722)
        ),
        "coverage_95": float(
            np.mean(np.abs(z) <= 1.959963984540054)
        ),
    }


def compute_metrics(raw, bins):
    return {
        condition: {
            channel: channel_metrics(
                arrays["error"][..., index].reshape(-1),
                arrays["sigma"][..., index].reshape(-1),
                bins,
            )
            for index, channel in enumerate(CHANNEL_NAMES)
        }
        for condition, arrays in raw.items()
    }


def plot_reliability(raw, output):
    nominal = np.linspace(0.05, 0.99, 20)
    threshold = norm.ppf((1.0 + nominal) / 2.0)
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.8), sharex=True, sharey=True)
    for index, (axis, channel) in enumerate(zip(axes, CHANNEL_NAMES)):
        axis.plot(nominal, nominal, "k--", linewidth=1, label="Ideal")
        for condition, arrays in raw.items():
            z = (
                arrays["error"][..., index].reshape(-1)
                / arrays["sigma"][..., index].reshape(-1)
            )
            empirical = [
                np.mean(np.abs(z) <= value) for value in threshold
            ]
            axis.plot(
                nominal,
                empirical,
                color=COLORS[condition],
                label=CONDITION_LABELS[condition],
            )
        axis.set_title(channel)
        axis.set_xlabel("Nominal coverage")
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("Empirical coverage")
    axes[-1].legend(fontsize=8, loc="lower right")
    fig.suptitle("Query residual reliability: predicted intervals vs errors")
    fig.tight_layout()
    fig.savefig(output / "01_reliability.png", dpi=180)
    plt.close(fig)


def plot_error_bins(raw, bins, output):
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.8))
    for index, (axis, channel) in enumerate(zip(axes, CHANNEL_NAMES)):
        maximum = 0.0
        for condition, arrays in raw.items():
            predicted, observed = equal_count_bins(
                arrays["sigma"][..., index].reshape(-1),
                arrays["error"][..., index].reshape(-1),
                bins,
            )
            maximum = max(
                maximum, float(predicted.max()), float(observed.max())
            )
            axis.plot(
                predicted,
                observed,
                marker="o",
                markersize=3,
                color=COLORS[condition],
                label=CONDITION_LABELS[condition],
            )
        axis.plot(
            [0, maximum], [0, maximum], "k--", linewidth=1, label="Ideal"
        )
        axis.set_title(channel)
        axis.set_xlabel("Predicted RMS sigma")
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("Observed RMSE in sigma bin")
    axes[-1].legend(fontsize=8)
    fig.suptitle("Query residual uncertainty-error association")
    fig.tight_layout()
    fig.savefig(output / "02_sigma_error_bins.png", dpi=180)
    plt.close(fig)


def plot_horizon(raw, output):
    horizon = np.arange(1, next(iter(raw.values()))["error"].shape[1] + 1)
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.8))
    for index, (axis, channel) in enumerate(zip(axes, CHANNEL_NAMES)):
        error = next(iter(raw.values()))["error"][..., index]
        observed = np.sqrt(np.mean(np.square(error), axis=0))
        axis.plot(
            horizon,
            observed,
            color="#4C78A8",
            linewidth=2,
            label="Observed RMSE",
        )
        for condition, arrays in raw.items():
            predicted = np.sqrt(
                np.mean(np.square(arrays["sigma"][..., index]), axis=0)
            )
            axis.plot(
                horizon,
                predicted,
                color=COLORS[condition],
                label=CONDITION_LABELS[condition],
            )
        axis.set_title(channel)
        axis.set_xlabel("Prediction horizon")
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("Physical scale")
    axes[-1].legend(fontsize=8)
    fig.suptitle("Query residual horizon calibration")
    fig.tight_layout()
    fig.savefig(output / "03_horizon_calibration.png", dpi=180)
    plt.close(fig)


def plot_z_histogram(raw, output):
    arrays = raw["horizon_temperature"]
    x = np.linspace(-5, 5, 500)
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.8), sharey=True)
    for index, (axis, channel) in enumerate(zip(axes, CHANNEL_NAMES)):
        z = (
            arrays["error"][..., index].reshape(-1)
            / arrays["sigma"][..., index].reshape(-1)
        )
        axis.hist(
            z,
            bins=160,
            range=(-5, 5),
            density=True,
            alpha=0.65,
            color=COLORS["horizon_temperature"],
        )
        axis.plot(x, norm.pdf(x), "k--", linewidth=1.2, label="N(0, 1)")
        axis.set_yscale("log")
        axis.set_ylim(1e-4, 2)
        axis.set_title(channel)
        axis.set_xlabel("Standardized residual z")
        axis.grid(alpha=0.2)
    axes[0].set_ylabel("Density (log scale)")
    axes[-1].legend(fontsize=8)
    fig.suptitle("Query residual standardized-error distribution")
    fig.tight_layout()
    fig.savefig(output / "04_standardized_residual_histogram.png", dpi=180)
    plt.close(fig)


def selective_risk(sigma, error):
    order = np.argsort(sigma)
    fractions = np.linspace(0.05, 1.0, 20)
    risk = []
    for fraction in fractions:
        count = max(1, int(len(order) * fraction))
        risk.append(np.sqrt(np.mean(np.square(error[order[:count]]))))
    return fractions, np.asarray(risk)


def plot_selective_risk(raw, output):
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.8), sharex=True)
    for index, (axis, channel) in enumerate(zip(axes, CHANNEL_NAMES)):
        for condition, arrays in raw.items():
            fractions, risk = selective_risk(
                arrays["sigma"][..., index].reshape(-1),
                arrays["error"][..., index].reshape(-1),
            )
            axis.plot(
                fractions,
                risk / risk[-1],
                color=COLORS[condition],
                label=CONDITION_LABELS[condition],
            )
        axis.axhline(1.0, color="k", linestyle="--", linewidth=1)
        axis.set_title(channel)
        axis.set_xlabel("Fraction retained (lowest sigma first)")
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("RMSE / full-set RMSE")
    axes[-1].legend(fontsize=8)
    fig.suptitle("Query residual selective risk")
    fig.tight_layout()
    fig.savefig(output / "05_selective_risk.png", dpi=180)
    plt.close(fig)


def plot_episode_intervals(raw, output):
    arrays = raw["horizon_temperature"]
    z = arrays["error"] / arrays["sigma"]
    episode_score = np.sqrt(np.mean(np.square(z), axis=(1, 2)))
    order = np.argsort(episode_score)
    selected = (order[len(order) // 2], order[int(len(order) * 0.95)])
    titles = (
        "Median standardized-error episode",
        "95th-percentile standardized-error episode",
    )
    horizon = np.arange(1, arrays["error"].shape[1] + 1)
    fig, axes = plt.subplots(4, 2, figsize=(15, 12), sharex=True)
    for row, channel in enumerate(CHANNEL_NAMES):
        for column, episode in enumerate(selected):
            axis = axes[row, column]
            error = arrays["error"][episode, :, row]
            sigma = arrays["sigma"][episode, :, row]
            axis.fill_between(
                horizon,
                -2 * sigma,
                2 * sigma,
                color="#4C78A8",
                alpha=0.15,
                label="±2 sigma",
            )
            axis.fill_between(
                horizon,
                -sigma,
                sigma,
                color="#4C78A8",
                alpha=0.30,
                label="±1 sigma",
            )
            axis.plot(
                horizon,
                error,
                color="#E45756",
                linewidth=1.2,
                label="Target - prediction",
            )
            axis.axhline(0.0, color="k", linewidth=0.8)
            axis.set_ylabel(channel)
            axis.grid(alpha=0.2)
            if row == 0:
                axis.set_title(f"{titles[column]} (index {episode})")
            if row == len(CHANNEL_NAMES) - 1:
                axis.set_xlabel("Prediction horizon")
    axes[0, 1].legend(fontsize=8, loc="upper right")
    fig.suptitle("Query residual inside calibrated probability intervals")
    fig.tight_layout()
    fig.savefig(output / "06_episode_error_intervals.png", dpi=180)
    plt.close(fig)


def write_metrics(metrics, output):
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    fields = [
        "condition",
        "channel",
        *next(iter(next(iter(metrics.values())).values())).keys(),
    ]
    with (output / "metrics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for condition, channels in metrics.items():
            for channel, values in channels.items():
                writer.writerow(
                    {"condition": condition, "channel": channel, **values}
                )


def main():
    args = parse_args()
    run_dir = args.large_holdout_dir.resolve()
    output = (args.output_dir or run_dir / "baseline_style_visualization").resolve()
    output.mkdir(parents=True, exist_ok=True)

    summary = json.loads((run_dir / "summary.json").read_text())
    data = np.load(run_dir / "confirmation_data.npz")
    error = data["confirm_test_error"].astype(np.float64)
    learned_sigma = data["confirm_test_sigma"].astype(np.float64)
    global_temperature = np.asarray(
        summary["global_temperature"], dtype=np.float64
    )
    horizon_temperature = np.asarray(
        summary["horizon_temperature"]["values"], dtype=np.float64
    )

    checkpoint = torch.load(
        summary["mean_checkpoint"], map_location="cpu", weights_only=False
    )
    residual_std = (
        checkpoint["stats"]["residual"][1]
        .detach()
        .cpu()
        .numpy()
        .astype(np.float64)
    )
    physical_error = error * residual_std.reshape(1, 1, -1)
    physical_sigma = learned_sigma * residual_std.reshape(1, 1, -1)
    raw = {
        "learned_raw": {
            "error": physical_error,
            "sigma": physical_sigma,
        },
        "global_temperature": {
            "error": physical_error,
            "sigma": physical_sigma * global_temperature.reshape(1, 1, -1),
        },
        "horizon_temperature": {
            "error": physical_error,
            "sigma": physical_sigma * horizon_temperature.reshape(
                1, *horizon_temperature.shape
            ),
        },
    }

    metrics = compute_metrics(raw, args.bins)
    write_metrics(metrics, output)
    np.savez_compressed(
        output / "probability_error_data.npz",
        error=physical_error,
        sigma_raw=raw["learned_raw"]["sigma"],
        sigma_global=raw["global_temperature"]["sigma"],
        sigma_horizon=raw["horizon_temperature"]["sigma"],
        channel_names=np.asarray(CHANNEL_NAMES),
    )
    plot_reliability(raw, output)
    plot_error_bins(raw, args.bins, output)
    plot_horizon(raw, output)
    plot_z_histogram(raw, output)
    plot_selective_risk(raw, output)
    plot_episode_intervals(raw, output)
    print(json.dumps({"output_dir": str(output), "metrics": metrics}, indent=2))


if __name__ == "__main__":
    main()
