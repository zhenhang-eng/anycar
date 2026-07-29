#!/usr/bin/env python3
"""Visualize how Transformer uncertainty relates to prediction error."""

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import kurtosis, norm, rankdata, spearmanr
from torch import nn


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "car_foundation"))

from scripts.model_verify.analyze_transformer_probability_shift import (
    CHANNEL_INDICES,
    CHANNEL_NAMES,
    collect,
    load_run,
    make_loader,
)


CONDITION_LABELS = {
    "simulation_on_simulation": "Sim model / sim test",
    "simulation_on_real": "Sim model / real test",
    "adapted_real_on_real": "Adapted model / real test",
}
COLORS = {
    "simulation_on_simulation": "#4C78A8",
    "simulation_on_real": "#E45756",
    "adapted_real_on_real": "#54A24B",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--probability-result", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--bins", type=int, default=12)
    parser.add_argument("--sigma-floor", type=float, default=1e-3)
    return parser.parse_args()


def load_head(state, prefix, device):
    head = nn.Linear(256, len(CHANNEL_INDICES)).to(device)
    head.load_state_dict(state[f"{prefix}_sigma_head"])
    head.eval()
    temperature = state[f"{prefix}_temperature"].to(device)
    return head, temperature


def evaluate_raw(system, head, temperature, files, args, device):
    loader = make_loader(files, args.batch_size, False)
    error, sigma, _ = collect(system, head, loader, device, args.sigma_floor)
    sigma = sigma * temperature.cpu().view(1, 50, len(CHANNEL_INDICES))
    indices = torch.tensor(CHANNEL_INDICES, device=system["std"].device)
    physical_std = system["std"].index_select(0, indices).cpu().view(1, 1, -1)
    return {
        "error": (error * physical_std).numpy(),
        "sigma": (sigma * physical_std).numpy(),
    }


def equal_count_bins(sigma, error, bins):
    order = np.argsort(sigma)
    groups = np.array_split(order, bins)
    predicted = np.array(
        [np.sqrt(np.mean(np.square(sigma[group]))) for group in groups]
    )
    observed = np.array(
        [np.sqrt(np.mean(np.square(error[group]))) for group in groups]
    )
    counts = np.array([len(group) for group in groups])
    return predicted, observed, counts


def binary_auc(score, target):
    target = np.asarray(target, dtype=bool)
    positives = int(target.sum())
    negatives = int((~target).sum())
    if positives == 0 or negatives == 0:
        return float("nan")
    ranks = rankdata(score)
    rank_sum = float(ranks[target].sum())
    return (rank_sum - positives * (positives + 1) / 2) / (positives * negatives)


def channel_metrics(error, sigma, bins):
    absolute_error = np.abs(error)
    z = error / sigma
    predicted, observed, _ = equal_count_bins(sigma, error, bins)
    nominal = np.linspace(0.05, 0.99, 20)
    threshold = norm.ppf((1.0 + nominal) / 2.0)
    empirical = np.array([np.mean(np.abs(z) <= value) for value in threshold])
    high_error = absolute_error >= np.quantile(absolute_error, 0.90)
    high_sigma = sigma >= np.quantile(sigma, 0.90)
    pearson = np.corrcoef(sigma, absolute_error)[0, 1]
    spearman = spearmanr(sigma, absolute_error).statistic
    return {
        "samples": int(error.size),
        "rmse": float(np.sqrt(np.mean(np.square(error)))),
        "mae": float(np.mean(absolute_error)),
        "mean_sigma": float(np.mean(sigma)),
        "rms_sigma": float(np.sqrt(np.mean(np.square(sigma)))),
        "pearson_sigma_abs_error": float(pearson),
        "spearman_sigma_abs_error": float(spearman),
        "ence": float(np.mean(np.abs(observed - predicted) / np.maximum(predicted, 1e-12))),
        "calibration_mae": float(np.mean(np.abs(empirical - nominal))),
        "top10_error_recall_by_top10_sigma": float(np.sum(high_error & high_sigma) / np.sum(high_error)),
        "top10_error_auc": float(binary_auc(sigma, high_error)),
        "z_mean": float(np.mean(z)),
        "z_rms": float(np.sqrt(np.mean(np.square(z)))),
        "z_excess_kurtosis": float(kurtosis(z, fisher=True, bias=False)),
        "coverage_68": float(np.mean(np.abs(z) <= 1.0)),
        "coverage_90": float(np.mean(np.abs(z) <= 1.6448536269514722)),
        "coverage_95": float(np.mean(np.abs(z) <= 1.959963984540054)),
    }


def compute_metrics(raw, bins):
    result = {}
    for condition, arrays in raw.items():
        result[condition] = {}
        for index, channel in enumerate(CHANNEL_NAMES):
            result[condition][channel] = channel_metrics(
                arrays["error"][..., index].reshape(-1),
                arrays["sigma"][..., index].reshape(-1),
                bins,
            )
    return result


def plot_reliability(raw, output):
    nominal = np.linspace(0.05, 0.99, 20)
    threshold = norm.ppf((1.0 + nominal) / 2.0)
    channel_count = len(CHANNEL_NAMES)
    fig, axes = plt.subplots(
        1,
        channel_count,
        figsize=(3.8 * channel_count, 3.8),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    axes = axes[0]
    for index, (axis, channel) in enumerate(zip(axes, CHANNEL_NAMES)):
        axis.plot(nominal, nominal, "k--", linewidth=1, label="Ideal")
        for condition, arrays in raw.items():
            z = arrays["error"][..., index].reshape(-1) / arrays["sigma"][..., index].reshape(-1)
            empirical = [np.mean(np.abs(z) <= value) for value in threshold]
            axis.plot(nominal, empirical, color=COLORS[condition], label=CONDITION_LABELS[condition])
        axis.set_title(channel)
        axis.grid(alpha=0.25)
        axis.set_xlabel("Nominal coverage")
    axes[0].set_ylabel("Empirical coverage")
    axes[-1].legend(fontsize=8, loc="lower right")
    fig.suptitle("Reliability diagram: predicted intervals vs observed errors")
    fig.tight_layout()
    fig.savefig(output / "01_reliability.png", dpi=180)
    plt.close(fig)


def plot_error_bins(raw, bins, output):
    channel_count = len(CHANNEL_NAMES)
    fig, axes = plt.subplots(
        1,
        channel_count,
        figsize=(3.8 * channel_count, 3.8),
        squeeze=False,
    )
    axes = axes[0]
    for index, (axis, channel) in enumerate(zip(axes, CHANNEL_NAMES)):
        maximum = 0.0
        for condition, arrays in raw.items():
            predicted, observed, _ = equal_count_bins(
                arrays["sigma"][..., index].reshape(-1),
                arrays["error"][..., index].reshape(-1),
                bins,
            )
            maximum = max(maximum, float(predicted.max()), float(observed.max()))
            axis.plot(
                predicted,
                observed,
                marker="o",
                markersize=3,
                color=COLORS[condition],
                label=CONDITION_LABELS[condition],
            )
        axis.plot([0, maximum], [0, maximum], "k--", linewidth=1, label="Ideal")
        axis.set_title(channel)
        axis.set_xlabel("Predicted RMS sigma")
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("Observed RMSE in sigma bin")
    axes[-1].legend(fontsize=8)
    fig.suptitle("Uncertainty-error association (equal-count sigma bins)")
    fig.tight_layout()
    fig.savefig(output / "02_sigma_error_bins.png", dpi=180)
    plt.close(fig)


def plot_horizon(raw, output):
    for condition, arrays in raw.items():
        channel_count = len(CHANNEL_NAMES)
        fig, axes = plt.subplots(
            1,
            channel_count,
            figsize=(3.8 * channel_count, 3.8),
            squeeze=False,
        )
        axes = axes[0]
        horizon = np.arange(1, arrays["error"].shape[1] + 1)
        for index, (axis, channel) in enumerate(zip(axes, CHANNEL_NAMES)):
            rmse = np.sqrt(np.mean(np.square(arrays["error"][..., index]), axis=0))
            rms_sigma = np.sqrt(np.mean(np.square(arrays["sigma"][..., index]), axis=0))
            axis.plot(horizon, rmse, label="Observed RMSE", color="#E45756")
            axis.plot(horizon, rms_sigma, label="Predicted RMS sigma", color="#4C78A8")
            axis.set_title(channel)
            axis.set_xlabel("Prediction horizon")
            axis.grid(alpha=0.25)
        axes[0].set_ylabel("Physical scale")
        axes[-1].legend(fontsize=8)
        fig.suptitle(f"Horizon calibration: {CONDITION_LABELS[condition]}")
        fig.tight_layout()
        fig.savefig(output / f"03_horizon_{condition}.png", dpi=180)
        plt.close(fig)


def plot_z_histogram(raw, output):
    condition = "adapted_real_on_real"
    arrays = raw[condition]
    x = np.linspace(-5, 5, 500)
    channel_count = len(CHANNEL_NAMES)
    fig, axes = plt.subplots(
        1,
        channel_count,
        figsize=(3.8 * channel_count, 3.8),
        sharey=True,
        squeeze=False,
    )
    axes = axes[0]
    for index, (axis, channel) in enumerate(zip(axes, CHANNEL_NAMES)):
        z = arrays["error"][..., index].reshape(-1) / arrays["sigma"][..., index].reshape(-1)
        axis.hist(z, bins=160, range=(-5, 5), density=True, alpha=0.65, color="#54A24B")
        axis.plot(x, norm.pdf(x), "k--", linewidth=1.2, label="N(0, 1)")
        axis.set_yscale("log")
        axis.set_ylim(1e-4, 2)
        axis.set_title(channel)
        axis.set_xlabel("Standardized residual z")
        axis.grid(alpha=0.2)
    axes[0].set_ylabel("Density (log scale)")
    axes[-1].legend(fontsize=8)
    fig.suptitle("Standardized residual distribution: adapted model / real test")
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
    channel_count = len(CHANNEL_NAMES)
    fig, axes = plt.subplots(
        1,
        channel_count,
        figsize=(3.8 * channel_count, 3.8),
        sharex=True,
        squeeze=False,
    )
    axes = axes[0]
    for index, (axis, channel) in enumerate(zip(axes, CHANNEL_NAMES)):
        for condition in ("simulation_on_real", "adapted_real_on_real"):
            arrays = raw[condition]
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
    fig.suptitle("Selective risk: does low predicted uncertainty identify easier samples?")
    fig.tight_layout()
    fig.savefig(output / "05_selective_risk.png", dpi=180)
    plt.close(fig)


def plot_episode_intervals(raw, output):
    arrays = raw["adapted_real_on_real"]
    z = arrays["error"] / arrays["sigma"]
    episode_score = np.sqrt(np.mean(np.square(z), axis=(1, 2)))
    order = np.argsort(episode_score)
    selected = (order[len(order) // 2], order[int(len(order) * 0.95)])
    titles = ("Median standardized-error episode", "95th-percentile error episode")
    horizon = np.arange(1, arrays["error"].shape[1] + 1)
    fig, axes = plt.subplots(
        len(CHANNEL_NAMES),
        2,
        figsize=(15, 3 * len(CHANNEL_NAMES)),
        sharex=True,
        squeeze=False,
    )
    for row, channel in enumerate(CHANNEL_NAMES):
        for column, episode in enumerate(selected):
            axis = axes[row, column]
            error = arrays["error"][episode, :, row]
            sigma = arrays["sigma"][episode, :, row]
            axis.fill_between(horizon, -2 * sigma, 2 * sigma, color="#4C78A8", alpha=0.15, label="±2 sigma")
            axis.fill_between(horizon, -sigma, sigma, color="#4C78A8", alpha=0.30, label="±1 sigma")
            axis.plot(horizon, error, color="#E45756", linewidth=1.2, label="Target - prediction")
            axis.axhline(0.0, color="k", linewidth=0.8)
            axis.grid(alpha=0.2)
            axis.set_ylabel(channel)
            if row == 0:
                axis.set_title(f"{titles[column]} (index {episode})")
            if row == len(CHANNEL_NAMES) - 1:
                axis.set_xlabel("Prediction horizon")
    axes[0, 1].legend(fontsize=8, loc="upper right")
    fig.suptitle("Prediction residual inside model probability intervals")
    fig.tight_layout()
    fig.savefig(output / "06_episode_error_intervals.png", dpi=180)
    plt.close(fig)


def write_metrics(metrics, output):
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    fieldnames = ["condition", "channel", *next(iter(next(iter(metrics.values())).values())).keys()]
    with (output / "metrics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for condition, channels in metrics.items():
            for channel, values in channels.items():
                writer.writerow({"condition": condition, "channel": channel, **values})


def main():
    args = parse_args()
    result_dir = args.probability_result.resolve()
    output = (args.output_dir or result_dir / "visualization").resolve()
    output.mkdir(parents=True, exist_ok=True)
    summary = json.loads((result_dir / "summary.json").read_text())
    device = torch.device("cuda")
    simulation = load_run(summary["args"]["simulation_run"], device)
    real = load_run(summary["args"]["real_run"], device)
    state = torch.load(result_dir / "probability_heads.pt", map_location=device, weights_only=False)
    sim_head, sim_temperature = load_head(state, "simulation", device)
    real_head, real_temperature = load_head(state, "real", device)

    conditions = (
        ("simulation_on_simulation", simulation, sim_head, sim_temperature, simulation["test_files"]),
        ("simulation_on_real", simulation, sim_head, sim_temperature, real["test_files"]),
        ("adapted_real_on_real", real, real_head, real_temperature, real["test_files"]),
    )
    raw = {}
    for name, system, head, temperature, files in conditions:
        print(f"Collecting {name}: {len(files)} files", flush=True)
        raw[name] = evaluate_raw(system, head, temperature, files, args, device)

    metrics = compute_metrics(raw, args.bins)
    write_metrics(metrics, output)
    np.savez_compressed(
        output / "probability_error_data.npz",
        **{
            f"{condition}_{kind}": values
            for condition, arrays in raw.items()
            for kind, values in arrays.items()
        },
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
