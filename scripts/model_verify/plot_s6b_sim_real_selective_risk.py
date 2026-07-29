#!/usr/bin/env python3
"""Plot S6-B selective risk using the original sim/real protocol."""

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
CHANNELS = ("dx_body", "dy_body", "dvx", "dyawrate")
COLORS = {
    "simulation_on_simulation": "#4C78A8",
    "simulation_on_real": "#E45756",
    "adapted_real_on_real": "#54A24B",
}
LABELS = {
    "simulation_on_simulation": "Sim model / sim test",
    "simulation_on_real": "Sim model / real test",
    "adapted_real_on_real": "Adapted S6-B / real test",
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline-data",
        type=Path,
        default=REPO_ROOT
        / "outputs/formal_query_probability_shift/20260728T144804/probability_error_data.npz",
    )
    parser.add_argument(
        "--s6b-data",
        type=Path,
        default=REPO_ROOT
        / "outputs/formal_real_query_layer_time_risk_s6b/20260728T180142/test_risk_outputs.npz",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT
        / "outputs/formal_real_query_layer_time_risk_s6b/20260728T180142/comparison",
    )
    return parser.parse_args()


def selective_rmse(score, error, fractions):
    score = np.asarray(score, dtype=np.float64).reshape(-1)
    error = np.asarray(error, dtype=np.float64).reshape(-1)
    order = np.argsort(score)
    cumulative_squared_error = np.cumsum(np.square(error[order]))
    counts = np.maximum(
        1, np.minimum(len(order), (fractions * len(order)).astype(int))
    )
    rmse = np.sqrt(cumulative_squared_error[counts - 1] / counts)
    return rmse / max(float(rmse[-1]), 1e-12)


def build_conditions(baseline, s6b, adapted_score):
    return {
        "simulation_on_simulation": {
            "error": baseline["simulation_on_simulation_error"],
            "score": baseline["simulation_on_simulation_sigma"],
        },
        "simulation_on_real": {
            "error": baseline["simulation_on_real_error"],
            "score": baseline["simulation_on_real_sigma"],
        },
        "adapted_real_on_real": {
            "error": s6b["error"],
            "score": s6b[adapted_score],
        },
    }


def plot_conditions(conditions, adapted_label, filename, output):
    fractions = np.linspace(0.05, 1.0, 20)
    fig, axes = plt.subplots(1, 4, figsize=(15.5, 4.0), sharex=True)
    metrics = {}
    for channel_index, (axis, channel) in enumerate(zip(axes, CHANNELS)):
        metrics[channel] = {}
        for condition, arrays in conditions.items():
            curve = selective_rmse(
                arrays["score"][..., channel_index],
                arrays["error"][..., channel_index],
                fractions,
            )
            label = LABELS[condition]
            if condition == "adapted_real_on_real":
                label += f" ({adapted_label})"
            axis.plot(
                fractions,
                curve,
                color=COLORS[condition],
                linewidth=2,
                label=label,
            )
            metrics[channel][condition] = {
                "normalized_aurc": float(
                    np.trapz(curve, fractions) / (fractions[-1] - fractions[0])
                ),
                "rmse_reduction_at_50pct_coverage": float(
                    1.0 - np.interp(0.5, fractions, curve)
                ),
                "rmse_reduction_at_90pct_coverage": float(
                    1.0 - np.interp(0.9, fractions, curve)
                ),
            }
        axis.axhline(1.0, color="black", linestyle="--", linewidth=1)
        axis.set_title(channel)
        axis.set_xlabel("Fraction retained (lowest risk first)")
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("RMSE / full-set RMSE")
    axes[-1].legend(fontsize=7)
    fig.suptitle(
        "Selective risk under the original sim/real protocol "
        f"(adapted score: {adapted_label})"
    )
    fig.tight_layout()
    fig.savefig(output / filename, dpi=180)
    plt.close(fig)
    return metrics


def main():
    args = parse_args()
    baseline = np.load(args.baseline_data.resolve())
    s6b = np.load(args.s6b_data.resolve())
    baseline_channels = tuple(str(value) for value in baseline["channel_names"])
    s6b_channels = tuple(str(value) for value in s6b["channel_names"])
    if baseline_channels != CHANNELS or s6b_channels != CHANNELS:
        raise ValueError(
            f"Channel mismatch: baseline={baseline_channels}, s6b={s6b_channels}"
        )
    maximum_error_difference = float(
        np.max(
            np.abs(
                baseline["adapted_real_on_real_error"].astype(np.float64)
                - s6b["error"].astype(np.float64)
            )
        )
    )
    if maximum_error_difference > 1e-8:
        raise ValueError(
            "Adapted real-test errors do not align: "
            f"max difference={maximum_error_difference}"
        )
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    results = {
        "protocol": "original_query_sim_real_selective_risk_with_s6b_v1",
        "baseline_data": str(args.baseline_data.resolve()),
        "s6b_data": str(args.s6b_data.resolve()),
        "adapted_real_error_max_difference": maximum_error_difference,
        "metric": "RMSE divided by full-condition RMSE",
        "sigma": plot_conditions(
            build_conditions(baseline, s6b, "u2_sigma"),
            "sigma",
            "05_selective_risk_sim_real_s6b_sigma.png",
            output,
        ),
        "expected_abs_error": plot_conditions(
            build_conditions(baseline, s6b, "u2_expected_abs"),
            "expected |error|",
            "06_selective_risk_sim_real_s6b_expected_error.png",
            output,
        ),
    }
    (output / "sim_real_selective_risk_metrics.json").write_text(
        json.dumps(results, indent=2) + "\n"
    )
    print(json.dumps({"output": str(output), "results": results}, indent=2))


if __name__ == "__main__":
    main()
