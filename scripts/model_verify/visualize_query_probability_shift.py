#!/usr/bin/env python3
"""Render the baseline six-figure suite for Query sim/real probability data."""

import argparse
import json
from pathlib import Path

import numpy as np

import visualize_transformer_probability_error as baseline


CHANNELS = ("dx_body", "dy_body", "dvx", "dyawrate")
CONDITIONS = (
    "simulation_on_simulation",
    "simulation_on_real",
    "adapted_real_on_real",
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--probability-result", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--bins", type=int, default=12)
    return parser.parse_args()


def main():
    args = parse_args()
    result = args.probability_result.resolve()
    output = (args.output_dir or result / "visualization").resolve()
    output.mkdir(parents=True, exist_ok=True)
    data = np.load(result / "probability_error_data.npz")
    names = tuple(str(value) for value in data["channel_names"].tolist())
    if names != CHANNELS:
        raise ValueError(f"Expected channels {CHANNELS}, got {names}")
    raw = {
        condition: {
            kind: data[f"{condition}_{kind}"].astype(np.float64)
            for kind in ("error", "sigma")
        }
        for condition in CONDITIONS
    }

    baseline.CHANNEL_NAMES = CHANNELS
    metrics = baseline.compute_metrics(raw, args.bins)
    baseline.write_metrics(metrics, output)
    baseline.plot_reliability(raw, output)
    baseline.plot_error_bins(raw, args.bins, output)
    baseline.plot_horizon(raw, output)
    baseline.plot_z_histogram(raw, output)
    baseline.plot_selective_risk(raw, output)
    baseline.plot_episode_intervals(raw, output)
    print(json.dumps({"output_dir": str(output), "metrics": metrics}, indent=2))


if __name__ == "__main__":
    main()
