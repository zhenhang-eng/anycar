#!/usr/bin/env python3
"""Validate the saved warm-relative OAC direct-center rescore artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from generate_dbm_proposal_teacher import sha256_file


DEFAULT_INPUT = Path("outputs/mppi_proposal/oac_warm_relative_centers_20260827_v1")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_dir", nargs="?", type=Path, default=DEFAULT_INPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = json.loads((args.input_dir / "summary.json").read_text())
    with np.load(args.input_dir / "evaluation.npz", allow_pickle=False) as loaded:
        data = {key: np.asarray(loaded[key]) for key in loaded.files}
    required = {
        "run", "seed", "state_index", "episode", "snapshot", "speed",
        "scenario", "warm_action", "actor_action", "warm_cost", "actor_cost",
        "gain_vs_warm",
    }
    if set(data) != required:
        raise AssertionError(f"evaluation fields differ: {set(data) ^ required}")
    if not np.allclose(
        data["gain_vs_warm"], data["warm_cost"] - data["actor_cost"],
        atol=1e-6, rtol=0,
    ):
        raise AssertionError("gain identity failed")
    if not np.all(np.isfinite(data["warm_cost"])) or not np.all(
        np.isfinite(data["actor_cost"])
    ):
        raise AssertionError("non-finite cost")
    runs = summary["manifest"]["runs"]
    if set(np.unique(data["run"]).tolist()) != set(runs):
        raise AssertionError("run manifest mismatch")
    checks = {}
    for name, manifest in runs.items():
        root = Path(manifest["path"])
        checks[f"{name}_contract_hash"] = (
            sha256_file(root / "contract.json") == manifest["contract_sha256"]
        )
        checks[f"{name}_summary_hash"] = (
            sha256_file(root / "summary.json") == manifest["summary_sha256"]
        )
        mask = data["run"] == name
        if int(np.sum(mask)) != 1800:
            raise AssertionError(f"unexpected row count for {name}")
        gain = data["gain_vs_warm"][mask].astype(np.float64)
        pooled = summary["runs"][name]["pooled"]
        checks[f"{name}_win"] = abs(
            float(np.mean(gain > 1e-6)) - pooled["actor_strict_win_fraction"]
        ) <= 1e-12
        tolerance = 1e-4  # summary is float64; compressed row artifact is float32
        checks[f"{name}_gain_mean"] = abs(
            float(np.mean(gain)) - pooled["gain_vs_warm"]["mean"]
        ) <= tolerance
        checks[f"{name}_gain_median"] = abs(
            float(np.median(gain)) - pooled["gain_vs_warm"]["median"]
        ) <= tolerance
        checks[f"{name}_gain_p05"] = abs(
            float(np.quantile(gain, 0.05)) - pooled["gain_vs_warm"]["p05"]
        ) <= tolerance
        checks[f"{name}_gain_worst"] = abs(
            float(np.min(gain)) - pooled["gain_vs_warm"]["minimum"]
        ) <= tolerance
    if not all(checks.values()):
        raise AssertionError({key: value for key, value in checks.items() if not value})
    report = {
        "qualification": "OAC_WARM_RELATIVE_DIRECT_CENTER_VALIDATION_PASS",
        "checks": checks,
        "summary_sha256": sha256_file(args.input_dir / "summary.json"),
        "evaluation_sha256": sha256_file(args.input_dir / "evaluation.npz"),
    }
    (args.input_dir / "validator_report.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
