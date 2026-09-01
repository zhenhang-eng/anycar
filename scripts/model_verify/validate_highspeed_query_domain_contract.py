#!/usr/bin/env python3
"""Independently validate the high-speed frozen-Query domain audit artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch


DEFAULT_ANALYSIS = Path(
    "outputs/query_mppi/highspeed_query_domain_audit_20260831_v3/analysis.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", type=Path, default=DEFAULT_ANALYSIS)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def close(name: str, actual: float, recorded: float, atol: float = 1e-6) -> dict:
    difference = abs(float(actual) - float(recorded))
    return {
        "name": name,
        "actual": float(actual),
        "recorded": float(recorded),
        "abs_difference": difference,
        "pass": difference <= atol,
    }


def main() -> None:
    args = parse_args()
    analysis = json.loads(args.analysis.read_text())
    inputs = analysis["inputs"]
    checks: list[dict] = []

    for name in (
        "checkpoint", "onnx", "train_summary", "metadata", "replay", "replay_summary"
    ):
        path = Path(inputs[name])
        actual_hash = sha256(path)
        checks.append({
            "name": f"sha256:{name}",
            "actual": actual_hash,
            "recorded": inputs[f"{name}_sha256"],
            "pass": actual_hash == inputs[f"{name}_sha256"],
        })

    checkpoint = torch.load(inputs["checkpoint"], map_location="cpu")
    metadata = json.loads(Path(inputs["metadata"]).read_text())
    with np.load(inputs["replay"], allow_pickle=False) as loaded:
        state_six = np.asarray(loaded["state_six"], np.float64)
        history = np.asarray(loaded["history"], np.float64)
        candidates = np.asarray(loaded["sampled_action_sequences"], np.float64)

    history_mean = np.asarray(checkpoint["stats"]["history"][0], np.float64)
    history_std = np.asarray(checkpoint["stats"]["history"][1], np.float64)
    context_mean = np.asarray(checkpoint["stats"]["context"][0], np.float64)
    context_std = np.asarray(checkpoint["stats"]["context"][1], np.float64)
    vx = state_six[:, 3]
    history_dx_abs_z = np.abs((history[..., 0] - history_mean[0]) / history_std[0])
    context_vx_abs_z = np.abs((vx - context_mean[0]) / context_std[0])

    state_record = analysis["state_support"]["vx"]
    checks.extend((
        close("current_vx_min", vx.min(), state_record["current"]["min"]),
        close("current_vx_median", np.median(vx), state_record["current"]["median"]),
        close("current_vx_max", vx.max(), state_record["current"]["max"]),
        close(
            "current_vx_outside_training_fraction",
            np.mean((vx < state_record["training_min"]) | (vx > state_record["training_max"])),
            state_record["current_fraction_outside_training_minmax"],
        ),
        close(
            "history_dx_abs_z_p95",
            np.quantile(history_dx_abs_z, 0.95),
            analysis["normalized_input_audit"]["history"]["dx_body"]["abs_z"]["p95"],
        ),
        close(
            "context_vx_abs_z_p95",
            np.quantile(context_vx_abs_z, 0.95),
            analysis["normalized_input_audit"]["context_all_64_candidates"]["vx"]["abs_z"]["p95"],
        ),
    ))

    physical_match = all(item["match"] for item in analysis["physical_parameter_checks"])
    checks.extend((
        {
            "name": "all_physical_contract_checks",
            "actual": physical_match,
            "recorded": analysis["matched_contracts"]["all_vehicle_and_dt_checks"],
            "pass": physical_match == analysis["matched_contracts"]["all_vehicle_and_dt_checks"],
        },
        {
            "name": "history_shape",
            "actual": list(history.shape),
            "recorded": analysis["current_contract"]["history_shape"],
            "pass": list(history.shape) == analysis["current_contract"]["history_shape"],
        },
        {
            "name": "candidate_shape",
            "actual": list(candidates.shape),
            "recorded": analysis["current_contract"]["candidate_action_shape"],
            "pass": list(candidates.shape) == analysis["current_contract"]["candidate_action_shape"],
        },
        {
            "name": "checkpoint_training_path",
            "actual": checkpoint["args"]["dataset_path"],
            "recorded": analysis["training_contract"]["dataset_path"],
            "pass": checkpoint["args"]["dataset_path"] == analysis["training_contract"]["dataset_path"],
        },
        {
            "name": "pre_transition_steer_shift_zero",
            "actual": [metadata["state_action_logging"], checkpoint["args"]["steer_shift"]],
            "recorded": ["pre_transition", 0],
            "pass": metadata["state_action_logging"] == "pre_transition"
            and checkpoint["args"]["steer_shift"] == 0,
        },
    ))

    report = {
        "format": "highspeed_query_domain_contract_validator_v1",
        "analysis": str(args.analysis.resolve()),
        "qualification": "PASS" if all(item["pass"] for item in checks) else "FAIL",
        "checks": checks,
    }
    report_path = args.analysis.parent / "validator_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if report["qualification"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
