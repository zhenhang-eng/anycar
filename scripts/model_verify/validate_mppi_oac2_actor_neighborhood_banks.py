#!/usr/bin/env python3
"""Independently validate the frozen-Actor neighborhood-bank audit artifact."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np

from generate_dbm_proposal_teacher import sha256_file


DEFAULT_INPUT = Path(
    "outputs/mppi_proposal/oac2_actor_neighborhood_bank_audit_20260828_v1"
)
TOLERANCE = 2e-6


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_dir", type=Path, nargs="?", default=DEFAULT_INPUT)
    return parser.parse_args()


def sha256_array(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def close(left: float, right: float) -> bool:
    return abs(float(left) - float(right)) <= TOLERANCE


def main() -> None:
    args = parse_args()
    summary_path = args.input_dir / "summary.json"
    arrays_path = args.input_dir / "evaluation.npz"
    summary = json.loads(summary_path.read_text())
    with np.load(arrays_path, allow_pickle=False) as loaded:
        arrays = {key: np.asarray(loaded[key]) for key in loaded.files}
    methods = [str(value) for value in arrays["method"]]
    cost = arrays["cost"].astype(np.float64)
    action = arrays["action"].astype(np.float64)
    warm = arrays["warm_cost"].astype(np.float64)
    expected_methods = ["current", "orthogonal", "guided_1x", "guided_2x"]
    checks = {
        "qualification": summary["qualification"]
        == "ACTOR_NEIGHBORHOOD_BANK_AUDIT_COMPLETE",
        "methods": methods == expected_methods,
        "shape": cost.shape == (3, 4, 600, 6),
        "action_shape": action.shape == (3, 4, 600, 6, 8, 2),
        "budget_six": int(summary["contract"]["evaluation_budget_per_state"]) == 6,
        "finite": bool(np.all(np.isfinite(cost))) and bool(np.all(np.isfinite(action))),
        "shared_center_action": float(np.max(np.abs(
            action[:, :, :, 0] - action[:, :1, :, 0]
        ))) <= 1e-7,
        "shared_center_cost": float(np.max(np.abs(cost[:, :, :, 0] - cost[:, :1, :, 0]))) <= 1e-7,
        "guided1_first_five_current": bool(np.array_equal(action[:, 2, :, :5], action[:, 0, :, :5])),
        "guided2_first_five_current": bool(np.array_equal(action[:, 3, :, :5], action[:, 0, :, :5])),
        "formal_validation_sealed": summary["checks"]["formal_validation_loaded"] is False,
        "test_sealed": summary["checks"]["test_loaded"] is False,
        "run_contract_hash": sha256_file(Path(summary["manifest"]["run"]) / "contract.json")
        == summary["manifest"]["run_contract_sha256"],
        "run_summary_hash": sha256_file(Path(summary["manifest"]["run"]) / "summary.json")
        == summary["manifest"]["run_summary_sha256"],
    }
    for key, expected in summary["array_sha256"].items():
        checks[f"array_hash_{key}"] = sha256_array(arrays[key]) == expected

    pooled_warm = np.tile(warm, 3)
    best = np.min(cost, axis=3).reshape(3, 4, -1)
    base = cost[:, :, :, 0].reshape(3, 4, -1)
    for index, method in enumerate(methods):
        gain = base[:, index].reshape(-1) - best[:, index].reshape(-1)
        actor_loses = base[:, index].reshape(-1) > pooled_warm + 1e-6
        recovered = float(np.mean(
            best[:, index].reshape(-1)[actor_loses]
            <= pooled_warm[actor_loses] + 1e-6
        ))
        stored = summary["pooled"][method]
        checks[f"{method}_gain_mean"] = close(
            np.mean(gain), stored["best_gain_vs_actor"]["mean"]
        )
        checks[f"{method}_hit"] = close(
            np.mean(gain > 1e-6), stored["strict_improvement_fraction"]
        )
        checks[f"{method}_recover_warm"] = close(
            recovered, stored["actor_loses_warm_recovered_fraction"]
        )
    current_best = best[:, 0].reshape(-1)
    for index, method in enumerate(methods[1:], start=1):
        delta = current_best - best[:, index].reshape(-1)
        stored = summary["pooled"]["paired_vs_current"][method]
        checks[f"{method}_paired_mean"] = close(
            np.mean(delta), stored["best_cost_gain_over_current"]["mean"]
        )
        checks[f"{method}_paired_win"] = close(
            np.mean(delta > 1e-6), stored["candidate_strict_win_fraction"]
        )
        checks[f"{method}_paired_loss"] = close(
            np.mean(delta < -1e-6), stored["current_strict_win_fraction"]
        )
    passed = bool(all(checks.values()))
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "qualification": (
            "OAC2_ACTOR_NEIGHBORHOOD_BANK_VALIDATION_PASS"
            if passed else "OAC2_ACTOR_NEIGHBORHOOD_BANK_VALIDATION_FAIL"
        ),
        "passed": passed,
        "checks": checks,
        "summary_sha256": sha256_file(summary_path),
        "evaluation_sha256": sha256_file(arrays_path),
        "formal_validation_loaded": False,
        "test_loaded": False,
    }
    (args.input_dir / "validator_report.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(report, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
