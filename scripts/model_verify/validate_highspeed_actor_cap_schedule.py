#!/usr/bin/env python3
"""Independently validate the high-speed Actor cap/LR schedule summary."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from analyze_highspeed_actor_cap_schedule import aggregate_records, cap_activity


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def assert_close(left, right, path: str = "root") -> None:
    if isinstance(left, dict):
        if left.keys() != right.keys():
            raise AssertionError(f"key mismatch at {path}: {left.keys()} != {right.keys()}")
        for key in left:
            assert_close(left[key], right[key], f"{path}.{key}")
    elif isinstance(left, list):
        if len(left) != len(right):
            raise AssertionError(f"length mismatch at {path}")
        for index, (left_item, right_item) in enumerate(zip(left, right)):
            assert_close(left_item, right_item, f"{path}[{index}]")
    elif isinstance(left, (float, int)) and isinstance(right, (float, int)):
        if not np.isclose(float(left), float(right), rtol=0.0, atol=1e-9):
            raise AssertionError(f"value mismatch at {path}: {left} != {right}")
    elif left != right:
        raise AssertionError(f"value mismatch at {path}: {left} != {right}")


def main() -> None:
    args = parse_args()
    root = args.input_dir.resolve()
    analysis_path = root / "analysis.json"
    analysis = json.loads(analysis_path.read_text())
    if analysis["qualification"] != "HIGHSPEED_CAP006_LRDECAY_FULL_FOLD_AUDIT_COMPLETE":
        raise AssertionError("unexpected analysis qualification")

    loaded = {}
    for name, source in analysis["sources"].items():
        source_root = Path(source["root"])
        paths = {key: source_root / filename for key, filename in (
            ("contract", "contract.json"),
            ("summary", "summary.json"),
            ("validator", "validator_report.json"),
        )}
        for key, path in paths.items():
            if sha256(path) != source[f"{key}_sha256"]:
                raise AssertionError(f"source hash mismatch: {path}")
        validator = json.loads(paths["validator"].read_text())
        if validator["qualification"] != "HIGHSPEED_ACTOR_K_SCAN_INDEPENDENT_RELOAD_REPLAY_PASS":
            raise AssertionError(f"source validator failed: {name}")
        contract = json.loads(paths["contract"].read_text())
        if contract["formal_validation_or_test_created"]:
            raise AssertionError(f"formal/test boundary violated: {name}")
        loaded[name] = json.loads(paths["summary"].read_text())

    expected_fold0 = {
        name: aggregate_records(loaded[name]["records"])
        for name in (
            "exact90_fold0", "cap90_fold0", "cap160_constant_fold0", "cap160_decay_fold0"
        )
    }
    assert_close(analysis["fold0_ablation"], expected_fold0, "fold0_ablation")

    final_records = (
        loaded["cap160_decay_fold0"]["records"]
        + loaded["cap160_decay_fold1to4"]["records"]
    )
    identities = [(int(row["fold"]), int(row["seed"])) for row in final_records]
    if sorted(identities) != [(fold, seed) for fold in range(5) for seed in range(3)]:
        raise AssertionError("full-fold identity mismatch")
    assert_close(analysis["final_full_fold"], aggregate_records(final_records), "final_full_fold")
    assert_close(analysis["final_cap_activity"], cap_activity(final_records), "cap_activity")

    final = analysis["final_full_fold"]
    expected_gate = {
        "all_run_mean_gain_vs_start_positive": final["gain_vs_start_mean"]["min"] > 0,
        "all_run_mean_gain_vs_warm_positive": final["gain_vs_warm_mean"]["min"] > 0,
        "all_run_warm_p05_nonnegative": final["gain_vs_warm_p05"]["min"] >= 0,
        "all_run_beats_warm_fraction_at_least_0_95": final["beats_warm_fraction"]["min"] >= 0.95,
    }
    expected_gate["passed"] = all(expected_gate.values())
    assert_close(analysis["mechanism_gate"], expected_gate, "mechanism_gate")

    report = {
        "format": "highspeed_actor_cap_schedule_validator_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "qualification": "HIGHSPEED_ACTOR_CAP_SCHEDULE_INDEPENDENT_SUMMARY_PASS",
        "checks": {
            "analysis_sha256": sha256(analysis_path),
            "source_hashes_verified": True,
            "source_independent_dbm_replay_validators_verified": True,
            "fold0_ablation_recomputed": True,
            "full_fold_metrics_recomputed": True,
            "cap_activity_recomputed": True,
            "mechanism_gate_recomputed": True,
            "formal_validation_or_test_created": False,
        },
    }
    (root / "validator_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
