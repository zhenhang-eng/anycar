#!/usr/bin/env python3
"""Validate source lineage and summary recomputation for the high-speed trust scan."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stats(values: list[float]) -> dict[str, float | int]:
    array = np.asarray(values, np.float64)
    return {
        "count": int(array.size), "min": float(array.min()),
        "median": float(np.median(array)), "mean": float(array.mean()),
        "max": float(array.max()),
    }


def extract(row: dict) -> dict[str, float]:
    return {
        "teacher_gain_recovery": float(row["teacher_gain_recovery"]),
        "gain_vs_start_mean": float(row["gain_vs_pretrained_actor"]["mean"]),
        "gain_vs_start_p05": float(row["gain_vs_pretrained_actor"]["p05"]),
        "gain_vs_warm_mean": float(row["gain_vs_warm"]["mean"]),
        "gain_vs_warm_p05": float(row["gain_vs_warm"]["p05"]),
        "beats_warm_fraction": float(row["beats_or_equals_warm_fraction"]),
    }


def assert_close(left, right, path: str = "root") -> None:
    if isinstance(left, dict):
        if left.keys() != right.keys():
            raise AssertionError(f"key mismatch at {path}")
        for key in left:
            assert_close(left[key], right[key], f"{path}.{key}")
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
    if analysis["qualification"] != "HIGHSPEED_ROUND_AND_TRUST_BUDGET_MATTER_TRUST006_MECHANISM_LEAD":
        raise AssertionError("unexpected qualification")
    recomputed_curve = {}
    recomputed_selected = {}
    for trust, source in analysis["sources"].items():
        source_root = Path(source["root"])
        contract_path = source_root / "contract.json"
        summary_path = source_root / "summary.json"
        validator_path = source_root / "validator_report.json"
        for path, key in (
            (contract_path, "contract_sha256"),
            (summary_path, "summary_sha256"),
            (validator_path, "validator_sha256"),
        ):
            if sha256(path) != source[key]:
                raise AssertionError(f"source hash mismatch: {path}")
        validator = json.loads(validator_path.read_text())
        if validator["qualification"] != "HIGHSPEED_ACTOR_K_SCAN_INDEPENDENT_RELOAD_REPLAY_PASS":
            raise AssertionError("source independent validation missing")
        summary = json.loads(summary_path.read_text())
        recomputed_curve[trust] = {}
        for round_index in analysis["contract"]["evaluation_rounds"]:
            rows = []
            for record in summary["records"]:
                round_row = next(
                    row for row in record["arms"]["16"]["rounds"]
                    if int(row["round"]) == int(round_index)
                )
                rows.append(extract(round_row["oof"]))
            recomputed_curve[trust][str(round_index)] = {
                name: stats([row[name] for row in rows]) for name in rows[0]
            }
        rows = [extract(record["arms"]["16"]["selected"]["oof"]) for record in summary["records"]]
        recomputed_selected[trust] = {
            name: stats([row[name] for row in rows]) for name in rows[0]
        }
        recomputed_selected[trust]["selected_round"] = stats([
            record["arms"]["16"]["selected_round"] for record in summary["records"]
        ])
    assert_close(analysis["curve"], recomputed_curve, "curve")
    assert_close(analysis["selected"], recomputed_selected, "selected")
    report = {
        "format": "highspeed_actor_trust_curve_validator_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "qualification": "HIGHSPEED_ACTOR_TRUST_CURVE_INDEPENDENT_SUMMARY_PASS",
        "checks": {
            "analysis_sha256": sha256(analysis_path),
            "source_hashes_verified": True,
            "source_independent_dbm_validators_verified": True,
            "curve_recomputed": True,
            "selected_metrics_recomputed": True,
            "formal_validation_or_test_created": False,
        },
    }
    (root / "validator_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
