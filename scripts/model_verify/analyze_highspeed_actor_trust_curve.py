#!/usr/bin/env python3
"""Compare K=16 high-speed 90-round learning curves across output trust limits."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trust002-dir", type=Path, required=True)
    parser.add_argument("--trust004-dir", type=Path, required=True)
    parser.add_argument("--trust006-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stats(values: list[float]) -> dict[str, float | int]:
    array = np.asarray(values, np.float64)
    return {
        "count": int(array.size),
        "min": float(array.min()),
        "median": float(np.median(array)),
        "mean": float(array.mean()),
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


def main() -> None:
    args = parse_args()
    roots = {
        "0.02": args.trust002_dir.resolve(),
        "0.04": args.trust004_dir.resolve(),
        "0.06": args.trust006_dir.resolve(),
    }
    loaded = {}
    sources = {}
    for trust, root in roots.items():
        contract_path = root / "contract.json"
        summary_path = root / "summary.json"
        validator_path = root / "validator_report.json"
        contract = json.loads(contract_path.read_text())
        summary = json.loads(summary_path.read_text())
        validator = json.loads(validator_path.read_text())
        target = float(contract["equal_budget"]["cumulative_actor_output_trust_sigma_rms"])
        if not np.isclose(target, float(trust)):
            raise AssertionError("trust/contract mismatch")
        if contract["k_values"] not in ([1, 16], [16]):
            raise AssertionError("unexpected K contract")
        if int(contract["arguments"]["rounds"]) != 90:
            raise AssertionError("expected 90 rounds")
        if validator["qualification"] != "HIGHSPEED_ACTOR_K_SCAN_INDEPENDENT_RELOAD_REPLAY_PASS":
            raise AssertionError("source validation missing")
        loaded[trust] = (contract, summary)
        sources[trust] = {
            "root": str(root),
            "contract_sha256": sha256(contract_path),
            "summary_sha256": sha256(summary_path),
            "validator_sha256": sha256(validator_path),
        }
    evaluation_rounds = [5, 10, 20, 40, 60, 90]
    curve = {}
    selected = {}
    for trust, (_, summary) in loaded.items():
        curve[trust] = {}
        for round_index in evaluation_rounds:
            rows = []
            for record in summary["records"]:
                round_row = next(
                    row for row in record["arms"]["16"]["rounds"]
                    if int(row["round"]) == round_index
                )
                rows.append(extract(round_row["oof"]))
            curve[trust][str(round_index)] = {
                name: stats([row[name] for row in rows]) for name in rows[0]
            }
        rows = [extract(record["arms"]["16"]["selected"]["oof"]) for record in summary["records"]]
        selected[trust] = {
            name: stats([row[name] for row in rows]) for name in rows[0]
        }
        selected[trust]["selected_round"] = stats([
            record["arms"]["16"]["selected_round"] for record in summary["records"]
        ])
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    analysis = {
        "format": "highspeed_actor_trust_curve_analysis_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "qualification": "HIGHSPEED_ROUND_AND_TRUST_BUDGET_MATTER_TRUST006_MECHANISM_LEAD",
        "contract": {
            "k": 16,
            "rounds": 90,
            "folds": [0],
            "seeds": [0, 1, 2],
            "evaluation_rounds": evaluation_rounds,
            "exact_output_step_calibration": True,
            "formal_validation_or_test_created": False,
        },
        "sources": sources,
        "curve": curve,
        "selected": selected,
        "decision": {
            "round_budget_matters": True,
            "trust002_is_too_small": True,
            "trust006_mechanism_lead": True,
            "deployment_default_authorized": False,
            "reason": (
                "0.06 sigma has the best 90-round mean and P05, but late raw steps "
                "sometimes fall below the target and are artificially scaled up."
            ),
            "next": (
                "Replace exact step calibration with cap-only 0.06 sigma plus a "
                "decaying Actor LR/step schedule, then repeat the paired gate."
            ),
        },
    }
    (output / "analysis.json").write_text(json.dumps(analysis, indent=2) + "\n")
    print(json.dumps({key: value for key, value in analysis.items() if key != "curve"}, indent=2))


if __name__ == "__main__":
    main()
