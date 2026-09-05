#!/usr/bin/env python3
"""Qualify the K1/K4 90-round curve from its full-replay legacy-validator report.

The full replay is performed by validate_query_oac_gamma1_k_scan.py.  That validator
was written for the earlier K1/K4/K8 pilot and intentionally fails only its hard-coded
arm-list check on the registered K1/K4 long curve.  This second layer verifies that
the arm-list mismatch is the sole failure and preserves every replay/run/pair check.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = REPO_ROOT / "outputs/query_mppi/query_oac_gamma1_k1_k4_90round_20260902_v1"
LEGACY_REPORT = "validation_legacy_k_contract_fail.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, nargs="?", default=DEFAULT_OUTPUT)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    output = args.output.resolve()
    manifest_path = output / "manifest.json"
    summary_path = output / "summary.json"
    legacy_path = output / LEGACY_REPORT
    manifest = json.loads(manifest_path.read_text())
    summary = json.loads(summary_path.read_text())
    legacy = json.loads(legacy_path.read_text())
    config_path = Path(manifest["config"])
    config = json.loads(config_path.read_text())
    legacy_validator = REPO_ROOT / "scripts/model_verify/validate_query_oac_gamma1_k_scan.py"

    legacy_checks = dict(legacy["checks"])
    non_arm_checks = {name: value for name, value in legacy_checks.items() if name != "k_arms_exact"}
    errors = {name: float(value) for name, value in legacy["maximum_absolute_errors"].items()}
    expected_arms = [
        {"name": "gamma1_k1", "actor_updates_per_round": 1},
        {"name": "gamma1_k4", "actor_updates_per_round": 4},
    ]
    checks = {
        "config_hash": sha256(config_path) == manifest["config_sha256"],
        "runner_hash": sha256(Path(manifest["script"])) == manifest["script_sha256"],
        "summary_hash": sha256(summary_path) == manifest["summary_sha256"],
        "legacy_report_is_expected_fail": legacy["qualification"] == "QUERY_OAC_GAMMA1_K_SCAN_INDEPENDENT_FAIL",
        "legacy_hardcoded_arm_check_is_false": legacy_checks.get("k_arms_exact") is False,
        "all_other_legacy_checks_pass": bool(non_arm_checks) and all(non_arm_checks.values()),
        "all_full_replay_errors_zero": all(value <= 1e-6 for value in errors.values()),
        "all_six_run_checks_pass": len(legacy["run_reports"]) == 6 and all(
            report["all_checks_pass"] for report in legacy["run_reports"]
        ),
        "all_pair_checks_pass": len(legacy["pair_checks"]) > 0 and all(legacy["pair_checks"].values()),
        "registered_arms_exact": config["arms"] == expected_arms,
        "registered_rounds_90": int(config["pilot"]["rounds"]) == 90,
        "registered_curve_checkpoints": config["pilot"].get("curve_checkpoints") == [10, 20, 40, 60, 90],
        "gamma_fixed_one": float(config["actor_cost_weight_gamma"]) == 1.0,
        "critic_updates_fixed_20": int(config["critic_updates"]["updates_per_round_per_twin"]) == 20,
        "candidate_checks_match": legacy["recomputed_candidate_checks"] == summary["candidate_checks"],
        "decision_match": legacy["decision"] == summary["decision"] == manifest["decision"],
        "legacy_validator_present": legacy_validator.is_file(),
        "formal_test_sealed": not bool(summary["formal_validation_or_test_consumed"])
        and not bool(manifest["formal_validation_or_test_consumed"]),
        "dbm_fields_absent": not bool(summary["dbm_fields_or_labels_consumed"])
        and not bool(manifest["dbm_fields_or_labels_consumed"]),
        "analytic_query_gradient_absent": not bool(summary["query_analytic_gradient_consumed"])
        and not bool(manifest["query_analytic_gradient_consumed"]),
    }
    passed = bool(all(checks.values()))
    qualification = (
        "QUERY_OAC_GAMMA1_K1_K4_90ROUND_INDEPENDENT_PASS"
        if passed else "QUERY_OAC_GAMMA1_K1_K4_90ROUND_INDEPENDENT_FAIL"
    )
    report = {
        "qualification": qualification,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "passed": passed,
        "decision": summary["decision"],
        "checks": checks,
        "legacy_full_replay_report": str(legacy_path),
        "legacy_full_replay_report_sha256": sha256(legacy_path),
        "legacy_validator": str(legacy_validator),
        "legacy_validator_sha256": sha256(legacy_validator),
        "legacy_failed_checks": [name for name, value in legacy_checks.items() if not value],
        "maximum_absolute_errors_from_full_replay": errors,
        "full_replay_run_count": len(legacy["run_reports"]),
        "full_replay_pair_check_count": len(legacy["pair_checks"]),
        "formal_validation_or_test_consumed": False,
        "dbm_fields_or_labels_consumed": [],
        "query_analytic_gradient_consumed": False,
    }
    validation_path = output / "validation.json"
    validation_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    if passed:
        manifest["qualification"] = qualification
        manifest["validation"] = str(validation_path)
        manifest["validation_sha256"] = sha256(validation_path)
        manifest["validator"] = str(Path(__file__).resolve())
        manifest["validator_sha256"] = sha256(Path(__file__).resolve())
        manifest["legacy_full_replay_validation"] = str(legacy_path)
        manifest["legacy_full_replay_validation_sha256"] = sha256(legacy_path)
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "qualification": qualification,
        "decision": summary["decision"],
        "failed_checks": [name for name, value in checks.items() if not value],
        "legacy_failed_checks": report["legacy_failed_checks"],
        "full_replay_run_count": report["full_replay_run_count"],
        "full_replay_pair_check_count": report["full_replay_pair_check_count"],
        "maximum_absolute_errors_from_full_replay": errors,
    }, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
