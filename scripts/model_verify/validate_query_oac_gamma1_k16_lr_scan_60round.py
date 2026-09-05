#!/usr/bin/env python3
"""Qualify the K16 Actor-LR screen after the shared full-replay validator."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import torch

import run_query_oac_gamma1_k16_lr_scan_60round as runner
import run_query_oac_gamma1_k_scan as base


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = REPO_ROOT / "outputs/query_mppi/query_oac_gamma1_k16_lr_scan_60round_20260903_v1"
LEGACY_REPORT = "validation_legacy_k_contract_fail.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, nargs="?", default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    output = parse_args().output.resolve()
    manifest_path = output / "manifest.json"
    summary_path = output / "summary.json"
    legacy_path = output / LEGACY_REPORT
    manifest = json.loads(manifest_path.read_text())
    summary = json.loads(summary_path.read_text())
    legacy = json.loads(legacy_path.read_text())
    config_path = Path(manifest["config"])
    config = json.loads(config_path.read_text())
    prior_dir = Path(manifest["k16_160round"])
    prior_validation = json.loads((prior_dir / "validation.json").read_text())
    legacy_checks = dict(legacy["checks"])
    non_arm_checks = {name: value for name, value in legacy_checks.items() if name != "k_arms_exact"}
    errors = {name: float(value) for name, value in legacy["maximum_absolute_errors"].items()}
    expected_arms = [
        {"name": "gamma1_k16_lr2e6", "actor_updates_per_round": 16, "learning_rate_per_microstep": 2e-6},
        {"name": "gamma1_k16_lr5e6", "actor_updates_per_round": 16, "learning_rate_per_microstep": 5e-6},
        {"name": "gamma1_k16_lr1e5", "actor_updates_per_round": 16, "learning_rate_per_microstep": 1e-5},
    ]
    baseline = expected_arms[0]["name"]
    reconstructed_tail = {}
    recommendation = "KEEP_ACTOR_LR_2E6"
    for candidate in config["decision_gate"]["candidate_order"]:
        local_checks = runner.tail_checks(
            summary["pooled_warm_relative"][candidate]["selected"]["inner"],
            summary["pooled_warm_relative"][baseline]["selected"]["inner"],
            config,
        )
        overall_passes = bool(summary["candidate_checks"][candidate]["passes"])
        reconstructed_tail[candidate] = {
            "checks": local_checks,
            "overall_gate_passes": overall_passes,
            "passes": bool(overall_passes and all(local_checks.values())),
        }
        if recommendation == "KEEP_ACTOR_LR_2E6" and overall_passes and all(local_checks.values()):
            recommendation = f"RAISE_ACTOR_LR_TO_{candidate.rsplit('_lr', 1)[-1].upper()}"

    lr_checkpoint_checks = {}
    for record in summary["records"]:
        checkpoint = torch.load(record["checkpoint"], map_location="cpu", weights_only=False)
        expected_lr = next(float(arm["learning_rate_per_microstep"]) for arm in expected_arms if arm["name"] == record["arm"])
        lr_checkpoint_checks[f"{record['arm']}_seed{record['seed']}"] = bool(
            int(record["actor_updates_per_round"]) == 16
            and float(record["learning_rate_per_microstep"]) == expected_lr
            and float(checkpoint["arm"]["learning_rate_per_microstep"]) == expected_lr
            and int(checkpoint["actor_update_count"]) == 60 * 16
        )

    checks = {
        "config_hash": base.sha256(config_path) == manifest["config_sha256"],
        "runner_hash": base.sha256(Path(manifest["script"])) == manifest["script_sha256"],
        "shared_runner_hash": base.sha256(Path(manifest["shared_runner"])) == manifest["shared_runner_sha256"],
        "summary_hash": base.sha256(summary_path) == manifest["summary_sha256"],
        "prior_k16_160_qualified": prior_validation["qualification"] == "QUERY_OAC_GAMMA1_K16_160ROUND_INDEPENDENT_PASS",
        "prior_validation_hash": base.sha256(prior_dir / "validation.json") == manifest["k16_160round_validation_sha256"],
        "legacy_report_expected_fail": legacy["qualification"] == "QUERY_OAC_GAMMA1_K_SCAN_INDEPENDENT_FAIL",
        "legacy_hardcoded_k_check_false": legacy_checks.get("k_arms_exact") is False,
        "all_other_legacy_checks_pass": bool(non_arm_checks) and all(non_arm_checks.values()),
        "all_full_replay_errors_zero": all(value <= 1e-6 for value in errors.values()),
        "all_nine_run_checks_pass": len(legacy["run_reports"]) == 9 and all(report["all_checks_pass"] for report in legacy["run_reports"]),
        "all_pair_checks_pass": len(legacy["pair_checks"]) > 0 and all(legacy["pair_checks"].values()),
        "registered_arms_exact": config["arms"] == expected_arms,
        "registered_rounds_60": int(config["pilot"]["rounds"]) == 60,
        "registered_curve_checkpoints": config["pilot"]["curve_checkpoints"] == [10, 20, 40, 60],
        "deterministic_warn_only_recorded": summary["deterministic_runtime_contract"]["mode"] == "warn_only"
        and bool(summary["deterministic_runtime_contract"]["deterministic_algorithms"]),
        "strict_determinism_failure_preserved": Path(config["pairing"]["strict_determinism_attempt"]["preserved_output"]).is_dir(),
        "all_lr_checkpoint_checks_pass": all(lr_checkpoint_checks.values()),
        "candidate_checks_match_full_replay": legacy["recomputed_candidate_checks"] == summary["candidate_checks"],
        "legacy_decision_match": legacy["decision"] == summary["decision"] == manifest["decision"],
        "tail_review_reconstructed": reconstructed_tail == summary["lr_tail_review"],
        "recommendation_reconstructed": recommendation == summary["lr_recommendation"] == manifest["lr_recommendation"],
        "formal_test_sealed": not bool(summary["formal_validation_or_test_consumed"]) and not bool(manifest["formal_validation_or_test_consumed"]),
        "dbm_fields_absent": not bool(summary["dbm_fields_or_labels_consumed"]) and not bool(manifest["dbm_fields_or_labels_consumed"]),
        "analytic_query_gradient_absent": not bool(summary["query_analytic_gradient_consumed"]) and not bool(manifest["query_analytic_gradient_consumed"]),
    }
    passed = bool(all(checks.values()))
    qualification = "QUERY_OAC_GAMMA1_K16_LR_SCAN_60ROUND_INDEPENDENT_PASS" if passed else "QUERY_OAC_GAMMA1_K16_LR_SCAN_60ROUND_INDEPENDENT_FAIL"
    report = {
        "qualification": qualification,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "passed": passed,
        "lr_recommendation": recommendation,
        "checks": checks,
        "lr_checkpoint_checks": lr_checkpoint_checks,
        "recomputed_tail_review": reconstructed_tail,
        "legacy_full_replay_report": str(legacy_path),
        "legacy_full_replay_report_sha256": base.sha256(legacy_path),
        "legacy_failed_checks": [name for name, value in legacy_checks.items() if not value],
        "maximum_absolute_errors_from_full_replay": errors,
        "full_replay_run_count": len(legacy["run_reports"]),
        "full_replay_pair_check_count": len(legacy["pair_checks"]),
        "formal_validation_or_test_consumed": False,
        "dbm_fields_or_labels_consumed": [],
        "query_analytic_gradient_consumed": False,
    }
    validation_path = output / "validation.json"
    base.dump_json(validation_path, report)
    if passed:
        manifest["qualification"] = qualification
        manifest["validation"] = str(validation_path)
        manifest["validation_sha256"] = base.sha256(validation_path)
        manifest["validator"] = str(Path(__file__).resolve())
        manifest["validator_sha256"] = base.sha256(Path(__file__).resolve())
        manifest["legacy_full_replay_validation"] = str(legacy_path)
        manifest["legacy_full_replay_validation_sha256"] = base.sha256(legacy_path)
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "qualification": qualification,
        "lr_recommendation": recommendation,
        "failed_checks": [name for name, value in checks.items() if not value],
        "legacy_failed_checks": report["legacy_failed_checks"],
        "full_replay_run_count": report["full_replay_run_count"],
        "full_replay_pair_check_count": report["full_replay_pair_check_count"],
        "maximum_absolute_errors_from_full_replay": errors,
        "tail_review": reconstructed_tail,
    }, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
