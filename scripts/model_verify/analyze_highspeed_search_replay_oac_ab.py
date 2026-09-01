#!/usr/bin/env python3
"""Validate and compare paired high-speed search-informed Replay OAC arms."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--treatment-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--critic-eval-dir", type=Path)
    return parser.parse_args()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def distribution(values: list[float] | np.ndarray) -> dict[str, float | int]:
    array = np.asarray(values, np.float64)
    return {
        "count": int(array.size), "min": float(array.min()),
        "p05": float(np.quantile(array, 0.05)),
        "median": float(np.median(array)), "mean": float(array.mean()),
        "p95": float(np.quantile(array, 0.95)), "max": float(array.max()),
    }


def load_root(root: Path, expected_mode: str) -> tuple[dict, dict, dict]:
    contract_path = root / "contract.json"
    summary_path = root / "summary.json"
    validator_path = root / "validator_report.json"
    contract = json.loads(contract_path.read_text())
    summary = json.loads(summary_path.read_text())
    validator = json.loads(validator_path.read_text())
    if contract["exploration"]["mode"] != expected_mode:
        raise AssertionError(f"unexpected exploration mode in {root}")
    if contract["analytic_dbm_gradient"]:
        raise AssertionError("analytic DBM gradient entered an OAC arm")
    if contract["formal_validation_or_test_created"]:
        raise AssertionError("formal validation/test was opened")
    if validator["qualification"] != "HIGHSPEED_ACTOR_VISITED_OAC_INDEPENDENT_REPLAY_PASS":
        raise AssertionError(f"independent replay did not pass: {root}")
    if validator["summary_sha256"] != sha256(summary_path):
        raise AssertionError(f"validator/summary hash mismatch: {root}")
    return contract, summary, validator


def paired_value(record: dict[str, Any], path: tuple[str, ...]) -> float:
    value: Any = record
    for key in path:
        value = value[key]
    return float(value)


def main() -> None:
    args = parse_args()
    baseline_root = args.baseline_dir.resolve()
    treatment_root = args.treatment_dir.resolve()
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to replace output: {output}")
    base_contract, base_summary, base_validator = load_root(
        baseline_root, "nonrecentered65"
    )
    treatment_contract, treatment_summary, treatment_validator = load_root(
        treatment_root, "search_recentered65"
    )

    ignored_arguments = {"output_dir", "exploration_mode"}
    base_args = {
        key: value for key, value in base_contract["arguments"].items()
        if key not in ignored_arguments
    }
    treatment_args = {
        key: value for key, value in treatment_contract["arguments"].items()
        if key not in ignored_arguments
    }
    if base_args != treatment_args:
        raise AssertionError("A/B arguments differ beyond output/mode")
    for key in (
        "source_pretrain_summary_sha256", "source_pretrain_validator_sha256",
        "folds", "seeds", "critic_actor_update_ratio", "actor_objective",
    ):
        if base_contract[key] != treatment_contract[key]:
            raise AssertionError(f"A/B contract mismatch: {key}")

    base_records = {
        (int(row["fold"]), int(row["seed"])): row
        for row in base_summary["records"]
    }
    treatment_records = {
        (int(row["fold"]), int(row["seed"])): row
        for row in treatment_summary["records"]
    }
    if base_records.keys() != treatment_records.keys():
        raise AssertionError("A/B fold-seed keys differ")

    replay_checks = []
    exact_source_max = 0.0
    absorption_first_max = 0.0
    first_online_first_max = 0.0
    for key in sorted(base_records):
        left = base_records[key]
        right = treatment_records[key]
        if left["initial"] != right["initial"]:
            raise AssertionError(f"initial metrics mismatch: {key}")
        with np.load(left["replay"], allow_pickle=False) as loaded:
            base_replay = {name: np.asarray(loaded[name]) for name in loaded.files}
        with np.load(right["replay"], allow_pickle=False) as loaded:
            treatment_replay = {name: np.asarray(loaded[name]) for name in loaded.files}
        source_count = int(base_replay["source_replay_candidates"])
        absorption_count = int(base_replay["absorption_candidates"])
        if source_count != int(treatment_replay["source_replay_candidates"]):
            raise AssertionError("source replay candidate count mismatch")
        if absorption_count != int(treatment_replay["absorption_candidates"]):
            raise AssertionError("absorption candidate count mismatch")
        if absorption_count not in (0, 65):
            raise AssertionError("unexpected absorption candidate count")
        if base_replay["actions"].shape != treatment_replay["actions"].shape:
            raise AssertionError("final replay shapes differ")
        exact_source_max = max(exact_source_max, float(np.max(np.abs(
            base_replay["actions"][:, :source_count]
            - treatment_replay["actions"][:, :source_count]
        ))))
        if absorption_count:
            start = source_count
            absorption_first_max = max(absorption_first_max, float(np.max(np.abs(
                base_replay["actions"][:, start : start + 33]
                - treatment_replay["actions"][:, start : start + 33]
            ))))
        first_online = source_count + absorption_count
        first_online_first_max = max(first_online_first_max, float(np.max(np.abs(
            base_replay["actions"][:, first_online : first_online + 33]
            - treatment_replay["actions"][:, first_online : first_online + 33]
        ))))
        replay_checks.append({
            "fold": key[0], "seed": key[1],
            "source_candidates": source_count,
            "absorption_candidates": absorption_count,
            "final_candidates": int(base_replay["actions"].shape[1]),
        })
    if max(exact_source_max, absorption_first_max, first_online_first_max) != 0.0:
        raise AssertionError("paired source/first-stage candidates are not bitwise equal")

    metric_paths = {
        "teacher_gain_recovery": ("selected", "oof", "teacher_gain_recovery"),
        "beats_warm_fraction": ("selected", "oof", "beats_or_equals_warm_fraction"),
        "warm_gain_mean": ("selected", "oof", "gain_vs_warm", "mean"),
        "warm_gain_median": ("selected", "oof", "gain_vs_warm", "median"),
        "warm_gain_p05": ("selected", "oof", "gain_vs_warm", "p05"),
        "warm_gain_worst": ("selected", "oof", "gain_vs_warm", "min"),
        "critic_centered_pearson": (
            "critic_oof_local_probe", "centered_log_cost_pearson"
        ),
        "critic_sign_accuracy": (
            "critic_oof_local_probe", "center_relative_sign_accuracy"
        ),
        "critic_bank_recovery": (
            "critic_oof_local_probe", "bank_gain_recovery"
        ),
    }
    metrics = {}
    for name, path in metric_paths.items():
        baseline = [paired_value(base_records[key], path) for key in sorted(base_records)]
        treatment = [
            paired_value(treatment_records[key], path) for key in sorted(base_records)
        ]
        metrics[name] = {
            "baseline": distribution(baseline),
            "treatment": distribution(treatment),
            "paired_delta_treatment_minus_baseline": distribution(
                np.asarray(treatment) - np.asarray(baseline)
            ),
        }

    def round_bank_value(record: dict, field: str) -> float:
        if field == "full_gain":
            values = [row["bank"]["full_bank_gain"]["mean"] for row in record["rounds"]]
        elif field == "stage2_increment":
            values = [
                row["bank"]["full_bank_gain"]["mean"]
                - row["bank"]["first_stage_gain"]["mean"]
                for row in record["rounds"]
            ]
        else:
            raise ValueError(field)
        return float(np.mean(values))

    for name, field in (
        ("online_bank_full_gain_mean_over_rounds", "full_gain"),
        ("online_bank_stage2_increment_mean_over_rounds", "stage2_increment"),
    ):
        baseline = [round_bank_value(base_records[key], field) for key in sorted(base_records)]
        treatment = [
            round_bank_value(treatment_records[key], field) for key in sorted(base_records)
        ]
        metrics[name] = {
            "baseline": distribution(baseline), "treatment": distribution(treatment),
            "paired_delta_treatment_minus_baseline": distribution(
                np.asarray(treatment) - np.asarray(baseline)
            ),
        }

    record_count = len(base_records)
    engineering_pass = bool(
        max(exact_source_max, absorption_first_max, first_online_first_max) == 0.0
        and base_validator["episode_group_leakage_count"] == 0
        and treatment_validator["episode_group_leakage_count"] == 0
    )
    critic_delta = metrics["critic_sign_accuracy"][
        "paired_delta_treatment_minus_baseline"
    ]["median"]
    actor_delta = metrics["warm_gain_mean"][
        "paired_delta_treatment_minus_baseline"
    ]["median"]
    critic_eval = None
    if args.critic_eval_dir is not None:
        critic_eval_path = args.critic_eval_dir.resolve() / "analysis.json"
        critic_eval = json.loads(critic_eval_path.read_text())
        if critic_eval["qualification"] != "FRESH_TWO_STAGE_CRITIC_AB_COMPLETE_TRAIN_ONLY":
            raise AssertionError("unexpected fresh two-stage Critic evaluation")
        critic_eval = {
            "analysis": critic_eval,
            "analysis_path": str(critic_eval_path),
            "analysis_sha256": sha256(critic_eval_path),
        }
    bank_delta = metrics["online_bank_full_gain_mean_over_rounds"][
        "paired_delta_treatment_minus_baseline"
    ]["median"]
    if record_count < 15:
        qualification = "SEARCH_INFORMED_REPLAY_OAC_AB_SMOKE_ENGINEERING_PASS"
    elif critic_eval is not None and bank_delta > 0.0 and actor_delta > 0.0:
        fresh = critic_eval["analysis"]["metrics"]
        fresh_pearson = fresh["centered_log_cost_pearson"][
            "paired_delta_treatment_minus_baseline"
        ]["median"]
        fresh_sign = fresh["center_relative_sign_accuracy"][
            "paired_delta_treatment_minus_baseline"
        ]["median"]
        if fresh_pearson < 0.01 and fresh_sign < 0.01:
            qualification = (
                "SEARCH_INFORMED_BANK_STRONG_CRITIC_TRANSFER_WEAK_ACTOR_SMALL_GAIN"
            )
        else:
            qualification = "SEARCH_INFORMED_REPLAY_OAC_AB_POSITIVE_PENDING_GATES"
    elif critic_delta > 0.0 and actor_delta > 0.0:
        qualification = "SEARCH_INFORMED_REPLAY_OAC_AB_POSITIVE_PENDING_GATES"
    elif critic_delta > 0.0:
        qualification = "SEARCH_INFORMED_REPLAY_CRITIC_POSITIVE_ACTOR_NOT_TRANSFERRED"
    else:
        qualification = "SEARCH_INFORMED_REPLAY_CRITIC_NO_GAIN"
    if not engineering_pass:
        qualification = "SEARCH_INFORMED_REPLAY_OAC_AB_ENGINEERING_FAIL"

    output.mkdir(parents=True)
    analysis = {
        "format": "highspeed_search_informed_replay_oac_ab_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "qualification": qualification,
        "record_count": record_count,
        "contract": {
            "baseline_mode": "nonrecentered65",
            "treatment_mode": "search_recentered65",
            "only_registered_variable": "stage-two candidate recentering",
            "analytic_dbm_gradient": False,
            "formal_validation_or_test_created": False,
        },
        "sources": {
            "baseline_summary": str((baseline_root / "summary.json").resolve()),
            "baseline_summary_sha256": sha256(baseline_root / "summary.json"),
            "baseline_validator_sha256": sha256(baseline_root / "validator_report.json"),
            "treatment_summary": str((treatment_root / "summary.json").resolve()),
            "treatment_summary_sha256": sha256(treatment_root / "summary.json"),
            "treatment_validator_sha256": sha256(treatment_root / "validator_report.json"),
        },
        "paired_checks": {
            "source_replay_action_max_abs_diff": exact_source_max,
            "absorption_first_stage_action_max_abs_diff": absorption_first_max,
            "first_online_first_stage_action_max_abs_diff": first_online_first_max,
            "episode_group_leakage_count": int(
                base_validator["episode_group_leakage_count"]
                + treatment_validator["episode_group_leakage_count"]
            ),
            "records": replay_checks,
        },
        "metrics": metrics,
        "fresh_two_stage_critic_evaluation": critic_eval,
    }
    path = output / "analysis.json"
    path.write_text(json.dumps(analysis, indent=2) + "\n")
    print(json.dumps(analysis, indent=2))


if __name__ == "__main__":
    main()
