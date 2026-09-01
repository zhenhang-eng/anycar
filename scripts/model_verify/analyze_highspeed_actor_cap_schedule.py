#!/usr/bin/env python3
"""Summarize the high-speed K16 cap-only/LR-decay Actor schedule audit."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


SOURCE_NAMES = (
    "exact90_fold0",
    "cap90_fold0",
    "cap160_constant_fold0",
    "cap160_decay_fold0",
    "cap160_decay_fold1to4",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in SOURCE_NAMES:
        parser.add_argument(f"--{name.replace('_', '-')}-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stats(values: list[float]) -> dict[str, float | int]:
    array = np.asarray(values, np.float64)
    return {
        "count": int(array.size),
        "min": float(array.min()),
        "p05": float(np.quantile(array, 0.05)),
        "median": float(np.median(array)),
        "mean": float(array.mean()),
        "p95": float(np.quantile(array, 0.95)),
        "max": float(array.max()),
    }


def selected_metrics(record: dict) -> dict[str, float]:
    arm = record["arms"]["16"]
    row = arm["selected"]["oof"]
    latest = arm["latest"]["oof"]
    return {
        "selected_round": float(arm["selected_round"]),
        "cost_mean": float(row["cost"]["mean"]),
        "cost_median": float(row["cost"]["median"]),
        "cost_p95": float(row["cost"]["p95"]),
        "cost_max": float(row["cost"]["max"]),
        "teacher_gain_recovery": float(row["teacher_gain_recovery"]),
        "gain_vs_start_mean": float(row["gain_vs_pretrained_actor"]["mean"]),
        "gain_vs_start_median": float(row["gain_vs_pretrained_actor"]["median"]),
        "gain_vs_start_p05": float(row["gain_vs_pretrained_actor"]["p05"]),
        "gain_vs_start_min": float(row["gain_vs_pretrained_actor"]["min"]),
        "gain_vs_warm_mean": float(row["gain_vs_warm"]["mean"]),
        "gain_vs_warm_median": float(row["gain_vs_warm"]["median"]),
        "gain_vs_warm_p05": float(row["gain_vs_warm"]["p05"]),
        "gain_vs_warm_min": float(row["gain_vs_warm"]["min"]),
        "beats_warm_fraction": float(row["beats_or_equals_warm_fraction"]),
        "regression_fraction_vs_warm": float(row["regression_fraction_vs_warm"]),
        "latest_minus_selected_gain_vs_start_mean": float(
            latest["gain_vs_pretrained_actor"]["mean"]
            - row["gain_vs_pretrained_actor"]["mean"]
        ),
        "latest_minus_selected_gain_vs_warm_p05": float(
            latest["gain_vs_warm"]["p05"] - row["gain_vs_warm"]["p05"]
        ),
    }


def aggregate_records(records: list[dict]) -> dict:
    rows = [selected_metrics(record) for record in records]
    result = {key: stats([row[key] for row in rows]) for key in rows[0]}
    by_speed: dict[str, dict] = {}
    speed_keys = records[0]["arms"]["16"]["selected"]["oof"]["by_speed_kph"]
    for speed in speed_keys:
        layer_rows = [
            record["arms"]["16"]["selected"]["oof"]["by_speed_kph"][speed]
            for record in records
        ]
        by_speed[speed] = {
            key: stats([float(row[key]) for row in layer_rows])
            for key in ("mean_gain_vs_warm", "median_gain_vs_warm", "p05_gain_vs_warm")
        }
    result["by_speed_fold_seed_balanced"] = by_speed
    result["per_run"] = [
        {"fold": int(record["fold"]), "seed": int(record["seed"]), **row}
        for record, row in zip(records, rows)
    ]
    return result


def cap_activity(records: list[dict], cap: float = 0.06) -> dict:
    per_run = []
    all_raw = []
    for record in records:
        rounds = record["arms"]["16"]["rounds"]
        raw = np.asarray(
            [row["actor_raw_cumulative_step_sigma_rms"] for row in rounds], np.float64
        )
        all_raw.extend(raw.tolist())
        per_run.append({
            "fold": int(record["fold"]),
            "seed": int(record["seed"]),
            "round_count": int(raw.size),
            "below_cap_count": int(np.sum(raw < cap - 1e-6)),
            "at_or_above_cap_count": int(np.sum(raw >= cap - 1e-6)),
            "below_cap_fraction": float(np.mean(raw < cap - 1e-6)),
            "raw_step": stats(raw.tolist()),
        })
    raw_all = np.asarray(all_raw, np.float64)
    return {
        "cap_sigma_rms": cap,
        "raw_step": stats(raw_all.tolist()),
        "below_cap_fraction": float(np.mean(raw_all < cap - 1e-6)),
        "per_run": per_run,
    }


def main() -> None:
    args = parse_args()
    roots = {name: getattr(args, f"{name}_dir").resolve() for name in SOURCE_NAMES}
    loaded: dict[str, tuple[dict, dict]] = {}
    sources = {}
    for name, root in roots.items():
        paths = {key: root / filename for key, filename in (
            ("contract", "contract.json"),
            ("summary", "summary.json"),
            ("validator", "validator_report.json"),
        )}
        for path in paths.values():
            if not path.exists():
                raise FileNotFoundError(path)
        contract = json.loads(paths["contract"].read_text())
        summary = json.loads(paths["summary"].read_text())
        validator = json.loads(paths["validator"].read_text())
        if contract["k_values"] != [16]:
            raise AssertionError(f"{name}: expected K16-only source")
        if validator["qualification"] != "HIGHSPEED_ACTOR_K_SCAN_INDEPENDENT_RELOAD_REPLAY_PASS":
            raise AssertionError(f"{name}: independent validator did not pass")
        loaded[name] = (contract, summary)
        sources[name] = {
            "root": str(root),
            **{f"{key}_sha256": sha256(path) for key, path in paths.items()},
        }

    expected = {
        "exact90_fold0": (90, [0], "exact", None, None),
        "cap90_fold0": (90, [0], "cap_only", None, None),
        "cap160_constant_fold0": (160, [0], "cap_only", None, None),
        "cap160_decay_fold0": (160, [0], "cap_only", 120, 0.25),
        "cap160_decay_fold1to4": (160, [1, 2, 3, 4], "cap_only", 120, 0.25),
    }
    for name, (rounds, folds, mode, decay_start, final_scale) in expected.items():
        contract, _ = loaded[name]
        arguments = contract["arguments"]
        actual_mode = arguments.get("round_step_mode", "exact")
        if int(arguments["rounds"]) != rounds or contract["folds"] != folds:
            raise AssertionError(f"{name}: rounds/folds mismatch")
        if actual_mode != mode or not np.isclose(float(arguments["max_round_step_sigma_rms"]), 0.06):
            raise AssertionError(f"{name}: step contract mismatch")
        if arguments.get("actor_lr_decay_start_round") != decay_start:
            raise AssertionError(f"{name}: LR decay start mismatch")
        if arguments.get("actor_lr_final_scale") != final_scale:
            raise AssertionError(f"{name}: LR final scale mismatch")

    final_records = (
        loaded["cap160_decay_fold0"][1]["records"]
        + loaded["cap160_decay_fold1to4"][1]["records"]
    )
    identities = {(int(row["fold"]), int(row["seed"])) for row in final_records}
    expected_identities = {(fold, seed) for fold in range(5) for seed in range(3)}
    if identities != expected_identities or len(final_records) != 15:
        raise AssertionError("final full-fold identity contract mismatch")

    fold0_comparison = {
        name: aggregate_records(summary["records"])
        for name, (_, summary) in loaded.items()
        if name != "cap160_decay_fold1to4"
    }
    final_aggregate = aggregate_records(final_records)
    min_run_warm_p05 = final_aggregate["gain_vs_warm_p05"]["min"]
    min_run_beats_warm = final_aggregate["beats_warm_fraction"]["min"]
    mechanism_gate = {
        "all_run_mean_gain_vs_start_positive": final_aggregate["gain_vs_start_mean"]["min"] > 0,
        "all_run_mean_gain_vs_warm_positive": final_aggregate["gain_vs_warm_mean"]["min"] > 0,
        "all_run_warm_p05_nonnegative": min_run_warm_p05 >= 0,
        "all_run_beats_warm_fraction_at_least_0_95": min_run_beats_warm >= 0.95,
    }
    mechanism_gate["passed"] = all(mechanism_gate.values())

    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    analysis = {
        "format": "highspeed_actor_cap_schedule_analysis_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "qualification": "HIGHSPEED_CAP006_LRDECAY_FULL_FOLD_AUDIT_COMPLETE",
        "contract": {
            "final_arm": {
                "k": 16,
                "rounds": 160,
                "folds": [0, 1, 2, 3, 4],
                "seeds": [0, 1, 2],
                "round_step_mode": "cap_only",
                "round_step_cap_sigma_rms": 0.06,
                "exploration_decay_rounds": 90,
                "actor_lr": 2e-5,
                "actor_lr_decay_start_round": 120,
                "actor_lr_final_scale": 0.25,
            },
            "evaluation_scope": "train-only episode-grouped OOF mechanism audit",
            "formal_validation_or_test_created": False,
        },
        "sources": sources,
        "fold0_ablation": fold0_comparison,
        "final_full_fold": final_aggregate,
        "final_cap_activity": cap_activity(final_records),
        "mechanism_gate": mechanism_gate,
        "decision": {
            "cap_only_replaces_exact_forcing": True,
            "late_lr_decay_is_preferred_over_constant_lr": True,
            "full_fold_mechanism_gate_passed": mechanism_gate["passed"],
            "deployment_default_authorized": False,
            "formal_validation_or_test_authorized": False,
            "reason": (
                "This artifact measures deterministic Actor-center quality relative to the "
                "pretrained Actor and warm center. It does not validate the MPPI wrapper or closed loop."
            ),
        },
    }
    analysis_path = output / "analysis.json"
    analysis_path.write_text(json.dumps(analysis, indent=2) + "\n")
    print(json.dumps({
        "qualification": analysis["qualification"],
        "mechanism_gate": mechanism_gate,
        "selected": {key: final_aggregate[key] for key in (
            "selected_round", "gain_vs_start_mean", "gain_vs_start_p05",
            "gain_vs_warm_mean", "gain_vs_warm_p05", "beats_warm_fraction",
        )},
        "output": str(output),
    }, indent=2))


if __name__ == "__main__":
    main()
