#!/usr/bin/env python3
"""Validate the OAC-2 Pairwise-Delta extension and its saved checkpoints."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import torch

from mppi_pair_delta_critic import ConfigurableAbsoluteActionValueCritic


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    contract = json.loads((args.run_dir / "contract.json").read_text())
    summary = json.loads((args.run_dir / "summary.json").read_text())
    pair_contract = contract["pair_delta_contract"]
    run_args = contract["arguments"]
    seeds = [int(value) for value in str(run_args["seeds"]).split(",")]
    device = torch.device(args.device)
    records = []
    for seed in seeds:
        seed_dir = args.run_dir / f"seed_{seed}"
        row = next(item for item in summary["records"] if int(item["seed"]) == seed)
        iteration_rows = [
            json.loads(line) for line in
            (seed_dir / "iteration_metrics.jsonl").read_text().splitlines()
            if line.strip()
        ]
        seed_checks: dict[str, bool] = {}
        network_records = []
        for twin in (1, 2):
            payload = torch.load(seed_dir / f"pair_critic{twin}.pt", map_location=device)
            model = ConfigurableAbsoluteActionValueCritic(pair_delta=True).to(device)
            model.load_state_dict(payload["model"], strict=True)
            model.eval()
            generator = torch.Generator(device="cpu").manual_seed(260825700 + 10 * seed + twin)
            history = torch.randn(4, 250, 7, generator=generator).to(device)
            reference = torch.randn(4, 50, 5, generator=generator).to(device)
            current = torch.randn(4, 4, generator=generator).to(device)
            left = torch.randn(4, 8, 2, generator=generator).to(device)
            right = torch.randn(4, 8, 2, generator=generator).to(device)
            with torch.no_grad():
                forward = model.pair_delta(history, reference, current, left, right)
                reverse = model.pair_delta(history, reference, current, right, left)
                identity = model.pair_delta(history, reference, current, left, left)
            antisymmetry_error = float((forward + reverse).abs().max().cpu())
            identity_error = float(identity.abs().max().cpu())
            network_records.append({
                "twin": twin,
                "parameter_count": int(sum(p.numel() for p in model.parameters())),
                "antisymmetry_max_abs_error": antisymmetry_error,
                "identity_delta_max_abs_error": identity_error,
            })
            seed_checks[f"twin_{twin}_strict_load"] = True
            seed_checks[f"twin_{twin}_antisymmetry"] = antisymmetry_error <= 1e-6
            seed_checks[f"twin_{twin}_identity_zero"] = identity_error <= 1e-6
            seed_checks[f"twin_{twin}_optimizer_serialized"] = "optimizer" in payload
            seed_checks[f"twin_{twin}_formal_test_sealed"] = not bool(
                payload["formal_validation_loaded"] or payload["test_loaded"]
            )
        expected_online = int(run_args["rounds"]) * int(run_args["critic_updates_per_round"])
        accuracies = [float(item["pair_delta_mean_comparison_accuracy"]) for item in iteration_rows]
        counts = [int(item["pair_delta_mean_comparison_count"]) for item in iteration_rows]
        seed_checks.update({
            "pretrain_update_count": int(row["pair_delta_pretrain_update_count"]) == int(pair_contract["pretrain_updates"]),
            "online_update_count": int(row["pair_delta_online_update_count"]) == expected_online,
            "iteration_count": len(iteration_rows) == int(run_args["rounds"]),
            "pair_metric_finite": all(0.0 <= value <= 1.0 for value in accuracies),
            "pair_metric_nonempty": all(value > 0 for value in counts),
            "formal_validation_sealed": not bool(summary["formal_validation_loaded"]),
            "test_sealed": not bool(summary["test_loaded"]),
        })
        records.append({
            "seed": seed,
            "passed": all(seed_checks.values()),
            "checks": seed_checks,
            "network_records": network_records,
            "pair_accuracy_initial": accuracies[0],
            "pair_accuracy_final": accuracies[-1],
            "pair_accuracy_min": min(accuracies),
            "pair_accuracy_max": max(accuracies),
        })
    checks = {
        "pair_enabled": bool(pair_contract["enabled"]),
        "absolute_main_gradient_unchanged": "absolute Twin Value still provides" in pair_contract["actor_use"],
        "training_only_deployment": "training-only" in pair_contract["deployment"],
        "all_seeds_pass": all(row["passed"] for row in records),
        "base_oac_validator_pass": json.loads(
            (args.run_dir / "validator_report.json").read_text()
        )["qualification"] == "OAC2_CONTINUOUS_ACTOR_VALIDATION_PASS",
    }
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "qualification": (
            "OAC2_PAIR_DELTA_VALIDATION_PASS" if all(checks.values())
            else "OAC2_PAIR_DELTA_VALIDATION_FAIL"
        ),
        "passed": all(checks.values()),
        "checks": checks,
        "records": records,
        "formal_validation_loaded": False,
        "test_loaded": False,
    }
    (args.run_dir / "pair_delta_validator_report.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
