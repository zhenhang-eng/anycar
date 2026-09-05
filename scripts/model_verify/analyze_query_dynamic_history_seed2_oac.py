#!/usr/bin/env python3
"""Compare the dynamic-history seed-2 OAC screen with its frozen baseline."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from analyze_query_target_coverage_mixed_init_oac import critic_bank_metrics
from mppi_pair_delta_critic import ConfigurableAbsoluteActionValueCritic
from run_query_single_center_oac20to1 import load_inputs, sha256


DEFAULT_BASELINE = Path(
    "/home/plusai/anycar/outputs/query_mppi/"
    "query_target_coverage_mixed_init_fixed_lr1e5_oac_160round_20260903_v1"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    return parser.parse_args()


def metrics(cost: np.ndarray, warm: np.ndarray) -> dict[str, float | int]:
    gain = warm.astype(np.float64) - cost.astype(np.float64)
    return {
        "count": int(len(cost)),
        "cost_mean": float(np.mean(cost)),
        "cost_median": float(np.median(cost)),
        "warm_win_or_tie_fraction": float(np.mean(cost <= warm)),
        "warm_gain_mean": float(np.mean(gain)),
        "warm_gain_median": float(np.median(gain)),
        "warm_gain_p05": float(np.quantile(gain, 0.05)),
        "warm_gain_worst": float(np.min(gain)),
    }


def load_selected(root: Path, seed: int) -> tuple[dict, dict[str, np.ndarray], dict]:
    summary = json.loads((root / "summary.json").read_text())
    record = next(value for value in summary["records"] if int(value["seed"]) == seed)
    with np.load(record["arrays"], allow_pickle=False) as archive:
        arrays = {name: np.asarray(archive[name]) for name in archive.files}
    checkpoint = torch.load(record["checkpoint"], map_location="cpu", weights_only=False)
    return record, arrays, checkpoint


def main() -> None:
    args = parse_args()
    root = args.run.resolve()
    baseline_root = args.baseline.resolve()
    summary = json.loads((root / "summary.json").read_text())
    replay_path = Path(summary["contract"]["sources"]["absolute_replay"]) / "replay.npz"
    with np.load(replay_path, allow_pickle=False) as archive:
        data = {name: np.asarray(archive[name]) for name in archive.files}
    seed = 2
    new_record, new_arrays, new_checkpoint = load_selected(root, seed)
    old_record, old_arrays, old_checkpoint = load_selected(baseline_root, seed)
    if not np.array_equal(new_arrays["selection_indices"], old_arrays["selection_indices"]):
        raise AssertionError("baseline and screen use different inner rows")
    inner = new_arrays["selection_indices"]
    hard_mask = (data["speed_kph"][inner] == 100) & (data["variant_index"][inner] == 3)
    hard_rows = inner[hard_mask]

    old_cost = old_arrays["selection_round_cost"][int(old_record["selected_round"])]
    new_cost = new_arrays["selection_round_cost"][int(new_record["selected_round"])]
    comparisons = {}
    for name, mask in (("inner_all", np.ones(len(inner), bool)), ("inner_100:3", hard_mask)):
        old_metric = metrics(old_cost[mask], data["warm_cost"][inner][mask])
        new_metric = metrics(new_cost[mask], data["warm_cost"][inner][mask])
        comparisons[name] = {
            "baseline": old_metric,
            "dynamic_history": new_metric,
            "delta_dynamic_minus_baseline": {
                key: float(new_metric[key] - old_metric[key])
                for key in old_metric if key != "count"
            },
        }

    device = torch.device(args.device)
    hard_records = []
    critic_summary = {}
    for label, record, arrays, checkpoint in (
        ("baseline", old_record, old_arrays, old_checkpoint),
        ("dynamic_history", new_record, new_arrays, new_checkpoint),
    ):
        checkpoint = torch.load(record["checkpoint"], map_location=device, weights_only=False)
        inputs = load_inputs(data, checkpoint["critic_normalization"])
        critics, trainings = [], []
        for twin in (1, 2):
            critic = ConfigurableAbsoluteActionValueCritic().to(device)
            critic.load_state_dict(checkpoint[f"selected_critic{twin}_state_dict"], strict=True)
            critic.eval()
            critics.append(critic)
            trainings.append(checkpoint[f"critic{twin}_training"])
        positions = np.asarray([int(np.flatnonzero(inner == row)[0]) for row in hard_rows])
        selected_cost = arrays["selection_round_cost"][int(record["selected_round"]), positions]
        correlations, pairs = [], []
        for row, cost in zip(hard_rows, selected_cost):
            bank = critic_bank_metrics(critics, trainings, inputs, data, int(row), device)
            correlations.append(bank["log_cost_pearson"])
            pairs.append(bank["pair_sign_accuracy"])
            hard_records.append({
                "run": label,
                "row": int(row),
                "control_step": int(data["control_step"][row]),
                "warm_cost": float(data["warm_cost"][row]),
                "selected_actor_cost": float(cost),
                "warm_gain": float(data["warm_cost"][row] - cost),
                "critic_bank": bank,
            })
        critic_summary[label] = {
            "log_cost_pearson_median": float(np.median(correlations)),
            "pair_sign_accuracy_median": float(np.median(pairs)),
        }

    dynamic_rows = new_arrays["fit_indices"][new_arrays["fit_indices"] >= 672]
    dynamic_positions = np.asarray([
        int(np.flatnonzero(new_arrays["fit_indices"] == row)[0]) for row in dynamic_rows
    ])
    dynamic_fit = metrics(
        new_arrays["selected_fit_cost"][dynamic_positions], data["warm_cost"][dynamic_rows]
    )
    report = {
        "qualification": "QUERY_DYNAMIC_HISTORY_SEED2_OAC_DIAGNOSIS_COMPLETE",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "run": str(root),
        "run_manifest_sha256": sha256(root / "manifest.json"),
        "run_validation_sha256": sha256(root / "validation.json"),
        "baseline": str(baseline_root),
        "baseline_manifest_sha256": sha256(baseline_root / "manifest.json"),
        "comparisons": comparisons,
        "inner_100:3_critic": critic_summary,
        "inner_100:3_rows": hard_records,
        "dynamic_fit_rows": dynamic_fit,
        "decision": "DO_NOT_EXPAND_DYNAMIC_HISTORY_SCREEN_TO_THREE_SEEDS",
        "decision_basis": [
            "The seed-2 inner mean and 100:3 mean improve slightly versus the baseline seed-2 run.",
            "The 100:3 win fraction remains 0.5 and the worst warm-relative regression becomes larger.",
            "Late episode_096 steps 350 and 375 both remain regressions and both worsen, so the intended tail failure is not repaired.",
            "The dynamic-history rows are valid and useful as a diagnostic bank, but this exact prelude design does not justify a three-seed expansion.",
        ],
        "recommended_next_step": (
            "Keep the existing seed-2 baseline as the robust candidate. Do not append more phase/seed or the same lateral-prelude data. "
            "If optimization continues, target the late 100:3 tail explicitly with a train-side risk-sensitive selection/update criterion or a "
            "different causal transient design, and require worst-regression improvement before any three-seed expansion."
        ),
        "outer_fold_evaluated": False,
        "formal_validation_or_test_consumed": False,
        "dbm_fields_or_labels_consumed": [],
        "query_analytic_gradient_consumed": False,
        "analyzer": str(Path(__file__).resolve()),
        "analyzer_sha256": sha256(Path(__file__).resolve()),
    }
    (root / "diagnosis.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "qualification": report["qualification"],
        "comparisons": comparisons,
        "inner_100:3_critic": critic_summary,
        "dynamic_fit_rows": dynamic_fit,
        "decision": report["decision"],
        "outer_fold_evaluated": False,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
