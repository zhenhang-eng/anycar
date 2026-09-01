#!/usr/bin/env python3
"""Paired analysis for the OAC-2 box1/box3 output-support experiment."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import torch

from run_mppi_absolute_action_value_critic_cv import make_folds
from train_mppi_oac2_continuous_actor import internal_split, load_actor
from train_mppi_online_absolute_sac import (
    actor_mean,
    load_actor_normalization,
    load_bank,
    make_actor_inputs,
)
from validate_mppi_oac2_continuous_actor import load_actor_checkpoint


DEFAULT_BOX1 = Path(
    "outputs/mppi_proposal/online_absolute_sac_oac2_support_box1_200round_20260825_v1"
)
DEFAULT_BOX3 = Path(
    "outputs/mppi_proposal/online_absolute_sac_oac2_support_box3_200round_20260825_v1"
)
DEFAULT_OUTPUT = Path("outputs/mppi_proposal/online_absolute_sac_oac2_support_ab_20260825_v1")


def average(rows: list[float]) -> float:
    return float(np.mean(np.asarray(rows, dtype=np.float64)))


def support_usage(actor, action: np.ndarray) -> dict[str, float]:
    center = actor.out_center.detach().cpu().numpy()
    scale = actor.out_scale.detach().cpu().numpy()
    multiplier = float(getattr(actor, "output_support_multiplier", 1.0))
    normalized = np.abs((action - center) / (scale * multiplier))
    return {
        "component_ge_0_95": float(np.mean(normalized >= 0.95)),
        "state_any_ge_0_95": float(np.mean(np.any(normalized >= 0.95, axis=(1, 2)))),
        "steering_component_ge_0_95": float(np.mean(normalized[:, :, 1] >= 0.95)),
        "acceleration_component_ge_0_95": float(np.mean(normalized[:, :, 0] >= 0.95)),
    }


def metric_view(record: dict, role: str) -> dict:
    metrics = record[f"{role}_metrics"]
    return {
        "selected_round": int(record["selected_round"]),
        "gain_mean": float(metrics["gain_vs_initial"]["mean"]),
        "gain_median": float(metrics["gain_vs_initial"]["median"]),
        "gain_p05": float(metrics["gain_vs_initial"]["p05"]),
        "gain_worst": float(metrics["gain_vs_initial"]["minimum"]),
        "regression_fraction": float(metrics["regression_fraction"]),
        "speed_2_4_gain_p05": float(metrics["by_speed"]["2.4"]["p05_gain"]),
        "speed_2_8_gain_p05": float(metrics["by_speed"]["2.8"]["p05_gain"]),
        "guard_mean_cost": float(metrics["two_center_guard"]["cost"]["mean"]),
        "guard_median_cost": float(metrics["two_center_guard"]["cost"]["median"]),
        "critic_pair_accuracy": float(
            record["critic_metrics"]["actor_visited_material_pair_accuracy"]
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--box1", type=Path, default=DEFAULT_BOX1)
    parser.add_argument("--box3", type=Path, default=DEFAULT_BOX3)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    runs = {}
    for name, path in (("box1", args.box1), ("box3", args.box3)):
        runs[name] = {
            "path": path,
            "contract": json.loads((path / "contract.json").read_text()),
            "summary": json.loads((path / "summary.json").read_text()),
            "validator": json.loads((path / "validator_report.json").read_text()),
        }
    contract1 = runs["box1"]["contract"]
    contract3 = runs["box3"]["contract"]
    arguments1 = dict(contract1["arguments"])
    arguments3 = dict(contract3["arguments"])
    ignored = {"output_dir", "actor_output_support_multiplier"}
    compared_keys = sorted((set(arguments1) | set(arguments3)) - ignored)
    paired_arguments = {
        key: arguments1[key] == arguments3[key]
        for key in compared_keys
    }
    if not all(paired_arguments.values()):
        raise AssertionError("non-support arguments are not paired")
    if float(arguments1["actor_output_support_multiplier"]) != 1.0:
        raise AssertionError("box1 support multiplier mismatch")
    if float(arguments3["actor_output_support_multiplier"]) != 3.0:
        raise AssertionError("box3 support multiplier mismatch")
    if float(arguments1["tail_regression_weight"]) != 0.0:
        raise AssertionError("fixed tail CVaR loss must remain disabled")

    bank_root = Path(arguments1["bank_root"])
    actor_root = Path(arguments1["actor_root"])
    base_ac = Path(arguments1["base_ac"])
    data = load_bank(bank_root)
    folds = make_folds(data, 3)
    outer_fold = int(contract1["outer_fold"])
    _, selection, _, _ = internal_split(data, folds, outer_fold)
    normalization, _ = load_actor_normalization(base_ac)
    actor_inputs = make_actor_inputs(data, normalization)
    device = torch.device(args.device)

    by_seed = []
    initial_max_error = 0.0
    for seed in (0, 1, 2):
        source_path = actor_root / f"a0_fold{outer_fold}_seed{seed}.pt"
        initial1, _ = load_actor(source_path, outer_fold, seed, device, 1.0)
        initial3, _ = load_actor(source_path, outer_fold, seed, device, 3.0)
        action1 = actor_mean(initial1, actor_inputs, selection, device)
        action3 = actor_mean(initial3, actor_inputs, selection, device)
        initial_error = float(np.max(np.abs(action1 - action3)))
        initial_max_error = max(initial_max_error, initial_error)
        row = {"seed": seed, "initial_action_max_abs_error": initial_error, "arms": {}}
        for name in ("box1", "box3"):
            source = next(
                value for value in runs[name]["summary"]["records"]
                if int(value["seed"]) == seed
            )
            selected_actor, _ = load_actor_checkpoint(
                runs[name]["path"] / f"seed_{seed}" / "actor_selected.pt", device
            )
            latest_actor, _ = load_actor_checkpoint(
                runs[name]["path"] / f"seed_{seed}" / "actor_latest.pt", device
            )
            selected_action = actor_mean(selected_actor, actor_inputs, selection, device)
            latest_action = actor_mean(latest_actor, actor_inputs, selection, device)
            failures = Counter()
            accepted_rounds = []
            for evaluation in source["evaluations"][1:]:
                if evaluation["accepted"]:
                    accepted_rounds.append(int(evaluation["round"]))
                for gate, passed in evaluation["acceptance_gate"].items():
                    if not passed:
                        failures[gate] += 1
            row["arms"][name] = {
                "selected": metric_view(source, "selected"),
                "latest": metric_view(source, "latest"),
                "selected_support_usage": support_usage(selected_actor, selected_action),
                "latest_support_usage": support_usage(latest_actor, latest_action),
                "accepted_rounds": accepted_rounds,
                "gate_failure_count": dict(sorted(failures.items())),
            }
        row["delta_box3_minus_box1"] = {
            role: {
                key: row["arms"]["box3"][role][key] - row["arms"]["box1"][role][key]
                for key in (
                    "gain_mean", "gain_median", "gain_p05", "gain_worst",
                    "regression_fraction", "speed_2_4_gain_p05",
                    "speed_2_8_gain_p05", "guard_mean_cost",
                )
            }
            for role in ("selected", "latest")
        }
        by_seed.append(row)

    aggregate = {}
    for role in ("selected", "latest"):
        aggregate[role] = {}
        for key in (
            "gain_mean", "gain_median", "gain_p05", "gain_worst",
            "regression_fraction", "speed_2_4_gain_p05",
            "speed_2_8_gain_p05", "guard_mean_cost", "critic_pair_accuracy",
        ):
            aggregate[role][key] = {
                arm: average([row["arms"][arm][role][key] for row in by_seed])
                for arm in ("box1", "box3")
            }
            aggregate[role][key]["box3_minus_box1"] = (
                aggregate[role][key]["box3"] - aggregate[role][key]["box1"]
            )
    selected_gain_better = all(
        row["delta_box3_minus_box1"]["selected"]["gain_mean"] > 0 for row in by_seed
    )
    latest_gain_better = all(
        row["delta_box3_minus_box1"]["latest"]["gain_mean"] > 0 for row in by_seed
    )
    same_selected_rounds = all(
        row["arms"]["box1"]["selected"]["selected_round"]
        == row["arms"]["box3"]["selected"]["selected_round"]
        for row in by_seed
    )
    validators_pass = all(
        runs[name]["validator"]["qualification"] == "OAC2_CONTINUOUS_ACTOR_VALIDATION_PASS"
        for name in ("box1", "box3")
    )
    qualification = (
        "BOX3_OAC_IMPROVES_GAIN_WITH_FIXED_TAIL_GATES_BUT_NO_SAFE_HORIZON_EXTENSION"
        if (
            initial_max_error <= 2e-6 and validators_pass and selected_gain_better
            and latest_gain_better and same_selected_rounds
        )
        else "BOX3_OAC_SUPPORT_AB_INCONCLUSIVE_OR_FAIL"
    )
    result = {
        "format_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "qualification": qualification,
        "contract": {
            "paired_non_support_arguments": paired_arguments,
            "initial_action_max_abs_error": initial_max_error,
            "tail_regression_weight": 0.0,
            "tail_floors": {
                "overall_p05": float(arguments1["selection_gain_p05_floor"]),
                "speed_2_4_p05": float(arguments1["selection_speed_2_4_gain_p05_floor"]),
                "speed_2_8_p05": float(arguments1["selection_speed_2_8_gain_p05_floor"]),
            },
            "formal_validation_loaded": False,
            "test_loaded": False,
        },
        "decision": {
            "validators_pass": validators_pass,
            "selected_gain_better_all_seeds": selected_gain_better,
            "latest_gain_better_all_seeds": latest_gain_better,
            "same_selected_rounds_all_seeds": same_selected_rounds,
            "box3_is_preferred_output_contract": selected_gain_better and validators_pass,
            "fixed_tail_cvar_loss_should_remain_disabled": True,
            "next_blocker": "state-wise tail control; support no longer primary",
        },
        "aggregate": aggregate,
        "by_seed": by_seed,
    }
    (args.output_dir / "analysis.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({
        "qualification": qualification,
        "initial_action_max_abs_error": initial_max_error,
        "selected_gain_mean": aggregate["selected"]["gain_mean"],
        "latest_gain_mean": aggregate["latest"]["gain_mean"],
        "selected_gain_p05": aggregate["selected"]["gain_p05"],
        "latest_gain_p05": aggregate["latest"]["gain_p05"],
        "same_selected_rounds": same_selected_rounds,
    }, indent=2))


if __name__ == "__main__":
    main()
