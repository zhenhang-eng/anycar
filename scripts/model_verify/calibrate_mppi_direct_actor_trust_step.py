#!/usr/bin/env python3
"""Calibrate a single-network trust step for a deterministic Direct Actor.

The candidate Actor is a parameter-space interpolation between the frozen
bootstrap Actor and a learned Actor.  Every interpolation candidate is scored
with the real deterministic DBM direct cost on validation episodes.  Test
episodes are never loaded.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch

from car_foundation.mppi_proposal_policy import TorchMPPIDeterministicCenterActor
from train_mppi_direct_center_actor_critic import (
    DEFAULT_BOOTSTRAP,
    DEFAULT_LABELS,
    DEFAULT_PARENT,
    DEFAULT_RISK,
    DEFAULT_SOURCE,
    actor_outputs,
    build_inputs,
    evaluate_actor_direct,
    load_direct_partition,
)
from train_mppi_two_pass_feedback_critic import load_partition as load_state_partition


DEFAULT_LEARNED = Path(
    "outputs/mppi_proposal/direct_center_actor_critic_diverse_20260806_v1/"
    "direct_center_actor_critic.pt"
)
DEFAULT_OUTPUT = Path(
    "outputs/mppi_proposal/direct_center_actor_trust_step_20260806_v1"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--parent-labels", type=Path, default=DEFAULT_PARENT)
    parser.add_argument("--risk-labels", type=Path, default=DEFAULT_RISK)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--bootstrap", type=Path, default=DEFAULT_BOOTSTRAP)
    parser.add_argument("--learned", type=Path, default=DEFAULT_LEARNED)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--alphas", type=float, nargs="+",
        default=(0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0),
    )
    parser.add_argument("--minimum-win-fraction", type=float, default=0.50)
    parser.add_argument("--minimum-median-gain", type=float, default=0.0)
    parser.add_argument("--tail-p05-floor", type=float, default=-5.0)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def deterministic_bootstrap(checkpoint: dict, device: torch.device) -> TorchMPPIDeterministicCenterActor:
    actor = TorchMPPIDeterministicCenterActor(
        float(checkpoint["maximum_delta_sigma"]), dropout=0.0
    ).to(device)
    if checkpoint.get("actor_class") == "TorchMPPIDeterministicCenterActor":
        actor.load_state_dict(checkpoint["actor_state_dict"], strict=True)
    else:
        actor.load_stochastic_actor_state_dict(checkpoint["actor_state_dict"])
    actor.eval()
    return actor


def interpolated_state(
    bootstrap: dict[str, torch.Tensor],
    learned: dict[str, torch.Tensor],
    alpha: float,
) -> dict[str, torch.Tensor]:
    if bootstrap.keys() != learned.keys():
        raise ValueError("bootstrap and learned Actor state dictionaries differ")
    result = {}
    for name, initial in bootstrap.items():
        target = learned[name]
        if initial.shape != target.shape:
            raise ValueError(f"Actor tensor shape differs for {name}")
        result[name] = (
            initial + float(alpha) * (target - initial)
            if torch.is_floating_point(initial)
            else initial.clone()
        )
    return result


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    if not args.alphas or any(alpha < 0.0 or alpha > 1.0 for alpha in args.alphas):
        raise ValueError("alphas must be non-empty and within [0, 1]")
    args.output_dir.mkdir(parents=True)
    device = torch.device(args.device)
    bootstrap = torch.load(args.bootstrap, map_location="cpu")
    learned = torch.load(args.learned, map_location="cpu")
    maximum = float(bootstrap["maximum_delta_sigma"])
    if maximum != float(learned["maximum_delta_sigma"]):
        raise ValueError("maximum_delta_sigma differs between checkpoints")

    splits = json.loads((args.labels / "splits.json").read_text())
    validation_episodes = splits["validation"]
    state = load_state_partition(
        args.source, args.parent_labels, validation_episodes, "selection"
    )
    replay = load_direct_partition(
        args.labels, args.parent_labels, args.risk_labels,
        validation_episodes, maximum,
    )
    inputs = build_inputs(state, replay, bootstrap)

    initial_actor = deterministic_bootstrap(bootstrap, device)
    initial_state = {
        name: value.detach().cpu().clone()
        for name, value in initial_actor.state_dict().items()
    }
    learned_state = learned["actor_state_dict"]
    initial_action, _ = actor_outputs(
        initial_actor, inputs, args.batch_size, device
    )
    rows = []
    states = {}
    for alpha in sorted(set(float(value) for value in args.alphas)):
        actor = TorchMPPIDeterministicCenterActor(maximum, dropout=0.0).to(device)
        state_dict = interpolated_state(initial_state, learned_state, alpha)
        actor.load_state_dict(state_dict, strict=True)
        metrics = evaluate_actor_direct(
            actor, inputs, replay, args.source, args.batch_size, device
        )
        action, _ = actor_outputs(actor, inputs, args.batch_size, device)
        delta_sigma = np.abs(action - initial_action) * maximum
        state_rms_sigma = np.sqrt(np.mean((action - initial_action) ** 2, axis=(1, 2))) * maximum
        row = {
            "alpha": alpha,
            **metrics,
            "component_abs_delta_sigma_p95": float(np.quantile(delta_sigma, 0.95)),
            "component_abs_delta_sigma_max": float(delta_sigma.max()),
            "state_rms_delta_sigma_p95": float(np.quantile(state_rms_sigma, 0.95)),
            "state_rms_delta_sigma_max": float(state_rms_sigma.max()),
        }
        rows.append(row)
        states[alpha] = state_dict
        print(
            f"[alpha={alpha:.3f}] cost={metrics['actor_direct_cost_mean']:.3f} "
            f"median_gain={metrics['gain_vs_bootstrap_median']:.3f} "
            f"p05={metrics['gain_vs_bootstrap_p05']:.3f} "
            f"win={metrics['win_fraction']:.3f}",
            flush=True,
        )

    eligible = [
        row for row in rows
        if row["win_fraction"] >= args.minimum_win_fraction
        and row["gain_vs_bootstrap_median"] >= args.minimum_median_gain
    ]
    if not eligible:
        raise RuntimeError("no trust-step candidate passed median/win constraints")
    selected = min(eligible, key=lambda row: row["actor_direct_cost_mean"])
    tail_eligible = [
        row for row in eligible
        if row["gain_vs_bootstrap_p05"] >= args.tail_p05_floor
    ]
    tail_selected = (
        min(tail_eligible, key=lambda row: row["actor_direct_cost_mean"])
        if tail_eligible else None
    )

    def save_checkpoint(filename: str, row: dict) -> str:
        path = (args.output_dir / filename).resolve()
        checkpoint = {
            key: value for key, value in learned.items()
            if key not in ("actor_state_dict", "training_args")
        }
        checkpoint.update({
            "format_version": 1,
            "method": "validation-calibrated deterministic Direct-Actor trust step",
            "actor_class": "TorchMPPIDeterministicCenterActor",
            "actor_state_dict": states[row["alpha"]],
            "trust_step_alpha": row["alpha"],
            "source_bootstrap_checkpoint": str(args.bootstrap.resolve()),
            "source_learned_checkpoint": str(args.learned.resolve()),
            "validation_metrics": row,
            "qualification": "validation_only_test_sealed",
        })
        torch.save(checkpoint, path)
        return str(path)

    selected_path = save_checkpoint("direct_center_actor_trust_selected.pt", selected)
    tail_path = (
        save_checkpoint("direct_center_actor_trust_tail_guarded.pt", tail_selected)
        if tail_selected is not None else None
    )
    summary = {
        "format_version": 1,
        "method": "single-network parameter-space trust-step calibration",
        "qualification": "VALIDATION_ONLY_TEST_SEALED",
        "validation_episodes": validation_episodes,
        "validation_contexts": len(replay.action),
        "selection_constraints": {
            "minimum_win_fraction": args.minimum_win_fraction,
            "minimum_median_gain": args.minimum_median_gain,
        },
        "tail_guard_constraint": {"minimum_p05_gain": args.tail_p05_floor},
        "candidates": rows,
        "selected": selected,
        "selected_checkpoint": selected_path,
        "tail_guarded_selected": tail_selected,
        "tail_guarded_checkpoint": tail_path,
        "test_policy": "episode_105..119 not loaded or evaluated",
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
