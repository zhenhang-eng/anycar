#!/usr/bin/env python3
"""Qualify the shared residual-Actor runtime component against frozen contexts."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import torch

from car_foundation.mppi_residual_actor_runtime import (
    ResidualActorRuntime,
    ResidualActorRuntimeConfig,
)
from generate_dbm_multicenter_teacher import load_config, make_controller
from generate_dbm_proposal_teacher import sha256_file


DEFAULT_CHECKPOINT = Path(
    "outputs/mppi_proposal/direct_residual_online_ac_20260811_v2/"
    "direct_residual_online_ac_selected.pt"
)
DEFAULT_REFERENCE = Path(
    "outputs/mppi_proposal/two_center_integration_20260813_v2/evaluation.npz"
)
DEFAULT_CONFIG = Path(__file__).with_name("dbm_teacher_t1_diverse_20260805_v1.json")
DEFAULT_OUTPUT = Path(
    "outputs/mppi_proposal/residual_actor_runtime_component_validation_20260813_v2"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--first-seed", type=int, default=24001)
    parser.add_argument("--max-snapshots", type=int, default=0)
    parser.add_argument("--tolerance", type=float, default=5e-5)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)
    payload = torch.load(args.checkpoint, map_location="cpu")
    labels = Path(payload["labels"])
    splits = json.loads((labels / "splits.json").read_text())
    selected = set(splits["internal_selection"])
    records = [
        path for path in sorted(labels.glob("episode_*/*.npz"))
        if path.parent.name in selected
    ]
    if args.max_snapshots:
        records = records[:args.max_snapshots]

    references = {}
    with np.load(args.reference, allow_pickle=False) as values:
        for source, repeat, center in zip(
            values["source_path"], values["context_in_file"], values["actor_center"]
        ):
            references[(str(source), int(repeat))] = np.asarray(center, np.float32)

    runtime = ResidualActorRuntime(
        args.checkpoint,
        config=ResidualActorRuntimeConfig(first_seed=args.first_seed),
        device=args.device,
    )
    generator_config = load_config(args.config)
    errors = {name: [] for name in (
        "bc_center", "first_knots", "first_cost", "guided_center", "feedback",
        "gradient_mean", "gradient_std", "old_center", "proposal_center", "final_center",
    )}
    durations = []
    for index, label_path in enumerate(records, 1):
        with np.load(label_path, allow_pickle=False) as label:
            source_path = Path(str(label["source_snapshot"]))
            context_path = Path(str(label["context_label"]))
            with np.load(source_path, allow_pickle=False) as source, np.load(
                context_path, allow_pickle=False
            ) as cached:
                seeds = np.asarray(cached["first_pass_seed"], np.int64).tolist()
                repeat = seeds.index(args.first_seed)
                controller, _ = make_controller(source, generator_config, torch.device(args.device))
                output = runtime.propose(
                    controller,
                    source["initial_state"],
                    source["current_action"],
                    source["history"],
                    source["reference"],
                    source["sampling_mean_knots"],
                    reference_ego=source["reference_ego"],
                )
                actual_expected = {
                    "bc_center": (output.bc_center_knots.cpu().numpy(), cached["base_center_knots"]),
                    "first_knots": (output.first_pass_knots.cpu().numpy(), cached["first_pass_knots"][repeat]),
                    "first_cost": (output.first_pass_cost.cpu().numpy(), cached["first_pass_cost"][repeat]),
                    "guided_center": (output.guided_center_knots.cpu().numpy(), cached["guided_center_knots"][repeat]),
                    "feedback": (output.first_pass_feedback.cpu().numpy(), cached["first_pass_feedback"][repeat]),
                    "gradient_mean": (output.critic_gradient_mean.cpu().numpy(), cached["critic_gradient_mean"][repeat]),
                    "gradient_std": (output.critic_gradient_std.cpu().numpy(), cached["critic_gradient_std"][repeat]),
                    "old_center": (output.old_center_knots.cpu().numpy(), label["old_actor_center"][repeat]),
                    "proposal_center": (output.proposal_center_knots.cpu().numpy(), label["proposal_actor_center"][repeat]),
                    "final_center": (output.center_knots.cpu().numpy(), references[(str(source_path), repeat)]),
                }
                for name, (actual, expected) in actual_expected.items():
                    errors[name].append(float(np.max(np.abs(
                        np.asarray(actual) - np.asarray(expected)
                    ))))
                durations.append(output.duration_s)
                if output.rollout_count != 129:
                    raise AssertionError("runtime first-pass rollout count is not 129")
        if index % 25 == 0 or index == len(records):
            print(f"[{index:03d}/{len(records):03d}] final={max(errors['final_center']):.3e}", flush=True)

    maxima = {name: float(max(values, default=0.0)) for name, values in errors.items()}
    tolerances = {
        "first_cost": 5e-4,
        "feedback": 5e-4,
        "gradient_mean": 5e-4,
        "gradient_std": 5e-4,
        "default": args.tolerance,
    }
    failures = {
        name: value for name, value in maxima.items()
        if value > tolerances.get(name, tolerances["default"])
    }
    result = {
        "format_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "qualification": "PASS_RESIDUAL_ACTOR_RUNTIME_COMPONENT" if not failures else "FAIL_RESIDUAL_ACTOR_RUNTIME_COMPONENT",
        "partition": "consumed internal_selection mechanism diagnostic",
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "reference_sha256": sha256_file(args.reference),
        "snapshot_count": len(records),
        "context_count": len(records),
        "first_seed": args.first_seed,
        "runtime_first_pass_rollouts": 129,
        "strict_guard_total_rollouts": 387,
        "maximum_absolute_error": maxima,
        "tolerance": tolerances,
        "failures": failures,
        "runtime_duration_seconds": {
            "mean": float(np.mean(durations)),
            "p95": float(np.quantile(durations, 0.95)),
            "maximum": float(np.max(durations)),
        },
        "test_policy": "formal validation and test remain sealed",
        "caveat": "This qualifies the shared component on frozen states, not closed-loop behavior.",
    }
    (args.output_dir / "validation.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)
    if failures:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
