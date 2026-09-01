#!/usr/bin/env python3
"""Validate direct-center replay hashes, reconstruction, split isolation, and DBM cost."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from car_foundation.mppi_proposal_policy import TorchMPPIDeterministicCenterActor
from evaluate_mppi_continuous_center_bootstrap import actor_inputs
from generate_dbm_direct_center_replay import (
    DEFAULT_BANK,
    DEFAULT_CHECKPOINT,
    DEFAULT_OUTPUT,
    DEFAULT_PARENT,
    DEFAULT_RISK,
    DEFAULT_SOURCE,
    DEFAULT_T1,
    center_names,
    local_centers,
    local_rank,
)
from generate_dbm_fullrank_local_labels import hadamard_directions
from generate_dbm_multicenter_teacher import evaluate_knots, make_controller
from generate_dbm_proposal_teacher import sha256_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("labels", type=Path, nargs="?", default=DEFAULT_OUTPUT)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--parent-labels", type=Path, default=DEFAULT_PARENT)
    parser.add_argument("--risk-labels", type=Path, default=DEFAULT_RISK)
    parser.add_argument("--bank-labels", type=Path, default=DEFAULT_BANK)
    parser.add_argument("--t1-labels", type=Path, default=DEFAULT_T1)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--max-snapshots", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = json.loads((args.labels / "summary.json").read_text())
    splits = json.loads((args.labels / "splits.json").read_text())
    sealed = set(splits["test_sealed_not_generated"])
    present = {path.name for path in args.labels.glob("episode_*") if path.is_dir()}
    if present & sealed:
        raise AssertionError("sealed test episode labels are present")
    expected = set(splits["train"]) | set(splits["validation"])
    if not present <= expected:
        raise AssertionError("unexpected episode directory in labels")

    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    if sha256_file(args.checkpoint) != summary["checkpoint_sha256"]:
        raise AssertionError("checkpoint hash mismatch")
    actor = TorchMPPIDeterministicCenterActor(
        float(checkpoint["maximum_delta_sigma"]), dropout=0.0
    ).to(device)
    if checkpoint.get("actor_class") == "TorchMPPIDeterministicCenterActor":
        actor.load_state_dict(checkpoint["actor_state_dict"], strict=True)
    else:
        actor.load_stochastic_actor_state_dict(checkpoint["actor_state_dict"])
    actor.eval()
    directions = hadamard_directions().astype(np.float32)
    paths = sorted(args.labels.glob("episode_*/*.npz"))
    if args.max_snapshots:
        paths = paths[: args.max_snapshots]
    maximum_center_error = 0.0
    maximum_cost_error = 0.0
    rank_min = 16
    context_count = 0
    for index, path in enumerate(paths, start=1):
        episode = path.parent.name
        source_path = args.source / episode / "snapshots" / path.name
        parent_path = args.parent_labels / episode / path.name
        risk_path = args.risk_labels / episode / path.name
        bank_path = args.bank_labels / episode / path.name
        t1_path = args.t1_labels / episode / path.name
        with np.load(path, allow_pickle=False) as label, np.load(
            source_path, allow_pickle=False
        ) as source, np.load(parent_path, allow_pickle=False) as parent, np.load(
            risk_path, allow_pickle=False
        ) as risk, np.load(bank_path, allow_pickle=False) as bank, np.load(
            t1_path, allow_pickle=False
        ) as t1:
            for key, source_file in (
                ("source_snapshot_sha256", source_path),
                ("parent_label_sha256", parent_path),
                ("risk_label_sha256", risk_path),
                ("bank_label_sha256", bank_path),
                ("t1_label_sha256", t1_path),
            ):
                if str(label[key]) != sha256_file(source_file):
                    raise AssertionError(f"{path}: {key} mismatch")
            config = {
                "objective": {
                    "cost_weights": json.loads(str(source["cost_weights_json"]))
                }
            }
            controller, backend = make_controller(source, config, device)
            params = json.loads(str(source["mppi_params_json"]))
            sigma = np.asarray(params["noise_sigma"], np.float32)
            action_min = np.asarray(params["action_min"], np.float32)
            action_max = np.asarray(params["action_max"], np.float32)
            radii = tuple(float(value) for value in label["local_radii_sigma"])
            anchors = np.asarray(parent["guided_center_knots"], np.float32)
            stored_bank = np.asarray(bank["centers"], np.float32)
            teacher = np.asarray(t1["teacher_center_knots"], np.float32)
            expected_names = center_names(len(stored_bank[0]), radii)
            if tuple(label["center_names"].astype(str)) != expected_names:
                raise AssertionError(f"{path}: center names mismatch")
            history = torch.from_numpy(source["history"]).to(device)
            initial = torch.from_numpy(source["initial_state"]).to(device).reshape(1, 5)
            current_action = torch.from_numpy(source["current_action"]).to(device).reshape(1, 2)
            reference = controller._prepare_reference(source["reference"])
            for context in range(len(anchors)):
                model_input = actor_inputs(
                    source, parent, risk, context, checkpoint, device
                )
                with torch.no_grad():
                    expected_action, expected_actor_tensor = actor(*model_input)
                expected_actor = expected_actor_tensor[0].cpu().numpy().astype(np.float32)
                raw_local, clipped_local = local_centers(
                    expected_actor, sigma, directions, radii, action_min, action_max
                )
                expected_centers = np.concatenate((
                    anchors[context][None], stored_bank[context], teacher[None],
                    expected_actor[None], clipped_local,
                )).astype(np.float32)
                maximum_center_error = max(
                    maximum_center_error,
                    float(np.max(np.abs(expected_centers - label["centers"][context]))),
                    float(np.max(np.abs(raw_local - label["raw_local_centers"][context]))),
                    float(np.max(np.abs(
                        expected_action[0].cpu().numpy() - label["bootstrap_actor_action"][context]
                    ))),
                )
                cost, _, _ = evaluate_knots(
                    controller, backend, expected_centers, history, initial,
                    current_action, reference,
                )
                maximum_cost_error = max(
                    maximum_cost_error,
                    float(np.max(np.abs(cost - label["direct_cost"][context]))),
                )
                one_rank = local_rank(clipped_local, expected_actor, sigma)
                rank_min = min(rank_min, one_rank)
                if one_rank != int(label["local_direction_rank"][context]):
                    raise AssertionError(f"{path}: local rank mismatch")
                context_count += 1
        if index == 1 or index % 50 == 0 or index == len(paths):
            print(
                f"[{index:04d}/{len(paths):04d}] contexts={context_count} "
                f"center_err={maximum_center_error:.3g} cost_err={maximum_cost_error:.3g}",
                flush=True,
            )
    if not args.max_snapshots:
        if len(paths) != int(summary["snapshot_count"]):
            raise AssertionError("snapshot count mismatch")
        if context_count != int(summary["context_count"]):
            raise AssertionError("context count mismatch")
    if maximum_center_error > 1e-6 or maximum_cost_error > 1e-5:
        raise AssertionError("reconstruction tolerance exceeded")
    print(json.dumps({
        "validated_snapshots": len(paths),
        "validated_contexts": context_count,
        "maximum_center_error": maximum_center_error,
        "maximum_cost_error": maximum_cost_error,
        "local_direction_rank_min": rank_min,
        "sealed_test_episode_count": len(sealed),
    }, indent=2))


if __name__ == "__main__":
    main()
