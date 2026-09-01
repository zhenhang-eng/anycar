#!/usr/bin/env python3
"""Independently reconstruct and replay the frozen Alpha AC TR3 output."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np
import torch

from evaluate_mppi_direct_alpha_ac_tr3 import (
    DEFAULT_OUTPUT,
    load_alpha_policy,
    summarize_arrays,
)
from generate_dbm_direct_trust_region_labels import (
    actor_inputs,
    line_centers,
    load_actor,
)
from generate_dbm_multicenter_teacher import evaluate_knots, make_controller
from generate_dbm_proposal_teacher import sha256_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path, nargs="?", default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def numeric_max_error(left, right) -> float:
    values = []
    if isinstance(left, dict) and isinstance(right, dict):
        for key in left.keys() & right.keys():
            values.append(numeric_max_error(left[key], right[key]))
    elif isinstance(left, (int, float)) and isinstance(right, (int, float)):
        values.append(abs(float(left) - float(right)))
    return max(values, default=0.0)


def main() -> None:
    args = parse_args()
    summary = json.loads((args.run / "summary.json").read_text())
    config = summary["config"]
    alpha_ac_path = Path(config["alpha_ac"])
    tr2b_path = Path(config["tr2b"])
    if sha256_file(alpha_ac_path) != config["alpha_ac_sha256"]:
        raise AssertionError("Alpha AC checkpoint hash mismatch")
    if sha256_file(tr2b_path) != config["tr2b_sha256"]:
        raise AssertionError("TR2-B checkpoint hash mismatch")
    ac_payload = torch.load(alpha_ac_path, map_location="cpu")
    tr2b_payload = torch.load(tr2b_path, map_location="cpu")
    old_path = Path(config["old_actor"])
    proposal_path = Path(config["proposal_actor"])
    device = torch.device(args.device)
    old_payload, old_actor = load_actor(old_path, device)
    proposal_payload, proposal_actor = load_actor(proposal_path, device)
    tr2b_policy = load_alpha_policy(tr2b_path, tr2b_payload, old_payload, device)
    ac_policy = load_alpha_policy(alpha_ac_path, ac_payload, old_payload, device)

    arrays: dict[str, list] = defaultdict(list)
    errors: dict[str, float] = defaultdict(float)
    label_paths = sorted(args.run.glob("episode_*/*.npz"))
    if len(label_paths) != 300:
        raise AssertionError(f"expected 300 TR3 labels, found {len(label_paths)}")
    expected_episodes = set(config["validation_episodes"])
    if {path.parent.name for path in label_paths} != expected_episodes:
        raise AssertionError("TR3 output episode set changed")
    if any(int(path.parent.name.split("_")[1]) >= 105 for path in label_paths):
        raise AssertionError("test episode found in TR3 output")

    for index, label_path in enumerate(label_paths, 1):
        with np.load(label_path, allow_pickle=False) as label:
            source_path = Path(str(label["source_snapshot"]))
            context_path = Path(str(label["context_label"]))
            risk_path = Path(str(label["risk_label"]))
            if sha256_file(source_path) != str(label["source_snapshot_sha256"]):
                raise AssertionError("saved source hash mismatch")
            if sha256_file(context_path) != str(label["context_label_sha256"]):
                raise AssertionError("saved context hash mismatch")
            if sha256_file(risk_path) != str(label["risk_label_sha256"]):
                raise AssertionError("saved risk hash mismatch")
            with np.load(source_path, allow_pickle=False) as source, np.load(
                context_path, allow_pickle=False
            ) as context, np.load(risk_path, allow_pickle=False) as risk:
                gradient_mean = np.asarray(risk["critic_gradient_mean"], np.float32)
                gradient_std = np.asarray(risk["critic_gradient_std"], np.float32)
                objective = {"objective": {"cost_weights": json.loads(str(source["cost_weights_json"]))}}
                controller, backend = make_controller(source, objective, device)
                params = json.loads(str(source["mppi_params_json"]))
                sigma = np.asarray(params["noise_sigma"], np.float32)
                low = np.asarray(params["action_min"], np.float32)
                high = np.asarray(params["action_max"], np.float32)
                history = torch.from_numpy(source["history"]).to(device)
                initial = torch.from_numpy(source["initial_state"]).to(device).reshape(1, 5)
                current_action = torch.from_numpy(source["current_action"]).to(device).reshape(1, 2)
                reference = controller._prepare_reference(source["reference"])
                for repeat in range(len(label["old_center"])):
                    old_input = actor_inputs(
                        source, context, gradient_mean, gradient_std,
                        repeat, old_payload, device,
                    )
                    proposal_input = actor_inputs(
                        source, context, gradient_mean, gradient_std,
                        repeat, proposal_payload, device,
                    )
                    with torch.no_grad():
                        _, old_center_t = old_actor(*old_input)
                        _, proposal_center_t = proposal_actor(*proposal_input)
                    old_center = old_center_t[0].cpu().numpy().astype(np.float32)
                    proposal_center = proposal_center_t[0].cpu().numpy().astype(np.float32)
                    line = line_centers(
                        old_center, proposal_center, sigma, low, high, 0.5,
                        np.asarray(label["alpha_grid"], np.float32),
                    )
                    direction = torch.from_numpy(line["projected_direction"][None]).to(device)
                    rho = torch.tensor([[line["requested_rho"]]], dtype=torch.float32, device=device)
                    scale = torch.tensor([[line["trust_scale"]]], dtype=torch.float32, device=device)
                    with torch.no_grad():
                        _, p_tr2b, a_tr2b = tr2b_policy(*old_input, direction, rho, scale)
                        _, p_ac, a_ac = ac_policy(*old_input, direction, rho, scale)
                    alpha_tr2b = tr2b_policy.hard_alpha(
                        p_tr2b, a_tr2b, float(config["tr2b_threshold"])
                    )
                    alpha_ac = ac_policy.hard_alpha(
                        p_ac, a_ac, float(config["alpha_ac_threshold"])
                    )
                    def center(alpha_value: float) -> np.ndarray:
                        raw = old_center + alpha_value * sigma.reshape(1, 2) * line["projected_direction"]
                        return np.clip(raw, low, high).astype(np.float32)
                    tr2b_center = center(float(alpha_tr2b.item()))
                    ac_center = center(float(alpha_ac.item()))
                    bank = np.concatenate((
                        line["centers"], tr2b_center[None], ac_center[None]
                    ), axis=0)
                    cost, _, _ = evaluate_knots(
                        controller, backend, bank, history, initial,
                        current_action, reference,
                    )
                    comparisons = {
                        "old_center": (old_center, label["old_center"][repeat]),
                        "proposal_center": (proposal_center, label["proposal_center"][repeat]),
                        "projected_direction": (line["projected_direction"], label["projected_direction"][repeat]),
                        "line_center": (line["centers"], label["line_centers"][repeat]),
                        "tr2b_center": (tr2b_center, label["tr2b_center"][repeat]),
                        "alpha_ac_center": (ac_center, label["alpha_ac_center"][repeat]),
                        "line_cost": (cost[:-2], label["line_cost"][repeat]),
                        "tr2b_cost": (np.asarray(cost[-2]), label["tr2b_cost"][repeat]),
                        "alpha_ac_cost": (np.asarray(cost[-1]), label["alpha_ac_cost"][repeat]),
                        "tr2b_probability": (p_tr2b.cpu().numpy()[0], label["tr2b_probability"][repeat]),
                        "alpha_ac_probability": (p_ac.cpu().numpy()[0], label["alpha_ac_probability"][repeat]),
                        "tr2b_alpha": (alpha_tr2b.cpu().numpy()[0], label["tr2b_alpha"][repeat]),
                        "alpha_ac_alpha": (alpha_ac.cpu().numpy()[0], label["alpha_ac_alpha"][repeat]),
                    }
                    for name, (actual, expected) in comparisons.items():
                        errors[name] = max(
                            errors[name], float(np.max(np.abs(
                                np.asarray(actual, np.float64) - np.asarray(expected, np.float64)
                            )))
                        )
                    safe = int(label["safe_index"][repeat])
                    argmin = int(label["argmin_index"][repeat])
                    values = {
                        "old_cost": float(cost[0]),
                        "tr2b_cost": float(cost[-2]),
                        "alpha_ac_cost": float(cost[-1]),
                        "safe_cost": float(cost[safe]),
                        "argmin_cost": float(cost[argmin]),
                        "tr2b_alpha": float(alpha_tr2b.item()),
                        "alpha_ac_alpha": float(alpha_ac.item()),
                        "reference_speed": float(label["reference_speed_mps"]),
                        "scenario": str(label["scenario"]),
                        "episode": str(label["episode"]),
                        "snapshot_key": f"{label_path.parent.name}/{label_path.stem}",
                    }
                    for name, value in values.items():
                        arrays[name].append(value)
        if index == 1 or index % 50 == 0 or index == len(label_paths):
            print(f"[{index:03d}/{len(label_paths):03d}] replayed", flush=True)

    replay_metrics = summarize_arrays({name: np.asarray(value) for name, value in arrays.items()})
    metric_error = numeric_max_error(summary["metrics"], replay_metrics)
    maximum_center_error = max(
        errors[name] for name in errors if "center" in name or "direction" in name
    )
    maximum_cost_error = max(errors[name] for name in errors if "cost" in name)
    qualification = (
        "TR3_INDEPENDENTLY_VALIDATED"
        if maximum_center_error <= 2e-6
        and maximum_cost_error <= 1e-5
        and metric_error <= 1e-5
        and replay_metrics["qualification"] == summary["qualification"]
        else "TR3_INDEPENDENT_VALIDATION_FAIL"
    )
    report = {
        "format_version": 1,
        "run": str(args.run.resolve()),
        "label_count": len(label_paths),
        "context_count": len(arrays["old_cost"]),
        "maximum_reconstruction_error": dict(errors),
        "maximum_center_error": maximum_center_error,
        "maximum_cost_error": maximum_cost_error,
        "summary_metric_max_error": metric_error,
        "replay_metrics": replay_metrics,
        "source_qualification": summary["qualification"],
        "qualification": qualification,
        "deviation": "D0",
        "test_policy": "test episodes 105--119 not opened",
    }
    (args.run / "validation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
