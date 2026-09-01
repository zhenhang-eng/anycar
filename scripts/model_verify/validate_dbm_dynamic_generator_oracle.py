#!/usr/bin/env python3
"""Reconstruct dynamic banks and independently replay their audit costs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from evaluate_dbm_dynamic_generator_oracle import construct_banks
from generate_dbm_multicenter_teacher import make_controller
from generate_dbm_sampling_center_gt_pilot import evaluate_centers


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_dir", type=Path)
    parser.add_argument("--parent-labels", type=Path, required=True)
    parser.add_argument("--risk-labels", type=Path, required=True)
    parser.add_argument("--current-labels", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--fit-ridge", type=float, default=0.10)
    parser.add_argument("--step-damping", type=float, default=0.10)
    parser.add_argument("--minimum-step", type=float, default=0.20)
    parser.add_argument("--maximum-step", type=float, default=1.50)
    parser.add_argument("--response-maximum-step", type=float, default=2.00)
    args = parser.parse_args()
    device = torch.device(args.device)
    checked = 0
    maximum_center_error = 0.0
    maximum_cost_error = 0.0
    for result_path in sorted(args.result_dir.glob("episode_*_step_*.npz")):
        with np.load(result_path, allow_pickle=False) as result:
            source_path = Path(str(result["source_snapshot"]))
            episode = source_path.parents[1].name
            context = int(result["context_index"])
            with np.load(source_path, allow_pickle=False) as source, np.load(
                args.parent_labels / episode / source_path.name, allow_pickle=False
            ) as parent, np.load(
                args.risk_labels / episode / source_path.name, allow_pickle=False
            ) as risk, np.load(
                args.current_labels / episode / source_path.name, allow_pickle=False
            ) as current:
                config = {"objective": {"cost_weights": json.loads(str(source["cost_weights_json"]))}}
                controller, backend = make_controller(source, config, device)
                generated = construct_banks(
                    source, parent, risk, current, context, controller, backend, device, args
                )
                center_error = float(np.max(np.abs(generated["banks"] - result["centers"])))
                maximum_center_error = max(maximum_center_error, center_error)
                if center_error > 1e-6:
                    raise AssertionError(f"{result_path}: center reconstruction {center_error}")
                replay = []
                candidate_sigma = generated["sigma"] * float(result["candidate_noise_scale"])
                for centers in generated["banks"]:
                    replay.append(
                        evaluate_centers(
                            centers, list(np.asarray(result["audit_seeds"], int)),
                            controller, backend, generated["history"], generated["initial"],
                            generated["current_action"], generated["reference"],
                            candidate_sigma, generated["action_min"], generated["action_max"],
                        )["output_cost"]
                    )
                cost_error = float(np.max(np.abs(np.asarray(replay) - result["audit_output_cost"])))
                maximum_cost_error = max(maximum_cost_error, cost_error)
                if cost_error > 1e-4:
                    raise AssertionError(f"{result_path}: audit replay {cost_error}")
            checked += 1
    if checked == 0:
        raise FileNotFoundError(args.result_dir)
    print(json.dumps({
        "status": "PASS", "checked": checked,
        "maximum_center_error": maximum_center_error,
        "maximum_cost_error": maximum_cost_error,
    }, indent=2))


if __name__ == "__main__":
    main()
