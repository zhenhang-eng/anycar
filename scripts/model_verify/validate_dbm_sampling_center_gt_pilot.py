#!/usr/bin/env python3
"""Replay audit costs saved by generate_dbm_sampling_center_gt_pilot.py."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from generate_dbm_multicenter_teacher import make_controller
from generate_dbm_sampling_center_gt_pilot import evaluate_centers


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_dir", type=Path)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    with np.load(args.result_dir / "center_oracle.npz", allow_pickle=False) as result:
        if str(result["candidate_noise_design"]) != "zero_extra_antithetic_pairs":
            raise ValueError("unsupported candidate noise design")
        snapshot_path = Path(str(result["source_snapshot"]))
        with np.load(snapshot_path, allow_pickle=False) as source:
            config = {"objective": {"cost_weights": json.loads(str(source["cost_weights_json"]))}}
            controller, backend = make_controller(source, config, torch.device(args.device))
            history = torch.from_numpy(source["history"]).to(args.device)
            initial = torch.from_numpy(source["initial_state"]).to(args.device).reshape(1, 5)
            current = torch.from_numpy(source["current_action"]).to(args.device).reshape(1, 2)
            reference = controller._prepare_reference(source["reference"])
            params = json.loads(str(source["mppi_params_json"]))
            replay = evaluate_centers(
                np.asarray(result["comparison_centers"]),
                list(np.asarray(result["audit_seeds"], dtype=int)),
                controller, backend, history, initial, current, reference,
                np.asarray(params["noise_sigma"], np.float32)
                * float(result["candidate_noise_scale"]),
                np.asarray(params["action_min"], np.float32),
                np.asarray(params["action_max"], np.float32),
            )
            saved = np.asarray(result["audit_output_cost"])
            error = float(np.max(np.abs(replay["output_cost"] - saved)))
            if not np.allclose(replay["output_cost"], saved, rtol=1e-5, atol=1e-4):
                raise AssertionError(f"audit output-cost mismatch: {error}")
    print(json.dumps({"status": "PASS", "maximum_cost_error": error}, indent=2))


if __name__ == "__main__":
    main()
