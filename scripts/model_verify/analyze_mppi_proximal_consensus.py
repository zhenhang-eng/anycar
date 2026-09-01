#!/usr/bin/env python3
"""Phase 1a post-analysis: consensus-center evaluation across seed banks.

The seed banks at the 128 budget find similar-cost teachers in different
directions (soft basin). Before any Actor distillation, check whether the
per-state mean of the three bank teachers (the target an MSE Actor would
converge toward) is itself a good center. Reads the Phase 1a manifest and
labels, evaluates the consensus centers, and reports cost against the bank
teachers and anchors. No anchor is updated.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from car_dynamics.controllers_torch.dbm import (
    TorchDBMParams,
    TorchDynamicBicycleRolloutBackend,
)
from car_dynamics.controllers_torch.mppi import (
    TorchMPPICostWeights,
    TorchMPPIParams,
)
from generate_dbm_direct_gt_validation import batched_cost, interpolate_knots
from run_mppi_proximal_search_phase1a import (
    DEFAULT_GT_TRAIN,
    DEFAULT_REPLAY_LABELS,
    DEFAULT_SCENARIO_PLAN,
    load_states,
)

BANK_KEYS = ("c_prefix_128", "c_bank2_128", "c_bank3_128")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path)
    parser.add_argument("--repeat", type=int, default=0)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run = args.run
    output_path = run / "consensus_analysis.json"
    if output_path.exists():
        raise FileExistsError(output_path)
    manifest = json.loads((run / "manifest.json").read_text())
    labels = np.load(run / "labels.npz", allow_pickle=False)
    episodes = [str(value) for value in labels["episodes"]]
    manifest_keys = [
        f"{row['episode']}#{row['snapshot']}" for row in manifest["states"]
    ]
    if episodes != manifest_keys:
        raise AssertionError("labels/manifest order mismatch")

    loader_args = SimpleNamespace(
        replay_labels=DEFAULT_REPLAY_LABELS,
        gt_train=DEFAULT_GT_TRAIN,
        scenario_plan=DEFAULT_SCENARIO_PLAN,
        repeat=args.repeat,
    )
    by_key = {
        f"{state['episode']}#{state['snapshot']}": state
        for state in load_states(loader_args)
    }
    states = [by_key[key] for key in manifest_keys]

    params = TorchMPPIParams(**json.loads(states[0]["mppi_params"]))
    weights = TorchMPPICostWeights(**json.loads(states[0]["cost_weights"]))
    backend = TorchDynamicBicycleRolloutBackend(
        TorchDBMParams(**json.loads(states[0]["dbm_params"]))
    )
    device = torch.device(args.device)
    low = np.asarray(params.action_min, np.float32)
    high = np.asarray(params.action_max, np.float32)

    teacher_knots = np.stack(
        [labels[f"{key}__knots"] for key in BANK_KEYS]
    )  # [3, N, 8, 2]
    consensus = np.clip(teacher_knots.mean(axis=0), low, high).astype(np.float32)
    actions = interpolate_knots(
        torch.as_tensor(
            consensus[:, None], dtype=torch.float32, device=device
        ),
        params.horizon,
    )
    initial = torch.as_tensor(
        np.stack([state["initial_state_six"] for state in states]),
        dtype=torch.float32, device=device,
    )
    current = torch.as_tensor(
        np.stack([state["current_action"] for state in states]),
        dtype=torch.float32, device=device,
    )
    reference = torch.as_tensor(
        np.stack([state["reference"] for state in states]),
        dtype=torch.float32, device=device,
    )
    with torch.no_grad():
        consensus_cost = batched_cost(
            backend, weights, actions, initial, current, reference
        ).squeeze(-1).cpu().numpy()

    bank_costs = np.stack(
        [labels[f"{key}__cost"] for key in BANK_KEYS]
    )  # [3, N]
    j_a0 = labels["j_a0"].astype(np.float64)
    j_warm = labels["j_warm"].astype(np.float64)
    j16 = labels["j16"].astype(np.float64)
    best_bank = bank_costs.min(axis=0)
    mean_bank = bank_costs.mean(axis=0)
    anchor = np.minimum(j_a0, j_warm)

    summary = {
        "format_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "qualification": "PHASE1A_CONSENSUS_DIAGNOSIS_ACTOR_FROZEN",
        "protocol": {
            "run": str(run),
            "states": len(states),
            "consensus": "elementwise mean of prefix_128/bank2/bank3 teachers",
            "objective": "deterministic J_direct",
        },
        "results": {
            "count": int(len(consensus_cost)),
            "j_consensus_mean": float(np.mean(consensus_cost)),
            "j_consensus_median": float(np.median(consensus_cost)),
            "j_best_bank_mean": float(np.mean(best_bank)),
            "j_mean_bank_mean": float(np.mean(mean_bank)),
            "j_a0_mean": float(np.mean(j_a0)),
            "j_warm_mean": float(np.mean(j_warm)),
            "j16_mean": float(np.mean(j16)),
            "headroom_recovery_vs_warm_consensus": float(
                np.sum(j_warm - consensus_cost) / np.sum(j_warm - j16)
            ),
            "headroom_recovery_vs_warm_best_bank": float(
                np.sum(j_warm - best_bank) / np.sum(j_warm - j16)
            ),
            "consensus_worse_than_anchor_count": int(
                np.sum(consensus_cost > anchor)
            ),
            "consensus_minus_best_bank_median": float(
                np.median(consensus_cost - best_bank)
            ),
            "consensus_minus_best_bank_p90": float(
                np.quantile(consensus_cost - best_bank, 0.90)
            ),
        },
    }
    output_path.write_text(json.dumps(summary, indent=1))
    np.savez_compressed(
        run / "consensus_labels.npz",
        episodes=np.asarray(episodes),
        consensus_knots=consensus,
        consensus_cost=consensus_cost.astype(np.float32),
    )
    print(json.dumps(summary["results"], indent=1))


if __name__ == "__main__":
    main()
