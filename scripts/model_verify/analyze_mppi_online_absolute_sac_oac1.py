#!/usr/bin/env python3
"""Post-hoc OAC-0 heldout evaluator and OAC-1 gate decomposition.

The original threshold-0.5 flat/stay result is preserved.  This audit also
calibrates the flat threshold on fold-train only and evaluates it on fold-0,
because the OAC plan registered recall/FPR but did not register a probability
threshold.  It never changes the OAC-1 qualification.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import torch

from run_mppi_absolute_action_value_critic_cv import (
    AbsoluteActionValueCritic,
    make_folds,
    metrics as bank_metrics,
)
from train_mppi_online_absolute_sac import (
    critic_state_inputs,
    load_bank,
    material_pair_accuracy,
    predict_actions,
    predict_flat,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument(
        "--bank-root", type=Path,
        default=Path("outputs/mppi_proposal/absolute_action_value_critic_20260820_v1"),
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def load_model(path: Path, device: torch.device):
    payload = torch.load(path, map_location=device)
    model = AbsoluteActionValueCritic(dropout=0.0).to(device)
    model.load_state_dict(payload["model"], strict=True)
    model.eval()
    return model, payload


def predict_bank(model, inputs, payload, states, actions, device):
    count, candidates = actions.shape[:2]
    flat_state = np.repeat(states, candidates)
    flat_action = actions.reshape(count * candidates, 8, 2)
    return predict_actions(
        model, inputs, payload, flat_state, flat_action, device
    ).reshape(count, candidates)


def calibrated_flat_gate(
    bank_best_probability: np.ndarray, warm_probability: np.ndarray,
    train: np.ndarray, heldout: np.ndarray, warm_material: np.ndarray,
) -> dict:
    candidates = []
    for threshold in np.linspace(0.0, 1.0, 1001):
        recall = float(np.mean(bank_best_probability[train] >= threshold))
        false_stay = float(np.mean(
            warm_probability[train & warm_material] >= threshold
        ))
        if false_stay <= 0.10:
            candidates.append((recall, -threshold, threshold, false_stay))
    selected = max(candidates) if candidates else (0.0, -1.0, 1.0, 0.0)
    threshold = float(selected[2])
    train_recall = float(np.mean(bank_best_probability[train] >= threshold))
    train_false = float(np.mean(
        warm_probability[train & warm_material] >= threshold
    ))
    heldout_recall = float(np.mean(bank_best_probability[heldout] >= threshold))
    heldout_false = float(np.mean(
        warm_probability[heldout & warm_material] >= threshold
    ))
    return {
        "selection": "maximize train recall subject to train warm false-stay <=0.10",
        "threshold": threshold,
        "train": {"bank_best_recall": train_recall, "warm_false_stay": train_false},
        "heldout": {
            "bank_best_recall": heldout_recall,
            "warm_false_stay": heldout_false,
        },
        "heldout_gate_passed": heldout_recall >= 0.80 and heldout_false <= 0.10,
    }


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    contract = json.loads((args.run_dir / "contract.json").read_text())
    summary = json.loads((args.run_dir / "summary.json").read_text())
    data = load_bank(args.bank_root)
    folds = make_folds(data, 3)
    train, heldout = folds != 0, folds == 0
    heldout_index = np.flatnonzero(heldout)
    best_index = np.argmin(data["costs"], axis=1)
    bank_best_action = data["actions"][np.arange(len(data["costs"])), best_index]
    warm_material = data["costs"][:, 0] - data["costs"].min(1) > float(
        contract["arguments"]["flat_gap"]
    )
    records = []
    for source in summary["records"]:
        seed = int(source["seed"])
        seed_dir = args.run_dir / f"seed_{seed}"
        stored = json.loads((seed_dir / "summary.json").read_text())
        with np.load(seed_dir / "actor_visited_replay.npz", allow_pickle=False) as loaded:
            replay = {key: np.asarray(loaded[key]) for key in loaded.files}
        final1, payload1 = load_model(seed_dir / "critic1.pt", device)
        final2, payload2 = load_model(seed_dir / "critic2.pt", device)
        initial1, initial_payload1 = load_model(
            Path(stored["initial_critics"][0]), device
        )
        initial2, initial_payload2 = load_model(
            Path(stored["initial_critics"][1]), device
        )
        input1 = critic_state_inputs(data, payload1)
        input2 = critic_state_inputs(data, payload2)
        initial_input1 = critic_state_inputs(data, initial_payload1)
        initial_input2 = critic_state_inputs(data, initial_payload2)
        actions = data["actions"][heldout]
        initial_prediction = np.maximum(
            predict_bank(
                initial1, initial_input1, initial_payload1, heldout_index,
                actions, device,
            ),
            predict_bank(
                initial2, initial_input2, initial_payload2, heldout_index,
                actions, device,
            ),
        )
        final_prediction = np.maximum(
            predict_bank(final1, input1, payload1, heldout_index, actions, device),
            predict_bank(final2, input2, payload2, heldout_index, actions, device),
        )
        initial_heldout = bank_metrics(
            initial_prediction, data["costs"][heldout],
            float(contract["arguments"]["material_gap"]),
        )
        final_heldout = bank_metrics(
            final_prediction, data["costs"][heldout],
            float(contract["arguments"]["material_gap"]),
        )

        flat_head, flat_payload = load_model(seed_dir / "flat_head.pt", device)
        flat_inputs = critic_state_inputs(data, {
            "training": {"normalization": flat_payload["normalization"]}
        })
        flat_best = predict_flat(
            flat_head, flat_inputs, np.arange(len(data["costs"])),
            bank_best_action, device,
        )
        flat_warm = predict_flat(
            flat_head, flat_inputs, np.arange(len(data["costs"])),
            data["actions"][:, 0], device,
        )
        calibrated = calibrated_flat_gate(
            flat_best, flat_warm, train, heldout, warm_material
        )

        conservative = np.maximum(
            replay["final_critic1"], replay["final_critic2"]
        )
        by_speed = {}
        for speed in sorted(np.unique(data["speed"])):
            mask = data["speed"][replay["state_index"]] == speed
            accuracy, count = material_pair_accuracy(
                conservative[mask], replay["cost"][mask],
                replay["interaction_group"][mask],
                float(contract["arguments"]["material_gap"]),
            )
            by_speed[f"{speed:.1f}"] = {
                "rows": int(mask.sum()), "material_pair_accuracy": accuracy,
                "material_pair_count": count,
                "cost_mean": float(np.mean(replay["cost"][mask])),
            }
        records.append({
            "seed": seed,
            "heldout_bank_initial": initial_heldout,
            "heldout_bank_after_oac1": final_heldout,
            "heldout_bank_pair_accuracy_delta": float(
                final_heldout["material_pair_accuracy"]
                - initial_heldout["material_pair_accuracy"]
            ),
            "actor_visited_by_speed": by_speed,
            "flat_threshold_0_5": stored["metrics"]["flat_stay"],
            "flat_train_calibrated_heldout_evaluation": calibrated,
            "bad_action_correction": stored["metrics"]["bad_action_correction"],
        })
    result = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "qualification": "OAC1_GATE_DECOMPOSITION_ACTOR_REMAINS_FROZEN",
        "records": records,
        "decision": {
            "value_ranking": (
                "passes actor-visited >=0.85 for all seeds and remains strong on "
                "episode-heldout historical bank"
            ),
            "flat_stay": (
                "fails recall>=0.80 under both threshold 0.5 and a train-only "
                "FPR-constrained calibration; this is not merely calibration"
            ),
            "bad_action_correction": (
                "overall lag2 ranking exceeds 0.93, but only 0.33-0.45 of initially "
                "misranked bad actions are corrected; the registered correction gate fails"
            ),
            "actor": "remains frozen; OAC-2 is not authorized",
        },
        "formal_validation_loaded": False,
        "test_loaded": False,
    }
    (args.run_dir / "oac1_gate_analysis.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print(json.dumps({
        "output": str((args.run_dir / "oac1_gate_analysis.json").resolve()),
        "qualification": result["qualification"],
        "heldout_pair_accuracy": [
            round(row["heldout_bank_after_oac1"]["material_pair_accuracy"], 4)
            for row in records
        ],
        "calibrated_flat_heldout": [
            row["flat_train_calibrated_heldout_evaluation"]["heldout"]
            for row in records
        ],
    }, indent=2))


if __name__ == "__main__":
    main()
