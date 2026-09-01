#!/usr/bin/env python3
"""Bounded amplitude scan along the existing consensus-search residual.

This is the teacher-side gate for the post-coverage branch-2 experiment.  It
does not search new directions.  For every state in the 1800-state diverse
pool it evaluates deterministic DBM J50 at scaled versions of the stored
three-bank consensus residual, with normalized RMS caps of 0.5 and 0.8 sigma.
The candidate set also contains a0 and warm, so each emitted label is a
cost-consistent zero-regression teacher.
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
from generate_dbm_proposal_teacher import sha256_file
from run_mppi_proximal_search_phase1a import (
    DEFAULT_GT_TRAIN,
    DEFAULT_REPLAY_LABELS,
    DEFAULT_SCENARIO_PLAN,
    load_states,
)


DEFAULT_LABELS = Path(
    "outputs/mppi_proposal/consensus64_labels_pool_20260818_v1/labels.npz"
)
DEFAULT_MANIFEST = Path(
    "outputs/mppi_proposal/consensus64_labels_pool_20260818_v1/manifest.json"
)
DEFAULT_OUTPUT = Path(
    "outputs/mppi_proposal/consensus_amplitude_scan_20260818_v1"
)
BETAS = np.asarray([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0], np.float32)
CAPS = (0.5, 0.8)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--replay-labels", type=Path, default=DEFAULT_REPLAY_LABELS)
    parser.add_argument("--gt-train", type=Path, default=DEFAULT_GT_TRAIN)
    parser.add_argument("--scenario-plan", type=Path, default=DEFAULT_SCENARIO_PLAN)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--repeat", type=int, default=0, choices=(0, 1))
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    return parser.parse_args()


def normalized_rms(center: np.ndarray, a0: np.ndarray, sigma: np.ndarray) -> float:
    safe = np.maximum(np.asarray(sigma, np.float32), 1e-6)
    return float(np.sqrt(np.mean(((center - a0) / safe.reshape(1, 2)) ** 2)))


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    labels = dict(np.load(args.labels, allow_pickle=False))
    manifest = json.loads(args.manifest.read_text())

    loader_args = SimpleNamespace(
        replay_labels=args.replay_labels,
        gt_train=args.gt_train,
        scenario_plan=args.scenario_plan,
        repeat=args.repeat,
    )
    states = sorted(
        load_states(loader_args),
        key=lambda item: (item["speed"], item["scenario"], item["residual_norm"]),
    )
    if len(states) != len(labels["episodes"]):
        raise AssertionError("state/label count mismatch")
    state_keys = [
        f'{state["episode"]}#{state["snapshot"]}' for state in states
    ]
    if state_keys != [str(value) for value in labels["episodes"]]:
        raise AssertionError("state order mismatch vs consensus pool")
    if len(manifest["states"]) != len(states):
        raise AssertionError("manifest state count mismatch")

    params = TorchMPPIParams(**json.loads(states[0]["mppi_params"]))
    weights = TorchMPPICostWeights(**json.loads(states[0]["cost_weights"]))
    backend = TorchDynamicBicycleRolloutBackend(
        TorchDBMParams(**json.loads(states[0]["dbm_params"]))
    )
    device = torch.device(args.device)
    low = np.asarray(params.action_min, np.float32)
    high = np.asarray(params.action_max, np.float32)

    def evaluate(state: dict, centers: np.ndarray) -> np.ndarray:
        clipped = np.clip(centers, low, high).astype(np.float32)
        count = len(clipped)
        with torch.no_grad():
            actions = interpolate_knots(
                torch.as_tensor(clipped[:, None], device=device), params.horizon
            )
            values = batched_cost(
                backend,
                weights,
                actions,
                torch.as_tensor(
                    np.repeat(state["initial_state_six"][None], count, axis=0),
                    dtype=torch.float32,
                    device=device,
                ),
                torch.as_tensor(
                    np.repeat(state["current_action"][None], count, axis=0),
                    dtype=torch.float32,
                    device=device,
                ),
                torch.as_tensor(
                    np.repeat(state["reference"][None], count, axis=0),
                    dtype=torch.float32,
                    device=device,
                ),
            )
        return values.squeeze(-1).cpu().numpy().astype(np.float32)

    per_cap = {
        cap: {
            "label_knots": [],
            "j_teacher": [],
            "selected_kind": [],
            "selected_beta": [],
            "movement_rms_sigma": [],
            "cap_saturated": [],
        }
        for cap in CAPS
    }
    stored_replay_error = {"j_a0": 0.0, "j_warm": 0.0, "j_consensus": 0.0}

    for index, state in enumerate(states):
        a0 = np.asarray(state["a0"], np.float32)
        warm = np.asarray(state["warm"], np.float32)
        sigma = np.asarray(state["sigma"], np.float32)
        consensus = np.asarray(labels["consensus_knots"][index], np.float32)
        residual = consensus - a0
        base_rms = normalized_rms(consensus, a0, sigma)

        for cap in CAPS:
            scaled = []
            saturated = []
            for beta in BETAS:
                effective = float(beta)
                if base_rms > 0:
                    effective = min(effective, cap / base_rms)
                scaled.append(a0 + effective * residual)
                saturated.append(bool(effective + 1e-7 < float(beta)))
            centers = np.concatenate(
                [a0[None], warm[None], np.asarray(scaled, np.float32)], axis=0
            )
            costs = evaluate(state, centers)
            if cap == CAPS[0]:
                stored_replay_error["j_a0"] = max(
                    stored_replay_error["j_a0"],
                    abs(float(costs[0]) - float(labels["j_a0"][index])),
                )
                stored_replay_error["j_warm"] = max(
                    stored_replay_error["j_warm"],
                    abs(float(costs[1]) - float(labels["j_warm"][index])),
                )
                stored_replay_error["j_consensus"] = max(
                    stored_replay_error["j_consensus"],
                    abs(float(costs[4]) - float(labels["j_teacher"][index]))
                    if not bool(labels["fallback"][index]) else 0.0,
                )
            best = int(np.argmin(costs))
            if best == 0:
                kind, beta, was_saturated = "a0", 0.0, False
            elif best == 1:
                kind, beta, was_saturated = "warm", np.nan, False
            else:
                local = best - 2
                kind = "scaled_consensus"
                beta = float(BETAS[local])
                was_saturated = saturated[local]
            chosen = np.clip(centers[best], low, high).astype(np.float32)
            target = per_cap[cap]
            target["label_knots"].append(chosen)
            target["j_teacher"].append(float(costs[best]))
            target["selected_kind"].append(kind)
            target["selected_beta"].append(beta)
            target["movement_rms_sigma"].append(normalized_rms(chosen, a0, sigma))
            target["cap_saturated"].append(was_saturated)

        if (index + 1) % 100 == 0:
            print(f"[{index + 1}/{len(states)}]", flush=True)

    anchor_best = np.minimum(labels["j_a0"], labels["j_warm"]).astype(np.float64)
    j16 = labels["j16"].astype(np.float64)
    denominator = float(np.sum(anchor_best - j16))
    summaries = {}
    for cap, raw in per_cap.items():
        label_knots = np.stack(raw["label_knots"]).astype(np.float32)
        j_teacher = np.asarray(raw["j_teacher"], np.float32)
        kind = np.asarray(raw["selected_kind"])
        beta = np.asarray(raw["selected_beta"], np.float32)
        movement = np.asarray(raw["movement_rms_sigma"], np.float32)
        saturated = np.asarray(raw["cap_saturated"], bool)
        stay = movement <= 1e-6
        key = f"cap{int(round(cap * 100)):03d}"
        np.savez_compressed(
            args.output / f"labels_{key}.npz",
            episodes=labels["episodes"],
            speeds=labels["speeds"],
            scenarios=labels["scenarios"],
            j_a0=labels["j_a0"],
            j_warm=labels["j_warm"],
            j16=labels["j16"],
            consensus_knots=labels["consensus_knots"],
            label_knots=label_knots,
            j_teacher=j_teacher,
            fallback=stay,
            stay=stay,
            selected_kind=kind,
            selected_beta=beta,
            movement_rms_sigma=movement,
            cap_saturated=saturated,
        )
        scaled = kind == "scaled_consensus"
        extended = scaled & (beta > 1.0)
        summaries[key] = {
            "cap_rms_sigma": cap,
            "j_teacher_mean": float(j_teacher.mean()),
            "aggregate_gain_vs_anchor": float(np.sum(anchor_best - j_teacher)),
            "headroom_recovery_to_j16": float(
                np.sum(anchor_best - j_teacher) / denominator
            ),
            "selection_fraction": {
                name: float(np.mean(kind == name))
                for name in ("a0", "warm", "scaled_consensus")
            },
            "extended_beta_gt_1_fraction": float(np.mean(extended)),
            "cap_saturated_fraction": float(np.mean(saturated)),
            "movement_rms_sigma": {
                "median_all": float(np.median(movement)),
                "median_movers": float(np.median(movement[movement > 1e-6])),
                "p90_movers": float(np.quantile(movement[movement > 1e-6], 0.90)),
            },
            "selected_beta_counts": {
                str(value): int(np.sum(beta == value)) for value in BETAS
            },
            "labels_file": f"labels_{key}.npz",
        }

    original_movement = []
    for state, center in zip(states, labels["label_knots"]):
        original_movement.append(normalized_rms(
            np.asarray(center, np.float32),
            np.asarray(state["a0"], np.float32),
            np.asarray(state["sigma"], np.float32),
        ))
    original_movement = np.asarray(original_movement, np.float32)
    summary = {
        "format_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "qualification": "CONSENSUS_AMPLITUDE_TEACHER_GATE",
        "sources": {
            "labels": str(args.labels.resolve()),
            "labels_sha256": sha256_file(args.labels),
            "manifest": str(args.manifest.resolve()),
            "manifest_sha256": sha256_file(args.manifest),
        },
        "contract": {
            "states": len(states),
            "betas": BETAS.tolist(),
            "caps_rms_sigma": list(CAPS),
            "candidate_set": "a0 + warm + scaled stored consensus residual",
            "objective": "deterministic DBM J50",
            "formal_validation_loaded": False,
            "test_loaded": False,
        },
        "replay_max_abs_error": stored_replay_error,
        "original": {
            "j_teacher_mean": float(labels["j_teacher"].mean()),
            "movement_median_all_rms_sigma": float(np.median(original_movement)),
            "movement_median_movers_rms_sigma": float(
                np.median(original_movement[original_movement > 1e-6])
            ),
        },
        "caps": summaries,
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
