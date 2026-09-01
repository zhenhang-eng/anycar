#!/usr/bin/env python3
"""Zero/low-cost mechanism join for the proximal search route.

Part A (zero rollout): direction canonicity vs move magnitude, from the
Phase 1a artifacts only. Hypothesis (review 11.41/11.43): near the optimum
the net gradient is a small difference of large opposing cost terms, so
small teacher moves live in the cancellation region where direction is not
canonical; larger moves (like the 0.86-sigma J16 residuals) should show
better cross-bank direction agreement. Tests: cross-bank teacher cosine and
teacher-to-J16 cosine stratified by move magnitude, excluding anchor-picking
states whose zero residual makes cosine trivial.

Part B (100 autograd passes, same pipeline as cost_term_flip_attribution):
per-term action gradients at the Phase 1a actor anchors, cancellation ratio
(net / sum of the three large-term norms) and position share. Joins:
consensus-worse-than-anchor states, teacher-is-anchor states, and cross-bank
direction agreement against cancellation. No FD cross-check exists for these
states; the pipeline itself was cross-validated at cosine 0.997 on the 600
selection contexts (review 11.41/11.42). No actor is updated.
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
from car_dynamics.controllers_torch.mppi import TorchMPPICostWeights
from generate_dbm_direct_gt_validation import interpolate_knots
from run_mppi_proximal_search_phase1a import (
    DEFAULT_GT_TRAIN,
    DEFAULT_REPLAY_LABELS,
    DEFAULT_SCENARIO_PLAN,
    load_states,
)

DEFAULT_RUN = Path("outputs/mppi_proposal/proximal_search_phase1a_20260817_v1")
DEFAULT_OUTPUT = Path(
    "outputs/mppi_proposal/proximal_mechanism_join_20260817_v1"
)
BANK_KEYS = ("c_prefix_128", "c_bank2_128", "c_bank3_128")
MAIN3 = ("position", "yaw", "vx")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--repeat", type=int, default=0)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    return parser.parse_args()


def unit(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)
    return vector / norm if norm > 1e-9 else np.zeros_like(vector)


def spearman(left: np.ndarray, right: np.ndarray) -> float:
    if len(left) < 3:
        return float("nan")
    rank_left = np.argsort(np.argsort(left)).astype(np.float64)
    rank_right = np.argsort(np.argsort(right)).astype(np.float64)
    left_c = rank_left - rank_left.mean()
    right_c = rank_right - rank_right.mean()
    denominator = np.sqrt(np.sum(left_c ** 2) * np.sum(right_c ** 2))
    if denominator <= 0:
        return float("nan")
    return float(np.sum(left_c * right_c) / denominator)


def term_gradients(
    backend: TorchDynamicBicycleRolloutBackend,
    weights: TorchMPPICostWeights,
    state: dict,
    knots_np: np.ndarray,
    device: torch.device,
) -> dict[str, np.ndarray]:
    horizon = backend.horizon
    knots = torch.from_numpy(knots_np.astype(np.float32)).to(device)[None]
    knots.requires_grad_(True)
    actions = interpolate_knots(knots, horizon)
    initial = torch.from_numpy(
        state["initial_state_six"].astype(np.float32)
    ).to(device)[None]
    full = backend.rollout_full_state_differentiable(initial, actions)
    trajectory = full[0][:, [0, 1, 2, 3, 5]]
    if trajectory.shape[0] != horizon:
        trajectory = trajectory[:horizon]
    reference = torch.from_numpy(state["reference"].astype(np.float32)).to(
        device
    )
    if reference.shape[0] == trajectory.shape[0] + 1:
        reference = reference[1:]
    pos = weights.position * ((trajectory[:, :2] - reference[:, :2]) ** 2).sum()
    yaw_delta = trajectory[:, 2] - reference[:, 2]
    yaw = weights.yaw * (
        torch.atan2(torch.sin(yaw_delta), torch.cos(yaw_delta)) ** 2
    ).sum()
    vx = weights.vx * ((trajectory[:, 3] - reference[:, 3]) ** 2).sum()
    current = torch.from_numpy(
        state["current_action"].astype(np.float32)
    ).to(device)
    previous = torch.cat((current[None], actions[0, :-1]), dim=0)
    steer_rate = weights.steering_rate * (
        (actions[0, :, 1] - previous[:, 1]) ** 2
    ).sum()
    accel_rate = weights.acceleration_rate * (
        (actions[0, :, 0] - previous[:, 0]) ** 2
    ).sum()
    out = {}
    for name, value in (
        ("position", pos), ("yaw", yaw), ("vx", vx),
        ("steer_rate", steer_rate), ("accel_rate", accel_rate),
    ):
        gradient = torch.autograd.grad(value, knots, retain_graph=True)[0]
        out[name] = gradient.detach().cpu().numpy().flatten()
    return out


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    labels = np.load(args.run / "labels.npz", allow_pickle=False)
    consensus = np.load(args.run / "consensus_labels.npz", allow_pickle=False)
    manifest = json.loads((args.run / "manifest.json").read_text())
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
    count = len(states)

    device = torch.device(args.device)
    backend = TorchDynamicBicycleRolloutBackend(
        TorchDBMParams(**json.loads(states[0]["dbm_params"]))
    )
    weights = TorchMPPICostWeights(**json.loads(states[0]["cost_weights"]))

    rows = []
    for position, state in enumerate(states):
        sigma = np.repeat(state["sigma"], 8)
        a0 = state["a0"].reshape(-1)
        j16_delta = (state["astar"].reshape(-1) - a0) / sigma
        bank_deltas = {}
        for key in BANK_KEYS:
            knots = labels[f"{key}__knots"][position].reshape(-1)
            bank_deltas[key] = (knots - a0) / sigma
        move = float(np.sqrt(np.mean(bank_deltas[BANK_KEYS[0]] ** 2)))
        cos12 = float(unit(bank_deltas[BANK_KEYS[0]]) @ unit(bank_deltas[BANK_KEYS[1]]))
        cos13 = float(unit(bank_deltas[BANK_KEYS[0]]) @ unit(bank_deltas[BANK_KEYS[2]]))
        cos_j16 = float(unit(bank_deltas[BANK_KEYS[0]]) @ unit(j16_delta))
        j16_norm = float(np.sqrt(np.mean(j16_delta ** 2)))

        grads = term_gradients(backend, weights, state, state["a0"], device)
        magnitudes = {name: float(np.linalg.norm(grads[name])) for name in MAIN3}
        net = -sum(grads.values())
        total_large = sum(magnitudes.values()) + 1e-12
        cancellation = float(np.linalg.norm(net) / total_large)
        position_share = magnitudes["position"] / total_large
        dominant = max(MAIN3, key=lambda name: magnitudes[name])

        anchor_cost = min(
            float(labels["j_a0"][position]), float(labels["j_warm"][position])
        )
        teacher_gain = anchor_cost - float(labels["c_prefix_128__cost"][position])
        rows.append({
            "episode": state["episode"],
            "snapshot": state["snapshot"],
            "speed": state["speed"],
            "scenario": state["scenario"],
            "move_sigma_rms": move,
            "teacher_gain": float(teacher_gain),
            "j16_residual_sigma_rms": j16_norm,
            "bank12_cosine": cos12,
            "bank13_cosine": cos13,
            "teacher_j16_cosine": cos_j16,
            "cancellation": cancellation,
            "position_share": position_share,
            "dominant_term": dominant,
            "teacher_is_anchor": bool(move < 1e-6),
            "consensus_worse_than_anchor": bool(
                float(consensus["consensus_cost"][position]) > anchor_cost + 1e-6
            ),
        })
        print(f"[{position + 1:03d}/{count:03d}] done", flush=True)

    move = np.asarray([row["move_sigma_rms"] for row in rows])
    cos12 = np.asarray([row["bank12_cosine"] for row in rows])
    cos13 = np.asarray([row["bank13_cosine"] for row in rows])
    cos_j16 = np.asarray([row["teacher_j16_cosine"] for row in rows])
    cancellation = np.asarray([row["cancellation"] for row in rows])
    position_share = np.asarray([row["position_share"] for row in rows])
    j16_norm = np.asarray([row["j16_residual_sigma_rms"] for row in rows])
    anchor_flag = np.asarray([row["teacher_is_anchor"] for row in rows])
    bad_flag = np.asarray([row["consensus_worse_than_anchor"] for row in rows])
    gain = np.asarray([row["teacher_gain"] for row in rows])
    # Cost-based mover definition matches the run summary: a geometric move
    # with gain <= 1e-6 is a numerical tie (the candidate and the anchor are
    # equal within float noise) and is treated as stay, not as a mover.
    moved = gain > 1e-6
    tie = (~anchor_flag) & (~moved)

    unique_moves = np.unique(np.round(move[moved], 4))
    canonicity_by_move = {}
    for level in unique_moves:
        mask = moved & np.isclose(np.round(move, 4), level)
        if not np.any(mask):
            continue
        canonicity_by_move[f"{level:.4f}"] = {
            "count": int(np.sum(mask)),
            "bank12_cosine_median": float(np.median(cos12[mask])),
            "bank13_cosine_median": float(np.median(cos13[mask])),
            "teacher_j16_cosine_median": float(np.median(cos_j16[mask])),
            "j16_residual_median": float(np.median(j16_norm[mask])),
        }

    def group_stats(mask: np.ndarray) -> dict:
        if not np.any(mask):
            return {"count": 0}
        return {
            "count": int(np.sum(mask)),
            "cancellation_median": float(np.median(cancellation[mask])),
            "position_share_median": float(np.median(position_share[mask])),
            "move_sigma_rms_median": float(np.median(move[mask])),
            "j16_residual_median": float(np.median(j16_norm[mask])),
            "bank12_cosine_median": float(np.median(cos12[mask])),
        }

    summary = {
        "format_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "qualification": "PROXIMAL_MECHANISM_JOIN_DIAGNOSIS_ACTOR_FROZEN",
        "sources": {
            "phase1a_run": str(args.run),
            "repeat": args.repeat,
            "term_gradient_pipeline": (
                "identical to analyze_mppi_cost_term_flip_attribution "
                "(cross-validated at cosine 0.997 on the 600 selection "
                "contexts); no FD cross-check on these states"
            ),
        },
        "part_a_canonicity_vs_move": {
            "anchor_state_count": int(np.sum(anchor_flag)),
            "tie_state_count": int(np.sum(tie)),
            "moved_state_count": int(np.sum(moved)),
            "note": (
                "teacher moves are quantized to ring steps; tie states move "
                "geometrically but improve by <=1e-6 and count as stay"
            ),
            "canonicity_by_move_level": canonicity_by_move,
            "spearman_move_vs_bank12": spearman(move[moved], cos12[moved]),
            "spearman_move_vs_bank13": spearman(move[moved], cos13[moved]),
            "spearman_move_vs_teacher_j16_cosine": spearman(
                move[moved], cos_j16[moved]
            ),
            "spearman_j16norm_vs_teacher_j16_cosine": spearman(
                j16_norm[moved], cos_j16[moved]
            ),
        },
        "part_b_cancellation_joins": {
            "all_states": group_stats(np.ones(count, bool)),
            "anchor_states": group_stats(anchor_flag),
            "tie_states": group_stats(tie),
            "moved_states": group_stats(moved),
            "consensus_worse_than_anchor": group_stats(bad_flag),
            "consensus_ok_moved": group_stats(moved & ~bad_flag),
            "spearman_cancellation_vs_move": spearman(
                cancellation[moved], move[moved]
            ),
            "spearman_cancellation_vs_bank12": spearman(
                cancellation[moved], cos12[moved]
            ),
            "spearman_position_share_vs_move": spearman(
                position_share[moved], move[moved]
            ),
            "dominant_term_counts_moved": {
                name: int(np.sum([
                    row["dominant_term"] == name
                    for row, keep in zip(rows, moved) if keep
                ]))
                for name in MAIN3
            },
        },
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "analysis.json").write_text(json.dumps(summary, indent=1))
    (args.output / "per_state.json").write_text(json.dumps(rows, indent=1))
    print(json.dumps({
        "part_a": summary["part_a_canonicity_vs_move"],
        "part_b_joins": {
            name: summary["part_b_cancellation_joins"][name]
            for name in (
                "anchor_states", "moved_states",
                "consensus_worse_than_anchor", "consensus_ok_moved",
            )
        },
    }, indent=1))


if __name__ == "__main__":
    main()
