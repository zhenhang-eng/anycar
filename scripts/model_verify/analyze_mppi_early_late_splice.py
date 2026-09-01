#!/usr/bin/env python3
"""Early/late splice oracle over the saved OOF actor predictions.

Causal locus test (review 11.52 follow-up): the actor's OOF predictions are
spliced with the teacher knots by control epoch and channel, then re-rolled
out. If replacing only the early knots (0-2) — or only early steering — with
teacher values closes most of the (teacher - actor) gap, the failure locus is
the high-leverage early segment and the A2 knot-aligned/temporal structure is
aimed at the right bottleneck; if late splices close it instead, the early-
leverage story fails.

Splices per state and seed (p = actor prediction, t = teacher label):
- actor: p                     (reference)
- teacher: t                   (reference)
- earlyT:  t[0:3] + p[3:8]
- lateT:   p[0:3] + t[3:8]
- earlyT_steer: p with steering dims of knots 0-2 from t
- earlyT_accel: p with accel dims of knots 0-2 from t

Gap closure = sum(gain_splice - gain_actor) / sum(gain_teacher - gain_actor),
aggregated (never per-frame ratios). Actor stays frozen.
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

DEFAULT_RUN = Path("outputs/mppi_proposal/actor_curve_n1800_predump_20260818_v1")
DEFAULT_POOL = Path("outputs/mppi_proposal/consensus64_labels_pool_20260818_v1")
DEFAULT_OUTPUT = Path("outputs/mppi_proposal/early_late_splice_20260818_v1")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--pool", type=Path, default=DEFAULT_POOL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--repeat", type=int, default=0)
    parser.add_argument("--batch", type=int, default=4096)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    oof = np.load(args.run / "oof_evaluation.npz", allow_pickle=False)
    pool = np.load(args.pool / "labels.npz", allow_pickle=False)
    pool_keys = [str(v) for v in pool["episodes"]]
    pool_index = {key: i for i, key in enumerate(pool_keys)}

    episodes = [str(v) for v in oof["state_keys"]]
    rows = [pool_index[key] for key in episodes]
    count = len(rows)
    predicted = oof["predicted_knots"].reshape(count, 8, 2).astype(np.float32)
    teacher = pool["label_knots"][rows].reshape(count, 8, 2).astype(np.float32)
    j_a0 = pool["j_a0"][rows].astype(np.float64)
    j16 = pool["j16"][rows].astype(np.float64)
    seed = oof["seed"].astype(np.int64)

    loader_args = SimpleNamespace(
        replay_labels=DEFAULT_REPLAY_LABELS,
        gt_train=DEFAULT_GT_TRAIN,
        scenario_plan=DEFAULT_SCENARIO_PLAN,
        repeat=args.repeat,
    )
    by_key = {f"{s['episode']}#{s['snapshot']}": s for s in load_states(loader_args)}
    states = [by_key[key] for key in episodes]

    device = torch.device(args.device)
    params = TorchMPPIParams(**json.loads(states[0]["mppi_params"]))
    weights = TorchMPPICostWeights(**json.loads(states[0]["cost_weights"]))
    backend = TorchDynamicBicycleRolloutBackend(
        TorchDBMParams(**json.loads(states[0]["dbm_params"]))
    )

    def splice(name: str) -> np.ndarray:
        p, t = predicted.copy(), teacher.copy()
        if name == "actor":
            out = p
        elif name == "teacher":
            out = t
        elif name == "earlyT":
            out = p.copy(); out[:, :3] = t[:, :3]
        elif name == "lateT":
            out = p.copy(); out[:, 3:] = t[:, 3:]
        elif name == "earlyT_steer":
            out = p.copy(); out[:, :3, 1] = t[:, :3, 1]
        elif name == "earlyT_accel":
            out = p.copy(); out[:, :3, 0] = t[:, :3, 0]
        else:
            raise ValueError(name)
        return out.astype(np.float32)

    names = (
        "actor", "teacher", "earlyT", "lateT", "earlyT_steer", "earlyT_accel",
    )
    gains = {}
    for name in names:
        centers = splice(name)
        knots = torch.as_tensor(
            centers[:, None], dtype=torch.float32, device=device
        )
        actions = interpolate_knots(knots, params.horizon)
        costs = []
        with torch.no_grad():
            for start in range(0, count, args.batch):
                chunk = slice(start, min(start + args.batch, count))
                costs.append(
                    batched_cost(
                        backend, weights, actions[chunk],
                        torch.as_tensor(
                            np.stack(
                                [s["initial_state_six"] for s in states[chunk]]
                            ), dtype=torch.float32, device=device,
                        ),
                        torch.as_tensor(
                            np.stack(
                                [s["current_action"] for s in states[chunk]]
                            ), dtype=torch.float32, device=device,
                        ),
                        torch.as_tensor(
                            np.stack(
                                [s["reference"] for s in states[chunk]]
                            ), dtype=torch.float32, device=device,
                        ),
                    ).squeeze(-1).cpu().numpy()
                )
        costs = np.concatenate(costs).astype(np.float64)
        gains[name] = j_a0 - costs
        print(f"{name}: mean J {float(np.mean(costs)):.3f}", flush=True)

    actor_gain = gains["actor"]
    gap = float(np.sum(gains["teacher"] - actor_gain))
    results = {}
    for name in names:
        per_seed = {}
        for value in sorted(set(seed.tolist())):
            mask = seed == value
            per_seed[str(value)] = {
                "recovery": float(
                    np.sum(gains[name][mask])
                    / np.sum(j_a0[mask] - j16[mask])
                ),
                "gap_closure": (
                    float(np.sum(gains[name][mask] - actor_gain[mask]))
                    / max(float(np.sum(
                        gains["teacher"][mask] - actor_gain[mask]
                    )), 1e-9)
                    if name not in ("actor", "teacher") else None
                ),
            }
        results[name] = {
            "recovery_pooled": float(np.sum(gains[name]) / np.sum(j_a0 - j16)),
            "gap_closure_pooled": (
                float(np.sum(gains[name] - actor_gain) / gap)
                if name not in ("actor", "teacher") else None
            ),
            "per_seed": per_seed,
        }
    summary = {
        "format_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "qualification": "EARLY_LATE_SPLICE_ORACLE_ACTOR_FROZEN",
        "sources": {"run": str(args.run), "pool": str(args.pool)},
        "protocol": {
            "rows": count,
            "splices": {
                "earlyT": "teacher knots 0-2 + actor knots 3-7",
                "lateT": "actor knots 0-2 + teacher knots 3-7",
                "earlyT_steer": "steering dims of knots 0-2 from teacher",
                "earlyT_accel": "accel dims of knots 0-2 from teacher",
            },
            "gap_closure": (
                "sum(gain_splice - gain_actor) / sum(gain_teacher - "
                "gain_actor), aggregated"
            ),
        },
        "results": results,
    }
    args.output.mkdir(parents=True)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps({
        name: {
            "recovery": round(value["recovery_pooled"], 3),
            "gap_closure": (
                round(value["gap_closure_pooled"], 3)
                if value["gap_closure_pooled"] is not None else None
            ),
        }
        for name, value in results.items()
    }, indent=1))


if __name__ == "__main__":
    main()
