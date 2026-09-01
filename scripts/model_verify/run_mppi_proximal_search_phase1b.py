#!/usr/bin/env python3
"""Phase 1b: proximal search label generation at scale (train-only).

Generates the Phase 2 distillation labels under the 11.45/11.46 contract:
- 600 stratified train-only states (deterministic even spacing over
  speed/scenario/J16-residual-norm).
- Main arm: Arm C multi-round 128 (64 bank prefix + 32 + 32 re-centred
  rounds); budget counts new unique evaluations with in-batch dedup.
- Guarded labels: the teacher is kept only when its deterministic gain over
  the anchor exceeds the float guard 1e-6; otherwise the label is the anchor
  itself (zero-regression stay). The near-zero gain distribution is reported
  so Phase 2 can choose a meaningful stay threshold with data.
- CRN audit on a deterministic 200-state subset: E_eps[J] of teacher and
  anchor under two independent 8-draw common-noise sets (selection and audit
  seeds, per-state deterministic). Reports margin consistency and the
  fraction of movers whose sampling expectation does not improve.
- Bypass diagnostics per state: cancellation ratio, position share and
  dominant term via per-term autograd (same pipeline as 11.41, cross-checked
  at 0.997 elsewhere). Diagnostic only; never an online contract input.
- Metrics: R_a0 aggregate with episode bootstrap CI, gain median/P05,
  per-stratum tables, stay fraction. J16 is a best-found reference, not a
  proven optimum. No Actor is trained; formal validation/test stay sealed.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import zlib

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
from generate_dbm_j16_local_curvature_labels import hadamard_directions
from generate_dbm_proposal_teacher import sha256_file
from run_mppi_proximal_search_phase1a import (
    DEFAULT_GT_TRAIN,
    DEFAULT_REPLAY_LABELS,
    DEFAULT_SCENARIO_PLAN,
    ARM_C_ROUND_RADII,
    ELITE_COUNT,
    RING_RADII,
    elite_mean,
    load_states,
    ring_candidates,
    select_states,
)
from analyze_mppi_proximal_mechanism_join import term_gradients

DEFAULT_OUTPUT = Path("outputs/mppi_proposal/proximal_search_phase1b_20260818_v1")
MAIN3 = ("position", "yaw", "vx")
CRN_DRAWS = 8
CRN_SUBSET = 200
STAY_GUARD = 1e-6
BOOTSTRAP = 2000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay-labels", type=Path, default=DEFAULT_REPLAY_LABELS)
    parser.add_argument("--gt-train", type=Path, default=DEFAULT_GT_TRAIN)
    parser.add_argument("--scenario-plan", type=Path, default=DEFAULT_SCENARIO_PLAN)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--states", type=int, default=600)
    parser.add_argument("--repeat", type=int, default=0, choices=(0, 1))
    parser.add_argument("--eval-chunk", type=int, default=512)
    parser.add_argument("--crn-subset", type=int, default=CRN_SUBSET)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    return parser.parse_args()


def state_seed(key: str, salt: int) -> int:
    return (zlib.crc32(key.encode()) ^ salt) % (2 ** 31)


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    directions = hadamard_directions()
    states = select_states(load_states(args), args.states)
    for key in ("cost_weights", "dbm_params"):
        if len({state[key] for state in states}) != 1:
            raise ValueError(f"{key} differs across states")
    params = TorchMPPIParams(**json.loads(states[0]["mppi_params"]))
    weights = TorchMPPICostWeights(**json.loads(states[0]["cost_weights"]))
    backend = TorchDynamicBicycleRolloutBackend(
        TorchDBMParams(**json.loads(states[0]["dbm_params"]))
    )
    device = torch.device(args.device)
    low = np.asarray(params.action_min, np.float32)
    high = np.asarray(params.action_max, np.float32)
    anchor_hashes = sorted({state["anchor_checkpoint_sha256"] for state in states})
    if len(anchor_hashes) != 1:
        raise AssertionError("anchor checkpoint differs across states")

    crn_keys = {
        f"{state['episode']}#{state['snapshot']}" for state in states
    }
    crn_selection = set(
        sorted(crn_keys)[:: max(1, len(crn_keys) // args.crn_subset)][: args.crn_subset]
    )

    rows = []
    for position, state in enumerate(states):
        key = f"{state['episode']}#{state['snapshot']}"
        cache: dict[bytes, float] = {}
        budget = {"total": 0}

        def evaluate(candidates: np.ndarray) -> np.ndarray:
            clipped = np.clip(candidates, low, high).astype(np.float32)
            keys = [np.round(value, 6).tobytes() for value in clipped]
            pending: dict[bytes, int] = {}
            for row, entry in enumerate(keys):
                if entry not in cache:
                    pending.setdefault(entry, row)
            if pending:
                pending_keys = list(pending)
                pending_array = np.asarray(
                    [clipped[row] for row in pending.values()], np.float32
                )
                flat = []
                with torch.no_grad():
                    for start in range(0, len(pending_array), args.eval_chunk):
                        chunk = torch.as_tensor(
                            pending_array[start:start + args.eval_chunk][None],
                            dtype=torch.float32, device=device,
                        )
                        actions = interpolate_knots(chunk, params.horizon)
                        flat.append(
                            batched_cost(
                                backend, weights, actions,
                                torch.as_tensor(state["initial_state_six"][None], dtype=torch.float32, device=device),
                                torch.as_tensor(state["current_action"][None], dtype=torch.float32, device=device),
                                torch.as_tensor(state["reference"][None], dtype=torch.float32, device=device),
                            )[0].cpu().numpy()
                        )
                budget["total"] += len(pending_keys)
                for entry, cost in zip(pending_keys, np.concatenate(flat)):
                    cache[entry] = float(cost)
            return np.asarray([cache[entry] for entry in keys], np.float32)

        anchor_costs = evaluate(np.stack([state["a0"], state["warm"]]))
        j_a0, j_warm = float(anchor_costs[0]), float(anchor_costs[1])

        bank = ring_candidates(state["a0"], state["sigma"], RING_RADII[:2], directions)
        bank_costs = evaluate(bank)
        m1 = elite_mean(
            np.concatenate([bank, state["a0"][None], state["warm"][None]]),
            np.concatenate([bank_costs, anchor_costs]),
        )
        round2 = ring_candidates(m1, state["sigma"], [ARM_C_ROUND_RADII[2]], directions)
        round2_costs = evaluate(round2)
        m2 = elite_mean(
            np.concatenate([bank, round2, state["a0"][None], state["warm"][None]]),
            np.concatenate([bank_costs, round2_costs, anchor_costs]),
        )
        round3 = ring_candidates(m2, state["sigma"], [ARM_C_ROUND_RADII[3]], directions)
        round3_costs = evaluate(round3)
        pool = np.concatenate(
            [state["a0"][None], state["warm"][None], bank, round2, round3]
        )
        pool_costs = np.concatenate([anchor_costs, bank_costs, round2_costs, round3_costs])
        best = int(np.argmin(pool_costs))
        teacher_knots, j_teacher = pool[best], float(pool_costs[best])
        gain = min(j_a0, j_warm) - j_teacher
        stay = gain <= STAY_GUARD
        label_knots = state["a0"] if stay else teacher_knots

        row = {
            "episode": state["episode"],
            "snapshot": state["snapshot"],
            "scenario": state["scenario"],
            "speed": state["speed"],
            "j_a0": j_a0,
            "j_warm": j_warm,
            "j16_best_found": state["j16_cost"],
            "j_teacher": j_teacher,
            "gain": float(gain),
            "stay": bool(stay),
            "teacher_knots": teacher_knots,
            "label_knots": label_knots,
            "budget": budget["total"],
        }

        grads = term_gradients(backend, weights, state, state["a0"], device)
        magnitudes = {name: float(np.linalg.norm(grads[name])) for name in MAIN3}
        net = -sum(grads.values())
        total_large = sum(magnitudes.values()) + 1e-12
        row["cancellation"] = float(np.linalg.norm(net) / total_large)
        row["position_share"] = magnitudes["position"] / total_large
        row["dominant_term"] = max(MAIN3, key=lambda name: magnitudes[name])

        if key in crn_selection:
            sigma_tiled = np.repeat(state["sigma"], 8)
            crn = {}
            for label, salt in (("selection", 26081801), ("audit", 26081802)):
                rng = np.random.default_rng(state_seed(key, salt))
                noises = rng.standard_normal((CRN_DRAWS, 16)).astype(np.float32)
                teacher_noisy = np.clip(
                    teacher_knots[None]
                    + (noises * sigma_tiled).reshape(CRN_DRAWS, 8, 2),
                    low, high,
                )
                anchor_noisy = np.clip(
                    state["a0"][None]
                    + (noises * sigma_tiled).reshape(CRN_DRAWS, 8, 2),
                    low, high,
                )
                crn[f"j_teacher_{label}"] = float(np.mean(evaluate(teacher_noisy)))
                crn[f"j_a0_{label}"] = float(np.mean(evaluate(anchor_noisy)))
            for label in ("selection", "audit"):
                crn[f"margin_{label}"] = (
                    crn[f"j_a0_{label}"] - crn[f"j_teacher_{label}"]
                )
            row["crn"] = crn

        rows.append(row)
        print(
            f"[{position + 1:03d}/{len(states):03d}] {key} "
            f"J={j_teacher:.3f} gain={gain:.3f} stay={row['stay']} "
            f"budget={budget['total']}",
            flush=True,
        )

    j_a0 = np.asarray([row["j_a0"] for row in rows], np.float64)
    j_warm = np.asarray([row["j_warm"] for row in rows], np.float64)
    j16 = np.asarray([row["j16_best_found"] for row in rows], np.float64)
    j_teacher = np.asarray([row["j_teacher"] for row in rows], np.float64)
    gain = np.asarray([row["gain"] for row in rows], np.float64)
    anchor = np.minimum(j_a0, j_warm)
    denominator = j_a0 - j16
    tiny = denominator <= 0.005
    r_a0 = float(np.sum(j_a0 - j_teacher) / np.sum(denominator))
    r_warm = float(np.sum(j_warm - j_teacher) / np.sum(j_warm - j16))

    episodes = np.asarray([row["episode"] for row in rows])
    unique_episodes, episode_index = np.unique(episodes, return_inverse=True)
    rng = np.random.default_rng(26081803)
    boot = []
    for _ in range(BOOTSTRAP):
        sample = rng.integers(0, len(unique_episodes), len(unique_episodes))
        mask = np.isin(episode_index, sample)
        denominator_sample = np.sum(denominator[mask])
        if denominator_sample > 1e-9:
            boot.append(np.sum(j_a0[mask] - j_teacher[mask]) / denominator_sample)
    boot = np.asarray(boot)

    def strata_table(field_a, field_b):
        table = {}
        for row in rows:
            name = f"{row[field_a]}|{row[field_b]}" if field_b else str(row[field_a])
            table.setdefault(name, {"count": 0, "j": [], "gain": [], "stay": 0})
            entry = table[name]
            entry["count"] += 1
            entry["j"].append(row["j_teacher"])
            entry["gain"].append(row["gain"])
            entry["stay"] += int(row["stay"])
        return {
            name: {
                "count": entry["count"],
                "j_teacher_mean": float(np.mean(entry["j"])),
                "gain_median": float(np.median(entry["gain"])),
                "stay_fraction": entry["stay"] / entry["count"],
            }
            for name, entry in table.items()
        }

    crn_rows = [row for row in rows if "crn" in row]
    movers = [row for row in crn_rows if not row["stay"]]
    crn_summary = {
        "states": len(crn_rows),
        "movers": len(movers),
        "selection_margin_median": float(np.median([
            row["crn"]["margin_selection"] for row in crn_rows
        ])),
        "audit_margin_median": float(np.median([
            row["crn"]["margin_audit"] for row in crn_rows
        ])),
        "mover_selection_margin_median": float(np.median([
            row["crn"]["margin_selection"] for row in movers
        ])),
        "mover_audit_margin_median": float(np.median([
            row["crn"]["margin_audit"] for row in movers
        ])),
        "movers_with_nonpositive_selection_margin": int(np.sum([
            row["crn"]["margin_selection"] <= 0 for row in movers
        ])),
        "movers_with_nonpositive_audit_margin": int(np.sum([
            row["crn"]["margin_audit"] <= 0 for row in movers
        ])),
        "sign_agreement_fraction": float(np.mean([
            (row["crn"]["margin_selection"] > 0) == (row["crn"]["margin_audit"] > 0)
            for row in crn_rows
        ])),
    }

    summary = {
        "format_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "qualification": "PHASE1B_LABEL_GENERATION_COMPLETED_ACTOR_FROZEN",
        "sources": {
            "replay_labels": str(args.replay_labels),
            "gt_train": str(args.gt_train),
            "gt_train_summary_sha256": sha256_file(args.gt_train / "summary.json"),
            "scenario_plan": str(args.scenario_plan),
            "scenario_plan_sha256": sha256_file(args.scenario_plan),
            "anchor_checkpoint_sha256": anchor_hashes[0],
        },
        "protocol": {
            "states": len(states),
            "repeat": args.repeat,
            "arm": "multi-round 128 (64 prefix + 32 + 32)",
            "objective": "deterministic J_direct; CRN E_eps[J] audit separate",
            "stay_guard": STAY_GUARD,
            "crn_draws_per_seed_set": CRN_DRAWS,
            "crn_subset": len(crn_selection),
            "j16_semantics": "best-found reference, not a proven optimum",
            "diagnostics": (
                "cancellation/position share are bypass fields only, never "
                "online contract inputs"
            ),
        },
        "results": {
            "j_a0_mean": float(np.mean(j_a0)),
            "j_warm_mean": float(np.mean(j_warm)),
            "j16_best_found_mean": float(np.mean(j16)),
            "j_teacher_mean": float(np.mean(j_teacher)),
            "j_teacher_p95": float(np.quantile(j_teacher, 0.95)),
            "r_a0": r_a0,
            "r_a0_bootstrap_ci": [
                float(np.quantile(boot, 0.025)),
                float(np.quantile(boot, 0.975)),
            ],
            "r_warm": r_warm,
            "gain_median": float(np.median(gain)),
            "gain_p05": float(np.quantile(gain, 0.05)),
            "gain_below_0p01": int(np.sum(gain < 0.01)),
            "gain_below_0p1": int(np.sum(gain < 0.1)),
            "stay_fraction": float(np.mean([row["stay"] for row in rows])),
            "baseline_violations": int(np.sum(j_teacher > anchor + 1e-9)),
            "tiny_headroom_states_denominator_le_0p005": int(np.sum(tiny)),
            "mean_budget_per_state": float(np.mean([
                row["budget"] for row in rows
            ])),
        },
        "strata_speed": strata_table("speed", None),
        "strata_speed_scenario": strata_table("speed", "scenario"),
        "crn_audit": crn_summary,
    }
    args.output.mkdir(parents=True)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=1))
    np.savez_compressed(
        args.output / "labels.npz",
        episodes=np.asarray([f"{row['episode']}#{row['snapshot']}" for row in rows]),
        speeds=np.asarray([row["speed"] for row in rows], np.float32),
        scenarios=np.asarray([row["scenario"] for row in rows]),
        j_a0=j_a0.astype(np.float32),
        j_warm=j_warm.astype(np.float32),
        j16=j16.astype(np.float32),
        j_teacher=j_teacher.astype(np.float32),
        gain=gain.astype(np.float32),
        stay=np.asarray([row["stay"] for row in rows], bool),
        label_knots=np.stack([row["label_knots"] for row in rows]),
        teacher_knots=np.stack([row["teacher_knots"] for row in rows]),
        cancellation=np.asarray([row["cancellation"] for row in rows], np.float32),
        position_share=np.asarray([row["position_share"] for row in rows], np.float32),
        dominant_term=np.asarray([row["dominant_term"] for row in rows]),
        crn_margin_selection=np.asarray([
            row.get("crn", {}).get("margin_selection", np.nan) for row in rows
        ], np.float32),
        crn_margin_audit=np.asarray([
            row.get("crn", {}).get("margin_audit", np.nan) for row in rows
        ], np.float32),
    )
    (args.output / "manifest.json").write_text(json.dumps({
        "states": [
            {
                "episode": row["episode"], "snapshot": row["snapshot"],
                "scenario": row["scenario"], "speed": row["speed"],
                "anchor_checkpoint_sha256": anchor_hashes[0],
                "crn_audited": "crn" in row,
            }
            for row in rows
        ],
    }, indent=1))
    print(json.dumps(summary["results"], indent=1))
    print(json.dumps(crn_summary, indent=1))


if __name__ == "__main__":
    main()
