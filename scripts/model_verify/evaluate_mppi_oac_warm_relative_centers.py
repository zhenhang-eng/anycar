#!/usr/bin/env python3
"""Rescore saved OAC Actor centers against the realized warm center.

This evaluator is intentionally narrower than the OAC training evaluator:

* one deterministic, unnoised Actor center per state;
* one deterministic, unnoised warm center from candidate-bank column zero;
* no MPPI sampling, wrapper, softmax, guard output, or J16/oracle metric;
* warm is an evaluation comparator only and never an Actor input or loss.

The output keeps the full per-state paired costs so warm-relative P05/worst and
future DBM/Query sign-agreement checks do not have to be reconstructed from
clipped guard summaries.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from car_dynamics.controllers_torch.dbm import (
    TorchDBMParams,
    TorchDynamicBicycleRolloutBackend,
)
from car_dynamics.controllers_torch.mppi import TorchMPPICostWeights, TorchMPPIParams
from analyze_mppi_absolute_action_value_critic_gradient import load_rollout_inputs
from generate_dbm_proposal_teacher import sha256_file
from mppi_a2_actors import DirectNoAnchorGTXSupportActor
from train_mppi_online_absolute_sac import (
    actor_mean,
    load_actor_normalization,
    load_bank,
    make_actor_inputs,
    rollout_bank,
)


DEFAULT_RUNS = (
    (
        "gamma0_200",
        Path("outputs/mppi_proposal/online_absolute_sac_oac2_box3_adaptive_tail_200round_20260825_v1"),
    ),
    (
        "gamma05_200",
        Path("outputs/mppi_proposal/online_absolute_sac_oac2_rawcost_gamma05_200round_20260825_v1"),
    ),
    (
        "gamma1_200",
        Path("outputs/mppi_proposal/online_absolute_sac_oac2_rawcost_gamma1_200round_20260825_v1"),
    ),
    (
        "stable_lr32e5_trust004",
        Path("outputs/mppi_proposal/online_absolute_sac_oac2_rawcost_gamma1_lr32e5_trust004_90round_20260827_v1"),
    ),
    (
        "high_lr64e5_trust006",
        Path("outputs/mppi_proposal/online_absolute_sac_oac2_rawcost_gamma1_lr64e5_trust006_90round_20260827_v1"),
    ),
)
DEFAULT_OUTPUT = Path("outputs/mppi_proposal/oac_warm_relative_centers_20260827_v1")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run", action="append", default=[], metavar="NAME=PATH",
        help="Saved OAC run to rescore; repeat for multiple runs.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--rollout-batch-size", type=int, default=64)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def parse_runs(values: list[str]) -> list[tuple[str, Path]]:
    if not values:
        return list(DEFAULT_RUNS)
    result: list[tuple[str, Path]] = []
    names: set[str] = set()
    for value in values:
        if "=" not in value:
            raise ValueError(f"run must be NAME=PATH, got {value!r}")
        name, path = value.split("=", 1)
        if not name or name in names:
            raise ValueError(f"invalid or repeated run name {name!r}")
        names.add(name)
        result.append((name, Path(path)))
    return result


def distribution(values: np.ndarray) -> dict[str, float | int]:
    values = np.asarray(values, np.float64)
    return {
        "count": int(len(values)),
        "mean": float(np.mean(values)),
        "p05": float(np.quantile(values, 0.05)),
        "p10": float(np.quantile(values, 0.10)),
        "p25": float(np.quantile(values, 0.25)),
        "median": float(np.median(values)),
        "p75": float(np.quantile(values, 0.75)),
        "p90": float(np.quantile(values, 0.90)),
        "p95": float(np.quantile(values, 0.95)),
        "minimum": float(np.min(values)),
        "maximum": float(np.max(values)),
    }


def paired_summary(
    warm_cost: np.ndarray,
    actor_cost: np.ndarray,
    speed: np.ndarray,
    scenario: np.ndarray,
) -> dict[str, Any]:
    warm_cost = np.asarray(warm_cost, np.float64)
    actor_cost = np.asarray(actor_cost, np.float64)
    gain = warm_cost - actor_cost
    result: dict[str, Any] = {
        "count": int(len(gain)),
        "warm_cost": distribution(warm_cost),
        "actor_cost": distribution(actor_cost),
        "gain_vs_warm": distribution(gain),
        "actor_strict_win_fraction": float(np.mean(gain > 1e-6)),
        "tie_fraction": float(np.mean(np.abs(gain) <= 1e-6)),
        "actor_loss_fraction": float(np.mean(gain < -1e-6)),
        "aggregate_relative_gain": float(
            np.sum(gain) / max(float(np.sum(warm_cost)), 1e-9)
        ),
        "by_speed": {},
        "by_scenario": {},
    }
    for value in sorted(np.unique(speed)):
        mask = np.isclose(speed, value)
        result["by_speed"][f"{value:.1f}"] = paired_summary_slice(
            warm_cost[mask], actor_cost[mask]
        )
    for value in sorted(np.unique(scenario)):
        mask = scenario == value
        result["by_scenario"][str(value)] = paired_summary_slice(
            warm_cost[mask], actor_cost[mask]
        )
    return result


def paired_summary_slice(
    warm_cost: np.ndarray, actor_cost: np.ndarray,
) -> dict[str, Any]:
    gain = np.asarray(warm_cost, np.float64) - np.asarray(actor_cost, np.float64)
    return {
        "count": int(len(gain)),
        "warm_cost_mean": float(np.mean(warm_cost)),
        "actor_cost_mean": float(np.mean(actor_cost)),
        "gain_mean": float(np.mean(gain)),
        "gain_median": float(np.median(gain)),
        "gain_p05": float(np.quantile(gain, 0.05)),
        "gain_worst": float(np.min(gain)),
        "actor_strict_win_fraction": float(np.mean(gain > 1e-6)),
        "aggregate_relative_gain": float(
            np.sum(gain) / max(float(np.sum(warm_cost)), 1e-9)
        ),
    }


def checkpoint_actor(
    run: Path, seed: int, support_multiplier: float, device: torch.device,
) -> tuple[DirectNoAnchorGTXSupportActor, Path]:
    path = run / f"seed_{seed}" / "actor_latest.pt"
    payload = torch.load(path, map_location=device)
    actor = DirectNoAnchorGTXSupportActor(
        support_multiplier=support_multiplier, dropout=0.0
    ).to(device)
    actor.load_state_dict(payload["model_state_dict"], strict=True)
    actor.eval()
    return actor, path


def main() -> None:
    args = parse_args()
    runs = parse_runs(args.run)
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)
    device = torch.device(args.device)

    contracts: dict[str, dict[str, Any]] = {}
    for name, root in runs:
        contract_path = root / "contract.json"
        summary_path = root / "summary.json"
        if not contract_path.is_file() or not summary_path.is_file():
            raise FileNotFoundError(f"incomplete run {name}: {root}")
        contract = json.loads(contract_path.read_text())
        if contract.get("formal_validation_loaded") or contract.get("test_loaded"):
            raise AssertionError(f"sealed split violation in {name}")
        contracts[name] = contract

    first_name, first_root = runs[0]
    first_contract = contracts[first_name]
    first_args = first_contract["arguments"]
    bank_root = Path(first_args["bank_root"])
    gt_v1 = Path(first_args["gt_v1"])
    base_ac = Path(first_args["base_ac"])
    data = load_bank(bank_root)
    states, current, references, params_json, weights_json, dbm_json = (
        load_rollout_inputs(data, gt_v1)
    )
    params = TorchMPPIParams(**json.loads(params_json))
    weights = TorchMPPICostWeights(**json.loads(weights_json))
    backend = TorchDynamicBicycleRolloutBackend(TorchDBMParams(**json.loads(dbm_json)))
    normalization, normalization_path = load_actor_normalization(base_ac)
    actor_inputs = make_actor_inputs(data, normalization)

    episode_contract = tuple(first_contract["internal_selection_episodes"])
    indices = np.flatnonzero(np.isin(data["episode"], np.asarray(episode_contract)))
    if len(indices) != int(first_contract["internal_selection_state_count"]):
        raise AssertionError("selection state count mismatch")
    for name, root in runs:
        contract = contracts[name]
        if tuple(contract["internal_selection_episodes"]) != episode_contract:
            raise AssertionError(f"selection episodes differ for {name}")
        local_args = contract["arguments"]
        for field, expected in (
            ("bank_root", bank_root), ("gt_v1", gt_v1), ("base_ac", base_ac)
        ):
            if Path(local_args[field]).resolve() != expected.resolve():
                raise AssertionError(f"{field} differs for {name}")

    warm_action = np.asarray(data["actions"][indices, 0], np.float32)
    warm_recomputed = rollout_bank(
        backend, weights, params, warm_action[:, None], states, current,
        references, indices, args.rollout_batch_size, device,
    )[:, 0]
    warm_stored = np.asarray(data["costs"][indices, 0], np.float32)
    warm_replay_error = float(np.max(np.abs(warm_recomputed - warm_stored)))
    # The stored bank and the current batched DBM path differ by a few float32
    # ulps at large costs; 1e-3 remains far below a meaningful cost change.
    if warm_replay_error > 1e-3:
        raise AssertionError(f"warm DBM replay mismatch {warm_replay_error}")

    row_run: list[np.ndarray] = []
    row_seed: list[np.ndarray] = []
    row_state: list[np.ndarray] = []
    row_actor_action: list[np.ndarray] = []
    row_actor_cost: list[np.ndarray] = []
    summaries: dict[str, Any] = {}
    run_manifest: dict[str, Any] = {}
    latest_mean_replay_max_abs_error = 0.0
    for name, root in runs:
        contract = contracts[name]
        local_args = contract["arguments"]
        support_multiplier = float(local_args["actor_output_support_multiplier"])
        per_seed = []
        run_costs = []
        checkpoint_hashes = []
        for seed in (0, 1, 2):
            actor, actor_path = checkpoint_actor(
                root, seed, support_multiplier, device
            )
            action = actor_mean(actor, actor_inputs, indices, device)
            cost = rollout_bank(
                backend, weights, params, action[:, None], states, current,
                references, indices, args.rollout_batch_size, device,
            )[:, 0]
            old_seed_summary = json.loads(
                (root / f"seed_{seed}" / "summary.json").read_text()
            )
            old_mean = float(old_seed_summary["latest_metrics"]["cost"]["mean"])
            mean_replay_error = abs(float(np.mean(cost)) - old_mean)
            latest_mean_replay_max_abs_error = max(
                latest_mean_replay_max_abs_error, mean_replay_error
            )
            if mean_replay_error > 1e-2:
                raise AssertionError(
                    f"latest Actor cost mismatch {name} seed {seed}: "
                    f"{mean_replay_error}"
                )
            per_seed.append(paired_summary(
                warm_recomputed, cost, data["speed"][indices],
                data["scenario"][indices],
            ))
            run_costs.append(cost)
            checkpoint_hashes.append(sha256_file(actor_path))
            count = len(indices)
            row_run.append(np.full(count, name))
            row_seed.append(np.full(count, seed, np.int16))
            row_state.append(indices.astype(np.int32))
            row_actor_action.append(np.asarray(action, np.float32))
            row_actor_cost.append(np.asarray(cost, np.float32))
        pooled_cost = np.concatenate(run_costs)
        pooled_warm = np.tile(warm_recomputed, 3)
        pooled_speed = np.tile(data["speed"][indices], 3)
        pooled_scenario = np.tile(data["scenario"][indices], 3)
        summaries[name] = {
            "per_seed": per_seed,
            "seed_mean": {
                "actor_strict_win_fraction": float(np.mean([
                    value["actor_strict_win_fraction"] for value in per_seed
                ])),
                "gain_median": float(np.mean([
                    value["gain_vs_warm"]["median"] for value in per_seed
                ])),
                "gain_mean": float(np.mean([
                    value["gain_vs_warm"]["mean"] for value in per_seed
                ])),
                "actor_cost_mean": float(np.mean([
                    value["actor_cost"]["mean"] for value in per_seed
                ])),
                "actor_cost_median": float(np.mean([
                    value["actor_cost"]["median"] for value in per_seed
                ])),
                "aggregate_relative_gain": float(np.mean([
                    value["aggregate_relative_gain"] for value in per_seed
                ])),
            },
            "pooled": paired_summary(
                pooled_warm, pooled_cost, pooled_speed, pooled_scenario
            ),
        }
        run_manifest[name] = {
            "path": str(root.resolve()),
            "contract_sha256": sha256_file(root / "contract.json"),
            "summary_sha256": sha256_file(root / "summary.json"),
            "actor_latest_sha256": checkpoint_hashes,
            "actor_cost_gamma": float(local_args.get("actor_cost_weight_gamma", 0.0)),
            "actor_learning_rate_schedule": local_args.get(
                "actor_learning_rate_schedule", "constant"
            ),
            "actor_learning_rate_initial": local_args.get("actor_learning_rate_initial"),
            "max_step_sigma_rms": float(local_args["max_step_sigma_rms"]),
            "actor_rounds": int(local_args["rounds"]),
        }
        print(
            f"{name}: win={summaries[name]['pooled']['actor_strict_win_fraction']:.4f} "
            f"gain_med={summaries[name]['pooled']['gain_vs_warm']['median']:.4f} "
            f"gain_p05={summaries[name]['pooled']['gain_vs_warm']['p05']:.4f} "
            f"gain_worst={summaries[name]['pooled']['gain_vs_warm']['minimum']:.4f}",
            flush=True,
        )

    actor_cost = np.concatenate(row_actor_cost)
    warm_cost = np.tile(warm_recomputed, len(runs) * 3)
    np.savez_compressed(
        args.output_dir / "evaluation.npz",
        run=np.concatenate(row_run),
        seed=np.concatenate(row_seed),
        state_index=np.concatenate(row_state),
        episode=np.tile(data["episode"][indices], len(runs) * 3),
        snapshot=np.tile(data["snapshot"][indices], len(runs) * 3),
        speed=np.tile(data["speed"][indices], len(runs) * 3),
        scenario=np.tile(data["scenario"][indices], len(runs) * 3),
        warm_action=np.tile(warm_action, (len(runs) * 3, 1, 1)),
        actor_action=np.concatenate(row_actor_action),
        warm_cost=warm_cost.astype(np.float32),
        actor_cost=actor_cost.astype(np.float32),
        gain_vs_warm=(warm_cost - actor_cost).astype(np.float32),
    )
    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "qualification": "OAC_WARM_RELATIVE_DIRECT_CENTER_RESCORE",
        "scope": (
            "fold-1 train-side internal-selection; deterministic DBM J50; "
            "single unnoised Actor center versus realized warm center; no MPPI "
            "sampling/wrapper/guard output/oracle; formal validation/test sealed"
        ),
        "contract": {
            "warm_role": "evaluation comparator only; never Actor input/loss/reward",
            "actor_role": "saved latest deterministic absolute center",
            "strict_win": "J_actor < J_warm - 1e-6",
            "gain": "J_warm - J_actor",
            "aggregate_relative_gain": "sum(gain) / sum(J_warm)",
        },
        "checks": {
            "formal_validation_sealed": True,
            "test_sealed": True,
            "selection_state_count": int(len(indices)),
            "selection_episode_count": int(len(episode_contract)),
            "warm_dbm_replay_max_abs_error": warm_replay_error,
            "same_selection_episodes_all_runs": True,
            "saved_latest_mean_replay_all_pass": True,
            "saved_latest_mean_replay_max_abs_error": (
                latest_mean_replay_max_abs_error
            ),
        },
        "manifest": {
            "bank": str((bank_root / "candidate_bank.npz").resolve()),
            "bank_sha256": sha256_file(bank_root / "candidate_bank.npz"),
            "normalization_checkpoint": str(normalization_path.resolve()),
            "normalization_checkpoint_sha256": sha256_file(normalization_path),
            "runs": run_manifest,
        },
        "warm_reference": paired_summary(
            warm_recomputed, warm_recomputed, data["speed"][indices],
            data["scenario"][indices],
        )["warm_cost"],
        "runs": summaries,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
