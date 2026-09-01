#!/usr/bin/env python3
"""Run the frozen TR3 formal-validation gate for the Alpha Actor--Critic.

Only validation episodes 090--104 are read.  Thresholds, checkpoints, bootstrap
seed and gates are frozen in the design document before this evaluator is run.
Test episodes are not opened.  All scores are deterministic direct DBM rollout
costs; no analytic dynamics gradient is used.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any

import numpy as np
import torch

from generate_dbm_direct_trust_region_labels import (
    actor_inputs,
    alpha_grid,
    line_centers,
    load_actor,
    select_safe_index,
)
from generate_dbm_multicenter_teacher import evaluate_knots, make_controller
from generate_dbm_proposal_teacher import repository_state, sha256_file
from train_mppi_direct_trust_alpha_policy import make_policy


ROOT = Path("/disk/collect_data_from_anycar/mppi_rl_closed_loop")
SOURCE = ROOT / "fixed_dbm_policy_diverse_20260805_v1"
CONTEXT = ROOT / "labels/dbm_two_pass_feedback_diverse_20260805_v1"
RISK = ROOT / "labels/dbm_two_pass_risk_replay_diverse_20260805_v1"
ALPHA_AC = Path(
    "outputs/mppi_proposal/direct_alpha_actor_critic_20260810_v1/"
    "alpha_actor_critic_selected.pt"
)
TR2B = Path(
    "outputs/mppi_proposal/direct_trust_alpha_policy_20260810_v1/"
    "trust_alpha_policy_tail_calibrated.pt"
)
DEFAULT_OUTPUT = Path(
    "outputs/mppi_proposal/direct_alpha_ac_tr3_validation_20260810_v1"
)
EXPECTED_ALPHA_AC_SHA256 = "ae862c09e09a46929e7cd172f54f2c2821cf6f20ba18ee7bef7b086c31ef9783"
EXPECTED_TR2B_SHA256 = "37cd09ab746f28fa7c3acf512e7f993d675ebc16b29104ba1f65e23497e5f012"
BOOTSTRAP_SEED = 20260810
BOOTSTRAP_REPEATS = 20000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--context", type=Path, default=CONTEXT)
    parser.add_argument("--risk", type=Path, default=RISK)
    parser.add_argument("--alpha-ac", type=Path, default=ALPHA_AC)
    parser.add_argument("--tr2b", type=Path, default=TR2B)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def distribution(value: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(np.mean(value)),
        "median": float(np.median(value)),
        "p05": float(np.quantile(value, 0.05)),
        "p95": float(np.quantile(value, 0.95)),
        "minimum": float(np.min(value)),
        "maximum": float(np.max(value)),
    }


def episode_bootstrap_ci(
    gain: np.ndarray, episodes: np.ndarray,
    seed: int = BOOTSTRAP_SEED, repeats: int = BOOTSTRAP_REPEATS,
) -> dict[str, float]:
    unique = np.unique(episodes)
    per_episode = np.asarray([np.mean(gain[episodes == one]) for one in unique])
    rng = np.random.default_rng(seed)
    values = np.empty(repeats, np.float64)
    for start in range(0, repeats, 1000):
        count = min(1000, repeats - start)
        sample = rng.integers(0, len(unique), size=(count, len(unique)))
        values[start:start + count] = np.mean(per_episode[sample], axis=1)
    return {
        "mean": float(np.mean(gain)),
        "episode_mean": float(np.mean(per_episode)),
        "lower_95": float(np.quantile(values, 0.025)),
        "upper_95": float(np.quantile(values, 0.975)),
        "bootstrap_seed": seed,
        "bootstrap_repeats": repeats,
        "episode_count": len(unique),
    }


def summarize_arrays(arrays: dict[str, np.ndarray]) -> dict[str, Any]:
    old = arrays["old_cost"]
    tr2b = arrays["tr2b_cost"]
    ac = arrays["alpha_ac_cost"]
    safe = arrays["safe_cost"]
    oracle = arrays["argmin_cost"]
    episode = arrays["episode"]
    speed = arrays["reference_speed"]
    scenario = arrays["scenario"]
    gain_old = old - ac
    gain_tr2b = tr2b - ac

    by_speed = {}
    for value in sorted(np.unique(speed)):
        mask = np.isclose(speed, value)
        by_speed[str(float(value))] = {
            "context_count": int(np.sum(mask)),
            "old_cost_mean": float(np.mean(old[mask])),
            "tr2b_cost_mean": float(np.mean(tr2b[mask])),
            "alpha_ac_cost_mean": float(np.mean(ac[mask])),
            "safe_cost_mean": float(np.mean(safe[mask])),
            "gain_vs_old": distribution(gain_old[mask]),
            "gain_vs_tr2b": distribution(gain_tr2b[mask]),
        }
    by_scenario = {}
    for value in sorted(np.unique(scenario)):
        mask = scenario == value
        by_scenario[str(value)] = {
            "context_count": int(np.sum(mask)),
            "old_cost_mean": float(np.mean(old[mask])),
            "tr2b_cost_mean": float(np.mean(tr2b[mask])),
            "alpha_ac_cost_mean": float(np.mean(ac[mask])),
            "safe_cost_mean": float(np.mean(safe[mask])),
            "gain_vs_old": distribution(gain_old[mask]),
            "gain_vs_tr2b": distribution(gain_tr2b[mask]),
        }

    old_ci = episode_bootstrap_ci(gain_old, episode)
    tr2b_ci = episode_bootstrap_ci(gain_tr2b, episode)
    high_speed = all(
        by_speed[key]["gain_vs_old"]["mean"] >= 0.0
        for key in by_speed
        if np.isclose(float(key), 2.4) or np.isclose(float(key), 2.8)
    )
    recovery_rows = [
        row for name, row in by_scenario.items() if "recovery" in name
    ]
    recovery = bool(recovery_rows) and all(
        row["gain_vs_old"]["mean"] >= 0.0 for row in recovery_rows
    )
    gates = {
        "vs_old_mean_ci_lower_positive": old_ci["lower_95"] > 0.0,
        "vs_old_median_nonnegative": float(np.median(gain_old)) >= -1e-6,
        "vs_old_p05_nonnegative": float(np.quantile(gain_old, 0.05)) >= -1e-6,
        "vs_old_worst_at_least_minus_5": float(np.min(gain_old)) >= -5.0,
        "high_speed_mean_nonregression": high_speed,
        "recovery_mean_nonregression": recovery,
        "vs_tr2b_mean_positive": float(np.mean(gain_tr2b)) > 0.0,
        "vs_tr2b_mean_ci_lower_positive": tr2b_ci["lower_95"] > 0.0,
    }
    return {
        "context_count": len(old),
        "snapshot_count": len(np.unique(arrays["snapshot_key"])),
        "episode_count": len(np.unique(episode)),
        "cost": {
            "old_actor": distribution(old),
            "tr2b": distribution(tr2b),
            "alpha_ac": distribution(ac),
            "safe_line_label": distribution(safe),
            "argmin_line_oracle": distribution(oracle),
        },
        "gain_vs_old": distribution(gain_old),
        "gain_vs_tr2b": distribution(gain_tr2b),
        "gain_vs_old_episode_bootstrap": old_ci,
        "gain_vs_tr2b_episode_bootstrap": tr2b_ci,
        "move_fraction": {
            "tr2b": float(np.mean(arrays["tr2b_alpha"] > 0.0)),
            "alpha_ac": float(np.mean(arrays["alpha_ac_alpha"] > 0.0)),
        },
        "by_reference_speed_mps": by_speed,
        "by_scenario": by_scenario,
        "gates": gates,
        "qualification": (
            "TR3_PASS_ALPHA_AC_FORMAL_VALIDATION"
            if all(gates.values())
            else "TR3_FAIL_RETAIN_TR2B_OR_OLD"
        ),
    }


def load_alpha_policy(path: Path, payload: dict, old_payload: dict, device):
    policy = make_policy(old_payload, device, dropout=0.0)
    policy.load_state_dict(payload["policy_state_dict"], strict=True)
    policy.eval()
    return policy


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    if sha256_file(args.alpha_ac) != EXPECTED_ALPHA_AC_SHA256:
        raise AssertionError("Alpha AC checkpoint differs from frozen TR3 protocol")
    if sha256_file(args.tr2b) != EXPECTED_TR2B_SHA256:
        raise AssertionError("TR2-B checkpoint differs from frozen TR3 protocol")
    ac_payload = torch.load(args.alpha_ac, map_location="cpu")
    tr2b_payload = torch.load(args.tr2b, map_location="cpu")
    if not np.isclose(float(ac_payload["move_threshold"]), 0.88):
        raise AssertionError("Alpha AC threshold is not frozen at 0.88")
    if not np.isclose(float(tr2b_payload["move_threshold"]), 0.99):
        raise AssertionError("TR2-B threshold is not frozen at 0.99")
    if ac_payload["old_actor_sha256"] != tr2b_payload["old_actor_sha256"]:
        raise AssertionError("Alpha policies use different old Actors")
    if ac_payload["proposal_actor_sha256"] != tr2b_payload["proposal_actor_sha256"]:
        raise AssertionError("Alpha policies use different proposal Actors")

    splits = json.loads((args.context / "splits.json").read_text())
    validation_episodes = list(splits["validation"])
    if validation_episodes != [f"episode_{index:03d}" for index in range(90, 105)]:
        raise AssertionError("formal validation episode contract changed")
    test_episodes = set(splits["test"])
    plan = json.loads((args.source / "scenario_plan.json").read_text())
    metadata = {
        row["episode_id"]: row for row in plan["episodes"]
        if row["episode_id"] in validation_episodes
    }
    if set(metadata) != set(validation_episodes):
        raise AssertionError("validation scenario metadata incomplete")

    records = []
    for episode in validation_episodes:
        if episode in test_episodes:
            raise AssertionError("test episode overlaps validation")
        for source_path in sorted((args.source / episode / "snapshots").glob("*.npz")):
            context_path = args.context / episode / source_path.name
            risk_path = args.risk / episode / source_path.name
            if not context_path.is_file() or not risk_path.is_file():
                raise FileNotFoundError(f"missing validation context: {context_path}")
            records.append((episode, source_path, context_path, risk_path))
    if len(records) != 300:
        raise AssertionError(f"expected 300 validation snapshots, found {len(records)}")

    device = torch.device(args.device)
    old_path = Path(ac_payload["old_actor"])
    proposal_path = Path(ac_payload["proposal_actor"])
    old_payload, old_actor = load_actor(old_path, device)
    proposal_payload, proposal_actor = load_actor(proposal_path, device)
    tr2b_policy = load_alpha_policy(args.tr2b, tr2b_payload, old_payload, device)
    ac_policy = load_alpha_policy(args.alpha_ac, ac_payload, old_payload, device)
    alphas = alpha_grid(0.05)

    staging = Path(tempfile.mkdtemp(
        prefix=f".{args.output_dir.name}.tmp-", dir=args.output_dir.parent
    ))
    arrays: dict[str, list] = defaultdict(list)
    try:
        for record_index, (episode, source_path, context_path, risk_path) in enumerate(records, 1):
            with np.load(source_path, allow_pickle=False) as source, np.load(
                context_path, allow_pickle=False
            ) as context, np.load(risk_path, allow_pickle=False) as risk:
                source_hash = sha256_file(source_path)
                if str(context["source_snapshot_sha256"]) != source_hash:
                    raise AssertionError("context/source validation hash mismatch")
                if str(risk["source_snapshot_sha256"]) != source_hash:
                    raise AssertionError("risk/source validation hash mismatch")
                gradient_mean = np.asarray(risk["critic_gradient_mean"], np.float32)
                gradient_std = np.asarray(risk["critic_gradient_std"], np.float32)
                config = {"objective": {"cost_weights": json.loads(str(source["cost_weights_json"]))}}
                controller, backend = make_controller(source, config, device)
                params = json.loads(str(source["mppi_params_json"]))
                sigma = np.asarray(params["noise_sigma"], np.float32)
                low = np.asarray(params["action_min"], np.float32)
                high = np.asarray(params["action_max"], np.float32)
                history = torch.from_numpy(source["history"]).to(device)
                initial = torch.from_numpy(source["initial_state"]).to(device).reshape(1, 5)
                current_action = torch.from_numpy(source["current_action"]).to(device).reshape(1, 2)
                reference = controller._prepare_reference(source["reference"])
                repeat_payload = []
                for repeat in range(len(context["guided_center_knots"])):
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
                        old_center, proposal_center, sigma, low, high, 0.5, alphas
                    )
                    direction = torch.from_numpy(line["projected_direction"][None]).to(device)
                    rho = torch.tensor([[line["requested_rho"]]], dtype=torch.float32, device=device)
                    scale = torch.tensor([[line["trust_scale"]]], dtype=torch.float32, device=device)
                    with torch.no_grad():
                        _, p_tr2b, a_tr2b = tr2b_policy(
                            *old_input, direction, rho, scale
                        )
                        _, p_ac, a_ac = ac_policy(
                            *old_input, direction, rho, scale
                        )
                    hard_tr2b = tr2b_policy.hard_alpha(
                        p_tr2b, a_tr2b, float(tr2b_payload["move_threshold"])
                    )
                    hard_ac = ac_policy.hard_alpha(
                        p_ac, a_ac, float(ac_payload["move_threshold"])
                    )
                    def one_center(alpha_value: float) -> np.ndarray:
                        raw = old_center + alpha_value * sigma.reshape(1, 2) * line["projected_direction"]
                        return np.clip(raw, low, high).astype(np.float32)
                    tr2b_center = one_center(float(hard_tr2b.item()))
                    ac_center = one_center(float(hard_ac.item()))
                    bank = np.concatenate((
                        line["centers"], tr2b_center[None], ac_center[None]
                    ), axis=0)
                    cost, _, _ = evaluate_knots(
                        controller, backend, bank, history, initial,
                        current_action, reference,
                    )
                    line_cost = cost[:len(alphas)]
                    argmin, safe, tolerance = select_safe_index(
                        line_cost, 0.05, 0.05, 0.05
                    )
                    scenario = str(metadata[episode]["scenario_class"])
                    speed = float(metadata[episode]["reference_speed_mps"])
                    row = {
                        "old_center": old_center,
                        "proposal_center": proposal_center,
                        "projected_direction": line["projected_direction"],
                        "requested_rho": float(line["requested_rho"]),
                        "trust_scale": float(line["trust_scale"]),
                        "line_centers": line["centers"],
                        "line_cost": line_cost.astype(np.float32),
                        "safe_index": safe,
                        "argmin_index": argmin,
                        "safe_tolerance": tolerance,
                        "tr2b_probability": float(p_tr2b.item()),
                        "tr2b_conditional_alpha": float(a_tr2b.item()),
                        "tr2b_alpha": float(hard_tr2b.item()),
                        "tr2b_center": tr2b_center,
                        "tr2b_cost": float(cost[-2]),
                        "alpha_ac_probability": float(p_ac.item()),
                        "alpha_ac_conditional_alpha": float(a_ac.item()),
                        "alpha_ac_alpha": float(hard_ac.item()),
                        "alpha_ac_center": ac_center,
                        "alpha_ac_cost": float(cost[-1]),
                        "old_cost": float(line_cost[0]),
                        "safe_cost": float(line_cost[safe]),
                        "argmin_cost": float(line_cost[argmin]),
                        "reference_speed": speed,
                        "scenario": scenario,
                    }
                    repeat_payload.append(row)
                    for name in (
                        "old_cost", "tr2b_cost", "alpha_ac_cost", "safe_cost",
                        "argmin_cost", "tr2b_alpha", "alpha_ac_alpha",
                        "reference_speed", "scenario",
                    ):
                        arrays[name].append(row[name])
                    arrays["episode"].append(episode)
                    arrays["snapshot_key"].append(f"{episode}/{source_path.stem}")

                out_dir = staging / episode
                out_dir.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(
                    out_dir / source_path.name,
                    format_version=np.asarray(1, np.int32),
                    source_snapshot=np.asarray(str(source_path.resolve())),
                    source_snapshot_sha256=np.asarray(source_hash),
                    context_label=np.asarray(str(context_path.resolve())),
                    context_label_sha256=np.asarray(sha256_file(context_path)),
                    risk_label=np.asarray(str(risk_path.resolve())),
                    risk_label_sha256=np.asarray(sha256_file(risk_path)),
                    episode=np.asarray(episode),
                    scenario=np.asarray(str(metadata[episode]["scenario_class"])),
                    reference_speed_mps=np.asarray(metadata[episode]["reference_speed_mps"], np.float32),
                    alpha_grid=alphas,
                    old_center=np.asarray([row["old_center"] for row in repeat_payload], np.float32),
                    proposal_center=np.asarray([row["proposal_center"] for row in repeat_payload], np.float32),
                    projected_direction=np.asarray([row["projected_direction"] for row in repeat_payload], np.float32),
                    requested_rho=np.asarray([row["requested_rho"] for row in repeat_payload], np.float32),
                    trust_scale=np.asarray([row["trust_scale"] for row in repeat_payload], np.float32),
                    line_centers=np.asarray([row["line_centers"] for row in repeat_payload], np.float32),
                    line_cost=np.asarray([row["line_cost"] for row in repeat_payload], np.float32),
                    safe_index=np.asarray([row["safe_index"] for row in repeat_payload], np.int32),
                    argmin_index=np.asarray([row["argmin_index"] for row in repeat_payload], np.int32),
                    safe_tolerance=np.asarray([row["safe_tolerance"] for row in repeat_payload], np.float32),
                    tr2b_probability=np.asarray([row["tr2b_probability"] for row in repeat_payload], np.float32),
                    tr2b_conditional_alpha=np.asarray([row["tr2b_conditional_alpha"] for row in repeat_payload], np.float32),
                    tr2b_alpha=np.asarray([row["tr2b_alpha"] for row in repeat_payload], np.float32),
                    tr2b_center=np.asarray([row["tr2b_center"] for row in repeat_payload], np.float32),
                    tr2b_cost=np.asarray([row["tr2b_cost"] for row in repeat_payload], np.float32),
                    alpha_ac_probability=np.asarray([row["alpha_ac_probability"] for row in repeat_payload], np.float32),
                    alpha_ac_conditional_alpha=np.asarray([row["alpha_ac_conditional_alpha"] for row in repeat_payload], np.float32),
                    alpha_ac_alpha=np.asarray([row["alpha_ac_alpha"] for row in repeat_payload], np.float32),
                    alpha_ac_center=np.asarray([row["alpha_ac_center"] for row in repeat_payload], np.float32),
                    alpha_ac_cost=np.asarray([row["alpha_ac_cost"] for row in repeat_payload], np.float32),
                )
            if record_index == 1 or record_index % 50 == 0 or record_index == len(records):
                print(f"[{record_index:03d}/{len(records):03d}] validation snapshots", flush=True)

        numpy_arrays = {name: np.asarray(value) for name, value in arrays.items()}
        metrics = summarize_arrays(numpy_arrays)
        config = {
            "format_version": 1,
            "method": "frozen Alpha AC TR3 direct-DBM formal validation",
            "source": str(args.source.resolve()),
            "context": str(args.context.resolve()),
            "risk": str(args.risk.resolve()),
            "alpha_ac": str(args.alpha_ac.resolve()),
            "alpha_ac_sha256": sha256_file(args.alpha_ac),
            "alpha_ac_threshold": float(ac_payload["move_threshold"]),
            "tr2b": str(args.tr2b.resolve()),
            "tr2b_sha256": sha256_file(args.tr2b),
            "tr2b_threshold": float(tr2b_payload["move_threshold"]),
            "old_actor": str(old_path.resolve()),
            "old_actor_sha256": sha256_file(old_path),
            "proposal_actor": str(proposal_path.resolve()),
            "proposal_actor_sha256": sha256_file(proposal_path),
            "validation_episodes": validation_episodes,
            "test_policy": "test episodes 105--119 not opened",
            "trust_radius_sigma_rms": 0.5,
            "alpha_grid": alphas.tolist(),
            "bootstrap_seed": BOOTSTRAP_SEED,
            "bootstrap_repeats": BOOTSTRAP_REPEATS,
            "deviation": "D0",
        }
        summary = {
            "format_version": 1,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "repository": repository_state(Path(__file__).resolve().parents[2]),
            "config": config,
            "metrics": metrics,
            "qualification": metrics["qualification"],
        }
        (staging / "config.json").write_text(json.dumps(config, indent=2) + "\n")
        (staging / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        args.output_dir.parent.mkdir(parents=True, exist_ok=True)
        os.replace(staging, args.output_dir)
        print(json.dumps(summary, indent=2), flush=True)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


if __name__ == "__main__":
    main()
