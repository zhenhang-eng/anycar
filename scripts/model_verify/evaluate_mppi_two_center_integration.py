#!/usr/bin/env python3
"""Evaluate deploy-shaped Gaussian MPPI center integration on frozen DBM states.

This is an internal mechanism diagnostic.  It compares, with common random
numbers and an unchanged 256-candidate budget:

* the original warm-centered Gaussian bank;
* an Actor-centered replacement bank;
* an Actor-centered bank whose candidate one is the exact warm center; and
* a hard model-cost fallback between that soft-bank weighted output and warm.

The hard arm costs one additional rollout of the weighted output.  No formal
validation/test episode is loaded and no model parameter is updated.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch

from car_dynamics.controllers_torch.dbm import (
    TorchDBMParams,
    TorchDynamicBicycleRolloutBackend,
)
from car_dynamics.controllers_torch.mppi import TorchMPPICostWeights, TorchMPPIParams
from car_foundation.mppi_proposal_policy import TorchMPPIDeterministicCenterActor
from generate_dbm_direct_gt_validation import batched_cost, interpolate_knots
from generate_dbm_proposal_teacher import repository_state, sha256_file
from train_mppi_direct_alpha_online_sac import deterministic_outputs
from train_mppi_direct_residual_online_ac import (
    actor_inputs,
    make_base_policy,
    residual_outputs,
)
from train_mppi_direct_trust_alpha_policy import extra_tensors
from train_mppi_direct_trust_region_actor import (
    distribution,
    load_actor_payload,
    load_dataset,
    tensorize,
)


DEFAULT_CHECKPOINT = Path(
    "outputs/mppi_proposal/direct_residual_online_ac_20260811_v2/"
    "direct_residual_online_ac_selected.pt"
)
DEFAULT_OUTPUT = Path(
    "outputs/mppi_proposal/two_center_integration_20260813_v2"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--partition", choices=("internal_fit", "internal_selection"),
        default="internal_selection",
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=(3407, 3408, 3409))
    parser.add_argument("--num-samples", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-contexts", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def load_warm_centers(labels: Path, train_episodes: set[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Recover warm centers in exactly the row order used by ``load_dataset``."""
    centers: list[np.ndarray] = []
    source_paths: list[str] = []
    context_in_file: list[int] = []
    for label_path in sorted(labels.glob("episode_*/*.npz")):
        if label_path.parent.name not in train_episodes:
            continue
        with np.load(label_path, allow_pickle=False) as label:
            source_path = Path(str(label["source_snapshot"]))
            repeats = len(label["safe_index"])
        with np.load(source_path, allow_pickle=False) as source:
            warm = np.asarray(source["sampling_mean_knots"], np.float32)
        for repeat in range(repeats):
            centers.append(warm)
            source_paths.append(str(source_path))
            context_in_file.append(repeat)
    return (
        np.asarray(centers, np.float32),
        np.asarray(source_paths),
        np.asarray(context_in_file, np.int16),
    )


def stable_subset(index: np.ndarray, episodes: np.ndarray, maximum: int) -> np.ndarray:
    """Take a deterministic episode-balanced subset without random tuning."""
    if maximum <= 0 or len(index) <= maximum:
        return index
    episode_order = list(dict.fromkeys(episodes[index].tolist()))
    selected: list[int] = []
    cursor = {episode: 0 for episode in episode_order}
    pools = {episode: index[episodes[index] == episode].tolist() for episode in episode_order}
    while len(selected) < maximum:
        progressed = False
        for episode in episode_order:
            position = cursor[episode]
            if position < len(pools[episode]):
                selected.append(pools[episode][position])
                cursor[episode] += 1
                progressed = True
                if len(selected) == maximum:
                    break
        if not progressed:
            break
    return np.asarray(selected, np.int64)


def percentile_distribution(value: np.ndarray) -> dict[str, float]:
    result = distribution(np.asarray(value, np.float64))
    result["p01"] = float(np.quantile(value, 0.01))
    result["p10"] = float(np.quantile(value, 0.10))
    result["p25"] = float(np.quantile(value, 0.25))
    result["p75"] = float(np.quantile(value, 0.75))
    result["p99"] = float(np.quantile(value, 0.99))
    return result


def gain_metrics(baseline: np.ndarray, cost: np.ndarray) -> dict[str, Any]:
    gain = np.asarray(baseline) - np.asarray(cost)
    return {
        "cost": percentile_distribution(cost),
        "gain_vs_warm_gaussian": percentile_distribution(gain),
        "improved_fraction": float(np.mean(gain > 1e-6)),
        "tied_or_improved_fraction": float(np.mean(gain >= -1e-6)),
        "regression_fraction": float(np.mean(gain < -1e-6)),
        "regression_sum": float(np.minimum(gain, 0.0).sum()),
        "improvement_sum": float(np.maximum(gain, 0.0).sum()),
    }


@torch.no_grad()
def direct_sequence_cost(
    backend: TorchDynamicBicycleRolloutBackend,
    weights: TorchMPPICostWeights,
    sequence: torch.Tensor,
    initial: torch.Tensor,
    current: torch.Tensor,
    reference: torch.Tensor,
) -> torch.Tensor:
    return batched_cost(
        backend, weights, sequence.unsqueeze(1), initial, current, reference
    )[:, 0]


@torch.no_grad()
def evaluate_bank(
    center: torch.Tensor,
    noise: torch.Tensor,
    warm: torch.Tensor,
    include_exact_warm: bool,
    params: TorchMPPIParams,
    backend: TorchDynamicBicycleRolloutBackend,
    weights: TorchMPPICostWeights,
    initial: torch.Tensor,
    current: torch.Tensor,
    reference: torch.Tensor,
) -> dict[str, torch.Tensor]:
    raw = center[:, None] + noise
    knots = torch.clamp(
        raw,
        torch.as_tensor(params.action_min, device=center.device),
        torch.as_tensor(params.action_max, device=center.device),
    )
    if include_exact_warm:
        knots[:, 1] = warm
        raw[:, 1] = warm
    sequence = interpolate_knots(knots, params.horizon)
    candidate_cost = batched_cost(
        backend, weights, sequence, initial, current, reference
    )
    candidate_weight = torch.softmax(
        -(candidate_cost - candidate_cost.min(dim=1, keepdim=True).values)
        / params.temperature,
        dim=1,
    )
    weighted_sequence = torch.sum(
        candidate_weight[:, :, None, None] * sequence, dim=1
    )
    weighted_cost = direct_sequence_cost(
        backend, weights, weighted_sequence, initial, current, reference
    )
    return {
        "candidate_cost": candidate_cost,
        "candidate_weight": candidate_weight,
        "weighted_sequence": weighted_sequence,
        "weighted_cost": weighted_cost,
        "clip_fraction": (knots != raw).float().mean(dim=(1, 2, 3)),
    }


def arm_summary(
    candidate_cost: np.ndarray,
    candidate_weight: np.ndarray,
    weighted_cost: np.ndarray,
    baseline: np.ndarray,
    clip_fraction: np.ndarray,
    exact_index: int,
) -> dict[str, Any]:
    ess = 1.0 / np.square(candidate_weight).sum(axis=-1)
    best = candidate_cost.argmin(axis=-1)
    return {
        **gain_metrics(baseline, weighted_cost),
        "candidate_best": percentile_distribution(candidate_cost.min(axis=-1)),
        "candidate_p10": percentile_distribution(np.quantile(candidate_cost, 0.10, axis=-1)),
        "candidate_median": percentile_distribution(np.median(candidate_cost, axis=-1)),
        "effective_sample_size": percentile_distribution(ess),
        "maximum_weight": percentile_distribution(candidate_weight.max(axis=-1)),
        "exact_candidate_weight": percentile_distribution(candidate_weight[..., exact_index]),
        "exact_candidate_best_fraction": float(np.mean(best == exact_index)),
        "candidate_clip_fraction": percentile_distribution(clip_fraction),
    }


def grouped_metrics(
    baseline: np.ndarray,
    costs: dict[str, np.ndarray],
    mask: np.ndarray,
) -> dict[str, Any]:
    repeated_mask = np.broadcast_to(mask[None, :], baseline.shape)
    result: dict[str, Any] = {"contexts": int(mask.sum())}
    for name, value in costs.items():
        result[name] = gain_metrics(baseline[repeated_mask], value[repeated_mask])
    return result


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    if args.num_samples != 256:
        raise ValueError("this frozen deployment diagnostic requires exactly 256 samples")
    if len(set(args.seeds)) != len(args.seeds):
        raise ValueError("evaluation seeds must be unique")
    args.output_dir.mkdir(parents=True)
    device = torch.device(args.device)

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    if checkpoint.get("actor_class") != "TorchMPPIDeterministicCenterActor":
        raise AssertionError("checkpoint is not the residual deterministic Actor")
    labels = Path(checkpoint["labels"])
    base_path = Path(checkpoint["base_alpha_checkpoint"])
    base_payload = torch.load(base_path, map_location="cpu")
    if sha256_file(base_path) != checkpoint["base_alpha_sha256"]:
        raise AssertionError("base Alpha checkpoint hash mismatch")
    old_payload = load_actor_payload(Path(base_payload["old_actor"]))
    data, _, splits = load_dataset(labels, old_payload)
    warm_all, source_paths, context_in_file = load_warm_centers(
        labels, set(splits["train"])
    )
    if len(warm_all) != len(data.episodes):
        raise AssertionError("warm-center rows do not match reconstructed Actor rows")

    episodes = list(splits[args.partition])
    selected = np.flatnonzero(np.isin(data.episodes, episodes))
    selected = stable_subset(selected, data.episodes, args.max_contexts)
    if not len(selected):
        raise ValueError("selected partition contains no contexts")
    if set(data.episodes[selected]) & set(splits.get("test_sealed_not_generated", [])):
        raise AssertionError("sealed test episode was selected")
    if set(data.episodes[selected]) & set(splits.get("formal_validation_sealed_not_generated", [])):
        raise AssertionError("sealed validation episode was selected")

    tensors = tensorize(data, device)
    extra = extra_tensors(data, device)
    base_policy = make_base_policy(base_payload, device)
    _, _, _, base_center = deterministic_outputs(
        base_policy, tensors, extra, np.arange(len(data.episodes)),
        float(base_payload["move_threshold"]), args.batch_size, device,
    )
    inputs = actor_inputs(tensors, base_center, device)
    actor = TorchMPPIDeterministicCenterActor(
        float(checkpoint["maximum_residual_sigma"]), dropout=0.0
    ).to(device)
    actor.load_state_dict(checkpoint["actor_state_dict"], strict=True)
    _, actor_center_all = residual_outputs(
        actor, inputs, np.arange(len(data.episodes)), args.batch_size, device
    )

    source_params = TorchMPPIParams(**data.mppi_params)
    params_dict = asdict(source_params)
    params_dict.update({
        "num_samples": args.num_samples,
        "num_iterations": 1,
        "sampling_mode": "gaussian",
    })
    params = TorchMPPIParams(**params_dict)
    if tuple(params.noise_sigma) != (0.25, 0.35) or params.temperature != 1.0:
        raise AssertionError("deployment noise/temperature contract changed")
    weights = TorchMPPICostWeights(**data.cost_weights)
    backend = TorchDynamicBicycleRolloutBackend(TorchDBMParams(**data.dbm_params))

    warm = torch.from_numpy(warm_all[selected]).to(device)
    actor_center = torch.from_numpy(actor_center_all[selected]).to(device)
    direct_warm_sequence = interpolate_knots(warm, params.horizon)
    direct_actor_sequence = interpolate_knots(actor_center, params.horizon)
    initial = tensors["initial_state_six"][torch.from_numpy(selected).to(device)]
    current = tensors["current_action"][torch.from_numpy(selected).to(device)]
    reference = tensors["direct_reference"][torch.from_numpy(selected).to(device)]
    direct_warm_cost = direct_sequence_cost(
        backend, weights, direct_warm_sequence, initial, current, reference
    ).cpu().numpy()
    direct_actor_cost = direct_sequence_cost(
        backend, weights, direct_actor_sequence, initial, current, reference
    ).cpu().numpy()

    names = ("warm_gaussian", "actor_replace", "soft_two_center")
    collected: dict[str, dict[str, list[np.ndarray]]] = {
        name: {field: [] for field in (
            "candidate_cost", "candidate_weight", "weighted_sequence",
            "weighted_cost", "clip_fraction",
        )} for name in names
    }
    elapsed: dict[str, float] = {name: 0.0 for name in names}
    sigma = torch.as_tensor(params.noise_sigma, dtype=torch.float32).reshape(1, 1, 1, 2)
    for seed in args.seeds:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        noise_cpu = torch.randn(
            len(selected), args.num_samples, params.num_knots, params.action_dim,
            generator=generator,
        ) * sigma
        noise_cpu[:, 0].zero_()
        for start in range(0, len(selected), args.batch_size):
            stop = min(start + args.batch_size, len(selected))
            noise = noise_cpu[start:stop].to(device)
            one_warm = warm[start:stop]
            one_actor = actor_center[start:stop]
            arguments = {
                "noise": noise,
                "warm": one_warm,
                "params": params,
                "backend": backend,
                "weights": weights,
                "initial": initial[start:stop],
                "current": current[start:stop],
                "reference": reference[start:stop],
            }
            arm_config = (
                ("warm_gaussian", one_warm, False),
                ("actor_replace", one_actor, False),
                ("soft_two_center", one_actor, True),
            )
            for name, center, include_warm in arm_config:
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                before = time.perf_counter()
                result = evaluate_bank(
                    center=center, include_exact_warm=include_warm, **arguments
                )
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                elapsed[name] += time.perf_counter() - before
                for field, value in result.items():
                    collected[name][field].append(value.cpu().numpy())

    arrays: dict[str, np.ndarray] = {}
    seed_count = len(args.seeds)
    for name in names:
        for field in collected[name]:
            concatenated = np.concatenate(collected[name][field], axis=0)
            arrays[f"{name}_{field}"] = concatenated.reshape(
                seed_count, len(selected), *concatenated.shape[1:]
            )

    baseline = arrays["warm_gaussian_weighted_cost"]
    soft_cost = arrays["soft_two_center_weighted_cost"]
    warm_direct = np.broadcast_to(direct_warm_cost[None, :], soft_cost.shape)
    hard_cost = np.minimum(soft_cost, warm_direct)
    hard_uses_warm = warm_direct < soft_cost
    paired_controller_oracle = np.minimum(
        arrays["actor_replace_weighted_cost"], baseline
    )
    actor_direct = np.broadcast_to(direct_actor_cost[None, :], baseline.shape)
    warm_mppi_vs_direct_actor_hard = np.minimum(baseline, actor_direct)

    summaries = {
        "warm_gaussian": arm_summary(
            arrays["warm_gaussian_candidate_cost"],
            arrays["warm_gaussian_candidate_weight"], baseline, baseline,
            arrays["warm_gaussian_clip_fraction"], 0,
        ),
        "actor_replace": arm_summary(
            arrays["actor_replace_candidate_cost"],
            arrays["actor_replace_candidate_weight"],
            arrays["actor_replace_weighted_cost"], baseline,
            arrays["actor_replace_clip_fraction"], 0,
        ),
        "soft_two_center": arm_summary(
            arrays["soft_two_center_candidate_cost"],
            arrays["soft_two_center_candidate_weight"], soft_cost, baseline,
            arrays["soft_two_center_clip_fraction"], 1,
        ),
        "hard_fallback": {
            **gain_metrics(baseline, hard_cost),
            "uses_warm_direct_fraction": float(np.mean(hard_uses_warm)),
            "model_floor_violation_max": float(np.max(hard_cost - warm_direct)),
        },
        "paired_controller_oracle_512_candidate_auxiliary": {
            **gain_metrics(baseline, paired_controller_oracle),
            "uses_warm_controller_fraction": float(np.mean(
                baseline < arrays["actor_replace_weighted_cost"]
            )),
        },
        "warm_mppi_vs_direct_actor_hard_258": {
            **gain_metrics(baseline, warm_mppi_vs_direct_actor_hard),
            "uses_actor_direct_fraction": float(np.mean(actor_direct < baseline)),
            "model_baseline_floor_violation_max": float(np.max(
                warm_mppi_vs_direct_actor_hard - baseline
            )),
        },
        "direct_centers": {
            "warm_cost": percentile_distribution(direct_warm_cost),
            "actor_cost": percentile_distribution(direct_actor_cost),
            "actor_gain_vs_warm": percentile_distribution(
                direct_warm_cost - direct_actor_cost
            ),
        },
    }
    summaries["soft_two_center"]["warm_candidate_weight"] = percentile_distribution(
        arrays["soft_two_center_candidate_weight"][..., 1]
    )
    summaries["soft_two_center"]["warm_candidate_best_fraction"] = float(np.mean(
        arrays["soft_two_center_candidate_cost"].argmin(axis=-1) == 1
    ))
    summaries["soft_two_center"]["actor_candidate_weight"] = percentile_distribution(
        arrays["soft_two_center_candidate_weight"][..., 0]
    )
    summaries["soft_two_center"]["actor_candidate_best_fraction"] = float(np.mean(
        arrays["soft_two_center_candidate_cost"].argmin(axis=-1) == 0
    ))

    costs = {
        "actor_replace": arrays["actor_replace_weighted_cost"],
        "soft_two_center": soft_cost,
        "hard_fallback": hard_cost,
        "warm_mppi_vs_direct_actor_hard_258": warm_mppi_vs_direct_actor_hard,
    }
    groups: dict[str, Any] = {}
    selected_speed = data.reference_speed[selected]
    selected_scenario = data.scenario[selected]
    for speed in sorted(np.unique(selected_speed)):
        mask = np.isclose(selected_speed, speed)
        groups[f"speed_{speed:.1f}_mps"] = grouped_metrics(baseline, costs, mask)
    for scenario in sorted(np.unique(selected_scenario)):
        mask = selected_scenario == scenario
        groups[f"scenario_{scenario}"] = grouped_metrics(baseline, costs, mask)
    recovery_mask = np.char.find(selected_scenario.astype(str), "recovery") >= 0
    if np.any(recovery_mask):
        groups["scenario_any_recovery"] = grouped_metrics(
            baseline, costs, recovery_mask
        )

    arrays.update({
        "selected_index": selected,
        "episode": data.episodes[selected],
        "source_path": source_paths[selected],
        "context_in_file": context_in_file[selected],
        "reference_speed_mps": selected_speed,
        "scenario": selected_scenario,
        "warm_center": warm_all[selected],
        "actor_center": actor_center_all[selected],
        "direct_warm_cost": direct_warm_cost,
        "direct_actor_cost": direct_actor_cost,
        "hard_fallback_cost": hard_cost,
        "hard_uses_warm": hard_uses_warm,
        "paired_controller_oracle_cost": paired_controller_oracle,
        "warm_mppi_vs_direct_actor_hard_258_cost": warm_mppi_vs_direct_actor_hard,
    })
    np.savez_compressed(args.output_dir / "evaluation.npz", **arrays)

    summary = {
        "format_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "method": "fixed-DBM deploy-Gaussian two-center integration diagnostic",
        "qualification": (
            "CONSUMED_INTERNAL_SELECTION_MECHANISM_ONLY"
            if args.partition == "internal_selection"
            else "TRAIN_INTERNAL_FIT_MECHANISM_ONLY"
        ),
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "base_checkpoint": str(base_path.resolve()),
        "base_checkpoint_sha256": sha256_file(base_path),
        "labels": str(labels.resolve()),
        "label_hashes": {
            name: sha256_file(labels / name)
            for name in ("config.json", "splits.json", "summary.json")
        },
        "partition": args.partition,
        "episode_count": int(len(np.unique(data.episodes[selected]))),
        "context_count": int(len(selected)),
        "evaluation_seeds": list(args.seeds),
        "candidate_contract": {
            **asdict(params),
            "common_random_numbers": True,
            "noise_generation": "CPU torch.Generator, one frozen stream per seed",
            "candidate_0": "exact bank center",
            "soft_candidate_1": "exact original warm center; replaces Gaussian candidate 1",
            "soft_remaining_candidates": 254,
            "hard_fallback": "min(model cost of soft weighted output, direct warm sequence)",
            "hard_extra_rollouts": 1,
            "baseline_preserving_258_guard": (
                "original warm Gaussian 256 + Actor direct 1 + warm weighted-output "
                "cost evaluation 1; hard min without action blending"
            ),
            "validation_output_cost_rollouts_not_deployment_budget": 3,
        },
        "cost_weights": asdict(weights),
        "dbm_params": asdict(TorchDBMParams(**data.dbm_params)),
        "elapsed_seconds": elapsed,
        "summaries": summaries,
        "groups": groups,
        "repository_state": repository_state(Path.cwd()),
        "caveats": [
            "No formal validation or sealed test episode was loaded.",
            "The internal-selection partition was already consumed by prior checkpoint selection.",
            "Frozen-state model cost does not establish closed-loop benefit.",
            "Hard fallback is a strict floor only in the rollout-model cost space.",
            "The paired-controller oracle doubles candidate rollout budget and is auxiliary only.",
            "The 258-rollout guard is deployable in model space but still needs Query/real ranking qualification.",
        ],
        "test_policy": "formal validation and test remain sealed and were not generated/evaluated",
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
