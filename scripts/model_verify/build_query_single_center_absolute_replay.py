#!/usr/bin/env python3
"""Build the validated absolute-action Query Replay for single-center OAC."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "car_foundation"))

from car_foundation.mppi_proposal_policy import ego_reference_features  # noqa: E402


DEFAULT_CONFIG = REPO_ROOT / (
    "scripts/model_verify/query_single_center_oac_config_20260902_v1.json"
)
CONTEXT_FIELDS = (
    "row_index",
    "episode_id",
    "episode_index",
    "row_in_episode",
    "control_step",
    "speed_kph",
    "speed_index",
    "variant_index",
    "road_name",
    "fold_id",
    "source_snapshot_sha256",
    "state",
    "current_action",
    "history",
    "reference",
    "reference_ego",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def dump_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def stats(values: np.ndarray) -> dict[str, float | int]:
    values = np.asarray(values, np.float64).reshape(-1)
    return {
        "count": int(values.size),
        "min": float(values.min()),
        "p05": float(np.quantile(values, 0.05)),
        "median": float(np.median(values)),
        "mean": float(values.mean()),
        "p95": float(np.quantile(values, 0.95)),
        "max": float(values.max()),
    }


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text())
    fullrank = Path(config["sources"]["fullrank_600"]).resolve()
    landscape = Path(config["sources"]["forward_response_full_100"]).resolve()
    output = Path(config["outputs"]["absolute_replay"]).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to replace existing output: {output}")
    if config["formal_validation_or_test_consumed"]:
        raise AssertionError("formal validation/test must remain sealed")
    if config.get("dbm_fields_or_labels_consumed"):
        raise AssertionError("DBM fields or labels are forbidden")

    fullrank_manifest_path = fullrank / "manifest.json"
    fullrank_validation_path = fullrank / "validation.json"
    fullrank_bank_path = fullrank / "bank.npz"
    landscape_manifest_path = landscape / "manifest.json"
    landscape_validation_path = landscape / "validation.json"
    landscape_bank_path = landscape / "landscape.npz"
    fullrank_manifest = json.loads(fullrank_manifest_path.read_text())
    fullrank_validation = json.loads(fullrank_validation_path.read_text())
    landscape_manifest = json.loads(landscape_manifest_path.read_text())
    landscape_validation = json.loads(landscape_validation_path.read_text())
    if fullrank_validation["qualification"] != "QUERY_EXPECTED_ROAD_FULLRANK_PASS":
        raise AssertionError("full-rank source did not pass")
    if landscape_validation["qualification"] != "QUERY_FORWARD_RESPONSE_FULL100_INDEPENDENT_PASS":
        raise AssertionError("landscape source did not pass")
    if sha256(fullrank_bank_path) != fullrank_manifest["bank_sha256"]:
        raise AssertionError("full-rank bank hash mismatch")
    if sha256(landscape_bank_path) != landscape_manifest["landscape_sha256"]:
        raise AssertionError("landscape bank hash mismatch")
    if fullrank_manifest.get("formal_validation_or_test_consumed", True):
        raise AssertionError("full-rank source consumed formal validation/test")
    if landscape_manifest.get("formal_validation_or_test_consumed", True):
        raise AssertionError("landscape source consumed formal validation/test")

    with np.load(fullrank_bank_path, allow_pickle=False) as archive:
        full = {name: np.asarray(archive[name]) for name in archive.files}
    with np.load(landscape_bank_path, allow_pickle=False) as archive:
        land = {name: np.asarray(archive[name]) for name in archive.files}
    if full["candidate_knots"].shape != (600, 132, 8, 2):
        raise AssertionError("unexpected full-rank candidate shape")
    if land["source_audit_row"].shape != (100,):
        raise AssertionError("unexpected landscape row count")
    full_row_lookup = {int(value): index for index, value in enumerate(full["row_index"])}
    mapped_rows = np.asarray([full_row_lookup[int(value)] for value in land["row_index"]])
    if len(np.unique(mapped_rows)) != 100:
        raise AssertionError("landscape/full-rank mapping is not one-to-one")
    for name in (
        "episode_id",
        "row_in_episode",
        "control_step",
        "state",
        "current_action",
        "history",
        "reference",
        "reference_ego",
        "fold_id",
    ):
        if not np.array_equal(full[name][mapped_rows], land[name]):
            raise AssertionError(f"landscape context mismatch: {name}")

    state_count, base_count, landscape_count = 600, 132, 765
    maximum_count = base_count + landscape_count
    candidate_knots = np.zeros((state_count, maximum_count, 8, 2), np.float32)
    candidate_cost = np.zeros((state_count, maximum_count), np.float32)
    candidate_valid = np.zeros((state_count, maximum_count), bool)
    candidate_clipped = np.zeros((state_count, maximum_count), bool)
    candidate_source = np.full((state_count, maximum_count), -1, np.int8)
    candidate_source_index = np.full((state_count, maximum_count), -1, np.int16)
    candidate_branch = np.full((state_count, maximum_count), -1, np.int8)
    candidate_round = np.full((state_count, maximum_count), -1, np.int8)
    candidate_local_index = np.full((state_count, maximum_count), -1, np.int16)
    candidate_shadow = np.zeros((state_count, maximum_count), bool)
    candidate_canonical_eligible = np.zeros((state_count, maximum_count), bool)

    candidate_knots[:, :base_count] = full["candidate_knots"]
    candidate_cost[:, :base_count] = full["candidate_cost"]
    candidate_valid[:, :base_count] = True
    candidate_clipped[:, :base_count] = np.any(
        full["candidate_clipped_mask"], axis=(2, 3)
    )
    candidate_source[:, :base_count] = 0
    candidate_source_index[:, :base_count] = np.arange(base_count, dtype=np.int16)
    candidate_canonical_eligible[:, :base_count] = True

    for local_row, replay_row in enumerate(mapped_rows):
        knots = [land["center_knots"][local_row, :, 0]]
        costs = [land["center_cost"][local_row, :, 0]]
        clipped = [np.zeros(5, bool)]
        sources = [np.ones(5, np.int8)]
        branches = [np.arange(5, dtype=np.int8)]
        rounds = [np.full(5, -1, np.int8)]
        local_indices = [np.arange(5, dtype=np.int16)]
        shadows = [np.arange(5) == 4]
        eligible = [np.arange(5) < 4]
        source_indices = [np.arange(5, dtype=np.int16)]
        running_source_index = 5
        for round_index in range(4):
            for branch in range(5):
                knots.append(land["probe_knots"][local_row, branch, round_index])
                costs.append(land["probe_cost"][local_row, branch, round_index])
                clipped.append(
                    np.any(
                        land["probe_clipped_mask"][local_row, branch, round_index],
                        axis=(1, 2),
                    )
                )
                sources.append(np.full(32, 2, np.int8))
                branches.append(np.full(32, branch, np.int8))
                rounds.append(np.full(32, round_index, np.int8))
                local_indices.append(np.arange(32, dtype=np.int16))
                shadows.append(np.full(32, branch == 4, bool))
                eligible.append(np.full(32, branch < 4, bool))
                source_indices.append(
                    np.arange(running_source_index, running_source_index + 32, dtype=np.int16)
                )
                running_source_index += 32
                knots.append(land["proposal_knots"][local_row, branch, round_index])
                costs.append(land["proposal_cost"][local_row, branch, round_index])
                clipped.append(
                    np.any(
                        land["proposal_clipped_mask"][local_row, branch, round_index],
                        axis=(1, 2),
                    )
                )
                sources.append(np.full(6, 3, np.int8))
                branches.append(np.full(6, branch, np.int8))
                rounds.append(np.full(6, round_index, np.int8))
                local_indices.append(np.arange(6, dtype=np.int16))
                shadows.append(np.full(6, branch == 4, bool))
                eligible.append(np.full(6, branch < 4, bool))
                source_indices.append(
                    np.arange(running_source_index, running_source_index + 6, dtype=np.int16)
                )
                running_source_index += 6
        flat_knots = np.concatenate(knots)
        flat_cost = np.concatenate(costs)
        flat_clipped = np.concatenate(clipped)
        flat_source = np.concatenate(sources)
        flat_branch = np.concatenate(branches)
        flat_round = np.concatenate(rounds)
        flat_local = np.concatenate(local_indices)
        flat_shadow = np.concatenate(shadows)
        flat_eligible = np.concatenate(eligible)
        flat_source_index = np.concatenate(source_indices)
        if len(flat_knots) != landscape_count or running_source_index != landscape_count:
            raise AssertionError("landscape flattening count mismatch")
        start, stop = base_count, maximum_count
        candidate_knots[replay_row, start:stop] = flat_knots
        candidate_cost[replay_row, start:stop] = flat_cost
        candidate_valid[replay_row, start:stop] = True
        candidate_clipped[replay_row, start:stop] = flat_clipped
        candidate_source[replay_row, start:stop] = flat_source
        candidate_source_index[replay_row, start:stop] = flat_source_index
        candidate_branch[replay_row, start:stop] = flat_branch
        candidate_round[replay_row, start:stop] = flat_round
        candidate_local_index[replay_row, start:stop] = flat_local
        candidate_shadow[replay_row, start:stop] = flat_shadow
        candidate_canonical_eligible[replay_row, start:stop] = flat_eligible

    data = {name: full[name] for name in CONTEXT_FIELDS}
    data.update(
        {
            "critic_reference": np.stack(
                [
                    ego_reference_features(value, float(state[3]))
                    for value, state in zip(full["reference_ego"], full["state"])
                ]
            ).astype(np.float32),
            "critic_current": np.concatenate(
                (full["state"][:, 3:5], full["current_action"]), axis=1
            ).astype(np.float32),
            "warm_knots": full["mean_knots_before"].astype(np.float32),
            "warm_cost": full["warm_direct_cost_replayed"].astype(np.float32),
            "fullrank_teacher_knots": full["fullrank_teacher_knots"].astype(np.float32),
            "fullrank_teacher_cost": full["fullrank_teacher_direct_cost"].astype(np.float32),
            "landscape_context_mask": np.isin(np.arange(state_count), mapped_rows),
            "landscape_source_audit_row": np.where(
                np.isin(np.arange(state_count), mapped_rows),
                np.searchsorted(mapped_rows, np.arange(state_count)),
                -1,
            ).astype(np.int16),
            "candidate_knots": candidate_knots,
            "candidate_cost": candidate_cost,
            "candidate_valid_mask": candidate_valid,
            "candidate_clipped_mask": candidate_clipped,
            "candidate_source": candidate_source,
            "candidate_source_index": candidate_source_index,
            "candidate_branch": candidate_branch,
            "candidate_round": candidate_round,
            "candidate_local_index": candidate_local_index,
            "candidate_shadow_mask": candidate_shadow,
            "candidate_canonical_eligible_mask": candidate_canonical_eligible,
            "candidate_count": np.sum(candidate_valid, axis=1).astype(np.int16),
        }
    )
    # searchsorted is only valid for members; explicitly set the exact local mapping.
    data["landscape_source_audit_row"][:] = -1
    data["landscape_source_audit_row"][mapped_rows] = np.arange(100, dtype=np.int16)
    valid_cost = candidate_cost[candidate_valid]
    if not np.all(np.isfinite(valid_cost)) or not np.all(np.isfinite(candidate_knots[candidate_valid])):
        raise AssertionError("non-finite valid candidate data")
    if int(np.sum(candidate_valid)) != 155700:
        raise AssertionError("valid candidate count mismatch")

    summary = {
        "qualification": "PENDING_INDEPENDENT_VALIDATION",
        "state_count": state_count,
        "episode_count": int(len(np.unique(data["episode_id"]))),
        "fullrank_candidates_per_state": base_count,
        "landscape_context_count": 100,
        "landscape_candidates_per_context": landscape_count,
        "maximum_candidates_per_state": maximum_count,
        "valid_candidate_count": int(np.sum(candidate_valid)),
        "candidate_cost": stats(valid_cost),
        "candidate_cost_by_source": {
            "fullrank": stats(candidate_cost[candidate_source == 0]),
            "landscape_initial": stats(candidate_cost[candidate_source == 1]),
            "landscape_probe": stats(candidate_cost[candidate_source == 2]),
            "landscape_response_proposal": stats(candidate_cost[candidate_source == 3]),
        },
        "clipped_candidate_fraction": float(
            np.mean(candidate_clipped[candidate_valid])
        ),
        "shadow_candidate_count": int(np.sum(candidate_shadow)),
        "canonical_eligible_candidate_count": int(np.sum(candidate_canonical_eligible)),
        "states_per_fold": {
            str(fold): int(np.sum(data["fold_id"] == fold)) for fold in range(5)
        },
        "candidate_sampling_contract": (
            "sample states/contexts first and candidates second; never sample uniformly "
            "over all 155700 rows"
        ),
        "formal_validation_or_test_consumed": False,
        "dbm_fields_or_labels_consumed": [],
    }

    output.mkdir(parents=True)
    replay_path = output / "replay.npz"
    np.savez_compressed(replay_path, **data)
    summary_path = output / "summary.json"
    dump_json(summary_path, summary)
    manifest = {
        "schema_version": "query-single-center-absolute-replay-v1",
        "dataset_type": "train-only-pure-query-absolute-action-cost-replay",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "qualification": "PENDING_INDEPENDENT_VALIDATION",
        "config": str(config_path),
        "config_sha256": sha256(config_path),
        "fullrank_source": str(fullrank),
        "fullrank_manifest_sha256": sha256(fullrank_manifest_path),
        "fullrank_validation_sha256": sha256(fullrank_validation_path),
        "fullrank_bank_sha256": sha256(fullrank_bank_path),
        "landscape_source": str(landscape),
        "landscape_manifest_sha256": sha256(landscape_manifest_path),
        "landscape_validation_sha256": sha256(landscape_validation_path),
        "landscape_bank_sha256": sha256(landscape_bank_path),
        "query_checkpoint": fullrank_manifest["query_checkpoint"],
        "query_checkpoint_sha256": fullrank_manifest["query_checkpoint_sha256"],
        "replay_sha256": sha256(replay_path),
        "summary_sha256": sha256(summary_path),
        "state_count": state_count,
        "episode_count": int(len(np.unique(data["episode_id"]))),
        "valid_candidate_count": int(np.sum(candidate_valid)),
        "formal_validation_or_test_consumed": False,
        "dbm_fields_or_labels_consumed": [],
        "script": str(Path(__file__).resolve()),
        "script_sha256": sha256(Path(__file__).resolve()),
        "limitations": [
            "This replay is for scalar absolute-cost/ranking Critic training, not hard-label Actor BC.",
            "Candidate generation paths are provenance fields and are never Critic inputs.",
            "The 100 landscape contexts have more candidates; training must sample states first.",
            "Formal validation/test, wrapper, and closed loop remain sealed.",
        ],
    }
    dump_json(output / "manifest.json", manifest)
    print(json.dumps(summary, indent=2))
    print(f"output: {output}")


if __name__ == "__main__":
    main()
