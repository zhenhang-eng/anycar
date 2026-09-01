#!/usr/bin/env python3
"""Build disjoint g0/H routing strata and a deterministic probe pilot manifest."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any


DEFAULT_SOURCE = Path(
    "outputs/mppi_proposal/g0_learnability_audit_20260814_v1/"
    "g0_priority_manifest.json"
)
DEFAULT_OUTPUT = Path(
    "outputs/mppi_proposal/targeted_local_probe_manifest_20260814_v1"
)
PILOT_TARGET_STRATA = ("JOINT_G0_H", "H_ONLY")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--matched-control-count", type=int, default=21)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stratum(row: dict[str, Any]) -> tuple[str, int]:
    g0 = bool(row["g0_knn_negative"])
    h = bool(row["h_oracle_cross_negative"])
    critic = bool(row["current_critic_hard"])
    if g0 and h:
        return "JOINT_G0_H", 0
    if h:
        return "H_ONLY", 1
    if g0 and critic:
        return "G0_ONLY_CRITIC_HARD", 2
    if g0:
        return "G0_ONLY_CRITIC_OK", 3
    if critic:
        return "CRITIC_ONLY", 4
    return "ALL_GOOD", 5


def group_key(row: dict[str, Any]) -> tuple[float, str, bool]:
    return (
        round(float(row["reference_speed_mps"]), 3),
        str(row["scenario"]),
        bool(row["clipped"]),
    )


def choose_controls(
    rows: list[dict[str, Any]], target: list[dict[str, Any]], count: int,
) -> set[int]:
    if count < 0:
        raise ValueError("matched-control-count must be nonnegative")
    controls: dict[tuple[float, str, bool], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["stratum_id"] == "ALL_GOOD":
            controls[group_key(row)].append(row)
    for values in controls.values():
        values.sort(key=lambda row: (row["episode"], row["context_index"]))
    demand = Counter(group_key(row) for row in target)
    groups = sorted(demand, key=lambda key: (-demand[key], key))
    chosen: set[int] = set()
    while len(chosen) < count:
        progressed = False
        for key in groups:
            if controls[key]:
                chosen.add(int(controls[key].pop(0)["context_index"]))
                progressed = True
                if len(chosen) == count:
                    break
        if not progressed:
            break
    if len(chosen) < count:
        remaining = sorted(
            (row for values in controls.values() for row in values),
            key=lambda row: (group_key(row), row["episode"], row["context_index"]),
        )
        for row in remaining[: count - len(chosen)]:
            chosen.add(int(row["context_index"]))
    if len(chosen) != count:
        raise AssertionError(f"could select only {len(chosen)}/{count} controls")
    return chosen


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    source = json.loads(args.source.read_text())
    rows = []
    for original in source["rows"]:
        row = dict(original)
        row["stratum_id"], row["priority_rank"] = stratum(row)
        row["critic_hard_knn_positive_overlay"] = bool(
            row["current_critic_hard"] and not row["g0_knn_negative"]
        )
        rows.append(row)
    target = [row for row in rows if row["stratum_id"] in PILOT_TARGET_STRATA]
    controls = choose_controls(rows, target, args.matched_control_count)
    for row in rows:
        if row["stratum_id"] in PILOT_TARGET_STRATA:
            role = "target"
        elif int(row["context_index"]) in controls:
            role = "matched_easy_control"
        else:
            role = "not_selected"
        row["pilot_role"] = role
        row["selected_for_probe_pilot"] = role != "not_selected"

    stratum_counts = Counter(row["stratum_id"] for row in rows)
    expected = {
        "JOINT_G0_H": 34,
        "H_ONLY": 45,
        "G0_ONLY_CRITIC_HARD": 125,
        "G0_ONLY_CRITIC_OK": 68,
        "CRITIC_ONLY": 66,
        "ALL_GOOD": 262,
    }
    if dict(stratum_counts) != expected:
        raise AssertionError(f"unexpected disjoint strata: {dict(stratum_counts)}")
    overlay_count = sum(row["critic_hard_knn_positive_overlay"] for row in rows)
    joint_hard = sum(
        row["stratum_id"] == "JOINT_G0_H" and row["current_critic_hard"]
        for row in rows
    )
    if overlay_count != 80 or joint_hard != 20:
        raise AssertionError(
            f"overlay/joint-hard mismatch: {overlay_count}/{joint_hard}"
        )
    selected = [row for row in rows if row["selected_for_probe_pilot"]]
    payload = {
        "format_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_manifest": str(args.source.resolve()),
        "source_manifest_sha256": sha256_file(args.source),
        "contract": {
            "actor_frozen": True,
            "formal_validation_loaded": False,
            "test_loaded": False,
            "new_dbm_rollouts": 0,
            "strata_are_mutually_exclusive": True,
            "pilot_target_strata": list(PILOT_TARGET_STRATA),
            "matched_control_selection": (
                "deterministic round-robin over target speed/scenario/clipping groups"
            ),
        },
        "counts": {
            "all_contexts": len(rows),
            "strata": dict(stratum_counts),
            "critic_hard_knn_positive_overlay": overlay_count,
            "joint_g0_h_current_critic_hard": joint_hard,
            "joint_g0_h_current_critic_not_hard": 34 - joint_hard,
            "pilot_target": len(target),
            "pilot_matched_easy_control": len(controls),
            "pilot_total": len(selected),
        },
        "rows": rows,
    }
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "manifest.json").write_text(json.dumps(payload, indent=2) + "\n")
    (args.output_dir / "pilot_rows.json").write_text(
        json.dumps(selected, indent=2) + "\n"
    )
    (args.output_dir / "README.md").write_text(
        "# Targeted local-probe manifest\n\n"
        "The six disjoint strata sum to 600 contexts. The first pilot contains "
        f"{len(target)} H-failure targets and {len(controls)} matched easy controls.\n"
    )
    print(json.dumps(payload["counts"], indent=2))


if __name__ == "__main__":
    main()
