#!/usr/bin/env python3
"""Independently validate the disjoint targeted-probe routing manifest."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def expected_stratum(row: dict) -> str:
    g0 = bool(row["g0_knn_negative"])
    h = bool(row["h_oracle_cross_negative"])
    critic = bool(row["current_critic_hard"])
    if g0 and h:
        return "JOINT_G0_H"
    if h:
        return "H_ONLY"
    if g0 and critic:
        return "G0_ONLY_CRITIC_HARD"
    if g0:
        return "G0_ONLY_CRITIC_OK"
    if critic:
        return "CRITIC_ONLY"
    return "ALL_GOOD"


def main() -> None:
    args = parse_args()
    manifest_path = args.output_dir / "manifest.json"
    pilot_path = args.output_dir / "pilot_rows.json"
    manifest = json.loads(manifest_path.read_text())
    rows = manifest["rows"]
    source = Path(manifest["source_manifest"])
    errors = []
    if sha256_file(source) != manifest["source_manifest_sha256"]:
        errors.append("source hash mismatch")
    if len(rows) != 600 or len({row["context_index"] for row in rows}) != 600:
        errors.append("context count/uniqueness mismatch")
    for row in rows:
        if row["stratum_id"] != expected_stratum(row):
            errors.append(f"stratum mismatch at {row['context_index']}")
    counts = Counter(row["stratum_id"] for row in rows)
    if dict(counts) != manifest["counts"]["strata"]:
        errors.append("stored stratum counts mismatch")
    if sum(counts.values()) != 600:
        errors.append("strata are not exhaustive")
    overlay = sum(
        row["current_critic_hard"] and not row["g0_knn_negative"] for row in rows
    )
    joint_hard = sum(
        row["stratum_id"] == "JOINT_G0_H" and row["current_critic_hard"]
        for row in rows
    )
    selected = [row for row in rows if row["selected_for_probe_pilot"]]
    pilot = json.loads(pilot_path.read_text())
    if overlay != 80 or joint_hard != 20:
        errors.append("overlay or joint-hard count mismatch")
    if len(selected) != 100 or len(pilot) != 100:
        errors.append("pilot count mismatch")
    if [row["context_index"] for row in selected] != [
        row["context_index"] for row in pilot
    ]:
        errors.append("pilot row mismatch")
    contract = manifest["contract"]
    if (
        not contract["actor_frozen"]
        or contract["formal_validation_loaded"]
        or contract["test_loaded"]
        or contract["new_dbm_rollouts"] != 0
    ):
        errors.append("contract mismatch")
    result = {
        "format_version": 1,
        "qualification": "PASS" if not errors else "FAIL",
        "manifest_sha256": sha256_file(manifest_path),
        "pilot_rows_sha256": sha256_file(pilot_path),
        "errors": errors,
        "stratum_counts": dict(counts),
        "overlay_count": overlay,
        "joint_hard_count": joint_hard,
        "pilot_count": len(selected),
    }
    (args.output_dir / "validation_summary.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print(json.dumps(result, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
