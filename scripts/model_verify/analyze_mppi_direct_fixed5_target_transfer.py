#!/usr/bin/env python3
"""Derive J_direct vs fixed-bank wrapper gain agreement from frozen pilot summaries.

Pure analysis: reads the three ``continuous_center_direct_fixed5_pilot_20260806_v*``
summaries and re-derives the numbers quoted in
``car_foundation/docs/mppi_sampling_center_review_20260812.md`` (section 5.3 / 7.2).
No DBM rollout, no checkpoint load, no training. Nothing is recomputed from the
dynamics model, so this cannot repair a wrong summary -- it only fixes the
provenance of the derived statistics.

Gain conventions (asymmetric on purpose, see ``caveats`` in the emitted JSON):

    direct_gain  = initial_actor_direct_cost - final_actor_direct_cost   (within run)
    wrapper_gain = baseline_wrapper_cost     - run_wrapper_cost          (vs baseline run)

Each row stores exactly one wrapper cost, evaluated at that run's final center, so
the wrapper baseline has to come from a run whose final center is still the initial
center. That is only true for ``best_iteration == 0``, which is asserted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

QUALIFICATION = "MECHANISM_PILOT_ONLY"
WRAPPER_KEY = "fixed_neighborhood_weighted_output_cost"
INITIAL_DIRECT_KEY = "initial_actor_direct_cost"
# v1 was written by the SAC pilot, v2/v3 by the deterministic actor-critic pilot.
FINAL_DIRECT_KEYS = ("deterministic_actor_direct_cost", "sac_actor_direct_cost")
CONTRACT_KEYS = ("candidate_count", "version", "contract_hash", "radii_source_sigma")

DEFAULT_ROOT = Path("outputs/mppi_proposal")
DEFAULT_RUNS = (
    "continuous_center_direct_fixed5_pilot_20260806_v1",
    "continuous_center_direct_fixed5_pilot_20260806_v2",
    "continuous_center_direct_fixed5_pilot_20260806_v3",
)
DEFAULT_OUTPUT = DEFAULT_ROOT / "direct_fixed5_target_transfer_20260812_v1"


class ContractError(RuntimeError):
    """Raised when the frozen pilot summaries are not mutually comparable."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument(
        "--runs",
        nargs="+",
        default=list(DEFAULT_RUNS),
        help="Run directory names; the first one is the wrapper baseline.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--baseline-tolerance",
        type=float,
        default=1e-6,
        help="Max allowed per-frame |initial-final| direct cost on the baseline run.",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_run(root: Path, name: str) -> dict[str, Any]:
    path = (root / name / "summary.json").resolve()
    if not path.is_file():
        raise ContractError(f"missing summary: {path}")
    summary = json.loads(path.read_text())
    rows = summary.get("rows")
    if not rows:
        raise ContractError(f"{name}: summary has no rows")

    final_key = next((k for k in FINAL_DIRECT_KEYS if k in rows[0]), None)
    if final_key is None:
        raise ContractError(
            f"{name}: no final actor direct-cost field among {FINAL_DIRECT_KEYS}"
        )

    by_episode: dict[str, dict[str, float]] = {}
    for row in rows:
        episode = row["episode"]
        if episode in by_episode:
            raise ContractError(f"{name}: duplicate episode {episode}")
        by_episode[episode] = {
            "initial_direct": float(row[INITIAL_DIRECT_KEY]),
            "final_direct": float(row[final_key]),
            "wrapper": float(row[WRAPPER_KEY]),
        }

    return {
        "name": name,
        "summary_path": str(path),
        "summary_sha256": sha256(path),
        "qualification": summary.get("qualification"),
        "method": summary.get("method"),
        "best_iteration": summary.get("best_iteration"),
        "final_direct_field": final_key,
        "contract": summary.get("secondary_center_quality", {}),
        "episodes": by_episode,
    }


def check_contracts(
    runs: list[dict[str, Any]], baseline_tolerance: float = 1e-6
) -> dict[str, Any]:
    baseline, *others = runs

    episodes = sorted(baseline["episodes"])
    for run in others:
        if sorted(run["episodes"]) != episodes:
            raise ContractError(
                f"episode set mismatch: {baseline['name']}={episodes} "
                f"vs {run['name']}={sorted(run['episodes'])}"
            )

    reference = baseline["contract"]
    missing = [key for key in CONTRACT_KEYS if key not in reference]
    if missing:
        raise ContractError(f"{baseline['name']}: contract missing {missing}")
    for run in others:
        for key in CONTRACT_KEYS:
            if run["contract"].get(key) != reference[key]:
                raise ContractError(
                    f"{key} mismatch: {baseline['name']}={reference[key]!r} "
                    f"vs {run['name']}={run['contract'].get(key)!r}"
                )

    # The wrapper column is evaluated at each run's final center, so using
    # baseline as the "initial" wrapper reference is only valid at iteration 0.
    if baseline["best_iteration"] != 0:
        raise ContractError(
            f"{baseline['name']}: wrapper baseline requires best_iteration == 0, "
            f"got {baseline['best_iteration']}"
        )

    # best_iteration == 0 only encodes the generator's intent. Assert the numbers
    # too, so a later change in summary semantics cannot silently invalidate the
    # baseline. Equal cost does not prove the centers are identical -- a knot hash
    # in the summary would be needed for that.
    baseline_error = max(
        abs(frame["initial_direct"] - frame["final_direct"])
        for frame in baseline["episodes"].values()
    )
    if baseline_error > baseline_tolerance:
        raise ContractError(
            f"{baseline['name']}: baseline initial vs final direct cost differ by "
            f"{baseline_error:.6g} > {baseline_tolerance:.6g}; its final center is "
            "not the initial center, so it cannot be the wrapper baseline"
        )

    for run in runs:
        if run["qualification"] != QUALIFICATION:
            raise ContractError(
                f"{run['name']}: qualification {run['qualification']!r} "
                f"!= {QUALIFICATION!r}"
            )

    return {
        "episodes": episodes,
        "episode_count": len(episodes),
        "shared_center_quality_contract": {key: reference[key] for key in CONTRACT_KEYS},
        "baseline_initial_final_max_abs_error": baseline_error,
        "baseline_initial_final_tolerance": baseline_tolerance,
        "baseline_center_identity_proven": False,
    }


def pearson(xs: list[float], ys: list[float]) -> float | None:
    count = len(xs)
    if count < 2:
        return None
    mean_x = sum(xs) / count
    mean_y = sum(ys) / count
    dev_x = sum((x - mean_x) ** 2 for x in xs) ** 0.5
    dev_y = sum((y - mean_y) ** 2 for y in ys) ** 0.5
    if dev_x == 0.0 or dev_y == 0.0:
        return None
    covariance = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    return covariance / (dev_x * dev_y)


def compare(baseline: dict[str, Any], run: dict[str, Any], episodes: list[str]) -> dict:
    per_episode = []
    for episode in episodes:
        current = run["episodes"][episode]
        direct_gain = current["initial_direct"] - current["final_direct"]
        wrapper_gain = baseline["episodes"][episode]["wrapper"] - current["wrapper"]
        per_episode.append(
            {
                "episode": episode,
                "initial_actor_direct_cost": current["initial_direct"],
                "final_actor_direct_cost": current["final_direct"],
                "direct_gain": direct_gain,
                "baseline_wrapper_cost": baseline["episodes"][episode]["wrapper"],
                "run_wrapper_cost": current["wrapper"],
                "wrapper_gain": wrapper_gain,
                "sign_agrees": (direct_gain > 0.0) == (wrapper_gain > 0.0),
            }
        )

    direct_gains = [row["direct_gain"] for row in per_episode]
    wrapper_gains = [row["wrapper_gain"] for row in per_episode]
    count = len(per_episode)
    contradictions = [
        row["episode"]
        for row in per_episode
        if row["direct_gain"] > 0.0 and row["wrapper_gain"] < 0.0
    ]
    # The opposite mismatch is a different phenomenon: the direct objective got
    # worse while the deployable wrapper metric improved. Reported separately so
    # "mismatch frames" is never read as only the contradiction set.
    inverse_mismatches = [
        row["episode"]
        for row in per_episode
        if row["direct_gain"] < 0.0 and row["wrapper_gain"] > 0.0
    ]

    return {
        "run": run["name"],
        "baseline": baseline["name"],
        "final_direct_field": run["final_direct_field"],
        "best_iteration": run["best_iteration"],
        "episode_count": count,
        "mean_direct_gain": sum(direct_gains) / count,
        "mean_wrapper_gain": sum(wrapper_gains) / count,
        "direct_wins": sum(1 for value in direct_gains if value > 0.0),
        "wrapper_wins": sum(1 for value in wrapper_gains if value > 0.0),
        "sign_agreement": sum(1 for row in per_episode if row["sign_agrees"]),
        "pearson_direct_vs_wrapper": pearson(direct_gains, wrapper_gains),
        "direct_improved_but_wrapper_regressed": contradictions,
        "wrapper_improved_but_direct_regressed": inverse_mismatches,
        "sign_mismatch_episodes": sorted(contradictions + inverse_mismatches),
        "per_episode": per_episode,
    }


def main() -> None:
    args = parse_args()
    if len(args.runs) < 2:
        raise ContractError("need a baseline run plus at least one comparison run")

    runs = [load_run(args.root, name) for name in args.runs]
    contract = check_contracts(runs, args.baseline_tolerance)
    baseline, *others = runs
    comparisons = [compare(baseline, run, contract["episodes"]) for run in others]

    report = {
        "format_version": 1,
        "analysis": "J_direct vs fixed-bank wrapper gain agreement on frozen pilots",
        "qualification": QUALIFICATION,
        "analysis_script": str(Path(__file__).resolve()),
        "analysis_script_sha256": sha256(Path(__file__).resolve()),
        "statistical_status": (
            f"n={contract['episode_count']} frames; diagnostic only. Pearson and win "
            "rates must not be reported as formal statistical claims."
        ),
        "recomputes_dynamics": False,
        "caveats": [
            "direct_gain is within-run (initial vs final center of the same run); "
            "wrapper_gain is cross-run against the baseline run. The two gains do "
            "not share a reference, so their correlation is directional evidence only.",
            "Each row holds one wrapper cost at that run's final center, so the "
            "baseline run must have best_iteration == 0 to stand in for the initial "
            "center. This is asserted, not assumed.",
            "Frozen-state single-step DBM rollout. This says nothing about closed-loop "
            "behaviour and does not substitute for the full same-budget wrapper check.",
            "Matching baseline initial/final direct cost does not prove the centers are "
            "identical; proving that needs initial/final knots or their hash in the "
            "summary. See baseline_center_identity_proven.",
            "Sign mismatches come in two kinds. Only "
            "direct_improved_but_wrapper_regressed is the target-transfer "
            "contradiction; wrapper_improved_but_direct_regressed is the opposite case "
            "and must not be merged into it.",
        ],
        "inputs": [
            {
                "run": run["name"],
                "summary_path": run["summary_path"],
                "summary_sha256": run["summary_sha256"],
                "method": run["method"],
                "qualification": run["qualification"],
                "best_iteration": run["best_iteration"],
                "final_direct_field": run["final_direct_field"],
            }
            for run in runs
        ],
        "contract": contract,
        "comparisons": comparisons,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "analysis.json"
    output_path.write_text(json.dumps(report, indent=2) + "\n")

    print(f"episodes: {contract['episode_count']} {contract['episodes']}")
    print(f"contract: {contract['shared_center_quality_contract']}")
    print(
        "baseline initial-vs-final max abs error: "
        f"{contract['baseline_initial_final_max_abs_error']:.3g} "
        f"(tol {contract['baseline_initial_final_tolerance']:.3g}, "
        "center identity not proven)"
    )
    for entry in comparisons:
        pearson_value = entry["pearson_direct_vs_wrapper"]
        pearson_text = "n/a" if pearson_value is None else f"{pearson_value:+.3f}"
        print(
            f"{entry['run']}: direct {entry['mean_direct_gain']:+.4f} "
            f"({entry['direct_wins']}/{entry['episode_count']}) "
            f"wrapper {entry['mean_wrapper_gain']:+.6f} "
            f"({entry['wrapper_wins']}/{entry['episode_count']}) "
            f"pearson {pearson_text}"
        )
        print(
            f"  direct+ wrapper- {entry['direct_improved_but_wrapper_regressed']}"
            f"  direct- wrapper+ {entry['wrapper_improved_but_direct_regressed']}"
            f"  sign agreement {entry['sign_agreement']}/{entry['episode_count']}"
        )
    print(f"wrote {output_path}")


if __name__ == "__main__":
    main()
