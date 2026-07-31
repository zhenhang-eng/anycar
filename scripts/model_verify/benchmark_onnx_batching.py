#!/usr/bin/env python3
"""Export the current AnyCar Query model and compare batching strategies.

Every configured strategy must process the same total number of candidates.
For example, ``100x10,1000x1,10x100`` means batch size x call count and
processes 1000 candidates with each strategy.

The exported graph includes history normalization, the nominal kinematic
rollout, the deterministic Query residual model, residual de-normalization,
and the physically consistent final rollout.  Runtime inputs follow the
training protocol exactly: raw transition history, absolute current state,
current action, and candidate future actions.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
for package_dir in ("car_foundation", "car_planner", "car_dynamics", "car_dataset"):
    sys.path.insert(0, str(REPO_ROOT / package_dir))
sys.path.insert(0, str(REPO_ROOT))

from car_foundation.query_deployment import (  # noqa: E402
    QueryDeploymentModel,
    export_query_onnx,
)


DEFAULT_CHECKPOINT = REPO_ROOT / (
    "outputs/formal_real_finetune_query_baseline_split/"
    "20260728T143256/query_best.pt"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs/onnx_batch_benchmark"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare equivalent ONNX batch-size x call-count strategies."
    )
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--onnx-path", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--provider", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--warmup-rounds", type=int, default=10)
    parser.add_argument("--measurement-rounds", type=int, default=100)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument(
        "--strategies",
        default="10x10,100x1",
        help=(
            "Comma-separated batch-size x call-count pairs. Every pair must "
            "process the same total candidate count."
        ),
    )
    parser.add_argument(
        "--skip-export",
        action="store_true",
        help="Reuse --onnx-path instead of exporting the checkpoint.",
    )
    return parser.parse_args()


def make_inputs(checkpoint: dict, batch_size: int, seed: int):
    args = checkpoint["args"]
    generator = np.random.default_rng(seed)

    # MPPI-style benchmark: all action candidates share one current history.
    history_one = np.zeros(
        (1, args["history_length"], 7), dtype=np.float32
    )
    history_one[:, :, 0] = 16.0 * 0.05
    history = history_one
    initial_state = np.array([[0.0, 0.0, 0.0, 16.0, 0.0]], dtype=np.float32)
    current_action = np.array([[0.0, 0.0]], dtype=np.float32)

    action = np.empty(
        (batch_size, args["prediction_length"], 2), dtype=np.float32
    )
    action[:, :, 0] = generator.normal(
        loc=0.0, scale=0.25, size=action[:, :, 0].shape
    )
    action[:, :, 1] = generator.normal(
        loc=0.0, scale=0.35, size=action[:, :, 1].shape
    )
    return (
        np.ascontiguousarray(history),
        initial_state,
        current_action,
        np.ascontiguousarray(action),
    )


def export_onnx(wrapper, checkpoint, onnx_path: Path, opset: int, seed: int):
    del checkpoint, seed
    export_query_onnx(wrapper, onnx_path, opset=opset, candidate_batch=2)


def create_session(onnx_path: Path, provider: str):
    options = ort.SessionOptions()
    options.log_severity_level = 3
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    providers = (
        ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if provider == "cuda"
        else ["CPUExecutionProvider"]
    )
    session = ort.InferenceSession(
        str(onnx_path), sess_options=options, providers=providers
    )
    if provider == "cuda" and "CUDAExecutionProvider" not in session.get_providers():
        raise RuntimeError(
            "CUDAExecutionProvider was requested but was not activated: "
            f"{session.get_providers()}"
        )
    return session


def run(session, history, initial_state, current_action, action):
    return session.run(
        None,
        {
            "history": history,
            "initial_state": initial_state,
            "current_action": current_action,
            "future_action": action,
        },
    )[0]


def parse_strategies(value):
    strategies = []
    for item in value.split(","):
        match = re.fullmatch(r"\s*(\d+)\s*[xX*]\s*(\d+)\s*", item)
        if match is None:
            raise ValueError(
                f"Invalid strategy {item!r}; expected batch_size x call_count"
            )
        batch_size, call_count = map(int, match.groups())
        if batch_size <= 0 or call_count <= 0:
            raise ValueError("Batch size and call count must be positive")
        strategies.append(
            {
                "batch_size": batch_size,
                "call_count": call_count,
                "total_candidates": batch_size * call_count,
                "name": f"batch{batch_size}_calls{call_count}",
            }
        )
    if not strategies:
        raise ValueError("At least one strategy is required")
    totals = {item["total_candidates"] for item in strategies}
    if len(totals) != 1:
        raise ValueError(
            "Every strategy must process the same total candidates, got "
            f"{sorted(totals)}"
        )
    return strategies, totals.pop()


def timed_strategy(
    session, history, initial_state, current_action, action, batch_size, call_count
):
    start = time.perf_counter_ns()
    outputs = []
    for call_index in range(call_count):
        begin = call_index * batch_size
        end = begin + batch_size
        outputs.append(
            run(
                session,
                history,
                initial_state,
                current_action,
                action[begin:end],
            )
        )
    elapsed_ms = (time.perf_counter_ns() - start) / 1e6
    return elapsed_ms, outputs


def distribution(values, total_candidates):
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean_ms": float(array.mean()),
        "std_ms": float(array.std()),
        "p50_ms": float(np.percentile(array, 50)),
        "p90_ms": float(np.percentile(array, 90)),
        "p99_ms": float(np.percentile(array, 99)),
        "min_ms": float(array.min()),
        "max_ms": float(array.max()),
        "per_sample_mean_ms": float(array.mean() / total_candidates),
        "throughput_samples_per_s": float(
            total_candidates * 1000.0 / array.mean()
        ),
    }


def main():
    args = parse_args()
    strategies, total_candidates = parse_strategies(args.strategies)
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.checkpoint.resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if checkpoint.get("variant") != "query":
        raise ValueError(
            f"Expected a Query checkpoint, got variant={checkpoint.get('variant')!r}"
        )

    onnx_path = (
        args.onnx_path.resolve()
        if args.onnx_path is not None
        else args.output_dir / "anycar_query_full_dynamic.onnx"
    )
    export_device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    wrapper = QueryDeploymentModel.from_checkpoint(
        checkpoint_path, export_device
    )
    if not args.skip_export:
        print(f"Exporting dynamic-batch ONNX: {onnx_path}")
        export_onnx(wrapper, checkpoint, onnx_path, args.opset, args.seed)
    elif not onnx_path.is_file():
        raise FileNotFoundError(onnx_path)

    session = create_session(onnx_path, args.provider)
    history, initial_state, current_action, action = make_inputs(
        checkpoint, batch_size=total_candidates, seed=args.seed
    )

    # A bounded numerical check avoids making validation dominate large tests.
    validation_batch_size = min(total_candidates, 100)
    with torch.no_grad():
        torch_output = wrapper(
            torch.from_numpy(history).to(export_device),
            torch.from_numpy(initial_state).to(export_device),
            torch.from_numpy(current_action).to(export_device),
            torch.from_numpy(action[:validation_batch_size]).to(export_device),
        ).cpu().numpy()
    onnx_output = run(
        session, history, initial_state, current_action, action[:validation_batch_size]
    )
    abs_diff = np.abs(torch_output - onnx_output)

    for _ in range(args.warmup_rounds):
        for strategy in strategies:
            timed_strategy(
                session,
                history,
                initial_state,
                current_action,
                action,
                strategy["batch_size"],
                strategy["call_count"],
            )

    times = {strategy["name"]: [] for strategy in strategies}
    last_outputs = {}
    rng = random.Random(args.seed)
    for _ in range(args.measurement_rounds):
        round_strategies = strategies.copy()
        rng.shuffle(round_strategies)
        for strategy in round_strategies:
            elapsed, outputs = timed_strategy(
                session,
                history,
                initial_state,
                current_action,
                action,
                strategy["batch_size"],
                strategy["call_count"],
            )
            times[strategy["name"]].append(elapsed)
            last_outputs[strategy["name"]] = np.concatenate(outputs, axis=0)

    summaries = {
        strategy["name"]: {
            **strategy,
            **distribution(times[strategy["name"]], total_candidates),
        }
        for strategy in strategies
    }
    fastest_mean = min(item["mean_ms"] for item in summaries.values())
    for item in summaries.values():
        item["relative_to_fastest"] = item["mean_ms"] / fastest_mean

    reference_strategy = max(strategies, key=lambda item: item["batch_size"])
    reference_output = last_outputs[reference_strategy["name"]]
    batching_diffs = {}
    for strategy in strategies:
        diff = np.abs(last_outputs[strategy["name"]] - reference_output)
        batching_diffs[strategy["name"]] = {
            "reference": reference_strategy["name"],
            "max_abs": float(diff.max()),
            "mean_abs": float(diff.mean()),
        }

    result = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "checkpoint": str(checkpoint_path),
        "onnx_path": str(onnx_path),
        "provider_requested": args.provider,
        "providers_active": session.get_providers(),
        "versions": {
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "onnx": onnx.__version__,
            "onnxruntime": ort.__version__,
            "gpu": torch.cuda.get_device_name(0),
        },
        "warmup_rounds": args.warmup_rounds,
        "measurement_rounds": args.measurement_rounds,
        "total_candidates_per_round": total_candidates,
        "strategies": summaries,
        "pytorch_onnx_max_abs": float(abs_diff.max()),
        "pytorch_onnx_mean_abs": float(abs_diff.mean()),
        "strategy_output_differences": batching_diffs,
        "timing_scope": "ORT session.run including host/device input and output copies",
        "workload": (
            f"{total_candidates} action candidates sharing one history "
            "(MPPI style)"
        ),
    }
    summary_path = (
        args.output_dir / f"batching_benchmark_{total_candidates}.json"
    )
    summary_path.write_text(json.dumps(result, indent=2) + "\n")

    print("\nONNX batching benchmark")
    print(f"checkpoint: {checkpoint_path}")
    print(f"onnx:       {onnx_path}")
    print(f"providers:  {session.get_providers()}")
    print(f"rounds:     {args.measurement_rounds} (+ {args.warmup_rounds} warmup)")
    print(f"candidates: {total_candidates} per strategy")
    print()
    for strategy in strategies:
        item = summaries[strategy["name"]]
        print(
            f"batch={item['batch_size']:<4} x calls={item['call_count']:<4} "
            f"mean={item['mean_ms']:8.3f} ms  "
            f"p50={item['p50_ms']:8.3f} ms  "
            f"p90={item['p90_ms']:8.3f} ms  "
            f"{item['throughput_samples_per_s']:8.1f} samples/s  "
            f"{item['relative_to_fastest']:.3f}x fastest"
        )
    print(
        "PyTorch vs ONNX: "
        f"max_abs={abs_diff.max():.6e}, mean_abs={abs_diff.mean():.6e}"
    )
    for strategy in strategies:
        diff = batching_diffs[strategy["name"]]
        print(
            f"{strategy['name']} vs {diff['reference']}: "
            f"max_abs={diff['max_abs']:.6e}, "
            f"mean_abs={diff['mean_abs']:.6e}"
        )
    print(f"summary: {summary_path}")


if __name__ == "__main__":
    main()
