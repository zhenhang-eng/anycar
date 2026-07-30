#!/usr/bin/env python3
"""Sweep dynamic ONNX batch sizes and estimate the GPU saturation range."""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(SCRIPT_DIR))

from benchmark_onnx_batching import (  # noqa: E402
    DEFAULT_CHECKPOINT,
    DEFAULT_OUTPUT_DIR,
    create_session,
    make_inputs,
    run,
)


DEFAULT_BATCHES = "1,2,4,8,16,32,64,96,128,192,256,384,512,768,1024,1536,2048,3072,4096"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Measure ONNX latency, throughput, utilization and memory by batch size."
    )
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument(
        "--onnx-path",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / "anycar_query_full_dynamic.onnx",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--provider", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--batches", default=DEFAULT_BATCHES)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument(
        "--timing-target-seconds",
        type=float,
        default=1.0,
        help="Approximate measured wall time per batch size.",
    )
    parser.add_argument("--min-iters", type=int, default=8)
    parser.add_argument("--max-iters", type=int, default=100)
    parser.add_argument("--monitor-seconds", type=float, default=1.5)
    parser.add_argument("--seed", type=int, default=3407)
    return parser.parse_args()


def parse_batches(value):
    batches = sorted({int(item.strip()) for item in value.split(",")})
    if not batches or batches[0] <= 0:
        raise ValueError("Batch sizes must be positive integers")
    return batches


def percentile_summary(times_ms):
    values = np.asarray(times_ms, dtype=np.float64)
    return {
        "mean_ms": float(values.mean()),
        "std_ms": float(values.std()),
        "p50_ms": float(np.percentile(values, 50)),
        "p90_ms": float(np.percentile(values, 90)),
        "p99_ms": float(np.percentile(values, 99)),
        "min_ms": float(values.min()),
        "max_ms": float(values.max()),
    }


def query_process_gpu_memory_mib():
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,used_gpu_memory",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    current_pid = os.getpid()
    for line in output.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) >= 2 and fields[0] == str(current_pid):
            try:
                return float(fields[1])
            except ValueError:
                return None
    return None


def monitor_gpu_while_running(fn, duration_seconds):
    command = [
        "nvidia-smi",
        "--query-gpu=utilization.gpu,memory.used,power.draw",
        "--format=csv,noheader,nounits",
        "-lms",
        "100",
    ]
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        bufsize=1,
    )
    samples = []
    collecting = threading.Event()
    collecting.set()

    def reader():
        assert process.stdout is not None
        for line in process.stdout:
            if not collecting.is_set():
                break
            fields = [field.strip() for field in line.split(",")]
            if len(fields) != 3:
                continue
            try:
                samples.append(tuple(float(field) for field in fields))
            except ValueError:
                continue

    thread = threading.Thread(target=reader, daemon=True)
    thread.start()
    calls = 0
    start = time.perf_counter()
    while time.perf_counter() - start < duration_seconds:
        fn()
        calls += 1
    collecting.clear()
    process.terminate()
    try:
        process.wait(timeout=2)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=2)
    thread.join(timeout=2)

    # Drop boundary samples when enough values are available.
    selected = samples[1:-1] if len(samples) >= 5 else samples
    if not selected:
        return {
            "monitor_calls": calls,
            "gpu_util_mean_pct": None,
            "gpu_util_p90_pct": None,
            "gpu_util_max_pct": None,
            "gpu_memory_max_mib": None,
            "gpu_power_mean_w": None,
            "gpu_power_max_w": None,
            "monitor_samples": 0,
        }
    values = np.asarray(selected, dtype=np.float64)
    return {
        "monitor_calls": calls,
        "gpu_util_mean_pct": float(values[:, 0].mean()),
        "gpu_util_p90_pct": float(np.percentile(values[:, 0], 90)),
        "gpu_util_max_pct": float(values[:, 0].max()),
        "gpu_memory_max_mib": float(values[:, 1].max()),
        "gpu_power_mean_w": float(values[:, 2].mean()),
        "gpu_power_max_w": float(values[:, 2].max()),
        "monitor_samples": int(values.shape[0]),
    }


def measure_batch(session, checkpoint, batch_size, args):
    history, action = make_inputs(checkpoint, batch_size, args.seed + batch_size)
    fn = lambda: run(session, history, action)
    for _ in range(args.warmup):
        fn()

    start = time.perf_counter_ns()
    fn()
    probe_ms = (time.perf_counter_ns() - start) / 1e6
    iterations = int(
        np.clip(
            math.ceil(args.timing_target_seconds * 1000.0 / max(probe_ms, 0.01)),
            args.min_iters,
            args.max_iters,
        )
    )
    times_ms = []
    for _ in range(iterations):
        start = time.perf_counter_ns()
        fn()
        times_ms.append((time.perf_counter_ns() - start) / 1e6)

    timing = percentile_summary(times_ms)
    timing.update(
        {
            "batch_size": batch_size,
            "iterations": iterations,
            "per_sample_mean_ms": timing["mean_ms"] / batch_size,
            "throughput_samples_per_s": batch_size * 1000.0 / timing["mean_ms"],
        }
    )
    if args.provider == "cuda" and args.monitor_seconds > 0:
        timing.update(monitor_gpu_while_running(fn, args.monitor_seconds))
        timing["process_gpu_memory_after_mib"] = query_process_gpu_memory_mib()
    return timing


def saturation_summary(results):
    successful = [item for item in results if "error" not in item]
    peak = max(successful, key=lambda item: item["throughput_samples_per_s"])
    peak_throughput = peak["throughput_samples_per_s"]

    def threshold(level):
        selected = [
            item
            for item in successful
            if item["throughput_samples_per_s"] >= level * peak_throughput
        ]
        return {
            "threshold_samples_per_s": level * peak_throughput,
            "first_batch": selected[0]["batch_size"] if selected else None,
            "batches": [item["batch_size"] for item in selected],
        }

    return {
        "peak_batch": peak["batch_size"],
        "peak_throughput_samples_per_s": peak_throughput,
        "peak_latency_ms": peak["mean_ms"],
        "at_least_90pct_peak": threshold(0.90),
        "at_least_95pct_peak": threshold(0.95),
    }


def plot_results(results, saturation, output_path):
    successful = [item for item in results if "error" not in item]
    batch = np.asarray([item["batch_size"] for item in successful])
    latency = np.asarray([item["mean_ms"] for item in successful])
    throughput = np.asarray(
        [item["throughput_samples_per_s"] for item in successful]
    )
    utilization = np.asarray(
        [
            np.nan
            if item.get("gpu_util_mean_pct") is None
            else item["gpu_util_mean_pct"]
            for item in successful
        ]
    )

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ax = axes[0]
    ax.plot(batch, throughput, "o-", color="#1565c0", label="Throughput")
    ax.axhline(
        saturation["peak_throughput_samples_per_s"] * 0.90,
        color="#1565c0",
        linestyle="--",
        alpha=0.5,
        label="90% peak throughput",
    )
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Batch size")
    ax.set_ylabel("Samples/s", color="#1565c0")
    ax.tick_params(axis="y", labelcolor="#1565c0")
    ax.grid(True, alpha=0.25)
    util_ax = ax.twinx()
    util_ax.plot(batch, utilization, "s-", color="#ef6c00", label="GPU util")
    util_ax.set_ylabel("Mean GPU utilization (%)", color="#ef6c00")
    util_ax.tick_params(axis="y", labelcolor="#ef6c00")
    ax.set_title("Throughput and GPU utilization")

    ax = axes[1]
    ax.plot(batch, latency, "o-", color="#6a1b9a", label="Batch latency")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("Batch size")
    ax.set_ylabel("Mean latency (ms)")
    ax.grid(True, which="both", alpha=0.25)
    ax.set_title("Latency after increasing batch size")
    for x, y in zip(batch, latency):
        if x in (32, 128, 512, 1024, 2048, 4096):
            ax.annotate(f"{y:.1f}", (x, y), xytext=(3, 4), textcoords="offset points")

    fig.suptitle("AnyCar full ONNX batch-size sweep (CUDAExecutionProvider)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main():
    args = parse_args()
    batches = parse_batches(args.batches)
    checkpoint = torch.load(args.checkpoint.resolve(), map_location="cpu")
    onnx_path = args.onnx_path.resolve()
    if not onnx_path.is_file():
        raise FileNotFoundError(onnx_path)
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    session = create_session(onnx_path, args.provider)

    results = []
    print(f"ONNX: {onnx_path}")
    print(f"providers: {session.get_providers()}")
    print(f"batches: {batches}")
    for batch_size in batches:
        try:
            result = measure_batch(session, checkpoint, batch_size, args)
        except Exception as error:
            result = {"batch_size": batch_size, "error": repr(error)}
            results.append(result)
            print(f"batch={batch_size}: ERROR {error}")
            break
        results.append(result)
        print(
            f"batch={batch_size:<4} "
            f"mean={result['mean_ms']:8.3f} ms  "
            f"throughput={result['throughput_samples_per_s']:9.1f}/s  "
            f"util={result.get('gpu_util_mean_pct')}%  "
            f"process_mem={result.get('process_gpu_memory_after_mib')} MiB"
        )

    saturation = saturation_summary(results)
    summary = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "checkpoint": str(args.checkpoint.resolve()),
        "onnx_path": str(onnx_path),
        "provider_requested": args.provider,
        "providers_active": session.get_providers(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "timing_scope": "ORT session.run including host/device copies",
        "monitor_note": (
            "nvidia-smi utilization includes desktop graphics activity; process "
            "memory is the ONNX process high-water allocation in an ascending sweep"
        ),
        "saturation": saturation,
        "results": results,
    }
    summary_path = args.output_dir / "onnx_batch_size_sweep.json"
    figure_path = args.output_dir / "onnx_batch_size_sweep.png"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    plot_results(results, saturation, figure_path)

    print("\nSaturation summary")
    print(json.dumps(saturation, indent=2))
    print(f"summary: {summary_path}")
    print(f"figure:  {figure_path}")


if __name__ == "__main__":
    main()
