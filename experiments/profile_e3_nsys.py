#!/usr/bin/env python3
"""Separate M6 profiling pass for E3 using Nsight Systems or CUPTI-friendly traces.

This script is intentionally distinct from the latency pass.  It does not
measure wall-clock p99 itself; instead it wraps the same TensorRT + secure-I/O
mode under a profiler so the caller can collect SM occupancy, GPU utilization,
DRAM traffic, UVM faults/migrations/stalls, and staging counters without
mixing profiling overhead into the latency measurements.

Typical usage:

  python3 experiments/profile_e3_nsys.py \
    --engine artifacts/yolov8n_fp16.plan \
    --model-name yolov8 \
    --out-dir artifacts/m6_profile_yolov8

  python3 experiments/profile_e3_nsys.py \
    --engine artifacts/<memory-heavy-engine>.plan \
    --model-name memory_heavy \
    --out-dir artifacts/m6_profile_memory_heavy

If `nsys` is available, the script runs `nsys profile` around the latency pass.
Otherwise it prints the exact profiling command so the user can run it manually.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
LATENCY_SCRIPT = ROOT / "experiments" / "benchmark_tensorrt_interference.py"


def main() -> int:
    ap = argparse.ArgumentParser(description="E3/M6 Nsight Systems profiling wrapper")
    ap.add_argument("--engine", required=True, help="TensorRT engine path")
    ap.add_argument("--model-name", required=True, help="Label for the profiled model")
    ap.add_argument("--out-dir", required=True, help="Directory for profiler output artifacts")
    ap.add_argument("--duration", type=int, default=5, help="Latency-pass duration in seconds")
    ap.add_argument("--latency-only", action="store_true", help="Do not invoke nsys; only print the command")
    args = ap.parse_args()

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    latency_cmd = [
        "python3",
        str(LATENCY_SCRIPT),
        str(out_dir / "latency_pass"),
        "--engine",
        str(Path(args.engine).resolve()),
        "--model-name",
        args.model_name,
        "--duration",
        str(args.duration),
    ]
    nsys = shutil.which("nsys")
    profile_cmd = None
    if nsys:
        profile_cmd = [
            nsys,
            "profile",
            "--stats=true",
            "--force-overwrite=true",
            "--trace=cuda,nvtx,osrt",
            "--output",
            str(out_dir / f"{args.model_name}_nsys"),
            *latency_cmd,
        ]

    manifest = {
        "model_name": args.model_name,
        "engine": str(Path(args.engine).resolve()),
        "out_dir": str(out_dir),
        "latency_cmd": latency_cmd,
        "nsys_available": bool(nsys),
        "profile_cmd": profile_cmd,
    }
    (out_dir / "profile_manifest.json").write_text(json.dumps(manifest, indent=2))

    print(json.dumps(manifest, indent=2))
    if profile_cmd is None:
        print("\nnsys not found. Run this manually if you want a profiling pass:\n")
        print(" ".join(latency_cmd))
        return 0

    print("\nRunning profiling command:\n")
    print(" ".join(profile_cmd))
    result = subprocess.run(profile_cmd, cwd=ROOT)
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
