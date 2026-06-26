#!/usr/bin/env python3
"""Run and retain evidence for the AEGIS-Q GPU/CPU microbenchmarks.

This harness intentionally accepts results only from the built binaries.  It
does not synthesize samples, estimate missing configurations, or relabel an
unsupported executor as measured.  Each raw stdout capture is retained so a
plot can be regenerated without relying on a transcribed paper number.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import re
import statistics
import subprocess
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / "build"
DEFAULT_OUT = ROOT / "artifacts" / "validation" / "microbench"
GPU_ROW = re.compile(r"^(ml_kem_(?:keygen|encaps|decaps)),gpu,(\d+),([0-9.]+),([0-9.]+)$")


def command_output(command: list[str]) -> str:
    return subprocess.run(command, cwd=ROOT, check=True, text=True,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE).stdout


def quantile(samples: list[float], q: float) -> float:
    ordered = sorted(samples)
    index = round((len(ordered) - 1) * q)
    return ordered[index]


def platform_manifest() -> dict[str, object]:
    model_path = Path("/proc/device-tree/model")
    model = model_path.read_bytes().rstrip(b"\0").decode(errors="replace") if model_path.exists() else "unknown"
    nvcc = subprocess.run(["nvcc", "--version"], text=True, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, check=False).stdout.strip()
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "kernel": platform.platform(),
        "machine": platform.machine(),
        "device_model": model,
        "cpu_count": os.cpu_count(),
        "nvcc": nvcc,
        "commands": {
            "workload_map": [str(BUILD / "workload_map_bench")],
            "gpu_mlkem": [str(BUILD / "bench_gpu_pqc")],
            "gpu_integrity": [str(BUILD / "bench_gpu_integrity"), "--only-tests"],
        },
    }


def summarize_workload(raw_paths: list[Path]) -> dict[str, dict[str, float | int | str]]:
    values: dict[tuple[str, str, int, int, str], list[dict[str, float]]] = defaultdict(list)
    for path in raw_paths:
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if row["status"] != "measured":
                    continue
                key = (row["operation"], row["target"], int(row["batch"]),
                       int(row["bytes"]), row["status"])
                values[key].append({
                    "p50_us": float(row["p50_us"]),
                    "p99_us": float(row["p99_us"]),
                    "throughput_per_s": float(row["throughput_per_s"]),
                })
    summary: dict[str, dict[str, float | int | str]] = {}
    for key, rows in values.items():
        operation, target, batch, byte_count, status = key
        summary["|".join(map(str, key))] = {
            "operation": operation,
            "target": target,
            "batch": batch,
            "bytes": byte_count,
            "status": status,
            "runs": len(rows),
            "p50_us_median": statistics.median(row["p50_us"] for row in rows),
            "p99_us_median": statistics.median(row["p99_us"] for row in rows),
            "throughput_per_s_median": statistics.median(row["throughput_per_s"] for row in rows),
            "throughput_per_s_p05": quantile([row["throughput_per_s"] for row in rows], 0.05),
            "throughput_per_s_p95": quantile([row["throughput_per_s"] for row in rows], 0.95),
        }
    return summary


def summarize_gpu_mlkem(raw_paths: list[Path]) -> dict[str, dict[str, float | int | str]]:
    values: dict[tuple[str, int], list[tuple[float, float]]] = defaultdict(list)
    for path in raw_paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            match = GPU_ROW.match(line.strip())
            if match:
                operation, batch, ops_per_s, wall_ms = match.groups()
                values[(operation, int(batch))].append((float(ops_per_s), float(wall_ms)))
    summary: dict[str, dict[str, float | int | str]] = {}
    for (operation, batch), rows in values.items():
        throughputs = [row[0] for row in rows]
        wall_times = [row[1] for row in rows]
        summary[f"{operation}|gpu|{batch}"] = {
            "operation": operation,
            "target": "gpu",
            "batch": batch,
            "runs": len(rows),
            "throughput_per_s_median": statistics.median(throughputs),
            "throughput_per_s_p05": quantile(throughputs, 0.05),
            "throughput_per_s_p95": quantile(throughputs, 0.95),
            "wall_ms_median": statistics.median(wall_times),
        }
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    if args.runs < 2:
        raise SystemExit("--runs must be at least 2")
    for binary in ("workload_map_bench", "bench_gpu_pqc", "bench_gpu_integrity"):
        if not (BUILD / binary).is_file():
            raise SystemExit(f"missing {BUILD / binary}; build the project first")

    args.out.mkdir(parents=True, exist_ok=True)
    workload_paths: list[Path] = []
    gpu_paths: list[Path] = []
    for run in range(args.runs):
        integrity = command_output([str(BUILD / "bench_gpu_integrity"), "--only-tests"])
        if "[SUCCESS] All correctness tests passed." not in integrity:
            raise RuntimeError(f"GPU integrity correctness failed in run {run}")
        integrity_path = args.out / f"integrity_{run:02d}.txt"
        integrity_path.write_text(integrity, encoding="utf-8")

        workload_path = args.out / f"workload_map_{run:02d}.csv"
        workload_path.write_text(command_output([str(BUILD / "workload_map_bench")]), encoding="utf-8")
        workload_paths.append(workload_path)

        gpu_path = args.out / f"gpu_mlkem_{run:02d}.csv"
        gpu_path.write_text(command_output([str(BUILD / "bench_gpu_pqc")]), encoding="utf-8")
        gpu_paths.append(gpu_path)

    result = {
        "manifest": platform_manifest(),
        "runs": args.runs,
        "workload_map": summarize_workload(workload_paths),
        "gpu_mlkem": summarize_gpu_mlkem(gpu_paths),
    }
    (args.out / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"out": str(args.out), "runs": args.runs}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
