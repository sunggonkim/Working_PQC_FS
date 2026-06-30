#!/usr/bin/env python3
"""
run_m5_admission_sweep.py

Drive the deterministic scheduler smoke test across supplied AI-slack values.
This is a controller unit test: it proves only that identical job state is
routed reproducibly as the supplied slack changes.  It is not an inference-QoS
measurement and must not be used as a TensorRT latency result.

Output:
  artifacts/results/qos/m5_admission_sweep.json
  artifacts/results/qos/m5_admission_sweep.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BUILD = ROOT / "build" / "pqc_fuse"


def run_smoke(ai_budget_ns: int, gpu_min_bytes: int, cpu_queue_depth: int, gpu_queue_depth: int) -> dict:
    env = os.environ.copy()
    env["PQC_SCHED_SMOKE_AI_BUDGET_NS"] = str(ai_budget_ns)
    env["PQC_GPU_MIN_BYTES"] = str(gpu_min_bytes)
    env["PQC_SCHED_SMOKE_CPU_QUEUE_DEPTH"] = str(cpu_queue_depth)
    env["PQC_SCHED_SMOKE_GPU_QUEUE_DEPTH"] = str(gpu_queue_depth)

    proc = subprocess.run(
        [str(BUILD), "--scheduler-smoke"],
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )

    jobs = []
    pressure = None
    for line in proc.stderr.splitlines():
        if "\"event\":\"scheduler_smoke_job\"" in line or "\"event\": \"scheduler_smoke_job\"" in line:
            jobs.append(json.loads(line))
        elif "\"event\":\"scheduler_smoke_pressure_job\"" in line or "\"event\": \"scheduler_smoke_pressure_job\"" in line:
            pressure = json.loads(line)

    return {
        "ai_budget_ns": ai_budget_ns,
        "gpu_min_bytes": gpu_min_bytes,
        "cpu_queue_depth": cpu_queue_depth,
        "gpu_queue_depth": gpu_queue_depth,
        "jobs": jobs,
        "pressure": pressure,
        "stderr": proc.stderr,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-json", default="artifacts/results/qos/m5_admission_sweep.json")
    parser.add_argument("--out-csv", default="artifacts/results/qos/m5_admission_sweep.csv")
    parser.add_argument("--gpu-min-bytes", type=int, default=131072)
    parser.add_argument("--cpu-queue-depths", type=int, nargs="+", default=[0, 1, 2, 4])
    parser.add_argument("--gpu-queue-depths", type=int, nargs="+", default=[0, 1, 2, 4])
    parser.add_argument(
        "--budgets-ns",
        type=int,
        nargs="+",
        default=[0, 65536, 131072, 2000000],
    )
    args = parser.parse_args()

    if not BUILD.exists():
        raise SystemExit("missing build/pqc_fuse; run cmake --build build first")

    rows = []
    for budget in args.budgets_ns:
        for cpu_q in args.cpu_queue_depths:
            for gpu_q in args.gpu_queue_depths:
                result = run_smoke(budget, args.gpu_min_bytes, cpu_q, gpu_q)
                route_summary = {
                    "budget_ns": budget,
                    "cpu_queue_depth": cpu_q,
                    "gpu_queue_depth": gpu_q,
                    "gpu_jobs": sum(1 for j in result["jobs"] if j.get("target") == "GPU"),
                    "cpu_jobs": sum(1 for j in result["jobs"] if j.get("target") == "CPU"),
                    "pressure_target": (result["pressure"] or {}).get("target"),
                    "pressure_gpu_wait_ns": (result["pressure"] or {}).get("gpu_wait_ns"),
                }
                rows.append(route_summary)

    out_json = ROOT / args.out_json
    out_csv = ROOT / args.out_csv
    out_json.parent.mkdir(parents=True, exist_ok=True)

    with out_json.open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)

    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["budget_ns", "cpu_queue_depth", "gpu_queue_depth", "gpu_jobs", "cpu_jobs", "pressure_target", "pressure_gpu_wait_ns"],
        )
        writer.writeheader()
        writer.writerows(rows)

    for row in rows:
        print(
            f"budget={row['budget_ns']:>8d} ns | gpu={row['gpu_jobs']} cpu={row['cpu_jobs']} "
            f"| pressure_target={row['pressure_target']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
