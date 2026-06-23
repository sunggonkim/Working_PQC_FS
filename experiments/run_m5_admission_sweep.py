#!/usr/bin/env python3
"""
run_m5_admission_sweep.py

Drive the scheduler smoke test across multiple AI budget settings and record
whether the admission policy causally changes route decisions.

Output:
  artifacts/m5_admission_sweep.json
  artifacts/m5_admission_sweep.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BUILD = ROOT / "build" / "pqc_fuse"


def run_smoke(ai_budget_ns: int, gpu_min_bytes: int) -> dict:
    env = os.environ.copy()
    env["PQC_AI_QOS_MIN_BUDGET_NS"] = str(ai_budget_ns)
    env["PQC_GPU_MIN_BYTES"] = str(gpu_min_bytes)

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
        "jobs": jobs,
        "pressure": pressure,
        "stderr": proc.stderr,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-json", default="artifacts/m5_admission_sweep.json")
    parser.add_argument("--out-csv", default="artifacts/m5_admission_sweep.csv")
    parser.add_argument("--gpu-min-bytes", type=int, default=131072)
    parser.add_argument(
        "--budgets-ns",
        type=int,
        nargs="+",
        default=[500000, 1000000, 2000000, 5000000],
    )
    args = parser.parse_args()

    if not BUILD.exists():
        raise SystemExit("missing build/pqc_fuse; run cmake --build build first")

    rows = []
    for budget in args.budgets_ns:
        result = run_smoke(budget, args.gpu_min_bytes)
        route_summary = {
            "budget_ns": budget,
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
            fieldnames=["budget_ns", "gpu_jobs", "cpu_jobs", "pressure_target", "pressure_gpu_wait_ns"],
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
