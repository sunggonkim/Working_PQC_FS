#!/usr/bin/env python3
"""
run_m5_fastlane_stress.py

Exercise the adaptive filesystem under four concurrent writers while the GPU
lane is artificially constrained, then report the fast-lane tail latency.

Output:
  artifacts/results/qos/m5_fastlane_stress.json
  artifacts/results/qos/m5_fastlane_stress.csv
"""

from __future__ import annotations

import csv
import json
import statistics
import sys
import multiprocessing
import tempfile
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ART = ROOT / "artifacts"

sys.path.insert(0, str(ROOT / "code" / "experiments"))
import run_motivation_bench as bench  # noqa: E402


def main() -> int:
    storage_dir = Path(tempfile.mkdtemp(prefix="skim_m5_store_"))
    mount_dir = Path(tempfile.mkdtemp(prefix="skim_m5_mnt_"))
    env = {
        "PQC_EXECUTION_MODE": "adaptive",
        "PQC_GPU_MAX_INFLIGHT_JOBS": "1",
        "PQC_GPU_MAX_INFLIGHT_BYTES": str(128 * 1024),
        "PQC_GPU_MIN_BYTES": "4096",
    }
    proc = bench.start_fuse(storage_dir, mount_dir, env)
    try:
        ctx = multiprocessing.get_context("spawn")
        queues = []
        writers = []
        for name in ["a", "b", "c", "d"]:
            q = ctx.Queue()
            p = ctx.Process(target=bench.writer_worker, args=(mount_dir, name, 131072, 25, q), daemon=True)
            queues.append(q)
            writers.append(p)
            p.start()
        per_writer = []
        all_samples = []
        for name, q, p in zip(["writer_a", "writer_b", "writer_c", "writer_d"], queues, writers):
            samples = q.get()
            p.join()
            per_writer.append({
                "mode": name,
                "samples_ms": samples,
                "median_ms": statistics.median(samples),
                "p95_ms": statistics.quantiles(samples, n=20, method="inclusive")[18],
                "p99_ms": statistics.quantiles(samples, n=100, method="inclusive")[98],
            })
            all_samples.extend(samples)
    finally:
        bench.stop_fuse(proc, mount_dir)
        shutil.rmtree(storage_dir, ignore_errors=True)
        shutil.rmtree(mount_dir, ignore_errors=True)

    out = [{
        "mode": "4-writer-aggregate",
        "samples_ms": all_samples,
        "median_ms": statistics.median(all_samples),
        "p95_ms": statistics.quantiles(all_samples, n=20, method="inclusive")[18],
        "p99_ms": statistics.quantiles(all_samples, n=100, method="inclusive")[98],
        "writer_summaries": per_writer,
    }]

    ART.mkdir(parents=True, exist_ok=True)
    with (ART / "m5_fastlane_stress.json").open("w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    with (ART / "m5_fastlane_stress.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["mode", "samples_ms", "median_ms", "p95_ms", "p99_ms"])
        writer.writeheader()
        for row in out:
            flat = {k: row[k] for k in ["mode", "samples_ms", "median_ms", "p95_ms", "p99_ms"]}
            writer.writerow(flat)

    for row in out:
        print(
            f"{row['mode']}: median={row.get('median_ms'):.3f} ms "
            f"p95={row.get('p95_ms'):.3f} ms p99={row.get('p99_ms'):.3f} ms "
            f"(n={len(row.get('samples_ms', []))})"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
