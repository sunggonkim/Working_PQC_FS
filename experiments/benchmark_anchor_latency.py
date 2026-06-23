#!/usr/bin/env python3
"""Benchmark freshness-anchor round trips.

The script measures the anchor store/load path exposed by pqc_fuse's self-test
interface. It supports both file-backed and hardware-backed backends through the
same environment variables the runtime already understands.
"""

from __future__ import annotations

import csv
import json
import os
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
BUILD = ROOT / "build" / "pqc_fuse"
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "artifacts" / "motivation"
OUT.mkdir(parents=True, exist_ok=True)


def run_once(env: dict[str, str]) -> tuple[float, str]:
    started = time.perf_counter_ns()
    proc = subprocess.run(
        [str(BUILD), "--anchor-self-test"],
        cwd=ROOT,
        env={**os.environ, **env},
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=True,
    )
    elapsed_ms = (time.perf_counter_ns() - started) / 1e6
    return elapsed_ms, proc.stdout


def bench_backend(name: str, env: dict[str, str], samples: int = 10) -> dict[str, object]:
    rows = []
    for _ in range(samples):
        try:
            elapsed, out = run_once(env)
            rows.append({"elapsed_ms": elapsed, "output": out})
        except subprocess.CalledProcessError as exc:
            return {
                "backend": name,
                "error": f"anchor-self-test failed: {exc.returncode}",
                "stdout": exc.stdout,
            }
    latencies = [row["elapsed_ms"] for row in rows]
    payload = {
        "backend": name,
        "samples_ms": latencies,
        "median_ms": statistics.median(latencies),
        "p95_ms": statistics.quantiles(latencies, n=20, method="inclusive")[18] if len(latencies) > 1 else latencies[0],
        "min_ms": min(latencies),
        "max_ms": max(latencies),
    }
    return payload


def main() -> int:
    results = []
    skipped = []
    file_anchor = OUT / "anchor_witness.bin"
    results.append(bench_backend(
        "file",
        {
            "PQC_FRESHNESS_ANCHOR_PATH": str(file_anchor),
            "PQC_FRESHNESS_ANCHOR_BACKEND": "file",
        },
    ))

    if os.environ.get("PQC_ENABLE_HARDWARE_ANCHOR") == "1":
        hw = bench_backend(
            "hardware",
            {
                "PQC_FRESHNESS_ANCHOR_PATH": str(file_anchor),
                "PQC_FRESHNESS_ANCHOR_BACKEND": "hardware",
                "PQC_TPM_TCTI": os.environ.get("PQC_TPM_TCTI", "device:/dev/tpmrm0"),
            },
        )
        if "error" in hw:
            skipped.append(hw)
        else:
            results.append(hw)

    json_path = OUT / "anchor_latency.json"
    csv_path = OUT / "anchor_latency.csv"
    json_path.write_text(json.dumps({"rows": results, "skipped": skipped}, indent=2))
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["backend", "median_ms", "p95_ms", "min_ms", "max_ms"])
        writer.writeheader()
        for row in results:
            writer.writerow({k: row[k] for k in writer.fieldnames})
    if skipped:
        (OUT / "anchor_latency_skipped.json").write_text(json.dumps(skipped, indent=2))

    fig, ax = plt.subplots(figsize=(4.2, 2.4))
    xs = list(range(len(results)))
    if results:
        labels = [row["backend"] for row in results]
        medians = [row["median_ms"] for row in results]
        p95s = [row["p95_ms"] for row in results]
        ax.bar([x - 0.15 for x in xs], medians, width=0.3, label="median")
        ax.bar([x + 0.15 for x in xs], p95s, width=0.3, label="p95")
        ax.set_xticks(xs)
        ax.set_xticklabels(labels)
        ax.set_ylabel("ms")
        ax.set_title("Anchor round-trip latency")
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(OUT / "anchor_latency.png", dpi=200)
        plt.close(fig)

    print(json.dumps({"rows": results, "skipped": skipped}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
