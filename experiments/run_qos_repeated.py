#!/usr/bin/env python3
"""Repeat the QoS prototype run and retain a conservative summary bundle.

This harness does not invent PMU-backed results. It simply reruns the existing
telemetry prototype script multiple times and records where the retained traces
land so the project can later attach a real counter-backed analysis to the same
workload.
"""

from __future__ import annotations

import argparse
import shutil
import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = ROOT / "artifacts" / "validation" / "qos_repeated_run"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    records = []
    for idx in range(args.runs):
        proc = subprocess.run(
            ["python3", "experiments/run_qos_gpu.py"],
            cwd=ROOT,
            text=True,
            capture_output=True,
        )
        trace_src = ROOT / "artifacts" / "validation" / "run_qos_gpu_trace.jsonl"
        summary_src = ROOT / "artifacts" / "validation" / "run_qos_gpu_summary.json"
        trace_dst = out_dir / f"run_{idx + 1}_run_qos_gpu_trace.jsonl"
        summary_dst = out_dir / f"run_{idx + 1}_run_qos_gpu_summary.json"
        if trace_src.exists():
            shutil.copy2(trace_src, trace_dst)
        if summary_src.exists():
            shutil.copy2(summary_src, summary_dst)
        run_record = {
            "run": idx + 1,
            "returncode": proc.returncode,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
            "trace_file": str(trace_dst),
            "summary_file": str(summary_dst),
            "trace_retained": trace_dst.exists(),
            "summary_retained": summary_dst.exists(),
        }
        (out_dir / f"run_{idx + 1}.json").write_text(json.dumps(run_record, indent=2), encoding="utf-8")
        records.append(run_record)

    report = {
        "runs": args.runs,
        "note": "Prototype repetition only; not a PMU/CUPTI-backed stability proof.",
        "records": [
            {
                "run": r["run"],
                "returncode": r["returncode"],
                "trace_file": r["trace_file"],
                "summary_file": r["summary_file"],
                "trace_retained": r["trace_retained"],
                "summary_retained": r["summary_retained"],
            }
            for r in records
        ],
    }
    (out_dir / "qos_repeated_run.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"out_dir": str(out_dir), "runs": args.runs}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
