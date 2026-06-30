#!/usr/bin/env python3
"""Feed Nsight-derived pressure into the existing admission controller path.

This script is intentionally narrow:

  * It does not invent PMU/CUPTI QoS results.
  * It parses retained Nsight Compute CSV metrics from an existing workload.
  * It maps those measured percentages to the existing admission telemetry
    inputs (`PQC_TELEMETRY_MEM_BANDWIDTH`, `PQC_TELEMETRY_TENSOR_CORE`).
  * It runs `pqc_fuse --admission-telemetry-smoke`, which calls
    `pqc_admission_update_telemetry()` and `pqc_admit()`.

The resulting bundle proves that a measured-pressure adapter reaches the
controller path and affects a deterministic admission decision.  It remains a
controller-path proof, not an end-to-end PMU-backed QoS claim.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
BUILD = ROOT / "build" / "pqc_fuse"
DEFAULT_NCU_CSV = (
    ROOT
    / "artifacts"
    / "validation"
    / "uma_storage_dma_profile_combined"
    / "ncu"
    / "io_uring_uvm_checksum.csv"
)


def parse_percent(value: str) -> float:
    value = value.strip().replace(",", "")
    if value.lower() == "nan" or not value:
        return 0.0
    return float(value)


def extract_ncu_pressure(csv_path: Path) -> dict[str, Any]:
    metrics: dict[str, float] = {}
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row.get("Metric Name", "")
            unit = row.get("Metric Unit", "")
            if unit != "%":
                continue
            if name in {
                "Memory Throughput",
                "Compute (SM) Throughput",
                "Mem Busy",
                "SM Busy",
                "Issue Slots Busy",
                "Achieved Occupancy",
            }:
                metrics[name] = parse_percent(row.get("Metric Value", "0"))

    required = ["Memory Throughput", "Compute (SM) Throughput"]
    missing = [name for name in required if name not in metrics]
    if missing:
        raise SystemExit(f"missing required NCU metrics in {csv_path}: {missing}")

    return {
        "source_csv": str(csv_path.relative_to(ROOT)),
        "raw_percent_metrics": metrics,
        "adapter_mapping": {
            "PQC_TELEMETRY_MEM_BANDWIDTH": "Memory Throughput / 100",
            "PQC_TELEMETRY_TENSOR_CORE": "Compute (SM) Throughput / 100",
        },
        "mem_bandwidth_util": max(0.0, min(1.0, metrics["Memory Throughput"] / 100.0)),
        "tensor_core_util": max(0.0, min(1.0, metrics["Compute (SM) Throughput"] / 100.0)),
    }


def run_case(out_dir: Path, name: str, mem_bw: float, tensor_core: float) -> dict[str, Any]:
    trace_path = out_dir / f"{name}.jsonl"
    stdout_path = out_dir / f"{name}.stdout"
    stderr_path = out_dir / f"{name}.stderr"
    if trace_path.exists():
        trace_path.unlink()

    env = os.environ.copy()
    env.update(
        {
            "PQC_ADMISSION_TRACE_PATH": str(trace_path),
            "PQC_TELEMETRY_MEM_BANDWIDTH": f"{mem_bw:.6f}",
            "PQC_TELEMETRY_TENSOR_CORE": f"{tensor_core:.6f}",
            "PQC_ADMISSION_SMOKE_AI_BUDGET_NS": "2000000",
            "PQC_ADMISSION_SMOKE_CPU_QUEUE_DEPTH": "1",
            "PQC_ADMISSION_SMOKE_GPU_QUEUE_DEPTH": "1",
            "PQC_ADMISSION_SMOKE_BYTES": "131072",
            "PQC_ADMISSION_SMOKE_GPU_KERNEL_NS": "100000",
            "PQC_ADMISSION_SMOKE_H2D_NS": "100000",
            "PQC_ADMISSION_SMOKE_D2H_NS": "100000",
        }
    )
    cmd = [str(BUILD), "--admission-telemetry-smoke"]
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True, check=False)
    stdout_path.write_text(proc.stdout, encoding="utf-8")
    stderr_path.write_text(proc.stderr, encoding="utf-8")

    summary: dict[str, Any] | None = None
    for line in proc.stdout.splitlines():
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            summary = json.loads(line)
            break
    if summary is None:
        raise SystemExit(f"{name}: smoke did not emit JSON summary; see {stdout_path}")

    trace_records = []
    if trace_path.exists():
        for line in trace_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line.startswith("{") and line.endswith("}"):
                trace_records.append(json.loads(line))

    return {
        "name": name,
        "command": cmd,
        "returncode": proc.returncode,
        "stdout": str(stdout_path.relative_to(ROOT)),
        "stderr": str(stderr_path.relative_to(ROOT)),
        "trace": str(trace_path.relative_to(ROOT)),
        "input": {
            "mem_bandwidth_util": mem_bw,
            "tensor_core_util": tensor_core,
        },
        "summary": summary,
        "trace_records": trace_records,
    }


def write_markdown(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# QoS measured-pressure adapter report",
        "",
        "This bundle feeds retained Nsight-derived pressure metrics into the existing admission controller path.",
        "It proves controller-path integration only; it is not an end-to-end PMU-backed QoS result.",
        "",
        "## Source metrics",
        "",
        f"- Source CSV: `{report['source']['source_csv']}`",
        f"- Memory bandwidth util: `{report['source']['mem_bandwidth_util']:.4f}`",
        f"- Tensor-core/SM util: `{report['source']['tensor_core_util']:.4f}`",
        "",
        "## Admission cases",
        "",
        "| Case | mem util | tensor util | target | trace |",
        "|---|---:|---:|---|---|",
    ]
    for case in report["cases"]:
        summary = case["summary"]
        lines.append(
            f"| {case['name']} | {case['input']['mem_bandwidth_util']:.4f} | "
            f"{case['input']['tensor_core_util']:.4f} | "
            f"{summary.get('chosen_target')} | `{case['trace']}` |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- `idle_control` records the deterministic baseline with no measured pressure.",
            "- `nsight_derived` records the same admission context after mapping retained NCU metrics into the admission telemetry inputs.",
            "- The JSONL trace contains `telemetry_mem_bandwidth_util` and `telemetry_tensor_core_util`, so the artifact directly ties the measured adapter values to the controller decision.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ncu-csv", type=Path, default=DEFAULT_NCU_CSV)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "artifacts" / "validation" / "qos_measured_pressure_adapter")
    args = parser.parse_args()

    if not BUILD.exists():
        raise SystemExit("missing build/pqc_fuse; run cmake --build build first")
    csv_path = args.ncu_csv if args.ncu_csv.is_absolute() else ROOT / args.ncu_csv
    if not csv_path.exists():
        raise SystemExit(f"missing NCU CSV: {csv_path}")

    out_dir = args.out_dir if args.out_dir.is_absolute() else ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    source = extract_ncu_pressure(csv_path)
    cases = [
        run_case(out_dir, "idle_control", 0.0, 0.0),
        run_case(out_dir, "nsight_derived", source["mem_bandwidth_util"], source["tensor_core_util"]),
    ]

    report = {
        "note": "Controller-path adapter proof only; not an end-to-end PMU/CUPTI-backed QoS claim.",
        "source": source,
        "cases": cases,
    }
    report_json = out_dir / "qos_measured_pressure_adapter.json"
    report_md = out_dir / "qos_measured_pressure_adapter.md"
    report_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_markdown(report, report_md)
    print(json.dumps({"report": str(report_json.relative_to(ROOT)), "cases": len(cases)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
