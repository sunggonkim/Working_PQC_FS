#!/usr/bin/env python3
"""Run the R3 GPU data-plane offload negative-control benchmark."""

from __future__ import annotations

import csv
import json
import statistics
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
BUILD = ROOT / "build"
BIN_CANDIDATES = [
    BUILD / "bench_gpu_dataplane_negative_control",
    BUILD / "code" / "bench_gpu_dataplane_negative_control",
]
OUT = ROOT / "artifacts" / "validation" / "gpu_dataplane_negative_control"
JSON_OUT = OUT / "gpu_dataplane_negative_control.json"
MD_OUT = OUT / "gpu_dataplane_negative_control.md"


def now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def run_command(argv: list[str], timeout: float = 60.0) -> dict[str, Any]:
    proc = subprocess.run(
        argv,
        cwd=ROOT,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
    )
    return {
        "argv": argv,
        "returncode": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
    }


def parse_rows(stdout: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in csv.DictReader(stdout.splitlines()):
        rows.append({
            "mode": row["mode"],
            "count": int(row["count"]),
            "total_bytes": int(row["total_bytes"]),
            "rep": int(row["rep"]),
            "ns": int(row["ns"]),
            "mib_s": float(row["mib_s"]),
            "verified": row["verified"] == "1",
            "gpu_available": row["gpu_available"] == "1",
        })
    return rows


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_case: dict[tuple[int, str], list[dict[str, Any]]] = {}
    for row in rows:
        by_case.setdefault((row["count"], row["mode"]), []).append(row)

    cases: list[dict[str, Any]] = []
    counts = sorted({row["count"] for row in rows})
    for count in counts:
        cpu = by_case.get((count, "cpu"), [])
        gpu = by_case.get((count, "gpu"), [])
        cpu_ns = [row["ns"] for row in cpu]
        gpu_ns = [row["ns"] for row in gpu]
        item: dict[str, Any] = {
            "count": count,
            "total_bytes": count * 4096,
            "cpu_median_ns": statistics.median(cpu_ns) if cpu_ns else None,
            "gpu_median_ns": statistics.median(gpu_ns) if gpu_ns else None,
            "cpu_median_mib_s": statistics.median([row["mib_s"] for row in cpu]) if cpu else None,
            "gpu_median_mib_s": statistics.median([row["mib_s"] for row in gpu]) if gpu else None,
            "verified": all(row["verified"] for row in cpu + gpu),
            "gpu_available": any(row["gpu_available"] for row in cpu + gpu),
        }
        if item["cpu_median_ns"] and item["gpu_median_ns"]:
            item["gpu_slower_ratio"] = item["gpu_median_ns"] / item["cpu_median_ns"]
            item["negative_control_supports_cpu_first"] = item["gpu_slower_ratio"] > 1.0
        else:
            item["gpu_slower_ratio"] = None
            item["negative_control_supports_cpu_first"] = False
        cases.append(item)

    small_cases = [case for case in cases if case["count"] in {1, 4, 16}]
    any_gpu_loss = any(case["negative_control_supports_cpu_first"] for case in small_cases)
    all_verified = all(case["verified"] for case in cases)
    gpu_available = any(case["gpu_available"] for case in cases)
    return {
        "cases": cases,
        "gpu_available": gpu_available,
        "all_verified": all_verified,
        "small_or_mid_gpu_loss_observed": any_gpu_loss,
        "overall_pass": gpu_available and all_verified and any_gpu_loss,
        "claim_boundary": (
            "This is a data-plane negative control for CPU-first AES-GCM placement. "
            "It does not claim GPU data-plane offload is always slower, only that "
            "the retained small/mid block evidence includes loss cases under the "
            "current UMA/CUDA executor."
        ),
    }


def write_outputs(command: dict[str, Any], rows: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    stdout_path = OUT / "bench_gpu_dataplane_negative_control.stdout.csv"
    stderr_path = OUT / "bench_gpu_dataplane_negative_control.stderr.txt"
    stdout_path.write_text(command["stdout"], encoding="utf-8")
    stderr_path.write_text(command["stderr"], encoding="utf-8")
    report = {
        "schema_version": 1,
        "generated_utc": now_utc(),
        "benchmark": command["argv"][0],
        "command": {
            "argv": command["argv"],
            "returncode": command["returncode"],
            "stdout_csv": rel(stdout_path),
            "stderr": rel(stderr_path),
        },
        "rows": rows,
        "summary": summary,
    }
    JSON_OUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "# GPU Data-Plane Negative Control",
        "",
        f"- Overall pass: `{summary['overall_pass']}`",
        f"- GPU available: `{summary['gpu_available']}`",
        f"- All CPU/GPU outputs verified: `{summary['all_verified']}`",
        f"- Small/mid GPU loss observed: `{summary['small_or_mid_gpu_loss_observed']}`",
        "",
        "| blocks | bytes | CPU median ns | GPU median ns | GPU/CPU |",
        "| ---: | ---: | ---: | ---: | ---: |",
    ]
    for case in summary["cases"]:
        ratio = case["gpu_slower_ratio"]
        lines.append(
            f"| {case['count']} | {case['total_bytes']} | {case['cpu_median_ns']} | "
            f"{case['gpu_median_ns']} | {ratio:.2f} |" if ratio is not None else
            f"| {case['count']} | {case['total_bytes']} | {case['cpu_median_ns']} | n/a | n/a |"
        )
    lines.extend(["", summary["claim_boundary"], ""])
    MD_OUT.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    binary = next((path for path in BIN_CANDIDATES if path.exists()), None)
    if binary is None:
        raise SystemExit(
            "missing benchmark binary; tried "
            + ", ".join(str(path) for path in BIN_CANDIDATES)
        )
    command = run_command([str(binary), "--warmups", "2", "--reps", "7"])
    rows = parse_rows(command["stdout"]) if command["returncode"] == 0 else []
    summary = summarize(rows) if rows else {
        "cases": [],
        "gpu_available": False,
        "all_verified": False,
        "small_or_mid_gpu_loss_observed": False,
        "overall_pass": False,
        "claim_boundary": "benchmark did not complete",
    }
    write_outputs(command, rows, summary)
    print(json.dumps({
        "json": rel(JSON_OUT),
        "md": rel(MD_OUT),
        "overall_pass": summary["overall_pass"],
        "gpu_available": summary["gpu_available"],
        "small_or_mid_gpu_loss_observed": summary["small_or_mid_gpu_loss_observed"],
    }, indent=2, sort_keys=True))
    return 0 if summary["overall_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
