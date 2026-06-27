#!/usr/bin/env python3
"""Validate the producer-facing admission slack/deadline interface.

This harness exercises the final ``build/pqc_fuse --admission-telemetry-smoke``
entry point under controlled producer inputs.  It validates the trace schema
and the safety fallbacks for supplied slack, no slack, elapsed deadline, and
stale slack samples.  It is an interface audit, not an end-to-end QoS result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
PQC_FUSE = ROOT / "build" / "pqc_fuse"
DEFAULT_OUT = ROOT / "artifacts" / "validation" / "admission_interface_audit"

REASON_AI_QOS_EXHAUSTED = 0x02
REASON_QUEUE_PRESSURE = 0x04
REASON_DEADLINE_ELAPSED = 0x08
REASON_STALE_TELEMETRY = 0x80

REQUIRED_TRACE_FIELDS = {
    "timestamp_ns",
    "trace_timestamp_clock",
    "age_clock",
    "batch_age_ns",
    "ai_inference_deadline_ns",
    "batch_count",
    "bytes_total",
    "queue_delay_ns",
    "service_time_ns",
    "gpu_kernel_est_ns",
    "gpu_h2d_staging_ns",
    "gpu_d2h_staging_ns",
    "cpu_queue_depth",
    "gpu_queue_depth",
    "telemetry_mem_bandwidth_util",
    "telemetry_tensor_core_util",
    "ai_qos_budget_remaining_ns",
    "producer_slack_age_ns",
    "producer_slack_stale_after_ns",
    "producer_slack_stale",
    "chosen_target",
    "deferral_reason",
    "decision_reason",
}


@dataclass(frozen=True)
class Case:
    name: str
    env: dict[str, str]
    expected_target: str
    expected_reason_mask: int
    expected_stale: bool


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def parse_json_lines(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            rows.append(json.loads(line))
    return rows


def parse_stdout_summary(stdout: str) -> dict[str, Any] | None:
    for line in stdout.splitlines():
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            return json.loads(line)
    return None


def run_case(case: Case, out_dir: Path) -> dict[str, Any]:
    case_dir = out_dir / case.name
    case_dir.mkdir(parents=True, exist_ok=True)
    trace_path = case_dir / "admission_trace.jsonl"
    stdout_path = case_dir / "stdout.txt"
    stderr_path = case_dir / "stderr.txt"

    env = os.environ.copy()
    env.update(
        {
            "PQC_ADMISSION_TRACE_PATH": str(trace_path),
            "PQC_TELEMETRY_MEM_BANDWIDTH": "0.10",
            "PQC_TELEMETRY_TENSOR_CORE": "0.10",
            "PQC_ADMISSION_SMOKE_BYTES": "131072",
            "PQC_ADMISSION_SMOKE_GPU_KERNEL_NS": "100000",
            "PQC_ADMISSION_SMOKE_H2D_NS": "100000",
            "PQC_ADMISSION_SMOKE_D2H_NS": "100000",
            "PQC_ADMISSION_SMOKE_CPU_QUEUE_DEPTH": "4",
            "PQC_ADMISSION_SMOKE_GPU_QUEUE_DEPTH": "0",
            "PQC_ADMISSION_SMOKE_DEADLINE_NS": "10000000",
            "PQC_ADMISSION_SMOKE_BATCH_AGE_NS": "0",
            "PQC_PRODUCER_SLACK_STALE_NS": "250000000",
        }
    )
    env.update(case.env)

    proc = subprocess.run(
        [str(PQC_FUSE), "--admission-telemetry-smoke"],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    stdout_path.write_text(proc.stdout, encoding="utf-8")
    stderr_path.write_text(proc.stderr, encoding="utf-8")

    rows = parse_json_lines(trace_path)
    decision_rows = [row for row in rows if "chosen_target" in row]
    decision = decision_rows[-1] if decision_rows else {}
    summary = parse_stdout_summary(proc.stdout) or {}
    missing_fields = sorted(REQUIRED_TRACE_FIELDS - set(decision))
    target_ok = decision.get("chosen_target") == case.expected_target
    reason_ok = bool(int(decision.get("decision_reason", 0)) & case.expected_reason_mask)
    stale_ok = bool(decision.get("producer_slack_stale", False)) == case.expected_stale
    clocks_ok = (
        decision.get("trace_timestamp_clock") == "CLOCK_REALTIME"
        and decision.get("age_clock") == "CLOCK_MONOTONIC"
    )
    deadline_ok = "ai_inference_deadline_ns" in decision and "batch_age_ns" in decision
    acceptable = (
        proc.returncode == 0
        and bool(decision)
        and not missing_fields
        and target_ok
        and reason_ok
        and stale_ok
        and clocks_ok
        and deadline_ok
    )
    return {
        "case": case.name,
        "acceptable": acceptable,
        "returncode": proc.returncode,
        "expected_target": case.expected_target,
        "expected_reason_mask": case.expected_reason_mask,
        "expected_stale": case.expected_stale,
        "summary": summary,
        "decision": decision,
        "missing_trace_fields": missing_fields,
        "checks": {
            "target_ok": target_ok,
            "reason_ok": reason_ok,
            "stale_ok": stale_ok,
            "clocks_ok": clocks_ok,
            "deadline_ok": deadline_ok,
        },
        "trace": str(trace_path.relative_to(ROOT)),
        "stdout": str(stdout_path.relative_to(ROOT)),
        "stderr": str(stderr_path.relative_to(ROOT)),
        "trace_sha256": sha256_bytes(trace_path.read_bytes()) if trace_path.exists() else None,
        "stdout_sha256": sha256_bytes(stdout_path.read_bytes()),
        "stderr_sha256": sha256_bytes(stderr_path.read_bytes()),
    }


def write_markdown(result: dict[str, Any], path: Path) -> None:
    lines = [
        "# Admission Interface Audit",
        "",
        f"- Overall pass: `{str(result['overall_pass']).lower()}`",
        "- Scope: producer-facing slack/deadline interface only; not an end-to-end QoS result.",
        "",
        "## Cases",
    ]
    for case in result["cases"]:
        decision = case["decision"]
        lines.append(
            f"- `{case['case']}` acceptable=`{str(case['acceptable']).lower()}`, "
            f"target=`{decision.get('chosen_target')}`, "
            f"decision_reason=`{decision.get('decision_reason')}`, "
            f"deferral_reason=`{decision.get('deferral_reason')}`, "
            f"deadline_ns=`{decision.get('ai_inference_deadline_ns')}`, "
            f"batch_age_ns=`{decision.get('batch_age_ns')}`, "
            f"slack_age_ns=`{decision.get('producer_slack_age_ns')}`, "
            f"stale=`{decision.get('producer_slack_stale')}`"
        )
    lines.extend(["", "## Interface Contract"])
    for item in result["interface_contract"]:
        lines.append(f"- {item}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    if not PQC_FUSE.exists():
        raise SystemExit(f"missing {PQC_FUSE}; run cmake --build build first")
    out_dir = args.out if args.out.is_absolute() else ROOT / args.out
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cases = [
        Case(
            name="slack_available_gpu",
            env={"PQC_ADMISSION_SMOKE_AI_BUDGET_NS": "2000000"},
            expected_target="GPU",
            expected_reason_mask=REASON_QUEUE_PRESSURE,
            expected_stale=False,
        ),
        Case(
            name="no_slack_cpu",
            env={"PQC_ADMISSION_SMOKE_AI_BUDGET_NS": "0"},
            expected_target="CPU",
            expected_reason_mask=REASON_AI_QOS_EXHAUSTED,
            expected_stale=False,
        ),
        Case(
            name="deadline_elapsed_cpu",
            env={
                "PQC_ADMISSION_SMOKE_AI_BUDGET_NS": "2000000",
                "PQC_ADMISSION_SMOKE_BATCH_AGE_NS": "20000000",
                "PQC_ADMISSION_SMOKE_DEADLINE_NS": "10000000",
            },
            expected_target="CPU",
            expected_reason_mask=REASON_DEADLINE_ELAPSED,
            expected_stale=False,
        ),
        Case(
            name="stale_slack_cpu",
            env={
                "PQC_ADMISSION_SMOKE_AI_BUDGET_NS": "2000000",
                "PQC_PRODUCER_SLACK_STALE_NS": "1",
                "PQC_ADMISSION_SMOKE_STALE_SLEEP_US": "2000",
            },
            expected_target="CPU",
            expected_reason_mask=REASON_STALE_TELEMETRY,
            expected_stale=True,
        ),
    ]

    results = [run_case(case, out_dir) for case in cases]
    result = {
        "artifact": "admission_interface_audit",
        "overall_pass": all(case["acceptable"] for case in results),
        "cases": results,
        "interface_contract": [
            "Deadline source: producer-supplied relative deadline/slack in nanoseconds.",
            "Timestamp domain: trace timestamp uses CLOCK_REALTIME; batch age and slack age use CLOCK_MONOTONIC-relative nanoseconds.",
            "Clock synchronization: no cross-process synchronization is assumed for relative age/deadline fields.",
            "Stale-sample behavior: producer slack older than PQC_PRODUCER_SLACK_STALE_NS routes to CPU with STALE_TELEMETRY.",
            "No-slack fallback: zero supplied slack routes to CPU with AI_QOS_EXHAUSTED.",
            "Safety default: admission initializes and falls back to CPU unless all GPU gates pass.",
        ],
    }
    json_path = out_dir / "admission_interface_audit.json"
    md_path = out_dir / "admission_interface_audit.md"
    json_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown(result, md_path)
    print(json.dumps({"overall_pass": result["overall_pass"], "out": str(json_path)}, sort_keys=True))
    return 0 if result["overall_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
