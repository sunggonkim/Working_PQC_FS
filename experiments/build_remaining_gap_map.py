#!/usr/bin/env python3
"""Summarize the remaining open checklist items and their evidence gaps.

This report does not change any claim status. It simply consolidates the
current retained evidence and the still-missing proof obligations so the next
verification pass can target the right work.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "artifacts" / "reports" / "remaining_gap_map"


GAPS = [
    {
        "item": "storage-DMA proof: raw NVMe O_DIRECT path crosses the intended NVMe-to-UVM boundary",
        "current_evidence": [
            "artifacts/probes/evidence/repro_malloc_register.out",
            "artifacts/probes/evidence/io_uring_uvm.out",
            "artifacts/probes/evidence/io_uring_uvm_nvme_sudo.out",
            "artifacts/validation/uma_storage_dma_same_buffer/",
            "artifacts/validation/uma_storage_dma_profile_combined_report/",
            "artifacts/validation/uma_storage_dma_profile_combined/ncu/",
            "artifacts/validation/uma_counter_availability/",
            "artifacts/validation/um_smoke.json",
            "artifacts/results/motivation/uvm_proxy_report/",
            "experiments/run_uma_storage_dma_probe.py",
        ],
        "missing_evidence": [
            "full NVMe-to-UVM DMA semantics",
            "a final FUSE data-path proof that the production path itself crosses the intended NVMe-to-UVM boundary",
        ],
    },
    {
        "item": "Foreground AI p99 QoS restoration",
        "current_evidence": [
            "pqc_admission.c/h software proxy path: pqc_admission_record_uma_event(), pqc_admission_update_telemetry()",
            "pqc_admission.c threshold-softening branch in pqc_admit()",
            "artifacts/validation/tegra_qos_daemon_trace.jsonl",
            "artifacts/validation/run_qos_gpu_trace.jsonl",
            "artifacts/validation/telemetry_trace_report/",
            "artifacts/validation/qos_repeated_run/",
            "artifacts/validation/qos_repeated_report/",
            "artifacts/validation/qos_measured_pressure_adapter/",
            "artifacts/validation/qos_live_telemetry_admission/",
            "artifacts/validation/qos_fuse_live_bridge/ (live tegrastats reaches mounted daemon; runtime_fuse_throttle_trace.jsonl records in-daemon throttle)",
            "artifacts/validation/qos_cupti_pm_fuse_bridge/ (live CUPTI PM samples drive the mounted daemon in the same execution)",
            "artifacts/reports/m3_qos_ci_report/",
            "artifacts/reports/tensorrt_ci_report/",
        ],
        "missing_evidence": [
            "a single foreground inference co-run that reports p50/p95/p99 latency under no-background, naïve-background, and AEGIS-Q-controlled background I/O",
            "repeated-run confidence intervals showing that PM/CUPTI-driven throttling recovers foreground inference tail latency rather than only throttling the background FUSE path",
        ],
    },
    {
        "item": "Power-loss / FUSE-daemon crash timing",
        "current_evidence": [
            "pqc_anchor.c / pqc_fuse.c file and hardware anchor paths with fail-closed loading",
            "artifacts/results/recovery/sqlite_strace.log",
            "artifacts/reports/sqlite_ci_report/",
            "artifacts/validation/sqlite_recovery_oracle/",
            "artifacts/validation/sqlite_fault_campaign/",
            "artifacts/validation/combined_durability_bundle/ (same-backing-store SQLite+TPM and dbm.dumb+TPM fail-closed stale-snapshot replay)",
            "artifacts/validation/sqlite_syscall_crash_tpm/ (fdatasync-exact SQLite app-process SIGKILL timing on TPM-backed FUSE)",
            "artifacts/results/recovery/crash_replay_e8_test_summary.json",
            "artifacts/reports/crash_audit_report/",
        ],
        "missing_evidence": [
            "power-loss or FUSE-daemon crash timing rather than app-process SIGKILL only",
            "arbitrary interruption safety across all cut points",
        ],
    },
    {
        "item": "Second hardware platform / driver-version matrix",
        "current_evidence": [
            "artifacts/validation/microbench/summary.json current platform manifest",
            "artifacts/reports/platform_inventory_report/",
            "artifacts/repro_bundle/",
        ],
        "missing_evidence": [
            "preserved raw outputs from the same scripts on at least one additional Jetson, kernel, or JetPack revision",
        ],
    },
    {
        "item": "Combined durability beyond app-process crash timing",
        "current_evidence": [
            "experiments/run_combined_durability_bundle.py (TPM-only and app-only checks returncode 0)",
            "artifacts/validation/combined_durability_bundle/combined_durability_bundle.json",
            "same-backing-store SQLite baseline row count 1, advanced row count 3, stale replay fail_closed at SQLite open",
            "same-backing-store dbm.dumb baseline row count 1, advanced row count 3, stale replay fail_closed at dbm read",
            "fdatasync-exact SQLite app-process SIGKILL timing: 3 trials, 0 unacceptable verdicts",
        ],
        "missing_evidence": [
            "power-loss or FUSE-daemon crash timing for the combined SQLite+TPM path",
        ],
    },
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    report = {
        "summary": {
            "broader_gap_count": len(GAPS),
            "status": "Broader claim gaps remain out of scope; QoS PM-sample-to-mounted-FUSE wiring, UMA managed-buffer diagnostics, and TPM replay-after-advance fail-closed evidence are verified separately",
        },
        "items": GAPS,
    }
    (args.out_dir / "remaining_gap_map.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    md = [
        "# Remaining gap map",
        "",
        "This report consolidates the currently open checklist items and the evidence that is already retained.",
        "",
    ]
    for entry in GAPS:
        md.append(f"## {entry['item']}")
        md.append("Current evidence:")
        for ev in entry["current_evidence"]:
            md.append(f"- {ev}")
        md.append("Missing evidence:")
        for miss in entry["missing_evidence"]:
            md.append(f"- {miss}")
        md.append("")
    (args.out_dir / "remaining_gap_map.md").write_text("\n".join(md).rstrip() + "\n", encoding="utf-8")

    print(json.dumps({"out_dir": str(args.out_dir), "broader_gaps": len(GAPS)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
