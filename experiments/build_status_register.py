#!/usr/bin/env python3
"""Build a conservative status register for all retained evidence packages.

This is a bookkeeping tool. It records which scaffold/report artifacts already
exist and which open claims remain open. It does not upgrade any claim.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = ROOT / "artifacts" / "status_register"


REPORTS = {
    "crash_audit_report": ROOT / "artifacts" / "crash_audit_report" / "crash_audit_report.json",
    "platform_inventory_report": ROOT / "artifacts" / "platform_inventory_report" / "platform_inventory_report.json",
    "tpm_freshness_bundle": ROOT / "artifacts" / "validation" / "tpm_freshness_bundle" / "tpm_freshness_bundle.json",
    "tpm_freshness_report": ROOT / "artifacts" / "validation" / "tpm_freshness_report" / "tpm_freshness_report.json",
    "tpm_only_bundle": ROOT / "artifacts" / "validation" / "tpm_only_bundle" / "tpm_only_bundle.json",
    "tpm_monotonic_replay": ROOT / "artifacts" / "validation" / "tpm_monotonic_replay" / "tpm_monotonic_replay.json",
    "tpm_recovery_verdict": ROOT / "artifacts" / "validation" / "tpm_recovery_verdict" / "tpm_recovery_verdict.json",
    "tpm_provisioning_probe": ROOT / "artifacts" / "validation" / "tpm_provisioning_probe" / "tpm_provisioning_probe.json",
    "tpm_pcr_policy_probe": ROOT / "artifacts" / "validation" / "tpm_pcr_policy_probe" / "tpm_pcr_policy_probe.json",
    "qos_repeated_run": ROOT / "artifacts" / "validation" / "qos_repeated_run" / "qos_repeated_run.json",
    "qos_repeated_report": ROOT / "artifacts" / "validation" / "qos_repeated_report" / "qos_repeated_report.json",
    "qos_live_telemetry_admission": ROOT / "artifacts" / "validation" / "qos_live_telemetry_admission" / "qos_live_telemetry_admission.json",
    "uma_storage_dma_repeated": ROOT / "artifacts" / "validation" / "uma_storage_dma_repeated" / "uma_storage_dma_repeated.json",
    "uma_storage_dma_report": ROOT / "artifacts" / "validation" / "uma_storage_dma_report" / "uma_storage_dma_report.json",
    "evidence_dashboard": ROOT / "artifacts" / "evidence_dashboard" / "evidence_dashboard.json",
    "app_recovery_bundle": ROOT / "artifacts" / "validation" / "app_recovery_bundle" / "app_recovery_bundle.json",
    "app_recovery_report": ROOT / "artifacts" / "validation" / "app_recovery_report" / "app_recovery_report.json",
    "combined_durability_bundle": ROOT / "artifacts" / "validation" / "combined_durability_bundle" / "combined_durability_bundle.json",
    "sqlite_recovery_oracle": ROOT / "artifacts" / "validation" / "sqlite_recovery_oracle" / "sqlite_recovery_oracle.json",
    "sqlite_fault_campaign": ROOT / "artifacts" / "validation" / "sqlite_fault_campaign" / "sqlite_fault_campaign.json",
}


OPEN_ITEMS = [
    "storage-DMA proof",
    "QoS PMU + hysteresis",
    "full multi-workload app recovery classification",
    "second-platform portability",
    "combined durability campaign",
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    register = {
        "reports": {name: str(path) for name, path in REPORTS.items()},
        "present": {name: path.exists() for name, path in REPORTS.items()},
        "open_items": OPEN_ITEMS,
        "note": "Status register only; open items remain open until direct evidence exists.",
    }
    (args.out_dir / "status_register.json").write_text(json.dumps(register, indent=2), encoding="utf-8")
    md = [
        "# Status register",
        "",
        "This register lists retained packaging artifacts and the open items that remain open.",
        "",
        "## Present reports",
    ]
    for name, present in register["present"].items():
        md.append(f"- {name}: {present}")
    md.append("")
    md.append("## Open items")
    for item in OPEN_ITEMS:
        md.append(f"- {item}")
    md.append("")
    md.append("This register does not upgrade any open item into a verified claim.")
    (args.out_dir / "status_register.md").write_text("\n".join(md), encoding="utf-8")
    print(json.dumps({"out_dir": str(args.out_dir)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
