#!/usr/bin/env python3
"""Build a conservative index of the current evidence packaging outputs.

This script is bookkeeping only. It records where the retained package/report
artifacts live and whether they exist.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = ROOT / "artifacts" / "reports" / "index_report"


PACKAGES = {
    "crash_audit_report": ROOT / "artifacts" / "reports" / "crash_audit_report" / "crash_audit_report.json",
    "platform_inventory_report": ROOT / "artifacts" / "reports" / "platform_inventory_report" / "platform_inventory_report.json",
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
    "app_recovery_bundle": ROOT / "artifacts" / "validation" / "app_recovery_bundle" / "app_recovery_bundle.json",
    "app_recovery_report": ROOT / "artifacts" / "validation" / "app_recovery_report" / "app_recovery_report.json",
    "combined_durability_bundle": ROOT / "artifacts" / "validation" / "combined_durability_bundle" / "combined_durability_bundle.json",
    "sqlite_recovery_oracle": ROOT / "artifacts" / "validation" / "sqlite_recovery_oracle" / "sqlite_recovery_oracle.json",
    "sqlite_fault_campaign": ROOT / "artifacts" / "validation" / "sqlite_fault_campaign" / "sqlite_fault_campaign.json",
    "evidence_dashboard": ROOT / "artifacts" / "reports" / "evidence_dashboard" / "evidence_dashboard.json",
    "status_register": ROOT / "artifacts" / "reports" / "status_register" / "status_register.json",
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    index = {
        "packages": {name: str(path) for name, path in PACKAGES.items()},
        "present": {name: path.exists() for name, path in PACKAGES.items()},
        "note": "Index report only; does not prove any open claim.",
    }
    (args.out_dir / "index_report.json").write_text(json.dumps(index, indent=2), encoding="utf-8")
    md = [
        "# Evidence index report",
        "",
        "This index points at the current report/package artifacts.",
        "",
        "## Packages",
    ]
    for name, present in index["present"].items():
        md.append(f"- {name}: {present}")
    md.append("")
    md.append("This report does not upgrade any open claim.")
    (args.out_dir / "index_report.md").write_text("\n".join(md), encoding="utf-8")
    print(json.dumps({"out_dir": str(args.out_dir)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
