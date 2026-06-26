#!/usr/bin/env python3
"""Build a conservative dashboard of verified vs open evidence.

This report is a bookkeeping aid only. It does not claim any unsupported
result and does not upgrade open gaps into verified claims.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = ROOT / "artifacts" / "evidence_dashboard"


def maybe_load(path: Path):
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    dashboard = {
        "verified_artifact_maps": {
            "crash_audit_report": str(ROOT / "artifacts" / "crash_audit_report" / "crash_audit_report.json"),
            "platform_inventory_report": str(ROOT / "artifacts" / "platform_inventory_report" / "platform_inventory_report.json"),
            "tpm_freshness_report": str(ROOT / "artifacts" / "validation" / "tpm_freshness_report" / "tpm_freshness_report.json"),
            "qos_repeated_report": str(ROOT / "artifacts" / "validation" / "qos_repeated_report" / "qos_repeated_report.json"),
            "uma_storage_dma_report": str(ROOT / "artifacts" / "validation" / "uma_storage_dma_report" / "uma_storage_dma_report.json"),
            "sqlite_fault_campaign": str(ROOT / "artifacts" / "validation" / "sqlite_fault_campaign" / "sqlite_fault_campaign.json"),
        },
        "status": {
            "storage_dma": "open",
            "qos_pmu_hysteresis": "open",
            "tpm_freshness_hardware": "open",
            "sqlite_app_recovery_campaign": "partial",
            "full_multi_workload_app_recovery": "open",
            "second_platform": "open",
            "combined_durability": "open",
        },
        "note": "Dashboard only; all open items remain open until concrete evidence is retained.",
    }

    (args.out_dir / "evidence_dashboard.json").write_text(json.dumps(dashboard, indent=2), encoding="utf-8")
    md = [
        "# Evidence dashboard",
        "",
        "This dashboard tracks retained artifact maps and the still-open evidence gaps.",
        "",
        "## Open items",
    ]
    for k, v in dashboard["status"].items():
        md.append(f"- {k}: {v}")
    md.append("")
    md.append("This report does not upgrade any open item into a verified claim.")
    (args.out_dir / "evidence_dashboard.md").write_text("\n".join(md), encoding="utf-8")
    print(json.dumps({"out_dir": str(args.out_dir)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
