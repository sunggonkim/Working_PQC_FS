#!/usr/bin/env python3
"""Package the conservative app-recovery bundle into a report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_IN = ROOT / "artifacts" / "validation" / "app_recovery_bundle"
DEFAULT_OUT = ROOT / "artifacts" / "validation" / "app_recovery_report"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", type=Path, default=DEFAULT_IN)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    bundle_path = args.in_dir / "app_recovery_bundle.json"
    if bundle_path.exists():
        bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    else:
        bundle = {"note": "bundle missing; run experiments/run_app_recovery_bundle.py first", "checks": []}
    sqlite_campaign = ROOT / "artifacts" / "validation" / "sqlite_fault_campaign" / "sqlite_fault_campaign.json"
    if sqlite_campaign.exists():
        campaign = json.loads(sqlite_campaign.read_text(encoding="utf-8"))
        unacceptable = sum(1 for row in campaign.get("rows", []) if not row.get("acceptable", False))
    else:
        campaign = None
        unacceptable = None
    report = {
        "note": "SQLite-only app recovery report; not evidence of full multi-workload crash certification.",
        "bundle": bundle,
        "sqlite_fault_campaign": {
            "path": str(sqlite_campaign),
            "present": sqlite_campaign.exists(),
            "unacceptable": unacceptable,
        },
    }
    (args.out_dir / "app_recovery_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    md = [
        "# App recovery report",
        "",
        "This bundle packages retained crash audit outputs plus the executable SQLite oracle campaign.",
        "",
        f"- Input directory: `{args.in_dir}`",
        f"- Checks: `{len(bundle.get('checks', []))}`",
        f"- Bundle present: `{bundle_path.exists()}`",
        "",
        f"- SQLite fault campaign present: `{sqlite_campaign.exists()}`",
        f"- SQLite unacceptable oracle verdicts: `{unacceptable}`",
        "",
        "This report does not claim full multi-workload crash certification.",
    ]
    (args.out_dir / "app_recovery_report.md").write_text("\n".join(md), encoding="utf-8")
    print(json.dumps({"out_dir": str(args.out_dir)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
