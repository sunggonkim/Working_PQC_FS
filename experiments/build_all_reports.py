#!/usr/bin/env python3
"""Run the retained bookkeeping/report generators in a conservative order.

This script is intentionally not a proof runner. It only invokes the existing
package/report generators so the evidence maps can be refreshed together.
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_LOG_DIR = ROOT / "artifacts" / "build_all_reports_logs"


SCRIPTS = [
    "experiments/build_sqlite_recovery_oracle.py",
    "experiments/run_sqlite_fault_campaign.py",
    "experiments/build_crash_audit_report.py",
    "experiments/build_platform_inventory_report.py",
    "experiments/build_tpm_freshness_report.py",
    "experiments/build_qos_repeated_report.py",
    "experiments/build_uma_storage_dma_report.py",
    "experiments/build_app_recovery_report.py",
    "experiments/build_evidence_dashboard.py",
    "experiments/build_status_register.py",
    "experiments/build_index_report.py",
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR)
    args = ap.parse_args()

    args.log_dir.mkdir(parents=True, exist_ok=True)
    for script in SCRIPTS:
        proc = subprocess.run(["python3", script], cwd=ROOT, text=True, capture_output=True)
        stem = script.replace("/", "_").replace(".py", "")
        (args.log_dir / f"{stem}.stdout.txt").write_text(proc.stdout, encoding="utf-8")
        (args.log_dir / f"{stem}.stderr.txt").write_text(proc.stderr, encoding="utf-8")
        if proc.returncode != 0:
            print(f"{script} failed with {proc.returncode}")
            return proc.returncode

    print(f"wrote logs to {args.log_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
