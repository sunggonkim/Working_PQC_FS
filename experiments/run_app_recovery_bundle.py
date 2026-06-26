#!/usr/bin/env python3
"""Run conservative app-recovery related probes as a single bundle.

This bundle is intentionally limited to the retained SQLite / replay evidence
and the executable SQLite oracle campaign.  It does not claim complete
multi-workload crash certification.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = ROOT / "artifacts" / "validation" / "app_recovery_bundle"


def run_cmd(cmd: list[str], out_dir: Path, name: str) -> dict:
    proc = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{name}.stdout.txt").write_text(proc.stdout, encoding="utf-8")
    (out_dir / f"{name}.stderr.txt").write_text(proc.stderr, encoding="utf-8")
    return {
        "name": name,
        "command": cmd,
        "returncode": proc.returncode,
        "stdout": str(out_dir / f"{name}.stdout.txt"),
        "stderr": str(out_dir / f"{name}.stderr.txt"),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    checks = []
    checks.append(run_cmd(["python3", "experiments/build_crash_audit_report.py"], out_dir, "crash_audit_report"))
    checks.append(run_cmd(["python3", "experiments/build_sqlite_recovery_oracle.py"], out_dir, "sqlite_recovery_oracle"))
    checks.append(run_cmd(["python3", "experiments/run_sqlite_fault_campaign.py"], out_dir, "sqlite_fault_campaign"))
    checks.append(run_cmd(["python3", "experiments/build_evidence_dashboard.py"], out_dir, "evidence_dashboard"))

    report = {
        "note": "SQLite app recovery bundle only; not a full multi-workload crash-certification claim.",
        "checks": checks,
    }
    (out_dir / "app_recovery_bundle.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"out_dir": str(out_dir), "checks": len(checks)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
