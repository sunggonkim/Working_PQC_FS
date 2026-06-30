#!/usr/bin/env python3
"""Package the conservative app-recovery bundle into a report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
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
    syscall_campaign = ROOT / "artifacts" / "validation" / "sqlite_syscall_crash_tpm" / "sqlite_syscall_crash_tpm.json"
    combined_bundle = ROOT / "artifacts" / "validation" / "combined_durability_bundle" / "combined_durability_bundle.json"
    if sqlite_campaign.exists():
        campaign = json.loads(sqlite_campaign.read_text(encoding="utf-8"))
        unacceptable = sum(1 for row in campaign.get("rows", []) if not row.get("acceptable", False))
    else:
        campaign = None
        unacceptable = None
    if syscall_campaign.exists():
        syscall = json.loads(syscall_campaign.read_text(encoding="utf-8"))
        syscall_unacceptable = syscall.get("summary", {}).get("unacceptable_trials")
    else:
        syscall = None
        syscall_unacceptable = None
    if combined_bundle.exists():
        combined = json.loads(combined_bundle.read_text(encoding="utf-8"))
        sqlite_replay = ((combined.get("unified_campaign") or {}).get("replay") or {})
        dbm_replay = ((combined.get("unified_dbm_campaign") or {}).get("replay") or {})
    else:
        sqlite_replay = {}
        dbm_replay = {}
    report = {
        "note": "App recovery report covering SQLite selected-boundary replay, SQLite syscall-exact app-crash timing, and combined SQLite/dbm.dumb stale-snapshot replay. It is not power-loss or FUSE-daemon crash certification.",
        "bundle": bundle,
        "sqlite_fault_campaign": {
            "path": str(sqlite_campaign),
            "present": sqlite_campaign.exists(),
            "unacceptable": unacceptable,
        },
        "sqlite_syscall_crash_tpm": {
            "path": str(syscall_campaign),
            "present": syscall_campaign.exists(),
            "unacceptable": syscall_unacceptable,
        },
        "combined_durability_bundle": {
            "path": str(combined_bundle),
            "present": combined_bundle.exists(),
            "sqlite_replay_verdict": sqlite_replay.get("verdict"),
            "sqlite_replay_acceptable": sqlite_replay.get("acceptable"),
            "dbm_replay_verdict": dbm_replay.get("verdict"),
            "dbm_replay_acceptable": dbm_replay.get("acceptable"),
        },
    }
    (args.out_dir / "app_recovery_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    md = [
        "# App recovery report",
        "",
        "This report packages retained crash audit outputs, the executable SQLite oracle campaign, syscall-exact SQLite app-crash timing, and combined SQLite/dbm.dumb stale-snapshot replay.",
        "",
        f"- Input directory: `{args.in_dir}`",
        f"- Checks: `{len(bundle.get('checks', []))}`",
        f"- Bundle present: `{bundle_path.exists()}`",
        "",
        f"- SQLite fault campaign present: `{sqlite_campaign.exists()}`",
        f"- SQLite unacceptable oracle verdicts: `{unacceptable}`",
        f"- SQLite syscall crash campaign present: `{syscall_campaign.exists()}`",
        f"- SQLite syscall crash unacceptable verdicts: `{syscall_unacceptable}`",
        f"- Combined SQLite replay: `{sqlite_replay.get('verdict')}` / acceptable `{sqlite_replay.get('acceptable')}`",
        f"- Combined dbm.dumb replay: `{dbm_replay.get('verdict')}` / acceptable `{dbm_replay.get('acceptable')}`",
        "",
        "This report does not claim power-loss or FUSE-daemon crash certification.",
    ]
    (args.out_dir / "app_recovery_report.md").write_text("\n".join(md), encoding="utf-8")
    print(json.dumps({"out_dir": str(args.out_dir)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
