#!/usr/bin/env python3
"""Build an audit report for the retained crash/recovery evidence.

This script does not rerun any recovery workload. It simply aggregates the
checked-in summaries so the remaining recovery gap is easier to audit.
The report is intentionally conservative: it records what is retained and
states that multi-workload app-level recovery remains incomplete.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "artifacts" / "crash_audit_report"


def load_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    replay_file = load_json(ROOT / "artifacts" / "validation" / "replay_file_summary.json")
    replay_file_final = load_json(ROOT / "artifacts" / "validation" / "replay_file_final_summary.json")
    replay_after_keyfix = load_json(ROOT / "artifacts" / "validation" / "replay_file_after_keyfix_summary.json")
    e8 = load_json(ROOT / "artifacts" / "crash_replay_e8_test_summary.json")
    tpm = load_json(ROOT / "artifacts" / "validation" / "tpm_unprovisioned.json")
    hardware_anchor = load_json(ROOT / "artifacts" / "anchor_refresh" / "hardware_anchor_latency.json")
    sqlite_fault_campaign_path = ROOT / "artifacts" / "validation" / "sqlite_fault_campaign" / "sqlite_fault_campaign.json"
    sqlite_fault_campaign = load_json(sqlite_fault_campaign_path)

    report = {
        "source_artifacts": {
            "file_replay": "artifacts/validation/replay_file_summary.json",
            "file_replay_final": "artifacts/validation/replay_file_final_summary.json",
            "file_replay_after_keyfix": "artifacts/validation/replay_file_after_keyfix_summary.json",
            "e8_crash_replay": "artifacts/crash_replay_e8_test_summary.json",
            "tpm_unprovisioned": "artifacts/validation/tpm_unprovisioned.json",
            "hardware_anchor_latency": "artifacts/anchor_refresh/hardware_anchor_latency.json",
            "sqlite_probe": "artifacts/sqlite_strace.log",
            "sqlite_fault_campaign": "artifacts/validation/sqlite_fault_campaign/sqlite_fault_campaign.json",
        },
        "retained_evidence": {
            "file_replay": replay_file,
            "file_replay_final": replay_file_final,
            "file_replay_after_keyfix": replay_after_keyfix,
            "e8_crash_replay": e8,
            "tpm_unprovisioned": tpm,
            "hardware_anchor_latency": hardware_anchor,
            "sqlite_probe_present": (ROOT / "artifacts" / "sqlite_strace.log").exists(),
            "sqlite_fault_campaign": sqlite_fault_campaign,
        },
        "gap": {
            "app_level_recovery": "SQLite-only selected-boundary campaign exists; full multi-workload recovery remains open",
            "fault_injection_across_durable_boundaries": "SQLite selected-boundary campaign exists; syscall-exact crash timing remains out of scope",
            "broader_workload_family": "second application workload remains open",
        },
    }

    (args.out_dir / "crash_audit_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    md = [
        "# Crash / recovery audit report",
        "",
        "This report aggregates the retained crash and freshness evidence without claiming full app-level recovery.",
        "",
        "## Retained evidence",
        f"- file replay negative control: {len(replay_file['rows'])} cut-point rows",
        f"- file replay final negative control: {len(replay_file_final['rows'])} cut-point rows",
        f"- post-keyfix replay summary: {len(replay_after_keyfix['rows'])} cut-point rows",
        f"- E8 crash/replay regression: {len(e8['rows'])} summarized rows, max success_rate={max(row['success_rate'] for row in e8['rows']):.2f}, min success_rate={min(row['success_rate'] for row in e8['rows']):.2f}",
        f"- TPM unprovisioned path: exit={tpm['hardware_anchor_without_preprovisioning_exit']}",
        f"- hardware anchor round-trip: median {hardware_anchor['rows'][0]['median_ms']:.6f} ms, p95 {hardware_anchor['rows'][0]['p95_ms']:.6f} ms",
        f"- sqlite probe present: {report['retained_evidence']['sqlite_probe_present']}",
        f"- sqlite fault campaign: {len(sqlite_fault_campaign['rows'])} trials, unacceptable={sum(1 for row in sqlite_fault_campaign['rows'] if not row['acceptable'])}",
        "",
        "## Current gap",
        "- app-level recovery remains open because the retained campaign is SQLite-only",
        "- syscall-exact crash timing and a second application workload remain open",
        "",
    ]
    (args.out_dir / "crash_audit_report.md").write_text("\n".join(md), encoding="utf-8")

    print(json.dumps({"out_dir": str(args.out_dir), "sections": len(md)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
