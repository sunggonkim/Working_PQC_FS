#!/usr/bin/env python3
"""Build an audit report for the retained crash/recovery evidence.

This script does not rerun any recovery workload. It simply aggregates the
checked-in summaries so the remaining recovery gap is easier to audit.
The report is intentionally conservative: it records what is retained and
states that power-loss/FUSE-daemon crash timing remains incomplete.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = ROOT / "artifacts" / "reports" / "crash_audit_report"


def load_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def summarize_verdict_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    out = []
    for row in rows:
        trials = int(row.get("trials", 0) or 0)
        fail_closed = int(row.get("fail_closed", 0) or 0)
        rollback_visible = int(row.get("rollback_visible", row.get("rollback_accept", 0)) or 0)
        unexpected = int(row.get("unexpected_error", 0) or 0)
        out.append({
            "backend": row.get("backend"),
            "cut_point_s": row.get("cut_point_s"),
            "trials": trials,
            "oracle_verdict_counts": {
                "fail_closed": fail_closed,
                "previous_committed": rollback_visible,
                "unexpected_liveness_failure": unexpected,
            },
            "fail_closed_rate": fail_closed / trials if trials else None,
        })
    return out


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
    sqlite_syscall_crash_path = ROOT / "artifacts" / "validation" / "sqlite_syscall_crash_tpm" / "sqlite_syscall_crash_tpm.json"
    sqlite_syscall_crash = load_json(sqlite_syscall_crash_path)

    report = {
        "source_artifacts": {
            "file_replay": "artifacts/validation/replay_file_summary.json",
            "file_replay_final": "artifacts/validation/replay_file_final_summary.json",
            "file_replay_after_keyfix": "artifacts/validation/replay_file_after_keyfix_summary.json",
            "e8_crash_replay": "artifacts/results/recovery/crash_replay_e8_test_summary.json",
            "tpm_unprovisioned": "artifacts/validation/tpm_unprovisioned.json",
            "hardware_anchor_latency": "artifacts/results/freshness/anchor_refresh/hardware_anchor_latency.json",
            "sqlite_probe": "artifacts/results/recovery/sqlite_strace.log",
            "sqlite_fault_campaign": "artifacts/validation/sqlite_fault_campaign/sqlite_fault_campaign.json",
            "sqlite_syscall_crash_tpm": "artifacts/validation/sqlite_syscall_crash_tpm/sqlite_syscall_crash_tpm.json",
        },
        "retained_evidence": {
            "file_replay": replay_file,
            "file_replay_final": replay_file_final,
            "file_replay_after_keyfix": {"rows": summarize_verdict_rows(replay_after_keyfix["rows"]), "skipped": replay_after_keyfix.get("skipped", [])},
            "e8_crash_replay": {"rows": summarize_verdict_rows(e8["rows"]), "skipped": e8.get("skipped", [])},
            "tpm_unprovisioned": tpm,
            "hardware_anchor_latency": hardware_anchor,
            "sqlite_probe_present": (ROOT / "artifacts" / "sqlite_strace.log").exists(),
            "sqlite_fault_campaign": sqlite_fault_campaign,
            "sqlite_syscall_crash_tpm": sqlite_syscall_crash,
        },
        "gap": {
            "app_level_recovery": "SQLite selected-boundary and syscall-exact app-crash timing exist; combined SQLite/dbm.dumb stale-snapshot campaigns are retained separately",
            "fault_injection_across_durable_boundaries": "power-loss and FUSE-daemon crash timing remain out of scope",
            "broader_workload_family": "dbm.dumb second-workload stale-snapshot evidence is retained; RocksDB remains unclaimed",
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
        f"- E8 crash/replay regression: {len(e8['rows'])} summarized rows, fail_closed={sum(row['fail_closed'] for row in e8['rows'])}, unexpected={sum(row['unexpected_error'] for row in e8['rows'])}",
        f"- TPM unprovisioned path: exit={tpm['hardware_anchor_without_preprovisioning_exit']}",
        f"- hardware anchor round-trip: median {hardware_anchor['rows'][0]['median_ms']:.6f} ms, p95 {hardware_anchor['rows'][0]['p95_ms']:.6f} ms",
        f"- sqlite probe present: {report['retained_evidence']['sqlite_probe_present']}",
        f"- sqlite fault campaign: {len(sqlite_fault_campaign['rows'])} trials, unacceptable={sum(1 for row in sqlite_fault_campaign['rows'] if not row['acceptable'])}",
        f"- sqlite syscall crash TPM campaign: {sqlite_syscall_crash['summary']['trial_count']} trials, unacceptable={sqlite_syscall_crash['summary']['unacceptable_trials']}",
        "",
        "## Current gap",
        "- app-level evidence includes selected-boundary SQLite replay and fdatasync-exact SQLite app-crash timing",
        "- power-loss and FUSE-daemon crash timing remain open",
        "",
    ]
    (args.out_dir / "crash_audit_report.md").write_text("\n".join(md), encoding="utf-8")

    print(json.dumps({"out_dir": str(args.out_dir), "sections": len(md)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
