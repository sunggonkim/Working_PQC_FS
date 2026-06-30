#!/usr/bin/env python3
"""Build the Gate 0.15 concurrency-contract closure artifact.

The smoke runner captures raw mounted-path timing.  This builder turns that
evidence into the parent-gate contract: every observed hot lock needs measured
wait/hold percentiles, ownership metadata, stress coverage, blocking-syscall
evidence, and a negative claim guard.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "artifacts" / "validation" / "concurrency_contract"
SUMMARY = OUT / "lock_profile_summary.json"
LOCK_INVENTORY = ROOT / "artifacts" / "validation" / "refactor_inventory" / "lock_inventory.json"
CONTRACT_JSON = OUT / "concurrency_contract.json"
CONTRACT_MD = OUT / "concurrency_contract.md"

PAPER_FILES = [
    ROOT / "Paper" / "1_Introduction.tex",
    ROOT / "Paper" / "3_Design.tex",
    ROOT / "Paper" / "4_Evaluation.tex",
    ROOT / "Paper" / "10_Discussion_and_Limitations.tex",
    ROOT / "Paper" / "6_Conclusion.tex",
    ROOT / "README.md",
]

FORBIDDEN_CLAIM_PATTERNS = (
    r"\bscales?\s+(linearly|with cores|with clients)",
    r"\bmulticore\s+scalability\b",
    r"\bconcurrent-client\s+scalability\b",
    r"\bparallel[- ]commit\s+(speedup|scalability|guarantee)",
    r"\bfine-grained\s+locking\s+(guarantees|proves|ensures)",
)

PROFILE_TO_INVENTORY_LOCK = {
    "admission_state_lock": "g_admission.state_lock",
    "admission_trace_lock": "g_admission.trace_lock",
    "anchor_epoch_record_lock": "g_epoch_record_lock",
    "anchor_lifecycle_lock": "g_anchor_lifecycle_lock",
    "anchor_worker_lock": "g_anchor_lock",
    "committed_map_lock": "g_committed_lock",
    "file_anchor_commit_lock": "g_file_anchor_commit_lock",
    "file_state_table_lock": "g_file_state_table_lock",
    "parallel_runtime_lock": "g_runtime_lock",
    "qos_gpu_load_lock": "g_gpu_load_lock",
    "qos_throttle_lock": "g_qos_throttle_lock",
    "rekey_lifecycle_lock": "g_rekey_lifecycle_lock",
    "rekey_queue_lock": "g_rekey_queue.lock",
    "scheduler_lock": "g_sched_lock",
    "trace_sink_lock": "sink.lock",
}


def now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def read_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def inventory_by_name() -> dict[str, dict[str, Any]]:
    inventory = read_json(LOCK_INVENTORY)
    locks = inventory.get("locks")
    if not isinstance(locks, list):
        locks = []
    return {str(item.get("name")): item for item in locks if isinstance(item, dict) and item.get("name")}


def aggregate_lock_rows(summary: dict[str, Any], inventory: dict[str, dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_site = summary.get("lock_profile", {}).get("by_lock_and_site", {})
    if not isinstance(by_site, dict):
        by_site = {}
    grouped: dict[str, list[tuple[str, dict[str, Any]]]] = {}
    for key, stats in by_site.items():
        lock, _, site = str(key).partition("@")
        grouped.setdefault(lock, []).append((site or "unknown", stats if isinstance(stats, dict) else {}))

    rows: list[dict[str, Any]] = []
    blockers: list[dict[str, Any]] = []
    for lock in sorted(grouped):
        sites = grouped[lock]
        inventory_name = PROFILE_TO_INVENTORY_LOCK.get(lock, lock)
        inv = inventory.get(inventory_name, {})
        missing = []
        if not inv.get("owner_module"):
            missing.append("owner_module")
        if not inv.get("protected_state"):
            missing.append("protected_state")
        if not inv.get("deadlock_order"):
            missing.append("deadlock_order")
        total_count = 0
        max_hold_p99 = 0.0
        max_wait_p99 = 0.0
        call_sites: list[str] = []
        measured_sites: list[dict[str, Any]] = []
        for site, stats in sites:
            hold = stats.get("hold_ns", {}) if isinstance(stats, dict) else {}
            wait = stats.get("wait_ns", {}) if isinstance(stats, dict) else {}
            if not all(key in hold for key in ("p50", "p95", "p99")):
                missing.append(f"{site}:hold_percentiles")
            if "p99" not in wait:
                missing.append(f"{site}:wait_p99")
            count = int(stats.get("count", 0) or 0)
            total_count += count
            max_hold_p99 = max(max_hold_p99, float(hold.get("p99", 0.0) or 0.0))
            max_wait_p99 = max(max_wait_p99, float(wait.get("p99", 0.0) or 0.0))
            call_sites.append(site)
            measured_sites.append(
                {
                    "site": site,
                    "count": count,
                    "hold_ns": hold,
                    "wait_ns": wait,
                }
            )
        row = {
            "lock": lock,
            "inventory_lock": inventory_name,
            "owner_module": inv.get("owner_module", "unknown"),
            "protected_state": inv.get("protected_state", "unknown"),
            "allowed_call_sites": sorted(set(call_sites)),
            "forbidden_blocking_boundary": inv.get(
                "deadlock_order",
                "must not hold this lock across crypto, CUDA, blocking I/O, fdatasync, TPM access, or user-visible waits",
            ),
            "sample_count": total_count,
            "max_hold_p99_ns": max_hold_p99,
            "max_wait_p99_ns": max_wait_p99,
            "measured_sites": measured_sites,
            "complete": total_count > 0 and not missing,
            "missing": sorted(set(missing)),
        }
        rows.append(row)
        if not row["complete"]:
            blockers.append({"lock": lock, "missing": row["missing"]})
    return rows, blockers


def scan_forbidden_claims() -> list[dict[str, Any]]:
    hits: list[dict[str, Any]] = []
    regexes = [re.compile(pattern, re.IGNORECASE) for pattern in FORBIDDEN_CLAIM_PATTERNS]
    for path in PAPER_FILES:
        for line_no, line in enumerate(read_text(path).splitlines(), 1):
            for regex in regexes:
                if regex.search(line):
                    hits.append(
                        {
                            "path": rel(path),
                            "line": line_no,
                            "pattern": regex.pattern,
                            "text": line.strip()[:240],
                        }
                    )
    return hits


def render_md(payload: dict[str, Any]) -> str:
    lines = [
        "# Concurrency Contract",
        "",
        f"- Generated: `{payload['generated_utc']}`",
        f"- Overall pass: `{str(payload['overall_pass']).lower()}`",
        f"- Observed hot locks: `{payload['observed_hot_lock_count']}`",
        f"- Blocking profile pass: `{str(payload['stress_evidence']['blocking_syscall_profile_pass']).lower()}`",
        f"- Deadlock/livelock negative pass: `{str(payload['stress_evidence']['deadlock_livelock_negative_pass']).lower()}`",
        f"- Forbidden claim hits: `{len(payload['forbidden_claim_hits'])}`",
        "",
        "| Lock | Owner | Samples | max hold p99 ns | max wait p99 ns | Complete |",
        "| --- | --- | ---: | ---: | ---: | --- |",
    ]
    for row in payload["hot_locks"]:
        lines.append(
            f"| `{row['lock']}` | `{row['owner_module']}` | {row['sample_count']} | "
            f"{row['max_hold_p99_ns']:.0f} | {row['max_wait_p99_ns']:.0f} | "
            f"`{str(row['complete']).lower()}` |"
        )
    if payload["blocking_items"]:
        lines.extend(["", "## Blocking Items", ""])
        for item in payload["blocking_items"]:
            lines.append(f"- `{item.get('lock', item.get('category', 'unknown'))}`: `{item}`")
    return "\n".join(lines) + "\n"


def main() -> int:
    summary = read_json(SUMMARY)
    inventory = inventory_by_name()
    hot_locks, lock_blockers = aggregate_lock_rows(summary, inventory)
    stress = {
        "lock_profile_summary_pass": summary.get("overall_pass") is True,
        "thread_counts": summary.get("thread_counts", []),
        "max_thread_count": summary.get("workload", {}).get("coverage", {}).get("max_thread_count"),
        "max_process_client_count": summary.get("workload", {}).get("coverage", {}).get("max_process_client_count"),
        "same_file_phases": summary.get("workload", {}).get("coverage", {}).get("same_file_writer_phases"),
        "disjoint_file_phases": summary.get("workload", {}).get("coverage", {}).get("disjoint_writer_phases"),
        "same_file_process_phases": summary.get("workload", {}).get("coverage", {}).get("same_file_process_lifecycle_phases"),
        "disjoint_file_process_phases": summary.get("workload", {}).get("coverage", {}).get("disjoint_process_lifecycle_phases"),
        "blocking_syscall_profile_pass": summary.get("blocking_syscall_profile", {}).get("overall_pass") is True,
        "reader_visibility_pass": summary.get("reader_visibility_probe", {}).get("overall_pass") is True,
        "deadlock_livelock_negative_pass": (
            summary.get("deadlock_livelock_negative", {}).get("timed_out_phase_count") == 0
            and summary.get("deadlock_livelock_negative", {}).get("worker_error_count") == 0
            and summary.get("deadlock_livelock_negative", {}).get("lifecycle_coverage_pass") is True
            and summary.get("deadlock_livelock_negative", {}).get("process_client_coverage_pass") is True
        ),
    }
    stress_blockers: list[dict[str, Any]] = []
    if not stress["lock_profile_summary_pass"]:
        stress_blockers.append({"category": "stress", "missing": "lock_profile_summary_pass"})
    if not stress["blocking_syscall_profile_pass"]:
        stress_blockers.append({"category": "stress", "missing": "blocking_syscall_profile_pass"})
    if not stress["reader_visibility_pass"]:
        stress_blockers.append({"category": "stress", "missing": "reader_visibility_pass"})
    if not stress["deadlock_livelock_negative_pass"]:
        stress_blockers.append({"category": "stress", "missing": "deadlock_livelock_negative_pass"})
    if (stress["max_thread_count"] or 0) < 4 or (stress["max_process_client_count"] or 0) < 4:
        stress_blockers.append({"category": "stress", "missing": "thread/process client count >= 4"})
    claim_hits = scan_forbidden_claims()
    blocking_items = lock_blockers + stress_blockers
    if claim_hits:
        blocking_items.append({"category": "claim_guard", "missing": "remove or qualify forbidden broad concurrency claims"})
    payload = {
        "schema_version": 1,
        "generated_utc": now_utc(),
        "overall_pass": not blocking_items and bool(hot_locks),
        "source_artifacts": {
            "lock_profile_summary": rel(SUMMARY),
            "lock_inventory": rel(LOCK_INVENTORY),
            "blocking_syscall_profile": summary.get("artifacts", {}).get("blocking_syscall_profile"),
            "reader_visibility_probe": summary.get("artifacts", {}).get("reader_visibility_probe"),
            "lock_order_table": summary.get("artifacts", {}).get("lock_order_table"),
        },
        "observed_hot_lock_count": len(hot_locks),
        "hot_locks": hot_locks,
        "stress_evidence": stress,
        "forbidden_claim_hits": claim_hits,
        "blocking_items": blocking_items,
        "negative_claim_guard": (
            "This contract supports measured mounted-path lock hold/wait boundaries and bounded deadlock/livelock smoke. "
            "It does not justify broad multicore scalability, deployment readiness, or general concurrent-client claims."
        ),
        "parent_0_15_closed": not blocking_items and bool(hot_locks),
    }
    CONTRACT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    CONTRACT_MD.write_text(render_md(payload), encoding="utf-8")
    print(
        json.dumps(
            {
                "overall_pass": payload["overall_pass"],
                "observed_hot_lock_count": payload["observed_hot_lock_count"],
                "blocking_items": payload["blocking_items"],
                "outputs": [rel(CONTRACT_JSON), rel(CONTRACT_MD)],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if payload["overall_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
