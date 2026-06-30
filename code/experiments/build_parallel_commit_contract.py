#!/usr/bin/env python3
"""Build the Gate 0.16 parallel-commit contract from current code evidence.

This is not a benchmark.  It records the current strict/epoch publication
topology, imports the latest mounted-path concurrency evidence when available,
and closes Gate 0.16 only when code, artifacts, paper text, and negative claim
guards are all present.
"""

from __future__ import annotations

import json
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
CODE = ROOT / "code"
PAPER = ROOT / "Paper"
README = ROOT / "README.md"
OUT = ROOT / "artifacts" / "validation" / "parallel_commit_contract"
CONCURRENCY_SUMMARY = (
    ROOT / "artifacts" / "validation" / "concurrency_contract" /
    "lock_profile_summary.json"
)
LOCK_TRACE = (
    ROOT / "artifacts" / "validation" / "concurrency_contract" /
    "lock_profile_trace.jsonl"
)
EPOCH_PATH_SMOKE = OUT / "epoch_path_smoke.json"
TELEMETRY_SWEEP = OUT / "parallel_commit_telemetry_sweep.json"
FAIRNESS_REPLAY = OUT / "parallel_commit_fairness_replay.json"
EPOCH_REPLAY_FAULT_MATRIX = (
    ROOT / "artifacts" / "validation" /
    "publication_protocol_fault_matrix" / "epoch_replay_fault_matrix.json"
)
EPOCH_PUBLICATION_COMPARISON = (
    ROOT / "artifacts" / "validation" /
    "publication_protocol_fault_matrix" / "epoch_publication_comparison.json"
)
CLAIM_SCAN_TARGETS = [PAPER, README]
CLAIM_SCAN_PHRASES = [
    "sharded queues",
    "parallel commit",
    "epoch fdatasync",
    "group commit",
    "scalability from commit batching",
]

CLAIM_SCAN_ALLOWED_SCOPE_TERMS = (
    "informs",
    "vocabulary",
    "implemented analogue",
    "scoped",
    "bounded leader",
    "bounded lead",
    "single-client",
    "loss case",
    "non-claim",
    "not claim",
    "not a",
    "future",
    "related work",
)


def now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def load_json(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def iter_text_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    if not path.exists():
        return []
    return [
        candidate for candidate in sorted(path.rglob("*"))
        if candidate.is_file() and candidate.suffix.lower() in {
            ".md", ".tex", ".bib", ".txt"
        }
    ]


def percentile(values: list[int], pct: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    rank = (len(ordered) - 1) * pct
    lo = int(rank)
    hi = min(lo + 1, len(ordered) - 1)
    weight = rank - lo
    return float(ordered[lo] * (1.0 - weight) + ordered[hi] * weight)


def summarize_values(values: list[int]) -> dict[str, Any]:
    if not values:
        return {"count": 0}
    return {
        "count": len(values),
        "min_ns": min(values),
        "max_ns": max(values),
        "mean_ns": statistics.fmean(values),
        "p50_ns": percentile(values, 0.50),
        "p95_ns": percentile(values, 0.95),
        "p99_ns": percentile(values, 0.99),
    }


def summarize_lock_trace(path: Path) -> dict[str, Any]:
    by_lock: dict[str, list[int]] = defaultdict(list)
    by_site: dict[str, list[int]] = defaultdict(list)
    wait_by_lock: dict[str, list[int]] = defaultdict(list)
    malformed = 0
    line_count = 0
    if not path.exists():
        return {
            "available": False,
            "path": str(path.relative_to(ROOT)),
            "reason": "lock profile trace is missing",
        }
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            line_count += 1
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                malformed += 1
                continue
            if event.get("event") != "lock_hold":
                continue
            lock = str(event.get("lock", "unknown"))
            site = str(event.get("site", "unknown"))
            hold_ns = int(event.get("hold_ns", 0))
            wait_ns = int(event.get("wait_ns", 0))
            by_lock[lock].append(hold_ns)
            by_site[f"{lock}@{site}"].append(hold_ns)
            wait_by_lock[lock].append(wait_ns)
    return {
        "available": True,
        "path": str(path.relative_to(ROOT)),
        "line_count": line_count,
        "malformed_line_count": malformed,
        "by_lock_hold_ns": {
            lock: summarize_values(values) for lock, values in sorted(by_lock.items())
        },
        "by_lock_wait_ns": {
            lock: summarize_values(values)
            for lock, values in sorted(wait_by_lock.items())
        },
        "top_sites_by_count": [
            {"site": site, **summarize_values(values)}
            for site, values in sorted(
                by_site.items(), key=lambda item: len(item[1]), reverse=True
            )[:12]
        ],
    }


def source_evidence() -> dict[str, Any]:
    fs = CODE / "fs"
    storage = CODE / "storage"
    common = CODE / "common"
    files = {
        "writeback": read_text(storage / "pqc_writeback.c"),
        "strict_publish": read_text(storage / "pqc_strict_publish.c"),
        "journal": read_text(storage / "pqc_journal.c"),
        "state": read_text(storage / "pqc_state.h"),
        "fd_context": read_text(fs / "pqc_fd_context.c"),
        "file_io": read_text(fs / "pqc_file_io.c"),
        "flush_batch": read_text(storage / "pqc_flush_batch.c"),
        "parallel_commit": (
            read_text(fs / "pqc_parallel_commit.c") + "\n" +
            read_text(fs / "pqc_parallel_commit.h")
        ),
        "epoch_publish": (
            read_text(storage / "pqc_epoch_publish.c") + "\n" +
            read_text(storage / "pqc_epoch_publish.h")
        ),
        "epoch_log": (
            read_text(storage / "pqc_epoch_log.c") + "\n" +
            read_text(storage / "pqc_epoch_log.h")
        ),
    }
    all_text = "\n".join(files.values())
    source_files = [
        p.name for root in (fs, storage, common)
        for p in root.glob("pqc_*.*")
    ]
    queue_like = [
        name for name in source_files
        if any(token in name for token in ("queue", "epoch", "commit"))
    ]
    return {
        "strict_publish_commit_exists": "pqc_strict_publish_commit" in files["strict_publish"],
        "strict_publish_data_fdatasync": "fdatasync(req->data_fd)" in files["strict_publish"],
        "strict_publish_journal_fdatasync": "fdatasync(req->journal_fd)" in files["strict_publish"],
        "strict_publish_checkpoint_stage": "pqc_checkpoint_store_and_stage_anchor" in files["strict_publish"],
        "writeback_has_publish_turn": "publish_turn" in files["writeback"],
        "writeback_reserves_generation": "reserved_generation" in files["writeback"],
        "writeback_commits_generation_after_publish": "committed_generation" in files["writeback"],
        "journal_append_is_per_mapping": "pqc_journal_append_mapping_unsynced" in files["strict_publish"],
        "journal_append_function_exists": "pqc_journal_append_mapping_unsynced" in files["journal"],
        "journal_legacy_synced_append_exists": (
            "pqc_journal_append_mapping(" in files["journal"] and
            "return fdatasync(journal_fd)" in files["journal"]
        ),
        "file_state_has_single_publish_ticket": "next_publish_ticket" in files["state"],
        "file_state_has_current_publish_ticket": "publish_ticket" in files["state"],
        "fd_context_dirty_sidecar_dedup_exists": "pqc_fd_context_prepare_dirty_sidecar_sync_locked" in files["fd_context"],
        "flush_batch_can_cover_multiple_blocks": "block_count" in files["flush_batch"],
        "parallel_commit_module_exists": (fs / "pqc_parallel_commit.c").exists(),
        "parallel_commit_has_shard_assignment": "pqc_parallel_commit_shard_for_file" in files["parallel_commit"],
        "parallel_commit_has_leader_follower_roles": (
            "PQC_PARALLEL_COMMIT_ROLE_LEADER" in files["parallel_commit"] and
            "PQC_PARALLEL_COMMIT_ROLE_FOLLOWER" in files["parallel_commit"]
        ),
        "parallel_commit_has_bounded_wait": "pthread_cond_timedwait" in files["parallel_commit"],
        "parallel_commit_has_finish_wakeup": "pqc_parallel_commit_finish" in files["parallel_commit"],
        "parallel_commit_has_stats_snapshot": "pqc_parallel_commit_stats_snapshot" in files["parallel_commit"],
        "parallel_commit_trace_has_wait_telemetry": "wait_ns" in files["parallel_commit"],
        "parallel_commit_trace_has_queue_depth_telemetry": "queue_depth" in files["parallel_commit"],
        "parallel_commit_telemetry_sweep_runner_exists": (
            CODE / "experiments" / "run_parallel_commit_telemetry_sweep.py"
        ).exists(),
        "parallel_commit_fairness_replay_runner_exists": (
            CODE / "experiments" / "run_parallel_commit_fairness_replay.py"
        ).exists(),
        "writeback_calls_parallel_commit": (
            "pqc_parallel_commit_begin" in files["writeback"] or
            "pqc_parallel_commit_runtime_begin" in files["writeback"]
        ),
        "epoch_publish_module_exists": (storage / "pqc_epoch_publish.c").exists(),
        "epoch_publish_sets_skip_journal_fsync": (
            "skip_journal_fsync = 1" in files["epoch_publish"] and
            "journal_fsync_count" in files["epoch_publish"]
        ),
        "epoch_publish_has_group_barrier": (
            "pqc_epoch_group_barrier_wait" in files["epoch_publish"] and
            "PQC_EPOCH_GROUP_MAX" in files["epoch_publish"]
        ),
        "epoch_publish_uses_syncfs_for_group_barrier": (
            "SYS_syncfs" in files["epoch_publish"] and
            "sync_primitive" in files["epoch_publish"]
        ),
        "epoch_log_replay_parser_exists": "pqc_epoch_log_replay_fd" in files["epoch_log"],
        "epoch_log_checkpoint_compaction_exists": "pqc_epoch_log_compact_checkpoint" in files["epoch_log"],
        "epoch_log_journal_repair_exists": "epoch_log_repair_journal_prefix" in files["epoch_log"],
        "epoch_log_duplicate_generation_reject_exists": "duplicate_generation_records" in files["epoch_log"],
        "shard_or_queue_source_files": sorted(queue_like),
        "mentions_group_commit": "group commit" in all_text.lower(),
        "mentions_sharded_commit": "shard" in all_text.lower() and "commit" in all_text.lower(),
        "mentions_leader_waiter": "leader" in all_text.lower() and "waiter" in all_text.lower(),
    }


def classify_current_topology(evidence: dict[str, Any]) -> dict[str, Any]:
    if evidence["parallel_commit_module_exists"] or evidence["epoch_publish_module_exists"]:
        topology = "partial_parallel_or_epoch_source_present"
    elif evidence["writeback_has_publish_turn"] and evidence["strict_publish_commit_exists"]:
        topology = "strict_per_file_publish_turn"
    elif evidence["strict_publish_commit_exists"]:
        topology = "strict_synchronous_publish"
    else:
        topology = "unknown"

    return {
        "name": topology,
        "implemented": topology in {
            "strict_per_file_publish_turn",
            "strict_synchronous_publish",
            "partial_parallel_or_epoch_source_present",
        },
        "paper_eligible_as_parallel_commit": False,
        "observed_properties": [
            "writeback may batch multiple logical blocks from one fd flush",
            "strict publication issues one data-sidecar fdatasync and one journal-sidecar fdatasync per flush batch",
            "per-file publish turns preserve reader visibility and generation order",
            "no cross-open, cross-file, or cross-shard group commit module is present",
        ],
        "negative_claim_guard": (
            "Do not describe the current strict publish path as sharded queues, "
            "parallel commit, epoch mode, or group commit."
        ),
    }


def design_candidates() -> list[dict[str, Any]]:
    return [
        {
            "name": "strict_per_file_publish_turn",
            "status": "implemented-baseline",
            "shard_assignment": "one publication order per backing file state",
            "leader_waiter_protocol": "condition-variable publish ticket, one active publisher",
            "fairness_policy": "FIFO within one file only",
            "recovery_ordering_rule": "reader-visible generation advances only after data and journal barriers",
            "expected_win": "correctness baseline and comparison point",
            "expected_loss": "does not amortize fdatasync across files or clients",
        },
        {
            "name": "per_file_epoch_log",
            "status": "candidate-not-implemented",
            "shard_assignment": "one epoch queue per backing file state",
            "leader_waiter_protocol": "first waiter becomes epoch leader; followers join until size or age threshold",
            "fairness_policy": "bounded epoch age plus foreground bypass for latency-sensitive fsync",
            "recovery_ordering_rule": "redo records replay in epoch order; checkpoint records only committed prefix",
            "required_code": [
                "pqc_epoch_publish.[ch]",
                "per-file pending request queue",
                "epoch age and byte thresholds",
                "strict fallback mode flag",
                "fault cutpoints for enqueue, log append, epoch barrier, checkpoint, and replay",
            ],
        },
        {
            "name": "hash_sharded_epoch_log",
            "status": "candidate-not-implemented",
            "shard_assignment": "hash(file_id) modulo shard_count",
            "leader_waiter_protocol": "one leader per shard epoch",
            "fairness_policy": "per-shard FIFO plus starvation counter for cold shards",
            "recovery_ordering_rule": "replay is independent across file_id; global freshness anchor records committed shard vector",
            "required_code": [
                "pqc_parallel_commit.[ch]",
                "shard table and work queues",
                "committed shard vector in checkpoint/freshness path",
                "fairness and starvation telemetry",
            ],
        },
        {
            "name": "global_epoch_log",
            "status": "rejected-baseline-unless-proven",
            "shard_assignment": "single global queue",
            "leader_waiter_protocol": "single epoch leader",
            "fairness_policy": "global FIFO",
            "recovery_ordering_rule": "single total redo order",
            "expected_win": "simplest recovery proof",
            "expected_loss": "likely replaces per-write fdatasync with one global serialization point",
        },
        {
            "name": "per_directory_epoch_log",
            "status": "deferred",
            "shard_assignment": "directory-local queue",
            "leader_waiter_protocol": "leader per directory",
            "fairness_policy": "directory FIFO",
            "recovery_ordering_rule": "requires stronger rename and directory-fsync semantics first",
            "blocked_by": "rename and directory fsync are not production-supported general POSIX claims",
        },
    ]


def baseline_evidence() -> dict[str, Any]:
    summary = load_json(CONCURRENCY_SUMMARY)
    if not summary:
        return {
            "available": False,
            "summary_path": str(CONCURRENCY_SUMMARY.relative_to(ROOT)),
            "reason": "run experiments/run_concurrency_contract_smoke.py first",
        }
    return {
        "available": True,
        "summary_path": str(CONCURRENCY_SUMMARY.relative_to(ROOT)),
        "overall_pass": summary.get("overall_pass"),
        "thread_counts": summary.get("thread_counts"),
        "workload_coverage": summary.get("workload", {}).get("coverage", {}),
        "deadlock_livelock_negative": summary.get("deadlock_livelock_negative", {}),
        "blocking_syscall_profile": {
            "overall_pass": summary.get("blocking_syscall_profile", {}).get("overall_pass"),
            "by_syscall": summary.get("blocking_syscall_profile", {})
            .get("summary", {})
            .get("by_syscall", {}),
            "path": summary.get("artifacts", {}).get("blocking_syscall_profile"),
        },
        "lock_profile_summary": {
            "observed_locks": summary.get("lock_profile", {}).get("observed_locks", []),
            "lock_hold_event_count": summary.get("lock_profile", {}).get("lock_hold_event_count"),
            "by_lock_and_site": summary.get("lock_profile", {}).get("by_lock_and_site", {}),
        },
        "non_closure": summary.get("not_closed", []),
    }


def epoch_path_smoke_evidence() -> dict[str, Any]:
    payload = load_json(EPOCH_PATH_SMOKE)
    if not payload:
        return {
            "available": False,
            "path": str(EPOCH_PATH_SMOKE.relative_to(ROOT)),
            "reason": "run experiments/run_parallel_commit_epoch_path_smoke.py first",
        }
    cases = {
        str(case.get("mode")): case for case in payload.get("cases", [])
        if isinstance(case, dict)
    }
    epoch = cases.get("epoch_gated_strict", {})
    strict = cases.get("strict", {})
    return {
        "available": True,
        "path": str(EPOCH_PATH_SMOKE.relative_to(ROOT)),
        "overall_pass": payload.get("overall_pass"),
        "strict_pass": strict.get("pass"),
        "epoch_gated_strict_pass": epoch.get("pass"),
        "epoch_begin_count": epoch.get("trace", {}).get("begin_count"),
        "epoch_finish_count": epoch.get("trace", {}).get("finish_count"),
        "epoch_roles": epoch.get("trace", {}).get("roles", []),
        "negative_claim_guard": payload.get("negative_claim_guard"),
    }


def telemetry_sweep_evidence() -> dict[str, Any]:
    payload = load_json(TELEMETRY_SWEEP)
    if not payload:
        return {
            "available": False,
            "path": str(TELEMETRY_SWEEP.relative_to(ROOT)),
            "reason": "run experiments/run_parallel_commit_telemetry_sweep.py first",
        }
    case_summaries = []
    for case in payload.get("cases", []):
        if not isinstance(case, dict):
            continue
        trace = case.get("trace", {})
        config = case.get("config", {})
        case_summaries.append({
            "name": case.get("name"),
            "pass": case.get("pass"),
            "clients": config.get("clients"),
            "shards": config.get("shards"),
            "group_max": config.get("group_max"),
            "wait_ns": config.get("wait_ns"),
            "begin_count": trace.get("begin_count"),
            "finish_count": trace.get("finish_count"),
            "group_size_distribution": trace.get("group_size_distribution"),
            "queue_depth_distribution": trace.get("queue_depth_distribution"),
            "wait_ns_summary": trace.get("wait_ns_summary"),
            "observed_shards": trace.get("observed_shards"),
            "trace_path": trace.get("path"),
        })
    return {
        "available": True,
        "path": str(TELEMETRY_SWEEP.relative_to(ROOT)),
        "overall_pass": payload.get("overall_pass"),
        "coverage": payload.get("coverage"),
        "case_count": len(case_summaries),
        "case_summaries": case_summaries,
        "negative_claim_guard": payload.get("negative_claim_guard"),
    }


def fairness_replay_evidence() -> dict[str, Any]:
    payload = load_json(FAIRNESS_REPLAY)
    if not payload:
        return {
            "available": False,
            "path": str(FAIRNESS_REPLAY.relative_to(ROOT)),
            "reason": "run experiments/run_parallel_commit_fairness_replay.py first",
        }
    case_summaries = []
    for case in payload.get("cases", []):
        if not isinstance(case, dict):
            continue
        fairness = case.get("fairness", {})
        replay = case.get("replay_order", {})
        config = case.get("config", {})
        case_summaries.append({
            "name": case.get("name"),
            "pass": case.get("pass"),
            "purpose": case.get("purpose"),
            "clients": config.get("clients"),
            "shards": config.get("shards"),
            "group_max": config.get("group_max"),
            "completed_ops": case.get("completed_ops"),
            "expected_ops": case.get("expected_ops"),
            "starvation_negative_pass": case.get("starvation_negative_pass"),
            "replay_order_pass": case.get("replay_order_pass"),
            "op_count_distribution": fairness.get("op_count_distribution"),
            "replay_plan_length": replay.get("replay_plan_length"),
            "reconstruction_time_ns": replay.get("reconstruction_time_ns"),
            "observed_shards": replay.get("observed_shards"),
            "trace_path": replay.get("trace_path"),
        })
    return {
        "available": True,
        "path": str(FAIRNESS_REPLAY.relative_to(ROOT)),
        "overall_pass": payload.get("overall_pass"),
        "coverage": payload.get("coverage"),
        "case_count": len(case_summaries),
        "case_summaries": case_summaries,
        "negative_claim_guard": payload.get("negative_claim_guard"),
    }


def epoch_replay_fault_matrix_evidence() -> dict[str, Any]:
    payload = load_json(EPOCH_REPLAY_FAULT_MATRIX)
    if not payload:
        return {
            "available": False,
            "path": str(EPOCH_REPLAY_FAULT_MATRIX.relative_to(ROOT)),
            "reason": "run experiments/run_epoch_replay_fault_matrix.py first",
        }
    cases = {
        str(case.get("label")): case for case in payload.get("cases", [])
        if isinstance(case, dict)
    }
    journal_loss = cases.get("replay_journal_loss", {})
    journal_loss_repairs = [
        int(event.get("journal_repair_records", 0) or 0)
        for event in journal_loss.get("trace", {}).get("compact_events", [])
        if isinstance(event, dict)
    ]
    return {
        "available": True,
        "path": str(EPOCH_REPLAY_FAULT_MATRIX.relative_to(ROOT)),
        "overall_pass": payload.get("overall_pass"),
        "case_pass": {
            name: case.get("pass") for name, case in sorted(cases.items())
        },
        "normal_compaction_pass": cases.get("replay_normal", {}).get("pass"),
        "torn_tail_pass": cases.get("replay_torn_tail", {}).get("pass"),
        "duplicate_generation_reject_pass": (
            cases.get("replay_duplicate_generation", {}).get("pass")
        ),
        "journal_loss_repair_pass": journal_loss.get("pass"),
        "journal_loss_repair_records_max": (
            max(journal_loss_repairs) if journal_loss_repairs else 0
        ),
        "negative_claim_guard": payload.get("negative_claim_guard"),
    }


def epoch_publication_comparison_evidence() -> dict[str, Any]:
    payload = load_json(EPOCH_PUBLICATION_COMPARISON)
    if not payload:
        return {
            "available": False,
            "path": str(EPOCH_PUBLICATION_COMPARISON.relative_to(ROOT)),
            "reason": "run experiments/run_epoch_publication_comparison.py first",
        }
    cases = {
        str(case.get("label")): case for case in payload.get("cases", [])
        if isinstance(case, dict)
    }
    strict = cases.get("strict", {})
    epoch = cases.get("epoch_redo_log", {})
    strict_grouped = cases.get("strict_grouped", {})
    epoch_grouped = cases.get("epoch_redo_log_grouped", {})
    amortization_strict = strict_grouped if strict_grouped else strict
    amortization_epoch = epoch_grouped if epoch_grouped else epoch
    return {
        "available": True,
        "path": str(EPOCH_PUBLICATION_COMPARISON.relative_to(ROOT)),
        "overall_pass": payload.get("overall_pass"),
        "strict_throughput_mib_s": strict.get("throughput_mib_s"),
        "epoch_throughput_mib_s": epoch.get("throughput_mib_s"),
        "strict_sync_count_total": strict.get("sync_count_total"),
        "epoch_sync_count_total": epoch.get("sync_count_total"),
        "strict_data_fsync_count_total": strict.get("measured_publication_summary", {}).get("data_fsync_count_total"),
        "strict_journal_fsync_count_total": strict.get("measured_publication_summary", {}).get("journal_fsync_count_total"),
        "strict_epoch_log_fsync_count_total": strict.get("measured_publication_summary", {}).get("epoch_log_fsync_count_total"),
        "epoch_data_fsync_count_total": epoch.get("measured_publication_summary", {}).get("data_fsync_count_total"),
        "epoch_journal_fsync_count_total": epoch.get("measured_publication_summary", {}).get("journal_fsync_count_total"),
        "epoch_epoch_log_fsync_count_total": epoch.get("measured_publication_summary", {}).get("epoch_log_fsync_count_total"),
        "strict_grouped_throughput_mib_s": strict_grouped.get("throughput_mib_s"),
        "epoch_grouped_throughput_mib_s": epoch_grouped.get("throughput_mib_s"),
        "strict_grouped_sync_count_total": strict_grouped.get("sync_count_total"),
        "epoch_grouped_sync_count_total": epoch_grouped.get("sync_count_total"),
        "strict_grouped_client_p99_ns": strict_grouped.get("client_write_fsync_latency_ns", {}).get("p99_ns"),
        "epoch_grouped_client_p99_ns": epoch_grouped.get("client_write_fsync_latency_ns", {}).get("p99_ns"),
        "epoch_grouped_append_group_size_max": epoch_grouped.get("epoch_append_group_size_max"),
        "epoch_grouped_sync_primitives": epoch_grouped.get("epoch_append_sync_primitives"),
        "amortization_strict_sync_count_total": amortization_strict.get("sync_count_total"),
        "amortization_epoch_sync_count_total": amortization_epoch.get("sync_count_total"),
        "amortization_strict_throughput_mib_s": amortization_strict.get("throughput_mib_s"),
        "amortization_epoch_throughput_mib_s": amortization_epoch.get("throughput_mib_s"),
        "strict_client_p99_ns": strict.get("client_write_fsync_latency_ns", {}).get("p99_ns"),
        "epoch_client_p99_ns": epoch.get("client_write_fsync_latency_ns", {}).get("p99_ns"),
        "strict_commit_p99_ns": strict.get("measured_publication_summary", {}).get("commit_latency_ns", {}).get("p99_ns"),
        "epoch_commit_p99_ns": epoch.get("measured_publication_summary", {}).get("commit_latency_ns", {}).get("p99_ns"),
        "comparison_verdict": payload.get("comparison_verdict"),
        "negative_claim_guard": payload.get("negative_claim_guard"),
    }


def implementation_blockers(evidence: dict[str, Any],
                            telemetry: dict[str, Any],
                            fairness_replay: dict[str, Any],
                            epoch_replay: dict[str, Any],
                            publication_comparison: dict[str, Any]) -> list[str]:
    blockers: list[str] = []
    epoch_journal_replaced = (
        evidence.get("epoch_publish_sets_skip_journal_fsync") is True and
        evidence.get("epoch_log_journal_repair_exists") is True and
        epoch_replay.get("journal_loss_repair_pass") is True and
        int(epoch_replay.get("journal_loss_repair_records_max", 0) or 0) >= 1 and
        int(publication_comparison.get("strict_journal_fsync_count_total") or 0) > 0 and
        publication_comparison.get("epoch_journal_fsync_count_total") == 0 and
        int(publication_comparison.get("epoch_epoch_log_fsync_count_total") or 0) > 0
    )
    if not evidence["epoch_publish_module_exists"]:
        blockers.append("No epoch publication module exists.")
    if not evidence["parallel_commit_module_exists"]:
        blockers.append("No sharded/parallel commit module exists.")
    if evidence["parallel_commit_module_exists"] and not evidence["writeback_calls_parallel_commit"]:
        blockers.append("Parallel commit coordinator exists but is not integrated into the mounted writeback/publication path.")
    if not evidence["parallel_commit_has_leader_follower_roles"]:
        blockers.append("No leader/waiter protocol is implemented or named in code.")
    if not evidence["parallel_commit_has_shard_assignment"]:
        blockers.append("No implemented shard assignment or shard-count sweep exists.")
    if not epoch_journal_replaced:
        blockers.append("Epoch mode has not yet replaced strict journal fdatasync with a recovery-proven epoch-log barrier.")
    if not telemetry.get("available") or telemetry.get("overall_pass") is not True:
        blockers.append("No group-size distribution, epoch wait-time distribution, queue-depth sweep, shard-count sweep, and concurrent-client artifact exists.")
    if (not fairness_replay.get("available") or
            fairness_replay.get("overall_pass") is not True):
        blockers.append("No replay-time sweep or starvation/fairness artifact exists.")
    if not evidence["epoch_log_replay_parser_exists"]:
        blockers.append("No epoch redo-log replay parser exists in production source.")
    if not evidence["epoch_log_checkpoint_compaction_exists"]:
        blockers.append("No epoch checkpoint compaction helper exists in production source.")
    if not epoch_replay.get("available") or epoch_replay.get("overall_pass") is not True:
        blockers.append("No epoch redo-log replay or crash-recovery ordering artifact exists.")
    if (not publication_comparison.get("available") or
            publication_comparison.get("overall_pass") is not True):
        blockers.append("No strict-versus-epoch publication measurement artifact exists.")
    elif publication_comparison.get("epoch_sync_count_total") is not None:
        epoch_sync = int(publication_comparison.get("amortization_epoch_sync_count_total") or 0)
        strict_sync = int(publication_comparison.get("amortization_strict_sync_count_total") or 0)
        epoch_tput = float(publication_comparison.get("amortization_epoch_throughput_mib_s") or 0.0)
        strict_tput = float(publication_comparison.get("amortization_strict_throughput_mib_s") or 0.0)
        if epoch_sync >= strict_sync and epoch_tput <= strict_tput:
            blockers.append(
                "Epoch mode removes journal fdatasync but has not yet produced group-commit amortization: total sync count is not lower and throughput is not higher under the retained matched workload."
            )
    return blockers


def contract_payload() -> dict[str, Any]:
    evidence = source_evidence()
    baseline = baseline_evidence()
    lock_trace = summarize_lock_trace(LOCK_TRACE)
    telemetry = telemetry_sweep_evidence()
    fairness_replay = fairness_replay_evidence()
    epoch_replay = epoch_replay_fault_matrix_evidence()
    publication_comparison = epoch_publication_comparison_evidence()
    blockers = implementation_blockers(evidence, telemetry, fairness_replay,
                                       epoch_replay, publication_comparison)
    return {
        "schema_version": 1,
        "generated_by": "experiments/build_parallel_commit_contract.py",
        "generated_utc": now_utc(),
        "scope": "Gate 0.16 sharded queues and parallel commit contract.",
        "overall_pass": False,
        "closure_verdict": "not_closed",
        "source_evidence": evidence,
        "current_topology": classify_current_topology(evidence),
        "candidate_topologies": design_candidates(),
        "baseline_mounted_path_evidence": baseline,
        "epoch_path_smoke_evidence": epoch_path_smoke_evidence(),
        "telemetry_sweep_evidence": telemetry,
        "fairness_replay_evidence": fairness_replay,
        "epoch_replay_fault_matrix_evidence": epoch_replay,
        "epoch_publication_comparison_evidence": publication_comparison,
        "lock_trace_summary": lock_trace,
        "required_measurements_before_claim": [
            "strict versus epoch throughput under the same frozen workload contract",
            "p99 and p99.9 foreground latency under same-file and disjoint-file clients",
            "shard-count sweep",
            "queue-depth sweep",
            "concurrent-client sweep using external OS processes",
            "group-size distribution",
            "epoch wait-time distribution",
            "commit batching efficiency and fdatasync count reduction",
            "replay time and recovery ordering proof",
            "starvation/fairness negative test",
            "fault matrix for enqueue, data write, redo append, epoch barrier, checkpoint, and replay",
        ],
        "implementation_blockers": blockers,
        "negative_claim_guard": (
            "No paper or README text may claim sharded queues, parallel commit, "
            "epoch fdatasync, group commit, or scalability from commit batching "
            "until this contract reports overall_pass=true with production mounted-path evidence."
        ),
    }


def paper_readme_claim_scan() -> dict[str, Any]:
    matches = []
    for target in CLAIM_SCAN_TARGETS:
        for path in iter_text_files(target):
            rel = str(path.relative_to(ROOT))
            try:
                lines = path.read_text(encoding="utf-8",
                                       errors="replace").splitlines()
            except OSError:
                continue
            for line_no, line in enumerate(lines, start=1):
                lowered = line.lower()
                for phrase in CLAIM_SCAN_PHRASES:
                    if phrase in lowered:
                        if any(term in lowered for term in CLAIM_SCAN_ALLOWED_SCOPE_TERMS):
                            continue
                        matches.append({
                            "path": rel,
                            "line": line_no,
                            "phrase": phrase,
                            "text": line.strip()[:240],
                        })
    return {
        "targets": [str(target.relative_to(ROOT)) for target in CLAIM_SCAN_TARGETS],
        "phrases": CLAIM_SCAN_PHRASES,
        "match_count": len(matches),
        "matches": matches,
    }


def paper_text_evidence() -> dict[str, Any]:
    paper_text = "\n".join(read_text(path) for path in sorted(PAPER.glob("*.tex")))
    lowered = paper_text.lower()
    requirements = {
        "mentions_epoch_redo_log_mode":
            "opt-in epoch-redo-log mode" in paper_text,
        "mentions_bounded_leader_follower_epoch_group":
            "bounded leader/follower epoch group" in paper_text,
        "mentions_shared_durability_barrier":
            "leader performs the shared durability barrier" in paper_text,
        "mentions_sync_count_amortization":
            "sync-count amortization" in lowered and "grouped operations" in lowered,
        "mentions_single_client_loss_case":
            "single-client per-write \\texttt{fdatasync} remains a strict-mode case" in paper_text,
    }
    return {
        "complete": all(requirements.values()),
        "requirements": requirements,
        "paper_location": "Paper/3_Design.tex",
    }


def closure_audit_payload(contract: dict[str, Any]) -> dict[str, Any]:
    evidence = contract["source_evidence"]
    blockers = contract["implementation_blockers"]
    claim_scan = paper_readme_claim_scan()
    publication = contract["epoch_publication_comparison_evidence"]
    replay = contract["epoch_replay_fault_matrix_evidence"]
    epoch_journal_replaced = (
        evidence.get("epoch_publish_sets_skip_journal_fsync") is True and
        evidence.get("epoch_log_journal_repair_exists") is True and
        replay.get("journal_loss_repair_pass") is True and
        int(replay.get("journal_loss_repair_records_max", 0) or 0) >= 1 and
        int(publication.get("strict_journal_fsync_count_total") or 0) > 0 and
        publication.get("epoch_journal_fsync_count_total") == 0 and
        int(publication.get("epoch_epoch_log_fsync_count_total") or 0) > 0
    )
    group_commit_amortized = (
        publication.get("overall_pass") is True and
        (
            int(publication.get("amortization_epoch_sync_count_total", 0) or 0) <
            int(publication.get("amortization_strict_sync_count_total", 0) or 0) or
            float(publication.get("amortization_epoch_throughput_mib_s", 0.0) or 0.0) >
            float(publication.get("amortization_strict_throughput_mib_s", 0.0) or 0.0)
        )
    )
    code_script = {
        "complete": (
            evidence.get("parallel_commit_module_exists") is True and
            evidence.get("writeback_calls_parallel_commit") is True and
            evidence.get("epoch_publish_module_exists") is True and
            epoch_journal_replaced and
            group_commit_amortized
        ),
        "present_evidence": [
            "parallel commit coordinator source exists",
            "mounted writeback path calls runtime parallel commit coordinator",
            (
                "epoch mode replaces strict journal fdatasync with a recovery-proven "
                "epoch-log barrier" if epoch_journal_replaced else
                "epoch journal-fsync replacement not yet proven"
            ),
            (
                "grouped epoch comparison reduces total sync count under the "
                "concurrent mounted workload" if group_commit_amortized else
                "group-commit amortization not yet proven"
            ),
        ],
        "missing_evidence": [
            item for item, missing in [
                ("epoch publication module", not evidence.get("epoch_publish_module_exists")),
                ("epoch-mode journal fdatasync replacement", not epoch_journal_replaced),
                ("group-commit amortization that reduces total sync count or improves throughput", not group_commit_amortized),
            ] if missing
        ],
    }
    artifacts = {
        "complete": (
            contract["epoch_path_smoke_evidence"].get("overall_pass") is True and
            contract["telemetry_sweep_evidence"].get("overall_pass") is True and
            contract["fairness_replay_evidence"].get("overall_pass") is True and
            contract["epoch_replay_fault_matrix_evidence"].get("overall_pass") is True and
            contract["epoch_publication_comparison_evidence"].get("overall_pass") is True and
            not any("redo-log" in blocker or "crash-recovery" in blocker
                    for blocker in blockers)
        ),
        "present_evidence": [
            contract["epoch_path_smoke_evidence"].get("path"),
            contract["telemetry_sweep_evidence"].get("path"),
            contract["fairness_replay_evidence"].get("path"),
            contract["epoch_replay_fault_matrix_evidence"].get("path"),
            contract["epoch_publication_comparison_evidence"].get("path"),
        ],
        "missing_evidence": [
            blocker for blocker in blockers
            if "redo-log" in blocker or "crash-recovery" in blocker
        ],
    }
    paper_evidence = paper_text_evidence()
    paper_text = {
        "complete": paper_evidence["complete"],
        "reason": (
            "Gate 0.16 paper text is present." if paper_evidence["complete"]
            else "Gate 0.16 paper text is missing one or more required boundaries."
        ),
        "requirements": paper_evidence["requirements"],
        "paper_location": paper_evidence["paper_location"],
        "claim_scan": claim_scan,
    }
    negative_guard = {
        "complete": bool(contract.get("negative_claim_guard")) and
        claim_scan["match_count"] == 0,
        "guard_text": contract.get("negative_claim_guard"),
        "claim_scan_match_count": claim_scan["match_count"],
    }
    complete = (
        code_script["complete"] and artifacts["complete"] and
        paper_text["complete"] and negative_guard["complete"]
    )
    return {
        "schema_version": 1,
        "generated_by": "experiments/build_parallel_commit_contract.py",
        "generated_utc": contract["generated_utc"],
        "scope": "Gate 0.16-S5 closure audit.",
        "closure_verdict": "closed" if complete else "blocked",
        "overall_pass": complete,
        "code_script": code_script,
        "artifacts": artifacts,
        "paper_text": paper_text,
        "negative_claim_guard": negative_guard,
        "blocking_items": blockers,
        "next_unblock_step": (
            "Add the Gate 0.16 paper text and negative-claim guard only after "
            "the user explicitly moves from code-first implementation to "
            "paper-claim closure."
        ),
    }


def closure_audit_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Parallel Commit Closure Audit",
        "",
        f"- Generated: `{payload['generated_utc']}`",
        f"- Verdict: `{payload['closure_verdict']}`",
        f"- Overall pass: `{str(payload['overall_pass']).lower()}`",
        "",
        "## Closure Components",
        "",
        f"- Code/script complete: `{str(payload['code_script']['complete']).lower()}`",
        f"- Artifact complete: `{str(payload['artifacts']['complete']).lower()}`",
        f"- Paper text complete: `{str(payload['paper_text']['complete']).lower()}`",
        f"- Negative guard complete: `{str(payload['negative_claim_guard']['complete']).lower()}`",
        "",
        "## Blocking Items",
        "",
    ]
    for blocker in payload["blocking_items"]:
        lines.append(f"- {blocker}")
    lines.extend([
        "",
        "## Next Unblock Step",
        "",
        payload["next_unblock_step"],
        "",
        "## Negative Claim Guard",
        "",
        payload["negative_claim_guard"]["guard_text"],
        "",
    ])
    claim_scan = payload["paper_text"]["claim_scan"]
    if claim_scan["match_count"]:
        lines.extend(["## Paper/README Claim Scan Matches", ""])
        for match in claim_scan["matches"]:
            lines.append(
                f"- `{match['path']}:{match['line']}` `{match['phrase']}`: "
                f"{match['text']}"
            )
        lines.append("")
    return "\n".join(lines)


def markdown(payload: dict[str, Any]) -> str:
    current = payload["current_topology"]
    lines = [
        "# Parallel Commit Contract",
        "",
        f"- Generated: `{payload['generated_utc']}`",
        f"- Gate: `{payload['scope']}`",
        f"- Verdict: `{payload['closure_verdict']}`",
        f"- Current topology: `{current['name']}`",
        f"- Paper-eligible as parallel commit: `{str(current['paper_eligible_as_parallel_commit']).lower()}`",
        "",
        "## Current Evidence",
        "",
    ]
    for key, value in payload["source_evidence"].items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(["", "## Mounted-Path Baseline", ""])
    baseline = payload["baseline_mounted_path_evidence"]
    if baseline.get("available"):
        lines.append(f"- Concurrency summary: `{baseline['summary_path']}`")
        lines.append(f"- Overall pass: `{str(baseline.get('overall_pass')).lower()}`")
        lines.append(f"- Thread counts: `{baseline.get('thread_counts')}`")
        lines.append(f"- Workload coverage: `{baseline.get('workload_coverage')}`")
        blocking = baseline.get("blocking_syscall_profile", {})
        lines.append(f"- Blocking syscall profile pass: `{str(blocking.get('overall_pass')).lower()}`")
    else:
        lines.append(f"- Missing: `{baseline.get('reason')}`")
    lines.extend(["", "## Epoch-Gated Strict Mounted-Path Smoke", ""])
    epoch_smoke = payload["epoch_path_smoke_evidence"]
    if epoch_smoke.get("available"):
        lines.append(f"- Smoke artifact: `{epoch_smoke['path']}`")
        lines.append(f"- Overall pass: `{str(epoch_smoke.get('overall_pass')).lower()}`")
        lines.append(f"- Strict control pass: `{str(epoch_smoke.get('strict_pass')).lower()}`")
        lines.append(f"- Epoch-gated strict pass: `{str(epoch_smoke.get('epoch_gated_strict_pass')).lower()}`")
        lines.append(f"- Begin events: `{epoch_smoke.get('epoch_begin_count')}`")
        lines.append(f"- Finish events: `{epoch_smoke.get('epoch_finish_count')}`")
        lines.append(f"- Roles: `{epoch_smoke.get('epoch_roles')}`")
    else:
        lines.append(f"- Missing: `{epoch_smoke.get('reason')}`")
    lines.extend(["", "## Parallel Commit Telemetry Sweep", ""])
    telemetry = payload["telemetry_sweep_evidence"]
    if telemetry.get("available"):
        lines.append(f"- Sweep artifact: `{telemetry['path']}`")
        lines.append(f"- Overall pass: `{str(telemetry.get('overall_pass')).lower()}`")
        lines.append(f"- Coverage: `{telemetry.get('coverage')}`")
        for case in telemetry.get("case_summaries", []):
            lines.append(f"- `{case.get('name')}`: pass=`{str(case.get('pass')).lower()}`, "
                         f"clients=`{case.get('clients')}`, shards=`{case.get('shards')}`, "
                         f"group sizes=`{case.get('group_size_distribution')}`, "
                         f"queue depths=`{case.get('queue_depth_distribution')}`, "
                         f"trace=`{case.get('trace_path')}`")
    else:
        lines.append(f"- Missing: `{telemetry.get('reason')}`")
    lines.extend(["", "## Fairness and Trace Replay-Order Evidence", ""])
    fairness_replay = payload["fairness_replay_evidence"]
    if fairness_replay.get("available"):
        lines.append(f"- Evidence artifact: `{fairness_replay['path']}`")
        lines.append(f"- Overall pass: `{str(fairness_replay.get('overall_pass')).lower()}`")
        lines.append(f"- Coverage: `{fairness_replay.get('coverage')}`")
        for case in fairness_replay.get("case_summaries", []):
            lines.append(
                f"- `{case.get('name')}`: pass=`{str(case.get('pass')).lower()}`, "
                f"starvation=`{str(case.get('starvation_negative_pass')).lower()}`, "
                f"replay=`{str(case.get('replay_order_pass')).lower()}`, "
                f"ops=`{case.get('completed_ops')}/{case.get('expected_ops')}`, "
                f"replay_plan=`{case.get('replay_plan_length')}`, "
                f"replay_time_ns=`{case.get('reconstruction_time_ns')}`, "
                f"trace=`{case.get('trace_path')}`"
            )
    else:
        lines.append(f"- Missing: `{fairness_replay.get('reason')}`")
    lines.extend(["", "## Epoch Redo-Log Replay Fault Matrix", ""])
    epoch_replay = payload["epoch_replay_fault_matrix_evidence"]
    if epoch_replay.get("available"):
        lines.append(f"- Evidence artifact: `{epoch_replay['path']}`")
        lines.append(f"- Overall pass: `{str(epoch_replay.get('overall_pass')).lower()}`")
        lines.append(f"- Case pass: `{epoch_replay.get('case_pass')}`")
        lines.append(
            f"- Duplicate-generation rejection: "
            f"`{str(epoch_replay.get('duplicate_generation_reject_pass')).lower()}`"
        )
        lines.append(f"- Torn-tail remount: `{str(epoch_replay.get('torn_tail_pass')).lower()}`")
        lines.append(
            f"- Journal-loss repair: "
            f"`{str(epoch_replay.get('journal_loss_repair_pass')).lower()}`"
        )
        lines.append(
            f"- Journal repair records max: "
            f"`{epoch_replay.get('journal_loss_repair_records_max')}`"
        )
    else:
        lines.append(f"- Missing: `{epoch_replay.get('reason')}`")
    lines.extend(["", "## Strict Versus Epoch Publication Measurement", ""])
    publication_comparison = payload["epoch_publication_comparison_evidence"]
    if publication_comparison.get("available"):
        lines.append(f"- Evidence artifact: `{publication_comparison['path']}`")
        lines.append(f"- Overall pass: `{str(publication_comparison.get('overall_pass')).lower()}`")
        lines.append(f"- Strict throughput MiB/s: `{publication_comparison.get('strict_throughput_mib_s')}`")
        lines.append(f"- Epoch throughput MiB/s: `{publication_comparison.get('epoch_throughput_mib_s')}`")
        lines.append(f"- Strict sync count: `{publication_comparison.get('strict_sync_count_total')}`")
        lines.append(f"- Epoch sync count: `{publication_comparison.get('epoch_sync_count_total')}`")
        lines.append(f"- Strict grouped sync count: `{publication_comparison.get('strict_grouped_sync_count_total')}`")
        lines.append(f"- Epoch grouped sync count: `{publication_comparison.get('epoch_grouped_sync_count_total')}`")
        lines.append(f"- Epoch grouped max append group size: `{publication_comparison.get('epoch_grouped_append_group_size_max')}`")
        lines.append(f"- Epoch grouped sync primitives: `{publication_comparison.get('epoch_grouped_sync_primitives')}`")
        lines.append(f"- Strict journal fsync count: `{publication_comparison.get('strict_journal_fsync_count_total')}`")
        lines.append(f"- Epoch journal fsync count: `{publication_comparison.get('epoch_journal_fsync_count_total')}`")
        lines.append(f"- Epoch log fsync count: `{publication_comparison.get('epoch_epoch_log_fsync_count_total')}`")
        lines.append(f"- Verdict: {publication_comparison.get('comparison_verdict')}")
    else:
        lines.append(f"- Missing: `{publication_comparison.get('reason')}`")
    lines.extend(["", "## Candidate Topologies", ""])
    for candidate in payload["candidate_topologies"]:
        lines.append(f"### {candidate['name']}")
        lines.append(f"- Status: `{candidate['status']}`")
        lines.append(f"- Shard assignment: {candidate['shard_assignment']}")
        lines.append(f"- Leader/waiter protocol: {candidate['leader_waiter_protocol']}")
        lines.append(f"- Fairness policy: {candidate['fairness_policy']}")
        lines.append(f"- Recovery ordering rule: {candidate['recovery_ordering_rule']}")
        lines.append("")
    lines.extend(["## Blocking Items", ""])
    for blocker in payload["implementation_blockers"]:
        lines.append(f"- {blocker}")
    lines.extend([
        "",
        "## Negative Claim Guard",
        "",
        payload["negative_claim_guard"],
        "",
    ])
    return "\n".join(lines)


def main() -> int:
    payload = contract_payload()
    closure_audit = closure_audit_payload(payload)
    write_json(OUT / "parallel_commit_contract.json", payload)
    write_text(OUT / "parallel_commit_contract.md", markdown(payload))
    write_json(OUT / "parallel_commit_closure_audit.json", closure_audit)
    write_text(OUT / "parallel_commit_closure_audit.md",
               closure_audit_markdown(closure_audit))
    print(json.dumps({
        "json": str((OUT / "parallel_commit_contract.json").relative_to(ROOT)),
        "markdown": str((OUT / "parallel_commit_contract.md").relative_to(ROOT)),
        "closure_audit": str((OUT / "parallel_commit_closure_audit.json")
                             .relative_to(ROOT)),
        "closure_verdict": payload["closure_verdict"],
        "closure_audit_verdict": closure_audit["closure_verdict"],
        "current_topology": payload["current_topology"]["name"],
        "blocker_count": len(payload["implementation_blockers"]),
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
