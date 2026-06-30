#!/usr/bin/env python3
"""Build the Gate 0.9 publication-protocol closeout from retained evidence."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "artifacts" / "validation" / "publication_protocol_fault_matrix"
DOC = ROOT / "docs" / "architecture" / "publication_protocol.md"
COMPARISON = OUT / "epoch_publication_comparison.json"
REPLAY = OUT / "epoch_replay_fault_matrix.json"
PARALLEL_AUDIT = (
    ROOT / "artifacts" / "validation" / "parallel_commit_contract" /
    "parallel_commit_closure_audit.json"
)
PAPER_DIR = ROOT / "Paper"


def now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def relpath(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected object in {relpath(path)}")
    return payload


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def case_by_label(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(case.get("label")): case
        for case in payload.get("cases", [])
        if isinstance(case, dict)
    }


def pub_summary(case: dict[str, Any]) -> dict[str, Any]:
    summary = case.get("measured_publication_summary", {})
    latency = case.get("client_write_fsync_latency_ns", {})
    return {
        "pass": case.get("pass"),
        "mode": case.get("mode"),
        "workload_kind": case.get("workload_kind", "sequential_unique_file_fdatasync"),
        "client_count": case.get("client_count", 1),
        "throughput_mib_s": case.get("throughput_mib_s"),
        "client_p99_ns": latency.get("p99_ns"),
        "client_p999_ns": latency.get("p999_ns"),
        "sync_count_total": summary.get("sync_count_total"),
        "data_fsync_count_total": summary.get("data_fsync_count_total"),
        "journal_fsync_count_total": summary.get("journal_fsync_count_total"),
        "epoch_log_fsync_count_total": summary.get("epoch_log_fsync_count_total"),
        "commit_p99_ns": summary.get("commit_latency_ns", {}).get("p99_ns"),
        "epoch_append_group_size_max": case.get("epoch_append_group_size_max"),
        "epoch_append_sync_primitives": case.get("epoch_append_sync_primitives"),
    }


def replay_summary(case: dict[str, Any]) -> dict[str, Any]:
    compact_events = case.get("trace", {}).get("compact_events", [])
    repairs = [
        int(event.get("journal_repair_records", 0) or 0)
        for event in compact_events
        if isinstance(event, dict)
    ]
    torn_tail = [
        int(event.get("torn_tail_bytes", 0) or 0)
        for event in compact_events
        if isinstance(event, dict)
    ]
    duplicates = [
        int(event.get("duplicate_generation_records", 0) or 0)
        for event in compact_events
        if isinstance(event, dict)
    ]
    rc_values = [
        int(event.get("rc", 0) or 0)
        for event in compact_events
        if isinstance(event, dict)
    ]
    return {
        "pass": case.get("pass"),
        "mutation": case.get("mutation"),
        "remount_read_matches": case.get("read_client", {}).get("matches"),
        "compact_rc_values": rc_values,
        "journal_repair_records_max": max(repairs) if repairs else 0,
        "torn_tail_bytes_max": max(torn_tail) if torn_tail else 0,
        "duplicate_generation_records_max": max(duplicates) if duplicates else 0,
    }


def source_evidence() -> dict[str, Any]:
    strict = read_text(ROOT / "code" / "storage" / "pqc_strict_publish.c")
    epoch = read_text(ROOT / "code" / "storage" / "pqc_epoch_publish.c")
    log = read_text(ROOT / "code" / "storage" / "pqc_epoch_log.c")
    fd_context = read_text(ROOT / "code" / "fs" / "pqc_fd_context.c")
    return {
        "strict_data_fsync": (
            "pqc_durability_fdatasync(\n                req->data_fd" in strict or
            "pqc_durability_fdatasync(req->data_fd" in strict
        ),
        "strict_journal_fsync": (
            "pqc_durability_fdatasync(\n            req->journal_fd" in strict or
            "pqc_durability_fdatasync(req->journal_fd" in strict
        ),
        "strict_checkpoint_stage": "pqc_checkpoint_store_and_stage_anchor" in strict,
        "epoch_mode_dispatch": "PQC_PUBLICATION_MODE_EPOCH_REDO_LOG" in epoch,
        "epoch_skips_strict_journal_fsync": "skip_journal_fsync = 1" in epoch,
        "epoch_group_barrier": "pqc_epoch_group_barrier_wait" in epoch,
        "epoch_group_config": "PQC_EPOCH_GROUP_MAX" in epoch,
        "epoch_group_syncfs": "pqc_durability_syncfs" in epoch,
        "epoch_trace_group_size": "group_size" in epoch,
        "epoch_log_replay_parser": "pqc_epoch_log_replay_fd" in log,
        "epoch_checkpoint_compaction": "pqc_epoch_log_compact_checkpoint" in log,
        "epoch_journal_repair": "epoch_log_repair_journal_prefix" in log,
        "epoch_duplicate_generation_reject": "duplicate_generation_records" in log,
        "fd_context_runs_epoch_compaction": "pqc_epoch_log_compact_checkpoint" in fd_context,
    }


def paper_text_evidence() -> dict[str, Any]:
    text = "\n".join(
        path.read_text(encoding="utf-8", errors="replace")
        for path in sorted(PAPER_DIR.glob("*.tex"))
    )
    lowered = text.lower()
    requirements = {
        "strict_publication_path": (
            "fsync} flushes the coalescing buffer" in text and
            "fdatasync} on both sidecars" in text
        ),
        "epoch_mode_named": "opt-in epoch-redo-log mode" in text,
        "bounded_group_topology": "bounded leader/follower epoch group" in text,
        "shared_barrier": "leader performs the shared durability barrier" in text,
        "sync_count_amortization": (
            "sync-count amortization" in lowered and
            "grouped operations" in lowered
        ),
        "single_client_loss_case": (
            "single-client per-write \\texttt{fdatasync} remains a strict-mode case"
            in text
        ),
        "replay_boundary": (
            "recovery accepts only committed epoch prefixes" in lowered or
            "replay accepts only committed epoch prefixes" in lowered
        ),
        "no_full_crash_claim": (
            "does not claim physical power-loss" in text or
            "not physical power-loss" in text
        ),
        "no_kernel_bypass_claim": (
            "does not claim verified \\texttt{O\\_DIRECT} NVMe-to-UVM DMA" in text and
            "\\texttt{io\\_uring}/eBPF completion bypass" in text
        ),
        "no_rollback_upgrade": "no persistent PCR-bound lifecycle claim" in text,
        "unsupported_posix_boundary": "unsupported POSIX modes remain open" in text,
    }
    return {
        "complete": all(requirements.values()),
        "paper_location": "Paper/3_Design.tex",
        "requirements": requirements,
    }


def safe_less(left: Any, right: Any) -> bool:
    return left is not None and right is not None and left < right


def percent_reduction(before: Any, after: Any) -> float | None:
    if before is None or after is None or before == 0:
        return None
    return (float(before) - float(after)) * 100.0 / float(before)


def build_payload() -> dict[str, Any]:
    comparison = load_json(COMPARISON)
    replay = load_json(REPLAY)
    parallel_audit = load_json(PARALLEL_AUDIT)
    pub_cases = case_by_label(comparison)
    replay_cases = case_by_label(replay)

    strict = pub_summary(pub_cases.get("strict", {}))
    epoch = pub_summary(pub_cases.get("epoch_redo_log", {}))
    strict_grouped = pub_summary(pub_cases.get("strict_grouped", {}))
    epoch_grouped = pub_summary(pub_cases.get("epoch_redo_log_grouped", {}))
    replay_cases_summary = {
        name: replay_summary(case)
        for name, case in sorted(replay_cases.items())
    }
    grouped_sync_reduction = percent_reduction(
        strict_grouped.get("sync_count_total"),
        epoch_grouped.get("sync_count_total"),
    )

    src = source_evidence()
    paper = paper_text_evidence()
    source_complete = all(src.values())
    grouped_amortized = safe_less(
        epoch_grouped.get("sync_count_total"),
        strict_grouped.get("sync_count_total"),
    )
    replay_complete = replay.get("overall_pass") is True and all(
        case.get("pass") is True for case in replay_cases.values()
    )

    closeout_complete = (
        source_complete and comparison.get("overall_pass") is True and
        replay_complete and grouped_amortized and
        parallel_audit.get("code_script", {}).get("complete") is True and
        parallel_audit.get("paper_text", {}).get("complete") is True and
        parallel_audit.get("negative_claim_guard", {}).get("complete") is True
    )
    parent_checklist_closed = closeout_complete and paper["complete"]

    return {
        "schema_version": 1,
        "generated_by": "code/experiments/build_publication_protocol_closeout.py",
        "generated_utc": now_utc(),
        "scope": "Gate 0.9-S7 publication protocol architecture/evidence closeout.",
        "artifacts_used": [
            relpath(COMPARISON),
            relpath(REPLAY),
            relpath(PARALLEL_AUDIT),
        ],
        "source_evidence": src,
        "source_complete": source_complete,
        "strict": strict,
        "epoch_redo_log": epoch,
        "strict_grouped": strict_grouped,
        "epoch_redo_log_grouped": epoch_grouped,
        "measured_result": {
            "sequential_epoch_journal_fsync_removed": (
                strict.get("journal_fsync_count_total", 0) and
                epoch.get("journal_fsync_count_total") == 0
            ),
            "grouped_sync_amortized": grouped_amortized,
            "grouped_sync_reduction_percent": grouped_sync_reduction,
            "grouped_epoch_throughput_win": (
                epoch_grouped.get("throughput_mib_s") is not None and
                strict_grouped.get("throughput_mib_s") is not None and
                epoch_grouped.get("throughput_mib_s") >
                strict_grouped.get("throughput_mib_s")
            ),
            "sequential_epoch_throughput_win": (
                epoch.get("throughput_mib_s") is not None and
                strict.get("throughput_mib_s") is not None and
                epoch.get("throughput_mib_s") > strict.get("throughput_mib_s")
            ),
        },
        "replay_fault_matrix": {
            "overall_pass": replay.get("overall_pass"),
            "cases": replay_cases_summary,
        },
        "parallel_commit_audit": {
            "closure_verdict": parallel_audit.get("closure_verdict"),
            "implementation_blockers": parallel_audit.get("blocking_items"),
            "code_script_complete": parallel_audit.get("code_script", {}).get("complete"),
            "paper_text_complete": parallel_audit.get("paper_text", {}).get("complete"),
            "negative_claim_guard_complete": (
                parallel_audit.get("negative_claim_guard", {}).get("complete")
            ),
        },
        "paper_text": paper,
        "invariants": [
            "Strict mode keeps data-sidecar fdatasync before journal fdatasync.",
            "Epoch mode writes ciphertext and data-sidecar fdatasync before appending committed epoch records.",
            "Epoch mode removes strict journal fdatasync from the foreground epoch path and repairs journal mappings from committed epoch prefixes at remount.",
            "Grouped epoch mode uses one filesystem-level syncfs barrier for joined foreground operations when PQC_EPOCH_GROUP_MAX is greater than one.",
            "Replay accepts only committed epoch prefixes, ignores torn tails, and rejects duplicate generation records.",
            "Reader-visible generation state advances only after publication dispatch returns successfully.",
        ],
        "known_losses": [
            "Sequential epoch-redo-log mode does not improve total sync count because each write still has one data fdatasync and one epoch-log barrier.",
            "Grouped epoch-redo-log reduces traced sync count in the retained concurrent workload, but the retained run does not show a throughput win.",
            "The grouped barrier uses syncfs, which can be broader and more expensive than per-file fdatasync.",
        ],
        "non_claims": [
            "No direct NVMe-to-UVM, GPUDirect/RDMA, dma-buf zero-copy, eBPF/io_uring bypass, or kernel-bypass publication path is claimed.",
            "No physical power-loss certification or full crash certification is claimed.",
            "No persistent PCR-bound freshness, TPM rollback resistance, or side-channel protection is claimed by this gate.",
            "No general-purpose POSIX filesystem support or ready-for-deployment wording is supported by this closeout.",
        ],
        "negative_claim_guard": (
            "Paper or README text may describe strict and epoch publication only "
            "within the retained artifact scope. It must not claim throughput "
            "improvement, full crash certification, kernel bypass, direct storage "
            "DMA, rollback resistance, side-channel defense, or general-purpose "
            "POSIX support from Gate 0.9 evidence."
        ),
        "closeout_complete": closeout_complete,
        "paper_text_status": "updated" if paper["complete"] else "missing",
        "parent_checklist_closed": parent_checklist_closed,
    }


def fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def metric_row(name: str, summary: dict[str, Any]) -> str:
    return (
        f"| {name} | {fmt(summary.get('workload_kind'))} | "
        f"{fmt(summary.get('client_count'))} | "
        f"{fmt(summary.get('sync_count_total'))} | "
        f"{fmt(summary.get('data_fsync_count_total'))} | "
        f"{fmt(summary.get('journal_fsync_count_total'))} | "
        f"{fmt(summary.get('epoch_log_fsync_count_total'))} | "
        f"{fmt(summary.get('throughput_mib_s'))} | "
        f"{fmt(summary.get('client_p99_ns'))} | "
        f"{fmt(summary.get('epoch_append_group_size_max'))} | "
        f"{fmt(summary.get('epoch_append_sync_primitives'))} |"
    )


def markdown(payload: dict[str, Any], title: str) -> str:
    lines = [
        f"# {title}",
        "",
        f"- Generated: `{payload['generated_utc']}`",
        f"- Scope: `{payload['scope']}`",
        f"- Closeout complete: `{str(payload['closeout_complete']).lower()}`",
        f"- Paper text status: `{payload['paper_text_status']}`",
        f"- Parent checklist closed: `{str(payload['parent_checklist_closed']).lower()}`",
        "",
        "## Evidence Inputs",
        "",
    ]
    for artifact in payload["artifacts_used"]:
        lines.append(f"- `{artifact}`")

    lines.extend([
        "",
        "## Production Mechanisms",
        "",
    ])
    for key, value in payload["source_evidence"].items():
        lines.append(f"- `{key}`: `{str(value).lower()}`")

    lines.extend([
        "",
        "## Measured Publication Modes",
        "",
        "| Mode | Workload | Clients | Syncs | Data fsync | Journal fsync | Epoch fsync | MiB/s | Client p99 ns | Max group | Sync primitive |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        metric_row("strict", payload["strict"]),
        metric_row("epoch-redo-log", payload["epoch_redo_log"]),
        metric_row("strict grouped", payload["strict_grouped"]),
        metric_row("epoch-redo-log grouped", payload["epoch_redo_log_grouped"]),
        "",
        "## Result Boundary",
        "",
        f"- Sequential epoch journal fsync removed: `{str(payload['measured_result']['sequential_epoch_journal_fsync_removed']).lower()}`",
        f"- Grouped sync amortized: `{str(payload['measured_result']['grouped_sync_amortized']).lower()}`",
        f"- Grouped sync reduction percent: `{fmt(payload['measured_result']['grouped_sync_reduction_percent'])}`",
        f"- Sequential throughput win: `{str(payload['measured_result']['sequential_epoch_throughput_win']).lower()}`",
        f"- Grouped throughput win: `{str(payload['measured_result']['grouped_epoch_throughput_win']).lower()}`",
        "",
        "## Replay And Recovery Evidence",
        "",
        f"- Overall pass: `{str(payload['replay_fault_matrix']['overall_pass']).lower()}`",
        "",
    ])
    for name, case in payload["replay_fault_matrix"]["cases"].items():
        lines.append(
            f"- `{name}`: pass=`{str(case.get('pass')).lower()}`, "
            f"mutation=`{case.get('mutation')}`, "
            f"repair_max=`{case.get('journal_repair_records_max')}`, "
            f"torn_tail_max=`{case.get('torn_tail_bytes_max')}`, "
            f"duplicate_max=`{case.get('duplicate_generation_records_max')}`"
        )

    lines.extend(["", "## Invariants", ""])
    for item in payload["invariants"]:
        lines.append(f"- {item}")
    lines.extend(["", "## Known Losses", ""])
    for item in payload["known_losses"]:
        lines.append(f"- {item}")
    lines.extend(["", "## Non-Claims", ""])
    for item in payload["non_claims"]:
        lines.append(f"- {item}")
    lines.extend([
        "",
        "## Negative Claim Guard",
        "",
        payload["negative_claim_guard"],
        "",
    ])
    return "\n".join(lines)


def main() -> int:
    payload = build_payload()
    OUT.mkdir(parents=True, exist_ok=True)
    DOC.parent.mkdir(parents=True, exist_ok=True)
    json_path = OUT / "publication_protocol_closeout.json"
    md_path = OUT / "publication_protocol_closeout.md"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n",
                         encoding="utf-8")
    md_text = markdown(payload, "Publication Protocol Closeout")
    md_path.write_text(md_text, encoding="utf-8")
    DOC.write_text(markdown(payload, "AEGIS-Q Publication Protocol"),
                   encoding="utf-8")
    print(json.dumps({
        "closeout_complete": payload["closeout_complete"],
        "json": relpath(json_path),
        "markdown": relpath(md_path),
        "doc": relpath(DOC),
        "paper_text_status": payload["paper_text_status"],
    }, indent=2, sort_keys=True))
    return 0 if payload["closeout_complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
