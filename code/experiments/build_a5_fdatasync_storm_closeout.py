#!/usr/bin/env python3
"""Build the Gate A5 VFS/FUSE and fdatasync-storm closeout.

The closeout does not claim a kernel bypass. It binds current evidence to a
small set of decisions: ordinary FUSE is accepted and measured through a
daemon-side proxy; strict per-write data/journal fdatasync is the cost boundary;
epoch/group commit amortizes a subset of publication syncs; eBPF/io_uring
completion bypass remains a non-claim.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = ROOT / "artifacts" / "validation" / "vfs_ebpf_fdatasync_storm"

A2 = ROOT / "artifacts" / "validation" / "a2_gap_attribution" / "a2_gap_attribution.json"
A3 = ROOT / "artifacts" / "validation" / "a3_batching_decision" / "a3_batching_decision.json"
A4 = ROOT / "artifacts" / "validation" / "a4_hidden_overhead_accounting" / "a4_hidden_overhead_closeout.json"
A4_SMOKE = ROOT / "artifacts" / "validation" / "a4_hidden_overhead_accounting" / "a4_overhead_trace_smoke.json"
PUB_CLOSEOUT = ROOT / "artifacts" / "validation" / "publication_protocol_fault_matrix" / "publication_protocol_closeout.json"
PUB_COMPARISON = ROOT / "artifacts" / "validation" / "publication_protocol_fault_matrix" / "epoch_publication_comparison.json"
REPLAY_MATRIX = ROOT / "artifacts" / "validation" / "publication_protocol_fault_matrix" / "epoch_replay_fault_matrix.json"
EBPF_AUDIT = ROOT / "artifacts" / "validation" / "ebpf_iouring_scope_audit" / "ebpf_iouring_scope_audit.json"

STRICT_PUBLISH = ROOT / "code" / "storage" / "pqc_strict_publish.c"
EPOCH_PUBLISH = ROOT / "code" / "storage" / "pqc_epoch_publish.c"
FUSE_SOURCE = ROOT / "code" / "frontend" / "pqc_fuse.c"
DESIGN_TEX = ROOT / "Paper" / "3_Design.tex"
EVAL_TEX = ROOT / "Paper" / "4_Evaluation.tex"


def now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT))


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def number(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def case_by_label(report: dict[str, Any], label: str) -> dict[str, Any]:
    for case in report.get("cases", []):
        if case.get("label") == label:
            return case
    return {}


def fuse_op(smoke: dict[str, Any], name: str) -> dict[str, Any]:
    for op in smoke.get("fuse_trace", {}).get("operations", []):
        if isinstance(op, dict) and op.get("op") == name:
            return op
    return {}


def avg_ns(op: dict[str, Any]) -> int:
    calls = int(number(op.get("calls")))
    if calls <= 0:
        return 0
    return int(number(op.get("total_ns"))) // calls


def summarize_publication(pub: dict[str, Any]) -> dict[str, Any]:
    strict = case_by_label(pub, "strict")
    epoch = case_by_label(pub, "epoch_redo_log")
    strict_grouped = case_by_label(pub, "strict_grouped")
    epoch_grouped = case_by_label(pub, "epoch_redo_log_grouped")
    def pub_summary(case: dict[str, Any]) -> dict[str, Any]:
        summary = case.get("measured_publication_summary", {})
        return {
            "sync_count_total": case.get("sync_count_total"),
            "sync_count_per_measured_op": case.get("sync_count_per_measured_op"),
            "throughput_mib_s": case.get("throughput_mib_s"),
            "p99_ns": case.get("client_write_fsync_latency_ns", {}).get("p99_ns"),
            "data_fsync_count_total": summary.get("data_fsync_count_total"),
            "journal_fsync_count_total": summary.get("journal_fsync_count_total"),
            "epoch_log_fsync_count_total": summary.get("epoch_log_fsync_count_total"),
        }
    return {
        "strict": pub_summary(strict),
        "epoch_redo_log": pub_summary(epoch),
        "strict_grouped": pub_summary(strict_grouped),
        "epoch_redo_log_grouped": pub_summary(epoch_grouped),
    }


def source_scan() -> dict[str, Any]:
    strict = read(STRICT_PUBLISH)
    epoch = read(EPOCH_PUBLISH)
    fuse = read(FUSE_SOURCE)
    return {
        "files": [rel(STRICT_PUBLISH), rel(EPOCH_PUBLISH), rel(FUSE_SOURCE)],
        "strict_has_data_fdatasync": "PQC_DURABILITY_SITE_DATA_SIDECAR" in strict
        and "pqc_durability_fdatasync" in strict,
        "strict_has_journal_fdatasync": "PQC_DURABILITY_SITE_JOURNAL_SIDECAR" in strict
        and "pqc_durability_fdatasync" in strict,
        "epoch_skips_foreground_journal_fsync": "skip_journal_fsync" in epoch,
        "epoch_has_group_syncfs_barrier": "syncfs" in epoch
        and "pqc_epoch_publish_sync_barrier" in epoch,
        "fuse_has_no_ebpf_iouring_path": "io_uring" not in fuse.lower()
        and "ebpf" not in fuse.lower(),
    }


def paper_scan() -> dict[str, Any]:
    design = read(DESIGN_TEX)
    evaluation = read(EVAL_TEX)
    combined = design + "\n" + evaluation
    dangerous = [
        "implements eBPF",
        "implements io_uring",
        "kernel-bypass publication",
        "eliminates per-write fdatasync",
        "removes all fdatasync",
        "kernel context-switch proof",
    ]
    return {
        "files": [rel(DESIGN_TEX), rel(EVAL_TEX)],
        "states_cost_boundary": "cost boundary for authenticated publication" in evaluation,
        "states_mounted_ebpf_not_established": "Mounted eBPF/io\\_uring path" in evaluation
        or "No eBPF notification path is part of the mounted implementation" in design,
        "states_strict_single_client_case": (
            "single-client per-write \\texttt{fdatasync} remains a strict-mode case"
            in design
        ),
        "states_sync_amortization": "sync-count amortization" in design,
        "unscoped_dangerous_hits": [phrase for phrase in dangerous if phrase in combined],
    }


def classify(a2: dict[str, Any], a3: dict[str, Any], a4: dict[str, Any],
             a4_smoke: dict[str, Any], pub_closeout: dict[str, Any],
             pub_comparison: dict[str, Any], replay: dict[str, Any],
             ebpf: dict[str, Any]) -> list[dict[str, Any]]:
    pub_summary = summarize_publication(pub_comparison)
    a2_durability = a2.get("summary", {}).get("durability_calls", {})
    pub_measured = pub_closeout.get("measured_result", {})
    a3_derived = a3.get("derived", {})
    a4_classes = a4.get("overhead_classification", [])
    return [
        {
            "path": "ordinary mounted FUSE path",
            "classification": "accepted-measured",
            "evidence": [
                f"fuse_create_avg_ns={avg_ns(fuse_op(a4_smoke, 'create'))}",
                f"fuse_write_avg_ns={avg_ns(fuse_op(a4_smoke, 'write'))}",
                f"fuse_fsync_avg_ns={avg_ns(fuse_op(a4_smoke, 'fsync'))}",
                "daemon-side proxy, not kernel context-switch count",
            ],
            "guard": "Do not claim FUSE bypass or kernel scheduler context-switch evidence.",
        },
        {
            "path": "eBPF/io_uring completion bypass",
            "classification": "not-claimed",
            "evidence": [
                f"ebpf_audit_overall_pass={ebpf.get('overall_pass')}",
                f"decision={ebpf.get('decision')}",
                "mounted FUSE source has no io_uring/eBPF completion path",
            ],
            "guard": "eBPF/io_uring remains diagnostic or future work, not a mounted-path mechanism.",
        },
        {
            "path": "strict per-write data/journal fdatasync storm",
            "classification": "accepted-cost-boundary",
            "evidence": [
                f"a2_fdatasync={a2_durability.get('fdatasync')}",
                f"a2_data_sidecar={a2_durability.get('data_sidecar')}",
                f"a2_journal_sidecar={a2_durability.get('journal_sidecar')}",
                f"strict_sync_count_per_op={pub_summary['strict']['sync_count_per_measured_op']}",
                f"strict_journal_fsync_total={pub_summary['strict']['journal_fsync_count_total']}",
            ],
            "guard": "Strict mode is a correctness boundary; do not present it as optimized throughput.",
        },
        {
            "path": "epoch foreground journal fdatasync",
            "classification": "eliminated-from-foreground-epoch-path",
            "evidence": [
                f"sequential_epoch_journal_fsync_removed={pub_measured.get('sequential_epoch_journal_fsync_removed')}",
                f"epoch_journal_fsync_total={pub_summary['epoch_redo_log']['journal_fsync_count_total']}",
                f"epoch_log_fsync_total={pub_summary['epoch_redo_log']['epoch_log_fsync_count_total']}",
            ],
            "guard": "This removes strict journal fdatasync from foreground epoch publication, not all durability barriers.",
        },
        {
            "path": "grouped epoch syncfs barrier",
            "classification": "amortized-for-grouped-work",
            "evidence": [
                f"grouped_sync_amortized={pub_measured.get('grouped_sync_amortized')}",
                f"grouped_sync_reduction_percent={pub_measured.get('grouped_sync_reduction_percent')}",
                f"a3_current_grouped_sync_count_reduced={a3.get('proof_checks', {}).get('current_grouped_sync_count_reduced')}",
                f"strict_grouped_sync_total={pub_summary['strict_grouped']['sync_count_total']}",
                f"epoch_grouped_sync_total={pub_summary['epoch_redo_log_grouped']['sync_count_total']}",
            ],
            "guard": "Scope to grouped/concurrent mounted work; single-client frozen rows remain cost-boundary evidence.",
        },
        {
            "path": "kernel context-switch count",
            "classification": "unavailable-not-claimed",
            "evidence": [
                "A4 records daemon-side FUSE operation latency only",
                "A2 records fio-client syscall counts, not daemon scheduler switches",
            ],
            "guard": "Do not convert daemon-side latency or fio-client syscall counts into context-switch counts.",
        },
        {
            "path": "correctness/fault boundary for amortization",
            "classification": "measured",
            "evidence": [
                f"replay_fault_matrix_pass={replay.get('overall_pass')}",
                f"a3_replay_fault_matrix_passes={a3.get('proof_checks', {}).get('replay_fault_matrix_passes')}",
                f"a4_measured_classes={sum(1 for row in a4_classes if row.get('classification') == 'measured')}",
            ],
            "guard": "Replay/fault evidence bounds epoch publication; it is not physical power-loss certification.",
        },
    ]


def build_report() -> dict[str, Any]:
    a2 = load_json(A2)
    a3 = load_json(A3)
    a4 = load_json(A4)
    a4_smoke = load_json(A4_SMOKE)
    pub_closeout = load_json(PUB_CLOSEOUT)
    pub_comparison = load_json(PUB_COMPARISON)
    replay = load_json(REPLAY_MATRIX)
    ebpf = load_json(EBPF_AUDIT)
    source = source_scan()
    paper = paper_scan()
    rows = classify(a2, a3, a4, a4_smoke, pub_closeout, pub_comparison, replay, ebpf)
    proof_checks = {
        "a2_pass": bool(a2.get("overall_pass")),
        "a3_pass": bool(a3.get("overall_pass")),
        "a4_pass": bool(a4.get("overall_pass")),
        "publication_closeout_complete": bool(pub_closeout.get("closeout_complete")),
        "publication_comparison_pass": bool(pub_comparison.get("overall_pass")),
        "replay_fault_matrix_pass": bool(replay.get("overall_pass")),
        "ebpf_iouring_audit_pass": bool(ebpf.get("overall_pass")),
        "source_guards_present": all(
            source[key]
            for key in (
                "strict_has_data_fdatasync",
                "strict_has_journal_fdatasync",
                "epoch_skips_foreground_journal_fsync",
                "epoch_has_group_syncfs_barrier",
                "fuse_has_no_ebpf_iouring_path",
            )
        ),
        "paper_scope_guards_present": paper["states_cost_boundary"]
        and paper["states_mounted_ebpf_not_established"]
        and paper["states_strict_single_client_case"]
        and paper["states_sync_amortization"],
        "no_unscoped_dangerous_paper_hits": not paper["unscoped_dangerous_hits"],
    }
    return {
        "overall_pass": all(proof_checks.values()),
        "schema": "a5-fdatasync-storm-closeout-v1",
        "generated_utc": now_utc(),
        "inputs": {
            "a2_gap_attribution": rel(A2),
            "a3_batching_decision": rel(A3),
            "a4_hidden_overhead_closeout": rel(A4),
            "a4_overhead_trace_smoke": rel(A4_SMOKE),
            "publication_protocol_closeout": rel(PUB_CLOSEOUT),
            "publication_comparison": rel(PUB_COMPARISON),
            "replay_fault_matrix": rel(REPLAY_MATRIX),
            "ebpf_iouring_scope_audit": rel(EBPF_AUDIT),
        },
        "path_classification": rows,
        "source_scan": source,
        "paper_scan": paper,
        "proof_checks": proof_checks,
        "paper_text_status": "already_scoped_no_update",
        "parent_checklist_closed": False,
        "non_claims": [
            "no mounted eBPF/io_uring completion bypass",
            "no kernel context-switch count",
            "no claim that epoch mode removes all fdatasync/syncfs barriers",
            "no general fast-filesystem throughput claim",
            "no physical power-loss certification",
        ],
    }


def markdown(report: dict[str, Any]) -> str:
    lines = [
        "# A5 VFS/FUSE and fdatasync Storm Closeout",
        "",
        f"- Overall pass: `{report['overall_pass']}`",
        f"- Paper text status: `{report['paper_text_status']}`",
        f"- Parent checklist closed: `{report['parent_checklist_closed']}`",
        "",
        "## Path Classification",
        "",
        "| Path | Classification | Evidence | Guard |",
        "| --- | --- | --- | --- |",
    ]
    for row in report["path_classification"]:
        evidence = "<br>".join(f"`{item}`" for item in row["evidence"])
        lines.append(
            f"| {row['path']} | `{row['classification']}` | {evidence} | {row['guard']} |"
        )
    lines.extend(["", "## Proof Checks", ""])
    for key, value in report["proof_checks"].items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(["", "## Non-Claims", ""])
    for item in report["non_claims"]:
        lines.append(f"- {item}")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    out_dir = args.out_dir if args.out_dir.is_absolute() else ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    report = build_report()
    json_path = out_dir / "a5_fdatasync_storm_closeout.json"
    md_path = out_dir / "a5_fdatasync_storm_closeout.md"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    md_path.write_text(markdown(report), encoding="utf-8")
    counts: dict[str, int] = {}
    for row in report["path_classification"]:
        counts[row["classification"]] = counts.get(row["classification"], 0) + 1
    print(json.dumps({
        "overall_pass": report["overall_pass"],
        "json": str(json_path),
        "markdown": str(md_path),
        "classification_counts": counts,
    }, indent=2, sort_keys=True))
    return 0 if report["overall_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
