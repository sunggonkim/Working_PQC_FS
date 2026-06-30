#!/usr/bin/env python3
"""Build the Gate A3 batching/redesign boundary decision.

This is a narrow closeout for the already-implemented epoch/group commit path.
It reads current production source, retained mounted-path comparisons, replay
fault evidence, and paper text. It does not tune the publication protocol.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = ROOT / "artifacts" / "validation" / "a3_batching_decision"

SOURCE_FILES = {
    "epoch_publish": ROOT / "code" / "storage" / "pqc_epoch_publish.c",
    "epoch_log": ROOT / "code" / "storage" / "pqc_epoch_log.c",
    "strict_publish": ROOT / "code" / "storage" / "pqc_strict_publish.c",
    "writeback": ROOT / "code" / "storage" / "pqc_writeback.c",
    "fd_context": ROOT / "code" / "fs" / "pqc_fd_context.c",
    "parallel_commit": ROOT / "code" / "fs" / "pqc_parallel_commit.c",
}

ARTIFACTS = {
    "current_viability": ROOT
    / "artifacts"
    / "validation"
    / "filesystem_viability_breakdown"
    / "epoch_publication_comparison.json",
    "wait1ms_viability": ROOT
    / "artifacts"
    / "validation"
    / "filesystem_viability_breakdown_wait1ms"
    / "epoch_publication_comparison.json",
    "fault_matrix_publication": ROOT
    / "artifacts"
    / "validation"
    / "publication_protocol_fault_matrix"
    / "epoch_publication_comparison.json",
    "replay_fault_matrix": ROOT
    / "artifacts"
    / "validation"
    / "publication_protocol_fault_matrix"
    / "epoch_replay_fault_matrix.json",
    "parallel_commit_closure": ROOT
    / "artifacts"
    / "validation"
    / "parallel_commit_contract"
    / "parallel_commit_closure_audit.json",
    "a1_decision": ROOT
    / "artifacts"
    / "validation"
    / "a1_throughput_decision"
    / "a1_throughput_decision.json",
}

PAPER_FILES = {
    "design": ROOT / "Paper" / "3_Design.tex",
    "evaluation": ROOT / "Paper" / "4_Evaluation.tex",
    "discussion": ROOT / "Paper" / "10_Discussion_and_Limitations.tex",
}


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT))


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def case_by_label(report: dict[str, Any], label: str) -> dict[str, Any]:
    for case in report.get("cases", []):
        if case.get("label") == label:
            return case
    return {}


def number(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def summarize_comparison(path: Path) -> dict[str, Any]:
    report = load_json(path)
    strict = case_by_label(report, "strict")
    epoch = case_by_label(report, "epoch_redo_log")
    strict_grouped = case_by_label(report, "strict_grouped")
    epoch_grouped = case_by_label(report, "epoch_redo_log_grouped")

    strict_grouped_sync = number(strict_grouped.get("sync_count_total"))
    epoch_grouped_sync = number(epoch_grouped.get("sync_count_total"))
    strict_grouped_tput = number(strict_grouped.get("throughput_mib_s"))
    epoch_grouped_tput = number(epoch_grouped.get("throughput_mib_s"))
    strict_grouped_p99 = number(
        strict_grouped.get("client_write_fsync_latency_ns", {}).get("p99_ns")
    )
    epoch_grouped_p99 = number(
        epoch_grouped.get("client_write_fsync_latency_ns", {}).get("p99_ns")
    )
    strict_single_tput = number(strict.get("throughput_mib_s"))
    epoch_single_tput = number(epoch.get("throughput_mib_s"))

    return {
        "path": rel(path),
        "overall_pass": bool(report.get("overall_pass")),
        "verdict": report.get("comparison_verdict"),
        "strict_single": {
            "throughput_mib_s": strict_single_tput,
            "sync_count_total": strict.get("sync_count_total"),
            "sync_count_per_measured_op": strict.get("sync_count_per_measured_op"),
            "journal_fsync_count_total": strict.get(
                "measured_publication_summary", {}
            ).get("journal_fsync_count_total"),
        },
        "epoch_single": {
            "throughput_mib_s": epoch_single_tput,
            "sync_count_total": epoch.get("sync_count_total"),
            "sync_count_per_measured_op": epoch.get("sync_count_per_measured_op"),
            "journal_fsync_count_total": epoch.get(
                "measured_publication_summary", {}
            ).get("journal_fsync_count_total"),
            "epoch_log_fsync_count_total": epoch.get(
                "measured_publication_summary", {}
            ).get("epoch_log_fsync_count_total"),
        },
        "strict_grouped": {
            "throughput_mib_s": strict_grouped_tput,
            "sync_count_total": strict_grouped.get("sync_count_total"),
            "sync_count_per_measured_op": strict_grouped.get(
                "sync_count_per_measured_op"
            ),
            "p99_ns": strict_grouped_p99,
            "client_count": strict_grouped.get("client_count"),
        },
        "epoch_grouped": {
            "throughput_mib_s": epoch_grouped_tput,
            "sync_count_total": epoch_grouped.get("sync_count_total"),
            "sync_count_per_measured_op": epoch_grouped.get(
                "sync_count_per_measured_op"
            ),
            "p99_ns": epoch_grouped_p99,
            "client_count": epoch_grouped.get("client_count"),
            "max_epoch_append_group_size": epoch_grouped.get("trace", {}).get(
                "epoch_append_group_size_max"
            ),
            "epoch_sync_primitives": epoch_grouped.get("trace", {}).get(
                "epoch_append_sync_primitives"
            ),
            "journal_fsync_count_total": epoch_grouped.get(
                "measured_publication_summary", {}
            ).get("journal_fsync_count_total"),
            "epoch_log_fsync_count_total": epoch_grouped.get(
                "measured_publication_summary", {}
            ).get("epoch_log_fsync_count_total"),
        },
        "derived": {
            "grouped_sync_count_reduced": epoch_grouped_sync < strict_grouped_sync,
            "grouped_throughput_improved": epoch_grouped_tput > strict_grouped_tput,
            "grouped_p99_improved": (
                epoch_grouped_p99 > 0
                and strict_grouped_p99 > 0
                and epoch_grouped_p99 < strict_grouped_p99
            ),
            "single_epoch_not_general_win": epoch_single_tput <= strict_single_tput,
            "epoch_grouped_journal_fsync_removed": number(
                epoch_grouped.get("measured_publication_summary", {}).get(
                    "journal_fsync_count_total"
                )
            )
            == 0,
        },
    }


def scan_source() -> dict[str, Any]:
    text = {name: read(path) for name, path in SOURCE_FILES.items()}
    return {
        "files": {name: rel(path) for name, path in SOURCE_FILES.items()},
        "epoch_mode_flag": "PQC_PUBLICATION_MODE_EPOCH_REDO_LOG"
        in text["epoch_publish"],
        "epoch_group_config": "PQC_EPOCH_GROUP_MAX" in text["epoch_publish"]
        and "PQC_EPOCH_GROUP_WAIT_NS" in text["epoch_publish"],
        "leader_follower_group_barrier": "group_role" in text["epoch_publish"]
        and "leader" in text["epoch_publish"]
        and "follower" in text["epoch_publish"],
        "shared_sync_barrier": "syncfs" in text["epoch_publish"]
        and "pqc_epoch_publish_sync_barrier" in text["epoch_publish"],
        "journal_fsync_can_be_skipped": "skip_journal_fsync" in text["epoch_publish"]
        and "journal_sidecar_epoch_repairable" in text["strict_publish"],
        "epoch_log_append_commit_records": "PQC_EPOCH_LOG_RECORD_COMMIT"
        in text["epoch_log"]
        and "pqc_epoch_log_append_records" in text["epoch_log"],
        "epoch_replay_and_compaction": "pqc_epoch_log_replay_fd"
        in text["epoch_log"]
        and "pqc_epoch_log_compact_checkpoint" in text["epoch_log"]
        and "pqc_epoch_log_compact_checkpoint" in text["fd_context"],
        "parallel_commit_runtime_integrated": "pqc_parallel_commit_runtime_begin"
        in text["writeback"]
        and "pqc_parallel_commit_runtime_finish" in text["writeback"],
        "parallel_commit_shards": "PQC_PARALLEL_COMMIT_MAX_SHARDS"
        in text["parallel_commit"]
        and "shard_count" in text["parallel_commit"],
    }


def scan_paper() -> dict[str, Any]:
    text = {name: read(path) for name, path in PAPER_FILES.items()}
    combined = "\n".join(text.values())
    return {
        "files": {name: rel(path) for name, path in PAPER_FILES.items()},
        "design_names_bounded_group": "bounded leader/follower epoch group"
        in text["design"],
        "design_names_sync_amortization": "sync-count amortization" in text["design"],
        "design_names_single_client_loss_case": (
            "single-client per-write \\texttt{fdatasync} remains a strict-mode case"
            in text["design"]
        ),
        "evaluation_cost_boundary": "cost boundary for authenticated publication"
        in text["evaluation"],
        "evaluation_not_headline_win": "not the headline win" in text["evaluation"],
        "discussion_no_high_throughput_claim": (
            "not to rank deployed encryption systems" in text["discussion"]
            and "high-throughput general-purpose encryption" in text["discussion"]
        ),
        "dangerous_general_win_claim": (
            "epoch mode always" in combined.lower()
            or "general-purpose filesystem replacement" in combined.lower()
            or "high-throughput general-purpose filesystem" in combined.lower()
        ),
    }


def build_report() -> dict[str, Any]:
    source = scan_source()
    paper = scan_paper()
    current = summarize_comparison(ARTIFACTS["current_viability"])
    wait1ms = summarize_comparison(ARTIFACTS["wait1ms_viability"])
    fault_publication = summarize_comparison(ARTIFACTS["fault_matrix_publication"])
    replay = load_json(ARTIFACTS["replay_fault_matrix"])
    parallel = load_json(ARTIFACTS["parallel_commit_closure"])
    a1 = load_json(ARTIFACTS["a1_decision"])

    comparisons = {
        "current_100us_viability": current,
        "wait1ms_viability": wait1ms,
        "fault_matrix_publication": fault_publication,
    }
    grouped_sync_reduced_somewhere = any(
        item["derived"]["grouped_sync_count_reduced"]
        for item in comparisons.values()
    )
    grouped_sync_reduced_current = current["derived"]["grouped_sync_count_reduced"]
    grouped_has_win_and_loss = any(
        item["derived"]["grouped_throughput_improved"]
        or item["derived"]["grouped_p99_improved"]
        for item in comparisons.values()
    ) and any(
        not item["derived"]["grouped_throughput_improved"]
        or not item["derived"]["grouped_p99_improved"]
        for item in comparisons.values()
    )
    single_not_general_win = any(
        item["derived"]["single_epoch_not_general_win"]
        for item in comparisons.values()
    )

    proof_checks = {
        "source_epoch_group_commit_present": all(
            source[key]
            for key in (
                "epoch_mode_flag",
                "epoch_group_config",
                "leader_follower_group_barrier",
                "shared_sync_barrier",
                "journal_fsync_can_be_skipped",
                "epoch_log_append_commit_records",
                "epoch_replay_and_compaction",
                "parallel_commit_runtime_integrated",
            )
        ),
        "current_grouped_sync_count_reduced": grouped_sync_reduced_current,
        "grouped_evidence_has_scoped_win_and_loss": grouped_has_win_and_loss,
        "single_or_frozen_path_not_general_win": single_not_general_win
        and bool(a1.get("overall_pass")),
        "replay_fault_matrix_passes": bool(replay.get("overall_pass")),
        "parallel_commit_closure_passes": bool(parallel.get("overall_pass"))
        and parallel.get("closure_verdict") == "closed",
        "paper_scopes_batching_claim": all(
            paper[key]
            for key in (
                "design_names_bounded_group",
                "design_names_sync_amortization",
                "design_names_single_client_loss_case",
                "evaluation_cost_boundary",
                "evaluation_not_headline_win",
                "discussion_no_high_throughput_claim",
            )
        )
        and not paper["dangerous_general_win_claim"],
        "negative_claim_guard_present": all(
            "negative_claim_guard" in load_json(path)
            for path in (
                ARTIFACTS["current_viability"],
                ARTIFACTS["replay_fault_matrix"],
                ARTIFACTS["parallel_commit_closure"],
            )
        ),
    }

    decision = {
        "verdict": "batching-boundary-closeout",
        "what_is_implemented": [
            "opt-in epoch-redo-log publication mode",
            "leader/follower bounded group barrier",
            "shared syncfs barrier for joined epoch groups",
            "journal-sidecar fsync removal from foreground epoch path",
            "checkpoint compaction and committed-prefix replay",
            "parallel commit coordinator integrated behind an explicit mode flag",
        ],
        "where_it_helps": (
            "Concurrent grouped mounted writes can reduce traced publication "
            "sync count and can improve throughput or tail latency in the "
            "runs where the grouped barrier is actually formed."
        ),
        "where_it_loses_or_is_not_a_claim": (
            "Single-client per-write fdatasync and frozen strict-path rows "
            "remain the authenticated-publication cost boundary. Evidence also "
            "contains grouped runs where throughput or p99 does not improve, so "
            "the paper must not claim a general fast-path or filesystem-wide "
            "throughput win."
        ),
        "next_valid_code_work": (
            "Only change publication performance further if the production "
            "protocol changes its crash ordering or grouping semantics and the "
            "same change is accompanied by replay/fault proof."
        ),
    }

    return {
        "overall_pass": all(proof_checks.values()),
        "schema": "a3-batching-decision-v1",
        "inputs": {name: rel(path) for name, path in ARTIFACTS.items()},
        "source_checks": source,
        "paper_checks": paper,
        "comparisons": comparisons,
        "replay_fault_matrix_summary": {
            "path": rel(ARTIFACTS["replay_fault_matrix"]),
            "overall_pass": replay.get("overall_pass"),
            "case_labels": [case.get("label") for case in replay.get("cases", [])],
        },
        "parallel_commit_summary": {
            "path": rel(ARTIFACTS["parallel_commit_closure"]),
            "overall_pass": parallel.get("overall_pass"),
            "closure_verdict": parallel.get("closure_verdict"),
        },
        "a1_cost_boundary_summary": {
            "path": rel(ARTIFACTS["a1_decision"]),
            "overall_pass": a1.get("overall_pass"),
            "verdict": a1.get("decision", {}).get("verdict"),
        },
        "derived": {
            "grouped_sync_reduced_somewhere": grouped_sync_reduced_somewhere,
            "grouped_sync_reduced_current": grouped_sync_reduced_current,
            "grouped_has_win_and_loss": grouped_has_win_and_loss,
            "single_not_general_win": single_not_general_win,
        },
        "proof_checks": proof_checks,
        "decision": decision,
        "non_claims": [
            "not a general-purpose filesystem throughput claim",
            "not a single-client frozen-contract speedup claim",
            "not proof that fdatasync can be removed without crash-ordering evidence",
            "not proof of physical power-loss certification",
        ],
    }


def build_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# A3 Batching Decision",
        "",
        f"- Overall pass: `{report['overall_pass']}`",
        f"- Verdict: `{report['decision']['verdict']}`",
        "",
        "## Decision",
        "",
        f"- Helps: {report['decision']['where_it_helps']}",
        f"- Loss/non-claim: {report['decision']['where_it_loses_or_is_not_a_claim']}",
        f"- Next valid code work: {report['decision']['next_valid_code_work']}",
        "",
        "## Comparison Summary",
        "",
    ]
    for name, comparison in report["comparisons"].items():
        derived = comparison["derived"]
        sg = comparison["strict_grouped"]
        eg = comparison["epoch_grouped"]
        lines.extend(
            [
                f"### {name}",
                "",
                f"- Overall pass: `{comparison['overall_pass']}`",
                f"- Strict grouped sync total: `{sg['sync_count_total']}`",
                f"- Epoch grouped sync total: `{eg['sync_count_total']}`",
                f"- Strict grouped throughput: `{sg['throughput_mib_s']:.3f} MiB/s`",
                f"- Epoch grouped throughput: `{eg['throughput_mib_s']:.3f} MiB/s`",
                f"- Strict grouped p99: `{sg['p99_ns']}` ns",
                f"- Epoch grouped p99: `{eg['p99_ns']}` ns",
                f"- Grouped sync reduced: `{derived['grouped_sync_count_reduced']}`",
                f"- Grouped throughput improved: `{derived['grouped_throughput_improved']}`",
                f"- Grouped p99 improved: `{derived['grouped_p99_improved']}`",
                "",
            ]
        )
    lines.extend(["## Proof Checks", ""])
    for key, value in report["proof_checks"].items():
        lines.append(f"- `{key}`: `{value}`")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    out_dir = args.out_dir if args.out_dir.is_absolute() else ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    report = build_report()
    json_path = out_dir / "a3_batching_decision.json"
    md_path = out_dir / "a3_batching_decision.md"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    md_path.write_text(build_markdown(report), encoding="utf-8")
    print(
        json.dumps(
            {
                "overall_pass": report["overall_pass"],
                "json": str(json_path),
                "markdown": str(md_path),
                "verdict": report["decision"]["verdict"],
                "grouped_sync_reduced_current": report["derived"][
                    "grouped_sync_reduced_current"
                ],
                "grouped_has_win_and_loss": report["derived"][
                    "grouped_has_win_and_loss"
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if report["overall_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
