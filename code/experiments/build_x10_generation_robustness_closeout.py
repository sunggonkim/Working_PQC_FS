#!/usr/bin/env python3
"""Build the X10 generation-robustness closeout.

The closeout answers the review question about generation monotonicity without
claiming a broad POSIX/concurrency proof. It combines retained final-binary
fault matrices, the nonce/generation verdict, mounted writer stress, and source
guards for generation reservation, commit visibility, and near-wrap failure.
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = ROOT / "artifacts" / "reports" / "x10_generation_robustness_closeout"

GEN_MATRIX = ROOT / "artifacts" / "validation" / "generation_fault_matrix" / "generation_fault_matrix.json"
NONCE_VERDICT = ROOT / "artifacts" / "validation" / "nonce_generation_fault_verdict" / "nonce_generation_fault_verdict.json"
CONCURRENCY_SUMMARY = ROOT / "artifacts" / "validation" / "concurrency_contract" / "lock_profile_summary.json"
CONCURRENCY_CONTRACT = ROOT / "artifacts" / "validation" / "concurrency_contract" / "concurrency_contract.json"

STATE_H = ROOT / "code" / "storage" / "pqc_state.h"
WRITEBACK_C = ROOT / "code" / "storage" / "pqc_writeback.c"
CHECKPOINT_C = ROOT / "code" / "storage" / "pqc_checkpoint.c"
JOURNAL_C = ROOT / "code" / "storage" / "pqc_journal.c"
EPOCH_LOG_C = ROOT / "code" / "storage" / "pqc_epoch_log.c"

DESIGN_TEX = ROOT / "Paper" / "3_Design.tex"
EVAL_TEX = ROOT / "Paper" / "4_Evaluation.tex"
DISCUSSION_TEX = ROOT / "Paper" / "10_Discussion_and_Limitations.tex"
CHECKLIST = ROOT / "SUBMISSION_CHECKLIST.md"


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{path} did not contain a JSON object")
    return data


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def contains_all(path: Path, snippets: list[str]) -> dict[str, Any]:
    text = read_text(path)
    present = {snippet: snippet in text for snippet in snippets}
    return {
        "path": rel(path),
        "present": present,
        "complete": all(present.values()),
    }


def source_guard() -> dict[str, Any]:
    guards = {
        "state_contract": contains_all(STATE_H, [
            "next_generation is a high-water mark",
            "strict mode now reserves generation ranges before ciphertext can be written",
            "committed_generation is the reader-visible journal boundary",
            "publish_ticket serializes read-modify-publish turns",
        ]),
        "reservation_and_wrap": contains_all(WRITEBACK_C, [
            "snapshot->state->next_generation >",
            "UINT64_MAX - (uint64_t)block_count",
            "return -EFBIG;",
            "first_generation = snapshot->state->next_generation + 1;",
            "reserved_generation = snapshot->state->next_generation + (uint64_t)block_count;",
            "pqc_checkpoint_reserve_generation",
            "snapshot->state->next_generation = reserved_generation",
            "snapshot->state->committed_generation = reserved_generation",
            "publish_turn_finish_locked",
        ]),
        "reservation_persistence": contains_all(CHECKPOINT_C, [
            "pqc_checkpoint_reserve_generation",
            "pqc_publish_checkpoint_store_xattr",
            "generation_reservation_xattr_after",
            "Reservation records keep generation reuse from moving backward",
        ]),
        "strict_highwater": contains_all(JOURNAL_C, [
            "#define PQC_JOURNAL_HIGHWATER_BLOCK UINT64_MAX",
            "pqc_journal_append_highwater_unsynced",
            "pqc_journal_tail_highwater_generation",
            "record.mapping.generation <= max_generation",
            "record.mapping.generation > best.generation",
        ]),
        "epoch_duplicate_rejection": contains_all(EPOCH_LOG_C, [
            "a->file_id == b->file_id",
            "a->logical_block == b->logical_block",
            "a->generation == b->generation",
            "++summary.duplicate_generation_records",
            "rc = -EEXIST;",
        ]),
    }
    return {
        "guards": guards,
        "complete": all(item["complete"] for item in guards.values()),
    }


def generation_evidence() -> dict[str, Any]:
    matrix = load_json(GEN_MATRIX)
    verdict = load_json(NONCE_VERDICT)
    rows = matrix.get("rows") if isinstance(matrix.get("rows"), list) else []
    cases = {str(row.get("case")): row for row in rows if isinstance(row, dict)}
    required_cases = [
        "self_test_older_generation_regression",
        "partial_update_and_remount",
        "torn_journal_write",
        "reserved_generation_skip_after_data_fsync_fault",
        "older_generation_append_after_newer_mapping",
        "stale_snapshot_replay_file_anchor_negative_control",
        "stale_snapshot_replay_tpm_anchor_existing_artifact",
    ]
    checks = {
        "matrix_overall_pass": matrix.get("overall_pass") is True,
        "no_generated_nonce_reuse": matrix.get("no_generated_nonce_reuse") is True,
        "no_silent_corruption": matrix.get("no_silent_corruption") is True,
        "unexpected_liveness_failures_zero": int(matrix.get("unexpected_liveness_failures", 1)) == 0,
        "required_cases_present": all(case in cases for case in required_cases),
        "nonce_verdict_pass": verdict.get("overall_pass") is True,
    }
    return {
        "matrix": rel(GEN_MATRIX),
        "nonce_verdict": rel(NONCE_VERDICT),
        "required_cases": required_cases,
        "checks": checks,
        "complete": all(checks.values()),
    }


def concurrency_evidence() -> dict[str, Any]:
    summary = load_json(CONCURRENCY_SUMMARY)
    contract = load_json(CONCURRENCY_CONTRACT)
    workload = summary.get("workload") if isinstance(summary.get("workload"), dict) else {}
    coverage = workload.get("coverage") if isinstance(workload.get("coverage"), dict) else {}
    phases = workload.get("phases") if isinstance(workload.get("phases"), list) else []
    errors = workload.get("errors") if isinstance(workload.get("errors"), list) else []
    timed_out = [phase for phase in phases if isinstance(phase, dict) and phase.get("timed_out")]
    worker_errors = [
        phase for phase in phases
        if isinstance(phase, dict) and phase.get("worker_errors")
    ]
    checks = {
        "summary_pass": summary.get("overall_pass") is True,
        "contract_pass": contract.get("overall_pass") is True,
        "same_file_threads_4": int(coverage.get("max_thread_count", 0) or 0) >= 4,
        "process_clients_4": int(coverage.get("max_process_client_count", 0) or 0) >= 4,
        "same_and_disjoint_phases": (
            int(coverage.get("same_file_writer_phases", 0) or 0) >= 3
            and int(coverage.get("disjoint_writer_phases", 0) or 0) >= 3
        ),
        "no_errors_or_timeouts": not errors and not timed_out and not worker_errors,
    }
    return {
        "summary": rel(CONCURRENCY_SUMMARY),
        "contract": rel(CONCURRENCY_CONTRACT),
        "coverage": coverage,
        "checks": checks,
        "complete": all(checks.values()),
        "residual": "Mounted writer stress exercises same-file/disjoint writers, not arbitrary application-level concurrency.",
    }


def paper_guard() -> dict[str, Any]:
    text = "\n".join(read_text(path) for path in [DESIGN_TEX, EVAL_TEX, DISCUSSION_TEX])
    required = {
        "publish_ticket_serialization": "publish-ticket" in text or "publish_ticket" in text,
        "near_wrap_policy": (
            ("UINT64_MAX" in text or "UINT64\\_MAX" in text)
            and "EFBIG" in text
        ),
        "stress_scope": "same-file/disjoint" in text and "4-thread/4-process" in text,
        "not_broad_concurrency": "not arbitrary application-level concurrency" in text,
        "not_power_loss": "physical power-loss" in text,
    }
    return {
        "paper_files": [rel(DESIGN_TEX), rel(EVAL_TEX), rel(DISCUSSION_TEX)],
        "required": required,
        "complete": all(required.values()),
    }


def checklist_guard() -> dict[str, Any]:
    text = read_text(CHECKLIST)
    required = {
        "x10_done": "| X10 | DONE |" in text,
        "x11_tracked": "| X11 | NEXT |" in text or "| X11 | DONE |" in text,
        "artifact_named": "x10_generation_robustness_closeout.json" in text,
    }
    return {
        "source": rel(CHECKLIST),
        "required": required,
        "complete": all(required.values()),
    }


def build_report() -> dict[str, Any]:
    source = source_guard()
    generation = generation_evidence()
    concurrency = concurrency_evidence()
    paper = paper_guard()
    checklist = checklist_guard()
    checks = {
        "source_complete": source["complete"],
        "generation_evidence_complete": generation["complete"],
        "concurrency_evidence_complete": concurrency["complete"],
        "paper_complete": paper["complete"],
        "checklist_complete": checklist["complete"],
    }
    overall = all(checks.values())
    return {
        "artifact": "x10_generation_robustness_closeout",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "scope": (
            "Generation monotonicity closeout for the current mounted runtime: "
            "per-file reservation/commit state, retained final-binary fault rows, "
            "mounted writer stress, and near-wrap fail-closed source guard."
        ),
        "source_guard": source,
        "generation_evidence": generation,
        "concurrency_evidence": concurrency,
        "paper_guard": paper,
        "checklist_guard": checklist,
        "checks": checks,
        "overall_pass": overall,
        "verdict": (
            "X10 is closed under the current claim boundary: generation ranges "
            "are per-file serialized, reserved before encryption, advanced by "
            "checkpoint/high-water state, rejected near UINT64_MAX, and tested "
            "against retained fault and bounded writer-stress rows."
            if overall else
            "X10 is not closed; inspect failed checks before claiming generation robustness."
        ),
        "residual_risk": (
            "This is not a physical power-loss proof, full POSIX concurrency proof, "
            "or long-running wraparound exhaustion campaign."
        ),
    }


def write_markdown(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# X10 generation robustness closeout",
        "",
        f"Overall pass: `{report['overall_pass']}`",
        "",
        "## Verdict",
        "",
        report["verdict"],
        "",
        "## Checks",
        "",
    ]
    for name, ok in report["checks"].items():
        lines.append(f"- `{name}`: `{ok}`")
    lines.extend([
        "",
        "## Residual risk",
        "",
        report["residual_risk"],
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    report = build_report()
    json_path = args.out_dir / "x10_generation_robustness_closeout.json"
    md_path = args.out_dir / "x10_generation_robustness_closeout.md"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown(report, md_path)
    print(json.dumps({
        "overall_pass": report["overall_pass"],
        "json": rel(json_path),
        "markdown": rel(md_path),
        "failed_checks": [name for name, ok in report["checks"].items() if not ok],
    }, indent=2, sort_keys=True))
    return 0 if report["overall_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
