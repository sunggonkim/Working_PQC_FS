#!/usr/bin/env python3
"""Audit retained recovery trials for explicit oracle verdicts.

The audit is intentionally conservative: it does not rerun recovery workloads.
Instead, it reads retained artifacts that the paper/README cite for recovery or
freshness behavior, normalizes their row-level labels into the checklist's
allowed oracle vocabulary, and fails if any retained trial lacks a
classification.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = ROOT / "artifacts" / "validation" / "recovery_oracle_audit"

ALLOWED_VERDICTS = {
    "previous_committed",
    "latest_committed",
    "fail_closed",
    "silent_corruption",
    "unexpected_liveness_failure",
}

NORMALIZE_VERDICT = {
    "previous": "previous_committed",
    "latest": "latest_committed",
    "previous_committed": "previous_committed",
    "latest_committed": "latest_committed",
    "fail_closed": "fail_closed",
    "fail_closed_update_not_committed": "fail_closed",
    "rollback_reject": "fail_closed",
    "transient_probe_rejects_drift": "fail_closed",
    "silent_corruption": "silent_corruption",
    "unexpected_liveness_failure": "unexpected_liveness_failure",
    "unexpected_error": "unexpected_liveness_failure",
    # File-backed replay negative controls expose the previous committed state.
    "rollback_visible": "previous_committed",
}

FORBIDDEN_SUCCESS_PATTERNS = [
    re.compile(r"success[_ -]?rate", re.IGNORECASE),
    re.compile(r"reported" r" as success", re.IGNORECASE),
]

TEXT_GATES = [
    ROOT / "Paper" / "4_Evaluation.tex",
    ROOT / "Paper" / "8_Security_Analysis.tex",
    ROOT / "README.md",
    ROOT / "SUBMISSION_CHECKLIST.md",
    ROOT / "code" / "experiments" / "build_tpm_recovery_verdict.py",
    ROOT / "code" / "experiments" / "build_crash_audit_report.py",
    ROOT / "artifacts" / "validation" / "tpm_recovery_verdict" / "tpm_recovery_verdict.md",
    ROOT / "artifacts" / "reports" / "crash_audit_report" / "crash_audit_report.md",
]

JSON_GATES = [
    ROOT / "artifacts" / "validation" / "tpm_recovery_verdict" / "tpm_recovery_verdict.json",
    ROOT / "artifacts" / "reports" / "crash_audit_report" / "crash_audit_report.json",
    ROOT / "artifacts" / "validation" / "replay_file_after_keyfix_summary.json",
]


def relpath(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def normalize(raw: object) -> str | None:
    if raw is None:
        return None
    return NORMALIZE_VERDICT.get(str(raw))


def verdict_row(source: Path, case: str, raw_verdict: object,
                acceptable: object | None = None,
                detail: str | None = None,
                scope: str | None = None) -> dict[str, Any]:
    normalized = normalize(raw_verdict)
    return {
        "source": relpath(source),
        "case": case,
        "raw_verdict": raw_verdict,
        "normalized_verdict": normalized,
        "allowed": normalized in ALLOWED_VERDICTS,
        "acceptable": acceptable,
        "detail": detail,
        "scope": scope,
    }


def add_generation(rows: list[dict[str, Any]]) -> None:
    path = ROOT / "artifacts" / "validation" / "generation_fault_matrix" / "generation_fault_matrix.json"
    data = load_json(path)
    for row in data.get("rows", []):
        rows.append(verdict_row(
            path,
            str(row.get("case")),
            row.get("oracle_verdict"),
            row.get("acceptable"),
            row.get("detail"),
            "generation_fault_matrix",
        ))


def add_hardware_freshness(rows: list[dict[str, Any]]) -> None:
    path = ROOT / "artifacts" / "validation" / "hardware_freshness_recovery_matrix" / "hardware_freshness_recovery_matrix.json"
    data = load_json(path)
    for row in data.get("rows", []):
        rows.append(verdict_row(
            path,
            str(row.get("case")),
            row.get("oracle_verdict"),
            row.get("pass"),
            row.get("evidence"),
            str(row.get("scope")),
        ))


def add_daemon(rows: list[dict[str, Any]]) -> None:
    path = ROOT / "artifacts" / "validation" / "daemon_power_fault_campaign" / "daemon_power_fault_campaign.json"
    data = load_json(path)
    for row in data.get("daemon_rows", []):
        rows.append(verdict_row(
            path,
            f"daemon:{row.get('case')}",
            row.get("verdict"),
            row.get("acceptable"),
            row.get("detail"),
            "daemon_sigkill_cutpoint",
        ))
    for row in data.get("application_rows", []):
        rows.append(verdict_row(
            path,
            f"application:{row.get('mode')}",
            row.get("verdict"),
            row.get("acceptable"),
            row.get("detail"),
            "application_mode_manifest",
        ))


def add_sqlite_fault(rows: list[dict[str, Any]]) -> None:
    path = ROOT / "artifacts" / "validation" / "sqlite_fault_campaign" / "sqlite_fault_campaign.json"
    data = load_json(path)
    for row in data.get("rows", []):
        rows.append(verdict_row(
            path,
            f"{row.get('cut_point')}#{row.get('trial')}",
            row.get("verdict"),
            row.get("acceptable"),
            row.get("detail"),
            "sqlite_wal_full_selected_boundary",
        ))


def add_sqlite_syscall(rows: list[dict[str, Any]]) -> None:
    path = ROOT / "artifacts" / "validation" / "sqlite_syscall_crash_tpm" / "sqlite_syscall_crash_tpm.json"
    data = load_json(path)
    for row in data.get("trials", []):
        replay = row.get("replay") or {}
        rows.append(verdict_row(
            path,
            f"fdatasync_when_{row.get('when')}",
            replay.get("verdict"),
            replay.get("acceptable"),
            replay.get("detail"),
            "sqlite_delete_extra_syscall_kill",
        ))


def add_combined(rows: list[dict[str, Any]]) -> None:
    path = ROOT / "artifacts" / "validation" / "combined_durability_bundle" / "combined_durability_bundle.json"
    data = load_json(path)
    for key, label in (
        ("unified_campaign", "sqlite_tpm_replay"),
        ("unified_dbm_campaign", "dbm_tpm_replay"),
    ):
        replay = ((data.get(key) or {}).get("replay") or {})
        rows.append(verdict_row(
            path,
            label,
            replay.get("verdict"),
            replay.get("acceptable"),
            replay.get("detail"),
            "same_backing_store_tpm_replay",
        ))


def add_tpm_monotonic(rows: list[dict[str, Any]]) -> None:
    path = ROOT / "artifacts" / "validation" / "tpm_monotonic_replay" / "tpm_monotonic_replay.json"
    data = load_json(path)
    replay = (((data.get("result") or {}).get("replay_result")) or {})
    rows.append(verdict_row(
        path,
        "tpm_monotonic_replay",
        replay.get("mode"),
        (data.get("result") or {}).get("fail_closed") is True,
        replay.get("detail"),
        "real_tpm_replay_after_advance",
    ))


def add_replay_file(rows: list[dict[str, Any]]) -> None:
    for filename, scope in (
        ("replay_file_matrix.json", "file_anchor_negative_control"),
        ("replay_file_final_matrix.json", "file_anchor_final_negative_control"),
        ("replay_file_after_keyfix_matrix.json", "post_keyfix_replay_rejection"),
    ):
        path = ROOT / "artifacts" / "validation" / filename
        data = load_json(path)
        for row in data.get("rows", []):
            rows.append(verdict_row(
                path,
                f"{row.get('backend')}@{row.get('cut_point_s')}#{row.get('trial')}",
                row.get("mode"),
                True,
                row.get("detail"),
                scope,
            ))


def add_tpm_recovery_verdict(rows: list[dict[str, Any]]) -> None:
    path = ROOT / "artifacts" / "validation" / "tpm_recovery_verdict" / "tpm_recovery_verdict.json"
    data = load_json(path)
    for index, row in enumerate(data.get("hardware_rows", [])):
        rows.append(verdict_row(
            path,
            f"hardware_e8_row_{index}",
            row.get("mode"),
            True,
            row.get("detail"),
            "legacy_hardware_e8_packaging",
        ))


def scan_forbidden_terms() -> list[dict[str, Any]]:
    hits: list[dict[str, Any]] = []
    for path in TEXT_GATES:
        if not path.exists():
            hits.append({"path": relpath(path), "missing": True})
            continue
        for lineno, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
            for pattern in FORBIDDEN_SUCCESS_PATTERNS:
                if pattern.search(line):
                    hits.append({
                        "path": relpath(path),
                        "line": lineno,
                        "pattern": pattern.pattern,
                        "text": line.strip(),
                    })
    for path in JSON_GATES:
        if not path.exists():
            hits.append({"path": relpath(path), "missing": True})
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for pattern in FORBIDDEN_SUCCESS_PATTERNS:
            for match in pattern.finditer(text):
                hits.append({
                    "path": relpath(path),
                    "offset": match.start(),
                    "pattern": pattern.pattern,
                })
    return hits


def count_verdicts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        verdict = row.get("normalized_verdict")
        counts[str(verdict)] = counts.get(str(verdict), 0) + 1
    return counts


def write_markdown(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# Recovery oracle audit",
        "",
        f"- Overall pass: `{report['overall_pass']}`",
        f"- Retained recovery rows audited: `{report['summary']['row_count']}`",
        f"- Missing/invalid verdict rows: `{report['summary']['invalid_rows']}`",
        f"- Forbidden success-report hits: `{len(report['forbidden_success_terms'])}`",
        "",
        "## Verdict counts",
        "",
    ]
    for verdict, count in sorted(report["summary"]["normalized_verdict_counts"].items()):
        lines.append(f"- `{verdict}`: `{count}`")
    if report["violations"]:
        lines.extend(["", "## Violations", ""])
        for row in report["violations"]:
            lines.append(f"- `{row['source']}` `{row['case']}` raw `{row['raw_verdict']}`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    out_dir = args.out_dir if args.out_dir.is_absolute() else ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    add_generation(rows)
    add_hardware_freshness(rows)
    add_daemon(rows)
    add_sqlite_fault(rows)
    add_sqlite_syscall(rows)
    add_combined(rows)
    add_tpm_monotonic(rows)
    add_replay_file(rows)
    add_tpm_recovery_verdict(rows)

    violations = [row for row in rows if row.get("normalized_verdict") not in ALLOWED_VERDICTS]
    forbidden = scan_forbidden_terms()
    report = {
        "schema_version": 1,
        "allowed_verdicts": sorted(ALLOWED_VERDICTS),
        "normalization": NORMALIZE_VERDICT,
        "rows": rows,
        "violations": violations,
        "forbidden_success_terms": forbidden,
        "summary": {
            "row_count": len(rows),
            "invalid_rows": len(violations),
            "normalized_verdict_counts": count_verdicts(rows),
        },
        "scope": [
            "Audits retained recovery/freshness artifacts cited by the paper and README.",
            "Does not rerun workloads and does not certify physical power loss or arbitrary workloads.",
            "Legacy negative-control labels are preserved as raw labels and normalized to the allowed oracle vocabulary.",
        ],
        "overall_pass": len(rows) > 0 and not violations and not forbidden,
    }

    json_path = out_dir / "recovery_oracle_audit.json"
    md_path = out_dir / "recovery_oracle_audit.md"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_markdown(report, md_path)
    print(json.dumps({
        "out_dir": relpath(out_dir),
        "overall_pass": report["overall_pass"],
        "row_count": len(rows),
        "invalid_rows": len(violations),
        "forbidden_success_terms": len(forbidden),
    }, indent=2))
    return 0 if report["overall_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
