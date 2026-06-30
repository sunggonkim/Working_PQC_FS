#!/usr/bin/env python3
"""Audit the paper decision for UMA/storage-DMA claims.

AEGIS-Q retains useful storage/GPU diagnostics, but the submitted paper should
not turn those diagnostics into a direct NVMe-to-UVM DMA contribution.  This
audit records that decision by checking both the manuscript scope language and
the retained diagnostic reports.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
PAPER_DIR = ROOT / "Paper"
DEFAULT_OUT = ROOT / "artifacts" / "validation" / "uma_dma_scope_audit"
PROFILE_REPORT = (
    ROOT
    / "artifacts"
    / "validation"
    / "uma_storage_dma_profile_combined_report"
    / "uma_storage_dma_profile_report.json"
)
COUNTER_REPORT = (
    ROOT
    / "artifacts"
    / "validation"
    / "uma_counter_availability"
    / "uma_counter_availability.json"
)


@dataclass(frozen=True)
class Pattern:
    name: str
    regex: re.Pattern[str]
    explanation: str


HIGH_RISK_PATTERNS = [
    Pattern(
        "direct_nvme_to_uvm_dma",
        re.compile(r"(?:direct\s+)?(?:\\texttt\{O\\_DIRECT\}\s+)?NVMe-to-UVM\s+DMA", re.IGNORECASE),
        "Direct NVMe-to-UVM DMA is intentionally excluded from the contribution.",
    ),
    Pattern(
        "storage_dma_path",
        re.compile(r"(?:storage-DMA|storage\s+DMA|O\\_DIRECT-to-UVM)", re.IGNORECASE),
        "Storage-DMA wording must be diagnostic, future work, or explicitly negated.",
    ),
    Pattern(
        "gpudirect_storage",
        re.compile(r"GPUDirect(?:-like|\s+Storage)?", re.IGNORECASE),
        "GPUDirect vocabulary must not imply the FUSE path implements that API.",
    ),
    Pattern(
        "migration_or_coherence_suppression",
        re.compile(r"(?:migration|coherence)(?:/coherence)?\s+suppression", re.IGNORECASE),
        "Migration/coherence suppression is not proven by retained counters.",
    ),
    Pattern(
        "final_fuse_dma_path",
        re.compile(r"final\s+FUSE\s+data-path\s+DMA", re.IGNORECASE),
        "The final mounted FUSE data path is POSIX sidecar I/O, not DMA into UVM.",
    ),
]

SCOPE_TERMS = (
    "does not",
    "do not",
    "not ",
    " no ",
    "neither",
    "without",
    "unless",
    "narrower",
    "future",
    "not a verified",
    "not a statement",
    "do not prove",
    "does not establish",
    "not established",
    "would need",
    "diagnostic",
    "scoped",
    "excluded",
    "ordinary",
    "separately proven",
    "must not",
)

REQUIRED_PAPER_PHRASES = [
    "does not claim direct NVMe-to-UVM DMA",
    "does not claim verified \\texttt{O\\_DIRECT} NVMe-to-UVM DMA",
    "not a verified NVMe-to-UVM storage-DMA path",
    "do not prove UVM migration suppression or final FUSE data-path DMA",
    "ordinary \\texttt{pread}/\\texttt{pwrite}",
]


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def context(text: str, start: int, end: int, radius: int = 160) -> str:
    begin = max(0, start - radius)
    finish = min(len(text), end + radius)
    return " ".join(text[begin:finish].split())


def is_scoped(match_context: str) -> bool:
    lowered = f" {match_context.lower()} "
    return any(term in lowered for term in SCOPE_TERMS)


def paper_files() -> list[Path]:
    return sorted(path for path in PAPER_DIR.glob("*.tex") if path.is_file())


def scan_paper() -> dict[str, Any]:
    files = paper_files()
    combined = "\n".join(path.read_text(encoding="utf-8") for path in files)
    combined_lower = combined.lower()
    mentions: list[dict[str, Any]] = []
    violations: list[dict[str, Any]] = []

    for path in files:
        text = path.read_text(encoding="utf-8")
        for pattern in HIGH_RISK_PATTERNS:
            for match in pattern.regex.finditer(text):
                ctx = context(text, match.start(), match.end())
                scoped = is_scoped(ctx)
                record = {
                    "pattern": pattern.name,
                    "file": str(path.relative_to(ROOT)),
                    "line": text.count("\n", 0, match.start()) + 1,
                    "match": match.group(0),
                    "scoped": scoped,
                    "context": ctx,
                    "explanation": pattern.explanation,
                }
                mentions.append(record)
                if not scoped:
                    violations.append(record)

    required = [{"phrase": phrase, "found": phrase.lower() in combined_lower} for phrase in REQUIRED_PAPER_PHRASES]
    return {
        "paper_files": [str(path.relative_to(ROOT)) for path in files],
        "high_risk_mentions": mentions,
        "unscoped_violations": violations,
        "required_scope_phrases": required,
        "paper_scope_pass": not violations and all(item["found"] for item in required),
    }


def audit_profile_report() -> dict[str, Any]:
    report = read_json(PROFILE_REPORT)
    raw = report.get("raw_probe") or {}
    managed = report.get("managed_storage_probe") or {}
    ncu = report.get("raw_probe_ncu") or {}
    checks = [
        {
            "name": "profile_report_exists",
            "passed": PROFILE_REPORT.exists(),
            "detail": str(PROFILE_REPORT.relative_to(ROOT)),
        },
        {
            "name": "report_is_scoped_not_dma_proof",
            "passed": "not a proof of NVMe-to-UVM DMA semantics" in str(report.get("note", "")),
            "detail": report.get("note", ""),
        },
        {
            "name": "raw_probe_same_buffer_checksum",
            "passed": bool(raw.get("same_buffer_checksum_match")),
            "detail": "raw pinned-host O_DIRECT diagnostic only",
        },
        {
            "name": "managed_storage_same_buffer_checksum",
            "passed": bool(managed.get("same_buffer_checksum_match")),
            "detail": "storage-filled cudaMallocManaged diagnostic only",
        },
        {
            "name": "managed_storage_pointer_attr",
            "passed": bool(managed.get("managed_pointer_attr")),
            "detail": "managed pointer attributes retained",
        },
        {
            "name": "managed_storage_prefetch_diagnostics",
            "passed": bool(managed.get("device_prefetch_seen")) and bool(managed.get("host_prefetch_seen")),
            "detail": "device and host prefetch-location diagnostics retained",
        },
        {
            "name": "ncu_same_buffer_checksum",
            "passed": bool(ncu.get("same_buffer_checksum_match")),
            "detail": "NCU checksum kernel diagnostic retained",
        },
    ]
    return {
        "path": str(PROFILE_REPORT.relative_to(ROOT)),
        "sha256": sha256_bytes(PROFILE_REPORT.read_bytes()) if PROFILE_REPORT.exists() else None,
        "checks": checks,
        "profile_scope_pass": all(item["passed"] for item in checks),
    }


def audit_counter_report() -> dict[str, Any]:
    report = read_json(COUNTER_REPORT)
    runs = report.get("nsys_runs") if isinstance(report.get("nsys_runs"), list) else []
    ncu_query = report.get("ncu_metric_query") if isinstance(report.get("ncu_metric_query"), dict) else {}
    run_checks = []
    for run in runs:
        reports = run.get("reports") if isinstance(run.get("reports"), list) else []
        um_reports = [r for r in reports if r.get("report") in {"um_sum", "um_total_sum", "um_cpu_page_faults_sum"}]
        run_checks.append(
            {
                "name": run.get("name"),
                "exists": bool(run.get("exists")),
                "um_report_count": len(um_reports),
                "all_um_reports_empty_or_skipped": all(
                    int(r.get("data_line_count", 0)) == 0 and bool(r.get("skipped_messages"))
                    for r in um_reports
                ),
                "matched_event_tables": run.get("sqlite", {}).get("matched_event_tables", []),
            }
        )

    checks = [
        {
            "name": "counter_report_exists",
            "passed": COUNTER_REPORT.exists(),
            "detail": str(COUNTER_REPORT.relative_to(ROOT)),
        },
        {
            "name": "counter_report_is_not_suppression_proof",
            "passed": "not a migration-suppression proof" in str(report.get("note", "")),
            "detail": report.get("note", ""),
        },
        {
            "name": "nsys_runs_present",
            "passed": len(run_checks) >= 3 and all(item["exists"] for item in run_checks),
            "detail": [item["name"] for item in run_checks],
        },
        {
            "name": "um_reports_empty_or_skipped",
            "passed": bool(run_checks)
            and all(item["all_um_reports_empty_or_skipped"] for item in run_checks),
            "detail": run_checks,
        },
        {
            "name": "ncu_query_has_no_uvm_migration_metric_matches",
            "passed": ncu_query.get("available") is True
            and ncu_query.get("returncode") == 0
            and ncu_query.get("matched_line_count") == 0,
            "detail": ncu_query,
        },
    ]
    return {
        "path": str(COUNTER_REPORT.relative_to(ROOT)),
        "sha256": sha256_bytes(COUNTER_REPORT.read_bytes()) if COUNTER_REPORT.exists() else None,
        "checks": checks,
        "counter_scope_pass": all(item["passed"] for item in checks),
    }


def write_markdown(result: dict[str, Any], path: Path) -> None:
    lines = [
        "# UMA / Storage-DMA Scope Audit",
        "",
        f"- Overall pass: `{str(result['overall_pass']).lower()}`",
        f"- Decision: `{result['decision']}`",
        f"- Paper scope pass: `{str(result['paper_scan']['paper_scope_pass']).lower()}`",
        f"- Profile diagnostic pass: `{str(result['profile_report']['profile_scope_pass']).lower()}`",
        f"- Counter diagnostic pass: `{str(result['counter_report']['counter_scope_pass']).lower()}`",
        "",
        "## High-Risk Paper Mentions",
    ]
    for mention in result["paper_scan"]["high_risk_mentions"]:
        lines.append(
            f"- `{mention['file']}:{mention['line']}` `{mention['pattern']}` "
            f"scoped=`{str(mention['scoped']).lower()}`: {mention['context']}"
        )
    if not result["paper_scan"]["high_risk_mentions"]:
        lines.append("- None.")
    lines.extend(["", "## Required Scope Phrases"])
    for item in result["paper_scan"]["required_scope_phrases"]:
        lines.append(f"- `{item['phrase']}` found=`{str(item['found']).lower()}`")
    lines.extend(["", "## Profile Checks"])
    for item in result["profile_report"]["checks"]:
        lines.append(f"- `{item['name']}` passed=`{str(item['passed']).lower()}`")
    lines.extend(["", "## Counter Checks"])
    for item in result["counter_report"]["checks"]:
        lines.append(f"- `{item['name']}` passed=`{str(item['passed']).lower()}`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    paper_scan = scan_paper()
    profile_report = audit_profile_report()
    counter_report = audit_counter_report()
    result = {
        "audit": "uma_dma_scope_audit",
        "decision": "accept_critical_path_direct_nvme_to_uvm_dma_scoped_out",
        "paper_scan": paper_scan,
        "profile_report": profile_report,
        "counter_report": counter_report,
    }
    result["overall_pass"] = (
        paper_scan["paper_scope_pass"]
        and profile_report["profile_scope_pass"]
        and counter_report["counter_scope_pass"]
    )

    json_path = args.out / "uma_dma_scope_audit.json"
    md_path = args.out / "uma_dma_scope_audit.md"
    json_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown(result, md_path)
    print(json.dumps({"overall_pass": result["overall_pass"], "out": str(json_path)}, sort_keys=True))
    return 0 if result["overall_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
