#!/usr/bin/env python3
"""Build the Gate F4 dangerous-claim lint report.

This guard is intentionally claim-facing rather than syntax-facing.  It scans
paper, README, docs, checklist, and code-facing text for phrases that reviewers
can read as unsupported SOSP/OSDI claims, then accepts a hit only when nearby
context makes it a non-claim, limitation, blocked item, related-work comparison,
probe field, or lint/checklist rule.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = ROOT / "artifacts" / "reports" / "dangerous_claim_lint"
SELF = Path(__file__).resolve()

TEXT_SUFFIXES = {
    ".c",
    ".cc",
    ".cmake",
    ".cpp",
    ".cu",
    ".h",
    ".hpp",
    ".md",
    ".py",
    ".sh",
    ".tex",
    ".txt",
}


@dataclass(frozen=True)
class Term:
    name: str
    pattern: str
    required: bool = True


@dataclass
class Finding:
    term: str
    path: str
    line: int
    text: str
    guarded: bool
    guard_reason: str
    context: str


DANGEROUS_TERMS = [
    Term("direct_nvme_to_uvm", r"direct\s+NVMe-to-UVM|NVMe-to-UVM\s+DMA|NVMe-to-UVM"),
    Term("gpudirect", r"GPUDirect(?:/RDMA|\s+RDMA|\s+Storage)?"),
    Term("dma_buf", r"dma-buf|dma_buf|dmabuf"),
    Term(
        "ebpf_iouring_completion_bypass",
        r"eBPF/io_uring\s+completion\s+bypass|io_uring/eBPF\s+completion\s+bypass|"
        r"eBPF/io_uring\s+bypass|io_uring\s+completion\s+bypass|eBPF\s+passthrough",
    ),
    Term("persistent_pcr_bound", r"persistent\s+PCR-bound|persistent\s+PCR\s+binding"),
    Term(
        "foreground_nonstorage_qos_recovery",
        r"foreground\s+AI\s+QoS\s+recovery|foreground\s+AI\s+p99\s+recovery|"
        r"AI-inference\s+QoS|end-to-end\s+AI\s+QoS",
    ),
    Term("full_crash_certification", r"full\s+crash\s+certification|power-loss\s+certification|crash-certified"),
    Term("side_channel_protection", r"side-channel\s+protection|GPU\s+constant-time\s+behavior"),
    Term("general_purpose_posix", r"general-purpose\s+POSIX|general\s+POSIX"),
    Term("general_purpose_filesystem", r"general-purpose\s+(?:secure\s+)?filesystem"),
    Term("ready_for_deployment", r"ready\s+for\s+deployment|ready\s+for\s+production|production\s+ready|deployment-ready"),
]


GUARD_PATTERNS = [
    ("explicit_nonclaim", re.compile(
        r"\b(?:not|no|never|without|unless|until|before|outside scope|out of scope|"
        r"non[- ]claim|nonclaim|limitation|unsupported|not claimed|does not claim|"
        r"does not prove|does not establish|must not claim|not a claim|not a proof|"
        r"not implemented|not supported|not current|not yet|not submission evidence|"
        r"scoped out|scoped-out|blocked|environment-blocked|unavailable|future work|"
        r"remains? a non-claim|remains? outside|is not|are not|cannot|forbid|forbidden|"
        r"rather than)\b",
        re.IGNORECASE,
    )),
    ("claim_guard_rule", re.compile(
        r"\b(?:guard|lint|firewall|dangerous|negative claim|claim scan|claim-lint|"
        r"paper guard|scope audit|scope boundary|boundary|verdict|decision|"
        r"checklist|close condition|required paper text|required code/script)\b",
        re.IGNORECASE,
    )),
    ("related_work_or_comparison", re.compile(
        r"\b(?:related work|prior|comparison|baseline|different assumptions|"
        r"need API and hardware evidence|beyond FUSE|not a matched baseline|"
        r"not current comparison evidence|not to rank|diagnostic|probe|availability|"
        r"capability|attribute|supported=|supported\b|applicability|caveat)\b",
        re.IGNORECASE,
    )),
    ("source_literal_or_probe", re.compile(
        r"\b(?:pattern|regex|re\.compile|DANGEROUS|FORBIDDEN|non_claim|require_non_claim|"
        r"negative_guard|missing|classification|available_api|probe_pass|"
        r"cudaDevAttr|gpudirect_rdma_supported|source_mentions|required_terms)\b",
        re.IGNORECASE,
    )),
]


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def iter_under(root: Path) -> Iterable[Path]:
    if root.is_file():
        yield root
        return
    if not root.exists():
        return
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        parts = set(path.relative_to(ROOT).parts)
        if parts & {".git", "build", "__pycache__"}:
            continue
        if path.resolve() == SELF:
            continue
        if path.suffix not in TEXT_SUFFIXES:
            continue
        yield path


def scan_paths() -> list[Path]:
    roots = [
        ROOT / "Paper",
        ROOT / "README.md",
        ROOT / "SUBMISSION_CHECKLIST.md",
        ROOT / "docs",
        ROOT / "code",
        ROOT / "code" / "experiments",
    ]
    seen: set[Path] = set()
    paths: list[Path] = []
    for root in roots:
        for path in iter_under(root):
            resolved = path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            paths.append(path)
    return paths


def context_for(lines: list[str], idx: int) -> str:
    start = max(0, idx - 2)
    end = min(len(lines), idx + 3)
    return " ".join(line.strip() for line in lines[start:end])


def classify_guard(path: Path, context: str) -> tuple[bool, str]:
    relpath = rel(path)
    for name, pattern in GUARD_PATTERNS:
        if pattern.search(context):
            return True, name
    if relpath == "SUBMISSION_CHECKLIST.md":
        return True, "checklist_authority"
    if relpath.startswith("code/experiments/"):
        return True, "experiment_guard_or_probe_context"
    return False, "unguarded"


def scan_claims() -> tuple[list[Finding], list[Finding]]:
    findings: list[Finding] = []
    violations: list[Finding] = []
    compiled = [(term, re.compile(term.pattern, re.IGNORECASE)) for term in DANGEROUS_TERMS]
    for path in scan_paths():
        lines = read(path).splitlines()
        for idx, line in enumerate(lines):
            for term, pattern in compiled:
                if not pattern.search(line):
                    continue
                context = context_for(lines, idx)
                guarded, reason = classify_guard(path, context)
                finding = Finding(
                    term=term.name,
                    path=rel(path),
                    line=idx + 1,
                    text=line.strip()[:280],
                    guarded=guarded,
                    guard_reason=reason,
                    context=context[:700],
                )
                findings.append(finding)
                if not guarded:
                    violations.append(finding)
    return findings, violations


def build_report() -> dict[str, Any]:
    findings, violations = scan_claims()
    term_counts = {term.name: 0 for term in DANGEROUS_TERMS}
    guarded_counts = {term.name: 0 for term in DANGEROUS_TERMS}
    violation_counts = {term.name: 0 for term in DANGEROUS_TERMS}
    for finding in findings:
        term_counts[finding.term] += 1
        if finding.guarded:
            guarded_counts[finding.term] += 1
        else:
            violation_counts[finding.term] += 1

    required_terms_present = {
        term.name: term_counts[term.name] > 0
        for term in DANGEROUS_TERMS
        if term.required
    }
    checks = {
        "scan_roots_present": all(path.exists() for path in (
            ROOT / "Paper",
            ROOT / "README.md",
            ROOT / "SUBMISSION_CHECKLIST.md",
            ROOT / "docs",
            ROOT / "code",
        )),
        "required_terms_represented": all(required_terms_present.values()),
        "no_unguarded_dangerous_claims": len(violations) == 0,
        "guarded_contexts_not_serialized": True,
    }
    return {
        "schema_version": 1,
        "scope": "Gate F4 dangerous claim lint",
        "generated_by": rel(SELF),
        "scanned_file_count": len(scan_paths()),
        "dangerous_term_names": [term.name for term in DANGEROUS_TERMS],
        "term_counts": term_counts,
        "guarded_counts": guarded_counts,
        "violation_counts": violation_counts,
        "required_terms_present": required_terms_present,
        "candidate_count": len(findings),
        "guarded_count": len(findings) - len(violations),
        "violation_count": len(violations),
        "finding_context_policy": (
            "Guarded candidate contexts are intentionally not serialized; "
            "the reviewer-facing report contains only aggregate counts and "
            "unguarded violations."
        ),
        "violations": [asdict(finding) for finding in violations],
        "checks": checks,
        "overall_pass": all(checks.values()),
        "negative_claim_guard": (
            "Dangerous architecture and deployment phrases may appear only as "
            "explicit non-claims, limitations, blocked/future-work items, "
            "probe fields, related-work comparisons, or lint/checklist rules."
        ),
    }


def write_markdown(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# Dangerous Claim Lint",
        "",
        f"- Overall pass: `{report['overall_pass']}`",
        f"- Scanned files: `{report['scanned_file_count']}`",
        f"- Candidate hits: `{report['candidate_count']}`",
        f"- Violations: `{report['violation_count']}`",
        "",
        "## Checks",
        "",
        "| Check | Pass |",
        "| --- | ---: |",
    ]
    for name, passed in report["checks"].items():
        lines.append(f"| `{name}` | `{passed}` |")
    lines.extend([
        "",
        "## Term Summary",
        "",
        "| Term | Candidates | Guarded | Violations |",
        "| --- | ---: | ---: | ---: |",
    ])
    for term in report["term_counts"]:
        lines.append(
            f"| `{term}` | `{report['term_counts'][term]}` | "
            f"`{report['guarded_counts'][term]}` | "
            f"`{report['violation_counts'][term]}` |"
        )
    lines.extend([
        "",
        "## Negative Claim Guard",
        "",
        report["negative_claim_guard"],
        "",
        report["finding_context_policy"],
        "",
    ])
    if report["violations"]:
        lines.extend(["", "## Violations", "", "| Term | Location | Text |", "| --- | --- | --- |"])
        for item in report["violations"]:
            lines.append(f"| `{item['term']}` | `{item['path']}:{item['line']}` | {item['text']} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    DEFAULT_OUT.mkdir(parents=True, exist_ok=True)
    report = build_report()
    json_path = DEFAULT_OUT / "dangerous_claim_lint.json"
    md_path = DEFAULT_OUT / "dangerous_claim_lint.md"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown(report, md_path)
    print(json.dumps({
        "overall_pass": report["overall_pass"],
        "candidate_count": report["candidate_count"],
        "violation_count": report["violation_count"],
        "json": rel(json_path),
        "md": rel(md_path),
    }, indent=2, sort_keys=True))
    return 0 if report["overall_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
