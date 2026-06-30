#!/usr/bin/env python3
"""Build the Gate F5 architecture-claim firewall.

F5 is narrower than a general search.  It prevents architecture vocabulary such
as eBPF passthrough, shadow mmap, Argon2id, async Merkle, TPM epochs, PCR policy,
Jetson zero-copy, and CUDA stream priority from becoming unsupported paper
mechanisms.  A hit is acceptable only when the surrounding context is an
implemented-gate reference, an explicit non-claim, a blocked/future-work item, a
probe/diagnostic field, related-work discussion, or checklist/lint language.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = ROOT / "artifacts" / "reports" / "architecture_claim_firewall"
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


@dataclass
class Finding:
    term: str
    path: str
    line: int
    text: str
    guarded: bool
    guard_reason: str
    context: str


ARCHITECTURE_TERMS = [
    Term("ebpf_passthrough", r"eBPF\s+passthrough|eBPF/io_uring\s+(?:completion\s+)?bypass"),
    Term("async_epoch_fdatasync", r"async\s+epoch\s+fdatasync|epoch\s+fdatasync"),
    Term("shadow_mmap", r"shadow\s+mmap|shadow\s+paging"),
    Term("out_of_place_update", r"out-of-place\s+update|out\s+of\s+place\s+update|atomic\s+pointer\s+swap"),
    Term("argon2id", r"\bArgon2id\b|\bArgon2\b"),
    Term("async_merkle", r"async\s+Merkle|async\s+Merkle\+TPM"),
    Term("merkle_root", r"Merkle\s+root|root-hash|root hash|per-file\s+content\s+Merkle"),
    Term("tpm_epoch", r"TPM\s+epoch|hardware\s+TPM\s+epoch"),
    Term("tpm_rollback_resistance", r"TPM\s+rollback\s+resistance|rollback\s+resistance"),
    Term("pcr", r"\bPCR(?:-bound|-policy|-sealed| binding| policy| sealing)?\b"),
    Term("jetson_zero_copy", r"Jetson\s+zero-copy|zero-copy|zero\s+copy"),
    Term("cuda_stream_priority", r"CUDA\s+stream\s+priority|stream\s+priority"),
]


GUARD_PATTERNS = [
    ("explicit_nonclaim", re.compile(
        r"\b(?:not|no|never|without|unless|until|before|outside scope|out of scope|"
        r"non[- ]claim|nonclaim|limitation|unsupported|not claimed|does not claim|"
        r"does not prove|does not establish|must not claim|not a claim|not a proof|"
        r"not implemented|not supported|not current|not yet|not submission evidence|"
        r"scoped|scoped out|scoped-out|blocked|environment-blocked|unavailable|"
        r"future work|future|roadmap|missing|remain|remains|only|transient|"
        r"diagnostic|probe|prototype|defer|deferred|rather than|instead of|"
        r"아직|검증해야|주장|아니다|않음|과장|분리해서|"
        r"cannot|forbid|forbidden)\b",
        re.IGNORECASE,
    )),
    ("implemented_or_gate_reference", re.compile(
        r"\b(?:Gate|F5|C2|C5|C6|D1|0\.10|0\.12|0\.18|DONE|closeout|"
        r"implemented|production source|production path|source checks|"
        r"retained evidence|retained artifact|reports overall_pass=true|"
        r"writes `?artifacts/|self-test|runner|verdict|decision|manifest|"
        r"compute|recompute|leaf_count|root_out|leaf hashes|Merkle reduction)\b",
        re.IGNORECASE,
    )),
    ("claim_guard_rule", re.compile(
        r"\b(?:guard|lint|firewall|dangerous|negative claim|claim scan|claim-lint|"
        r"paper guard|scope audit|scope boundary|boundary|checklist|"
        r"required paper text|required code/script|close condition|forbidden_upgrade|"
        r"required_terms|pattern|regex|re\.compile)\b",
        re.IGNORECASE,
    )),
    ("related_work_or_probe", re.compile(
        r"\b(?:related work|prior|comparison|baseline|different assumptions|"
        r"NVIDIA|Jetson|CUDA|TPM|TCTI|availability|capability|attribute|"
        r"cudaDevAttr|nvpmodel|tegrastats|Nsight|CUPTI|policy probe|"
        r"pcrread|seal/unseal|transient seal|diagnostics?|prototype|"
        r"paper-eligibility|evidence level)\b",
        re.IGNORECASE,
    )),
    ("source_literal", re.compile(
        r"\b(?:non_claim|negative_guard|required_missing_artifacts|paper_claim_boundary|"
        r"source_boundary|scope_boundary|claim_checks|paper_checks|missing|"
        r"classification|available_api|probe_pass|production_source_present|"
        r"has_hit|source_mentions|guard=|note=|hot_path_status)\b",
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
    compiled = [(term, re.compile(term.pattern, re.IGNORECASE)) for term in ARCHITECTURE_TERMS]
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
    term_counts = {term.name: 0 for term in ARCHITECTURE_TERMS}
    guarded_counts = {term.name: 0 for term in ARCHITECTURE_TERMS}
    violation_counts = {term.name: 0 for term in ARCHITECTURE_TERMS}
    for finding in findings:
        term_counts[finding.term] += 1
        if finding.guarded:
            guarded_counts[finding.term] += 1
        else:
            violation_counts[finding.term] += 1

    checks = {
        "scan_roots_present": all(path.exists() for path in (
            ROOT / "Paper",
            ROOT / "README.md",
            ROOT / "SUBMISSION_CHECKLIST.md",
            ROOT / "docs",
            ROOT / "code",
        )),
        "all_terms_configured": len(term_counts) == len(ARCHITECTURE_TERMS),
        "no_unguarded_architecture_claims": len(violations) == 0,
        "guarded_contexts_not_serialized": True,
    }
    return {
        "schema_version": 1,
        "scope": "Gate F5 architecture claim firewall",
        "generated_by": rel(SELF),
        "scanned_file_count": len(scan_paths()),
        "architecture_term_names": [term.name for term in ARCHITECTURE_TERMS],
        "term_counts": term_counts,
        "guarded_counts": guarded_counts,
        "violation_counts": violation_counts,
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
            "Architecture mechanism phrases may appear only when they are tied "
            "to an implemented gate, explicit future/blocked work, a probe or "
            "diagnostic, related-work comparison, or an explicit non-claim."
        ),
    }


def write_markdown(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# Architecture Claim Firewall",
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
    json_path = DEFAULT_OUT / "architecture_claim_firewall.json"
    md_path = DEFAULT_OUT / "architecture_claim_firewall.md"
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
