#!/usr/bin/env python3
"""Audit that AEGIS-Q integrity claims stay within retained evidence.

This is a claim-scope audit, not an implementation of per-file content
integrity.  It verifies that the paper states the retained scope
(committed-prefix anchor/parity evidence) and does not make an unqualified
claim that AEGIS-Q persists a complete per-file content Merkle tree.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
PAPER_DIR = ROOT / "Paper"
DEFAULT_OUT = ROOT / "artifacts" / "validation" / "integrity_scope_audit"
INTEGRITY_BENCH = ROOT / "build" / "bench_gpu_integrity"


@dataclass(frozen=True)
class Pattern:
    name: str
    regex: re.Pattern[str]
    explanation: str


HIGH_RISK_PATTERNS = [
    Pattern(
        "per_file_content_merkle_tree",
        re.compile(r"per-file\s+content\s+Merkle\s+tree", re.IGNORECASE),
        "Mentions of a per-file content Merkle tree must be explicitly negated or scoped.",
    ),
    Pattern(
        "complete_per_file_merkle_protection",
        re.compile(r"(?:complete|full)\s+per-file\s+Merkle\s+protection", re.IGNORECASE),
        "Complete per-file Merkle protection is not implemented.",
    ),
    Pattern(
        "verification_on_read_merkle",
        re.compile(r"(?:verification|verify|verifies)\s*-?\s*on\s*-?\s*read.{0,80}Merkle", re.IGNORECASE),
        "Verification-on-read Merkle claims require a persisted content tree.",
    ),
    Pattern(
        "persisted_merkle_root_storage",
        re.compile(r"(?:persist(?:s|ed|ing)?|store(?:s|d|ing)?)\s+.{0,80}Merkle\s+root", re.IGNORECASE),
        "Persisted Merkle-root storage claims must identify the committed-prefix anchor, not a content tree.",
    ),
]

NEGATION_TERMS = (
    "does not",
    "do not",
    "not ",
    " no ",
    "without",
    "unless",
)

SCOPE_TERMS = (
    "committed-prefix",
    "parity",
    "helper",
    "negative control",
    "narrow",
    "not a",
)

REQUIRED_PAPER_PHRASES = [
    "committed-prefix anchor root",
    "does not persist a per-file content Merkle tree",
    "do not mean the filesystem persists a per-file content Merkle tree",
]

REQUIRED_SOURCE_PHRASES = {
    "pqc_anchor.c": "Committed-prefix root",
    "cuda_integrity.h": "committed-prefix root",
}


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def context(text: str, start: int, end: int, radius: int = 140) -> str:
    begin = max(0, start - radius)
    finish = min(len(text), end + radius)
    return " ".join(text[begin:finish].split())


def is_scoped(match_context: str) -> bool:
    lowered = f" {match_context.lower()} "
    return any(term in lowered for term in NEGATION_TERMS) or any(
        term in lowered for term in SCOPE_TERMS
    )


def paper_files() -> list[Path]:
    return sorted(path for path in PAPER_DIR.glob("*.tex") if path.is_file())


def scan_paper() -> dict[str, Any]:
    files = paper_files()
    combined = "\n".join(path.read_text(encoding="utf-8") for path in files)
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

    required = []
    for phrase in REQUIRED_PAPER_PHRASES:
        found = phrase in combined
        required.append({"phrase": phrase, "found": found})

    return {
        "paper_files": [str(path.relative_to(ROOT)) for path in files],
        "high_risk_mentions": mentions,
        "unscoped_violations": violations,
        "required_scope_phrases": required,
        "paper_scope_pass": not violations and all(item["found"] for item in required),
    }


def check_source_scope() -> dict[str, Any]:
    checks = []
    for rel, phrase in REQUIRED_SOURCE_PHRASES.items():
        path = ROOT / rel
        found = path.exists() and phrase in path.read_text(encoding="utf-8", errors="replace")
        checks.append({"file": rel, "phrase": phrase, "found": found})
    return {
        "checks": checks,
        "source_scope_pass": all(item["found"] for item in checks),
    }


def run_integrity_bench(out_dir: Path) -> dict[str, Any]:
    stdout_path = out_dir / "bench_gpu_integrity.stdout.txt"
    stderr_path = out_dir / "bench_gpu_integrity.stderr.txt"
    if not INTEGRITY_BENCH.exists():
        stdout_path.write_text("", encoding="utf-8")
        stderr_path.write_text(f"missing binary: {INTEGRITY_BENCH}\n", encoding="utf-8")
        return {
            "command": [str(INTEGRITY_BENCH), "--only-tests"],
            "returncode": None,
            "stdout": str(stdout_path.relative_to(ROOT)),
            "stderr": str(stderr_path.relative_to(ROOT)),
            "passed": False,
            "reason": "missing_bench_binary",
        }

    with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
        proc = subprocess.run(
            [str(INTEGRITY_BENCH), "--only-tests"],
            cwd=ROOT,
            stdout=stdout,
            stderr=stderr,
            check=False,
        )

    stdout = stdout_path.read_bytes()
    stderr = stderr_path.read_bytes()
    success = proc.returncode == 0 and b"[SUCCESS] All correctness tests passed." in stdout
    return {
        "command": [str(INTEGRITY_BENCH), "--only-tests"],
        "returncode": proc.returncode,
        "stdout": str(stdout_path.relative_to(ROOT)),
        "stderr": str(stderr_path.relative_to(ROOT)),
        "stdout_sha256": sha256_bytes(stdout),
        "stderr_sha256": sha256_bytes(stderr),
        "passed": success,
        "scope": (
            "Parity evidence for SHA-256 leaf/Merkle helper behavior only; "
            "not persisted per-file content-integrity protection."
        ),
    }


def write_markdown(result: dict[str, Any], path: Path) -> None:
    lines = [
        "# Integrity Scope Audit",
        "",
        f"- Overall pass: `{str(result['overall_pass']).lower()}`",
        f"- Paper scope pass: `{str(result['paper_scan']['paper_scope_pass']).lower()}`",
        f"- Source scope pass: `{str(result['source_scope']['source_scope_pass']).lower()}`",
        f"- Integrity parity bench pass: `{str(result['integrity_bench']['passed']).lower()}`",
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
    lines.extend(["", "## Integrity Bench"])
    lines.append(f"- Command: `{' '.join(result['integrity_bench']['command'])}`")
    lines.append(f"- Return code: `{result['integrity_bench']['returncode']}`")
    lines.append(f"- Stdout: `{result['integrity_bench']['stdout']}`")
    lines.append(f"- Stderr: `{result['integrity_bench']['stderr']}`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    paper_scan = scan_paper()
    source_scope = check_source_scope()
    integrity_bench = run_integrity_bench(out_dir)
    result = {
        "audit": "integrity_scope_audit",
        "paper_scan": paper_scan,
        "source_scope": source_scope,
        "integrity_bench": integrity_bench,
    }
    result["overall_pass"] = (
        paper_scan["paper_scope_pass"]
        and source_scope["source_scope_pass"]
        and integrity_bench["passed"]
    )

    json_path = out_dir / "integrity_scope_audit.json"
    md_path = out_dir / "integrity_scope_audit.md"
    json_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown(result, md_path)
    print(json.dumps({"overall_pass": result["overall_pass"], "out": str(json_path)}, sort_keys=True))
    return 0 if result["overall_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
