#!/usr/bin/env python3
"""Audit that Evaluation is organized around positive research questions."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
PAPER = ROOT / "Paper"
DEFAULT_OUT = ROOT / "artifacts" / "reports" / "evaluation_rq_audit"

REQUIRED_RQS = [
    {
        "id": "RQ1",
        "category": "CPU/GPU/PQC placement",
        "terms": ["cpu", "gpu", "pqc", "placement", "ml-kem"],
        "evidence_needles": [
            "\\label{sec:eval_performance}",
            "This is why AEGIS-Q's data plane is CPU-first",
            "mounted ML-KEM-768 key-plane workflow",
        ],
    },
    {
        "id": "RQ2",
        "category": "mounted app QoS",
        "terms": ["storage-visible", "sqlite", "edge-storage", "remount"],
        "evidence_needles": ["\\label{sec:eval_qos}", "SQLite transaction latency", "Kernel controls preserve"],
    },
    {
        "id": "RQ3",
        "category": "correctness",
        "terms": ["correctness", "authenticated", "publication"],
        "evidence_needles": ["\\label{sec:eval_workloads}", "generation fault matrix", "EKEYREJECTED"],
    },
    {
        "id": "RQ4",
        "category": "replay and recovery",
        "terms": ["replay", "recovery", "oracle", "tpm"],
        "evidence_needles": ["hardware replay matrix", "daemon campaign", "oracle verdict"],
    },
    {
        "id": "RQ5",
        "category": "cost boundaries and sensitivity",
        "terms": ["strict-publication", "mode-aligned", "baselines", "controller", "key-plane"],
        "evidence_needles": ["frozen filesystem contract", "sensitivity bundle", "ablation manifest"],
    },
]

DISCUSSION_BOUNDARY_NEEDLES = [
    "fscrypt is environment-blocked",
    "physical power-loss",
    "kernel-crash",
    "full cross-SoC portability",
    "GPU side channels",
]

DEFENSIVE_EVAL_PATTERNS = [
    r"RQ\d+:\s*Scope",
    r"Which common systems claims remain unsupported",
    r"\\subsection\{What this evaluation leaves open\}",
    r"\\label\{sec:eval_stability\}",
]


def relpath(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def run_pdfinfo_pages(path: Path) -> int | None:
    proc = subprocess.run(["pdfinfo", str(path)], check=True, text=True,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    for line in proc.stdout.splitlines():
        if line.startswith("Pages:"):
            return int(line.split(":", 1)[1].strip())
    return None


def extract_rqs(evaluation: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    pattern = re.compile(r"\\item\s+\\textbf\{(RQ\d+):\s*([^}]+?)\.\}\s*(.*)")
    for line in evaluation.splitlines():
        match = pattern.search(line.strip())
        if match:
            rows.append({
                "id": match.group(1),
                "title": match.group(2),
                "question": match.group(3).strip(),
            })
    return rows


def text_has_terms(text: str, terms: list[str]) -> bool:
    lowered = text.lower()
    return all(term.lower() in lowered for term in terms)


def build_report() -> dict[str, Any]:
    evaluation = read_text(PAPER / "4_Evaluation.tex")
    discussion = read_text(PAPER / "10_Discussion_and_Limitations.tex")
    rqs = extract_rqs(evaluation)

    rq_rows: list[dict[str, Any]] = []
    for idx, required in enumerate(REQUIRED_RQS):
        found = rqs[idx] if idx < len(rqs) else None
        rq_text = f"{found['title']} {found['question']}" if found else ""
        evidence_present = all(needle in evaluation for needle in required["evidence_needles"])
        terms_present = text_has_terms(rq_text, required["terms"])
        rq_rows.append({
            "required_id": required["id"],
            "required_category": required["category"],
            "found_id": found["id"] if found else None,
            "found_title": found["title"] if found else None,
            "found_question": found["question"] if found else None,
            "terms": required["terms"],
            "terms_present": terms_present,
            "evidence_needles": required["evidence_needles"],
            "evidence_present": evidence_present,
            "passes": bool(found)
            and found["id"] == required["id"]
            and terms_present
            and evidence_present,
        })

    defensive_hits = []
    for pattern in DEFENSIVE_EVAL_PATTERNS:
        if re.search(pattern, evaluation, re.IGNORECASE):
            defensive_hits.append(pattern)

    discussion_boundaries = {
        needle: needle in discussion
        for needle in DISCUSSION_BOUNDARY_NEEDLES
    }

    violations: list[str] = []
    if len(rqs) != len(REQUIRED_RQS):
        violations.append(f"expected {len(REQUIRED_RQS)} evaluation RQs, found {len(rqs)}")
    for row in rq_rows:
        if not row["passes"]:
            violations.append(f"{row['required_id']} does not satisfy required {row['required_category']} shape")
    if defensive_hits:
        violations.append("defensive scope question or evaluation-leaves-open subsection remains in Evaluation")
    if not all(discussion_boundaries.values()):
        violations.append("Discussion/Limits does not retain all required unsupported-claim boundaries")
    pages = run_pdfinfo_pages(PAPER / "main.pdf")
    if pages is None or pages > 13:
        violations.append("Paper/main.pdf exceeds 13 pages")

    return {
        "schema_version": 1,
        "scope": [
            "Paper/4_Evaluation.tex research-question structure",
            "Paper/10_Discussion_and_Limitations.tex boundary retention",
            "Paper/main.pdf page-count gate",
        ],
        "rqs": rq_rows,
        "defensive_eval_patterns": DEFENSIVE_EVAL_PATTERNS,
        "defensive_hits": defensive_hits,
        "discussion_boundaries": discussion_boundaries,
        "pages": run_pdfinfo_pages(PAPER / "main.pdf"),
        "violations": violations,
        "overall_pass": not violations,
    }


def write_markdown(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# Evaluation RQ audit",
        "",
        f"- Overall pass: `{report['overall_pass']}`",
        f"- AEGIS-Q page count at audit time: `{report['pages']}`",
        f"- Defensive Evaluation hits: `{len(report['defensive_hits'])}`",
        "",
        "## Research-question shape",
        "",
        "| RQ | Required category | Found title | Terms present | Evidence present | Pass |",
        "| --- | --- | --- | ---: | ---: | ---: |",
    ]
    for row in report["rqs"]:
        lines.append(
            f"| {row['required_id']} | {row['required_category']} | "
            f"{row.get('found_title') or 'missing'} | `{row['terms_present']}` | "
            f"`{row['evidence_present']}` | `{row['passes']}` |"
        )

    lines += [
        "",
        "## Discussion boundary retention",
        "",
        "| Boundary needle | Present |",
        "| --- | ---: |",
    ]
    for needle, present in report["discussion_boundaries"].items():
        lines.append(f"| `{needle}` | `{present}` |")

    if report["violations"]:
        lines += ["", "## Violations", ""]
        lines += [f"- {v}" for v in report["violations"]]

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    report = build_report()
    json_path = args.out / "evaluation_rq_audit.json"
    md_path = args.out / "evaluation_rq_audit.md"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown(report, md_path)
    print(json.dumps({
        "out_dir": relpath(args.out),
        "overall_pass": report["overall_pass"],
        "pages": report["pages"],
        "rqs": len(report["rqs"]),
        "defensive_hits": len(report["defensive_hits"]),
        "violations": len(report["violations"]),
    }, indent=2, sort_keys=True))
    return 0 if report["overall_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
