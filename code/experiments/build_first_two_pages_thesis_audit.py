#!/usr/bin/env python3
"""Audit that the first two pages present a positive thesis spine."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
PAPER = ROOT / "Paper"
DEFAULT_OUT = ROOT / "artifacts" / "reports" / "first_two_pages_thesis"

SOURCE_ANCHORS = [
    {
        "name": "pressure_result",
        "path": "Paper/1_Introduction.tex",
        "needles": ["Figure~\\ref{fig:first_page_qos}", "9.62", "8.15", "3.02"],
    },
    {
        "name": "capability_table",
        "path": "Paper/1_Introduction.tex",
        "needles": ["Capability comparison", "design gap"],
    },
    {
        "name": "concrete_gap",
        "path": "Paper/1_Introduction.tex",
        "needles": ["systems challenge", "compose ordinary storage invariants", "shared-resource control"],
    },
    {
        "name": "central_thesis",
        "path": "Paper/1_Introduction.tex",
        "needles": ["Its thesis is that secure edge storage", "one durable authenticated format", "CPU-first AES-GCM data plane"],
    },
    {
        "name": "design_insight",
        "path": "Paper/1_Introduction.tex",
        "needles": ["accelerator placement must be subordinate to storage correctness"],
    },
    {
        "name": "contribution_intro",
        "path": "Paper/1_Introduction.tex",
        "needles": ["This paper makes four contributions"],
    },
    {
        "name": "contribution_c1",
        "path": "Paper/1_Introduction.tex",
        "needles": ["C1: Placement-safe storage format"],
    },
    {
        "name": "contribution_c2",
        "path": "Paper/1_Introduction.tex",
        "needles": ["C2: CPU data lane, GPU/PQC maintenance lane"],
    },
    {
        "name": "contribution_c3",
        "path": "Paper/1_Introduction.tex",
        "needles": ["C3: Recovery and replay boundary"],
    },
    {
        "name": "contribution_c4",
        "path": "Paper/1_Introduction.tex",
        "needles": ["C4: Storage-visible QoS control"],
    },
    {
        "name": "not_weaker_component_stack",
        "path": "Paper/2_Background.tex",
        "needles": ["not merely gocryptfs/fscrypt plus CUDA/TPM scripts", "one mounted runtime", "one evidence contract"],
    },
]

PDF_CHECKS = [
    {
        "name": "first_page_pressure_figure",
        "needles": ["Figure 1", "SQLite p99", "9.62", "8.15", "3.0 MB/s"],
    },
    {
        "name": "first_page_capability_table",
        "needles": [
            "Table 1",
            "Capability comparison",
            "Plaintext",
            "gocryptfs",
            "fscrypt",
            "dm-crypt",
            "GPU-storage systems",
            "AEGIS-Q",
        ],
    },
    {
        "name": "first_two_pages_gap",
        "needles": ["systems challenge", "compose ordinary storage invariants", "accelerator placement", "shared-resource control"],
    },
    {
        "name": "first_two_pages_thesis",
        "needles": [
            "secure edge",
            "storage should expose one durable authenticated format",
            "CPU-first AES-GCM data plane",
        ],
    },
    {
        "name": "first_two_pages_contributions",
        "needles": [
            "This paper makes four contributions",
            "C1: Placement-safe",
            "C2: CPU data lane",
            "C3: Recovery and replay",
            "C4: Storage-visible QoS",
        ],
    },
    {
        "name": "first_two_pages_component_stack_answer",
        "needles": ["not merely", "gocryptfs/fscrypt", "CUDA/TPM", "scripts:", "one mounted", "evidence", "contract"],
    },
]

FORBIDDEN_FIRST_PAGE_TERMS = [
    "Fresh root",
    "storage-visible accelerator/QoS/freshness policy",
]

SUPPORT_GATES = [
    "artifacts/reports/paper_spine_gate/paper_spine_gate.json",
    "artifacts/reports/hero_result_contract/hero_result_contract.json",
    "artifacts/reports/novelty_isolation/novelty_isolation.json",
    "artifacts/reports/accepted_paper_structure_audit/accepted_paper_structure_audit.json",
]


def relpath(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def run_pdfinfo_pages(path: Path) -> int | None:
    proc = subprocess.run(["pdfinfo", str(path)], check=True, text=True,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    for line in proc.stdout.splitlines():
        if line.startswith("Pages:"):
            return int(line.split(":", 1)[1].strip())
    return None


def run_pdftotext_pages(path: Path, first: int, last: int) -> str:
    proc = subprocess.run(
        ["pdftotext", "-f", str(first), "-l", str(last), "-layout", str(path), "-"],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return " ".join(proc.stdout.split())


def contains_all(text: str, needles: list[str]) -> bool:
    lowered = text.lower()
    return all(needle.lower() in lowered for needle in needles)


def find_line(path_str: str, needles: list[str]) -> dict[str, Any]:
    path = ROOT / path_str
    text = path.read_text(encoding="utf-8")
    for line_no, line in enumerate(text.splitlines(), start=1):
        if contains_all(line, needles):
            return {
                "path": path_str,
                "line": line_no,
                "text": " ".join(line.strip().split()),
                "present": True,
            }
    return {"path": path_str, "line": None, "text": None, "present": False}


def gate_status(path_str: str) -> dict[str, Any]:
    path = ROOT / path_str
    row: dict[str, Any] = {"path": path_str, "present": path.exists(), "overall_pass": None}
    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
        row["overall_pass"] = data.get("overall_pass")
        if "violations" in data:
            row["violations"] = len(data["violations"])
    return row


def first_index(text: str, needle: str) -> int:
    return text.lower().find(needle.lower())


def build_report() -> dict[str, Any]:
    pdf_text = run_pdftotext_pages(PAPER / "main.pdf", 1, 2)
    intro_text = (PAPER / "1_Introduction.tex").read_text(encoding="utf-8")
    source_rows = []
    for anchor in SOURCE_ANCHORS:
        source_rows.append({"name": anchor["name"], **find_line(anchor["path"], anchor["needles"])})

    pdf_rows = []
    for check in PDF_CHECKS:
        pdf_rows.append({
            "name": check["name"],
            "needles": check["needles"],
            "present": contains_all(pdf_text, check["needles"]),
        })

    forbidden_hits = [
        term for term in FORBIDDEN_FIRST_PAGE_TERMS
        if term.lower() in intro_text.lower() or term.lower() in pdf_text.lower()
    ]

    positive_idx = first_index(intro_text, "This paper makes four contributions")
    defensive_idx = first_index(intro_text, "The claim boundary excludes")
    order_ok = positive_idx >= 0 and defensive_idx >= 0 and positive_idx < defensive_idx

    gates = [gate_status(path) for path in SUPPORT_GATES]
    gates_ok = all(
        row["present"]
        and row["overall_pass"] is True
        and row.get("violations", 0) == 0
        for row in gates
    )

    pages = run_pdfinfo_pages(PAPER / "main.pdf")
    checks = {
        "paper_pages_12": pages == 12,
        "source_anchors_present": all(row["present"] for row in source_rows),
        "compiled_first_two_pages_present": all(row["present"] for row in pdf_rows),
        "no_overstrong_freshness_root_language": not forbidden_hits,
        "positive_contributions_before_defensive_scope": order_ok,
        "support_gates_pass": gates_ok,
    }
    violations = [name for name, passed in checks.items() if not passed]

    return {
        "schema_version": 1,
        "scope": [
            "first two compiled PDF pages",
            "source anchors for pressure figure/table, gap, thesis, contributions, and component-stack answer",
            "supporting paper-spine, hero-result, novelty-isolation, and accepted-structure gates",
        ],
        "pages": pages,
        "source_anchors": source_rows,
        "pdf_checks": pdf_rows,
        "forbidden_first_page_terms": {
            "terms": FORBIDDEN_FIRST_PAGE_TERMS,
            "hits": forbidden_hits,
        },
        "support_gates": gates,
        "ordering": {
            "contributions_index": positive_idx,
            "defensive_scope_index": defensive_idx,
            "positive_before_defensive": order_ok,
        },
        "checks": checks,
        "violations": violations,
        "overall_pass": not violations,
    }


def write_markdown(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# First-two-pages positive-thesis audit",
        "",
        f"- Overall pass: `{report['overall_pass']}`",
        f"- Paper pages: `{report['pages']}`",
        "",
        "## Source Anchors",
        "",
        "| Anchor | Source | Present |",
        "| --- | --- | ---: |",
    ]
    for row in report["source_anchors"]:
        lines.append(f"| `{row['name']}` | `{row['path']}:{row['line']}` | `{row['present']}` |")

    lines += [
        "",
        "## Compiled PDF Checks",
        "",
        "| Check | Present |",
        "| --- | ---: |",
    ]
    for row in report["pdf_checks"]:
        lines.append(f"| `{row['name']}` | `{row['present']}` |")

    lines += [
        "",
        "## Forbidden First-Page Terms",
        "",
        f"- Hits: `{report['forbidden_first_page_terms']['hits']}`",
    ]

    lines += [
        "",
        "## Support Gates",
        "",
        "| Gate | Present | Overall pass | Violations |",
        "| --- | ---: | ---: | ---: |",
    ]
    for row in report["support_gates"]:
        lines.append(
            f"| `{row['path']}` | `{row['present']}` | `{row['overall_pass']}` | "
            f"`{row.get('violations', 'n/a')}` |"
        )

    lines += [
        "",
        "## Checks",
        "",
        "| Check | Pass |",
        "| --- | ---: |",
    ]
    for name, passed in report["checks"].items():
        lines.append(f"| `{name}` | `{passed}` |")

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
    json_path = args.out / "first_two_pages_thesis.json"
    md_path = args.out / "first_two_pages_thesis.md"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown(report, md_path)
    print(json.dumps({
        "out_dir": relpath(args.out),
        "overall_pass": report["overall_pass"],
        "pages": report["pages"],
        "source_anchors": len(report["source_anchors"]),
        "pdf_checks": len(report["pdf_checks"]),
        "violations": len(report["violations"]),
    }, indent=2, sort_keys=True))
    return 0 if report["overall_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
