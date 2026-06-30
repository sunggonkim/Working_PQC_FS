#!/usr/bin/env python3
"""Build an accepted-paper structure audit from the local PDF corpus.

This is a structural gate, not a content-quality oracle.  It verifies that the
required local examples are present and text-extractable, records their early
figure/table signals, and checks that AEGIS-Q exposes the same visible spine:
early pressure result, capability table, design before evaluation, mechanisms
mapped to evaluation closures, and a deployment takeaway.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
PAPER = ROOT / "Paper"
PREVIOUS = PAPER / "Previous paper"
DEFAULT_OUT = ROOT / "artifacts" / "reports" / "accepted_paper_structure_audit"

REQUIRED_SOURCES = {
    "ScaleQsim": "*ScaleQsim*.pdf",
    "AURORA-Q": "*AURORA_Q*.pdf",
    "CITADEL": "*CITADEL.pdf",
    "AS2": "*AS2.pdf",
    "previous paper": "previous paper.pdf",
}

MAIN_ORDER = [
    "1_Introduction.tex",
    "2_Background.tex",
    "3_Design.tex",
    "7_Implementation_Details.tex",
    "8_Security_Analysis.tex",
    "4_Evaluation.tex",
    "10_Discussion_and_Limitations.tex",
    "5_Related_Works.tex",
    "6_Conclusion.tex",
]

MECHANISM_MAP = [
    {
        "mechanism": "first-page pressure/gap",
        "design_needles": ["fig:first_page_qos", "tab:capability_matrix"],
        "evaluation_needles": ["sec:evaluation"],
    },
    {
        "mechanism": "authenticated format and D/J/C publication",
        "design_needles": ["tab:design_goals", "fig:djc_state_machine"],
        "evaluation_needles": ["sec:eval_workloads", "generation fault matrix"],
    },
    {
        "mechanism": "CPU-first data lane and elastic key lane",
        "design_needles": ["sec:design_uma", "elastic lane"],
        "evaluation_needles": ["sec:eval_performance", "CPU for bulk data"],
    },
    {
        "mechanism": "external replay boundary",
        "design_needles": ["sec:design_security", "TPM NV index"],
        "evaluation_needles": ["replay-after-advance", "hardware replay matrix"],
    },
    {
        "mechanism": "telemetry-to-storage QoS",
        "design_needles": ["user.pqc\\_qos\\_class", "mounted-FUSE throttling"],
        "evaluation_needles": ["sec:eval_qos", "SQLite recovery"],
    },
]

WARNING_TEXT: list[str] = []


def relpath(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def run_pdftotext(path: Path, first_page: int | None = None,
                  last_page: int | None = None) -> str:
    cmd = ["pdftotext", "-layout"]
    if first_page is not None:
        cmd += ["-f", str(first_page)]
    if last_page is not None:
        cmd += ["-l", str(last_page)]
    cmd += [str(path), "-"]
    proc = subprocess.run(cmd, check=True, text=True,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return proc.stdout


def run_pdfinfo_pages(path: Path) -> int | None:
    proc = subprocess.run(["pdfinfo", str(path)], check=True, text=True,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    for line in proc.stdout.splitlines():
        if line.startswith("Pages:"):
            return int(line.split(":", 1)[1].strip())
    return None


def normalize_line(line: str) -> str:
    return " ".join(line.strip().split())


def first_matching_line(text: str, patterns: list[str]) -> str | None:
    compiled = [re.compile(p, re.IGNORECASE) for p in patterns]
    for line in text.splitlines():
        clean = normalize_line(line)
        if not clean:
            continue
        if any(p.search(clean) for p in compiled):
            return clean[:240]
    return None


def extract_headings(text: str) -> list[str]:
    headings: list[str] = []
    pattern = re.compile(r"^(?:Abstract\b|\d+\.?\s+[A-Z][A-Za-z].*|[IVX]+\.\s+[A-Z][A-Za-z].*)$")
    for line in text.splitlines():
        clean = normalize_line(line)
        if pattern.match(clean):
            headings.append(clean[:200])
    return headings[:20]


def find_required_sources() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for name, pattern in REQUIRED_SOURCES.items():
        matches = sorted(PREVIOUS.glob(pattern))
        row: dict[str, Any] = {
            "name": name,
            "pattern": pattern,
            "found": len(matches) == 1,
            "matches": [relpath(p) for p in matches],
        }
        if len(matches) == 1:
            path = matches[0]
            full_text = run_pdftotext(path)
            early_text = run_pdftotext(path, 1, min(3, run_pdfinfo_pages(path) or 3))
            row.update({
                "path": relpath(path),
                "sha256": sha256(path),
                "pages": run_pdfinfo_pages(path),
                "text_chars": len(full_text),
                "early_result_signal": first_matching_line(
                    early_text,
                    [
                        r"Fig(?:ure)?\.?\s*\d+",
                        r"Table\s*\d+",
                        r"latency",
                        r"throughput",
                        r"simulation time",
                        r"trade[- ]?off",
                        r"comparison",
                    ],
                ),
                "comparison_table_signal": first_matching_line(
                    full_text,
                    [
                        r"Table\s*1.*comparison",
                        r"Categories and comparison",
                        r"comparison with previous",
                        r"capabilit",
                    ],
                ),
                "early_headings": extract_headings(early_text),
            })
        rows.append(row)
    return rows


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def input_order(main_tex: str) -> list[str]:
    return [m.group(1) for m in re.finditer(r"\\input\{([^}]+)\}", main_tex)]


def ordered_before(order: list[str], left: str, right: str) -> bool:
    try:
        return order.index(left) < order.index(right)
    except ValueError:
        return False


def all_needles_present(text: str, needles: list[str]) -> bool:
    return all(n in text for n in needles)


def build_aegis_outline() -> dict[str, Any]:
    main_tex = read_text(PAPER / "main.tex")
    intro_tex = read_text(PAPER / "1_Introduction.tex")
    design_tex = read_text(PAPER / "3_Design.tex")
    evaluation_tex = read_text(PAPER / "4_Evaluation.tex")
    discussion_tex = read_text(PAPER / "10_Discussion_and_Limitations.tex")
    first_page = run_pdftotext(PAPER / "main.pdf", 1, 1)
    full_pdf = run_pdftotext(PAPER / "main.pdf")

    order = input_order(main_tex)
    top_level = {
        "order": order,
        "matches_required_order": order == MAIN_ORDER,
        "design_before_evaluation": ordered_before(order, "3_Design.tex", "4_Evaluation.tex"),
        "related_after_evaluation": ordered_before(order, "4_Evaluation.tex", "5_Related_Works.tex"),
        "discussion_before_related": ordered_before(order, "10_Discussion_and_Limitations.tex", "5_Related_Works.tex"),
    }

    mechanism_rows = []
    for row in MECHANISM_MAP:
        design_present = all_needles_present(design_tex + "\n" + main_tex + "\n" + intro_tex,
                                             row["design_needles"])
        eval_present = all_needles_present(evaluation_tex + "\n" + full_pdf, row["evaluation_needles"])
        mechanism_rows.append({
            "mechanism": row["mechanism"],
            "design_needles": row["design_needles"],
            "evaluation_needles": row["evaluation_needles"],
            "design_present": design_present,
            "evaluation_present": eval_present,
            "mapped": design_present and eval_present,
        })

    return {
        "pdf": relpath(PAPER / "main.pdf"),
        "pages": run_pdfinfo_pages(PAPER / "main.pdf"),
        "first_page_pressure_result": {
            "present": ("Figure 1:" in first_page and "SQLite p99" in first_page)
            or "Table 1: The edge-storage design pressure" in first_page,
            "evidence": first_matching_line(
                first_page,
                [
                    r"Figure\s*1: First-page pressure result",
                    r"SQLite p99",
                    r"Table\s*1: The edge-storage design pressure",
                ],
            ),
        },
        "capability_table": {
            "present": "tab:capability_matrix" in intro_tex
            and all(term in intro_tex for term in [
                "Plaintext", "gocryptfs", "fscrypt", "dm-crypt",
                "fs-verity/dm-integrity", "TPM/TEE-backed",
                "GPU-storage systems", "AEGIS-Q",
            ]),
            "evidence": first_matching_line(full_pdf, [r"Table\s*1: Capability comparison"]),
        },
        "top_level_order": top_level,
        "mechanism_map": mechanism_rows,
        "deployment_takeaway": {
            "present": "\\subsection{Deployment takeaway}" in discussion_tex
            and "SQLite" in discussion_tex
            and "append-log/cache-manifest remounts" in discussion_tex
            and "CPU AES-GCM publication" in discussion_tex
            and "slack-gated PQC maintenance" in discussion_tex,
            "source": "Paper/10_Discussion_and_Limitations.tex",
        },
        "known_next_gate_warnings": WARNING_TEXT,
    }


def write_markdown(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# Accepted-paper structure audit",
        "",
        f"- Overall pass: `{report['overall_pass']}`",
        f"- Required source examples found: `{report['summary']['required_sources_found']}` / `{report['summary']['required_sources_total']}`",
        f"- AEGIS-Q page count at audit time: `{report['aegis_outline']['pages']}`",
        "",
        "## Required local sources",
        "",
        "| Source | Pages | Early result/table signal | Comparison/capability signal |",
        "| --- | ---: | --- | --- |",
    ]
    for row in report["required_sources"]:
        signal = row.get("early_result_signal") or "missing"
        comparison = row.get("comparison_table_signal") or "not detected"
        lines.append(f"| {row['name']} | {row.get('pages', 'missing')} | {signal} | {comparison} |")

    outline = report["aegis_outline"]
    lines += [
        "",
        "## AEGIS-Q spine",
        "",
        f"- First-page pressure result: `{outline['first_page_pressure_result']['present']}` ({outline['first_page_pressure_result']['evidence']})",
        f"- Capability table: `{outline['capability_table']['present']}` ({outline['capability_table']['evidence']})",
        f"- Top-level order matches audit target: `{outline['top_level_order']['matches_required_order']}`",
        f"- Deployment takeaway present: `{outline['deployment_takeaway']['present']}`",
        "",
        "## Mechanism-to-evaluation map",
        "",
        "| Mechanism | Design present | Evaluation present | Mapped |",
        "| --- | ---: | ---: | ---: |",
    ]
    for row in outline["mechanism_map"]:
        lines.append(
            f"| {row['mechanism']} | `{row['design_present']}` | "
            f"`{row['evaluation_present']}` | `{row['mapped']}` |"
        )

    if report["violations"]:
        lines += ["", "## Violations", ""]
        lines += [f"- {v}" for v in report["violations"]]
    if outline["known_next_gate_warnings"]:
        lines += ["", "## Next-gate warnings", ""]
        lines += [f"- {w}" for w in outline["known_next_gate_warnings"]]

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_report() -> dict[str, Any]:
    required_sources = find_required_sources()
    aegis_outline = build_aegis_outline()

    violations: list[str] = []
    if not all(row.get("found") and row.get("text_chars", 0) > 0 for row in required_sources):
        violations.append("not all required local source PDFs are uniquely found and text-extractable")
    if not aegis_outline["first_page_pressure_result"]["present"]:
        violations.append("AEGIS-Q lacks a first-page pressure/result table or figure")
    if not aegis_outline["capability_table"]["present"]:
        violations.append("AEGIS-Q lacks a system-boundary/capability table")
    if not aegis_outline["top_level_order"]["matches_required_order"]:
        violations.append("AEGIS-Q top-level source order does not match the accepted-paper spine target")
    if not all(row["mapped"] for row in aegis_outline["mechanism_map"]):
        violations.append("one or more AEGIS-Q mechanisms lack a design/evaluation mapping")
    if not aegis_outline["deployment_takeaway"]["present"]:
        violations.append("AEGIS-Q lacks a deployment-style takeaway in Discussion")

    return {
        "schema_version": 1,
        "scope": [
            "local accepted-paper PDFs under Paper/Previous paper/",
            "AEGIS-Q compiled PDF and LaTeX source order",
            "structure only; no benchmark numbers are inferred from source PDFs",
        ],
        "required_sources": required_sources,
        "aegis_outline": aegis_outline,
        "summary": {
            "required_sources_total": len(REQUIRED_SOURCES),
            "required_sources_found": sum(1 for row in required_sources if row.get("found")),
            "required_sources_text_extractable": sum(1 for row in required_sources if row.get("text_chars", 0) > 0),
            "mechanisms_mapped": sum(1 for row in aegis_outline["mechanism_map"] if row["mapped"]),
            "mechanisms_total": len(aegis_outline["mechanism_map"]),
            "violations": len(violations),
        },
        "violations": violations,
        "overall_pass": not violations,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    report = build_report()
    json_path = args.out / "accepted_paper_structure_audit.json"
    md_path = args.out / "accepted_paper_structure_audit.md"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown(report, md_path)
    print(json.dumps({
        "out_dir": relpath(args.out),
        "overall_pass": report["overall_pass"],
        "required_sources_found": report["summary"]["required_sources_found"],
        "mechanisms_mapped": report["summary"]["mechanisms_mapped"],
        "violations": report["summary"]["violations"],
    }, indent=2, sort_keys=True))
    return 0 if report["overall_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
