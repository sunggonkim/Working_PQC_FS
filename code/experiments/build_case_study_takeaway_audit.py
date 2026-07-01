#!/usr/bin/env python3
"""Audit that the deployment takeaway reuses the retained hero result."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
PAPER = ROOT / "Paper"
DEFAULT_OUT = ROOT / "artifacts" / "reports" / "case_study_takeaway"
HERO_CONTRACT = ROOT / "artifacts" / "reports" / "hero_result_contract" / "hero_result_contract.json"

REQUIRED_TAKEAWAY_TERMS = [
    "evaluated envelope is local",
    "SQLite",
    "elastic background writes",
    "append-log/cache-manifest remounts",
    "authenticated FUSE",
    "Figure~\\ref{fig:first_page_qos}",
    "Table~\\ref{tab:qos_sqlite_recovery}",
    "not a separate deployment anecdote",
]

BOUNDARY_TERMS = [
    "kernel/FUSE-daemon trust",
    "Jetson/CUDA/TPM dependencies",
    "CPU AES-GCM publication",
    "slack-gated PQC maintenance",
    "executor-local managed memory",
]


def relpath(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def run_pdfinfo_pages(path: Path) -> int | None:
    proc = subprocess.run(["pdfinfo", str(path)], check=True, text=True,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    for line in proc.stdout.splitlines():
        if line.startswith("Pages:"):
            return int(line.split(":", 1)[1].strip())
    return None


def extract_subsection(text: str, heading: str) -> str:
    pattern = re.compile(
        r"\\subsection\{" + re.escape(heading) + r"\}(.*?)(?=\\subsection\{|\\section\{|\\bibliographystyle|$)",
        re.DOTALL,
    )
    match = pattern.search(text)
    return match.group(1).strip() if match else ""


def build_report() -> dict[str, Any]:
    discussion = read_text(PAPER / "10_Discussion_and_Limitations.tex")
    intro = read_text(PAPER / "1_Introduction.tex")
    eval_text = read_text(PAPER / "4_Evaluation.tex")
    generated_table = read_text(PAPER / "generated_qos_recovery_table.tex")
    takeaway = extract_subsection(discussion, "Deployment takeaway")
    hero = read_json(HERO_CONTRACT) if HERO_CONTRACT.exists() else {}

    takeaway_terms = {term: term in takeaway for term in REQUIRED_TAKEAWAY_TERMS}
    boundary_terms = {term: term in takeaway for term in BOUNDARY_TERMS}
    artifact_path = hero.get("artifact_path")
    artifact_present = bool(artifact_path) and (ROOT / artifact_path).exists()
    hero_id = hero.get("hero_id")

    checks = {
        "takeaway_subsection_present": bool(takeaway),
        "takeaway_terms_present": all(takeaway_terms.values()),
        "boundary_terms_present": all(boundary_terms.values()),
        "hero_contract_pass": HERO_CONTRACT.exists() and hero.get("overall_pass") is True,
        "hero_artifact_present": artifact_present,
        "figure_label_present": "\\label{fig:first_page_qos}" in intro,
        "table_label_present": "\\label{tab:qos_sqlite_recovery}" in generated_table,
        "evaluation_uses_same_table": "Table~\\ref{tab:qos_sqlite_recovery}" in eval_text,
        "paper_pages_le_13": (run_pdfinfo_pages(PAPER / "main.pdf") or 999) <= 13,
    }
    violations = [name for name, passed in checks.items() if not passed]

    return {
        "schema_version": 1,
        "scope": [
            "Deployment takeaway in Paper/10_Discussion_and_Limitations.tex",
            "reuse of hero-result contract and retained SQLite/FUSE artifact",
            "case-study boundaries against unrelated deployment claims",
        ],
        "takeaway_text": " ".join(takeaway.split()),
        "takeaway_terms": takeaway_terms,
        "boundary_terms": boundary_terms,
        "hero_contract": {
            "path": relpath(HERO_CONTRACT),
            "present": HERO_CONTRACT.exists(),
            "overall_pass": hero.get("overall_pass"),
            "hero_id": hero_id,
            "artifact_path": artifact_path,
            "artifact_present": artifact_present,
            "headline_claim_displayed": hero.get("headline_claim_displayed"),
        },
        "checks": checks,
        "pages": run_pdfinfo_pages(PAPER / "main.pdf"),
        "violations": violations,
        "overall_pass": not violations,
    }


def write_markdown(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# Case-study takeaway audit",
        "",
        f"- Overall pass: `{report['overall_pass']}`",
        f"- Paper pages: `{report['pages']}`",
        f"- Hero id: `{report['hero_contract']['hero_id']}`",
        f"- Hero artifact: `{report['hero_contract']['artifact_path']}`",
        "",
        "## Takeaway",
        "",
        report["takeaway_text"],
        "",
        "## Required Terms",
        "",
        "| Term | Present |",
        "| --- | ---: |",
    ]
    for term, present in report["takeaway_terms"].items():
        lines.append(f"| `{term}` | `{present}` |")

    lines += [
        "",
        "## Boundary Terms",
        "",
        "| Term | Present |",
        "| --- | ---: |",
    ]
    for term, present in report["boundary_terms"].items():
        lines.append(f"| `{term}` | `{present}` |")

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
    json_path = args.out / "case_study_takeaway.json"
    md_path = args.out / "case_study_takeaway.md"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown(report, md_path)
    print(json.dumps({
        "out_dir": relpath(args.out),
        "overall_pass": report["overall_pass"],
        "pages": report["pages"],
        "violations": len(report["violations"]),
    }, indent=2, sort_keys=True))
    return 0 if report["overall_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
