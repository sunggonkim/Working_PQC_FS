#!/usr/bin/env python3
"""Audit that artifact bookkeeping stays out of the submitted paper body."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
PAPER = ROOT / "Paper"
README = ROOT / "README.md"
REPRO_MANIFEST = ROOT / "artifacts" / "repro_bundle" / "manifest.json"
DEFAULT_OUT = ROOT / "artifacts" / "reports" / "artifact_ledger_separation"

PAPER_TEX = [
    PAPER / "main.tex",
    PAPER / "1_Introduction.tex",
    PAPER / "2_Background.tex",
    PAPER / "3_Design.tex",
    PAPER / "4_Evaluation.tex",
    PAPER / "5_Related_Works.tex",
    PAPER / "6_Conclusion.tex",
    PAPER / "7_Implementation_Details.tex",
    PAPER / "8_Security_Analysis.tex",
    PAPER / "10_Discussion_and_Limitations.tex",
    PAPER / "generated_qos_recovery_table.tex",
]

FORBIDDEN_PAPER_PATTERNS = {
    "repository_artifact_path": re.compile(r"\b(?:artifacts|code/experiments|experiments)/[A-Za-z0-9_.\-/]+"),
    "artifact_index_heading": re.compile(r"\\(?:section|subsection|paragraph)\*?\{[^}]*artifact\s+(?:index|ledger|archive|dump)[^}]*\}", re.IGNORECASE),
    "repro_bundle_heading": re.compile(r"\\(?:section|subsection|paragraph)\*?\{[^}]*repro(?:ducibility)?\s+bundle[^}]*\}", re.IGNORECASE),
    "appendix_artifact_body": re.compile(r"\\appendix|artifact-style appendix|artifact ledger|artifact dump", re.IGNORECASE),
    "raw_path_list_wording": re.compile(r"raw\s+(?:artifact|path|file)\s+list|artifact\s+path\s+list", re.IGNORECASE),
}

README_REQUIRED_PHRASES = [
    "Completed evidence archive moved from `SUBMISSION_CHECKLIST.md`",
    "checks and artifact bookkeeping live here",
    "The detailed artifact index remains",
    "The main paper should reference only the small number of artifacts",
    "The submitted paper no longer includes an artifact-style appendix",
    "### Artifact index",
]

PAPER_CONCLUSION_PHRASES = [
    "not peak filesystem throughput",
    "known cost boundary",
    "not current comparison evidence",
    "not a broad workload suite",
    "not a claim that the persistent filesystem anchor itself is PCR-sealed",
]


def relpath(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def run_pdfinfo_pages(path: Path) -> int | None:
    proc = subprocess.run(["pdfinfo", str(path)], check=True, text=True,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    for line in proc.stdout.splitlines():
        if line.startswith("Pages:"):
            return int(line.split(":", 1)[1].strip())
    return None


def scan_paper() -> dict[str, Any]:
    hits: list[dict[str, Any]] = []
    term_counts = {"retained": 0, "artifact": 0, "audit": 0, "bundle": 0, "manifest": 0}
    conclusion_hits: dict[str, bool] = {phrase: False for phrase in PAPER_CONCLUSION_PHRASES}

    for path in PAPER_TEX:
        text = read(path)
        lowered = text.lower()
        for term in term_counts:
            term_counts[term] += lowered.count(term)
        for phrase in conclusion_hits:
            if phrase.lower() in lowered:
                conclusion_hits[phrase] = True
        for name, pattern in FORBIDDEN_PAPER_PATTERNS.items():
            for match in pattern.finditer(text):
                line = text.count("\n", 0, match.start()) + 1
                snippet = " ".join(text[match.start():match.end()].split())
                hits.append({
                    "pattern": name,
                    "path": relpath(path),
                    "line": line,
                    "text": snippet,
                })

    return {
        "files": [relpath(path) for path in PAPER_TEX],
        "forbidden_hits": hits,
        "forbidden_hit_count": len(hits),
        "term_counts": term_counts,
        "conclusion_phrases": conclusion_hits,
        "conclusion_phrase_hits": sum(1 for present in conclusion_hits.values() if present),
    }


def inspect_readme() -> dict[str, Any]:
    text = read(README)
    required = {phrase: phrase in text for phrase in README_REQUIRED_PHRASES}
    return {
        "path": relpath(README),
        "present": README.exists(),
        "required_phrases": required,
        "complete": all(required.values()),
    }


def inspect_repro_manifest() -> dict[str, Any]:
    row: dict[str, Any] = {
        "path": relpath(REPRO_MANIFEST),
        "present": REPRO_MANIFEST.exists(),
        "file_count": 0,
        "paper_pages": None,
        "has_commands": False,
        "has_hashes": (ROOT / "artifacts" / "repro_bundle" / "sha256sums.txt").exists(),
    }
    if REPRO_MANIFEST.exists():
        data = json.loads(read(REPRO_MANIFEST))
        files = data.get("files", [])
        row.update({
            "file_count": len(files),
            "paper_pages": data.get("paper_pages"),
            "paper_pdf": data.get("paper_pdf"),
            "has_commands": bool(data.get("commands")),
        })
    row["complete"] = (
        row["present"]
        and row["file_count"] > 0
        and row["paper_pages"] == 12
        and row["has_commands"]
        and row["has_hashes"]
    )
    return row


def build_report() -> dict[str, Any]:
    paper = scan_paper()
    readme = inspect_readme()
    repro = inspect_repro_manifest()
    pages = run_pdfinfo_pages(PAPER / "main.pdf")
    checks = {
        "paper_pdf_pages_12": pages == 12,
        "paper_has_no_repository_artifact_paths": paper["forbidden_hit_count"] == 0,
        "paper_keeps_conclusion_level_evidence_language": paper["conclusion_phrase_hits"] >= 4,
        "readme_hosts_artifact_index": readme["complete"],
        "repro_bundle_manifest_complete": repro["complete"],
    }
    violations = [name for name, passed in checks.items() if not passed]
    return {
        "schema_version": 1,
        "scope": [
            "Paper/*.tex must not contain repository artifact path lists or artifact-ledger sections",
            "README.md must host the artifact index and state that the paper is not an artifact appendix",
            "artifacts/repro_bundle/manifest.json must remain the machine-readable repro manifest",
        ],
        "paper_pages": pages,
        "paper": paper,
        "readme": readme,
        "repro_manifest": repro,
        "checks": checks,
        "violations": violations,
        "overall_pass": not violations,
    }


def write_markdown(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# Artifact-ledger separation audit",
        "",
        f"- Overall pass: `{report['overall_pass']}`",
        f"- Paper pages: `{report['paper_pages']}`",
        f"- Forbidden paper hits: `{report['paper']['forbidden_hit_count']}`",
        f"- README artifact index complete: `{report['readme']['complete']}`",
        f"- Repro manifest complete: `{report['repro_manifest']['complete']}`",
        "",
        "## Checks",
        "",
        "| Check | Pass |",
        "| --- | ---: |",
    ]
    for name, passed in report["checks"].items():
        lines.append(f"| `{name}` | `{passed}` |")

    lines += [
        "",
        "## Paper Term Counts",
        "",
        "| Term | Count |",
        "| --- | ---: |",
    ]
    for term, count in report["paper"]["term_counts"].items():
        lines.append(f"| `{term}` | {count} |")

    lines += [
        "",
        "## README Required Phrases",
        "",
        "| Phrase | Present |",
        "| --- | ---: |",
    ]
    for phrase, present in report["readme"]["required_phrases"].items():
        lines.append(f"| `{phrase}` | `{present}` |")

    if report["paper"]["forbidden_hits"]:
        lines += ["", "## Forbidden Paper Hits", ""]
        for hit in report["paper"]["forbidden_hits"]:
            lines.append(f"- `{hit['pattern']}` at `{hit['path']}:{hit['line']}`: {hit['text']}")
    if report["violations"]:
        lines += ["", "## Violations", ""]
        lines.extend(f"- {violation}" for violation in report["violations"])

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    report = build_report()
    json_path = args.out / "artifact_ledger_separation.json"
    md_path = args.out / "artifact_ledger_separation.md"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown(report, md_path)
    print(json.dumps({
        "out_dir": relpath(args.out),
        "overall_pass": report["overall_pass"],
        "paper_pages": report["paper_pages"],
        "forbidden_paper_hits": report["paper"]["forbidden_hit_count"],
        "violations": len(report["violations"]),
    }, indent=2, sort_keys=True))
    return 0 if report["overall_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
