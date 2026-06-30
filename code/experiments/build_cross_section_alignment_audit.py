#!/usr/bin/env python3
"""Audit hero-claim alignment across the paper's main narrative sections."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
PAPER = ROOT / "Paper"
DEFAULT_OUT = ROOT / "artifacts" / "reports" / "cross_section_alignment"
HERO_CONTRACT = ROOT / "artifacts" / "reports" / "hero_result_contract" / "hero_result_contract.json"

SECTION_FILES = {
    "abstract": "Paper/main.tex",
    "introduction": "Paper/1_Introduction.tex",
    "design": "Paper/3_Design.tex",
    "evaluation": "Paper/4_Evaluation.tex",
    "conclusion": "Paper/6_Conclusion.tex",
}

HERO_TERMS = [
    "SQLite",
    "p99",
    "9.62",
    "8.15",
    "3.02",
    "background",
]

MODE_TERMS = [
    "secure-storage pressure",
    "AEGIS-Q",
]

SECTION_REQUIREMENTS = {
    "abstract": [
        "SQLite",
        "p99",
        "9.62",
        "8.15",
        "3.02",
        "background",
        "not deployed-filesystem",
    ],
    "introduction": [
        "SQLite",
        "p99",
        "9.62",
        "8.15",
        "3.02",
        "not peak filesystem throughput",
    ],
    "design": [
        "SQLite",
        "9.62",
        "8.15",
        "3.02",
        "not by claiming application scheduling",
    ],
    "evaluation": [
        "Table~\\ref{tab:qos_sqlite_recovery}",
        "9.62",
        "8.15",
        "3.02",
        "bounded storage-visible control",
    ],
    "conclusion": [
        "secure edge file encryption",
        "CPU",
        "ML-KEM",
        "8.15",
        "SQLite",
    ],
}

UNSUPPORTED_PATTERNS = [
    r"direct\s+NVMe-to-UVM\s+DMA",
    r"eBPF/io\\?_uring\s+completion\s+bypass",
    r"io\\?_uring/eBPF\s+completion\s+bypass",
    r"persistent\s+PCR-bound\s+freshness",
    r"power-loss\s+(?:crash\s+)?certification",
    r"foreground\s+AI\s+p99\s+recovery",
    r"TensorRT\s+p99\s+recovery",
    r"CUDA-independent\s+deployment",
    r"portability\s+beyond\s+the\s+tested\s+stack",
]

NEGATION_TERMS = [
    "not",
    "no ",
    "does not",
    "do not",
    "without",
    "excludes",
    "scoped out",
    "not claimed",
    "still lacks",
    "does not extend",
]


def relpath(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def normalize(text: str) -> str:
    return " ".join(text.split())


def contains_all(text: str, needles: list[str]) -> bool:
    lowered = text.lower()
    return all(needle.lower() in lowered for needle in needles)


def extract_abstract(main_tex: str) -> str:
    match = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", main_tex, re.S)
    return match.group(1) if match else ""


def run_pdfinfo_pages(path: Path) -> int | None:
    proc = subprocess.run(["pdfinfo", str(path)], check=True, text=True,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    for line in proc.stdout.splitlines():
        if line.startswith("Pages:"):
            return int(line.split(":", 1)[1].strip())
    return None


def find_first_line(path_str: str, needles: list[str]) -> dict[str, Any]:
    path = ROOT / path_str
    for line_no, line in enumerate(read_text(path).splitlines(), start=1):
        if contains_all(line, needles):
            return {
                "path": path_str,
                "line": line_no,
                "text": normalize(line.strip()),
                "present": True,
            }
    return {"path": path_str, "line": None, "text": None, "present": False}


def section_texts() -> dict[str, str]:
    main = read_text(PAPER / "main.tex")
    return {
        "abstract": extract_abstract(main),
        "introduction": read_text(PAPER / "1_Introduction.tex"),
        "design": read_text(PAPER / "3_Design.tex"),
        "evaluation": read_text(PAPER / "4_Evaluation.tex"),
        "conclusion": read_text(PAPER / "6_Conclusion.tex"),
    }


def unsupported_hits(texts: dict[str, str]) -> list[dict[str, str]]:
    hits: list[dict[str, str]] = []
    for section, text in texts.items():
        flattened = normalize(text)
        for pattern in UNSUPPORTED_PATTERNS:
            for match in re.finditer(pattern, flattened, flags=re.IGNORECASE):
                start = max(0, match.start() - 200)
                end = min(len(flattened), match.end() + 120)
                context = flattened[start:end]
                if any(term in context.lower() for term in NEGATION_TERMS):
                    continue
                hits.append({
                    "section": section,
                    "pattern": pattern,
                    "context": context,
                })
    return hits


def build_report() -> dict[str, Any]:
    texts = section_texts()
    section_rows = []
    for name, text in texts.items():
        terms_present = {term: term.lower() in text.lower() for term in HERO_TERMS}
        mode_present = {term: term.lower() in text.lower() for term in MODE_TERMS}
        required_terms = SECTION_REQUIREMENTS[name]
        required_present = {term: term.lower() in text.lower() for term in required_terms}
        section_rows.append({
            "section": name,
            "path": SECTION_FILES[name],
            "hero_terms": terms_present,
            "mode_terms": mode_present,
            "required_terms": required_present,
            "passes": all(required_present.values()) and all(mode_present.values()),
        })

    hero_contract = json.loads(HERO_CONTRACT.read_text(encoding="utf-8"))
    support = {
        "path": relpath(HERO_CONTRACT),
        "present": HERO_CONTRACT.exists(),
        "overall_pass": hero_contract.get("overall_pass"),
        "hero_id": hero_contract.get("hero_id"),
        "violations": len(hero_contract.get("violations", [])),
        "displayed_claim": hero_contract.get("hero", {}).get("headline_claim_displayed"),
    }

    source_anchors = {
        "abstract": find_first_line("Paper/main.tex", ["SQLite p99", "9.62", "8.15", "3.02"]),
        "introduction": find_first_line("Paper/1_Introduction.tex", ["Figure~\\ref{fig:first_page_qos}", "9.62", "8.15", "3.02"]),
        "design": find_first_line("Paper/3_Design.tex", ["SQLite", "9.62", "8.15", "3.02"]),
        "evaluation": find_first_line("Paper/4_Evaluation.tex", ["Table~\\ref{tab:qos_sqlite_recovery}", "9.62", "8.15", "3.02"]),
        "conclusion": find_first_line("Paper/6_Conclusion.tex", ["SQLite", "8.15", "ML-KEM"]),
    }

    unsupported = unsupported_hits(texts)
    pages = run_pdfinfo_pages(PAPER / "main.pdf")
    checks = {
        "paper_pages_12": pages == 12,
        "hero_contract_pass": support["present"] and support["overall_pass"] is True and support["violations"] == 0,
        "all_sections_have_same_rounded_hero_terms": all(row["passes"] for row in section_rows),
        "source_anchors_present": all(row["present"] for row in source_anchors.values()),
        "no_unsupported_positive_claims": not unsupported,
    }
    violations = [name for name, passed in checks.items() if not passed]

    return {
        "schema_version": 1,
        "scope": [
            "abstract, introduction, design, evaluation, and conclusion",
            "rounded SQLite hero claim terms: 9.62, 8.15, 3.02, p99, background",
            "unsupported-claim scan for unnegated restored claims",
            "retained hero-result contract",
        ],
        "pages": pages,
        "hero_contract": support,
        "section_alignment": section_rows,
        "source_anchors": source_anchors,
        "unsupported_hits": unsupported,
        "checks": checks,
        "violations": violations,
        "overall_pass": not violations,
    }


def write_markdown(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# Cross-section alignment audit",
        "",
        f"- Overall pass: `{report['overall_pass']}`",
        f"- Paper pages: `{report['pages']}`",
        f"- Hero id: `{report['hero_contract']['hero_id']}`",
        "",
        "## Section Alignment",
        "",
        "| Section | Path | Hero terms | Mode terms | Pass |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for row in report["section_alignment"]:
        lines.append(
            f"| `{row['section']}` | `{row['path']}` | "
            f"`{all(row['required_terms'].values())}` | `{all(row['mode_terms'].values())}` | `{row['passes']}` |"
        )

    lines += [
        "",
        "## Source Anchors",
        "",
        "| Section | Source | Present |",
        "| --- | --- | ---: |",
    ]
    for section, row in report["source_anchors"].items():
        lines.append(f"| `{section}` | `{row['path']}:{row['line']}` | `{row['present']}` |")

    lines += [
        "",
        "## Checks",
        "",
        "| Check | Pass |",
        "| --- | ---: |",
    ]
    for name, passed in report["checks"].items():
        lines.append(f"| `{name}` | `{passed}` |")

    if report["unsupported_hits"]:
        lines += ["", "## Unsupported Hits", ""]
        for hit in report["unsupported_hits"]:
            lines.append(f"- `{hit['section']}` `{hit['pattern']}`: {hit['context']}")

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
    json_path = args.out / "cross_section_alignment.json"
    md_path = args.out / "cross_section_alignment.md"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown(report, md_path)
    print(json.dumps({
        "out_dir": relpath(args.out),
        "overall_pass": report["overall_pass"],
        "pages": report["pages"],
        "sections": len(report["section_alignment"]),
        "violations": len(report["violations"]),
    }, indent=2, sort_keys=True))
    return 0 if report["overall_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
