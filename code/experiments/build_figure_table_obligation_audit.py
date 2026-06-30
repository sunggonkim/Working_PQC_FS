#!/usr/bin/env python3
"""Audit that every main-paper figure/table answers an RQ or design obligation."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
PAPER = ROOT / "Paper"
DEFAULT_OUT = ROOT / "artifacts" / "reports" / "figure_table_obligations"

PAPER_SOURCES = [
    "Paper/1_Introduction.tex",
    "Paper/2_Background.tex",
    "Paper/3_Design.tex",
    "Paper/7_Implementation_Details.tex",
    "Paper/8_Security_Analysis.tex",
    "Paper/4_Evaluation.tex",
    "Paper/generated_qos_recovery_table.tex",
    "Paper/10_Discussion_and_Limitations.tex",
    "Paper/5_Related_Works.tex",
    "Paper/6_Conclusion.tex",
]

EXPECTED_OBLIGATIONS = {
    "fig:first_page_qos": {
        "kind": "RQ3",
        "obligation": "first-page SQLite pressure/hero result",
        "caption_terms": ["SQLite", "RQ3", "deadline"],
    },
    "tab:capability_matrix": {
        "kind": "design",
        "obligation": "capability table defining the design gap",
        "caption_terms": ["Capability", "design gap"],
    },
    "tab:design_goals": {
        "kind": "design",
        "obligation": "formal storage-protocol invariant table",
        "caption_terms": ["invariants", "boundaries"],
    },
    "fig:overall_procedure": {
        "kind": "design",
        "obligation": "architecture and plane-separation figure",
        "caption_terms": ["runtime architecture", "claim firewall"],
    },
    "fig:djc_state_machine": {
        "kind": "design",
        "obligation": "D/J/C/xattr publication state machine",
        "caption_terms": ["publication protocol", "claim boundary"],
    },
    "tab:memory_compat": {
        "kind": "design/RQ2/RQ5",
        "obligation": "memory-path claim boundary",
        "caption_terms": ["Memory-path", "hardware"],
    },
    "tab:impl_boundaries": {
        "kind": "design",
        "obligation": "implementation boundary summary",
        "caption_terms": ["Implementation", "boundaries"],
    },
    "tab:threat_boundary": {
        "kind": "RQ1/RQ4",
        "obligation": "threat-model and recovery boundary",
        "caption_terms": ["Threat", "boundaries"],
    },
    "tab:benchmark_workloads": {
        "kind": "RQ1-RQ5",
        "obligation": "evaluation scope by research question",
        "caption_terms": ["Evaluation", "research question"],
    },
    "fig:problem_boundary": {
        "kind": "motivation",
        "obligation": "problem boundary for the edge-runtime thesis",
        "caption_terms": ["Motivating boundary", "authenticated publication"],
    },
    "fig:evaluation_summary": {
        "kind": "RQ2/RQ3/RQ5",
        "obligation": "evaluation spine across cost, QoS, and key-plane placement",
        "caption_terms": ["Evaluation summary", "RQ2", "RQ3", "RQ5"],
    },
    "tab:qos_sqlite_recovery": {
        "kind": "RQ3",
        "obligation": "SQLite QoS recovery hero result",
        "caption_terms": ["SQLite", "p99", "medians"],
    },
}

PATH_LIST_PATTERNS = [
    r"artifacts\s*/",
    r"Raw data\s*:",
    r"Raw captures\s*:",
]

ALLOWED_ALIAS_LABELS = {"tab:applicability_boundary"}


def relpath(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def strip_latex_comments(text: str) -> str:
    lines = []
    for line in text.splitlines():
        out = []
        escaped = False
        for char in line:
            if char == "%" and not escaped:
                break
            out.append(char)
            escaped = char == "\\" and not escaped
            if char != "\\":
                escaped = False
        lines.append("".join(out))
    return "\n".join(lines) + "\n"


def extract_balanced_brace(text: str, start: int) -> str | None:
    brace = text.find("{", start)
    if brace < 0:
        return None
    depth = 0
    for idx in range(brace, len(text)):
        if text[idx] == "{":
            depth += 1
        elif text[idx] == "}":
            depth -= 1
            if depth == 0:
                return text[brace + 1:idx]
    return None


def normalize(text: str) -> str:
    return " ".join(text.split())


def extract_caption(env_body: str) -> str:
    pos = env_body.find(r"\caption")
    if pos < 0:
        return ""
    return normalize(extract_balanced_brace(env_body, pos) or "")


def run_pdfinfo_pages(path: Path) -> int | None:
    proc = subprocess.run(["pdfinfo", str(path)], check=True, text=True,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    for line in proc.stdout.splitlines():
        if line.startswith("Pages:"):
            return int(line.split(":", 1)[1].strip())
    return None


def extract_environments(path: Path) -> list[dict[str, Any]]:
    text = strip_latex_comments(read_text(path))
    rows: list[dict[str, Any]] = []
    pattern = re.compile(r"\\begin\{(figure\*?|table\*?)\}(.*?)\\end\{\1\}", re.S)
    for match in pattern.finditer(text):
        body = match.group(2)
        labels = re.findall(r"\\label\{([^}]+)\}", body)
        label = labels[0] if labels else None
        rows.append({
            "path": relpath(path),
            "line": text[:match.start()].count("\n") + 1,
            "environment": match.group(1),
            "label": label,
            "extra_labels": labels[1:],
            "caption": extract_caption(body),
        })
    return rows


def contains_terms(text: str, terms: list[str]) -> bool:
    lowered = text.lower()
    return all(term.lower() in lowered for term in terms)


def reference_count(full_text: str, label: str) -> int:
    pattern = re.compile(r"\\(?:ref|pageref|autoref)\{" + re.escape(label) + r"\}")
    return len(pattern.findall(full_text))


def path_list_hits(source_texts: dict[str, str]) -> list[dict[str, str]]:
    hits = []
    for source, text in source_texts.items():
        for line_no, line in enumerate(text.splitlines(), start=1):
            for pattern in PATH_LIST_PATTERNS:
                if re.search(pattern, line, re.IGNORECASE):
                    hits.append({
                        "path": source,
                        "line": line_no,
                        "pattern": pattern,
                        "text": normalize(line.strip()),
                    })
    return hits


def build_report() -> dict[str, Any]:
    source_texts = {
        source: strip_latex_comments(read_text(ROOT / source))
        for source in PAPER_SOURCES
    }
    full_text = "\n".join(source_texts.values())
    envs = []
    for source in PAPER_SOURCES:
        envs.extend(extract_environments(ROOT / source))

    labels = [row["label"] for row in envs if row["label"]]
    duplicate_labels = sorted({label for label in labels if labels.count(label) > 1})
    found = set(labels)
    expected = set(EXPECTED_OBLIGATIONS)

    rows = []
    for env in envs:
        label = env["label"]
        obligation = EXPECTED_OBLIGATIONS.get(label or "")
        caption_terms_present = (
            bool(obligation) and contains_terms(env["caption"], obligation["caption_terms"])
        )
        ref_count = reference_count(full_text, label) if label else 0
        allowed_extra_labels = set(env["extra_labels"]).issubset(ALLOWED_ALIAS_LABELS)
        rows.append({
            **env,
            "kind": obligation["kind"] if obligation else None,
            "obligation": obligation["obligation"] if obligation else None,
            "caption_terms": obligation["caption_terms"] if obligation else [],
            "caption_terms_present": caption_terms_present,
            "reference_count": ref_count,
            "passes": bool(label)
            and bool(obligation)
            and allowed_extra_labels
            and caption_terms_present
            and ref_count > 0,
        })

    violations = []
    if found != expected:
        missing = sorted(expected - found)
        extra = sorted(found - expected)
        if missing:
            violations.append(f"missing expected figure/table labels: {', '.join(missing)}")
        if extra:
            violations.append(f"unexpected figure/table labels: {', '.join(extra)}")
    if duplicate_labels:
        violations.append(f"duplicate figure/table labels: {', '.join(duplicate_labels)}")
    for row in rows:
        if not row["passes"]:
            violations.append(f"figure/table obligation failed for {row['label'] or row['path']}")

    path_hits = path_list_hits(source_texts)
    if path_hits:
        violations.append("rendered paper source still contains artifact/raw path-list text")

    pages = run_pdfinfo_pages(PAPER / "main.pdf")
    if pages != 12:
        violations.append("Paper/main.pdf is not 12 pages")

    return {
        "schema_version": 1,
        "scope": [
            "all figure/table environments in the main paper inputs",
            "one obligation mapping per figure/table label",
            "caption text names the RQ/design obligation rather than a raw artifact path",
            "labels are referenced from the paper and Paper/main.pdf has 12 pages",
        ],
        "expected_labels": sorted(EXPECTED_OBLIGATIONS),
        "found_labels": sorted(found),
        "figure_tables": rows,
        "duplicate_labels": duplicate_labels,
        "rendered_path_list_hits": path_hits,
        "checks": {
            "all_expected_labels_present": found == expected,
            "no_duplicate_labels": not duplicate_labels,
            "all_figure_tables_have_obligations": all(row["passes"] for row in rows),
            "no_rendered_artifact_path_lists": not path_hits,
            "paper_pages_12": pages == 12,
        },
        "pages": pages,
        "violations": violations,
        "overall_pass": not violations,
    }


def write_markdown(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# Figure/table obligation audit",
        "",
        f"- Overall pass: `{report['overall_pass']}`",
        f"- Paper pages: `{report['pages']}`",
        f"- Figure/table count: `{len(report['figure_tables'])}`",
        "",
        "## Obligations",
        "",
        "| Label | Source | Kind | Obligation | Caption terms | References | Pass |",
        "| --- | --- | --- | --- | ---: | ---: | ---: |",
    ]
    for row in report["figure_tables"]:
        lines.append(
            f"| `{row['label']}` | `{row['path']}:{row['line']}` | "
            f"`{row['kind']}` | {row['obligation']} | "
            f"`{row['caption_terms_present']}` | `{row['reference_count']}` | `{row['passes']}` |"
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

    if report["rendered_path_list_hits"]:
        lines += ["", "## Rendered Path-List Hits", ""]
        for hit in report["rendered_path_list_hits"]:
            lines.append(f"- `{hit['path']}:{hit['line']}` `{hit['pattern']}`: {hit['text']}")

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
    json_path = args.out / "figure_table_obligations.json"
    md_path = args.out / "figure_table_obligations.md"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown(report, md_path)
    print(json.dumps({
        "out_dir": relpath(args.out),
        "overall_pass": report["overall_pass"],
        "pages": report["pages"],
        "figure_tables": len(report["figure_tables"]),
        "violations": len(report["violations"]),
    }, indent=2, sort_keys=True))
    return 0 if report["overall_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
