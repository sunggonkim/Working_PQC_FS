#!/usr/bin/env python3
"""Audit that AEGIS-Q's novelty is isolated against deployed baselines."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
PAPER = ROOT / "Paper"
DEFAULT_OUT = ROOT / "artifacts" / "reports" / "novelty_isolation"

BASELINE_CLASSES = [
    "Plaintext",
    "gocryptfs",
    "fscrypt",
    "dm-crypt",
    "fs-verity/dm-integrity",
    "TPM/TEE-backed",
    "GPU-storage systems",
    "AEGIS-Q",
]

COMBINED_CAPABILITY_TERMS = [
    "authenticated-block publication",
    "storage-visible QoS/admission",
    "optional batched GPU key-plane work",
    "external replay-after-advance checks",
    "one evidence contract",
]

RELATED_WORK_TERMS = [
    "not a free kernel substitute",
    "authenticated FUSE records",
    "storage-visible policy",
    "optional GPU maintenance",
    "external replay checks",
    "secure edge-storage runtime",
]

EVIDENCE_GATES = [
    "artifacts/reports/paper_spine_gate/paper_spine_gate.json",
    "artifacts/reports/hero_result_contract/hero_result_contract.json",
    "artifacts/reports/design_eval_isomorphism/design_eval_isomorphism.json",
    "artifacts/validation/mechanism_ablation_manifest/mechanism_ablation_manifest.json",
    "artifacts/validation/integrity_comparison_manifest/integrity_comparison_manifest.json",
    "artifacts/validation/kernel_baseline_feasibility/kernel_baseline_feasibility.json",
    "artifacts/validation/frozen_gocryptfs_contract/frozen_gocryptfs_contract.json",
    "artifacts/validation/frozen_aegisq_contract/frozen_aegisq_contract.json",
]

FORBIDDEN_BROAD_CLAIMS = [
    r"AEGIS-Q\s+replaces\s+fscrypt",
    r"AEGIS-Q\s+replaces\s+dm-crypt",
    r"superior\s+to\s+fscrypt",
    r"superior\s+to\s+dm-crypt",
    r"deployed-filesystem superiority",
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


def first_line_with(text: str, needles: list[str]) -> dict[str, Any]:
    for idx, line in enumerate(text.splitlines(), start=1):
        if all(needle in line for needle in needles):
            return {
                "line": idx,
                "text": " ".join(line.strip().split()),
                "present": True,
            }
    return {"line": None, "text": None, "present": False}


def gate_status(path: str) -> dict[str, Any]:
    full = ROOT / path
    row: dict[str, Any] = {"path": path, "present": full.exists(), "overall_pass": None}
    if full.exists():
        data = read_json(full)
        row["overall_pass"] = data.get("overall_pass")
        if "checks" in data:
            row["checks_passed"] = all(data["checks"].values())
        if "violations" in data:
            row["violations"] = len(data["violations"])
    return row


def build_report() -> dict[str, Any]:
    intro = read_text(PAPER / "1_Introduction.tex")
    background = read_text(PAPER / "2_Background.tex")
    related = read_text(PAPER / "5_Related_Works.tex")
    discussion = read_text(PAPER / "10_Discussion_and_Limitations.tex")
    main_tex = read_text(PAPER / "main.tex")
    paper_text = "\n".join([intro, background, related, discussion, main_tex])

    direct_answer = first_line_with(
        background,
        ["not merely gocryptfs/fscrypt plus CUDA/TPM scripts"],
    )
    combined_capability = {
        term: term in background for term in COMBINED_CAPABILITY_TERMS
    }
    related_boundary = {
        term: term in related for term in RELATED_WORK_TERMS
    }
    baseline_classes = {
        cls: cls in intro for cls in BASELINE_CLASSES
    }
    evidence_gates = [gate_status(path) for path in EVIDENCE_GATES]

    forbidden_hits = []
    for pattern in FORBIDDEN_BROAD_CLAIMS:
        for match in re.finditer(pattern, paper_text, re.IGNORECASE):
            start = max(0, match.start() - 80)
            end = min(len(paper_text), match.end() + 80)
            context = " ".join(paper_text[start:end].split())
            if (
                "not a claim of deployed-filesystem superiority" in context
                or "rather than a claim of deployed-filesystem superiority" in context
            ):
                continue
            forbidden_hits.append({"pattern": pattern, "context": context})

    checks = {
        "direct_answer_present": direct_answer["present"],
        "combined_capability_terms_present": all(combined_capability.values()),
        "related_boundary_present": all(related_boundary.values()),
        "baseline_classes_present": all(baseline_classes.values()),
        "evidence_gates_pass": all(
            row["present"] and row["overall_pass"] is True and row.get("violations", 0) == 0
            for row in evidence_gates
        ),
        "no_broad_deployed_superiority_claim": not forbidden_hits,
        "paper_pages_le_13": (run_pdfinfo_pages(PAPER / "main.pdf") or 999) <= 13,
        "maturity_boundary_present": "mature deployed filesystems" in background,
        "full_kernel_matrix_boundary_present": (
            "not a full fscrypt, fs-verity, dm-integrity, OP-TEE, SPDK, or GPUDirect Storage comparison"
            in discussion
        ),
    }

    violations = [name for name, passed in checks.items() if not passed]

    return {
        "schema_version": 1,
        "scope": [
            "novelty isolation against deployed secure-storage baselines",
            "direct answer to gocryptfs/fscrypt plus CUDA/TPM script concern",
            "combined capability and non-superiority boundaries in paper text",
            "retained artifact gates that support the combined capability",
        ],
        "direct_answer": {"path": "Paper/2_Background.tex", **direct_answer},
        "combined_capability_terms": combined_capability,
        "related_boundary_terms": related_boundary,
        "baseline_classes": baseline_classes,
        "evidence_gates": evidence_gates,
        "forbidden_hits": forbidden_hits,
        "checks": checks,
        "pages": run_pdfinfo_pages(PAPER / "main.pdf"),
        "violations": violations,
        "overall_pass": not violations,
    }


def write_markdown(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# Novelty-isolation audit",
        "",
        f"- Overall pass: `{report['overall_pass']}`",
        f"- Paper pages: `{report['pages']}`",
        f"- Direct answer present: `{report['direct_answer']['present']}`",
        "",
        "## Direct Answer",
        "",
        f"- Source: `{report['direct_answer']['path']}:{report['direct_answer']['line']}`",
        f"- Text: {report['direct_answer']['text']}",
        "",
        "## Combined Capability",
        "",
        "| Term | Present |",
        "| --- | ---: |",
    ]
    for term, present in report["combined_capability_terms"].items():
        lines.append(f"| `{term}` | `{present}` |")

    lines += [
        "",
        "## Deployed-Baseline Classes",
        "",
        "| Class | Present in first-page table |",
        "| --- | ---: |",
    ]
    for cls, present in report["baseline_classes"].items():
        lines.append(f"| `{cls}` | `{present}` |")

    lines += [
        "",
        "## Evidence Gates",
        "",
        "| Gate | Present | Overall pass | Violations |",
        "| --- | ---: | ---: | ---: |",
    ]
    for row in report["evidence_gates"]:
        lines.append(
            f"| `{row['path']}` | `{row['present']}` | "
            f"`{row['overall_pass']}` | `{row.get('violations', 'n/a')}` |"
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
    json_path = args.out / "novelty_isolation.json"
    md_path = args.out / "novelty_isolation.md"
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
