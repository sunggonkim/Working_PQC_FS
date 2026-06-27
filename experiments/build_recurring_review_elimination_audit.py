#!/usr/bin/env python3
"""Audit paper locations that answer repeated review objections."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
PAPER = ROOT / "Paper"
DEFAULT_OUT = ROOT / "artifacts" / "reports" / "recurring_review_elimination"

THEMES = [
    {
        "theme": "no end-to-end benefit",
        "first_page_source": ("Paper/main.tex", ["foreground SQLite p99 rises", "13.8", "8.8"]),
        "first_page_pdf": ["SQLite p99 rises from 6.4", "13.8", "8.8"],
        "design": ("Paper/3_Design.tex", ["user.pqc\\_qos\\_class", "elastic files are throttle-eligible"]),
        "evaluation": ("Paper/4_Evaluation.tex", ["Table~\\ref{tab:qos_sqlite_recovery}", "13.8", "8.8"]),
        "limits": ("Paper/10_Discussion_and_Limitations.tex", ["sqlite-mounted-qos-recovery-2026-06-27", "not a separate anecdotal demo"]),
        "artifacts": [
            "artifacts/reports/hero_result_contract/hero_result_contract.json",
            "artifacts/validation/qos_sqlite_hero_bundle/qos_sqlite_hero_bundle.json",
        ],
    },
    {
        "theme": "disconnected GPU/KEM story",
        "first_page_source": ("Paper/main.tex", ["optional batched GPU maintenance lane", "rekey fallback"]),
        "first_page_pdf": ["optional batched GPU", "rekey fallback"],
        "design": ("Paper/3_Design.tex", ["optional batched ML-KEM", "key-plane workflow"]),
        "evaluation": ("Paper/4_Evaluation.tex", ["ML-KEM-768 key-plane workflow", "HMAC-authenticated envelopes"]),
        "limits": ("Paper/10_Discussion_and_Limitations.tex", ["GPU helps independent ML-KEM batches", "does not justify foreground AI-QoS recovery"]),
        "artifacts": [
            "artifacts/validation/keyplane_rekey_workflow/keyplane_rekey_workflow.json",
            "artifacts/validation/keyplane_rekey_methodology/keyplane_rekey_workflow.json",
            "artifacts/reports/design_eval_isomorphism/design_eval_isomorphism.json",
        ],
    },
    {
        "theme": "incomplete baselines",
        "first_page_source": ("Paper/1_Introduction.tex", ["gocryptfs", "fscrypt", "dm-crypt"]),
        "first_page_pdf": ["gocryptfs", "fscrypt", "dm-crypt"],
        "design": ("Paper/2_Background.tex", ["protection boundary", "not a replacement for fscrypt or dm-crypt"]),
        "evaluation": ("Paper/4_Evaluation.tex", ["Frozen contract", "fscrypt/dm-crypt and cold rows"]),
        "limits": ("Paper/10_Discussion_and_Limitations.tex", ["not yet a complete fscrypt, dm-crypt", "traceability"]),
        "artifacts": [
            "artifacts/reports/novelty_isolation/novelty_isolation.json",
            "artifacts/validation/kernel_baseline_feasibility/kernel_baseline_feasibility.json",
            "artifacts/validation/frozen_plaintext_contract/frozen_plaintext_contract.json",
            "artifacts/validation/frozen_gocryptfs_contract/frozen_gocryptfs_contract.json",
            "artifacts/validation/frozen_aegisq_contract/frozen_aegisq_contract.json",
        ],
    },
    {
        "theme": "wiring-only QoS",
        "first_page_source": ("Paper/1_Introduction.tex", ["Figure~\\ref{fig:first_page_qos}", "8.8", "2.7"]),
        "first_page_pdf": ["Figure 1", "8.8", "2.7 MB/s"],
        "design": ("Paper/3_Design.tex", ["QoS", "mounted-FUSE throttling"]),
        "evaluation": ("Paper/4_Evaluation.tex", ["This is SQLite recovery", "not general AI-inference QoS"]),
        "limits": ("Paper/10_Discussion_and_Limitations.tex", ["TensorRT p99 recovery", "not a separate anecdotal demo"]),
        "artifacts": [
            "artifacts/validation/qos_sqlite_hero_bundle/qos_sqlite_hero_bundle.json",
            "artifacts/validation/qos_sensitivity_analysis/qos_sensitivity_analysis.json",
            "artifacts/reports/case_study_takeaway/case_study_takeaway.json",
        ],
    },
    {
        "theme": "partial TPM/freshness",
        "first_page_source": ("Paper/main.tex", ["TPM replay-after-advance", "fail-closed external-anchor"]),
        "first_page_pdf": ["fail-closed external", "anchor TPM replay-after-advance"],
        "design": ("Paper/3_Design.tex", ["pre-provisioned TPM NV index", "fail-closed replay-after-advance"]),
        "evaluation": ("Paper/4_Evaluation.tex", ["hardware-freshness recovery matrix", "replay-after-advance"]),
        "limits": ("Paper/10_Discussion_and_Limitations.tex", ["not persistent PCR-bound key release", "provisioned TPM policy"]),
        "artifacts": [
            "artifacts/validation/hardware_freshness_recovery_matrix/hardware_freshness_recovery_matrix.json",
            "artifacts/validation/tpm_freshness_policy/tpm_freshness_policy.json",
            "artifacts/validation/pcr_anchor_decision/pcr_anchor_decision.json",
        ],
    },
    {
        "theme": "narrow POSIX semantics",
        "first_page_source": ("Paper/1_Introduction.tex", ["FUSE prototype with scoped evidence"]),
        "first_page_pdf": ["FUSE", "prototype", "scoped evidence"],
        "design": ("Paper/3_Design.tex", ["Unsupported POSIX modes", "ordinary read/write/\\texttt{fsync}"]),
        "evaluation": ("Paper/4_Evaluation.tex", ["POSIX-scope audit", "rename and directory \\texttt{fsync}"]),
        "limits": ("Paper/10_Discussion_and_Limitations.tex", ["POSIX envelope is also narrow", "shared \\texttt{mmap}"]),
        "artifacts": [
            "artifacts/validation/posix_scope_audit/posix_scope_audit.json",
            "artifacts/validation/generation_fault_matrix/generation_fault_matrix.json",
        ],
    },
    {
        "theme": "password-derived credential boundary",
        "first_page_source": ("Paper/main.tex", ["password-derived mount-key boundary"]),
        "first_page_pdf": ["password", "mount-key"],
        "design": ("Paper/3_Design.tex", ["mount key is password-derived", "never hardware-released"]),
        "evaluation": ("Paper/4_Evaluation.tex", ["HMAC-authenticated envelopes", "EKEYREJECTED"]),
        "limits": ("Paper/10_Discussion_and_Limitations.tex", ["mount password remains the root credential", "no hardware-backed credential release"]),
        "artifacts": [
            "artifacts/validation/mount_key_lifecycle/mount_key_lifecycle.json",
            "artifacts/validation/fuse_tamper_rejection.json",
            "artifacts/validation/keyplane_rekey_workflow/keyplane_rekey_workflow.json",
        ],
    },
    {
        "theme": "single-platform evidence",
        "first_page_source": ("Paper/main.tex", ["Jetson AGX Thor Developer Kit"]),
        "first_page_pdf": ["Jetson AGX", "Thor Developer Kit"],
        "design": ("Paper/7_Implementation_Details.tex", ["CUDA~13.0.48", "cuPQC"]),
        "evaluation": ("Paper/4_Evaluation.tex", ["NVIDIA Jetson AGX Thor Developer Kit", "must not be generalized"]),
        "limits": ("Paper/10_Discussion_and_Limitations.tex", ["non-NVIDIA UMA platform", "CUDA-independent deployment"]),
        "artifacts": [
            "artifacts/validation/stat_thermal_methodology/stat_thermal_methodology_audit.json",
            "artifacts/validation/qos_sqlite_hero_bundle/qos_sqlite_hero_bundle.json",
        ],
    },
    {
        "theme": "microbenchmark-only methodology",
        "first_page_source": ("Paper/main.tex", ["retained mounted artifacts show the intended spine"]),
        "first_page_pdf": ["retained mounted artifacts", "intended", "spine"],
        "design": ("Paper/3_Design.tex", ["accelerator placement must be subordinate to storage correctness"]),
        "evaluation": ("Paper/4_Evaluation.tex", ["not confidence intervals", "single retained workflow artifacts"]),
        "limits": ("Paper/10_Discussion_and_Limitations.tex", ["three-run primitive measurements cannot establish"]),
        "artifacts": [
            "artifacts/reports/hero_result_contract/hero_result_contract.json",
            "artifacts/validation/stat_thermal_methodology/stat_thermal_methodology_audit.json",
            "artifacts/reports/evaluation_rq_audit/evaluation_rq_audit.json",
        ],
    },
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


def run_pdftotext_first_page(path: Path) -> str:
    proc = subprocess.run(
        ["pdftotext", "-f", "1", "-l", "1", "-layout", str(path), "-"],
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
    text = read_text(path)
    for line_no, line in enumerate(text.splitlines(), start=1):
        if contains_all(line, needles):
            return {
                "path": path_str,
                "line": line_no,
                "text": " ".join(line.strip().split()),
                "present": True,
            }
    return {"path": path_str, "line": None, "text": None, "present": False}


def artifact_status(path_str: str) -> dict[str, Any]:
    path = ROOT / path_str
    row: dict[str, Any] = {"path": path_str, "present": path.exists(), "overall_pass": None}
    if path.exists() and path.suffix == ".json":
        data = read_json(path)
        row["overall_pass"] = data.get("overall_pass")
        if "violations" in data:
            row["violations"] = len(data["violations"])
        if "summary" in data and "violations" in data["summary"]:
            row["summary_violations"] = data["summary"]["violations"]
    return row


def build_report() -> dict[str, Any]:
    first_page_text = run_pdftotext_first_page(PAPER / "main.pdf")
    theme_rows = []
    for theme in THEMES:
        first_source = find_line(*theme["first_page_source"])
        design = find_line(*theme["design"])
        evaluation = find_line(*theme["evaluation"])
        limits = find_line(*theme["limits"])
        artifacts = [artifact_status(path) for path in theme["artifacts"]]
        first_page_pdf_present = contains_all(first_page_text, theme["first_page_pdf"])
        artifacts_ok = all(
            row["present"]
            and (row["overall_pass"] is True or row["overall_pass"] is None)
            and row.get("violations", 0) == 0
            and row.get("summary_violations", 0) == 0
            for row in artifacts
        )
        passes = (
            first_source["present"]
            and first_page_pdf_present
            and design["present"]
            and evaluation["present"]
            and limits["present"]
            and artifacts_ok
        )
        theme_rows.append({
            "theme": theme["theme"],
            "first_page_source": first_source,
            "first_page_pdf_needles": theme["first_page_pdf"],
            "first_page_pdf_present": first_page_pdf_present,
            "design": design,
            "evaluation": evaluation,
            "limits": limits,
            "artifacts": artifacts,
            "artifacts_ok": artifacts_ok,
            "passes": passes,
        })

    violations: list[str] = []
    for row in theme_rows:
        if not row["passes"]:
            violations.append(row["theme"])
    if run_pdfinfo_pages(PAPER / "main.pdf") != 12:
        violations.append("Paper/main.pdf is not 12 pages")

    return {
        "schema_version": 1,
        "scope": [
            "nine recurring review themes from SUBMISSION_CHECKLIST.md",
            "first-page source and compiled first-page PDF anchors",
            "design, evaluation, limitation/applicability paper locations",
            "retained artifact gates for each theme",
        ],
        "themes": theme_rows,
        "pages": run_pdfinfo_pages(PAPER / "main.pdf"),
        "violations": violations,
        "overall_pass": not violations,
    }


def write_markdown(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# Recurring-review elimination audit",
        "",
        f"- Overall pass: `{report['overall_pass']}`",
        f"- Paper pages: `{report['pages']}`",
        f"- Themes checked: `{len(report['themes'])}`",
        "",
        "| Theme | First page | Design | Evaluation | Limits | Artifacts | Pass |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in report["themes"]:
        lines.append(
            f"| {row['theme']} | `{row['first_page_pdf_present'] and row['first_page_source']['present']}` | "
            f"`{row['design']['present']}` | `{row['evaluation']['present']}` | "
            f"`{row['limits']['present']}` | `{row['artifacts_ok']}` | `{row['passes']}` |"
        )

    lines += ["", "## Theme Details", ""]
    for row in report["themes"]:
        lines += [
            f"### {row['theme']}",
            "",
            f"- First page: `{row['first_page_source']['path']}:{row['first_page_source']['line']}`",
            f"- Design: `{row['design']['path']}:{row['design']['line']}`",
            f"- Evaluation: `{row['evaluation']['path']}:{row['evaluation']['line']}`",
            f"- Limits: `{row['limits']['path']}:{row['limits']['line']}`",
            f"- Artifacts: {', '.join('`' + a['path'] + '`' for a in row['artifacts'])}",
            "",
        ]

    if report["violations"]:
        lines += ["## Violations", ""]
        lines += [f"- {v}" for v in report["violations"]]

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    report = build_report()
    json_path = args.out / "recurring_review_elimination.json"
    md_path = args.out / "recurring_review_elimination.md"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown(report, md_path)
    print(json.dumps({
        "out_dir": relpath(args.out),
        "overall_pass": report["overall_pass"],
        "pages": report["pages"],
        "themes": len(report["themes"]),
        "violations": len(report["violations"]),
    }, indent=2, sort_keys=True))
    return 0 if report["overall_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
