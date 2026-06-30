#!/usr/bin/env python3
"""Audit one-to-one mapping from design mechanisms to evaluation closures."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
PAPER = ROOT / "Paper"
DEFAULT_OUT = ROOT / "artifacts" / "reports" / "design_eval_isomorphism"

MECHANISMS = [
    {
        "mechanism": "authenticated block format",
        "closure_id": "E1-generation-and-envelope-authentication",
        "design_needles": ["AES-256-GCM block format", "file identifier", "Table~\\ref{tab:design_goals}"],
        "evaluation_needles": ["generation fault matrix", "EKEYREJECTED"],
        "rq": "RQ1",
        "artifact_paths": [
            "artifacts/validation/generation_fault_matrix/generation_fault_matrix.json",
            "artifacts/validation/fuse_tamper_rejection.json",
        ],
    },
    {
        "mechanism": "D/J/C publication",
        "closure_id": "E2-oracle-labeled-publication-cutpoints",
        "design_needles": ["Figure~\\ref{fig:djc_state_machine}", "Data-before-mapping publication"],
        "evaluation_needles": ["daemon campaign kills", "D/J/C", "previous committed payload", "latest committed payload"],
        "rq": "RQ4",
        "artifact_paths": [
            "artifacts/validation/daemon_power_fault_campaign/daemon_power_fault_campaign.json",
            "artifacts/validation/recovery_oracle_audit/recovery_oracle_audit.json",
        ],
    },
    {
        "mechanism": "TPM replay check",
        "closure_id": "E3-hardware-freshness-verdict-matrix",
        "design_needles": ["TPM NV index", "Replay boundary"],
        "evaluation_needles": ["hardware replay matrix", "replay-after-advance"],
        "rq": "RQ4",
        "artifact_paths": [
            "artifacts/validation/hardware_freshness_recovery_matrix/hardware_freshness_recovery_matrix.json",
            "artifacts/validation/tpm_monotonic_replay/tpm_monotonic_replay.json",
        ],
    },
    {
        "mechanism": "QoS controller",
        "closure_id": "E4-sqlite-recovery-and-controller-sensitivity",
        "design_needles": ["user.pqc\\_qos\\_class", "elastic files are throttle-eligible"],
        "evaluation_needles": ["SQLite transaction latency", "sensitivity bundle", "hysteresis records"],
        "rq": "RQ3/RQ5",
        "artifact_paths": [
            "artifacts/validation/qos_sqlite_hero_bundle/qos_sqlite_hero_bundle.json",
            "artifacts/validation/qos_sensitivity_analysis/qos_sensitivity_analysis.json",
        ],
    },
    {
        "mechanism": "CPU/GPU placement",
        "closure_id": "E5-data-plane-placement-asymmetry",
        "design_needles": ["placement function balances", "managed allocation", "stream synchronization"],
        "evaluation_needles": ["CPU/OpenSSL", "managed-buffer GPU path", "data plane is CPU-first"],
        "rq": "RQ2/RQ5",
        "artifact_paths": [
            "artifacts/validation/microbench/summary.json",
            "artifacts/validation/frozen_aegisq_contract/frozen_aegisq_contract.json",
        ],
    },
    {
        "mechanism": "optional ML-KEM batch lane",
        "closure_id": "E6-mounted-open-file-rekey-workflow",
        "design_needles": ["ML-KEM envelope refresh", "key-plane workflow", "producer supplies nonzero relative slack"],
        "evaluation_needles": ["ML-KEM-768", "1.186$\\times$ median speedup", "break-even model", "open-file envelope refresh"],
        "rq": "RQ5",
        "artifact_paths": [
            "artifacts/validation/keyplane_rekey_workflow/keyplane_rekey_workflow.json",
            "artifacts/validation/keyplane_rekey_methodology/keyplane_rekey_workflow.json",
            "artifacts/reports/x11_mlkem_break_even_model/x11_mlkem_break_even_model.json",
        ],
    },
]

FIGURE_TABLE_OBLIGATIONS = [
    {"label": "fig:first_page_qos", "source": "Paper/1_Introduction.tex", "obligation": "first-page QoS pressure/hero result"},
    {"label": "tab:capability_matrix", "source": "Paper/1_Introduction.tex", "obligation": "design-gap capability comparison"},
    {"label": "tab:design_goals", "source": "Paper/3_Design.tex", "obligation": "formal invariant table"},
    {"label": "fig:overall_procedure", "source": "Paper/3_Design.tex", "obligation": "architecture and plane separation"},
    {"label": "fig:djc_state_machine", "source": "Paper/3_Design.tex", "obligation": "publication protocol state machine"},
    {"label": "tab:memory_compat", "source": "Paper/3_Design.tex", "obligation": "memory/claim boundary"},
    {"label": "tab:impl_boundaries", "source": "Paper/7_Implementation_Details.tex", "obligation": "implementation boundary summary"},
    {"label": "tab:threat_boundary", "source": "Paper/8_Security_Analysis.tex", "obligation": "security threat boundary"},
    {"label": "tab:benchmark_workloads", "source": "Paper/4_Evaluation.tex", "obligation": "evaluation provenance"},
    {"label": "fig:evaluation_summary", "source": "Paper/4_Evaluation.tex", "obligation": "evaluation spine"},
    {"label": "tab:qos_sqlite_recovery", "source": "Paper/generated_qos_recovery_table.tex", "obligation": "SQLite QoS hero result"},
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


def all_present(text: str, needles: list[str]) -> bool:
    return all(needle in text for needle in needles)


def find_label(path: Path, label: str) -> bool:
    return f"\\label{{{label}}}" in read_text(path)


def label_referenced(paper_text: str, label: str) -> bool:
    return bool(re.search(r"\\(?:ref|pageref|autoref)\{" + re.escape(label) + r"\}", paper_text))


def build_report() -> dict[str, Any]:
    design_text = read_text(PAPER / "3_Design.tex")
    eval_text = read_text(PAPER / "4_Evaluation.tex")
    full_paper_text = "\n".join(read_text(path) for path in PAPER.glob("*.tex"))

    mechanism_rows = []
    for mechanism in MECHANISMS:
        artifact_rows = [
            {"path": path, "present": (ROOT / path).exists()}
            for path in mechanism["artifact_paths"]
        ]
        mechanism_rows.append({
            "mechanism": mechanism["mechanism"],
            "closure_id": mechanism["closure_id"],
            "rq": mechanism["rq"],
            "design_needles": mechanism["design_needles"],
            "evaluation_needles": mechanism["evaluation_needles"],
            "design_present": all_present(design_text, mechanism["design_needles"]),
            "evaluation_present": all_present(eval_text, mechanism["evaluation_needles"]),
            "artifacts": artifact_rows,
            "artifacts_present": all(row["present"] for row in artifact_rows),
        })

    closure_ids = [row["closure_id"] for row in mechanism_rows]
    duplicate_closures = sorted({cid for cid in closure_ids if closure_ids.count(cid) > 1})

    figure_table_rows = []
    for item in FIGURE_TABLE_OBLIGATIONS:
        source = ROOT / item["source"]
        figure_table_rows.append({
            **item,
            "label_present": find_label(source, item["label"]),
            "referenced": label_referenced(full_paper_text, item["label"]),
        })

    mapping_text_present = (
        "The design/evaluation contract is one closure per main mechanism" in eval_text
        and "the mounted rekey result comes from the key-plane workflow harness" in eval_text
    )

    violations: list[str] = []
    for row in mechanism_rows:
        if not row["design_present"]:
            violations.append(f"missing design source for {row['mechanism']}")
        if not row["evaluation_present"]:
            violations.append(f"missing evaluation closure for {row['mechanism']}")
        if not row["artifacts_present"]:
            violations.append(f"missing artifact path for {row['mechanism']}")
    if duplicate_closures:
        violations.append(f"duplicate closure ids: {', '.join(duplicate_closures)}")
    if len(set(closure_ids)) != len(MECHANISMS):
        violations.append("not every mechanism has exactly one unique closure id")
    for row in figure_table_rows:
        if not row["label_present"]:
            violations.append(f"missing figure/table label {row['label']}")
        if not row["referenced"]:
            violations.append(f"figure/table label {row['label']} is not referenced")
    if not mapping_text_present:
        violations.append("paper lacks explicit design/evaluation contract paragraph")
    if run_pdfinfo_pages(PAPER / "main.pdf") != 12:
        violations.append("Paper/main.pdf is not 12 pages")

    return {
        "schema_version": 1,
        "scope": [
            "six required design mechanisms from SUBMISSION_CHECKLIST.md",
            "unique evaluation closure id per mechanism",
            "main-paper figure/table obligation labels",
            "paper source mapping paragraph and 12-page PDF gate",
        ],
        "mechanisms": mechanism_rows,
        "figure_table_obligations": figure_table_rows,
        "mapping_text_present": mapping_text_present,
        "duplicate_closures": duplicate_closures,
        "pages": run_pdfinfo_pages(PAPER / "main.pdf"),
        "violations": violations,
        "overall_pass": not violations,
    }


def write_markdown(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# Design-evaluation isomorphism audit",
        "",
        f"- Overall pass: `{report['overall_pass']}`",
        f"- Paper pages: `{report['pages']}`",
        f"- Mapping text present: `{report['mapping_text_present']}`",
        "",
        "## Mechanism closures",
        "",
        "| Mechanism | Closure id | RQ | Design present | Evaluation present | Artifacts present |",
        "| --- | --- | --- | ---: | ---: | ---: |",
    ]
    for row in report["mechanisms"]:
        lines.append(
            f"| {row['mechanism']} | `{row['closure_id']}` | {row['rq']} | "
            f"`{row['design_present']}` | `{row['evaluation_present']}` | "
            f"`{row['artifacts_present']}` |"
        )

    lines += [
        "",
        "## Figure/table obligations",
        "",
        "| Label | Obligation | Label present | Referenced |",
        "| --- | --- | ---: | ---: |",
    ]
    for row in report["figure_table_obligations"]:
        lines.append(
            f"| `{row['label']}` | {row['obligation']} | "
            f"`{row['label_present']}` | `{row['referenced']}` |"
        )

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
    json_path = args.out / "design_eval_isomorphism.json"
    md_path = args.out / "design_eval_isomorphism.md"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown(report, md_path)
    print(json.dumps({
        "out_dir": relpath(args.out),
        "overall_pass": report["overall_pass"],
        "mechanisms": len(report["mechanisms"]),
        "pages": report["pages"],
        "violations": len(report["violations"]),
    }, indent=2, sort_keys=True))
    return 0 if report["overall_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
