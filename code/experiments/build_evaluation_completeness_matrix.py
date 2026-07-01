#!/usr/bin/env python3
"""Build the evaluation completeness matrix for the submission paper."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
PAPER = ROOT / "Paper"
DEFAULT_OUT = ROOT / "artifacts" / "reports" / "evaluation_completeness_matrix"

FIGURE_TABLE_AUDIT = (
    "artifacts/reports/figure_table_obligations/figure_table_obligations.json"
)

REQUIRED_ROW_IDS = {
    "baseline_sota_comparison",
    "scalability_pressure_behavior",
    "workload_diversity",
    "time_overhead_breakdown",
    "sensitivity_analysis",
    "stability_variance",
    "case_study",
}

ALLOWED_REQUIRED_STATUSES = {
    "implemented",
    "implemented_scoped",
    "scoped_partial",
}

MATRIX_ROWS: list[dict[str, Any]] = [
    {
        "id": "baseline_sota_comparison",
        "required": True,
        "requirement": "baseline/SOTA comparison",
        "status": "scoped_partial",
        "figure_table_labels": ["tab:capability_matrix"],
        "evidence": [
            {
                "path": "artifacts/validation/frozen_plaintext_contract/frozen_plaintext_contract.json",
                "expected_overall_pass": True,
            },
            {
                "path": "artifacts/validation/frozen_gocryptfs_contract/frozen_gocryptfs_contract.json",
                "expected_overall_pass": True,
            },
            {
                "path": "artifacts/validation/frozen_dmcrypt_contract/frozen_dmcrypt_contract.json",
                "expected_overall_pass": True,
            },
            {
                "path": "artifacts/validation/frozen_aegisq_contract/frozen_aegisq_contract.json",
                "expected_overall_pass": True,
            },
            {
                "path": "artifacts/validation/kernel_baseline_feasibility/kernel_baseline_feasibility.json",
                "expected_overall_pass": True,
            },
            {
                "path": "artifacts/validation/kernel_baseline_feasibility/paper_fscrypt_verdict.json",
                "expected_overall_pass": True,
            },
            {
                "path": "artifacts/reports/novelty_isolation/novelty_isolation.json",
                "expected_overall_pass": True,
            },
        ],
        "paper_anchors": [
            {
                "path": "Paper/1_Introduction.tex",
                "terms": [
                    "Table~\\ref{tab:capability_matrix} positions that gap against plaintext, gocryptfs, fscrypt, dm-crypt",
                    "storage-visible policy at the edge encryption boundary",
                ],
            },
            {
                "path": "Paper/10_Discussion_and_Limitations.tex",
                "terms": [
                    "The frozen matrix covers plaintext, gocryptfs, dm-crypt, and AEGIS-Q",
                    "not a full fscrypt, fs-verity, dm-integrity",
                ],
            },
        ],
        "scoped_out": [
            "fscrypt/fs-verity/dm-integrity rows are not current matched throughput measurements",
            "cold-cache rows remain invalid without privileged cache-drop control",
        ],
    },
    {
        "id": "scalability_pressure_behavior",
        "required": True,
        "requirement": "scalability or pressure behavior",
        "status": "implemented_scoped",
        "figure_table_labels": ["fig:first_page_qos", "fig:evaluation_summary", "fig:recovery_qos_detail"],
        "evidence": [
            {
                "path": "artifacts/reports/hero_result_contract/hero_result_contract.json",
                "expected_overall_pass": True,
            },
            {
                "path": "artifacts/validation/qos_sqlite_hero_bundle/qos_sqlite_hero_bundle.json",
                "expected_overall_pass": True,
            },
        ],
        "paper_anchors": [
            {
                "path": "Paper/4_Evaluation.tex",
                "terms": [
                    "AEGIS-Q reports 8.15~ms p99",
                    "unthrottled storage reports 9.62~ms",
                    "bounded storage-visible control, not a uniqueness or non-storage QoS claim",
                ],
            },
        ],
        "scoped_out": [
            "pressure result is mounted SQLite storage pressure, not non-storage application p99 recovery",
        ],
    },
    {
        "id": "workload_diversity",
        "required": True,
        "requirement": "workload diversity",
        "status": "implemented_scoped",
        "figure_table_labels": ["tab:benchmark_workloads"],
        "evidence": [
            {
                "path": "artifacts/validation/workload_diversity_matrix/workload_diversity_matrix.json",
                "expected_overall_pass": True,
            },
        ],
        "paper_anchors": [
            {
                "path": "Paper/4_Evaluation.tex",
                "terms": [
                    "Table~\\ref{tab:benchmark_workloads}",
                    "SQLite",
                    "\\texttt{dbm.dumb}",
                    "append-log/cache-manifest",
                ],
            },
            {
                "path": "Paper/10_Discussion_and_Limitations.tex",
                "terms": [
                    "SQLite, elastic background writes, append-log/cache-manifest remounts",
                    "not a general POSIX",
                ],
            },
        ],
        "scoped_out": [
            "workloads do not establish broad workload generalization or full crash certification",
        ],
    },
    {
        "id": "time_overhead_breakdown",
        "required": True,
        "requirement": "time/overhead breakdown",
        "status": "implemented_scoped",
        "figure_table_labels": ["fig:dataplane_negative_control", "fig:evaluation_summary"],
        "evidence": [
            {
                "path": "artifacts/validation/mechanism_ablation_manifest/mechanism_ablation_manifest.json",
                "expected_overall_pass": True,
            },
            {
                "path": "artifacts/validation/keyplane_rekey_workflow/keyplane_rekey_workflow.json",
                "expected_overall_pass": True,
            },
            {
                "path": "artifacts/reports/x11_mlkem_break_even_model/x11_mlkem_break_even_model.json",
                "expected_overall_pass": True,
            },
        ],
        "paper_anchors": [
            {
                "path": "Paper/4_Evaluation.tex",
                "terms": [
                    "CPU/OpenSSL reaches 1.61~GB/s",
                    "cuPQC reaches 1.50~M operations/s",
                    "1.186$\\times$ median speedup",
                    "break-even model",
                    "The ablation manifest then attributes cost",
                ],
            },
        ],
        "scoped_out": [
            "primitive placement is not reported as an end-to-end FUSE write speedup",
        ],
    },
    {
        "id": "sensitivity_analysis",
        "required": True,
        "requirement": "sensitivity analysis",
        "status": "implemented_scoped",
        "figure_table_labels": [],
        "evidence": [
            {
                "path": "artifacts/validation/qos_sensitivity_analysis/qos_sensitivity_analysis.json",
                "expected_overall_pass": True,
            },
        ],
        "paper_anchors": [
            {
                "path": "Paper/4_Evaluation.tex",
                "terms": [
                    "The sensitivity bundle varies budget",
                    "sampling interval",
                    "queue depth",
                    "writer intensity",
                    "hysteresis",
                ],
            },
        ],
        "scoped_out": [
            "sensitivity is controller-parameter coverage, not a statistical confidence study",
        ],
    },
    {
        "id": "stability_variance",
        "required": True,
        "requirement": "stability/variance",
        "status": "scoped_partial",
        "figure_table_labels": [],
        "evidence": [
            {
                "path": "artifacts/validation/stat_thermal_methodology/stat_thermal_methodology_audit.json",
                "expected_overall_pass": True,
            },
            {
                "path": "artifacts/validation/qos_sqlite_hero_methodology/qos_sqlite_hero_bundle.json",
                "expected_overall_pass": False,
                "role": "negative repeated-run stability artifact",
            },
        ],
        "paper_anchors": [
            {
                "path": "Paper/4_Evaluation.tex",
                "terms": [
                    "same-run power/thermal observations",
                    "five repetitions",
                    "five-run methodology row",
                ],
            },
        ],
        "scoped_out": [
            "SQLite QoS headline remains a single retained workflow artifact",
            "future headline comparisons require the methodology gate before generalization",
        ],
    },
    {
        "id": "case_study",
        "required": True,
        "requirement": "case study",
        "status": "implemented_scoped",
        "figure_table_labels": [],
        "evidence": [
            {
                "path": "artifacts/reports/case_study_takeaway/case_study_takeaway.json",
                "expected_overall_pass": True,
            },
            {
                "path": "artifacts/reports/hero_result_contract/hero_result_contract.json",
                "expected_overall_pass": True,
            },
        ],
        "paper_anchors": [
            {
                "path": "Paper/4_Evaluation.tex",
                "terms": [
                    "The app-level result uses SQLite transaction latency",
                    "secure append-log macro",
                    "cache-manifest workload",
                ],
            },
        ],
        "scoped_out": [
            "deployment takeaway is the SQLite/FUSE hero contract, not a separate anecdote",
        ],
    },
    {
        "id": "protocol_correctness_and_security_obligations",
        "required": False,
        "requirement": "non-evaluation figure/table protocol obligations",
        "status": "implemented_scoped",
        "figure_table_labels": [
            "fig:problem_boundary",
            "tab:design_goals",
            "fig:overall_procedure",
            "fig:djc_state_machine",
            "tab:memory_compat",
            "tab:impl_boundaries",
            "tab:threat_boundary",
        ],
        "evidence": [
            {
                "path": "artifacts/reports/design_eval_isomorphism/design_eval_isomorphism.json",
                "expected_overall_pass": True,
            },
            {
                "path": "artifacts/validation/recovery_oracle_audit/recovery_oracle_audit.json",
                "expected_overall_pass": True,
            },
            {
                "path": "artifacts/validation/hardware_freshness_recovery_matrix/hardware_freshness_recovery_matrix.json",
                "expected_overall_pass": True,
            },
        ],
        "paper_anchors": [
            {
                "path": "Paper/3_Design.tex",
                "terms": [
                    "Figure~\\ref{fig:overall_procedure}",
                    "Table~\\ref{tab:design_goals}",
                    "Figure~\\ref{fig:djc_state_machine}",
                    "Table~\\ref{tab:memory_compat}",
                ],
            },
            {
                "path": "Paper/7_Implementation_Details.tex",
                "terms": ["Table~\\ref{tab:impl_boundaries}"],
            },
            {
                "path": "Paper/8_Security_Analysis.tex",
                "terms": [
                    "Table~\\ref{tab:threat_boundary}",
                    "not a formal proof of crash consistency",
                ],
            },
        ],
        "scoped_out": [
            "protocol rows close design/security obligations rather than adding evaluation completeness rows",
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


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def normalize(text: str) -> str:
    return " ".join(text.split())


def text_has_terms(text: str, terms: list[str]) -> tuple[bool, list[str]]:
    normalized = normalize(text).lower()
    missing = [term for term in terms if normalize(term).lower() not in normalized]
    return not missing, missing


def run_pdfinfo_pages(path: Path) -> int | None:
    proc = subprocess.run(
        ["pdfinfo", str(path)],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    for line in proc.stdout.splitlines():
        if line.startswith("Pages:"):
            return int(line.split(":", 1)[1].strip())
    return None


def check_artifact(spec: dict[str, Any]) -> dict[str, Any]:
    path = ROOT / spec["path"]
    row: dict[str, Any] = {
        "path": spec["path"],
        "exists": path.exists(),
        "expected_overall_pass": spec.get("expected_overall_pass"),
        "role": spec.get("role", "retained evidence"),
        "overall_pass": None,
        "passes": False,
    }
    if not path.exists():
        return row

    if path.suffix == ".json":
        try:
            data = load_json(path)
        except json.JSONDecodeError as exc:
            row["json_error"] = str(exc)
            return row
        row["overall_pass"] = data.get("overall_pass")

    expected = spec.get("expected_overall_pass")
    if expected is None:
        row["passes"] = True
    else:
        row["passes"] = row["overall_pass"] is expected
    return row


def check_paper_anchor(spec: dict[str, Any]) -> dict[str, Any]:
    path = ROOT / spec["path"]
    row: dict[str, Any] = {
        "path": spec["path"],
        "terms": spec["terms"],
        "exists": path.exists(),
        "missing_terms": spec["terms"],
        "passes": False,
    }
    if not path.exists():
        return row
    present, missing = text_has_terms(read_text(path), spec["terms"])
    row["missing_terms"] = missing
    row["passes"] = present
    return row


def load_figure_table_labels() -> tuple[set[str], dict[str, Any], list[str]]:
    audit_path = ROOT / FIGURE_TABLE_AUDIT
    violations: list[str] = []
    if not audit_path.exists():
        return set(), {}, [f"missing figure/table audit: {FIGURE_TABLE_AUDIT}"]
    audit = load_json(audit_path)
    if audit.get("overall_pass") is not True:
        violations.append("figure/table obligation audit is not passing")
    labels = set(audit.get("found_labels") or [])
    if not labels:
        labels = {
            row["label"]
            for row in audit.get("figure_tables", [])
            if row.get("label")
        }
    return labels, audit, violations


def evaluate_matrix_rows() -> list[dict[str, Any]]:
    evaluated = []
    for row in MATRIX_ROWS:
        evidence = [check_artifact(spec) for spec in row["evidence"]]
        paper_anchors = [check_paper_anchor(spec) for spec in row["paper_anchors"]]
        evaluated.append({
            **row,
            "evidence": evidence,
            "paper_anchors": paper_anchors,
            "evidence_pass": all(item["passes"] for item in evidence),
            "paper_anchor_pass": all(item["passes"] for item in paper_anchors),
        })
    return evaluated


def build_report() -> dict[str, Any]:
    figure_labels, figure_audit, violations = load_figure_table_labels()
    rows = evaluate_matrix_rows()

    required_rows = {row["id"] for row in rows if row["required"]}
    missing_required = sorted(REQUIRED_ROW_IDS - required_rows)
    extra_required = sorted(required_rows - REQUIRED_ROW_IDS)
    if missing_required:
        violations.append(f"missing required matrix rows: {', '.join(missing_required)}")
    if extra_required:
        violations.append(f"unexpected required matrix rows: {', '.join(extra_required)}")

    for row in rows:
        if row["required"] and row["status"] not in ALLOWED_REQUIRED_STATUSES:
            violations.append(f"{row['id']} has unsupported status {row['status']}")
        if row["required"] and row["status"] == "scoped_partial" and not row["scoped_out"]:
            violations.append(f"{row['id']} is scoped_partial without scoped_out text")
        if not row["evidence_pass"]:
            violations.append(f"{row['id']} does not have passing retained evidence")
        if not row["paper_anchor_pass"]:
            violations.append(f"{row['id']} is missing paper scope anchors")

    label_to_rows: dict[str, list[str]] = {}
    for row in rows:
        for label in row["figure_table_labels"]:
            label_to_rows.setdefault(label, []).append(row["id"])

    mapped_labels = set(label_to_rows)
    missing_label_mappings = sorted(figure_labels - mapped_labels)
    extra_label_mappings = sorted(mapped_labels - figure_labels)
    duplicate_label_mappings = {
        label: owners
        for label, owners in sorted(label_to_rows.items())
        if len(owners) > 1
    }

    if missing_label_mappings:
        violations.append(
            "figure/table labels missing from completeness matrix: "
            + ", ".join(missing_label_mappings)
        )
    if extra_label_mappings:
        violations.append(
            "completeness matrix maps unknown labels: " + ", ".join(extra_label_mappings)
        )
    if duplicate_label_mappings:
        violations.append("figure/table labels have multiple matrix owners")

    pages = run_pdfinfo_pages(PAPER / "main.pdf")
    if pages is None or pages > 13:
        violations.append("Paper/main.pdf exceeds 13 pages")

    checks = {
        "figure_table_audit_pass": figure_audit.get("overall_pass") is True,
        "all_required_rows_present": not missing_required and not extra_required,
        "required_rows_have_allowed_status": all(
            not row["required"] or row["status"] in ALLOWED_REQUIRED_STATUSES
            for row in rows
        ),
        "all_rows_have_retained_evidence": all(row["evidence_pass"] for row in rows),
        "all_rows_have_paper_scope_anchors": all(row["paper_anchor_pass"] for row in rows),
        "all_figure_table_labels_mapped_once": (
            not missing_label_mappings
            and not extra_label_mappings
            and not duplicate_label_mappings
        ),
        "paper_pages_le_13": pages is not None and pages <= 13,
    }

    return {
        "schema_version": 1,
        "scope": [
            "seven required evaluation completeness rows",
            "one supplemental protocol-obligation row for non-evaluation figures/tables",
            "primary ownership for every retained main-paper figure/table label",
            "paper scope anchors for implemented rows and explicitly scoped gaps",
        ],
        "required_row_ids": sorted(REQUIRED_ROW_IDS),
        "rows": rows,
        "figure_table_labels": sorted(figure_labels),
        "label_to_matrix_row": label_to_rows,
        "missing_label_mappings": missing_label_mappings,
        "extra_label_mappings": extra_label_mappings,
        "duplicate_label_mappings": duplicate_label_mappings,
        "checks": checks,
        "pages": pages,
        "violations": violations,
        "overall_pass": not violations,
    }


def markdown_bool(value: bool) -> str:
    return f"`{value}`"


def write_markdown(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# Evaluation completeness matrix",
        "",
        f"- Overall pass: `{report['overall_pass']}`",
        f"- Paper pages: `{report['pages']}`",
        f"- Required rows: `{len(report['required_row_ids'])}`",
        f"- Figure/table labels mapped: `{len(report['label_to_matrix_row'])}`",
        "",
        "## Matrix",
        "",
        "| Row | Required | Status | Primary labels | Evidence | Paper anchors | Scoped gaps |",
        "| --- | ---: | --- | --- | ---: | ---: | --- |",
    ]
    for row in report["rows"]:
        labels = ", ".join(f"`{label}`" for label in row["figure_table_labels"]) or "-"
        scoped = "; ".join(row["scoped_out"]) or "-"
        lines.append(
            f"| `{row['id']}` | {markdown_bool(row['required'])} | `{row['status']}` | "
            f"{labels} | {markdown_bool(row['evidence_pass'])} | "
            f"{markdown_bool(row['paper_anchor_pass'])} | {scoped} |"
        )

    lines += [
        "",
        "## Label ownership",
        "",
        "| Figure/table label | Matrix row |",
        "| --- | --- |",
    ]
    for label in sorted(report["figure_table_labels"]):
        owners = report["label_to_matrix_row"].get(label, [])
        lines.append(f"| `{label}` | {', '.join(f'`{owner}`' for owner in owners) or '-'} |")

    lines += [
        "",
        "## Evidence contracts",
        "",
        "| Matrix row | Artifact | Expected pass | Actual pass | Matrix pass | Role |",
        "| --- | --- | ---: | ---: | ---: | --- |",
    ]
    for row in report["rows"]:
        for item in row["evidence"]:
            lines.append(
                f"| `{row['id']}` | `{item['path']}` | "
                f"`{item['expected_overall_pass']}` | `{item['overall_pass']}` | "
                f"`{item['passes']}` | {item['role']} |"
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
        lines += [f"- {violation}" for violation in report["violations"]]

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    report = build_report()
    json_path = args.out / "evaluation_completeness_matrix.json"
    md_path = args.out / "evaluation_completeness_matrix.md"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown(report, md_path)
    print(json.dumps({
        "out_dir": relpath(args.out),
        "overall_pass": report["overall_pass"],
        "pages": report["pages"],
        "required_rows": len(report["required_row_ids"]),
        "mapped_labels": len(report["label_to_matrix_row"]),
        "violations": len(report["violations"]),
    }, indent=2, sort_keys=True))
    return 0 if report["overall_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
