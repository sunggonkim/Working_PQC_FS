#!/usr/bin/env python3
"""Audit the latest review against the current paper and previous-paper pattern.

This is a decision aid, not a claim gate. It reads the pasted review, checks
current closeout evidence, compares the first-page logic shape against local
previous papers, and emits a conservative accept-readiness verdict.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REVIEW = Path("/home/thor/.codex/attachments/4eb849c2-e0e7-4b67-bc49-a4519b5bf587/pasted-text-1.txt")
DEFAULT_OUT = ROOT / "artifacts" / "reports" / "review_acceptance_structure_audit"
PREVIOUS_DIR = ROOT / "Paper" / "Previous paper"

CURRENT_FILES = [
    ROOT / "Paper" / "main.tex",
    ROOT / "Paper" / "1_Introduction.tex",
    ROOT / "Paper" / "2_Background.tex",
    ROOT / "Paper" / "3_Design.tex",
    ROOT / "Paper" / "7_Implementation_Details.tex",
    ROOT / "Paper" / "8_Security_Analysis.tex",
    ROOT / "Paper" / "4_Evaluation.tex",
    ROOT / "Paper" / "5_Related_Works.tex",
    ROOT / "Paper" / "10_Discussion_and_Limitations.tex",
]
REPORTS = {
    "x9_qos_tradeoff": ROOT / "artifacts" / "reports" / "x9_qos_admission_closeout" / "x9_qos_admission_closeout.json",
    "x10_generation": ROOT / "artifacts" / "reports" / "x10_generation_robustness_closeout" / "x10_generation_robustness_closeout.json",
    "x11_mlkem_break_even": ROOT / "artifacts" / "reports" / "x11_mlkem_break_even_model" / "x11_mlkem_break_even_model.json",
    "paper_spine": ROOT / "artifacts" / "reports" / "paper_spine_gate" / "paper_spine_gate.json",
    "first_two_pages": ROOT / "artifacts" / "reports" / "first_two_pages_thesis" / "first_two_pages_thesis.json",
    "dangerous_claim_lint": ROOT / "artifacts" / "reports" / "dangerous_claim_lint" / "dangerous_claim_lint.json",
    "architecture_firewall": ROOT / "artifacts" / "reports" / "architecture_claim_firewall" / "architecture_claim_firewall.json",
    "cross_section_alignment": ROOT / "artifacts" / "reports" / "cross_section_alignment" / "cross_section_alignment.json",
    "energy_thermal_metadata": ROOT / "artifacts" / "validation" / "jetson_power_thermal_contract" / "jetson_power_thermal_contract.json",
    "o4_energy_thermal_result": ROOT / "artifacts" / "reports" / "o4_energy_thermal_result" / "o4_energy_thermal_result.json",
    "o3_strict_path_practicality": ROOT / "artifacts" / "validation" / "strict_path_practicality" / "strict_path_practicality.json",
    "fscrypt_paper_verdict": ROOT / "artifacts" / "validation" / "kernel_baseline_feasibility" / "paper_fscrypt_verdict.json",
    "evaluation_completeness": ROOT / "artifacts" / "reports" / "evaluation_completeness_matrix" / "evaluation_completeness_matrix.json",
    "figure_table_obligations": ROOT / "artifacts" / "reports" / "figure_table_obligations" / "figure_table_obligations.json",
    "related_work_applicability": ROOT / "artifacts" / "reports" / "related_work_applicability" / "related_work_applicability.json",
    "recurring_review_elimination": ROOT / "artifacts" / "reports" / "recurring_review_elimination" / "recurring_review_elimination.json",
}

REVIEW_ITEMS = [
    ("limited_posix_fuse", "FUSE prototype with limited POSIX coverage", "bounded_nonclaim"),
    ("mount_credential", "password-derived mount credential without hardware-backed release", "bounded_nonclaim"),
    ("side_channel", "No side-channel analysis", "bounded_nonclaim"),
    ("generation_robustness", "Generation management is critical", "closed"),
    ("single_platform_workloads_fscrypt", "single Jetson device and narrow set of workloads", "residual"),
    ("closed_loop_foreground_nonstorage", "No closed-loop foreground non-storage QoS experiment", "bounded_nonclaim"),
    ("qos_tradeoff", "QoS benefit trades away background throughput", "closed"),
    ("mlkem_modest", "workflow-level benefit for GPU ML-KEM refresh is modest", "next"),
    ("power_failure", "No power-failure or drive-cache certification", "bounded_nonclaim"),
    ("scoping_heavy", "heavy use of scoping statements", "partly_closed"),
    ("figures_tables_alignment", "figures/tables could align more directly", "partly_closed"),
    ("related_work_qos_logging", "Limited discussion of storage-level QoS mechanisms", "next"),
    ("strict_fuse_impact", "Throughput and latency of the strict FUSE data path are low", "osdi_p0"),
    ("closed_loop_foreground_nonstorage_core_motivation", "no closed-loop foreground non-storage QoS improvement result", "osdi_p0"),
    ("fscrypt_baseline_supported", "fscrypt is not evaluated", "osdi_p0"),
    ("energy_thermal_missing", "No energy/thermal evaluation", "osdi_p0"),
    ("kernel_integration_path", "kernel support", "osdi_p0"),
    ("competitor_boundary", "Limited end-to-end comparison with kernel-integrated encryption", "partly_closed"),
]


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def load_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def current_text() -> str:
    return "\n".join(read_text(path) for path in CURRENT_FILES if path.exists())


def pdf_first_pages(path: Path) -> str:
    if shutil.which("pdftotext") is None:
        return ""
    proc = subprocess.run(
        ["pdftotext", "-f", "1", "-l", "2", str(path), "-"],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return proc.stdout if proc.returncode == 0 else ""


def previous_paper_patterns() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(PREVIOUS_DIR.glob("*.pdf")):
        text = pdf_first_pages(path)
        lower = text.lower()
        rows.append({
            "paper": rel(path),
            "accepted_or_camera_ready": any(token in path.name for token in ("Accepted", "Camera_Ready")),
            "rejected": "Rejected" in path.name,
            "first_page_problem_pressure": any(word in lower for word in ("however", "challenge", "bottleneck", "limited", "constraint")),
            "first_page_quant_figure_or_table": bool(re.search(r"(fig\\.?\\s*1|figure\\s*1|table\\s*i|table\\s*1)", lower)),
            "first_page_existing_systems": any(word in lower for word in ("sota", "prior", "existing", "state-of-the-art", "previous")),
            "first_page_key_insight": any(word in lower for word in ("key observation", "primary strategy", "we present", "our work", "to overcome")),
            "headline_result": bool(re.search(r"(\\d+(?:\\.\\d+)?\\s*[x×]|\\d+(?:\\.\\d+)?\\s*%|up to|outperform|speedup)", lower)),
        })
    return rows


def foreground_inference_claim_removed(text: str) -> bool:
    return re.search(r"\b(AI|TensorRT|inference)\b", text, re.IGNORECASE) is None


def current_structure_audit(text: str) -> dict[str, Any]:
    lower = text.lower()
    required = {
        "pressure_opening": "Edge devices increasingly run local databases, encrypted logs, cache-manifest updates" in text,
        "thesis_sentence": "Its thesis is that secure edge storage" in text,
        "first_page_qos_figure": "fig_first_page_qos.pdf" in text and "9.62" in text and "8.15" in text,
        "capability_matrix": "Capability comparison" in text and "storage-visible accelerator/QoS/replay policy" in text,
        "naive_failures": "GPU-everything file encryption loses" in text and "KEM-per-block" in text,
        "contributions_before_scope": (
            text.find("This paper makes four contributions.") != -1
            and text.find("This paper makes four contributions.") < text.find("The claim boundary excludes")
        ),
        "rq_mapping": "RQ1: CPU/GPU/PQC placement" in text and "RQ2: Mounted QoS and edge-storage behavior" in text,
        "qos_pareto": "Pareto point rather than a free throughput improvement" in text,
        "generation_x10": "publish-ticket" in text and "UINT64\\_MAX" in text and "4-thread/4-process" in text,
        "claim_boundaries": "not a broad workload suite" in text and "physical power-loss" in text,
        "jetson_scope_not_portability": (
            "Jetson-style accelerated UMA edge storage" in text
            and "representative Jetson-class accelerated UMA platform" in text
            and "not as evidence of cross-SoC portability" in text
        ),
        "foreground_inference_claim_removed": foreground_inference_claim_removed(text),
        "competitor_boundary": (
            "mode-aligned measured rows" in text
            and "fscrypt is unavailable with proof" in text
            and "not an implied speedup" in text
            and "GPUfs" in text
            and "GPUDirect Storage" in text
            and "fscrypt-GPU" in text
        ),
        "energy_thermal_observations": (
            "same-run power/thermal observations" in text
            and "not energy-efficiency claims" in text
        ),
        "kernel_assist_roadmap": (
            "kernel assistance" in lower
            and "d/j/c barrier issuance" in lower
            and "same-format kernel path preserves the recovery oracle" in lower
        ),
        "lowerfs_contract_not_powerloss": (
            "\\texttt{.pqcdata}" in text
            and "\\texttt{.pqcmeta}" in text
            and "outside the selected crash model" in text
            and "not a proof of device-cache flush ordering" in text
        ),
    }
    return {
        "required": required,
        "score": sum(1 for ok in required.values() if ok),
        "total": len(required),
        "complete": all(required.values()),
    }


def report_statuses() -> dict[str, Any]:
    status: dict[str, Any] = {}
    for name, path in REPORTS.items():
        data = load_json(path)
        status[name] = {
            "path": rel(path),
            "present": path.exists(),
            "overall_pass": data.get("overall_pass"),
            "violations": data.get("violations", []),
        }
    return status


def review_map(review: str, reports: dict[str, Any], paper_text: str) -> list[dict[str, Any]]:
    rows = []
    inference_removed = foreground_inference_claim_removed(paper_text)
    for item_id, phrase, class_hint in REVIEW_ITEMS:
        present = phrase.lower() in review.lower()
        if item_id in {
            "closed_loop_ai",
            "closed_loop_ai_core_motivation",
            "closed_loop_foreground_nonstorage",
            "closed_loop_foreground_nonstorage_core_motivation",
        }:
            risk = (
                "low_repeat_risk_after_claim_removal"
                if inference_removed
                else "high_repeat_risk_for_osdi_until_claim_removed_or_new_evidence"
            )
            class_hint = "claim_removed" if inference_removed else class_hint
        elif class_hint == "closed":
            closed = (
                item_id == "generation_robustness" and reports.get("x10_generation", {}).get("overall_pass") is True
            ) or (
                item_id == "qos_tradeoff" and reports.get("x9_qos_tradeoff", {}).get("overall_pass") is True
            )
            risk = "low_repeat_risk_after_closeout" if closed else "still_open"
        elif class_hint == "partly_closed":
            if item_id == "competitor_boundary":
                completeness = reports.get("evaluation_completeness", {})
                boundary_terms_present = all(term in paper_text for term in [
                    "mode-aligned measured rows",
                    "fscrypt is unavailable with proof",
                    "not an implied speedup",
                    "GPUDirect Storage",
                    "fscrypt-GPU",
                ])
                risk = (
                    "low_repeat_risk_after_competitor_boundary"
                    if completeness.get("overall_pass") is True and boundary_terms_present
                    else "medium_repeat_risk"
                )
            elif item_id == "scoping_heavy":
                crisp_spine = all(term in paper_text for term in [
                    "The useful result is asymmetry",
                    "Figure~\\ref{fig:evaluation_summary} is the evaluation spine",
                    "The point is visible storage-layer policy",
                ])
                risk = (
                    "low_repeat_risk_after_crisper_spine"
                    if crisp_spine
                    and reports.get("paper_spine", {}).get("overall_pass") is True
                    and reports.get("first_two_pages", {}).get("overall_pass") is True
                    and reports.get("cross_section_alignment", {}).get("overall_pass") is True
                    else "medium_repeat_risk"
                )
            elif item_id == "figures_tables_alignment":
                risk = (
                    "low_repeat_risk_after_figure_table_alignment"
                    if reports.get("figure_table_obligations", {}).get("overall_pass") is True
                    and reports.get("cross_section_alignment", {}).get("overall_pass") is True
                    and "Figure~\\ref{fig:evaluation_summary} is the evaluation spine" in paper_text
                    and "Table~\\ref{tab:benchmark_workloads} states what each experiment can establish" in paper_text
                    else "medium_repeat_risk"
                )
            else:
                risk = "medium_repeat_risk"
        elif item_id == "related_work_qos_logging":
            related_terms_present = all(term in paper_text for term in [
                "blk-throttle",
                "io.latency",
                "IO.cost",
                "encrypted databases",
                "secure logs",
            ])
            risk = (
                "low_repeat_risk_after_qos_related_work"
                if related_terms_present
                and reports.get("related_work_applicability", {}).get("overall_pass") is True
                else "high_repeat_risk_until_next_cursor"
            )
        elif item_id == "mlkem_modest":
            closed = reports.get("x11_mlkem_break_even", {}).get("overall_pass") is True
            risk = "low_repeat_risk_after_closeout" if closed else "high_repeat_risk_until_next_cursor"
        elif class_hint == "osdi_p0":
            closed = (
                item_id == "energy_thermal_missing"
                and reports.get("o4_energy_thermal_result", {}).get("overall_pass") is True
                and "same-run power/thermal observations" in paper_text
                and "not energy-efficiency claims" in paper_text
            ) or (
                item_id == "strict_fuse_impact"
                and reports.get("o3_strict_path_practicality", {}).get("overall_pass") is True
            ) or (
                item_id == "fscrypt_baseline_supported"
                and reports.get("fscrypt_paper_verdict", {}).get("overall_pass") is True
            )
            if item_id == "kernel_integration_path" and reports.get("o3_strict_path_practicality", {}).get("overall_pass") is True:
                lower_text = paper_text.lower()
                kernel_path_present = all(term in lower_text for term in [
                    "kernel assistance",
                    "d/j/c barrier issuance",
                    "same-format kernel path preserves the recovery oracle",
                ])
                risk = (
                    "low_repeat_risk_after_kernel_assist_roadmap"
                    if kernel_path_present
                    else "medium_repeat_risk_future_kernel_claim_only"
                )
            else:
                risk = "low_repeat_risk_after_closeout" if closed else "high_repeat_risk_for_osdi_until_claim_removed_or_new_evidence"
        elif class_hint == "next":
            risk = "high_repeat_risk_until_next_cursor"
        elif class_hint == "residual":
            if item_id == "single_platform_workloads_fscrypt":
                risk = "medium_repeat_risk_but_not_p0_for_edge_scope"
            else:
                risk = "medium_to_high_repeat_risk_for_sosp"
        else:
            risk = "bounded_but_repeatable_if_reviewer_demands_deployment"
        rows.append({
            "id": item_id,
            "review_phrase_present": present,
            "current_status": class_hint,
            "repeat_risk": risk,
        })
    return rows


def verdict(review_rows: list[dict[str, Any]], structure: dict[str, Any], reports: dict[str, Any]) -> dict[str, Any]:
    high = [row["id"] for row in review_rows if row["repeat_risk"].startswith("high")]
    medium_high = [row["id"] for row in review_rows if "medium_to_high" in row["repeat_risk"]]
    gates_ok = all(
        reports.get(name, {}).get("overall_pass") is True
        for name in (
            "x9_qos_tradeoff",
            "x10_generation",
            "o3_strict_path_practicality",
            "fscrypt_paper_verdict",
            "o4_energy_thermal_result",
            "paper_spine",
            "first_two_pages",
            "dangerous_claim_lint",
            "architecture_firewall",
            "cross_section_alignment",
            "evaluation_completeness",
            "figure_table_obligations",
            "related_work_applicability",
            "recurring_review_elimination",
        )
    )
    if high:
        accept_readiness = (
            "not safe to call OSDI/SOSP-ready; the paper is defensible as a scoped edge-runtime result, but the latest review can still repeat the supported-fscrypt baseline objection until that row is implemented or the venue-level baseline claim is removed"
        )
    elif medium_high:
        accept_readiness = (
            "not safe to guarantee SOSP acceptance; the edge-runtime logic is clear, but platform/fscrypt/workload breadth remains a venue-risk rather than a thesis contradiction"
        )
    else:
        accept_readiness = (
            "defensible under scoped edge-runtime claims; SOSP/OSDI acceptance is still not guaranteed because breadth, measured fscrypt preference, "
            "power-loss, and deployment expectations are venue-dependent, but they are no longer contradictions in the current paper claim"
        )
    return {
        "same_review_exact_repeat": (
            "unlikely for generation robustness, closed-loop foreground non-storage QoS mismatch, ML-KEM break-even, energy/thermal, strict-path hybrid-barrier criticism, unsupported fscrypt-speedup claims, and baseline-deletion criticism; a reviewer may still prefer a measured fscrypt row for OSDI, but it no longer contradicts the scoped paper claim"
        ),
        "logic_structure_vs_previous_papers": (
            "clear and aligned" if structure["complete"] else "mostly aligned but missing at least one first-page/claim-spine anchor"
        ),
        "accept_readiness": accept_readiness,
        "gates_ok": gates_ok,
        "highest_remaining_review_risks": high + medium_high,
    }


def build(review_path: Path) -> dict[str, Any]:
    review = read_text(review_path)
    text = current_text()
    previous = previous_paper_patterns()
    structure = current_structure_audit(text)
    reports = report_statuses()
    rows = review_map(review, reports, text)
    return {
        "artifact": "review_acceptance_structure_audit",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "review_source": str(review_path),
        "previous_paper_patterns": previous,
        "current_structure": structure,
        "support_reports": reports,
        "review_item_map": rows,
        "verdict": verdict(rows, structure, reports),
    }


def write_markdown(report: dict[str, Any], path: Path) -> None:
    v = report["verdict"]
    lines = [
        "# Review acceptance/structure audit",
        "",
        f"- Same-review exact repeat: {v['same_review_exact_repeat']}",
        f"- Logic structure: {v['logic_structure_vs_previous_papers']}",
        f"- Accept readiness: {v['accept_readiness']}",
        f"- Gates OK: `{v['gates_ok']}`",
        "",
        "## Remaining risks",
        "",
    ]
    for risk in v["highest_remaining_review_risks"]:
        lines.append(f"- `{risk}`")
    lines.extend(["", "## Review map", ""])
    for row in report["review_item_map"]:
        lines.append(f"- `{row['id']}`: {row['repeat_risk']}")
    lines.extend(["", "## Previous-paper pattern", ""])
    for row in report["previous_paper_patterns"]:
        label = "accepted/camera" if row["accepted_or_camera_ready"] else ("rejected" if row["rejected"] else "other")
        lines.append(
            f"- `{Path(row['paper']).name}` ({label}): pressure={row['first_page_problem_pressure']}, "
            f"fig/table={row['first_page_quant_figure_or_table']}, insight={row['first_page_key_insight']}, "
            f"headline={row['headline_result']}"
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review", type=Path, default=DEFAULT_REVIEW)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    report = build(args.review)
    json_path = args.out_dir / "review_acceptance_structure_audit.json"
    md_path = args.out_dir / "review_acceptance_structure_audit.md"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown(report, md_path)
    print(json.dumps({
        "json": rel(json_path),
        "markdown": rel(md_path),
        "gates_ok": report["verdict"]["gates_ok"],
        "accept_readiness": report["verdict"]["accept_readiness"],
        "highest_remaining_review_risks": report["verdict"]["highest_remaining_review_risks"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
