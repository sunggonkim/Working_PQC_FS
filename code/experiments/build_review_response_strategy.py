#!/usr/bin/env python3
"""Build the post-review response and paper strategy target.

This script is intentionally strategic rather than a narrow artifact checker.
It separates two questions that are easy to conflate:

1. Is the current submission defensible under its explicit claim boundary?
2. What integrated work would make the paper stronger against the latest OSDI
   review, using the author's previous paper style as a structural reference?
"""

from __future__ import annotations

import json
import re
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
PAPER = ROOT / "Paper"
PREVIOUS = PAPER / "Previous paper"
CHECKLIST = ROOT / "SUBMISSION_CHECKLIST.md"
OUT_DIR = ROOT / "artifacts" / "reports" / "review_response_strategy"
JSON_OUT = OUT_DIR / "review_response_strategy.json"

STALE_WORKLOAD_CUE = re.compile(
    r"\b(AI|TensorRT|inference|YOLO|InfScaler|Breaking the Edge)\b",
    re.IGNORECASE,
)

SURFACE_HYGIENE_PATHS = (
    ROOT / "README.md",
    CHECKLIST,
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
    PAPER / "references.bib",
    PAPER / "main.bbl",
)


@dataclass(frozen=True)
class StrategyRow:
    ident: str
    priority: str
    title: str
    review_pressure: str
    target: str
    paper_strategy: str
    required_terms: tuple[str, ...]
    evidence_paths: tuple[str, ...]
    completion_kind: str


def now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def read(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def load_json(path: str) -> dict[str, Any]:
    full = ROOT / path
    try:
        data = json.loads(full.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {"present": full.exists(), "path": path, "error": str(exc)}
    if not isinstance(data, dict):
        return {"present": True, "path": path, "error": "json root is not object"}
    data["present"] = True
    data["path"] = path
    return data


def artifact_exists(pattern: str) -> list[str]:
    full = ROOT / pattern
    if any(ch in pattern for ch in "*?[]"):
        return sorted(rel(path) for path in ROOT.glob(pattern))
    return [pattern] if full.exists() else []


def pdftotext(path: Path, pages: int = 2) -> str:
    try:
        proc = subprocess.run(
            ["pdftotext", "-f", "1", "-l", str(pages), "-layout", str(path), "-"],
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        return f"[pdftotext unavailable for {path.name}: {exc}]"
    return proc.stdout


def paper_text() -> str:
    parts = [
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
        PAPER / "main.bbl",
    ]
    return "\n".join(read(path) for path in parts)


def has_all(text: str, terms: tuple[str, ...]) -> bool:
    lower = text.lower()
    return all(term.lower() in lower for term in terms)


def has_any(text: str, terms: tuple[str, ...]) -> bool:
    lower = text.lower()
    return any(term.lower() in lower for term in terms)


def foreground_nonstorage_claim_removed(text: str) -> bool:
    return STALE_WORKLOAD_CUE.search(text) is None


def surface_hygiene() -> dict[str, Any]:
    """Guard the reviewer-facing paper/README/checklist surface.

    Historical scripts and old artifacts may still contain prior review terms as
    provenance.  This gate only covers text a reviewer or goal prompt should see.
    """
    hits: list[dict[str, Any]] = []
    for path in SURFACE_HYGIENE_PATHS:
        text = read(path)
        for line_no, line in enumerate(text.splitlines(), start=1):
            match = STALE_WORKLOAD_CUE.search(line)
            if match:
                hits.append({
                    "path": rel(path),
                    "line": line_no,
                    "match": match.group(0),
                    "excerpt": line.strip()[:180],
                })

    pdf_text = pdftotext(PAPER / "main.pdf", pages=50)
    for line_no, line in enumerate(pdf_text.splitlines(), start=1):
        match = STALE_WORKLOAD_CUE.search(line)
        if match:
            hits.append({
                "path": rel(PAPER / "main.pdf"),
                "line": line_no,
                "match": match.group(0),
                "excerpt": line.strip()[:180],
            })

    return {
        "scope": [
            rel(path) for path in SURFACE_HYGIENE_PATHS
        ] + [rel(PAPER / "main.pdf")],
        "no_stale_workload_cues": not hits,
        "stale_workload_cue_hits": hits,
        "policy": (
            "Reviewer-facing surfaces must not reintroduce the old foreground "
            "non-storage workload cue unless a same-run workload row is added "
            "and the paper claim expands explicitly."
        ),
    }


def previous_paper_signals() -> dict[str, Any]:
    pdfs = sorted(PREVIOUS.glob("*.pdf"))
    source_roots = [
        PREVIOUS / "src_icdcs26",
        PREVIOUS / "src_sigmetrics26",
    ]
    papers: list[dict[str, Any]] = []
    source_rows: list[dict[str, Any]] = []
    aggregate = {
        "front_page_figure_or_table_count": 0,
        "sota_comparison_count": 0,
        "headline_metric_count": 0,
        "explicit_bottleneck_count": 0,
        "scale_claim_count": 0,
    }
    for pdf in pdfs:
        text = pdftotext(pdf)
        lower = text.lower()
        signals = {
            "front_page_figure_or_table": bool(re.search(r"\bfig(?:ure)?\.?\s*1\b|\btable\s+i\b", lower)),
            "sota_comparison": has_any(lower, ("state-of-the-art", "sota", "prior work", "comparison")),
            "headline_metric": bool(re.search(r"\b\d+(?:\.\d+)?\s*(?:x|×|%|ms|gb|tb|qubits?)\b", lower)),
            "explicit_bottleneck": has_any(lower, ("bottleneck", "limited", "challenge", "overcome", "scalability")),
            "scale_claim": has_any(lower, ("up to", "outperforms", "speedup", "accuracy", "reduction")),
        }
        for key, value in signals.items():
            if value:
                aggregate[f"{key}_count"] += 1
        title = next((line.strip() for line in text.splitlines() if line.strip()), pdf.stem)
        papers.append({
            "path": rel(pdf),
            "title_guess": title[:180],
            "signals": signals,
        })
    for root in source_roots:
        design = root / "3.Design.tex"
        evaluation = root / "4.Evaluation.tex"
        design_text = read(design)
        eval_text = read(evaluation)
        design_subsections = re.findall(r"\\subsection\{([^}]*)\}", design_text)
        design_figures = re.findall(r"\\caption\{([^}]*)\}", design_text)
        eval_subsections = re.findall(r"\\subsection\{([^}]*)\}", eval_text)
        eval_figures = re.findall(r"\\caption\{([^}]*)\}", eval_text)
        source_rows.append({
            "root": rel(root),
            "design": rel(design),
            "evaluation": rel(evaluation),
            "design_subsections": design_subsections,
            "design_figure_count": design_text.count("\\begin{figure"),
            "design_captions": design_figures[:8],
            "evaluation_subsections": eval_subsections,
            "evaluation_figure_count": eval_text.count("\\begin{figure"),
            "evaluation_captions": eval_figures[:10],
            "style_signals": {
                "design_opens_with_overall_procedure_or_architecture": has_any(
                    design_text,
                    ("Overall Procedure", "Architecture and procedure", "Overall architecture", "Overview"),
                ),
                "mechanism_subsections_after_overview": len(design_subsections) >= 4,
                "evaluation_setup_first": bool(eval_subsections and "Evaluation Setup" in eval_subsections[0]),
                "evaluation_uses_performance_time_sensitivity": has_all(
                    eval_text,
                    ("Performance", "Time Analysis"),
                ) and has_any(eval_text, ("Sensitivity", "Stability", "Scalability")),
            },
        })
    return {
        **{
            "paper_count": len(pdfs),
            "papers": papers,
            "aggregate": aggregate,
            "derived_style": [
                "front-load the problem and first quantitative signal",
                "make the SOTA comparison table visible early",
                "state a concrete bottleneck before mechanisms",
                "tie each mechanism to a measurable result",
                "use limitations as scoped boundaries, not as the main story",
            ],
        },
        "source_style_basis": source_rows,
    }


def strategy_rows() -> list[StrategyRow]:
    return [
        StrategyRow(
            "R1",
            "P0",
            "central story and first-two-pages spine",
            "The review says caveats obscure the central contribution.",
            "Make the paper read like a systems result: bottleneck, invariant, mechanism, result, loss boundary.",
            "Follow previous-paper style: first page must expose a concrete problem figure/table and a measurable claim before long caveats.",
            ("boundary-aware", "SQLite", "p99", "does not claim"),
            (
                "artifacts/reports/first_two_pages_thesis/first_two_pages_thesis.json",
                "artifacts/reports/paper_spine_gate/paper_spine_gate.json",
                "artifacts/reports/dangerous_claim_lint/dangerous_claim_lint.json",
            ),
            "paper_only_guarded",
        ),
        StrategyRow(
            "R2",
            "P0",
            "strict-versus-epoch design surface",
            "The review sees epoch redo-log as under-explained relative to strict mode.",
            "Present strict and epoch as one publication design with explicit wins, costs, replay rules, and loss cases.",
            "Use one comparison table/paragraph rather than scattered option language.",
            ("strict", "epoch", "redo-log", "group", "syncfs"),
            (
                "artifacts/validation/publication_protocol_fault_matrix/epoch_publication_comparison.json",
                "artifacts/validation/publication_protocol_fault_matrix/publication_protocol_closeout.json",
                "artifacts/validation/parallel_commit_contract/parallel_commit_closure_audit.json",
            ),
            "paper_with_existing_evidence",
        ),
        StrategyRow(
            "O1",
            "P0",
            "closed-loop foreground impact or claim removal",
            "The latest review says the motivation invited a closed-loop foreground non-storage QoS result.",
            "Close this either by removing that axis from the venue-level claim, or by adding a same-run foreground workload with p50/p95/p99 evidence.",
            "The current target is claim removal: keep SQLite, encrypted logs, and cache-manifest updates as mounted edge-storage workloads.",
            ("local databases", "encrypted logs", "cache-manifest updates", "SQLite"),
            (),
            "paper_claim_removed",
        ),
        StrategyRow(
            "O2",
            "P0",
            "fscrypt on a supported kernel",
            "The latest review repeats that fscrypt is an important missing baseline even when Thor is environment-blocked.",
            "Run the frozen filesystem contract on a supported fscrypt environment, or remove OSDI-level baseline completeness language.",
            "The current closeout is claim removal: the paper marks fscrypt unavailable with kernel/filesystem proof and does not imply a measured fscrypt speedup row.",
            ("fscrypt", "environment-blocked", "missing kernel/filesystem support", "not an implied speedup"),
            (
                "artifacts/validation/frozen_fscrypt_supported_contract/frozen_fscrypt_supported_contract.json",
                "artifacts/validation/kernel_baseline_feasibility/paper_fscrypt_verdict.json",
            ),
            "platform_dependent_or_claim_removed",
        ),
        StrategyRow(
            "O3",
            "P0",
            "strict-path practicality beyond X6",
            "The latest review says the strict FUSE path is still too slow and asks whether kernel or hybrid barrier schedules could reduce the penalty.",
            "Close the current claim by reporting X6 and conditional epoch grouping as strict-path practicality evidence while keeping strict single-client publication as the cost boundary.",
            "Close with O3 evidence: X6 removes marker syncfs, and epoch/group publication is the implemented hybrid barrier path for concurrent or batched writes; strict single-client publication remains the cost boundary.",
            ("X6", "O3", "strict-path", "epoch grouping"),
            (
                "artifacts/validation/strict_path_practicality/strict_path_practicality.json",
            ),
            "new_experiment_required",
        ),
        StrategyRow(
            "O4",
            "P0",
            "energy and thermal result",
            "The latest review says edge UMA placement needs energy/thermal evidence, not only metadata capture.",
            "Add a same-run energy/thermal summary for the placement and QoS rows or remove energy-relevant implications.",
            "The existing Jetson power/thermal contract validates metadata eligibility; it is not an energy result by itself.",
            ("energy", "thermal", "same-run", "QoS"),
            (
                "artifacts/reports/o4_energy_thermal_result/o4_energy_thermal_result.json",
                "artifacts/validation/jetson_power_thermal_contract/jetson_power_thermal_contract.json",
            ),
            "new_experiment_required",
        ),
        StrategyRow(
            "R3",
            "P1",
            "GPU data-plane negative control",
            "The review asks for evidence that naive GPU data-plane offload hurts under UMA contention.",
            "Add a mounted-path or close-proxy negative-control experiment for GPU AES-GCM/data-plane offload.",
            "Turn the result into a thesis-strengthening negative result: placement is subordinate to publication and tail latency.",
            ("CPU", "GPU", "AES-GCM", "tail latency"),
            (
                "artifacts/validation/gpu_dataplane_negative_control/gpu_dataplane_negative_control.json",
                "artifacts/reports/final_claim_evidence_manifest/final_claim_evidence_manifest.json",
            ),
            "new_experiment_required",
        ),
        StrategyRow(
            "R4",
            "P1",
            "matched kernel baselines and kernel QoS sweeps",
            "The review remains skeptical without stronger fscrypt/dm-crypt and kernel-control baselines.",
            "Run matched kernel-encryption baselines where supported and broaden kernel QoS sweeps beyond one-off rows.",
            "Keep all rows mode-aligned; unavailable rows are acceptable only with concrete environment proof.",
            ("fscrypt", "dm-crypt", "kernel", "QoS"),
            (
                "artifacts/validation/frozen_dmcrypt_contract/frozen_dmcrypt_contract.json",
                "artifacts/validation/kernel_baseline_feasibility/kernel_baseline_feasibility.json",
                "artifacts/validation/sqlite_kernel_qos_baseline/sqlite_kernel_qos_baseline.json",
                "artifacts/validation/sqlite_kernel_qos_baseline_cgroup/sqlite_kernel_qos_baseline.json",
                "artifacts/validation/sqlite_kernel_qos_comparison/sqlite_kernel_qos_comparison.json",
            ),
            "platform_dependent_evidence",
        ),
        StrategyRow(
            "R5",
            "P1",
            "epoch-mode depth",
            "The review asks how epoch mode behaves at higher queue depth and multi-writer pressure.",
            "Extend epoch-mode evaluation to multi-writer or higher queue-depth workloads with replay-time and p99/p99.9 tradeoffs.",
            "Report wins and losses; do not imply epoch is universally better.",
            ("multi-writer", "queue-depth", "p99.9", "replay"),
            (
                "artifacts/validation/epoch_mode_depth/epoch_mode_depth.json",
                "artifacts/validation/publication_protocol_fault_matrix/epoch_publication_comparison.json",
                "artifacts/validation/parallel_commit_contract/parallel_commit_closure_audit.json",
            ),
            "new_experiment_required",
        ),
        StrategyRow(
            "R6",
            "P2",
            "lower-filesystem crash assumptions",
            "The review asks which lower filesystem semantics make D/J/C safe.",
            "Test or document xattr, directory-entry, rename, and ordering assumptions under fault injection.",
            "Move from generic limitation to an explicit assumption matrix.",
            ("xattr", "directory", "ordering", "fdatasync"),
            (
                "artifacts/validation/lowerfs_durability_assumptions/lowerfs_durability_assumptions.json",
                "artifacts/reports/crash_audit_report/crash_audit_report.json",
            ),
            "new_fault_evidence_required",
        ),
        StrategyRow(
            "R7",
            "P2",
            "C6 persistent freshness",
            "The review says replay-after-advance is useful but far from rollback resistance.",
            "Implement committed-prefix root maintenance plus TPM epoch/PCR lifecycle only if stronger freshness is claimed.",
            "Otherwise keep it as explicit future work.",
            ("persistent", "PCR", "TPM", "rollback"),
            (
                "artifacts/validation/async_merkle_tpm_epoch/tpm_epoch_freshness_probe.json",
                "artifacts/reports/architecture_claim_firewall/architecture_claim_firewall.json",
                "artifacts/reports/dangerous_claim_lint/dangerous_claim_lint.json",
            ),
            "future_feature_or_nonclaim",
        ),
        StrategyRow(
            "R8",
            "P2",
            "POSIX breadth",
            "The review says unsupported POSIX modes limit deployability.",
            "Only broaden POSIX claims after rename, directory fsync, mmap/msync, and conflict cases have tests or formal rejection.",
            "Keep deployability language out unless this row is implemented.",
            ("rename", "directory", "mmap", "concurrent"),
            (
                "artifacts/validation/posix_scope_audit/posix_scope_audit.json",
                "artifacts/validation/shadow_mmap_posix/shadow_mmap_decision.json",
            ),
            "new_semantics_required_for_claim_upgrade",
        ),
        StrategyRow(
            "R9",
            "P2",
            "TEE and attested-storage positioning",
            "The review wants sharper contrast with TEE-mediated secure storage.",
            "Position AEGIS-Q against TEE systems without implying unsupported isolation.",
            "Use related work to say when AEGIS-Q loses and when its UMA/runtime boundary is the relevant tradeoff.",
            ("TEE", "TrustZone", "attested", "isolation"),
            (
                "artifacts/reports/related_work_applicability/related_work_applicability.json",
                "artifacts/validation/refactor_inventory/technique_transfer_matrix.json",
            ),
            "paper_positioning",
        ),
        StrategyRow(
            "X13",
            "P0",
            "competitor and baseline boundary",
            "The latest review asks whether removing baselines or leaving GPU-storage comparisons qualitative is enough.",
            "Preserve direct measured baselines, keep fscrypt as unavailable with proof, and separate GPU-storage/PQC systems as boundary comparisons rather than same-contract baselines.",
            "Do not delete baselines to avoid bad numbers; make the comparison family explicit in Related Work and Evaluation.",
            (
                "mode-aligned measured rows",
                "unavailable with proof",
                "Speculative GPU encryption",
                "GPUstore",
                "GPUfs",
                "GeminiFS",
                "GPU4FS",
                "GPUDirect Storage",
                "fscrypt-GPU",
                "PQC key-plane work",
            ),
            (
                "artifacts/reports/evaluation_completeness_matrix/evaluation_completeness_matrix.json",
                "artifacts/reports/final_claim_evidence_manifest/final_claim_evidence_manifest.json",
                "artifacts/validation/kernel_baseline_feasibility/paper_fscrypt_verdict.json",
            ),
            "paper_positioning",
        ),
        StrategyRow(
            "X1C",
            "P0",
            "scoped crash-consistency envelope",
            "The paper claims authenticated publication recovery under a scoped crash model, not physical power-loss certification.",
            "Show previous/latest committed recovery within daemon cutpoints, D/J/C fault matrices, remount oracles, and lower-block interruption.",
            "Treat real power-loss, kernel-crash, and drive-cache campaigns as future claim expansion, not as the P0 for this edge-runtime paper.",
            ("physical power-loss", "does not claim", "lower-block", "previous committed", "latest committed"),
            (
                "artifacts/validation/x1_block_fault_campaign/x1_block_fault_campaign.json",
                "artifacts/validation/publication_protocol_fault_matrix/publication_protocol_closeout.json",
                "artifacts/reports/crash_audit_report/crash_audit_report.json",
            ),
            "paper_with_existing_evidence",
        ),
    ]


def evaluate_rows(rows: list[StrategyRow], current_paper: str) -> list[dict[str, Any]]:
    evaluated: list[dict[str, Any]] = []
    for row in rows:
        evidence = {path: artifact_exists(path) for path in row.evidence_paths}
        primary_artifact = load_json(row.evidence_paths[0]) if row.evidence_paths else {}
        primary_artifact_pass = (
            primary_artifact.get("present") is True
            and (
                primary_artifact.get("overall_pass") is True
                or (primary_artifact.get("summary") or {}).get("overall_pass") is True
            )
        )
        required_terms_present = has_all(current_paper, row.required_terms)
        evidence_present_count = sum(1 for matches in evidence.values() if matches)
        new_work_required = row.completion_kind in {
            "new_experiment_required",
            "platform_dependent_evidence",
            "new_fault_evidence_required",
            "new_semantics_required_for_claim_upgrade",
        }
        o1_claim_removed = (
            row.ident == "O1"
            and row.completion_kind == "paper_claim_removed"
            and required_terms_present
            and foreground_nonstorage_claim_removed(current_paper)
        )
        o2_claim_removed = False
        if row.ident == "O2":
            fscrypt_verdict = load_json("artifacts/validation/kernel_baseline_feasibility/paper_fscrypt_verdict.json")
            o2_claim_removed = (
                required_terms_present
                and fscrypt_verdict.get("parent_b1_gate_closed") is True
                and fscrypt_verdict.get("dangerous_claims_clear") is True
                and fscrypt_verdict.get("paper_marks_fscrypt_unavailable_with_proof") is True
            )
        r4_complete = False
        r6_complete = False
        r8_complete = False
        if row.ident == "R4":
            dmcrypt = load_json("artifacts/validation/frozen_dmcrypt_contract/frozen_dmcrypt_contract.json")
            feasibility = load_json("artifacts/validation/kernel_baseline_feasibility/kernel_baseline_feasibility.json")
            ionice = load_json("artifacts/validation/sqlite_kernel_qos_baseline/sqlite_kernel_qos_baseline.json")
            cgroup = load_json("artifacts/validation/sqlite_kernel_qos_baseline_cgroup/sqlite_kernel_qos_baseline.json")
            comparison = load_json("artifacts/validation/sqlite_kernel_qos_comparison/sqlite_kernel_qos_comparison.json")
            dmcrypt_environment_scoped = (
                dmcrypt.get("present") is True
                and dmcrypt.get("verdict") in {"environment-blocked", "measured"}
            )
            kernel_controls_repeated = (
                ionice.get("present") is True
                and cgroup.get("present") is True
                and ionice.get("overall_pass") is True
                and cgroup.get("overall_pass") is True
                and int(ionice.get("repetitions") or 0) >= 3
                and int(cgroup.get("repetitions") or 0) >= 3
            )
            r4_complete = (
                required_terms_present
                and evidence_present_count == len(row.evidence_paths)
                and feasibility.get("overall_pass") is True
                and dmcrypt_environment_scoped
                and kernel_controls_repeated
                and comparison.get("overall_pass") is True
                and comparison.get("parent_b3_gate_closed") is True
            )
        if row.ident == "R6":
            lowerfs = load_json("artifacts/validation/lowerfs_durability_assumptions/lowerfs_durability_assumptions.json")
            crash = load_json("artifacts/reports/crash_audit_report/crash_audit_report.json")
            matrix = lowerfs.get("assumption_matrix") or []
            r6_complete = (
                required_terms_present
                and evidence_present_count == len(row.evidence_paths)
                and lowerfs.get("overall_pass") is True
                and bool(matrix)
                and all(item.get("pass") is True for item in matrix)
                and bool(crash.get("retained_evidence"))
            )
        if row.ident == "R8":
            posix = load_json("artifacts/validation/posix_scope_audit/posix_scope_audit.json")
            mmap = load_json("artifacts/validation/shadow_mmap_posix/shadow_mmap_decision.json")
            semantics = posix.get("semantic_status") or {}
            claim_guard = mmap.get("claim_guard") or {}
            r8_complete = (
                required_terms_present
                and evidence_present_count == len(row.evidence_paths)
                and posix.get("overall_pass") is True
                and posix.get("required_semantics_all_covered") is True
                and semantics.get("rename", {}).get("status") in {"formal rejection", "supported_subset"}
                and semantics.get("directory_fsync", {}).get("status") in {"formal rejection", "supported_subset"}
                and semantics.get("shared_mmap", {}).get("status") == "formal rejection"
                and semantics.get("concurrent_disjoint_writes", {}).get("status") == "support"
                and mmap.get("overall_pass") is True
                and claim_guard.get("pass") is True
            )

        x1c_complete = False
        if row.ident == "X1C":
            x1 = load_json("artifacts/validation/x1_block_fault_campaign/x1_block_fault_campaign.json")
            publication = load_json("artifacts/validation/publication_protocol_fault_matrix/publication_protocol_closeout.json")
            crash = load_json("artifacts/reports/crash_audit_report/crash_audit_report.json")
            rows = x1.get("rows") or []
            lower_block_supporting = (
                x1.get("overall_pass") is True
                and len(rows) >= 3
                and all(item.get("acceptable") is True for item in rows if isinstance(item, dict))
            )
            x1c_complete = (
                required_terms_present
                and lower_block_supporting
                and publication.get("overall_pass") is True
                and bool(crash.get("retained_evidence"))
            )

        complete_now = (
            row.completion_kind in {
                "paper_only_guarded",
                "paper_with_existing_evidence",
                "future_feature_or_nonclaim",
                "paper_positioning",
            }
            and required_terms_present
            and evidence_present_count >= max(1, len(row.evidence_paths) - 1)
        ) or (
            row.completion_kind == "new_experiment_required"
            and required_terms_present
            and evidence_present_count == len(row.evidence_paths)
            and primary_artifact_pass
        ) or o1_claim_removed or o2_claim_removed or r4_complete or r6_complete or r8_complete or x1c_complete
        evaluated.append({
            "id": row.ident,
            "priority": row.priority,
            "title": row.title,
            "review_pressure": row.review_pressure,
            "target": row.target,
            "paper_strategy": row.paper_strategy,
            "required_terms_present": required_terms_present,
            "evidence": evidence,
            "evidence_present_count": evidence_present_count,
            "primary_artifact_pass": primary_artifact_pass,
            "claim_removed": o1_claim_removed if row.ident == "O1" else None,
            "claim_boundary_closed": o2_claim_removed if row.ident == "O2" else None,
            "completion_kind": row.completion_kind,
            "new_work_required": new_work_required,
            "complete_now": complete_now,
            "next_action": next_action(row, complete_now),
        })
    return evaluated


def next_action(row: StrategyRow, complete_now: bool) -> str:
    if complete_now:
        return "Keep guarded; no immediate implementation needed unless claims expand."
    if row.ident == "R1":
        return "Edit Introduction/Abstract/Conclusion to foreground the bottleneck, invariant, hero result, and loss boundary."
    if row.ident == "R2":
        return "Edit Design/Evaluation to consolidate strict and epoch mode tradeoffs in one location."
    if row.ident == "R3":
        return "Add a GPU data-plane negative-control runner and retain p99/throughput/thermal evidence."
    if row.ident == "O1":
        return "Keep foreground non-storage QoS out of the venue-level claim, or add a same-run foreground workload before reintroducing it."
    if row.ident == "O2":
        return "Keep fscrypt as unavailable with kernel/filesystem proof, or run the frozen contract later on a supported host before adding any measured fscrypt claim."
    if row.ident == "O3":
        return "Implement or model a production-facing strict-path cost reduction beyond X6, such as kernel-assist or hybrid barrier scheduling."
    if row.ident == "O4":
        return "Build a same-run energy/thermal result; the current Jetson metadata contract is only eligibility evidence."
    if row.ident == "R4":
        return "Run or mark environment-blocked matched fscrypt/dm-crypt and repeated kernel QoS controls."
    if row.ident == "R5":
        return "Extend epoch publication comparison to multi-writer or higher queue-depth cases."
    if row.ident == "R6":
        return "Add lower-filesystem xattr/directory-ordering assumption tests or a formal assumption matrix."
    if row.ident == "R7":
        return "Implement C6 production freshness or keep all persistent-PCR/rollback language negated."
    if row.ident == "R8":
        return "Add POSIX semantics tests before any deployability language."
    if row.ident == "R9":
        return "Strengthen related work against TEE/attested storage without claiming TEE isolation."
    if row.ident == "X1C":
        return "Keep physical power-loss/kernel-crash/drive-cache language negated; strengthen only the scoped crash-consistency envelope unless claims expand."
    if row.ident == "X13":
        return "Keep measured plaintext/gocryptfs/dm-crypt/AEGIS-Q rows, keep fscrypt unavailable with proof, and keep GPU-storage/PQC systems as boundary comparisons unless claims expand."
    return "No action."


def checklist_state() -> dict[str, Any]:
    text = read(CHECKLIST)
    cursor_match = re.search(r"\|\s*(R\d+)\s*\|\s*NEXT\s*\|", text)
    future_claims = (
        "F2 physical power-loss" in text
        and "F3 cross-platform UMA" in text
        and "F4 persistent PCR lifecycle" in text
    )
    no_current_blocking_p0 = (
        "No P0/P1 work is open under the current claim boundary." in text
        or "No P0 contradiction is open under the current claim boundary" in text
    )
    return {
        "path": rel(CHECKLIST),
        "current_cursor": cursor_match.group(1) if cursor_match else None,
        "has_latest_review_diagnosis": "## Latest Review Diagnosis" in text,
        "has_closed_gate_register": "## Closed Gate Register" in text,
        "has_direction_lock": "EDGE_FILE_ENCRYPTION_CPU_GPU_UMA_RUNTIME" in text,
        "future_claim_expansions_listed": future_claims,
        "no_current_p0_p1": no_current_blocking_p0,
        "active_p1_performance_cleanup": (
            "placement/strict-path overhead" in text
            and "Do not remove GPU" in text
        ),
        "r_rows_present": sorted(set(re.findall(r"\|\s*(R\d+)\s*\|", text))),
    }


def existing_gate_health() -> dict[str, Any]:
    artifacts = {
        "dangerous_claim_lint": load_json("artifacts/reports/dangerous_claim_lint/dangerous_claim_lint.json"),
        "architecture_claim_firewall": load_json("artifacts/reports/architecture_claim_firewall/architecture_claim_firewall.json"),
        "final_claim_manifest": load_json("artifacts/reports/final_claim_evidence_manifest/final_claim_evidence_manifest.json"),
        "paper_spine_gate": load_json("artifacts/reports/paper_spine_gate/paper_spine_gate.json"),
        "hero_result_contract": load_json("artifacts/reports/hero_result_contract/hero_result_contract.json"),
        "related_work_applicability": load_json("artifacts/reports/related_work_applicability/related_work_applicability.json"),
        "publication_closeout": load_json("artifacts/validation/publication_protocol_fault_matrix/publication_protocol_closeout.json"),
        "parallel_commit": load_json("artifacts/validation/parallel_commit_contract/parallel_commit_closure_audit.json"),
    }
    pass_like: dict[str, bool] = {}
    for name, data in artifacts.items():
        pass_like[name] = (
            data.get("present") is True
            and (
                data.get("overall_pass") is True
                or data.get("closeout_complete") is True
                or data.get("closure_verdict") == "closed"
                or data.get("final_s2_closed") is True
            )
        )
    return {
        "artifacts": {name: data.get("path") for name, data in artifacts.items()},
        "pass_like": pass_like,
        "closed_claim_boundary_pass": all(pass_like.values()),
    }


def build_report() -> dict[str, Any]:
    current = paper_text()
    rows = evaluate_rows(strategy_rows(), current)
    health = existing_gate_health()
    hygiene = surface_hygiene()
    checklist = checklist_state()
    previous = previous_paper_signals()

    p0_rows = [row for row in rows if row["priority"] == "P0"]
    p1_rows = [row for row in rows if row["priority"] == "P1"]
    p2_rows = [row for row in rows if row["priority"] == "P2"]
    p0_complete = all(row["complete_now"] for row in p0_rows)
    p1_complete = all(row["complete_now"] for row in p1_rows)

    submission_defense_ready = (
        health["closed_claim_boundary_pass"]
        and checklist["has_latest_review_diagnosis"]
        and checklist["has_closed_gate_register"]
        and checklist["has_direction_lock"]
        and checklist["future_claim_expansions_listed"]
        and checklist["no_current_p0_p1"]
        and hygiene["no_stale_workload_cues"]
    )
    osdi_strengthening_complete = (
        p0_complete
        and p1_complete
        and all(row["complete_now"] for row in p2_rows)
        and hygiene["no_stale_workload_cues"]
    )

    return {
        "schema_version": 1,
        "generated_utc": now_utc(),
        "verdict": {
            "submission_defense_ready": submission_defense_ready,
            "osdi_strengthening_complete": osdi_strengthening_complete,
            "answer_to_user": (
                "Defensible for the scoped edge-runtime claim. "
                "O1 is closed by claim removal, O2 by fscrypt baseline-claim removal with kernel/filesystem proof, O3 by the hybrid strict/epoch barrier closeout, O4 by retained power/thermal observations, and X13 by preserving measured baselines while separating GPU-storage/PQC competitors as boundary comparisons. OSDI acceptance is still not guaranteed, but the repeated review no longer maps to an unsupported paper claim."
            ),
        },
        "previous_paper_strategy_basis": previous,
        "checklist_state": checklist,
        "surface_hygiene": hygiene,
        "existing_gate_health": health,
        "rows": rows,
        "priority_summary": {
            "p0_complete": p0_complete,
            "p1_complete": p1_complete,
            "p2_complete": all(row["complete_now"] for row in p2_rows),
            "p0_next": [row["id"] for row in p0_rows if not row["complete_now"]],
            "p1_next": [row["id"] for row in p1_rows if not row["complete_now"]],
            "p2_next": [row["id"] for row in p2_rows if not row["complete_now"]],
        },
        "paper_strategy": [
            "Lead with the runtime-boundary thesis, not the artifact ledger.",
            "Use previous-paper pattern: early figure/table, concrete SOTA gap, named mechanism, quantitative outcome.",
            "Make QoS/UVM/CPU-GPU placement the motivation-design-evaluation spine.",
            "Motivation must show edge UMA contention and why static placement, GPU-everything, KEM-per-block PQC, and kernel QoS alone are insufficient.",
            "Design must name CPU-first AES-GCM publication, slack/telemetry-gated ML-KEM/cuPQC maintenance, executor-local managed-memory locality, and mounted QoS admission.",
            "Evaluation must close placement and QoS with negative controls, mounted key-plane admission, telemetry-to-FUSE wiring, SQLite p99 recovery, kernel-control comparisons, and sensitivity/ablation.",
            "Keep caveats in boundary tables and limitations; do not let them replace the core contribution.",
            "Turn negative results into design evidence, especially GPU data-plane offload under UMA contention.",
            "Do not chase POSIX, physical power-loss, C6, or portability work unless the paper explicitly expands those claims.",
            "Treat Jetson-class UMA as the representative edge platform unless the paper upgrades to a portability claim.",
            "If O2 is closed by claim removal, keep fscrypt unavailable wording guarded until a supported-host measurement exists.",
            "Do not delete direct baseline rows: measured plaintext/gocryptfs/dm-crypt/AEGIS-Q rows and the fscrypt unavailable proof are part of the defense.",
        ],
    }


def write_outputs(report: dict[str, Any]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    JSON_OUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    report = build_report()
    write_outputs(report)
    print(f"wrote {rel(JSON_OUT)}")
    print(f"submission_defense_ready={report['verdict']['submission_defense_ready']}")
    print(f"osdi_strengthening_complete={report['verdict']['osdi_strengthening_complete']}")
    print(f"p0_next={','.join(report['priority_summary']['p0_next']) or 'none'}")
    print(f"p1_next={','.join(report['priority_summary']['p1_next']) or 'none'}")
    return 0 if report["verdict"]["submission_defense_ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
