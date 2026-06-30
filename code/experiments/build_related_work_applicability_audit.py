#!/usr/bin/env python3
"""Audit related-work and applicability closure.

The audit is intentionally narrow: it verifies that the paper engages the
required system families, that the retained literature gates are present and
passing, and that the paper contains an explicit applicability boundary rather
than upgrading AEGIS-Q into a broad filesystem claim.
"""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
PAPER = ROOT / "Paper"
DEFAULT_OUT = ROOT / "artifacts" / "reports" / "related_work_applicability"
CHECKLIST = ROOT / "SUBMISSION_CHECKLIST.md"
REFERENCES = PAPER / "references.bib"

SYSTEMS_GATE = ROOT / "artifacts" / "validation" / "refactor_inventory" / "systems_literature_gate.json"
TRANSFER_MATRIX = ROOT / "artifacts" / "validation" / "refactor_inventory" / "technique_transfer_matrix.json"

PAPER_TEX = [
    PAPER / "1_Introduction.tex",
    PAPER / "2_Background.tex",
    PAPER / "3_Design.tex",
    PAPER / "4_Evaluation.tex",
    PAPER / "5_Related_Works.tex",
    PAPER / "6_Conclusion.tex",
    PAPER / "7_Implementation_Details.tex",
    PAPER / "8_Security_Analysis.tex",
    PAPER / "10_Discussion_and_Limitations.tex",
]

RELATED_REQUIRED = {
    "kernel_file_encryption": ("fscrypt", "dm-crypt", "kernel"),
    "userspace_encryption": ("gocryptfs", "FUSE"),
    "integrity_boundaries": ("fs-verity", "dm-integrity"),
    "kernel_qos_controls": ("kernel QoS", "ionice", "IOWeight"),
    "log_structured_storage": ("F2FS", "ScaleFS", "FastCommit"),
    "journaling_group_commit": ("journal", "epoch", "barrier"),
    "tpm_tee_freshness": ("TPM", "TEE", "OP-TEE", "PCR"),
    "gpu_crypto_storage_staging": ("GPU", "fscrypt-GPU", "FlashNeuron", "Fastensor", "SnuQS", "Speculative GPU encryption"),
    "gpu_filesystem_boundary": ("GPUstore", "GPUfs", "GeminiFS", "GPU4FS", "GPU-side execution"),
    "storage_bypass_nonclaims": ("SPDK", "GPUDirect Storage", "cuFile", "direct storage/GPU data path", "FUSE"),
    "same_contract_baseline_boundary": ("same-contract rows", "artificial throughput row", "mode-aligned measured rows"),
}

APPLICABILITY_REQUIRED = {
    "first_page_boundary": ("tab:applicability_boundary",),
    "protection_domain": ("authenticated FUSE", "mounted runtime"),
    "trust_assumptions": ("kernel/FUSE-daemon trust",),
    "platform_dependencies": ("Jetson/CUDA/TPM dependencies",),
    "known_loss_cases": ("gocryptfs", "free storage path"),
    "unsupported_guarantees": ("not a general POSIX", "PCR-bound", "power-loss"),
}

WIN_LOSE_APPLY = {
    "wins_or_value": ("storage-visible", "SQLite", "replay-after-advance"),
    "loses": ("gocryptfs", "free storage path", "not kernel-replacement maturity"),
    "does_not_apply": ("not a full fscrypt", "not as a replacement", "not a general POSIX"),
}

CHECKLIST_COMPETITOR_STRATEGY = {
    "baseline_deletion_forbidden": (
        "Baseline deletion verdict: no",
        "Do not delete baselines",
        "hiding the cost boundary",
    ),
    "direct_measured_rows_preserved": (
        "Direct measured baselines",
        "Plaintext/lowerfs",
        "gocryptfs",
        "dm-crypt",
        "AEGIS-Q",
    ),
    "fscrypt_unavailable_not_speedup": (
        "Required unavailable baseline",
        "fscrypt",
        "no measured fscrypt speedup/throughput is claimed",
    ),
    "gpu_storage_related_work_boundary": (
        "Related-work competitors",
        "FAST'19 speculative GPU CFS",
        "GPUstore",
        "GPUfs",
        "GeminiFS",
        "GPU4FS",
        "GPUDirect Storage",
    ),
    "sufficiency_scope": (
        "Current baseline sufficiency decision",
        "Enough for:",
        "Not enough for:",
        "Do not remove baselines",
    ),
}

BIBLIOGRAPHY_COMPETITOR_SET = {
    "storage_encryption_baselines": (
        "@article{fscrypt",
        "@article{dmcrypt",
        "@inproceedings{gocryptfs",
    ),
    "fuse_cost_boundary": (
        "@inproceedings{fuse_perf",
        "To {FUSE} or Not to {FUSE}",
    ),
    "gpu_crypto_file_system": (
        "@inproceedings{spec_gpu_cfs",
        "Speculative Encryption on {GPU} Applied to Cryptographic File Systems",
        "FAST 19",
    ),
    "gpu_storage_systems": (
        "@inproceedings{gpustore",
        "@inproceedings{gpufs",
        "@inproceedings{geminifs",
        "@inproceedings{gpu4fs",
    ),
    "direct_storage_gpu_path": (
        "@manual{gpudirect_storage",
        "cuFile API Reference Guide",
    ),
    "pqc_cost_model": (
        "@misc{pppqefs",
        "Predicting Performance for Post-Quantum Encrypted-File Systems",
    ),
}

DANGEROUS_CLAIMS = {
    "ready_for_deployment": re.compile(r"\bready for deployment\b|\bproduction ready\b", re.IGNORECASE),
    "general_purpose_filesystem": re.compile(r"\bgeneral-purpose filesystem\b", re.IGNORECASE),
    "direct_nvme_to_uvm": re.compile(r"direct NVMe-to-UVM|NVMe-to-UVM DMA", re.IGNORECASE),
    "gpudirect_rdma_claim": re.compile(r"GPUDirect/RDMA", re.IGNORECASE),
    "ebpf_iouring_bypass": re.compile(r"eBPF/io_uring completion bypass|io_uring/eBPF completion bypass", re.IGNORECASE),
    "persistent_pcr_freshness": re.compile(r"persistent PCR-bound freshness|persistent PCR binding", re.IGNORECASE),
    "foreground_ai_qos": re.compile(r"foreground AI QoS recovery|AI-inference QoS", re.IGNORECASE),
    "full_crash_certification": re.compile(r"full crash certification|power-loss certification", re.IGNORECASE),
    "side_channel_protection": re.compile(r"side-channel protection|GPU constant-time behavior", re.IGNORECASE),
}

NEGATION_HINTS = (
    "not ",
    "does not ",
    "no ",
    "without ",
    "excludes ",
    "unproven ",
    "outside scope",
    "remain open",
    "not claimed",
    "non-claim",
    "lacks ",
)


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def paper_text() -> str:
    return "\n".join(read(path) for path in PAPER_TEX)


def find_terms(text: str, terms: tuple[str, ...]) -> dict[str, list[dict[str, Any]]]:
    rows: dict[str, list[dict[str, Any]]] = {}
    for term in terms:
        hits: list[dict[str, Any]] = []
        pattern = re.compile(re.escape(term), re.IGNORECASE)
        for path in PAPER_TEX:
            lines = read(path).splitlines()
            for line_no, line in enumerate(lines, 1):
                if pattern.search(line):
                    hits.append({"path": rel(path), "line": line_no, "text": line.strip()[:240]})
                    break
        rows[term] = hits
    return rows


def has_all_terms(text: str, terms: tuple[str, ...]) -> bool:
    lowered = text.lower()
    return all(term.lower() in lowered for term in terms)


def term_group_report(text: str, groups: dict[str, tuple[str, ...]]) -> dict[str, Any]:
    return {
        name: {
            "required_terms": terms,
            "complete": has_all_terms(text, terms),
            "missing_terms": [
                term for term in terms if term.lower() not in text.lower()
            ],
        }
        for name, terms in groups.items()
    }


def inspect_json(path: Path, required_count: int = 9) -> dict[str, Any]:
    row: dict[str, Any] = {"path": rel(path), "present": path.exists(), "overall_pass": False}
    if not path.exists():
        return row
    data = json.loads(read(path))
    row.update({
        "overall_pass": bool(data.get("overall_pass")),
        "complete_count": data.get("complete_count"),
        "technique_count": data.get("technique_count"),
        "blocking_items": data.get("blocking_items", []),
    })
    row["complete"] = (
        row["overall_pass"]
        and row["complete_count"] == required_count
        and row["technique_count"] == required_count
        and not row["blocking_items"]
    )
    return row


def run_pdfinfo_pages(path: Path) -> int | None:
    proc = subprocess.run(["pdfinfo", str(path)], check=True, text=True,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    for line in proc.stdout.splitlines():
        if line.startswith("Pages:"):
            return int(line.split(":", 1)[1].strip())
    return None


def scan_dangerous_claims() -> list[dict[str, Any]]:
    hits: list[dict[str, Any]] = []
    for path in PAPER_TEX:
        lines = read(path).splitlines()
        for line_no, line in enumerate(lines, 1):
            lowered = line.lower()
            guarded = any(hint in lowered for hint in NEGATION_HINTS)
            for name, pattern in DANGEROUS_CLAIMS.items():
                if pattern.search(line) and not guarded:
                    hits.append({
                        "claim": name,
                        "path": rel(path),
                        "line": line_no,
                        "text": line.strip()[:260],
                    })
    return hits


def build_report() -> dict[str, Any]:
    text = paper_text()
    checklist_text = read(CHECKLIST)
    references_text = read(REFERENCES)
    related = {
        name: {
            "required_terms": terms,
            "complete": has_all_terms(text, terms),
            "hits": find_terms(text, terms),
        }
        for name, terms in RELATED_REQUIRED.items()
    }
    applicability = {
        name: {
            "required_terms": terms,
            "complete": has_all_terms(text, terms),
            "hits": find_terms(text, terms),
        }
        for name, terms in APPLICABILITY_REQUIRED.items()
    }
    win_loss = {
        name: {
            "required_terms": terms,
            "complete": has_all_terms(text, terms),
            "hits": find_terms(text, terms),
        }
        for name, terms in WIN_LOSE_APPLY.items()
    }
    systems_gate = inspect_json(SYSTEMS_GATE)
    transfer_matrix = inspect_json(TRANSFER_MATRIX)
    checklist_strategy = term_group_report(checklist_text, CHECKLIST_COMPETITOR_STRATEGY)
    bibliography_set = term_group_report(references_text, BIBLIOGRAPHY_COMPETITOR_SET)
    dangerous_hits = scan_dangerous_claims()
    pages = run_pdfinfo_pages(PAPER / "main.pdf")

    checks = {
        "paper_pdf_pages_12": pages == 12,
        "systems_literature_gate_passes": bool(systems_gate.get("complete")),
        "technique_transfer_matrix_passes": bool(transfer_matrix.get("complete")),
        "all_related_topics_present": all(row["complete"] for row in related.values()),
        "applicability_boundary_present": all(row["complete"] for row in applicability.values()),
        "win_loss_nonapply_language_present": all(row["complete"] for row in win_loss.values()),
        "checklist_competitor_strategy_present": all(
            row["complete"] for row in checklist_strategy.values()
        ),
        "bibliography_competitor_set_present": all(
            row["complete"] for row in bibliography_set.values()
        ),
        "dangerous_claims_guarded": len(dangerous_hits) == 0,
    }
    violations = [name for name, passed in checks.items() if not passed]
    return {
        "schema_version": 1,
        "scope": "related-work and applicability closure",
        "paper_pages": pages,
        "systems_literature_gate": systems_gate,
        "technique_transfer_matrix": transfer_matrix,
        "related_topics": related,
        "applicability_boundary": applicability,
        "win_loss_nonapply": win_loss,
        "checklist_competitor_strategy": checklist_strategy,
        "bibliography_competitor_set": bibliography_set,
        "dangerous_claim_hits": dangerous_hits,
        "checks": checks,
        "violations": violations,
        "overall_pass": not violations,
    }


def write_markdown(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# Related-work and applicability audit",
        "",
        f"- Overall pass: `{report['overall_pass']}`",
        f"- Paper pages: `{report['paper_pages']}`",
        f"- Dangerous unguarded claim hits: `{len(report['dangerous_claim_hits'])}`",
        "",
        "## Checks",
        "",
        "| Check | Pass |",
        "| --- | ---: |",
    ]
    for name, passed in report["checks"].items():
        lines.append(f"| `{name}` | `{passed}` |")

    lines += ["", "## Related Topics", "", "| Topic | Complete | Required terms |", "| --- | ---: | --- |"]
    for name, row in report["related_topics"].items():
        terms = ", ".join(f"`{term}`" for term in row["required_terms"])
        lines.append(f"| `{name}` | `{row['complete']}` | {terms} |")

    lines += ["", "## Applicability Boundary", "", "| Row | Complete | Required terms |", "| --- | ---: | --- |"]
    for name, row in report["applicability_boundary"].items():
        terms = ", ".join(f"`{term}`" for term in row["required_terms"])
        lines.append(f"| `{name}` | `{row['complete']}` | {terms} |")

    lines += ["", "## Win/Loss/Non-Apply Language", "", "| Row | Complete | Required terms |", "| --- | ---: | --- |"]
    for name, row in report["win_loss_nonapply"].items():
        terms = ", ".join(f"`{term}`" for term in row["required_terms"])
        lines.append(f"| `{name}` | `{row['complete']}` | {terms} |")

    lines += ["", "## Checklist Competitor Strategy", "", "| Row | Complete | Missing terms |", "| --- | ---: | --- |"]
    for name, row in report["checklist_competitor_strategy"].items():
        missing = ", ".join(f"`{term}`" for term in row["missing_terms"]) or "-"
        lines.append(f"| `{name}` | `{row['complete']}` | {missing} |")

    lines += ["", "## Bibliography Competitor Set", "", "| Row | Complete | Missing terms |", "| --- | ---: | --- |"]
    for name, row in report["bibliography_competitor_set"].items():
        missing = ", ".join(f"`{term}`" for term in row["missing_terms"]) or "-"
        lines.append(f"| `{name}` | `{row['complete']}` | {missing} |")

    if report["dangerous_claim_hits"]:
        lines += ["", "## Unguarded Dangerous Claims", "", "| Claim | Location | Text |", "| --- | --- | --- |"]
        for hit in report["dangerous_claim_hits"]:
            lines.append(f"| `{hit['claim']}` | `{hit['path']}:{hit['line']}` | {hit['text']} |")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    DEFAULT_OUT.mkdir(parents=True, exist_ok=True)
    report = build_report()
    (DEFAULT_OUT / "related_work_applicability.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_markdown(report, DEFAULT_OUT / "related_work_applicability.md")
    print(json.dumps({
        "overall_pass": report["overall_pass"],
        "paper_pages": report["paper_pages"],
        "violations": report["violations"],
        "dangerous_claim_hits": len(report["dangerous_claim_hits"]),
    }, indent=2, sort_keys=True))
    return 0 if report["overall_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
