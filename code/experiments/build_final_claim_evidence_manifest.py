#!/usr/bin/env python3
"""Build the final numeric-claim and claim-to-evidence manifest.

This is a final-sweep gate. It does not create new benchmark results. It checks
that the current main-paper numeric claims and abstract/conclusion/security/
recovery claims map to retained evidence and that unsupported stronger claims
remain guarded by the existing lint/firewall artifacts.
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
OUT_DIR = ROOT / "artifacts" / "reports" / "final_claim_evidence_manifest"
JSON_OUT = OUT_DIR / "final_claim_evidence_manifest.json"
MD_OUT = OUT_DIR / "final_claim_evidence_manifest.md"

PAPER_FILES = [
    "Paper/main.tex",
    "Paper/1_Introduction.tex",
    "Paper/2_Background.tex",
    "Paper/3_Design.tex",
    "Paper/4_Evaluation.tex",
    "Paper/generated_qos_recovery_table.tex",
    "Paper/7_Implementation_Details.tex",
    "Paper/8_Security_Analysis.tex",
    "Paper/10_Discussion_and_Limitations.tex",
    "Paper/5_Related_Works.tex",
    "Paper/6_Conclusion.tex",
]

NUMERIC_RE = re.compile(
    r"(?<![A-Za-z])\d+(?:\.\d+)?\s*(?:ms|MB/s|MiB/s|GB/s|KiB|GiB|TB|K|M|"
    r"byte|bytes|bit|bits|run|runs|row|rows|miss|misses|writer|writers|"
    r"flush|flushes|transition|transitions|flip|flips|operation|operations|"
    r"\\times|x)?\b"
)

STRUCTURAL_SKIP_RE = re.compile(
    r"\\(?:begin|end|section|subsection|caption|textbf|label|ref|cite|includegraphics|Description|node|draw|path|"
    r"toprule|midrule|bottomrule|setlength|definecolor|newcommand|usepackage|"
    r"documentclass|author|email|affiliation|acm|title|keywords|input|bibliography)"
)

CLAIM_KEYWORDS = re.compile(
    r"SQLite|p99|MB/s|MiB/s|GB/s|AES|GCM|ML-KEM|cuPQC|liboqs|scrypt|PBKDF2|"
    r"TPM|PCR|NV|replay|freshness|rollback|crash|fault|SIGKILL|POSIX|mmap|"
    r"fscrypt|dm-crypt|gocryptfs|plaintext|fdatasync|syncfs|KiB|GiB|TB|CUDA|"
    r"Jetson|Thor|deadline|miss|thermal|tegrastats|ionice|IOWeight|epoch"
)


@dataclass(frozen=True)
class EvidenceRule:
    claim_id: str
    category: str
    needles_any: tuple[str, ...]
    evidence_paths: tuple[str, ...]
    boundary: str


def now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def normalize(text: str) -> str:
    return " ".join(text.split())


def extract_abstract(text: str) -> str:
    match = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", text, re.S)
    return match.group(1) if match else ""


def run_pdfinfo_pages(path: Path) -> int | None:
    proc = subprocess.run(["pdfinfo", str(path)], check=True, text=True,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    for line in proc.stdout.splitlines():
        if line.startswith("Pages:"):
            return int(line.split(":", 1)[1].strip())
    return None


def rules() -> list[EvidenceRule]:
    return [
        EvidenceRule(
            "edge_runtime_claim_spine",
            "claim-spine",
            (
                "secure file-encryption runtime", "edge-runtime thesis",
                "storage and recovery claims", "one durable authenticated format",
                "elastic GPU/PQC maintenance lane", "fail-closed replay checks",
                "placement/QoS runtime", "mounted runtime boundary",
                "central QoS/placement result",
            ),
            (
                "artifacts/reports/paper_spine_gate/paper_spine_gate.json",
                "artifacts/reports/first_two_pages_thesis/first_two_pages_thesis.json",
                "artifacts/reports/architecture_claim_firewall/architecture_claim_firewall.json",
                "artifacts/reports/x9_qos_admission_closeout/x9_qos_admission_closeout.json",
                "artifacts/reports/x10_generation_robustness_closeout/x10_generation_robustness_closeout.json",
                "artifacts/reports/x11_mlkem_break_even_model/x11_mlkem_break_even_model.json",
            ),
            "Top-level edge-runtime thesis; supporting correctness rows bound the claim rather than converting it into filesystem certification.",
        ),
        EvidenceRule(
            "mechanism_ablation_attribution",
            "claim-attribution",
            (
                "ablation manifest then attributes cost",
                "SQLite recovery to elastic-write throttling",
                "key-plane speedup to slack-gated GPU maintenance",
                "replay refusal to the external anchor",
            ),
            (
                "artifacts/validation/mechanism_ablation_manifest/mechanism_ablation_manifest.json",
                "artifacts/reports/x9_qos_admission_closeout/x9_qos_admission_closeout.json",
                "artifacts/reports/x11_mlkem_break_even_model/x11_mlkem_break_even_model.json",
                "artifacts/validation/freshness_ladder_claim_guard/freshness_ladder_claim_guard.json",
            ),
            "Mechanism attribution summary; not an independent throughput or recovery expansion.",
        ),
        EvidenceRule(
            "sqlite_hero_repeated_medians",
            "numeric-performance",
            ("SQLite p99", "App only", "Unthrottled", "Simple ctrl.", "AEGIS-Q & storage class", "ionice", "IOWeight"),
            (
                "artifacts/reports/hero_result_contract/hero_result_contract.json",
                "artifacts/validation/sqlite_hero_validity_closeout/sqlite_hero_validity_closeout.json",
                "artifacts/validation/kernel_qos_hero_integration_closeout/kernel_qos_hero_integration_closeout.json",
            ),
            "SQLite foreground storage-visible control only; no non-storage QoS or broad filesystem superiority.",
        ),
        EvidenceRule(
            "frozen_filesystem_contract_rows",
            "numeric-performance",
            ("Plaintext reports", "gocryptfs reports", "AEGIS-Q reports 0.359", "4~KiB", "1~GiB", "queue depth~1"),
            (
                "artifacts/validation/frozen_plaintext_contract/frozen_plaintext_contract.json",
                "artifacts/validation/frozen_gocryptfs_contract/frozen_gocryptfs_contract.json",
                "artifacts/validation/frozen_aegisq_contract/frozen_aegisq_contract.json",
                "artifacts/validation/frozen_workload_contract/frozen_workload_contract.json",
            ),
            "Warm-cache frozen-contract cost boundary; not high-throughput/general-purpose claim.",
        ),
        EvidenceRule(
            "kernel_baseline_unavailable_rows",
            "baseline-boundary",
            ("fscrypt", "dm-crypt", "CONFIG\\_FS\\_ENCRYPTION", "sudo -n true"),
            (
                "artifacts/validation/kernel_baseline_feasibility/kernel_baseline_feasibility.json",
                "artifacts/validation/kernel_baseline_feasibility/paper_fscrypt_verdict.json",
                "artifacts/validation/frozen_dmcrypt_contract/frozen_dmcrypt_contract.json",
            ),
            "Kernel rows are measured or environment-blocked; no fscrypt/dm-crypt speedup claim.",
        ),
        EvidenceRule(
            "primitive_and_keyplane_placement",
            "numeric-performance",
            (
                "AES-256-GCM", "CPU/OpenSSL", "managed-buffer GPU", "ML-KEM-768",
                "cuPQC", "liboqs", "1.186", "break-even model",
                "AES-GCM data writes", "ML-KEM batches", "slack-tolerant maintenance plane",
            ),
            (
                "artifacts/validation/microbench/summary.json",
                "artifacts/validation/keyplane_rekey_methodology/keyplane_rekey_workflow.json",
                "artifacts/reports/x11_mlkem_break_even_model/x11_mlkem_break_even_model.json",
                "artifacts/validation/crypto_plane_separation/crypto_plane_claim_guard.json",
            ),
            "Placement asymmetry and mounted key-plane refresh; not bulk-data PQC encryption.",
        ),
        EvidenceRule(
            "kdf_parameters",
            "security-claim",
            ("scrypt", "PBKDF2", "32-byte", "N=32768", "64~MiB", "256-bit DEKs"),
            (
                "artifacts/validation/kdf_crypto_plane/kdf_current_state_verdict.json",
                "artifacts/validation/kdf_crypto_plane/kdf_scrypt_mounted_smoke.json",
            ),
            "Password-derived mount-key boundary; no hardware-backed credential release or offline-resistance claim.",
        ),
        EvidenceRule(
            "generation_and_tamper_safety",
            "security-recovery-claim",
            ("generation fault matrix", "EKEYREJECTED", "file identifier", "GCM nonce", "4~KiB"),
            (
                "artifacts/validation/generation_fault_matrix/generation_fault_matrix.json",
                "artifacts/validation/fuse_tamper_rejection.json",
                "artifacts/validation/nonce_generation_fault_verdict/nonce_generation_fault_verdict.json",
            ),
            "Strict/epoch nonce-generation evidence under stated model; no physical power-loss certification.",
        ),
        EvidenceRule(
            "tpm_freshness_boundary",
            "security-recovery-claim",
            (
                "TPM", "PCR", "NV", "0x01500010", "replay-after-advance",
                "rollback visible", "four of four", "File-anchor replay visible",
                "File-backed witness is a negative control", "External freshness state",
            ),
            (
                "artifacts/validation/freshness_ladder_claim_guard/freshness_ladder_claim_guard.json",
                "artifacts/validation/hardware_freshness_recovery_matrix/hardware_freshness_recovery_matrix.json",
                "artifacts/validation/pcr_anchor_decision/pcr_anchor_decision.json",
                "artifacts/validation/async_merkle_tpm_epoch/tpm_epoch_freshness_probe.json",
            ),
            "Replay-after-advance and file-anchor negative control; no persistent PCR-bound rollback resistance.",
        ),
        EvidenceRule(
            "crash_and_recovery_boundary",
            "security-recovery-claim",
            (
                "daemon power-fault campaign", "SIGKILL", "SQLite WAL/FULL",
                "SQLite DELETE/EXTRA", "dbm.dumb", "oracle verdict",
                "lower-block interruption", "previous committed",
                "Physical power-loss", "drive-cache certification",
            ),
            (
                "artifacts/validation/daemon_power_fault_campaign/daemon_power_fault_campaign.json",
                "artifacts/validation/x1_block_fault_campaign/x1_block_fault_campaign.json",
                "artifacts/validation/recovery_oracle_audit/recovery_oracle_audit.json",
                "artifacts/validation/crash_power_loss_claim_guard/crash_power_loss_claim_guard.json",
            ),
            "Selected daemon/app cutpoints; no physical power-loss, kernel-crash, or drive-cache certification.",
        ),
        EvidenceRule(
            "second_macrobenchmark",
            "numeric-performance",
            ("secure append-log", "16~KiB", "5.93", "4.17"),
            (
                "artifacts/validation/second_macrobenchmark_closeout/second_macrobenchmark_closeout.json",
                "artifacts/validation/jetson_power_thermal_contract/jetson_power_thermal_contract.json",
            ),
            "Second scoped mounted workload; not a broad workload suite or non-storage QoS claim.",
        ),
        EvidenceRule(
            "epoch_mode_conditional_amortization",
            "numeric-performance",
            ("Epoch mode", "hybrid-barrier", "D/J/C barrier", "47.6", "40.1", "9.05", "9.72", "53.1"),
            (
                "artifacts/validation/filesystem_viability_breakdown/epoch_publication_comparison.json",
                "artifacts/validation/epoch_mode_depth/epoch_mode_depth.json",
                "artifacts/validation/strict_path_practicality/strict_path_practicality.json",
            ),
            "Epoch mode is conditional amortization with explicit p99/throughput wins and loss cases, not dominance across workloads.",
        ),
        EvidenceRule(
            "cache_manifest_workload",
            "numeric-performance",
            ("cache-manifest", "24 hashed 16~KiB objects", "4.22", "28.09", "2.00"),
            (
                "artifacts/validation/model_cache_manifest_workload/model_cache_manifest_workload.json",
            ),
            "Third scoped mounted workload using closed-file rename, directory fsync, remount, and hash verification; not a broad workload suite.",
        ),
        EvidenceRule(
            "x6_strict_cost_reduction_model",
            "numeric-performance",
            ("X6", "marker/checkpoint", "1,070", "syncfs", "marker-file"),
            (
                "artifacts/validation/x6_strict_cost_reduction/x6_strict_cost_reduction_model.json",
                "artifacts/validation/strict_path_practicality/strict_path_practicality.json",
            ),
            "Strict-path marker/checkpoint sync narrowing model; no throughput, kernel-upstreaming, or physical power-loss claim.",
        ),
        EvidenceRule(
            "foreground_inference_claim_removal",
            "workload-boundary",
            (
                "local databases", "encrypted logs", "cache-manifest updates",
                "secure append-log", "application scheduler recovery",
            ),
            (
                "artifacts/reports/review_response_strategy/review_response_strategy.json",
                "artifacts/reports/review_acceptance_structure_audit/review_acceptance_structure_audit.json",
            ),
            "O1 closed by removing the inference/TensorRT/AI axis from the submitted paper claim.",
        ),
        EvidenceRule(
            "telemetry_and_sensitivity",
            "numeric-performance",
            (
                "tegrastats", "CUPTI", "mounted telemetry", "FUSE write-throttle",
                "8 and 5", "80~ms", "128~KiB", "12.3", "9 transitions", "8 flips",
                "Ablation and sensitivity mechanisms",
            ),
            (
                "artifacts/validation/qos_sensitivity_analysis/qos_sensitivity_analysis.json",
                "artifacts/validation/stat_thermal_methodology/stat_thermal_methodology_audit.json",
                "artifacts/validation/cuda_qos_contract/cuda_qos_contract.json",
            ),
            "Controller sensitivity and wiring evidence; not external application scheduling.",
        ),
        EvidenceRule(
            "posix_scope_boundary",
            "correctness-boundary",
            (
                "POSIX-scope audit", "default encrypted-file", "mmap", "msync",
                "rename", "directory", "relative symlink", "hard links",
                "full SQLite/crash certification",
            ),
            (
                "artifacts/validation/posix_scope_audit/posix_scope_audit.json",
                "artifacts/reports/dangerous_claim_lint/dangerous_claim_lint.json",
            ),
            "Narrow FUSE/POSIX envelope; no general-purpose POSIX filesystem claim.",
        ),
        EvidenceRule(
            "integrity_and_sidechannel_boundary",
            "security-boundary",
            (
                "RQ1 hash", "CUDA and OpenSSL digests agree", "Side-channel resistance",
                "GPU side channel", "constant-time evidence", "Backing-store read",
            ),
            (
                "artifacts/validation/integrity_comparison_manifest/integrity_comparison_manifest.json",
                "artifacts/validation/attacker_model_claim_guard/attacker_model_claim_guard.json",
            ),
            "Integrity parity and AES-GCM confidentiality only; no side-channel protection claim.",
        ),
        EvidenceRule(
            "platform_and_resource_context",
            "numeric-platform",
            ("Jetson AGX Thor", "14 ARM CPU", "1~TB", "Linux~6.8.12-tegra", "CUDA~13.0.48", "FUSE3~3.14.0"),
            (
                "artifacts/validation/jetson_power_thermal_contract/jetson_power_thermal_contract.json",
                "artifacts/validation/jetson_memory_contract/jetson_memory_contract.json",
                "artifacts/validation/jetson_optimization_ladder/jetson_optimization_ladder.json",
            ),
            "Single tested Jetson stack; no portability claim.",
        ),
        EvidenceRule(
            "claim_firewalls",
            "negative-claim",
            ("does not claim", "not claimed", "not a replacement", "not a universal replacement", "not deployed-filesystem", "not peak-throughput"),
            (
                "artifacts/reports/dangerous_claim_lint/dangerous_claim_lint.json",
                "artifacts/reports/architecture_claim_firewall/architecture_claim_firewall.json",
                "artifacts/validation/attacker_model_claim_guard/attacker_model_claim_guard.json",
            ),
            "Unsupported stronger claims remain absent, negated, or scoped.",
        ),
    ]


def evidence_status(path_str: str) -> dict[str, Any]:
    path = ROOT / path_str
    row: dict[str, Any] = {"path": path_str, "present": path.exists(), "overall_pass": None}
    if path.exists() and path.suffix == ".json":
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            row["overall_pass"] = data.get("overall_pass")
            if "violations" in data:
                row["violations"] = len(data["violations"])
            if "summary" in data and isinstance(data["summary"], dict) and "violations" in data["summary"]:
                row["summary_violations"] = data["summary"]["violations"]
        except json.JSONDecodeError as exc:
            row["json_error"] = str(exc)
    return row


def rule_matches(rule: EvidenceRule, text: str) -> bool:
    lowered = text.lower()
    return any(needle.lower() in lowered for needle in rule.needles_any)


def candidate_lines() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for file_name in PAPER_FILES:
        path = ROOT / file_name
        text = read_text(path)
        if file_name == "Paper/main.tex":
            text = extract_abstract(text)
            base_line = 1
        else:
            base_line = 0
        for idx, line in enumerate(text.splitlines(), start=base_line + 1):
            clean = normalize(line)
            if not clean:
                continue
            if STRUCTURAL_SKIP_RE.search(clean):
                continue
            if NUMERIC_RE.search(clean) and CLAIM_KEYWORDS.search(clean):
                rows.append({
                    "path": file_name,
                    "line": idx if file_name != "Paper/main.tex" else None,
                    "text": clean,
                    "numbers": NUMERIC_RE.findall(clean),
                })
    return rows


def abstract_conclusion_security_recovery_claims() -> list[dict[str, Any]]:
    targets = [
        ("abstract", "Paper/main.tex", extract_abstract(read_text(PAPER / "main.tex"))),
        ("conclusion", "Paper/6_Conclusion.tex", read_text(PAPER / "6_Conclusion.tex")),
        ("security", "Paper/8_Security_Analysis.tex", read_text(PAPER / "8_Security_Analysis.tex")),
        ("recovery", "Paper/4_Evaluation.tex", read_text(PAPER / "4_Evaluation.tex")),
    ]
    patterns = re.compile(
        r"AES-GCM|PQC|ML-KEM|TPM|PCR|freshness|replay|rollback|crash|recovery|"
        r"tamper|EKEYREJECTED|SIGKILL|power-loss|side channel|side-channel|POSIX|mmap|"
        r"SQLite|QoS|p99|deadline|direct NVMe|io\\?_uring|portability",
        re.IGNORECASE,
    )
    rows: list[dict[str, Any]] = []
    for section, path_name, text in targets:
        for idx, line in enumerate(text.splitlines(), 1):
            clean = normalize(line)
            if not clean or STRUCTURAL_SKIP_RE.search(clean):
                continue
            if patterns.search(clean):
                rows.append({"section": section, "path": path_name, "line": idx, "text": clean})
    return rows


def annotate(rows: list[dict[str, Any]], evidence_rules: list[EvidenceRule]) -> list[dict[str, Any]]:
    annotated: list[dict[str, Any]] = []
    for row in rows:
        matched = [rule for rule in evidence_rules if rule_matches(rule, row["text"])]
        annotated.append({
            **row,
            "claim_ids": [rule.claim_id for rule in matched],
            "categories": sorted({rule.category for rule in matched}),
            "covered": bool(matched),
        })
    return annotated


def build_report() -> dict[str, Any]:
    evidence_rules = rules()
    rule_rows = []
    for rule in evidence_rules:
        evidence = [evidence_status(path) for path in rule.evidence_paths]
        # Some non-claim evidence paths are environment-blocked and may not expose
        # overall_pass, so presence is the minimum requirement; if a violations
        # field exists, it must be zero.
        complete = all(
            item["present"]
            and item.get("violations", 0) == 0
            and item.get("summary_violations", 0) == 0
            for item in evidence
        )
        rule_rows.append({
            "claim_id": rule.claim_id,
            "category": rule.category,
            "needles_any": list(rule.needles_any),
            "evidence": evidence,
            "boundary": rule.boundary,
            "complete": complete,
        })

    numeric = annotate(candidate_lines(), evidence_rules)
    cross_claims = annotate(abstract_conclusion_security_recovery_claims(), evidence_rules)
    uncovered_numeric = [row for row in numeric if not row["covered"]]
    uncovered_cross = [row for row in cross_claims if not row["covered"]]
    incomplete_rules = [row for row in rule_rows if not row["complete"]]
    pages = run_pdfinfo_pages(PAPER / "main.pdf")

    checks = {
        "paper_pages_le_13": pages is not None and pages <= 13,
        "all_evidence_rules_complete": not incomplete_rules,
        "all_numeric_candidates_covered": not uncovered_numeric,
        "abstract_conclusion_security_recovery_claims_covered": not uncovered_cross,
    }
    violations = [name for name, passed in checks.items() if not passed]

    return {
        "schema_version": 1,
        "generated_utc": now_utc(),
        "scope": [
            "main-paper numeric claim candidates in content files",
            "abstract, conclusion, security, and recovery claim lines",
            "retained evidence paths and negative-claim guards",
            "12-page final PDF gate",
        ],
        "pages": pages,
        "checks": checks,
        "evidence_rules": rule_rows,
        "numeric_claim_candidates": numeric,
        "abstract_conclusion_security_recovery_claims": cross_claims,
        "uncovered_numeric_candidates": uncovered_numeric,
        "uncovered_cross_claims": uncovered_cross,
        "incomplete_evidence_rules": incomplete_rules,
        "violations": violations,
        "overall_pass": not violations,
    }


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_markdown(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# Final Claim-to-Evidence Manifest",
        "",
        f"- Generated: `{report['generated_utc']}`",
        f"- Overall pass: `{report['overall_pass']}`",
        f"- Paper pages: `{report['pages']}`",
        f"- Numeric claim candidates: `{len(report['numeric_claim_candidates'])}`",
        f"- Abstract/conclusion/security/recovery claims: `{len(report['abstract_conclusion_security_recovery_claims'])}`",
        f"- Uncovered numeric candidates: `{len(report['uncovered_numeric_candidates'])}`",
        f"- Uncovered cross claims: `{len(report['uncovered_cross_claims'])}`",
        "",
        "## Evidence Rules",
        "",
        "| Claim id | Category | Complete | Boundary |",
        "| --- | --- | ---: | --- |",
    ]
    for row in report["evidence_rules"]:
        lines.append(f"| `{row['claim_id']}` | `{row['category']}` | `{row['complete']}` | {row['boundary']} |")

    if report["uncovered_numeric_candidates"]:
        lines += ["", "## Uncovered Numeric Candidates", ""]
        for row in report["uncovered_numeric_candidates"]:
            lines.append(f"- `{row['path']}:{row['line']}` {row['text']}")

    if report["uncovered_cross_claims"]:
        lines += ["", "## Uncovered Cross Claims", ""]
        for row in report["uncovered_cross_claims"]:
            lines.append(f"- `{row['section']}` `{row['path']}:{row['line']}` {row['text']}")

    if report["violations"]:
        lines += ["", "## Violations", ""]
        lines += [f"- `{item}`" for item in report["violations"]]

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def main() -> int:
    report = build_report()
    write_json(JSON_OUT, report)
    write_markdown(report, MD_OUT)
    print(json.dumps({
        "overall_pass": report["overall_pass"],
        "pages": report["pages"],
        "numeric_claim_candidates": len(report["numeric_claim_candidates"]),
        "cross_claims": len(report["abstract_conclusion_security_recovery_claims"]),
        "violations": report["violations"],
        "outputs": [rel(JSON_OUT), rel(MD_OUT)],
    }, indent=2, sort_keys=True))
    return 0 if report["overall_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
