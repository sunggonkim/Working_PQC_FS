#!/usr/bin/env python3
"""Build Gate 0.13/0.17 systems-technique transfer matrices.

This is a claim-boundary artifact, not a literature survey generator.  It
records which storage/OS mechanisms have a production analogue in AEGIS-Q,
which are only design inspiration, and which remain explicit non-claims.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "artifacts" / "validation" / "refactor_inventory"
SYSTEMS_JSON = OUT_DIR / "systems_literature_gate.json"
TECHNIQUE_JSON = OUT_DIR / "technique_transfer_matrix.json"
TECHNIQUE_MD = OUT_DIR / "technique_transfer_matrix.md"


@dataclass(frozen=True)
class Technique:
    prior_mechanism: str
    prior_source: str
    aegisq_analogue: str
    transfer_status: str
    code_modules: tuple[str, ...]
    artifact_paths: tuple[str, ...]
    paper_patterns: tuple[str, ...]
    contribution_boundary: str
    nonclaim_boundary: str


def now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def path_matches(pattern: str) -> list[str]:
    path = ROOT / pattern
    if any(ch in pattern for ch in "*?[]"):
        return sorted(rel(match) for match in ROOT.glob(pattern))
    return [pattern] if path.exists() else []


def pattern_hits(patterns: tuple[str, ...]) -> list[dict[str, Any]]:
    paper_files = [
        ROOT / "Paper" / "1_Introduction.tex",
        ROOT / "Paper" / "2_Background.tex",
        ROOT / "Paper" / "3_Design.tex",
        ROOT / "Paper" / "4_Evaluation.tex",
        ROOT / "Paper" / "5_Related_Works.tex",
        ROOT / "Paper" / "10_Discussion_and_Limitations.tex",
    ]
    hits: list[dict[str, Any]] = []
    for paper in paper_files:
        text = read_text(paper)
        lines = text.splitlines()
        for pattern in patterns:
            regex = re.compile(pattern, re.IGNORECASE)
            for line_no, line in enumerate(lines, 1):
                if regex.search(line):
                    hits.append(
                        {
                            "path": rel(paper),
                            "line": line_no,
                            "pattern": pattern,
                            "text": line.strip()[:240],
                        }
                    )
                    break
    return hits


def specs() -> list[Technique]:
    return [
        Technique(
            prior_mechanism="Append-only logging and roll-forward recovery",
            prior_source="F2FS FAST'15",
            aegisq_analogue="Strict journal plus opt-in epoch redo-log with replay of committed prefixes.",
            transfer_status="implemented-scoped-analogue",
            code_modules=(
                "code/storage/pqc_journal.c",
                "code/storage/pqc_epoch_log.c",
                "code/storage/pqc_epoch_publish.c",
                "code/fs/pqc_recovery.c",
            ),
            artifact_paths=(
                "docs/architecture/publication_protocol.md",
                "artifacts/validation/publication_protocol_fault_matrix/epoch_replay_fault_matrix.json",
                "artifacts/validation/publication_protocol_fault_matrix/epoch_publication_comparison.json",
            ),
            paper_patterns=(r"epoch-redo-log", r"committed epoch prefixes", r"roll-forward|replay"),
            contribution_boundary=(
                "AEGIS-Q uses an append/replay pattern for its FUSE sidecar format; it does not claim an F2FS segment cleaner, "
                "NAT/SIT design, or kernel-integrated flash filesystem."
            ),
            nonclaim_boundary="Do not call the epoch log an F2FS implementation or a flash-translation/filesystem replacement.",
        ),
        Technique(
            prior_mechanism="Operation logs and decentralized filesystem metadata",
            prior_source="ScaleFS SOSP'17",
            aegisq_analogue="Operation-style epoch records and a sharded leader/follower commit coordinator.",
            transfer_status="implemented-scoped-analogue",
            code_modules=(
                "code/storage/pqc_epoch_log.c",
                "code/fs/pqc_parallel_commit.c",
                "code/storage/pqc_epoch_publish.c",
            ),
            artifact_paths=(
                "artifacts/validation/parallel_commit_contract/parallel_commit_contract.json",
                "artifacts/validation/publication_protocol_fault_matrix/epoch_publication_comparison.json",
            ),
            paper_patterns=(r"leader/follower", r"epoch group", r"operation log|operation logs"),
            contribution_boundary=(
                "AEGIS-Q borrows the idea of operation-shaped metadata records, but keeps a FUSE sidecar runtime with scoped "
                "parallel commit; it does not claim ScaleFS decentralization, kernel shared-memory metadata, or POSIX scalability."
            ),
            nonclaim_boundary="No ScaleFS-equivalent scalability or decentralized metadata claim is allowed.",
        ),
        Technique(
            prior_mechanism="Compact metadata logging and selective flushing",
            prior_source="FastCommit ATC'24",
            aegisq_analogue="Batched strict journal append, epoch commit records, checkpoint compaction, and opt-in windowed anchor flushing.",
            transfer_status="implemented-scoped-analogue",
            code_modules=(
                "code/storage/pqc_strict_publish.c",
                "code/storage/pqc_epoch_log.c",
                "code/storage/pqc_epoch_publish.c",
                "code/storage/pqc_anchor_worker.c",
            ),
            artifact_paths=(
                "artifacts/validation/publication_protocol_fault_matrix/epoch_publication_comparison.json",
                "artifacts/validation/publication_protocol_fault_matrix/epoch_replay_fault_matrix.json",
            ),
            paper_patterns=(r"checkpoint compaction", r"sync-count amortization", r"window|selective|compact"),
            contribution_boundary=(
                "AEGIS-Q has compact sidecar records and selective/windowed freshness tradeoffs, not an ext4 FastCommit path "
                "or a complete journaling filesystem redesign."
            ),
            nonclaim_boundary="Do not imply ext4 FastCommit semantics, broad selective flushing safety, or physical power-loss certification.",
        ),
        Technique(
            prior_mechanism="Group commit and parallel journaling",
            prior_source="FAST/SOSP/OSDI storage systems",
            aegisq_analogue="Opt-in epoch leader/follower group barrier with sharded runtime telemetry and replay ordering.",
            transfer_status="implemented-scoped-analogue",
            code_modules=(
                "code/fs/pqc_parallel_commit.c",
                "code/storage/pqc_epoch_publish.c",
                "code/support/pqc_lock_profile.c",
            ),
            artifact_paths=(
                "artifacts/validation/parallel_commit_contract/parallel_commit_closure_audit.json",
                "artifacts/validation/parallel_commit_contract/parallel_commit_contract.json",
            ),
            paper_patterns=(r"group commit", r"leader/follower", r"bounded leader/follower"),
            contribution_boundary=(
                "AEGIS-Q can claim a scoped epoch-group mechanism only for the measured mounted path; it cannot claim broad "
                "multicore filesystem scalability without the concurrency-contract closeout."
            ),
            nonclaim_boundary="No broad scalability, multicore, or concurrent-client claim before Gate 0.15/A3 close.",
        ),
        Technique(
            prior_mechanism="Kernel QoS and I/O scheduling controls",
            prior_source="Linux ionice/cgroup/BFQ-style controls",
            aegisq_analogue="Storage-visible QoS classification and throttle path compared against two kernel-level controls.",
            transfer_status="measured-comparison",
            code_modules=(
                "code/runtime/pqc_qos.c",
                "code/runtime/pqc_admission.c",
                "code/experiments/run_sqlite_kernel_qos_baseline.py",
            ),
            artifact_paths=(
                "artifacts/validation/sqlite_kernel_qos_comparison/sqlite_kernel_qos_comparison.json",
                "artifacts/validation/sqlite_kernel_qos_baseline/sqlite_kernel_qos_baseline.json",
                "artifacts/validation/sqlite_kernel_qos_baseline_cgroup/sqlite_kernel_qos_baseline.json",
            ),
            paper_patterns=(r"ionice", r"IOWeight", r"bounded storage-visible control"),
            contribution_boundary=(
                "AEGIS-Q may claim storage-visible classification and observability tradeoffs; it may not claim kernel QoS is "
                "generally impossible or always worse."
            ),
            nonclaim_boundary="No SQLite uniqueness or general kernel-QoS impossibility claim.",
        ),
        Technique(
            prior_mechanism="Kernel-native encryption baselines",
            prior_source="fscrypt and dm-crypt",
            aegisq_analogue="Mode-aligned baseline verdicts with measured, environment-blocked, or unavailable rows.",
            transfer_status="baseline-boundary",
            code_modules=("code/experiments/build_kernel_baseline_feasibility.py", "code/experiments/run_frozen_dmcrypt_contract.py"),
            artifact_paths=(
                "artifacts/validation/kernel_baseline_feasibility/kernel_baseline_feasibility.json",
                "artifacts/validation/frozen_dmcrypt_contract/*.json",
            ),
            paper_patterns=(r"fscrypt", r"dm-crypt", r"environment-blocked|unavailable"),
            contribution_boundary=(
                "AEGIS-Q is positioned against kernel-native alternatives through measured or blocked rows, not by replacing their "
                "maturity or kernel trust boundary."
            ),
            nonclaim_boundary="No apples-to-apples fscrypt/dm-crypt speedup without matched measured rows.",
        ),
        Technique(
            prior_mechanism="TPM/TEE freshness and rollback boundaries",
            prior_source="TPM/TEE systems",
            aegisq_analogue="TPM NV replay-after-advance checks plus freshness ladder, without sealed-key release or PCR-bound rollback resistance.",
            transfer_status="implemented-negative-boundary",
            code_modules=(
                "code/storage/pqc_anchor.c",
                "code/storage/pqc_anchor_worker.c",
                "code/experiments/run_hardware_freshness_recovery_matrix.py",
            ),
            artifact_paths=(
                "artifacts/validation/freshness_ladder_claim_guard/*.json",
                "artifacts/validation/hardware_freshness_recovery_matrix/*.json",
                "artifacts/validation/tpm_freshness_bundle/*.json",
            ),
            paper_patterns=(r"TPM replay-after-advance", r"persistent PCR-bound", r"sealed-key|rollback"),
            contribution_boundary=(
                "AEGIS-Q may claim replay-after-advance fail-closed behavior under the retained model; it may not claim full rollback "
                "resistance, sealed-key release, or persistent PCR-bound freshness."
            ),
            nonclaim_boundary="No TPM rollback-resistance or PCR-bound freshness claim before Gate C6 closes.",
        ),
        Technique(
            prior_mechanism="FUSE passthrough, eBPF, io_uring, and kernel-bypass completion",
            prior_source="Modern kernel bypass and tracing paths",
            aegisq_analogue="Measured non-claim and ordinary FUSE path; no production bypass is part of the verified system.",
            transfer_status="rejected-nonclaim",
            code_modules=("code/frontend/pqc_fuse.c", "code/experiments/run_ebpf_iouring_scope_audit.py"),
            artifact_paths=("artifacts/validation/ebpf_iouring_scope_audit/ebpf_iouring_scope_audit.json",),
            paper_patterns=(r"eBPF", r"io\\_uring|io_uring", r"completion bypass|not implemented"),
            contribution_boundary=(
                "AEGIS-Q keeps ordinary FUSE semantics in the verified path; eBPF/io_uring completion bypass is not a contribution."
            ),
            nonclaim_boundary="No eBPF/io_uring bypass claim unless production mounted-path evidence exists.",
        ),
        Technique(
            prior_mechanism="GPU/UMA scheduling and storage staging",
            prior_source="GPUfs, GPUDirect, FlashNeuron/Fastensor-style staging",
            aegisq_analogue="CPU-first AES-GCM data path, optional GPU key-plane/maintenance lane, and explicit Jetson memory non-claims.",
            transfer_status="implemented-scoped-placement",
            code_modules=(
                "code/runtime/pqc_scheduler.c",
                "code/runtime/pqc_rekey.c",
                "code/gpu/cuda_pqc.cu",
                "code/gpu/cuda_aead.cu",
            ),
            artifact_paths=(
                "artifacts/validation/jetson_memory_contract/*.json",
                "artifacts/validation/jetson_optimization_ladder/*.json",
                "artifacts/validation/cuda_qos_contract/*.json",
            ),
            paper_patterns=(r"CPU-first", r"GPU elastic lane", r"NVMe-to-UVM|GPUDirect|zero-copy"),
            contribution_boundary=(
                "AEGIS-Q may claim measured placement asymmetry and optional GPU maintenance work; it may not claim direct storage DMA, "
                "GPUDirect, dma-buf zero-copy, or foreground AI QoS recovery."
            ),
            nonclaim_boundary="No NVIDIA mechanism is a paper mechanism without production mounted-path benefit evidence.",
        ),
    ]


def build_record(spec: Technique) -> dict[str, Any]:
    module_hits = [{"path": path, "exists": (ROOT / path).exists()} for path in spec.code_modules]
    artifact_hits = []
    missing_artifacts = []
    for pattern in spec.artifact_paths:
        matches = path_matches(pattern)
        if matches:
            artifact_hits.extend({"pattern": pattern, "path": match} for match in matches[:8])
            if len(matches) > 8:
                artifact_hits.append({"pattern": pattern, "path": f"{len(matches) - 8} more matches omitted"})
        else:
            missing_artifacts.append(pattern)
    hits = pattern_hits(spec.paper_patterns)
    missing_modules = [item["path"] for item in module_hits if not item["exists"]]
    return {
        "prior_mechanism": spec.prior_mechanism,
        "prior_source": spec.prior_source,
        "aegisq_analogue": spec.aegisq_analogue,
        "transfer_status": spec.transfer_status,
        "code_modules": module_hits,
        "artifact_evidence": artifact_hits,
        "missing_modules": missing_modules,
        "missing_artifacts": missing_artifacts,
        "paper_hits": hits[:12],
        "paper_hit_count": len(hits),
        "paper_patterns": list(spec.paper_patterns),
        "contribution_boundary": spec.contribution_boundary,
        "nonclaim_boundary": spec.nonclaim_boundary,
        "complete": not missing_modules and not missing_artifacts and bool(hits),
    }


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def render_md(payload: dict[str, Any]) -> str:
    lines = [
        "# Technique Transfer Matrix",
        "",
        f"- Generated: `{payload['generated_utc']}`",
        f"- Overall pass: `{str(payload['overall_pass']).lower()}`",
        f"- Complete rows: `{payload['complete_count']}/{payload['technique_count']}`",
        "",
        "| Prior mechanism | Source | AEGIS-Q analogue | Status | Complete | Boundary |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in payload["techniques"]:
        lines.append(
            "| "
            + " | ".join(
                [
                    row["prior_mechanism"],
                    row["prior_source"],
                    row["aegisq_analogue"],
                    f"`{row['transfer_status']}`",
                    f"`{str(row['complete']).lower()}`",
                    row["nonclaim_boundary"],
                ]
            )
            + " |"
        )
    if payload["blocking_items"]:
        lines.extend(["", "## Blocking Items", ""])
        for item in payload["blocking_items"]:
            lines.append(
                f"- `{item['prior_mechanism']}`: missing modules `{item['missing_modules']}`, "
                f"missing artifacts `{item['missing_artifacts']}`, paper hits `{item['paper_hit_count']}`"
            )
    return "\n".join(lines) + "\n"


def main() -> int:
    records = [build_record(spec) for spec in specs()]
    blocking = [row for row in records if not row["complete"]]
    payload = {
        "schema_version": 1,
        "generated_utc": now_utc(),
        "overall_pass": not blocking,
        "technique_count": len(records),
        "complete_count": sum(1 for row in records if row["complete"]),
        "blocking_items": [
            {
                "prior_mechanism": row["prior_mechanism"],
                "missing_modules": row["missing_modules"],
                "missing_artifacts": row["missing_artifacts"],
                "paper_hit_count": row["paper_hit_count"],
            }
            for row in blocking
        ],
        "techniques": records,
        "gate_0_13_close_condition": (
            "The spine is a secure edge-storage runtime bottleneck claim, and each borrowed OS/storage mechanism is "
            "classified as implemented analogue, measured comparison, baseline boundary, rejected non-claim, or scoped placement."
        ),
        "gate_0_17_close_condition": (
            "No FAST/SOSP/OSDI/ATC mechanism is treated as an AEGIS-Q contribution unless the row maps to production code, "
            "artifact evidence, paper location, and a non-claim boundary."
        ),
    }
    systems_payload = {
        key: payload[key]
        for key in (
            "schema_version",
            "generated_utc",
            "overall_pass",
            "technique_count",
            "complete_count",
            "blocking_items",
            "gate_0_13_close_condition",
        )
    }
    systems_payload["required_mechanisms"] = [
        {
            "prior_mechanism": row["prior_mechanism"],
            "prior_source": row["prior_source"],
            "transfer_status": row["transfer_status"],
            "complete": row["complete"],
            "contribution_boundary": row["contribution_boundary"],
            "nonclaim_boundary": row["nonclaim_boundary"],
        }
        for row in records
    ]
    write_json(SYSTEMS_JSON, systems_payload)
    write_json(TECHNIQUE_JSON, payload)
    TECHNIQUE_MD.parent.mkdir(parents=True, exist_ok=True)
    TECHNIQUE_MD.write_text(render_md(payload), encoding="utf-8")
    print(
        json.dumps(
            {
                "overall_pass": payload["overall_pass"],
                "complete_count": payload["complete_count"],
                "technique_count": payload["technique_count"],
                "blocking_items": payload["blocking_items"],
                "outputs": [rel(SYSTEMS_JSON), rel(TECHNIQUE_JSON), rel(TECHNIQUE_MD)],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if payload["overall_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
