#!/usr/bin/env python3
"""Gate 0.2-S0 recurring review-objection map builder."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "artifacts" / "validation" / "refactor_inventory"
JSON_OUT = OUT_DIR / "review_objection_map.json"
MD_OUT = OUT_DIR / "review_objection_map.md"
FREEZE = OUT_DIR / "worktree_freeze.json"

STATUS_VALUES = {
    "code-required",
    "artifact-required",
    "paper-required",
    "negative-claim-only",
    "dropped-claim",
}


@dataclass(frozen=True)
class ArtifactSpec:
    label: str
    pattern: str


@dataclass(frozen=True)
class ObjectionSpec:
    ident: str
    title: str
    status: str
    reviewer_objection: str
    gate_links: tuple[str, ...]
    primary_code_module: str
    code_modules: tuple[str, ...]
    artifact_specs: tuple[ArtifactSpec, ...]
    required_missing_artifacts: tuple[str, ...]
    paper_claim_boundary: str
    negative_guard: str


def now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def relpath(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def read_json(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def path_exists(path_or_pattern: str) -> list[str]:
    path = ROOT / path_or_pattern
    if any(ch in path_or_pattern for ch in "*?[]"):
        return sorted(relpath(match) for match in ROOT.glob(path_or_pattern))
    if path.exists():
        return [path_or_pattern]
    return []


def classify_artifacts(specs: tuple[ArtifactSpec, ...]) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    retained: list[dict[str, str]] = []
    missing: list[dict[str, str]] = []
    for spec in specs:
        matches = path_exists(spec.pattern)
        if matches:
            retained.extend({"label": spec.label, "path": match} for match in matches[:8])
            if len(matches) > 8:
                retained.append(
                    {
                        "label": spec.label,
                        "path": f"{spec.pattern} ({len(matches) - 8} more matches omitted)",
                    }
                )
        else:
            missing.append({"label": spec.label, "path": spec.pattern})
    return retained, missing


def module_status(modules: tuple[str, ...]) -> list[dict[str, Any]]:
    return [
        {
            "path": module,
            "exists": bool(path_exists(module)),
        }
        for module in modules
    ]


def specs() -> list[ObjectionSpec]:
    return [
        ObjectionSpec(
            ident="O01",
            title="codebase architecture still looks patch-accumulated",
            status="code-required",
            reviewer_objection=(
                "The current filesystem path still needs a defensible storage-runtime architecture, "
                "not a set of accumulated FUSE, CUDA, TPM, and checklist patches."
            ),
            gate_links=("0.1", "0.4", "0.5", "0.6", "0.7", "0.8", "0.14"),
            primary_code_module="code/frontend/pqc_fuse.c",
            code_modules=(
                "code/frontend/pqc_fuse.c",
                "code/runtime/pqc_config.c",
                "code/common/pqc_format.h",
                "code/fs/pqc_file_io.c",
                "code/storage/pqc_publish.c",
                "code/storage/pqc_recovery.c",
                "code/storage/pqc_anchor.c",
                "code/support/pqc_test_hooks.c",
            ),
            artifact_specs=(
                ArtifactSpec("worktree freeze", "artifacts/validation/refactor_inventory/worktree_freeze.json"),
                ArtifactSpec("source ownership map", "artifacts/validation/refactor_inventory/source_ownership_map.json"),
                ArtifactSpec("phase1 behavior equivalence", "artifacts/validation/refactor_inventory/phase1_behavior_equivalence.json"),
            ),
            required_missing_artifacts=(
                "Gate 0.14 behavior-equivalence closeout after decomposition",
                "paper negative guard for clean-module claims",
            ),
            paper_claim_boundary=(
                "Implementation text may describe current modules only as refactored components after behavior-equivalence evidence; "
                "until then, it must not claim a production-clean architecture."
            ),
            negative_guard=(
                "No production-clean, modular-storage-runtime, or behavior-equivalent refactor claim before Gate 0.14 closes."
            ),
        ),
        ObjectionSpec(
            ident="O02",
            title="filesystem throughput viability remains the top blocker",
            status="artifact-required",
            reviewer_objection=(
                "AEGIS-Q frozen-contract throughput is far below plaintext and gocryptfs, and the paper must explain "
                "whether the bottleneck is semantic, implementation, or redesign debt."
            ),
            gate_links=("A1", "A2", "A3", "A4", "A5", "0.9", "0.15", "0.16"),
            primary_code_module="code/fs/pqc_file_io.c",
            code_modules=(
                "code/fs/pqc_file_io.c",
                "code/storage/pqc_strict_publish.c",
                "code/storage/pqc_epoch_publish.c",
                "code/storage/pqc_epoch_log.c",
                "code/storage/pqc_parallel_commit.c",
                "code/support/pqc_metrics.c",
            ),
            artifact_specs=(
                ArtifactSpec("frozen AEGIS-Q contract", "artifacts/validation/frozen_aegisq_contract/frozen_aegisq_contract.json"),
                ArtifactSpec("epoch publication comparison", "artifacts/validation/publication_protocol_fault_matrix/epoch_publication_comparison.json"),
                ArtifactSpec("parallel commit contract", "artifacts/validation/parallel_commit_contract/parallel_commit_contract.json"),
            ),
            required_missing_artifacts=(
                "filesystem_viability_breakdown end-to-end component attribution",
                "strict-versus-epoch lock-hold and p99/p99.9 closeout",
                "paper text classifying the dominant bottleneck",
            ),
            paper_claim_boundary=(
                "The paper may report conservative strict-mode results and epoch-mode diagnostics, but no high-throughput or general-purpose "
                "filesystem implication is allowed until A1/A3 close."
            ),
            negative_guard=(
                "No high-throughput, competitive-with-gocryptfs, or general-purpose filesystem wording before viability breakdown and epoch evidence close."
            ),
        ),
        ObjectionSpec(
            ident="O03",
            title="fscrypt matched baseline is unresolved",
            status="artifact-required",
            reviewer_objection="fscrypt is missing from the matched frozen-contract baseline table.",
            gate_links=("B1", "B4"),
            primary_code_module="code/experiments/build_kernel_baseline_feasibility.py",
            code_modules=(
                "code/experiments/build_kernel_baseline_feasibility.py",
                "code/experiments/build_frozen_workload_contract.py",
            ),
            artifact_specs=(
                ArtifactSpec("kernel baseline feasibility", "artifacts/validation/kernel_baseline_feasibility/kernel_baseline_feasibility.json"),
                ArtifactSpec("frozen workload contract", "artifacts/validation/frozen_workload_contract/frozen_workload_contract.json"),
            ),
            required_missing_artifacts=("matched fscrypt fio row or environment-blocked proof",),
            paper_claim_boundary="fscrypt must appear only as measured, environment-blocked, or removed from the comparison.",
            negative_guard="No fscrypt speedup or apples-to-apples implication without a measured fscrypt row.",
        ),
        ObjectionSpec(
            ident="O04",
            title="dm-crypt matched baseline is unresolved",
            status="artifact-required",
            reviewer_objection="dm-crypt appears runnable but lacks a matched frozen-contract row.",
            gate_links=("B2", "B4"),
            primary_code_module="code/experiments/build_kernel_baseline_feasibility.py",
            code_modules=(
                "code/experiments/build_kernel_baseline_feasibility.py",
                "code/experiments/build_frozen_workload_contract.py",
            ),
            artifact_specs=(
                ArtifactSpec("kernel baseline feasibility", "artifacts/validation/kernel_baseline_feasibility/kernel_baseline_feasibility.json"),
                ArtifactSpec("dm-crypt raw row", "artifacts/validation/frozen_dmcrypt_contract/*.json"),
            ),
            required_missing_artifacts=("LUKS/dmsetup setup logs and matched fio JSON for dm-crypt/ext4",),
            paper_claim_boundary="dm-crypt is either a measured row or an unavailable row with root/device proof.",
            negative_guard="Historical dm-crypt diagnostics are not a paper comparison row.",
        ),
        ObjectionSpec(
            ident="O05",
            title="SQLite QoS lacks kernel-control baselines",
            status="artifact-required",
            reviewer_objection="SQLite p99 recovery may be achievable with standard Linux QoS controls.",
            gate_links=("B3", "E2"),
            primary_code_module="code/runtime/pqc_qos.c",
            code_modules=(
                "code/runtime/pqc_qos.c",
                "code/runtime/pqc_admission.c",
                "code/runtime/pqc_scheduler.c",
                "code/experiments/run_qos_sqlite_hero_bundle.py",
                "code/experiments/run_qos_repeated.py",
            ),
            artifact_specs=(
                ArtifactSpec("SQLite hero bundle", "artifacts/validation/qos_sqlite_hero_bundle/*.json"),
                ArtifactSpec("QoS repeated report", "artifacts/reports/qos_repeated/*.json"),
            ),
            required_missing_artifacts=("at least two cgroup/blk-throttle/ionice/BFQ baseline runs",),
            paper_claim_boundary="SQLite p99 improvement must be compared against kernel QoS or scoped as internal controller evidence.",
            negative_guard="No SQLite p99 uniqueness claim before two kernel QoS baselines exist.",
        ),
        ObjectionSpec(
            ident="O06",
            title="POSIX envelope is too broad unless closed by tests or formal rejection",
            status="paper-required",
            reviewer_objection="Shared mmap, rename, directory fsync, and concurrent writes are not closed enough for broad POSIX wording.",
            gate_links=("C1", "C2", "0.14"),
            primary_code_module="code/fs/pqc_posix.c",
            code_modules=(
                "code/fs/pqc_posix.c",
                "code/fs/pqc_namespace.c",
                "code/fs/pqc_file_io.c",
                "code/experiments/run_posix_scope_audit.py",
            ),
            artifact_specs=(
                ArtifactSpec("POSIX scope audit", "artifacts/validation/posix_scope_audit/posix_scope_audit.json"),
                ArtifactSpec("shadow mmap matrix", "artifacts/validation/shadow_mmap_posix/*.json"),
            ),
            required_missing_artifacts=("paper POSIX envelope table with support/formal rejection/paper limitation",),
            paper_claim_boundary="The paper must read as a scoped FUSE secure-storage prototype unless shared mmap and envelope semantics pass.",
            negative_guard="No general-purpose POSIX wording before shared mmap and POSIX envelope evidence close.",
        ),
        ObjectionSpec(
            ident="O07",
            title="crash and power-loss model is not fully bounded",
            status="artifact-required",
            reviewer_objection="SIGKILL-at-cutpoint evidence is not the same as power-loss certification.",
            gate_links=("C3", "C4", "0.8", "0.9"),
            primary_code_module="code/support/pqc_test_hooks.c",
            code_modules=(
                "code/support/pqc_test_hooks.c",
                "code/storage/pqc_recovery.c",
                "code/storage/pqc_epoch_log.c",
                "code/experiments/run_generation_fault_matrix.py",
                "code/experiments/run_daemon_power_fault_matrix.py",
            ),
            artifact_specs=(
                ArtifactSpec("generation fault matrix", "artifacts/validation/generation_fault_matrix/generation_fault_matrix.json"),
                ArtifactSpec("daemon power fault campaign", "artifacts/validation/daemon_power_fault_campaign/daemon_power_fault_campaign.json"),
                ArtifactSpec("epoch replay fault matrix", "artifacts/validation/publication_protocol_fault_matrix/epoch_replay_fault_matrix.json"),
            ),
            required_missing_artifacts=("physical power-loss or explicit unavailable-environment boundary artifact",),
            paper_claim_boundary="Crash claims must be bounded to the strongest retained model: process kill, daemon kill, remount, or physical power cut.",
            negative_guard="No power-loss safe, full crash certification, or crash-certified wording without physical or explicitly bounded evidence.",
        ),
        ObjectionSpec(
            ident="O08",
            title="freshness and rollback resistance remain overclaim-prone",
            status="code-required",
            reviewer_objection="File anchors and TPM NV evidence do not prove persistent PCR-bound rollback resistance.",
            gate_links=("C5", "C6", "D3"),
            primary_code_module="code/storage/pqc_anchor.c",
            code_modules=(
                "code/storage/pqc_anchor.c",
                "code/storage/pqc_anchor_worker.c",
                "code/storage/pqc_checkpoint.c",
                "code/experiments/run_tpm_monotonic_replay.py",
                "code/experiments/run_tpm_pcr_policy_probe.py",
                "code/experiments/build_tpm_freshness_policy_manifest.py",
            ),
            artifact_specs=(
                ArtifactSpec("TPM freshness bundle", "artifacts/validation/tpm_freshness_bundle/*.json"),
                ArtifactSpec("TPM monotonic replay", "artifacts/validation/tpm_monotonic_replay/*.json"),
                ArtifactSpec("TPM PCR policy probe", "artifacts/validation/tpm_pcr_policy_probe/*.json"),
                ArtifactSpec("async Merkle TPM epoch", "artifacts/validation/async_merkle_tpm_epoch/*.json"),
            ),
            required_missing_artifacts=("production async Merkle or committed-prefix root plus TPM epoch evidence",),
            paper_claim_boundary="Freshness text must map to the retained ladder level and stop before PCR-bound rollback resistance unless C6 exists.",
            negative_guard="No TPM rollback resistance, persistent PCR-bound freshness, sealed-key release, or full rollback-resistance wording before Gate C6.",
        ),
        ObjectionSpec(
            ident="O09",
            title="KDF and key lifecycle need a security-facing decision",
            status="code-required",
            reviewer_objection="PBKDF2-HMAC-SHA256 and key lifecycle boundaries need a parameterized decision backed by artifacts.",
            gate_links=("D1", "D2", "D3"),
            primary_code_module="code/crypto/pqc_keyring.c",
            code_modules=(
                "code/crypto/pqc_keyring.c",
                "code/crypto/pqc_crypto.c",
                "code/crypto/pqc_key_lifecycle.c",
                "code/runtime/pqc_rekey.c",
                "code/experiments/build_mount_key_lifecycle_manifest.py",
                "code/experiments/run_keyplane_rekey_workflow.py",
            ),
            artifact_specs=(
                ArtifactSpec("mount key lifecycle", "artifacts/validation/mount_key_lifecycle/*.json"),
                ArtifactSpec("KDF crypto plane", "artifacts/validation/kdf_crypto_plane/*.json"),
            ),
            required_missing_artifacts=("Argon2id/scrypt implementation or PBKDF2 parameter/salt/offline-attack benchmark artifact",),
            paper_claim_boundary="Security text must state the KDF, salt, parameters, offline attacker model, and PQC/data-plane separation.",
            negative_guard="No offline-attack resistance or strong password-hardening claim while PBKDF2 remains unexplained.",
        ),
        ObjectionSpec(
            ident="O10",
            title="GPU and Jetson optimization claims need a ladder",
            status="negative-claim-only",
            reviewer_objection=(
                "Pinned memory, UVM, dma-buf, GPUDirect, stream priority, and Jetson power claims need local probes, "
                "production traces, and official-document boundaries."
            ),
            gate_links=("0.10", "0.11", "0.12", "0.18", "A4", "E5"),
            primary_code_module="code/gpu/cuda_aead.cu",
            code_modules=(
                "code/gpu/cuda_aead.cu",
                "code/gpu/cuda_pqc.cu",
                "code/experiments/run_jetson_memory_contract.py",
                "code/experiments/run_jetson_power_thermal_contract.py",
                "code/experiments/run_cuda_qos_contract.py",
                "code/experiments/build_jetson_optimization_ladder.py",
            ),
            artifact_specs=(
                ArtifactSpec("Jetson memory contract", "artifacts/validation/jetson_memory_contract/*.json"),
                ArtifactSpec("Jetson power thermal contract", "artifacts/validation/jetson_power_thermal_contract/*.json"),
                ArtifactSpec("CUDA QoS contract", "artifacts/validation/cuda_qos_contract/*.json"),
                ArtifactSpec("Jetson optimization ladder", "artifacts/validation/jetson_optimization_ladder/*.json"),
            ),
            required_missing_artifacts=("production mounted-path benefit traces before any NVIDIA term becomes a paper mechanism",),
            paper_claim_boundary="NVIDIA terms may appear only at their retained evidence level: API available, probe passed, diagnostic, or non-claim.",
            negative_guard="No NVMe-to-UVM, GPUDirect, dma-buf zero-copy, CUDA isolation, or stream-priority QoS claim without production evidence.",
        ),
        ObjectionSpec(
            ident="O11",
            title="hero workload generality is too narrow",
            status="code-required",
            reviewer_objection="SQLite/FUSE QoS alone is not enough for SOSP/OSDI workload generality.",
            gate_links=("E1", "E2", "E3", "E4", "E5"),
            primary_code_module="code/experiments/run_qos_sqlite_hero_bundle.py",
            code_modules=(
                "code/experiments/run_qos_sqlite_hero_bundle.py",
                "code/experiments/build_hero_result_contract.py",
                "code/experiments/build_workload_diversity_matrix.py",
            ),
            artifact_specs=(
                ArtifactSpec("hero result contract", "artifacts/validation/hero_result_contract/*.json"),
                ArtifactSpec("workload diversity matrix", "artifacts/validation/workload_diversity_matrix/*.json"),
                ArtifactSpec("second macrobenchmark", "artifacts/validation/second_macrobenchmark/*.json"),
            ),
            required_missing_artifacts=("RocksDB/WAL, checkpointing, or secure inference-log macrobenchmark runner and raw logs",),
            paper_claim_boundary="SQLite can remain the primary hero, but generality requires a second macrobenchmark or downgraded venue ambition.",
            negative_guard="No broad workload-general or SOSP/OSDI-ready claim before a second macrobenchmark exists.",
        ),
        ObjectionSpec(
            ident="O12",
            title="paper narrative can overclaim the evidence",
            status="paper-required",
            reviewer_objection="The paper can sound broader than the retained production evidence.",
            gate_links=("F1", "F2", "F3", "F4", "F5"),
            primary_code_module="code/experiments/build_paper_spine_gate.py",
            code_modules=(
                "code/experiments/build_paper_spine_gate.py",
                "code/experiments/build_first_two_pages_thesis_audit.py",
                "code/experiments/build_recurring_review_elimination_audit.py",
                "code/experiments/build_evaluation_completeness_matrix.py",
            ),
            artifact_specs=(
                ArtifactSpec("paper spine gate", "artifacts/reports/paper_spine_gate/*.json"),
                ArtifactSpec("first two pages audit", "artifacts/validation/first_two_pages_thesis_audit/*.json"),
                ArtifactSpec("recurring review audit", "artifacts/validation/recurring_review_elimination_audit/*.json"),
            ),
            required_missing_artifacts=("paper text pass with dangerous phrases absent, negated, or evidence-backed",),
            paper_claim_boundary="Paper body must prioritize thesis, hard problem, mechanism, result, and loss case, not artifact-ledger narration.",
            negative_guard="Dangerous phrases must be absent, explicitly negated, or tied to a closed gate.",
        ),
        ObjectionSpec(
            ident="O13",
            title="FUSE and fdatasync storm need amortization or measured rejection",
            status="artifact-required",
            reviewer_objection="FUSE context switching and per-write fdatasync can dominate the path and must be amortized or honestly measured.",
            gate_links=("A5", "0.9", "0.15", "0.16"),
            primary_code_module="code/storage/pqc_epoch_publish.c",
            code_modules=(
                "code/storage/pqc_epoch_publish.c",
                "code/storage/pqc_parallel_commit.c",
                "code/fs/pqc_file_io.c",
                "code/experiments/run_ebpf_iouring_scope_audit.py",
                "code/experiments/bench_io_uring_ebpf.cu",
            ),
            artifact_specs=(
                ArtifactSpec("eBPF/io_uring scope audit", "artifacts/validation/ebpf_iouring_scope_audit/ebpf_iouring_scope_audit.json"),
                ArtifactSpec("publication closeout", "artifacts/validation/publication_protocol_fault_matrix/publication_protocol_closeout.json"),
                ArtifactSpec("parallel commit closure", "artifacts/validation/parallel_commit_contract/parallel_commit_closure_audit.json"),
            ),
            required_missing_artifacts=("vfs_ebpf_fdatasync_storm context-switch/syscall/fdatasync/tail-latency breakdown",),
            paper_claim_boundary="The design may claim fdatasync amortization only for the measured epoch path; eBPF/io_uring bypass remains a non-claim.",
            negative_guard="No eBPF passthrough, io_uring completion bypass, or async epoch fdatasync claim without production evidence.",
        ),
        ObjectionSpec(
            ident="O14",
            title="attacker model and deployment claims must stay scoped",
            status="negative-claim-only",
            reviewer_objection=(
                "GPU side channels, multi-tenant attackers, privileged local attackers, and deployment readiness are not implemented defenses."
            ),
            gate_links=("D4", "F4", "F5"),
            primary_code_module="code/experiments/run_sidechannel_eval.py",
            code_modules=(
                "code/experiments/run_sidechannel_eval.py",
                "code/experiments/run_tvla_leakage_eval.py",
                "code/experiments/build_integrity_comparison_manifest.py",
            ),
            artifact_specs=(
                ArtifactSpec("side-channel diagnostics", "artifacts/validation/sidechannel_eval/*.json"),
                ArtifactSpec("TVLA diagnostics", "artifacts/validation/tvla_leakage_eval/*.json"),
            ),
            required_missing_artifacts=("claim lint output proving out-of-scope language remains explicit",),
            paper_claim_boundary="Security scope must exclude compromised kernel/driver/FUSE, privileged local attackers, multi-tenant isolation, and GPU side-channel defense.",
            negative_guard="No side-channel protection, multi-tenant defense, compromised-kernel defense, deployment-ready, or ready-for-production wording.",
        ),
    ]


def build_objection(spec: ObjectionSpec) -> dict[str, Any]:
    retained, missing = classify_artifacts(spec.artifact_specs)
    missing.extend({"label": "required closure artifact", "path": item} for item in spec.required_missing_artifacts)
    return {
        "id": spec.ident,
        "title": spec.title,
        "status": spec.status,
        "reviewer_objection": spec.reviewer_objection,
        "gate_links": list(spec.gate_links),
        "primary_code_module": spec.primary_code_module,
        "code_modules": module_status(spec.code_modules),
        "retained_artifacts": retained,
        "missing_artifacts": missing,
        "paper_claim_boundary": spec.paper_claim_boundary,
        "negative_guard": spec.negative_guard,
    }


def freeze_summary() -> dict[str, Any]:
    freeze = read_json(FREEZE)
    git = freeze.get("git", {})
    patch = freeze.get("patch_ownership", {})
    return {
        "source": relpath(FREEZE),
        "exists": FREEZE.exists(),
        "generated_utc": freeze.get("generated_utc"),
        "git_branch": git.get("branch"),
        "diff_name_only_count": len(git.get("diff_name_only", [])) if isinstance(git.get("diff_name_only"), list) else None,
        "untracked_files_count": git.get("untracked_files_count"),
        "code_refactor_strategy_absent": freeze.get("authoritative_checklist", {}).get("code_refactor_strategy_absent"),
        "dirty_sidecar_sync_dedup_patch": patch.get("dirty_sidecar_sync_dedup_patch", {}).get("classification"),
        "parent_checklist_closed": freeze.get("parent_checklist_closed"),
        "paper_text_status": freeze.get("paper_text_status"),
    }


def artifact_verdict(objections: list[dict[str, Any]]) -> dict[str, Any]:
    invalid_statuses = [
        item["id"]
        for item in objections
        if item.get("status") not in STATUS_VALUES
    ]
    missing_primary_modules = [
        item["id"]
        for item in objections
        if not path_exists(str(item.get("primary_code_module", "")))
    ]
    missing_boundary = [
        item["id"]
        for item in objections
        if not item.get("paper_claim_boundary") or not item.get("negative_guard")
    ]
    covered_topics = {
        "filesystem_viability": any("A1" in item["gate_links"] for item in objections),
        "fscrypt_dmcrypt_baselines": any(item["id"] == "O03" for item in objections)
        and any(item["id"] == "O04" for item in objections),
        "kernel_qos": any("B3" in item["gate_links"] for item in objections),
        "posix_envelope": any("C1" in item["gate_links"] for item in objections),
        "crash_model": any("C4" in item["gate_links"] for item in objections),
        "freshness_ladder": any("C5" in item["gate_links"] for item in objections),
        "kdf": any("D1" in item["gate_links"] for item in objections),
        "gpu_jetson": any("0.18" in item["gate_links"] for item in objections),
        "hero_generality": any("E3" in item["gate_links"] for item in objections),
        "narrative_overclaiming": any("F4" in item["gate_links"] for item in objections),
    }
    status_counts = {status: 0 for status in sorted(STATUS_VALUES)}
    for item in objections:
        status_counts[item["status"]] = status_counts.get(item["status"], 0) + 1
    return {
        "overall_pass": (
            not invalid_statuses
            and not missing_primary_modules
            and not missing_boundary
            and all(covered_topics.values())
        ),
        "status_counts": status_counts,
        "invalid_status_ids": invalid_statuses,
        "missing_primary_module_ids": missing_primary_modules,
        "missing_boundary_ids": missing_boundary,
        "covered_topics": covered_topics,
    }


def write_markdown(payload: dict[str, Any]) -> None:
    lines = [
        "# Review Objection Map",
        "",
        "This Gate 0.2-S0 artifact maps repeated review objections to the concrete code, retained or missing evidence, paper boundary, and negative guard that prevent another loop of paper-only fixes.",
        "",
        f"- Generated: `{payload['generated_utc']}`",
        f"- Worktree freeze source: `{payload['freeze_context']['source']}`",
        f"- Dirty-sidecar sync dedup patch: `{payload['freeze_context']['dirty_sidecar_sync_dedup_patch']}`",
        f"- Parent checklist closed by this artifact: `{payload['parent_checklist_closed']}`",
        "",
        "## Status Counts",
        "",
    ]
    for status, count in payload["artifact_verdict"]["status_counts"].items():
        lines.append(f"- `{status}`: {count}")
    lines.extend(
        [
            "",
            "## Objections",
            "",
            "| ID | Status | Primary module | Gates | Missing closure evidence | Negative guard |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
    )
    for item in payload["objections"]:
        missing = "; ".join(entry["path"] for entry in item["missing_artifacts"][:3])
        if len(item["missing_artifacts"]) > 3:
            missing += f"; +{len(item['missing_artifacts']) - 3} more"
        gates = ", ".join(item["gate_links"])
        guard = item["negative_guard"].replace("|", "/")
        lines.append(
            f"| {item['id']} | `{item['status']}` | `{item['primary_code_module']}` | {gates} | {missing} | {guard} |"
        )
    lines.extend(
        [
            "",
            "## Immediate Execution Consequence",
            "",
            "The next row should refresh the venue gate. Broad checklist boxes remain unchecked because this artifact is an inventory/proof cursor, not code, paper text, and negative-claim closure for any parent gate.",
            "",
        ]
    )
    MD_OUT.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    objections = [build_objection(spec) for spec in specs()]
    payload = {
        "schema_version": 2,
        "generated_by": relpath(Path(__file__)),
        "generated_utc": now_utc(),
        "scope": "Gate 0.2-S0 recurring review-objection map against the frozen worktree and active gates.",
        "status_values": sorted(STATUS_VALUES),
        "freeze_context": freeze_summary(),
        "objection_count": len(objections),
        "objections": objections,
        "artifact_verdict": artifact_verdict(objections),
        "paper_text_status": "not_updated",
        "parent_checklist_closed": False,
        "next_cursor": {
            "row_id": "0.3-S0",
            "reason": "Refresh SOSP/OSDI/ATC/FAST venue readiness against the same frozen worktree and evidence map.",
        },
    }
    JSON_OUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown(payload)
    print(json.dumps(payload["artifact_verdict"], indent=2, sort_keys=True))
    return 0 if payload["artifact_verdict"]["overall_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
