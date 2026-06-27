#!/usr/bin/env python3
"""Build the mount-key lifecycle scope manifest.

This manifest closes the lifecycle-definition checklist item by tying the
implemented mounted path to retained artifacts and by forcing the paper to state
the credential boundary.  It intentionally does not upgrade the prototype to a
hardware-backed credential service or a transactional key-rotation system.
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "artifacts" / "validation" / "mount_key_lifecycle"

PQC_FUSE_C = ROOT / "pqc_fuse.c"
PQC_FILE_KEY_C = ROOT / "pqc_file_key.c"
PQC_FILE_KEY_H = ROOT / "pqc_file_key.h"
CMAKE = ROOT / "CMakeLists.txt"
DESIGN_TEX = ROOT / "Paper" / "3_Design.tex"
IMPLEMENTATION_TEX = ROOT / "Paper" / "7_Implementation_Details.tex"
SECURITY_TEX = ROOT / "Paper" / "8_Security_Analysis.tex"
DISCUSSION_TEX = ROOT / "Paper" / "10_Discussion_and_Limitations.tex"
EVALUATION_TEX = ROOT / "Paper" / "4_Evaluation.tex"

FUSE_ROUNDTRIP = ROOT / "artifacts" / "validation" / "fuse_roundtrip.json"
FUSE_TAMPER = ROOT / "artifacts" / "validation" / "fuse_tamper_rejection.json"
KEYPLANE = ROOT / "artifacts" / "validation" / "keyplane_rekey_workflow" / "keyplane_rekey_workflow.json"
KEYPLANE_METHOD = ROOT / "artifacts" / "validation" / "keyplane_rekey_methodology" / "keyplane_rekey_workflow.json"
GENERATION_MATRIX = ROOT / "artifacts" / "validation" / "generation_fault_matrix" / "generation_fault_matrix.json"
TPM_POLICY = ROOT / "artifacts" / "validation" / "tpm_freshness_policy" / "tpm_freshness_policy.json"

REQUIRED_DECISIONS = {
    "password_derived_mount_key_boundary",
    "hardware_backed_credential_release_plan",
    "key_rotation",
    "envelope_rewrap",
    "epoch_counter_interaction",
    "recovery_after_failed_rewrap",
    "rollback_behavior_for_old_envelopes",
}


def relpath(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(relpath(path))
    return json.loads(path.read_text(encoding="utf-8"))


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def paper_text() -> str:
    return "\n".join(
        read(path)
        for path in (
            DESIGN_TEX,
            IMPLEMENTATION_TEX,
            SECURITY_TEX,
            DISCUSSION_TEX,
            EVALUATION_TEX,
        )
        if path.exists()
    )


def source_segment(text: str, start: str, end: str) -> str:
    start_idx = text.find(start)
    if start_idx < 0:
        return ""
    end_idx = text.find(end, start_idx + len(start))
    if end_idx < 0:
        return text[start_idx:]
    return text[start_idx:end_idx]


def mode_names(report: dict[str, Any]) -> set[str]:
    return {str(mode.get("mode")) for mode in report.get("modes", [])}


def all_modes_acceptable(report: dict[str, Any]) -> bool:
    return bool(report.get("modes")) and all(mode.get("acceptable") is True for mode in report.get("modes", []))


def methodology_modes_acceptable(report: dict[str, Any]) -> bool:
    summaries = report.get("mode_summaries", [])
    return bool(summaries) and all(summary.get("all_acceptable") is True for summary in summaries)


def generation_case(report: dict[str, Any], case: str) -> dict[str, Any] | None:
    for row in report.get("rows", []):
        if row.get("case") == case:
            return row
    return None


def build_manifest() -> dict[str, Any]:
    fuse_src = read(PQC_FUSE_C)
    file_key_src = read(PQC_FILE_KEY_C)
    file_key_hdr = read(PQC_FILE_KEY_H)
    cmake = read(CMAKE)
    rekey_src = source_segment(fuse_src, "static void *rekey_worker_main", "static int pqc_subsystem_init")
    metadata_load_src = source_segment(fuse_src, "static int metadata_load", "static int logical_size_load")
    metadata_store_src = source_segment(fuse_src, "static int metadata_store", "static int metadata_load")

    roundtrip = load_json(FUSE_ROUNDTRIP)
    tamper = load_json(FUSE_TAMPER)
    keyplane = load_json(KEYPLANE)
    keyplane_method = load_json(KEYPLANE_METHOD)
    generation = load_json(GENERATION_MATRIX)
    tpm_policy = load_json(TPM_POLICY)

    stale_file = generation_case(generation, "stale_snapshot_replay_file_anchor_negative_control") or {}
    stale_tpm = generation_case(generation, "stale_snapshot_replay_tpm_anchor_existing_artifact") or {}

    fuse_target_excludes_file_key = (
        re.search(
            r"set\s*\(\s*PQC_FUSE_SOURCES\s+pqc_fuse\.c\s+pqc_anchor\.c\s+pqc_admission\.c\s*\)",
            cmake,
        )
        is not None
    )

    source_checks = {
        "mount_password_required": "PQC_MASTER_PASSWORD is required" in fuse_src,
        "pbkdf2_sha256_master_key": "PKCS5_PBKDF2_HMAC" in fuse_src and "EVP_sha256()" in fuse_src and "g_master_key" in fuse_src,
        "random_dek_and_file_id_on_create": "RAND_bytes(ss, sizeof(ss))" in fuse_src and "RAND_bytes((unsigned char *)&fid, sizeof(fid))" in fuse_src,
        "metadata_store_wraps_under_mount_key": "wrap_shared_secret(ss, ss_len, file_id, 0, meta.wrapped_ss)" in metadata_store_src,
        "metadata_store_hmac_authenticates_envelope": "HMAC(EVP_sha256(), g_master_key" in metadata_store_src,
        "metadata_load_hmac_before_unwrap": "CRYPTO_memcmp(digest, meta.digest" in metadata_load_src
        and metadata_load_src.find("CRYPTO_memcmp") < metadata_load_src.find("unwrap_shared_secret"),
        "metadata_load_rejects_bad_hmac": "return -EKEYREJECTED" in metadata_load_src,
        "rekey_worker_refreshes_open_file_secret": "rekey_worker_main" in rekey_src
        and "memcpy(g_fd_ctx[idx].ss" in rekey_src
        and "metadata_store(g_fd_ctx[idx].marker_path" in rekey_src,
        "rekey_worker_updates_runtime_epoch": "g_fd_ctx[idx].key_epoch++" in rekey_src,
        "rekey_rewrap_not_transactional": "metadata_store(g_fd_ctx[idx].marker_path" in rekey_src
        and "REKEY WORKER: batched rekey FAILED" in rekey_src,
        "mounted_target_excludes_legacy_file_key_helper": fuse_target_excludes_file_key,
        "legacy_epoch_helper_is_not_mounted_path": "pqc_file_key_verify_epoch" in file_key_src
        and "Rotation policy enforces epoch-based access control" in file_key_hdr
        and fuse_target_excludes_file_key,
    }

    artifact_checks = {
        "clean_remount_passes": roundtrip.get("normal_remount") == "pass" and roundtrip.get("raw_plaintext_search") == "pass",
        "tampered_envelope_rejected": tamper.get("pass") is True
        and tamper.get("metadata_xattr") == "user.pqc_metadata"
        and tamper.get("open_rejected") is True,
        "keyplane_workflow_passes": keyplane.get("overall_pass") is True
        and keyplane.get("maintenance_visible_benefit") is True
        and all_modes_acceptable(keyplane)
        and mode_names(keyplane) == {"cpu_only", "gpu_batch", "policy_fallback"},
        "keyplane_methodology_passes": keyplane_method.get("overall_pass") is True
        and methodology_modes_acceptable(keyplane_method),
        "generation_matrix_passes": generation.get("overall_pass") is True
        and generation.get("no_silent_corruption") is True,
        "file_anchor_old_state_is_negative_control": stale_file.get("negative_control") is True
        and stale_file.get("oracle_verdict") == "previous_committed"
        and stale_file.get("acceptable") is True,
        "tpm_old_state_fails_closed": stale_tpm.get("oracle_verdict") == "fail_closed"
        and stale_tpm.get("acceptable") is True,
        "tpm_policy_scopes_no_credential_release": tpm_policy.get("overall_pass") is True
        and "no hardware-backed credential release" in set(tpm_policy.get("non_claims", [])),
    }

    lifecycle_rows = [
        {
            "decision": "password_derived_mount_key_boundary",
            "current_evidence": (
                "pqc_subsystem_init() requires PQC_MASTER_PASSWORD, derives g_master_key "
                "with PBKDF2-HMAC-SHA256, and pqc_create() generates a random 256-bit "
                "per-file DEK plus file identifier."
            ),
            "policy_or_scope": (
                "The mount key is the prototype root credential.  This closes only the "
                "storage-format correctness boundary, not deployed credential protection."
            ),
            "paper_gate_phrase": "mount key is password-derived and never hardware-released",
        },
        {
            "decision": "hardware_backed_credential_release_plan",
            "current_evidence": (
                "The TPM policy manifest explicitly records no hardware-backed credential "
                "release, and the mounted FUSE source has no TPM/PCR release path for "
                "g_master_key."
            ),
            "policy_or_scope": (
                "Hardware-backed release remains out of scope; TPM evidence is limited to "
                "freshness-anchor behavior."
            ),
            "paper_gate_phrase": "no hardware-backed credential release path for the mount key",
        },
        {
            "decision": "key_rotation",
            "current_evidence": (
                "The rekey worker batches open file descriptors, encapsulates fresh key "
                "material on the admitted executor, installs it in the open fd context, "
                "increments an in-memory key_epoch, and persists a new envelope."
            ),
            "policy_or_scope": (
                "This is open-file DEK refresh.  It is not deployed mount credential "
                "rotation, administrator key rollover, or a persistent KEM hierarchy."
            ),
            "paper_gate_phrase": "rekey is open-file DEK refresh rather than deployed credential rotation",
        },
        {
            "decision": "envelope_rewrap",
            "current_evidence": (
                "metadata_store() masks the per-file DEK under the mount key and file id "
                "and HMAC-authenticates the envelope; metadata_load() verifies the HMAC "
                "before unwrapping.  The retained key-plane workflow refreshes 1,024 open "
                "files per mode, and the tamper regression rejects a corrupted metadata "
                "xattr with EKEYREJECTED."
            ),
            "policy_or_scope": (
                "Envelope rewrap evidence applies to mounted open files and authenticated "
                "open/remount behavior only."
            ),
            "paper_gate_phrase": "HMAC envelope rewrap for open files",
        },
        {
            "decision": "epoch_counter_interaction",
            "current_evidence": (
                "The mounted path has an in-memory key_epoch incremented by the rekey "
                "worker.  The older pqc_file_key epoch/grace helper is present in source "
                "but is not part of the pqc_fuse CMake target."
            ),
            "policy_or_scope": (
                "Epochs are runtime bookkeeping for the evaluated path, not a persisted "
                "anti-rollback journal or stale-handle proof."
            ),
            "paper_gate_phrase": "runtime key epoch is in-memory bookkeeping, not a persistent anti-rollback journal",
        },
        {
            "decision": "recovery_after_failed_rewrap",
            "current_evidence": (
                "No retained artifact exercises an interrupted envelope rewrap, and the "
                "rekey worker does not implement a transactional rewrap journal.  Existing "
                "retained evidence covers HMAC fail-closed behavior on later open and clean "
                "remount after normal writes."
            ),
            "policy_or_scope": (
                "The prototype makes no transactional recovery claim for failed rewrap; "
                "future work must add a cut-point campaign before making that claim."
            ),
            "paper_gate_phrase": "no transactional rewrap recovery claim",
        },
        {
            "decision": "rollback_behavior_for_old_envelopes",
            "current_evidence": (
                "The generation fault matrix records file-anchor stale snapshot replay as "
                "an expected negative control and records the existing TPM stale snapshot "
                "artifact as fail closed."
            ),
            "policy_or_scope": (
                "A password-derived envelope by itself is replayable with a whole backing "
                "snapshot.  Rollback resistance is claimed only when generation/checkpoint "
                "and external-anchor evidence reject stale state."
            ),
            "paper_gate_phrase": "password-derived envelope alone is not rollback resistance",
        },
    ]

    paper = paper_text()
    paper_gates = {row["decision"]: row["paper_gate_phrase"] in paper for row in lifecycle_rows}
    lifecycle_checks = {
        "all_required_decisions_present": {row["decision"] for row in lifecycle_rows} == REQUIRED_DECISIONS,
        "all_rows_have_evidence_and_scope": all(row["current_evidence"] and row["policy_or_scope"] for row in lifecycle_rows),
        "source_checks_pass": all(source_checks.values()),
        "artifact_checks_pass": all(artifact_checks.values()),
        "paper_covers_all_lifecycle_decisions": all(paper_gates.values()),
    }

    return {
        "schema_version": 1,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "overall_pass": all(lifecycle_checks.values()),
        "scope": (
            "Mount-key lifecycle scope artifact.  It supports authenticated "
            "storage-format correctness, clean remount, tamper rejection, and mounted "
            "open-file envelope refresh.  It does not claim hardware-backed credential "
            "release, deployed credential rotation, transactional rewrap recovery, or "
            "credential-only rollback resistance."
        ),
        "source_artifacts": [
            relpath(PQC_FUSE_C),
            relpath(PQC_FILE_KEY_C),
            relpath(PQC_FILE_KEY_H),
            relpath(CMAKE),
            relpath(FUSE_ROUNDTRIP),
            relpath(FUSE_TAMPER),
            relpath(KEYPLANE),
            relpath(KEYPLANE_METHOD),
            relpath(GENERATION_MATRIX),
            relpath(TPM_POLICY),
        ],
        "lifecycle_rows": lifecycle_rows,
        "source_checks": source_checks,
        "artifact_checks": artifact_checks,
        "paper_gates": paper_gates,
        "checks": lifecycle_checks,
        "non_claims": [
            "no hardware-backed credential release path for the mount key",
            "no deployed mount credential rotation",
            "no persistent KEM hierarchy",
            "no persistent epoch anti-rollback journal in the mounted path",
            "no transactional rewrap recovery claim",
            "no credential-only rollback resistance",
            "no recovery after a lost mount credential",
        ],
    }


def markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Mount-Key Lifecycle Manifest",
        "",
        f"- Overall pass: `{str(report['overall_pass']).lower()}`",
        f"- Scope: {report['scope']}",
        "",
        "## Checks",
        "",
    ]
    for key, value in report["checks"].items():
        lines.append(f"- `{key}`: `{str(value).lower()}`")
    lines.extend(["", "## Source Checks", ""])
    for key, value in report["source_checks"].items():
        lines.append(f"- `{key}`: `{str(value).lower()}`")
    lines.extend(["", "## Artifact Checks", ""])
    for key, value in report["artifact_checks"].items():
        lines.append(f"- `{key}`: `{str(value).lower()}`")
    lines.extend(["", "## Lifecycle Rows", ""])
    for row in report["lifecycle_rows"]:
        lines.extend(
            [
                f"### {row['decision']}",
                "",
                f"- Current evidence: {row['current_evidence']}",
                f"- Policy or scope: {row['policy_or_scope']}",
                f"- Paper gate phrase: `{row['paper_gate_phrase']}`",
                "",
            ]
        )
    lines.extend(["## Non-Claims", ""])
    for item in report["non_claims"]:
        lines.append(f"- {item}")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()

    out = args.out if args.out.is_absolute() else ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    report = build_manifest()
    json_path = out / "mount_key_lifecycle.json"
    md_path = out / "mount_key_lifecycle.md"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md_path.write_text(markdown(report), encoding="utf-8")
    print(
        json.dumps(
            {
                "overall_pass": report["overall_pass"],
                "json": relpath(json_path),
                "markdown": relpath(md_path),
                "checks": report["checks"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    if args.require_complete and not report["overall_pass"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
