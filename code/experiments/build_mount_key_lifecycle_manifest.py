#!/usr/bin/env python3
"""Build the key-management lifecycle closeout manifest.

The source of truth is the production key-lifecycle table in
``code/crypto/pqc_key_lifecycle.c``.  This builder ties that table to the
current KDF/key-plane/freshness evidence and to paper wording that prevents
hardware-backed credential or rollback-resistance overclaims.
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
CODE = ROOT / "code"
DEFAULT_OUT = ROOT / "artifacts" / "validation" / "mount_key_lifecycle"

KEY_LIFECYCLE_C = CODE / "crypto" / "pqc_key_lifecycle.c"
KEY_LIFECYCLE_H = CODE / "crypto" / "pqc_key_lifecycle.h"
KEYRING_C = CODE / "crypto" / "pqc_keyring.c"
CRYPTO_C = CODE / "crypto" / "pqc_crypto.c"
REKEY_C = CODE / "runtime" / "pqc_rekey.c"
RUNTIME_C = CODE / "runtime" / "pqc_runtime.c"
ANCHOR_C = CODE / "storage" / "pqc_anchor.c"
CHECKPOINT_C = CODE / "storage" / "pqc_checkpoint.c"
SELFTEST_C = CODE / "support" / "pqc_selftest.c"
MAIN_C = CODE / "frontend" / "pqc_main.c"
CRYPTO_SOURCES = CODE / "crypto" / "sources.cmake"

PAPER_FILES = [
    ROOT / "Paper" / "3_Design.tex",
    ROOT / "Paper" / "4_Evaluation.tex",
    ROOT / "Paper" / "8_Security_Analysis.tex",
    ROOT / "Paper" / "10_Discussion_and_Limitations.tex",
    ROOT / "Paper" / "6_Conclusion.tex",
]

FUSE_ROUNDTRIP = ROOT / "artifacts" / "validation" / "fuse_roundtrip.json"
FUSE_TAMPER = ROOT / "artifacts" / "validation" / "fuse_tamper_rejection.json"
KDF_VERDICT = (
    ROOT / "artifacts" / "validation" / "kdf_crypto_plane" /
    "kdf_current_state_verdict.json"
)
CRYPTO_PLANE = (
    ROOT / "artifacts" / "validation" / "crypto_plane_separation" /
    "crypto_plane_claim_guard.json"
)
KEYPLANE = (
    ROOT / "artifacts" / "validation" / "keyplane_rekey_workflow" /
    "keyplane_rekey_workflow.json"
)
GENERATION_MATRIX = (
    ROOT / "artifacts" / "validation" / "generation_fault_matrix" /
    "generation_fault_matrix.json"
)
FRESHNESS_LADDER = (
    ROOT / "artifacts" / "validation" / "freshness_ladder_claim_guard" /
    "freshness_ladder_claim_guard.json"
)
TPM_POLICY = (
    ROOT / "artifacts" / "validation" / "tpm_freshness_policy" /
    "tpm_freshness_policy.json"
)

REQUIRED_MATERIALS = {
    "PQC_KEY_MATERIAL_MOUNT_KEY": "mount-key",
    "PQC_KEY_MATERIAL_FILE_ENVELOPE_SECRET": "per-file-envelope-secret",
    "PQC_KEY_MATERIAL_DATA_BLOCK_KEY": "aes-gcm-data-block-key",
    "PQC_KEY_MATERIAL_MOUNT_KEM_KEYPAIR": "mount-lifetime-kem-keypair",
    "PQC_KEY_MATERIAL_FRESHNESS_ANCHOR": "committed-prefix-freshness-anchor",
    "PQC_KEY_MATERIAL_TPM_PCR_POLICY":
        "persistent-tpm-pcr-key-release-policy",
}

REQUIRED_PAPER_PHRASES = {
    "scrypt_new_root": "New roots use OpenSSL scrypt",
    "mount_key_not_hardware_released":
        "The password-derived mount key is never hardware-released",
    "no_hardware_credential_release":
        "There is no hardware-backed credential release path for the mount key",
    "rekey_boundary":
        "The rekey is open-file DEK refresh rather than deployed credential rotation",
    "runtime_epoch_boundary":
        "runtime key epoch is in-memory bookkeeping",
    "no_transactional_rewrap": "There is no transactional rewrap recovery claim",
    "envelope_not_rollback": "password-derived envelope alone is not rollback resistance",
    "bulk_data_boundary": "does not encrypt ordinary file blocks",
    "persistent_pcr_nonclaim": "not persistent PCR-bound key release",
}


def relpath(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(relpath(path))
    return json.loads(path.read_text(encoding="utf-8"))


def paper_text() -> str:
    return "\n".join(read(path) for path in PAPER_FILES if path.exists())


def source_segment(text: str, start: str, end: str) -> str:
    start_idx = text.find(start)
    if start_idx < 0:
        return ""
    end_idx = text.find(end, start_idx + len(start))
    if end_idx < 0:
        return text[start_idx:]
    return text[start_idx:end_idx]


def extract_lifecycle_rows(src: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for match in re.finditer(r"\{\s*\.material\s*=\s*(PQC_KEY_MATERIAL_[A-Z_]+),([\s\S]*?)\n\s*\}", src):
        material = match.group(1)
        body = match.group(2)
        row: dict[str, str] = {"material": material}
        for field in (
            "name",
            "owner_module",
            "producer",
            "storage",
            "protector",
            "rotation",
            "recovery",
            "failure_boundary",
        ):
            field_match = re.search(rf"\.{field}\s*=\s*\"([^\"]*)\"", body)
            row[field] = field_match.group(1) if field_match else ""
        plane_match = re.search(r"\.plane\s*=\s*(PQC_KEY_LIFECYCLE_PLANE_[A-Z]+)", body)
        status_match = re.search(r"\.status\s*=\s*(PQC_KEY_LIFECYCLE_STATUS_[A-Z_]+)", body)
        hw_match = re.search(r"\.hardware_released\s*=\s*([01])", body)
        data_match = re.search(r"\.data_plane_critical\s*=\s*([01])", body)
        row["plane"] = plane_match.group(1) if plane_match else ""
        row["status"] = status_match.group(1) if status_match else ""
        row["hardware_released"] = hw_match.group(1) if hw_match else ""
        row["data_plane_critical"] = data_match.group(1) if data_match else ""
        rows.append(row)
    return rows


def generation_case(report: dict[str, Any], case: str) -> dict[str, Any]:
    for row in report.get("rows", []):
        if row.get("case") == case:
            return row
    return {}


def build_manifest() -> dict[str, Any]:
    key_lifecycle_src = read(KEY_LIFECYCLE_C)
    key_lifecycle_hdr = read(KEY_LIFECYCLE_H)
    keyring_src = read(KEYRING_C)
    crypto_src = read(CRYPTO_C)
    rekey_src = read(REKEY_C)
    runtime_src = read(RUNTIME_C)
    anchor_src = read(ANCHOR_C)
    checkpoint_src = read(CHECKPOINT_C)
    selftest_src = read(SELFTEST_C)
    main_src = read(MAIN_C)
    crypto_sources = read(CRYPTO_SOURCES)
    paper = paper_text()

    metadata_load_src = source_segment(
        keyring_src, "int pqc_keyring_metadata_load", "__end_of_file__"
    )
    metadata_store_src = source_segment(
        keyring_src, "int pqc_keyring_metadata_store",
        "int pqc_keyring_metadata_load"
    )

    roundtrip = load_json(FUSE_ROUNDTRIP)
    tamper = load_json(FUSE_TAMPER)
    kdf = load_json(KDF_VERDICT)
    crypto_plane = load_json(CRYPTO_PLANE)
    keyplane = load_json(KEYPLANE)
    generation = load_json(GENERATION_MATRIX)
    freshness = load_json(FRESHNESS_LADDER)
    tpm_policy = load_json(TPM_POLICY)

    stale_file = generation_case(
        generation, "stale_snapshot_replay_file_anchor_negative_control"
    )
    stale_tpm = generation_case(
        generation, "stale_snapshot_replay_tpm_anchor_existing_artifact"
    )

    lifecycle_rows = extract_lifecycle_rows(key_lifecycle_src)
    rows_by_material = {row["material"]: row for row in lifecycle_rows}

    lifecycle_table_checks = {
        "all_required_materials_present":
            set(rows_by_material) == set(REQUIRED_MATERIALS),
        "all_names_match_materials": all(
            rows_by_material.get(material, {}).get("name") == name
            for material, name in REQUIRED_MATERIALS.items()
        ),
        "all_rows_have_owner_producer_storage_protector_recovery": all(
            all(row.get(field) for field in (
                "owner_module", "producer", "storage", "protector",
                "rotation", "recovery", "failure_boundary",
            ))
            for row in lifecycle_rows
        ),
        "data_key_is_only_data_plane_critical": all(
            (row["material"] == "PQC_KEY_MATERIAL_DATA_BLOCK_KEY") ==
            (row.get("data_plane_critical") == "1")
            for row in lifecycle_rows
        ),
        "tpm_pcr_policy_is_non_claim":
            rows_by_material.get(
                "PQC_KEY_MATERIAL_TPM_PCR_POLICY", {}
            ).get("status") == "PQC_KEY_LIFECYCLE_STATUS_NON_CLAIM",
        "no_material_is_hardware_released":
            all(row.get("hardware_released") == "0" for row in lifecycle_rows),
        "self_test_validates_table":
            "pqc_key_lifecycle_self_test" in key_lifecycle_src and
            "entry->hardware_released" in key_lifecycle_src and
            "PQC_KEY_MATERIAL_TPM_PCR_POLICY" in key_lifecycle_src,
        "self_test_wired_to_binary":
            "pqc_selftest_key_lifecycle" in selftest_src and
            "PQC-FUSE key lifecycle self-test" in main_src,
        "build_includes_lifecycle_source":
            "crypto/pqc_key_lifecycle.c" in crypto_sources,
    }

    source_checks = {
        "runtime_requires_mount_password":
            "PQC_MASTER_PASSWORD is required" in runtime_src,
        "new_roots_use_scrypt_metadata":
            "EVP_PBE_scrypt" in keyring_src and
            "PQC_KDF_ALG_SCRYPT" in keyring_src and
            "PQC_KDF_SALT_SIZE" in keyring_src,
        "legacy_pbkdf2_is_compatibility_path":
            "derive_pbkdf2_legacy" in keyring_src and
            "PBKDF2-HMAC-SHA256-legacy" in keyring_src,
        "envelope_wrapped_under_mount_key_and_file_id":
            "wrap_shared_secret(ss, ss_len, file_id, 0, meta.wrapped_ss)"
            in metadata_store_src,
        "envelope_hmac_before_unwrap":
            "CRYPTO_memcmp(digest, meta.digest" in metadata_load_src and
            metadata_load_src.find("CRYPTO_memcmp") <
            metadata_load_src.find("unwrap_shared_secret"),
        "tampered_envelope_rejects":
            "return -EKEYREJECTED" in metadata_load_src,
        "data_plane_uses_aes_gcm_aad":
            "EVP_aes_256_gcm()" in crypto_src and
            "pqc_crypto_build_block_aad" in crypto_src,
        "rekey_refreshes_open_file_envelope_not_data_key_rotation":
            "The current format uses one file key for all authenticated" in
            rekey_src and
            "pqc_keyring_metadata_store(snapshot.marker_path" in rekey_src and
            "ctx->key_epoch++" in rekey_src,
        "freshness_anchor_uses_file_or_tpm_nv_backend":
            "PQC_ANCHOR_BACKEND_HARDWARE" in anchor_src and
            "pqc_checkpoint_record_file_anchor" in checkpoint_src,
        "persistent_pcr_key_release_not_in_anchor_source":
            "TPM2_PolicyPCR" not in anchor_src and
            "TPM2_Unseal" not in anchor_src,
    }

    artifact_checks = {
        "d1_kdf_verdict_passes":
            kdf.get("overall_pass") is True and
            kdf.get("verdict", {}).get("decision") ==
            "scrypt_metadata_paper_guard_complete",
        "d2_crypto_plane_guard_passes":
            crypto_plane.get("overall_pass") is True,
        "c5_freshness_ladder_guard_passes":
            freshness.get("overall_pass") is True,
        "tpm_policy_scopes_no_credential_release":
            tpm_policy.get("overall_pass") is True and
            "no hardware-backed credential release" in
            set(tpm_policy.get("non_claims", [])),
        "clean_remount_passes":
            roundtrip.get("normal_remount") == "pass" and
            roundtrip.get("raw_plaintext_search") == "pass",
        "tampered_envelope_rejected":
            tamper.get("pass") is True and
            tamper.get("open_rejected") is True and
            tamper.get("metadata_xattr") == "user.pqc_metadata",
        "keyplane_workflow_passes":
            keyplane.get("overall_pass") is True and
            {mode.get("mode") for mode in keyplane.get("modes", [])} >=
            {"cpu_only", "gpu_batch", "policy_fallback"},
        "generation_matrix_passes":
            generation.get("overall_pass") is True and
            generation.get("no_silent_corruption") is True,
        "file_anchor_old_state_is_negative_control":
            stale_file.get("negative_control") is True and
            stale_file.get("acceptable") is True,
        "tpm_old_state_fails_closed":
            stale_tpm.get("oracle_verdict") == "fail_closed" and
            stale_tpm.get("acceptable") is True,
    }

    paper_gates = {
        key: phrase in paper
        for key, phrase in REQUIRED_PAPER_PHRASES.items()
    }

    close_conditions = {
        "production_lifecycle_table_complete":
            all(lifecycle_table_checks.values()),
        "source_boundary_matches_lifecycle_table":
            all(source_checks.values()),
        "retained_d1_d2_c5_evidence_linked":
            all(artifact_checks.values()),
        "paper_summarizes_key_lifecycle_and_non_claims":
            all(paper_gates.values()),
    }

    overall_pass = all(close_conditions.values())

    return {
        "schema_version": 2,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "overall_pass": overall_pass,
        "scope": (
            "Key-management lifecycle closeout for the mounted prototype. "
            "It covers the password-derived mount key, scrypt/PBKDF2 boundary, "
            "per-file envelope secret, AES-GCM data-block key, mount-lifetime "
            "ML-KEM keypair, freshness anchor material, TPM NV/PCR policy status, "
            "open-file rekey, recovery, rotation boundary, and failure modes. "
            "It does not claim hardware-backed credential release, deployed "
            "credential rotation, persistent PCR-bound key release, transactional "
            "rewrap recovery, or credential-only rollback resistance."
        ),
        "lifecycle_rows": lifecycle_rows,
        "lifecycle_table_checks": lifecycle_table_checks,
        "source_checks": source_checks,
        "artifact_checks": artifact_checks,
        "paper_gates": paper_gates,
        "close_conditions": close_conditions,
        "source_artifacts": [
            relpath(path) for path in (
                KEY_LIFECYCLE_C,
                KEY_LIFECYCLE_H,
                KEYRING_C,
                CRYPTO_C,
                REKEY_C,
                RUNTIME_C,
                ANCHOR_C,
                CHECKPOINT_C,
                SELFTEST_C,
                MAIN_C,
                CRYPTO_SOURCES,
                KDF_VERDICT,
                CRYPTO_PLANE,
                FRESHNESS_LADDER,
                TPM_POLICY,
                FUSE_ROUNDTRIP,
                FUSE_TAMPER,
                KEYPLANE,
                GENERATION_MATRIX,
            )
        ],
        "non_claims": [
            "no hardware-backed credential release path for the mount key",
            "no deployed mount credential rotation",
            "no persistent KEM hierarchy",
            "no persistent PCR-bound key release",
            "no TPM/PCR sealed-key recovery",
            "no persistent epoch anti-rollback journal in the mounted path",
            "no transactional rewrap recovery claim",
            "no credential-only rollback resistance",
            "no recovery after a lost mount credential",
        ],
    }


def markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Key-Management Lifecycle Manifest",
        "",
        f"- Overall pass: `{str(report['overall_pass']).lower()}`",
        f"- Scope: {report['scope']}",
        "",
        "## Close Conditions",
        "",
    ]
    for key, value in report["close_conditions"].items():
        lines.append(f"- `{key}`: `{str(value).lower()}`")

    for section in (
        "lifecycle_table_checks",
        "source_checks",
        "artifact_checks",
        "paper_gates",
    ):
        lines.extend(["", f"## {section}", ""])
        for key, value in report[section].items():
            lines.append(f"- `{key}`: `{str(value).lower()}`")

    lines.extend(["", "## Lifecycle Rows", ""])
    for row in report["lifecycle_rows"]:
        lines.extend(
            [
                f"### {row['name']}",
                "",
                f"- Material: `{row['material']}`",
                f"- Plane/status: `{row['plane']}` / `{row['status']}`",
                f"- Owner: {row['owner_module']}",
                f"- Producer: {row['producer']}",
                f"- Storage: {row['storage']}",
                f"- Protector: {row['protector']}",
                f"- Rotation: {row['rotation']}",
                f"- Recovery: {row['recovery']}",
                f"- Failure boundary: {row['failure_boundary']}",
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
    json_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    md_path.write_text(markdown(report), encoding="utf-8")
    print(
        json.dumps(
            {
                "overall_pass": report["overall_pass"],
                "json": relpath(json_path),
                "markdown": relpath(md_path),
                "close_conditions": report["close_conditions"],
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
