#!/usr/bin/env python3
"""Build the intended TPM/freshness policy manifest.

The manifest closes only the policy-definition checklist item.  It records the
current TPM evidence, the intended operational policy, and the explicit
non-claims.  It does not mutate TPM state or upgrade the paper to a persistent
PCR-bound freshness claim.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "artifacts" / "validation" / "tpm_freshness_policy"

ANCHOR_C = ROOT / "pqc_anchor.c"
SECURITY_TEX = ROOT / "Paper" / "8_Security_Analysis.tex"
DISCUSSION_TEX = ROOT / "Paper" / "10_Discussion_and_Limitations.tex"

TPM_PROVISION = ROOT / "artifacts" / "validation" / "tpm_provisioning_probe_sudo" / "tpm_provisioning_probe.json"
TPM_PCR = ROOT / "artifacts" / "validation" / "tpm_pcr_policy_probe" / "tpm_pcr_policy_probe.json"
TPM_REPLAY = ROOT / "artifacts" / "validation" / "tpm_monotonic_replay" / "tpm_monotonic_replay.json"
COMBINED = ROOT / "artifacts" / "validation" / "combined_durability_bundle" / "combined_durability_bundle.json"
TPM_UNPROVISIONED = ROOT / "artifacts" / "validation" / "tpm_unprovisioned.json"

REQUIRED_TOPICS = {
    "nv_authorization_model",
    "pcr_binding_role",
    "index_lifecycle",
    "reprovisioning",
    "software_update",
    "backup_migration",
    "lost_credentials",
    "expected_failure_behavior",
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


def paper_text() -> str:
    return "\n".join(
        path.read_text(encoding="utf-8", errors="replace")
        for path in (SECURITY_TEX, DISCUSSION_TEX)
        if path.exists()
    )


def any_check(report: dict[str, Any], needle: str) -> dict[str, Any] | None:
    for item in report.get("checks", []):
        if needle in " ".join(str(part) for part in item.get("command", [])):
            return item
    return None


def build_manifest() -> dict[str, Any]:
    anchor_src = ANCHOR_C.read_text(encoding="utf-8", errors="replace")
    provision = load_json(TPM_PROVISION)
    pcr = load_json(TPM_PCR)
    replay = load_json(TPM_REPLAY)
    combined = load_json(COMBINED)
    unprovisioned = load_json(TPM_UNPROVISIONED)

    nv_public = any_check(provision, "tpm2_nvreadpublic")
    nv_stdout = (nv_public or {}).get("stdout", "")
    pcr_results = pcr.get("results", {})
    replay_result = ((replay.get("result") or {}).get("replay_result") or {})
    sqlite_replay = (combined.get("unified_campaign") or {}).get("replay") or {}
    dbm_replay = (combined.get("unified_dbm_campaign") or {}).get("replay") or {}

    policy_rows = [
        {
            "topic": "nv_authorization_model",
            "current_evidence": (
                "The retained provisioning probe observes NV index 0x01500010 "
                "with ownerread|ownerwrite attributes, and pqc_anchor.c uses "
                "the TPM owner hierarchy for nvread/nvwrite."
            ),
            "intended_policy": (
                "The prototype treats the NV index as an administrator-owned "
                "freshness register.  It reads and writes only the authenticated "
                "prefix anchor record and does not claim a deployed TPM "
                "authorization policy."
            ),
            "paper_gate_phrase": "owner-authorized TPM NV index",
        },
        {
            "topic": "pcr_binding_role",
            "current_evidence": (
                "The transient PCR probe unseals under the current PCR file and "
                "rejects a drifted PCR file; it does not provision the persistent "
                "filesystem NV index."
            ),
            "intended_policy": (
                "PCR binding is a future release gate for stronger claims, not a "
                "property of the current persistent filesystem anchor."
            ),
            "paper_gate_phrase": "PCR binding is a transient probe rather than a persistent filesystem policy",
        },
        {
            "topic": "index_lifecycle",
            "current_evidence": (
                "pqc_anchor.c checks tpm2_nvreadpublic and refuses to define an "
                "index implicitly; the unprovisioned TPM artifact records a "
                "nonzero hardware-anchor startup exit."
            ),
            "intended_policy": (
                "Administrators create and rotate the NV index outside the "
                "daemon; the daemon fails closed when the configured index is "
                "absent."
            ),
            "paper_gate_phrase": "administrators create and rotate the index outside the daemon",
        },
        {
            "topic": "reprovisioning",
            "current_evidence": (
                "The implementation has no automatic NV define/redefine path and "
                "no artifact showing safe automatic anchor migration."
            ),
            "intended_policy": (
                "Reprovisioning is a deliberate administrative reset or migration "
                "step that must be paired with a compatible storage snapshot or "
                "a fresh anchor bootstrap."
            ),
            "paper_gate_phrase": "explicit reprovisioning/resealing",
        },
        {
            "topic": "software_update",
            "current_evidence": (
                "PCR drift is rejected only in the transient PCR-policy probe; no "
                "persistent PCR-sealed filesystem anchor is installed."
            ),
            "intended_policy": (
                "Software updates or PCR changes require explicit policy renewal "
                "before the paper can claim persistent PCR-bound freshness."
            ),
            "paper_gate_phrase": "software update or PCR change therefore requires explicit reprovisioning/resealing",
        },
        {
            "topic": "backup_migration",
            "current_evidence": (
                "The monotonic replay and combined SQLite/dbm.dumb campaigns show "
                "that stale backing-store snapshots fail closed against the "
                "advanced TPM anchor."
            ),
            "intended_policy": (
                "Backup and migration are not blind directory copies; a "
                "destination needs an authorized anchor bootstrap or the restored "
                "state should fail closed."
            ),
            "paper_gate_phrase": "Backup or migration cannot copy only the backing directory",
        },
        {
            "topic": "lost_credentials",
            "current_evidence": (
                "The paper and code keep the mount password as the root "
                "credential; no hardware-backed credential release or recovery "
                "artifact exists."
            ),
            "intended_policy": (
                "Lost mount credentials are unrecoverable in this prototype."
            ),
            "paper_gate_phrase": "Lost mount credentials are unrecoverable",
        },
        {
            "topic": "expected_failure_behavior",
            "current_evidence": (
                "The source fails closed on missing NV index, TPM I/O failure, "
                "anchor digest mismatch, and stored sequence ahead of local state; "
                "retained replay artifacts expose fail-closed read/open behavior."
            ),
            "intended_policy": (
                "Missing index, authorization failure, TPM I/O failure, digest "
                "mismatch, or a TPM sequence ahead of local state must fail "
                "closed instead of exposing stale data."
            ),
            "paper_gate_phrase": "Missing index, authorization failure, TPM I/O failure, digest mismatch, or a TPM sequence ahead of local state fail closed",
        },
    ]

    paper = paper_text()
    paper_gates = {row["topic"]: row["paper_gate_phrase"] in paper for row in policy_rows}
    source_checks = {
        "nvreadpublic_succeeds": (nv_public or {}).get("returncode") == 0,
        "nv_index_is_expected": (provision.get("configured") or {}).get("nv_index") == "0x01500010",
        "nv_ownerread_ownerwrite_recorded": "ownerwrite|ownerread" in nv_stdout,
        "nv_size_compatible_with_anchor_record": "size: 88" in nv_stdout,
        "code_uses_owner_hierarchy_nv_ops": "tpm2_nvwrite -C o" in anchor_src and "tpm2_nvread -C o" in anchor_src,
        "code_requires_preprovisioned_index": "Creating a persistent" in anchor_src and "implicitly would make ownership" in anchor_src,
        "code_detects_stale_sequence": "stored.global_sequence > local_seq" in anchor_src and "return -ESTALE" in anchor_src,
        "transient_pcr_probe_passes": pcr_results.get("good_unseal_matches_secret") is True and pcr_results.get("drift_rejected") is True,
        "monotonic_replay_fail_closed": replay_result.get("mode") == "fail_closed" and ((replay.get("result") or {}).get("fail_closed") is True),
        "sqlite_tpm_stale_snapshot_fail_closed": sqlite_replay.get("verdict") == "fail_closed" and sqlite_replay.get("acceptable") is True,
        "dbm_tpm_stale_snapshot_fail_closed": dbm_replay.get("verdict") == "fail_closed" and dbm_replay.get("acceptable") is True,
        "unprovisioned_tpm_fails_closed": unprovisioned.get("hardware_anchor_without_preprovisioning_exit") not in (None, 0),
    }
    policy_checks = {
        "all_required_topics_present": {row["topic"] for row in policy_rows} == REQUIRED_TOPICS,
        "all_rows_have_evidence_and_policy": all(row["current_evidence"] and row["intended_policy"] for row in policy_rows),
        "paper_covers_all_policy_topics": all(paper_gates.values()),
        "source_and_artifact_evidence_pass": all(source_checks.values()),
    }
    return {
        "schema_version": 1,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "overall_pass": all(policy_checks.values()),
        "scope": (
            "Policy-definition artifact only.  It records the intended TPM/freshness "
            "lifecycle and gates paper wording; it does not claim persistent "
            "PCR-bound freshness or deployed credential recovery."
        ),
        "source_artifacts": [
            relpath(TPM_PROVISION),
            relpath(TPM_PCR),
            relpath(TPM_REPLAY),
            relpath(COMBINED),
            relpath(TPM_UNPROVISIONED),
            relpath(ANCHOR_C),
        ],
        "policy_rows": policy_rows,
        "source_checks": source_checks,
        "paper_gates": paper_gates,
        "checks": policy_checks,
        "non_claims": [
            "no persistent PCR-bound filesystem freshness",
            "no hardware-backed credential release",
            "no automatic backup or migration recovery",
            "no recovery after lost mount credential",
            "no full power-loss or FUSE-daemon crash certification",
        ],
    }


def markdown(report: dict[str, Any]) -> str:
    lines = [
        "# TPM Freshness Policy Manifest",
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
    lines.extend(["", "## Policy Rows", ""])
    for row in report["policy_rows"]:
        lines.extend(
            [
                f"### {row['topic']}",
                "",
                f"- Current evidence: {row['current_evidence']}",
                f"- Intended policy: {row['intended_policy']}",
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
    json_path = out / "tpm_freshness_policy.json"
    md_path = out / "tpm_freshness_policy.md"
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
