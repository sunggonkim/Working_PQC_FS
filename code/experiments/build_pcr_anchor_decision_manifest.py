#!/usr/bin/env python3
"""Build the persistent-anchor PCR-binding decision manifest.

This manifest closes the checklist decision item by recording the current
choice: the persistent filesystem anchor is not PCR-bound in this revision.
The retained PCR artifact remains a transient seal/unseal probe, while the
persistent AEGIS-Q anchor path uses an administrator-provisioned TPM NV index.
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
DEFAULT_OUT = ROOT / "artifacts" / "validation" / "pcr_anchor_decision"

ANCHOR_C = CODE / "storage" / "pqc_anchor.c"
PCR_PROBE_SCRIPT = ROOT / "code" / "experiments" / "run_tpm_pcr_policy_probe.py"
TPM_POLICY_SCRIPT = ROOT / "code" / "experiments" / "build_tpm_freshness_policy_manifest.py"
TPM_PCR = ROOT / "artifacts" / "validation" / "tpm_pcr_policy_probe" / "tpm_pcr_policy_probe.json"
TPM_POLICY = ROOT / "artifacts" / "validation" / "tpm_freshness_policy" / "tpm_freshness_policy.json"
TPM_PROVISION = ROOT / "artifacts" / "validation" / "tpm_provisioning_probe_sudo" / "tpm_provisioning_probe.json"
TPM_REPLAY = ROOT / "artifacts" / "validation" / "tpm_monotonic_replay" / "tpm_monotonic_replay.json"

PAPER_FILES = sorted((ROOT / "Paper").glob("*.tex"))

FORBIDDEN_PERSISTENT_PCR_COMMANDS = (
    "tpm2_createpolicy",
    "tpm2_policypcr",
    "tpm2_startauthsession",
    "tpm2_unseal",
    "--policy-pcr",
    "--policy-session",
    "TPM2_PolicyPCR",
)

REQUIRED_PAPER_GATES = {
    "security_transient_policy": "PCR binding is a transient probe rather than a persistent filesystem policy",
    "evaluation_not_pcr_sealed": "not a claim that the persistent filesystem anchor itself is PCR-sealed",
    "design_no_persistent_lifecycle": "no persistent PCR-bound lifecycle claim",
    "abstract_nonclaim": "PCR-bound persistent freshness",
    "conclusion_nonclaim": "persistent PCR-bound freshness",
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
    return "\n".join(read(path) for path in PAPER_FILES)


def paragraph_for_offset(text: str, offset: int) -> str:
    start = text.rfind("\n\n", 0, offset)
    end = text.find("\n\n", offset)
    if start < 0:
        start = 0
    else:
        start += 2
    if end < 0:
        end = len(text)
    return text[start:end]


def high_risk_pcr_mentions(text: str) -> list[dict[str, str]]:
    accepted_markers = (
        "does not claim",
        "does not assert",
        "not a claim",
        "not persistent",
        "no persistent",
        "transient",
        "rather than a persistent",
        "Remaining gap",
        "Still missing",
        "stronger rollback-resistance claim requires",
        "still does not claim",
        "not a persistent",
        "not stale-handle",
        "not ... proof",
    )
    mentions: list[dict[str, str]] = []
    for match in re.finditer(r"PCR(?:-bound|-policy|-sealed| binding| policy)?|PCR-bound|PCR-sealed", text, re.IGNORECASE):
        para = paragraph_for_offset(text, match.start())
        normalized = " ".join(para.split())
        mentions.append(
            {
                "needle": match.group(0),
                "context": normalized[:500],
                "accepted_scope_context": str(any(marker in normalized for marker in accepted_markers)).lower(),
            }
        )
    return mentions


def any_check(report: dict[str, Any], needle: str) -> dict[str, Any] | None:
    for item in report.get("checks", []):
        if needle in " ".join(str(part) for part in item.get("command", [])):
            return item
    return None


def build_manifest() -> dict[str, Any]:
    anchor_src = read(ANCHOR_C)
    pcr_script = read(PCR_PROBE_SCRIPT)
    tpm_policy_script = read(TPM_POLICY_SCRIPT)
    pcr = load_json(TPM_PCR)
    policy = load_json(TPM_POLICY)
    provision = load_json(TPM_PROVISION)
    replay = load_json(TPM_REPLAY)
    paper = paper_text()

    pcr_results = pcr.get("results", {})
    nv_public = any_check(provision, "tpm2_nvreadpublic")
    nv_stdout = (nv_public or {}).get("stdout", "")
    replay_result = ((replay.get("result") or {}).get("replay_result") or {})
    risk_mentions = high_risk_pcr_mentions(paper)

    decision_rows = [
        {
            "topic": "persistent_anchor_decision",
            "decision": "no_persistent_pcr_bound_filesystem_anchor_in_this_revision",
            "evidence": (
                "pqc_anchor.c stores the persistent hardware anchor as an "
                "authenticated prefix record in a pre-provisioned TPM NV index "
                "using owner-hierarchy nvread/nvwrite commands.  It contains no "
                "PCR policy-session, sealed-object, or unseal path."
            ),
            "paper_gate_phrase": REQUIRED_PAPER_GATES["security_transient_policy"],
        },
        {
            "topic": "transient_pcr_probe_role",
            "decision": "retain_pcr_only_as_probe",
            "evidence": (
                "The PCR probe artifact seals a transient object, unseals under "
                "the current PCR digest, and rejects a drifted PCR digest; its "
                "own note says it is not a filesystem freshness proof."
            ),
            "paper_gate_phrase": REQUIRED_PAPER_GATES["evaluation_not_pcr_sealed"],
        },
        {
            "topic": "paper_claim_boundary",
            "decision": "paper_must_negate_persistent_pcr_binding",
            "evidence": (
                "The main paper gates the claim in design, evaluation, security, "
                "abstract, and conclusion wording; all high-risk PCR mentions are "
                "in a negated, transient, or remaining-gap context."
            ),
            "paper_gate_phrase": REQUIRED_PAPER_GATES["design_no_persistent_lifecycle"],
        },
    ]

    source_checks = {
        "anchor_uses_preprovisioned_nv_index": "tpm2_nvreadpublic" in anchor_src
        and "Creating a persistent" in anchor_src
        and "implicitly would make ownership" in anchor_src,
        "anchor_uses_owner_hierarchy_nv_ops": "tpm2_nvwrite -C o" in anchor_src
        and "tpm2_nvread -C o" in anchor_src,
        "anchor_source_has_no_persistent_pcr_commands": not any(cmd in anchor_src for cmd in FORBIDDEN_PERSISTENT_PCR_COMMANDS),
        "pcr_probe_script_declares_transient_scope": "This is PCR-policy evidence for a transient object" in pcr_script
        and "does not provision the persistent AEGIS-Q NV index" in pcr_script,
        "tpm_policy_manifest_declares_nonclaim": "no persistent PCR-bound filesystem freshness" in set(policy.get("non_claims", [])),
        "tpm_policy_script_preserves_nonclaim": "does not claim persistent" in tpm_policy_script,
    }
    artifact_checks = {
        "pcr_probe_passes_current_unseal": pcr_results.get("good_unseal_rc") == 0
        and pcr_results.get("good_unseal_matches_secret") is True,
        "pcr_probe_rejects_drift": pcr_results.get("drift_rejected") is True,
        "pcr_probe_note_is_transient": "Transient PCR policy probe only" in pcr.get("note", ""),
        "provisioned_nv_index_recorded": (provision.get("configured") or {}).get("nv_index") == "0x01500010"
        and "ownerwrite|ownerread" in nv_stdout,
        "tpm_replay_after_advance_fail_closed": replay_result.get("mode") == "fail_closed"
        and ((replay.get("result") or {}).get("fail_closed") is True),
        "tpm_policy_manifest_passes": policy.get("overall_pass") is True,
    }
    paper_gates = {name: phrase in paper for name, phrase in REQUIRED_PAPER_GATES.items()}
    paper_checks = {
        "required_paper_gate_phrases_present": all(paper_gates.values()),
        "all_high_risk_pcr_mentions_scoped": bool(risk_mentions)
        and all(item["accepted_scope_context"] == "true" for item in risk_mentions),
    }
    checks = {
        "decision_rows_present": len(decision_rows) == 3,
        "source_checks_pass": all(source_checks.values()),
        "artifact_checks_pass": all(artifact_checks.values()),
        "paper_checks_pass": all(paper_checks.values()),
    }

    return {
        "schema_version": 1,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "overall_pass": all(checks.values()),
        "decision": "no_persistent_pcr_bound_filesystem_anchor_in_this_revision",
        "scope": (
            "The persistent filesystem anchor is not PCR-bound in this revision. "
            "PCR evidence is retained only as a transient policy probe; persistent "
            "freshness evidence is limited to pre-provisioned TPM NV "
            "replay-after-advance fail-closed behavior."
        ),
        "source_artifacts": [
            relpath(ANCHOR_C),
            relpath(PCR_PROBE_SCRIPT),
            relpath(TPM_POLICY_SCRIPT),
            relpath(TPM_PCR),
            relpath(TPM_POLICY),
            relpath(TPM_PROVISION),
            relpath(TPM_REPLAY),
        ],
        "decision_rows": decision_rows,
        "source_checks": source_checks,
        "artifact_checks": artifact_checks,
        "paper_gates": paper_gates,
        "paper_checks": paper_checks,
        "checks": checks,
        "high_risk_pcr_mentions": risk_mentions,
        "non_claims": [
            "no persistent PCR-bound filesystem freshness",
            "no persistent PCR-sealed filesystem anchor",
            "no PCR-bound key release",
            "no software-update PCR migration protocol",
            "no PCR-bound backup or restore protocol",
        ],
    }


def markdown(report: dict[str, Any]) -> str:
    lines = [
        "# PCR Anchor Decision Manifest",
        "",
        f"- Overall pass: `{str(report['overall_pass']).lower()}`",
        f"- Decision: `{report['decision']}`",
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
    lines.extend(["", "## Paper Checks", ""])
    for key, value in report["paper_checks"].items():
        lines.append(f"- `{key}`: `{str(value).lower()}`")
    lines.extend(["", "## Decision Rows", ""])
    for row in report["decision_rows"]:
        lines.extend(
            [
                f"### {row['topic']}",
                "",
                f"- Decision: `{row['decision']}`",
                f"- Evidence: {row['evidence']}",
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
    json_path = out / "pcr_anchor_decision.json"
    md_path = out / "pcr_anchor_decision.md"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md_path.write_text(markdown(report), encoding="utf-8")
    print(
        json.dumps(
            {
                "overall_pass": report["overall_pass"],
                "decision": report["decision"],
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
