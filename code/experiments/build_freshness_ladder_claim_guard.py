#!/usr/bin/env python3
"""C5 freshness ladder and rollback-resistance claim guard.

The gate is intentionally conservative.  It separates evidence levels instead
of treating every TPM/PCR artifact as rollback resistance.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = ROOT / "artifacts" / "validation" / "freshness_ladder_claim_guard"

FILE_REPLAY = ROOT / "artifacts" / "validation" / "replay_file_final_matrix.json"
TPM_PROVISION = (
    ROOT / "artifacts" / "validation" / "tpm_provisioning_probe_sudo" /
    "tpm_provisioning_probe.json"
)
TPM_PCR = (
    ROOT / "artifacts" / "validation" / "tpm_pcr_policy_probe" /
    "tpm_pcr_policy_probe.json"
)
TPM_REPLAY = (
    ROOT / "artifacts" / "validation" / "tpm_monotonic_replay" /
    "tpm_monotonic_replay.json"
)
TPM_POLICY = (
    ROOT / "artifacts" / "validation" / "tpm_freshness_policy" /
    "tpm_freshness_policy.json"
)
PCR_DECISION = (
    ROOT / "artifacts" / "validation" / "pcr_anchor_decision" /
    "pcr_anchor_decision.json"
)
HW_MATRIX = (
    ROOT / "artifacts" / "validation" / "hardware_freshness_recovery_matrix" /
    "hardware_freshness_recovery_matrix.json"
)
COMBINED = (
    ROOT / "artifacts" / "validation" / "combined_durability_bundle" /
    "combined_durability_bundle.json"
)
ASYNC_MERKLE = (
    ROOT / "artifacts" / "validation" / "async_merkle_tpm_epoch" /
    "async_merkle_tpm_epoch.json"
)
TPM_EPOCH_PROBE = (
    ROOT / "artifacts" / "validation" / "async_merkle_tpm_epoch" /
    "tpm_epoch_freshness_probe.json"
)

ANCHOR_C = ROOT / "code" / "storage" / "pqc_anchor.c"
ANCHOR_H = ROOT / "code" / "storage" / "pqc_anchor.h"
ANCHOR_WORKER_C = ROOT / "code" / "storage" / "pqc_anchor_worker.c"

REQUIRED_HW_CASES = {
    "stale_disk_new_tpm",
    "new_disk_stale_tpm",
    "missing_index",
    "changed_pcrs",
    "authorization_failure",
    "interrupted_nv_update",
    "normal_replay_after_advance",
}

DANGER_PATTERNS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"persistent\s+PCR[- ]bound",
        r"PCR[- ]bound\s+freshness",
        r"PCR[- ]bound\s+rollback",
        r"PCR[- ]sealed",
        r"persistent\s+PCR\s+(?:binding|policy)",
        r"sealed[- ]key",
        r"hardware[- ]backed\s+credential\s+release",
        r"TPM\s+rollback\s+resistance",
        r"full\s+rollback\s+resistance",
        r"offline\s+rollback\s+resistance",
        r"rollback\s+resistance",
        r"anti[- ]rollback",
        r"async\s+Merkle",
        r"TPM\s+epoch\s+freshness",
        r"full\s+rollback",
        r"prevents?\s+rollback",
        r"rollback\s+proof",
        r"롤백\s*방지",
    )
]

C6_EPOCH_DANGER_PATTERNS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"TPM\s+rollback\s+resistance",
        r"persistent\s+PCR[- ]bound\s+(?:filesystem\s+)?freshness",
        r"persistent\s+PCR[- ]bound\s+(?:policy|lifecycle)",
        r"PCR[- ]bound\s+rollback\s+resistance",
        r"async\s+Merkle(?:\+TPM)?\s+(?:TPM\s+)?epoch",
        r"TPM\s+epoch\s+freshness",
        r"full\s+replay\s+protection",
        r"full\s+rollback\s+resistance",
        r"offline\s+rollback\s+resistance",
        r"persistent\s+filesystem\s+anchor.{0,80}PCR[- ]bound",
    )
]

GUARD_PATTERNS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"does not (?:claim|assert|prove|establish|extend)",
        r"not (?:claimed|a claim|a proof|freshness|rollback|persistent|PCR|.*proof|.*resistance)",
        r"no (?:persistent|hardware|full|TPM|PCR|rollback|recovery|transactional|deployed)",
        r"non[- ]claim",
        r"forbidden_until_closed",
        r"remains? (?:open|a non-claim)",
        r"still (?:requires|need|needs|does not|not|open)",
        r"(?:future|stronger|separate) (?:claim|gate|proof|evidence|release|work|path|experiment)",
        r"rather than (?:persistent|rollback|a persistent)",
        r"transient",
        r"negative control",
        r"only replay-after-advance",
        r"replay-after-advance",
        r"scope(?:d| boundary)?",
        r"boundary",
        r"explicit",
        r"separat(?:e|es|ing)",
        r"trade[- ]?off",
        r"evaluat(?:e|es|ed|ing|ion)",
        r"need(?:s|ed)?",
        r"requires?",
        r"missing",
        r"unsupported",
        r"아직",
        r"주장하지",
        r"검증",
        r"평가",
        r"필요",
        r"분리",
        r"증거",
        r"트레이드오프",
        r"보존하지만",
        r"과장해서는 안",
        r"별도",
    )
]


@dataclass
class Finding:
    path: str
    line: int
    text: str
    matched: list[str]
    guarded: bool
    context: str


def relpath(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fp:
        data = json.load(fp)
    if not isinstance(data, dict):
        raise ValueError(f"{relpath(path)} is not a JSON object")
    return data


def maybe_load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return load_json(path)


def scan_paths() -> list[Path]:
    paths = sorted((ROOT / "Paper").glob("*.tex"))
    paths.append(ROOT / "README.md")
    docs = ROOT / "docs"
    if docs.exists():
        paths.extend(sorted(docs.rglob("*.md")))
    return [path for path in paths if path.exists()]


def source_comment_scan_paths() -> list[Path]:
    suffixes = {".c", ".h", ".cu", ".cuh", ".cc", ".cpp", ".hpp", ".py", ".sh"}
    paths: list[Path] = []
    code_root = ROOT / "code"
    if code_root.exists():
        paths.extend(
            path for path in sorted(code_root.rglob("*"))
            if path.is_file() and path.suffix in suffixes
        )
    return paths


def c6_epoch_scan_paths() -> list[Path]:
    seen: set[Path] = set()
    paths: list[Path] = []
    for path in [*scan_paths(), *source_comment_scan_paths()]:
        if path in seen:
            continue
        seen.add(path)
        paths.append(path)
    return paths


def pattern_names(patterns: Iterable[re.Pattern[str]], text: str) -> list[str]:
    return [pattern.pattern for pattern in patterns if pattern.search(text)]


def context_guarded(context: str) -> bool:
    return bool(pattern_names(GUARD_PATTERNS, context))


def scan_claims() -> list[Finding]:
    findings: list[Finding] = []
    for path in scan_paths():
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        for index, line in enumerate(lines):
            matched = pattern_names(DANGER_PATTERNS, line)
            if not matched:
                continue
            start = max(0, index - 12)
            end = min(len(lines), index + 3)
            context = " ".join(lines[start:end])
            findings.append(Finding(
                path=relpath(path),
                line=index + 1,
                text=line.strip(),
                matched=matched,
                guarded=context_guarded(context),
                context=context.strip(),
            ))
    return findings


def scan_c6_epoch_claims() -> list[Finding]:
    findings: list[Finding] = []
    for path in c6_epoch_scan_paths():
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        for index, line in enumerate(lines):
            matched = pattern_names(C6_EPOCH_DANGER_PATTERNS, line)
            if not matched:
                continue
            start = max(0, index - 12)
            end = min(len(lines), index + 3)
            context = " ".join(lines[start:end])
            findings.append(Finding(
                path=relpath(path),
                line=index + 1,
                text=line.strip(),
                matched=matched,
                guarded=context_guarded(context),
                context=context.strip(),
            ))
    return findings


def artifact_paths_exist(paths: list[str]) -> bool:
    ok = True
    for item in paths:
        path = Path(item)
        if not path.is_absolute():
            path = ROOT / item
        ok = ok and path.exists()
    return ok


def any_check(report: dict[str, Any], needle: str) -> dict[str, Any] | None:
    for item in report.get("checks", []):
        if needle in " ".join(str(part) for part in item.get("command", [])):
            return item
    return None


def paper_text() -> str:
    return "\n".join(
        path.read_text(encoding="utf-8", errors="replace")
        for path in scan_paths()
    )


def source_contains(path: Path, snippets: list[str]) -> bool:
    text = path.read_text(encoding="utf-8", errors="replace")
    return all(snippet in text for snippet in snippets)


def build_ladder() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    file_replay = load_json(FILE_REPLAY)
    provision = load_json(TPM_PROVISION)
    pcr = load_json(TPM_PCR)
    replay = load_json(TPM_REPLAY)
    policy = load_json(TPM_POLICY)
    pcr_decision = load_json(PCR_DECISION)
    hw_matrix = load_json(HW_MATRIX)
    combined = load_json(COMBINED)
    async_merkle = maybe_load_json(ASYNC_MERKLE)
    tpm_epoch_probe = maybe_load_json(TPM_EPOCH_PROBE)

    file_rows = file_replay.get("rows", [])
    pcr_results = pcr.get("results", {})
    replay_result = ((replay.get("result") or {}).get("replay_result") or {})
    nv_public = any_check(provision, "tpm2_nvreadpublic")
    nv_stdout = (nv_public or {}).get("stdout", "")
    hw_rows = hw_matrix.get("rows", [])
    sqlite_replay = (combined.get("unified_campaign") or {}).get("replay") or {}
    dbm_replay = (combined.get("unified_dbm_campaign") or {}).get("replay") or {}

    file_anchor_ok = (
        bool(file_rows) and
        all(row.get("mode") == "rollback_visible" for row in file_rows) and
        all(row.get("rollback_blocked") is False for row in file_rows)
    )
    tpm_nv_ok = (
        (provision.get("configured") or {}).get("nv_index") == "0x01500010" and
        "ownerwrite|ownerread" in nv_stdout and
        "size: 88" in nv_stdout and
        replay_result.get("mode") == "fail_closed" and
        ((replay.get("result") or {}).get("fail_closed") is True) and
        hw_matrix.get("overall_pass") is True and
        set(hw_matrix.get("required_cases", [])) == REQUIRED_HW_CASES and
        all(row.get("pass") is True for row in hw_rows) and
        all(artifact_paths_exist(row.get("raw_artifacts", [])) for row in hw_rows)
    )
    pcr_transient_ok = (
        pcr_results.get("good_unseal_rc") == 0 and
        pcr_results.get("good_unseal_matches_secret") is True and
        pcr_results.get("drift_rejected") is True and
        "Transient PCR policy probe only" in pcr.get("note", "")
    )
    persistent_pcr_nonclaim_ok = (
        pcr_decision.get("overall_pass") is True and
        pcr_decision.get("decision") == "no_persistent_pcr_bound_filesystem_anchor_in_this_revision" and
        "no persistent PCR-bound filesystem freshness" in pcr_decision.get("non_claims", [])
    )
    sealed_key_nonclaim_ok = (
        policy.get("overall_pass") is True and
        "no hardware-backed credential release" in policy.get("non_claims", [])
    )
    tpm_epoch_probe_ok = (
        tpm_epoch_probe is not None and
        tpm_epoch_probe.get("overall_pass") is True and
        tpm_epoch_probe.get("verdict") == "environment-blocked" and
        tpm_epoch_probe.get("environment_blocked") is True and
        tpm_epoch_probe.get("hardware_epoch_committed") is False and
        ((tpm_epoch_probe.get("trace_summary") or {}).get(
            "hardware_pending_observed"
        ) is True) and
        ((tpm_epoch_probe.get("trace_summary") or {}).get(
            "hardware_flush_attempted"
        ) is True)
    )
    async_merkle_nonclaim_ok = (
        (async_merkle is None or async_merkle.get("overall_pass") is True) and
        tpm_epoch_probe_ok
    )
    full_rollback_nonclaim_ok = (
        "full rollback resistance is not claimed" in paper_text() and
        "offline rollback resistance를 완전 보장한다고 주장하지 않는다" in paper_text()
    )
    app_replay_ok = (
        sqlite_replay.get("verdict") == "fail_closed" and
        sqlite_replay.get("acceptable") is True and
        dbm_replay.get("verdict") == "fail_closed" and
        dbm_replay.get("acceptable") is True
    )

    ladder = [
        {
            "level": "L0_file_anchor_negative_control",
            "supported": file_anchor_ok,
            "allowed_claim": "file-backed witness is replayable; stale directory snapshot remains visible",
            "forbidden_upgrade": "rollback resistance",
            "evidence": [relpath(FILE_REPLAY)],
        },
        {
            "level": "L1_tpm_nv_replay_after_advance",
            "supported": tpm_nv_ok and app_replay_ok,
            "allowed_claim": "pre-provisioned TPM NV replay-after-advance can fail closed under retained rows",
            "forbidden_upgrade": "persistent PCR-bound freshness or full rollback resistance",
            "evidence": [relpath(TPM_PROVISION), relpath(TPM_REPLAY),
                         relpath(HW_MATRIX), relpath(COMBINED)],
        },
        {
            "level": "L2_transient_pcr_policy_probe",
            "supported": pcr_transient_ok,
            "allowed_claim": "transient PCR-policy seal/unseal probe rejects drift",
            "forbidden_upgrade": "persistent filesystem PCR policy",
            "evidence": [relpath(TPM_PCR)],
        },
        {
            "level": "L3_persistent_pcr_bound_policy",
            "supported": False,
            "allowed_claim": "non-claim only in this revision",
            "forbidden_upgrade": "persistent PCR-bound freshness",
            "nonclaim_guarded": persistent_pcr_nonclaim_ok,
            "evidence": [relpath(PCR_DECISION), relpath(TPM_POLICY)],
        },
        {
            "level": "L4_sealed_key_release",
            "supported": False,
            "allowed_claim": "non-claim only in this revision",
            "forbidden_upgrade": "hardware-backed credential release or sealed-key release",
            "nonclaim_guarded": sealed_key_nonclaim_ok,
            "evidence": [relpath(TPM_POLICY)],
        },
        {
            "level": "L5_async_merkle_tpm_epoch_freshness",
            "supported": False,
            "allowed_claim": (
                "non-claim only while the bounded hardware epoch probe is "
                "environment-blocked and no TPM epoch is committed"
            ),
            "forbidden_upgrade": "async Merkle + TPM epoch freshness",
            "nonclaim_guarded": async_merkle_nonclaim_ok,
            "evidence": [relpath(TPM_EPOCH_PROBE)] +
            ([relpath(ASYNC_MERKLE)] if async_merkle else []),
        },
        {
            "level": "L6_full_rollback_resistance",
            "supported": False,
            "allowed_claim": "non-claim only in this revision",
            "forbidden_upgrade": "full/offline rollback resistance",
            "nonclaim_guarded": full_rollback_nonclaim_ok,
            "evidence": [relpath(HW_MATRIX), relpath(PCR_DECISION)],
        },
    ]
    details = {
        "file_anchor_ok": file_anchor_ok,
        "tpm_nv_ok": tpm_nv_ok,
        "pcr_transient_ok": pcr_transient_ok,
        "persistent_pcr_nonclaim_ok": persistent_pcr_nonclaim_ok,
        "sealed_key_nonclaim_ok": sealed_key_nonclaim_ok,
        "async_merkle_artifact_present": async_merkle is not None,
        "tpm_epoch_probe_present": tpm_epoch_probe is not None,
        "tpm_epoch_probe_environment_blocked": (
            tpm_epoch_probe.get("environment_blocked") is True
            if tpm_epoch_probe else False
        ),
        "tpm_epoch_probe_hardware_epoch_committed": (
            tpm_epoch_probe.get("hardware_epoch_committed") is True
            if tpm_epoch_probe else False
        ),
        "tpm_epoch_probe_guard_basis_ok": tpm_epoch_probe_ok,
        "async_merkle_nonclaim_ok": async_merkle_nonclaim_ok,
        "full_rollback_nonclaim_ok": full_rollback_nonclaim_ok,
        "application_replay_after_advance_ok": app_replay_ok,
    }
    return ladder, details


def build_source_checks() -> dict[str, bool]:
    return {
        "anchor_has_file_and_hardware_backends": source_contains(ANCHOR_H, [
            "PQC_ANCHOR_BACKEND_FILE",
            "PQC_ANCHOR_BACKEND_HARDWARE",
        ]),
        "anchor_stages_hardware_and_flushes": source_contains(ANCHOR_C, [
            "stage_pending_hardware_anchor",
            "pqc_anchor_flush",
            "write_anchor_tpm",
        ]),
        "anchor_load_fails_closed_on_ahead_tpm_sequence": source_contains(ANCHOR_C, [
            "stored.global_sequence > local_seq",
            "return -ESTALE",
        ]),
        "anchor_worker_separates_background_stage_from_force_flush": source_contains(ANCHOR_WORKER_C, [
            "pqc_anchor_worker_stage",
            "pqc_anchor_store_force",
            "pqc_anchor_flush",
        ]),
        "source_has_no_persistent_pcr_or_sealed_key_path": not any(
            needle in ANCHOR_C.read_text(encoding="utf-8", errors="replace")
            for needle in (
                "tpm2_unseal",
                "TPM2_PolicyPCR",
                "--policy-session",
                "--policy-pcr",
                "sealed-key",
            )
        ),
    }


def build_paper_checks() -> dict[str, bool]:
    text = paper_text()
    required = {
        "file_anchor_negative_control": "file-backed anchor produces the deliberately negative result" in text,
        "tpm_replay_after_advance_scope": "This is replay-after-advance evidence" in text,
        "pcr_transient_scope": "PCR binding is a transient probe rather than a persistent filesystem policy" in text,
        "persistent_pcr_nonclaim": "persistent PCR-bound freshness" in text and "does not claim" in text,
        "full_rollback_nonclaim": "full rollback resistance is not claimed" in text,
        "credential_release_nonclaim": "no hardware-backed credential release" in text,
        "async_merkle_not_claimed": "async Merkle" not in "\n".join(
            path.read_text(encoding="utf-8", errors="replace")
            for path in sorted((ROOT / "Paper").glob("*.tex"))
        ),
    }
    return required


def write_outputs(out_dir: Path, payload: dict[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "freshness_ladder_claim_guard.json"
    md_path = out_dir / "freshness_ladder_claim_guard.md"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n",
                         encoding="utf-8")

    lines = [
        "# Freshness Ladder Claim Guard",
        "",
        f"- Generated: `{payload['generated_utc']}`",
        f"- Overall pass: `{str(payload['overall_pass']).lower()}`",
        f"- Unguarded dangerous lines: `{payload['unguarded_count']}`",
        "",
        "## Ladder",
        "",
    ]
    for row in payload["ladder"]:
        lines.append(
            f"- `{row['level']}`: supported=`{str(row.get('supported')).lower()}`; "
            f"allowed={row['allowed_claim']}; forbidden={row['forbidden_upgrade']}"
        )
    lines.extend(["", "## Unguarded Findings", ""])
    if payload["unguarded_findings"]:
        for finding in payload["unguarded_findings"]:
            lines.append(f"- `{finding['path']}:{finding['line']}` {finding['text']}")
    else:
        lines.append("- None.")
    lines.extend(["", "## C6 TPM Epoch Guard", ""])
    lines.append(
        f"- TPM epoch probe verdict: "
        f"`{payload['c6_epoch_probe'].get('verdict', 'missing')}`"
    )
    lines.append(
        f"- Hardware epoch committed: "
        f"`{str(payload['c6_epoch_probe'].get('hardware_epoch_committed')).lower()}`"
    )
    lines.append(
        f"- C6 unguarded dangerous lines: "
        f"`{payload['c6_epoch_unguarded_count']}`"
    )
    if payload["c6_epoch_unguarded_findings"]:
        for finding in payload["c6_epoch_unguarded_findings"]:
            lines.append(f"- `{finding['path']}:{finding['line']}` {finding['text']}")
    else:
        lines.append("- None.")
    lines.extend(["", "## Boundary", "", payload["boundary"], ""])
    md_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    ladder, ladder_details = build_ladder()
    source_checks = build_source_checks()
    paper_checks = build_paper_checks()
    findings = scan_claims()
    unguarded = [finding for finding in findings if not finding.guarded]
    c6_findings = scan_c6_epoch_claims()
    c6_unguarded = [finding for finding in c6_findings if not finding.guarded]

    supported_required = {
        "file_anchor_negative_control_supported": ladder[0]["supported"] is True,
        "tpm_replay_after_advance_supported": ladder[1]["supported"] is True,
        "pcr_transient_probe_supported": ladder[2]["supported"] is True,
        "persistent_pcr_nonclaim_guarded": ladder[3].get("nonclaim_guarded") is True,
        "sealed_key_nonclaim_guarded": ladder[4].get("nonclaim_guarded") is True,
        "async_merkle_not_accidentally_claimed": ladder[5]["supported"] is False and ladder[5].get("nonclaim_guarded") is True,
        "full_rollback_nonclaim_guarded": ladder[6].get("nonclaim_guarded") is True,
    }
    tpm_epoch_probe = maybe_load_json(TPM_EPOCH_PROBE) or {}
    c6_epoch_checks = {
        "probe_artifact_exists": TPM_EPOCH_PROBE.exists(),
        "probe_overall_pass": tpm_epoch_probe.get("overall_pass") is True,
        "probe_environment_blocked": (
            tpm_epoch_probe.get("verdict") == "environment-blocked" and
            tpm_epoch_probe.get("environment_blocked") is True
        ),
        "probe_hardware_epoch_not_committed":
            tpm_epoch_probe.get("hardware_epoch_committed") is False,
        "probe_reaches_hardware_stage": (
            (tpm_epoch_probe.get("trace_summary") or {}).get(
                "hardware_pending_observed"
            ) is True
        ),
        "probe_attempts_hardware_flush": (
            (tpm_epoch_probe.get("trace_summary") or {}).get(
                "hardware_flush_attempted"
            ) is True
        ),
        "strong_c6_claims_guarded": len(c6_unguarded) == 0,
    }
    payload: dict[str, Any] = {
        "schema_version": 1,
        "generated_by": "code/experiments/build_freshness_ladder_claim_guard.py",
        "generated_utc": now_utc(),
        "ladder": ladder,
        "ladder_details": ladder_details,
        "source_checks": source_checks,
        "paper_checks": paper_checks,
        "supported_required": supported_required,
        "dangerous_findings": [finding.__dict__ for finding in findings],
        "unguarded_findings": [finding.__dict__ for finding in unguarded],
        "unguarded_count": len(unguarded),
        "c6_epoch_probe": {
            "path": relpath(TPM_EPOCH_PROBE),
            "verdict": tpm_epoch_probe.get("verdict"),
            "environment_blocked": tpm_epoch_probe.get("environment_blocked"),
            "hardware_epoch_committed":
                tpm_epoch_probe.get("hardware_epoch_committed"),
            "trace_summary": tpm_epoch_probe.get("trace_summary"),
        },
        "c6_epoch_checks": c6_epoch_checks,
        "c6_epoch_findings": [finding.__dict__ for finding in c6_findings],
        "c6_epoch_unguarded_findings": [
            finding.__dict__ for finding in c6_unguarded
        ],
        "c6_epoch_unguarded_count": len(c6_unguarded),
        "overall_pass": (
            all(supported_required.values()) and
            all(c6_epoch_checks.values()) and
            all(source_checks.values()) and
            all(paper_checks.values()) and
            not unguarded and
            not c6_unguarded
        ),
        "boundary": (
            "Freshness claims are ordered by evidence: file anchor replay is a "
            "negative control; TPM NV evidence supports only retained "
            "replay-after-advance fail-closed rows; PCR evidence is transient; "
            "the C6 hardware epoch probe is environment-blocked on this machine; "
            "persistent PCR-bound policy, sealed-key release, async Merkle TPM "
            "epoch freshness, TPM rollback resistance, full replay protection, "
            "and full rollback resistance remain non-claims."
        ),
    }
    write_outputs(args.out_dir, payload)
    print(json.dumps({
        "overall_pass": payload["overall_pass"],
        "json": relpath(args.out_dir / "freshness_ladder_claim_guard.json"),
        "unguarded_count": len(unguarded),
        "failed_supported_required": [
            key for key, ok in supported_required.items() if not ok
        ],
        "failed_source_checks": [key for key, ok in source_checks.items() if not ok],
        "failed_paper_checks": [key for key, ok in paper_checks.items() if not ok],
        "failed_c6_epoch_checks": [
            key for key, ok in c6_epoch_checks.items() if not ok
        ],
    }, indent=2, sort_keys=True))
    return 0 if payload["overall_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
