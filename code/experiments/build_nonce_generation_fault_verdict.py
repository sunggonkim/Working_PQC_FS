#!/usr/bin/env python3
"""Build the C3 nonce/generation verdict from retained strict and epoch data.

This is a small gate script, not a new fault campaign.  It fails closed unless
the existing strict generation matrix, epoch replay matrix, source invariants,
and paper claim guards all agree on the same narrow model.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
STRICT_MATRIX = (
    ROOT / "artifacts" / "validation" / "generation_fault_matrix" /
    "generation_fault_matrix.json"
)
EPOCH_MATRIX = (
    ROOT / "artifacts" / "validation" / "publication_protocol_fault_matrix" /
    "epoch_replay_fault_matrix.json"
)
DEFAULT_OUT = ROOT / "artifacts" / "validation" / "nonce_generation_fault_verdict"


@dataclass
class Check:
    name: str
    ok: bool
    evidence: str
    residual: str = ""


def relpath(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fp:
        data = json.load(fp)
    if not isinstance(data, dict):
        raise ValueError(f"{relpath(path)} is not a JSON object")
    return data


def now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def row_by_case(strict: dict[str, Any], case: str) -> dict[str, Any]:
    for row in strict.get("rows", []):
        if row.get("case") == case:
            return row
    return {}


def epoch_case(epoch: dict[str, Any], label: str) -> dict[str, Any]:
    for case in epoch.get("cases", []):
        if case.get("label") == label:
            return case
    return {}


def compact_events(case: dict[str, Any]) -> list[dict[str, Any]]:
    trace = case.get("trace", {})
    events = trace.get("compact_events", [])
    return [event for event in events if isinstance(event, dict)]


def has_compact_event(case: dict[str, Any], **expected: int) -> bool:
    for event in compact_events(case):
        matched = True
        for key, value in expected.items():
            if int(event.get(key, -999999)) != value:
                matched = False
                break
        if matched:
            return True
    return False


def max_compact_value(case: dict[str, Any], key: str) -> int:
    values = [int(event.get(key, 0) or 0) for event in compact_events(case)]
    return max(values) if values else 0


def retained_path_exists(path_text: str | None) -> bool:
    if not path_text:
        return False
    path = Path(path_text)
    if not path.is_absolute():
        path = ROOT / path
    return path.exists()


def retained_journal_paths(row: dict[str, Any]) -> list[str]:
    paths: list[str] = []
    for key in (
        "journal",
        "journal_before_attack",
        "journal_after_attack",
        "journal_after_fault",
        "journal_after_remount_write",
    ):
        value = row.get(key)
        if isinstance(value, dict):
            retained = value.get("retained_file")
            if isinstance(retained, dict) and isinstance(retained.get("path"), str):
                paths.append(retained["path"])
    return paths


def source_contains(path: Path, snippets: list[str]) -> bool:
    text = path.read_text(encoding="utf-8", errors="replace")
    return all(snippet in text for snippet in snippets)


def paper_text() -> str:
    parts: list[str] = []
    for path in sorted((ROOT / "Paper").glob("*.tex")):
        parts.append(path.read_text(encoding="utf-8", errors="replace"))
    readme = ROOT / "README.md"
    if readme.exists():
        parts.append(readme.read_text(encoding="utf-8", errors="replace"))
    return "\n".join(parts)


def build_checks(strict: dict[str, Any], epoch: dict[str, Any]) -> list[Check]:
    checks: list[Check] = []

    strict_rows = strict.get("rows", [])
    strict_rows_ok = (
        strict.get("overall_pass") is True and
        strict.get("no_generated_nonce_reuse") is True and
        strict.get("no_silent_corruption") is True and
        int(strict.get("unexpected_liveness_failures", 1)) == 0 and
        all(row.get("acceptable") is True for row in strict_rows)
    )
    checks.append(Check(
        "strict_matrix_overall",
        strict_rows_ok,
        "strict matrix reports overall_pass, no generated nonce reuse, no silent corruption, and no unexpected liveness failures",
        "process/final-binary fault model only",
    ))

    partial = row_by_case(strict, "partial_update_and_remount")
    torn = row_by_case(strict, "torn_journal_write")
    reserved = row_by_case(strict, "reserved_generation_skip_after_data_fsync_fault")
    older = row_by_case(strict, "older_generation_append_after_newer_mapping")
    stale_file = row_by_case(strict, "stale_snapshot_replay_file_anchor_negative_control")
    stale_tpm = row_by_case(strict, "stale_snapshot_replay_tpm_anchor_existing_artifact")

    checks.append(Check(
        "strict_partial_update_remount",
        partial.get("acceptable") is True and
        partial.get("oracle_verdict") == "latest_committed" and
        partial.get("live_verdict") == "latest_committed" and
        not partial.get("generated_duplicate_nonce_pairs"),
        "partial update/remount returns latest committed state with no generated duplicate journal pairs",
    ))
    checks.append(Check(
        "strict_torn_record_tail",
        torn.get("acceptable") is True and
        torn.get("oracle_verdict") == "latest_committed" and
        int(torn.get("journal", {}).get("torn_tail_bytes", 0) or 0) > 0 and
        not torn.get("generated_duplicate_nonce_pairs"),
        "torn strict journal tail is ignored and remount exposes the latest committed payload",
    ))
    final_generations = reserved.get("final_generations_for_block0", [])
    checks.append(Check(
        "strict_interrupted_generation_advance",
        reserved.get("acceptable") is True and
        reserved.get("skipped_reserved_generation") is True and
        reserved.get("fault_cutpoint") == "data_fsync_after" and
        not reserved.get("generated_duplicate_nonce_pairs") and
        isinstance(final_generations, list) and len(final_generations) >= 2,
        "daemon kill after generation reservation/data fsync skips the unpublished generation on remount",
        "not physical power loss",
    ))
    checks.append(Check(
        "strict_replayed_metadata",
        older.get("acceptable") is True and
        older.get("oracle_verdict") == "latest_committed" and
        bool(older.get("adversarial_replay_duplicate_pairs")) and
        not older.get("generated_duplicate_nonce_pairs"),
        "adversarial append of an older mapping creates a replay duplicate but does not supersede the newer mapping",
    ))
    checks.append(Check(
        "strict_stale_checkpoint_ladder",
        stale_file.get("acceptable") is True and
        stale_file.get("oracle_verdict") == "previous_committed" and
        stale_file.get("negative_control") is True and
        stale_tpm.get("acceptable") is True and
        stale_tpm.get("oracle_verdict") == "fail_closed" and
        stale_tpm.get("negative_control") is False,
        "file-backed stale snapshot remains replayable, while retained TPM replay-after-advance artifact fails closed",
        "not PCR-bound rollback resistance",
    ))

    retained_paths = []
    for row in (partial, torn, reserved, older):
        retained_paths.extend(retained_journal_paths(row))
    missing_paths = [p for p in retained_paths if not retained_path_exists(p)]
    checks.append(Check(
        "strict_retained_raw_logs",
        bool(retained_paths) and not missing_paths,
        f"{len(retained_paths)} retained strict journal files exist",
        ", ".join(missing_paths),
    ))

    epoch_ok = epoch.get("overall_pass") is True and all(
        case.get("pass") is True for case in epoch.get("cases", [])
    )
    checks.append(Check(
        "epoch_matrix_overall",
        epoch_ok,
        "epoch replay matrix reports all cases pass",
        "committed-prefix replay only",
    ))

    epoch_torn = epoch_case(epoch, "replay_torn_tail")
    epoch_duplicate = epoch_case(epoch, "replay_duplicate_generation")
    epoch_journal_loss = epoch_case(epoch, "replay_journal_loss")
    epoch_normal = epoch_case(epoch, "replay_normal")

    checks.append(Check(
        "epoch_torn_log_tail",
        epoch_torn.get("pass") is True and
        epoch_torn.get("read_client", {}).get("matches") is True and
        has_compact_event(epoch_torn, rc=0, torn_tail_bytes=17),
        "epoch committed-prefix replay ignores a torn log tail and recovers matching data",
    ))
    checks.append(Check(
        "epoch_duplicate_append_fail_closed",
        epoch_duplicate.get("pass") is True and
        epoch_duplicate.get("read_client", {}).get("opened") is False and
        max_compact_value(epoch_duplicate, "duplicate_generation_records") >= 1 and
        any(int(event.get("rc", 0) or 0) == -17 for event in compact_events(epoch_duplicate)),
        "epoch replay rejects duplicate generation records instead of exposing data",
    ))
    checks.append(Check(
        "epoch_stale_journal_repair",
        epoch_journal_loss.get("pass") is True and
        epoch_journal_loss.get("read_client", {}).get("matches") is True and
        int(epoch_journal_loss.get("mutation_result", {}).get("bytes_after", -1)) == 0 and
        max_compact_value(epoch_journal_loss, "journal_repair_records") >= 1,
        "epoch replay repairs a deliberately lost strict journal from the committed epoch prefix",
    ))
    checks.append(Check(
        "epoch_normal_committed_prefix",
        epoch_normal.get("pass") is True and
        epoch_normal.get("read_client", {}).get("matches") is True and
        has_compact_event(epoch_normal, rc=0, torn_tail_bytes=0),
        "normal epoch remount compacts the committed prefix without duplicate generations",
    ))

    source_checks = (
        source_contains(ROOT / "code" / "crypto" / "pqc_crypto.c", [
            "memcpy(nonce_seed, &file_id, sizeof(file_id));",
            "memcpy(nonce_seed + 8, &block, sizeof(block));",
            "memcpy(nonce_seed + 16, &generation, sizeof(generation));",
            "pqc_crypto_build_block_aad(aad, file_id, block, generation, length);",
        ]) and
        source_contains(ROOT / "code" / "storage" / "pqc_journal.c", [
            "record.mapping.logical_block == logical_block",
            "record.mapping.generation <= max_generation",
            "record.mapping.generation > best.generation",
        ]) and
        source_contains(ROOT / "code" / "storage" / "pqc_epoch_log.c", [
            "a->file_id == b->file_id",
            "a->logical_block == b->logical_block",
            "a->generation == b->generation",
            "rc = -EEXIST;",
        ])
    )
    checks.append(Check(
        "source_invariant_guard",
        source_checks,
        "nonce/AAD bind file_id, block, generation; strict replay chooses highest committed generation; epoch replay rejects duplicate keys",
    ))

    text = paper_text()
    duplicate_guard = (
        "no duplicate generated" in text or
        "no generated \\((\\mathit{block},\\mathit{generation})\\) duplicates" in text
    )
    paper_checks = (
        duplicate_guard and
        "not physical power-loss" in text and
        "not a formal proof of crash consistency" in text and
        "persistent PCR-bound freshness" in text
    )
    checks.append(Check(
        "paper_negative_claim_guard",
        paper_checks,
        "paper states duplicate-generation invariant and residual power-loss/PCR/crash-proof limits",
    ))

    return checks


def write_outputs(out_dir: Path, payload: dict[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "nonce_generation_fault_verdict.json"
    md_path = out_dir / "nonce_generation_fault_verdict.md"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n",
                         encoding="utf-8")

    lines = [
        "# Nonce/Generation Fault Verdict",
        "",
        f"- Generated: `{payload['generated_utc']}`",
        f"- Overall pass: `{str(payload['overall_pass']).lower()}`",
        f"- Strict matrix: `{payload['inputs']['strict_matrix']}`",
        f"- Epoch matrix: `{payload['inputs']['epoch_matrix']}`",
        "",
        "## Checks",
        "",
    ]
    for check in payload["checks"]:
        lines.append(
            f"- `{check['name']}`: `{str(check['ok']).lower()}` - "
            f"{check['evidence']}"
        )
        if check.get("residual"):
            lines.append(f"  - Residual: {check['residual']}")
    lines.extend([
        "",
        "## Claim Boundary",
        "",
        payload["claim_boundary"],
        "",
    ])
    md_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--strict-matrix", type=Path, default=STRICT_MATRIX)
    parser.add_argument("--epoch-matrix", type=Path, default=EPOCH_MATRIX)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    strict = load_json(args.strict_matrix)
    epoch = load_json(args.epoch_matrix)
    checks = build_checks(strict, epoch)
    payload = {
        "schema_version": 1,
        "generated_by": "code/experiments/build_nonce_generation_fault_verdict.py",
        "generated_utc": now_utc(),
        "inputs": {
            "strict_matrix": relpath(args.strict_matrix),
            "epoch_matrix": relpath(args.epoch_matrix),
        },
        "checks": [check.__dict__ for check in checks],
        "overall_pass": all(check.ok for check in checks),
        "claim_boundary": (
            "C3 proves nonce/generation safety only for the retained final-binary "
            "strict and epoch fault models. Strict evidence is per-file sidecar "
            "(block,generation) evidence; epoch replay additionally checks "
            "(file_id,block,generation) duplicate records. This does not prove "
            "physical power-loss, kernel-crash, drive-cache, PCR-bound rollback "
            "resistance, or complete POSIX crash certification."
        ),
    }
    write_outputs(args.out_dir, payload)
    summary = {
        "overall_pass": payload["overall_pass"],
        "json": relpath(args.out_dir / "nonce_generation_fault_verdict.json"),
        "failed_checks": [check.name for check in checks if not check.ok],
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if payload["overall_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
