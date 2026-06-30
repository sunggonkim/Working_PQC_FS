#!/usr/bin/env python3
"""Build the baseline-contract immutability verdict for Gate B4.

This is a verifier, not a benchmark runner.  It ties the current plaintext,
gocryptfs, AEGIS-Q, fscrypt, and dm-crypt rows to the single frozen filesystem
contract and fails if the paper implies historical diagnostics are current
comparison rows.
"""

from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "artifacts" / "validation" / "baseline_contract_immutability"
MASTER_CONTRACT = ROOT / "artifacts" / "validation" / "frozen_workload_contract" / "frozen_workload_contract.json"

MEASURED_ROWS = {
    "plaintext_lowerfs": ROOT / "artifacts" / "validation" / "frozen_plaintext_contract" / "frozen_plaintext_contract.json",
    "gocryptfs": ROOT / "artifacts" / "validation" / "frozen_gocryptfs_contract" / "frozen_gocryptfs_contract.json",
    "dm_crypt_ext4": ROOT / "artifacts" / "validation" / "frozen_dmcrypt_contract" / "frozen_dmcrypt_contract.json",
    "aegis_q": ROOT / "artifacts" / "validation" / "frozen_aegisq_contract" / "frozen_aegisq_contract.json",
}

BLOCKED_ROWS = {
    "fscrypt": {
        "status_artifact": ROOT / "artifacts" / "validation" / "kernel_baseline_feasibility" / "paper_fscrypt_verdict.json",
        "status_key": "fscrypt_status",
        "paper_proof_key": "paper_marks_fscrypt_unavailable_with_proof",
    },
}

PAPER_DIR = ROOT / "Paper"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def paper_text() -> tuple[str, list[dict[str, Any]]]:
    chunks: list[str] = []
    lines: list[dict[str, Any]] = []
    for path in sorted(PAPER_DIR.glob("*.tex")):
        text = path.read_text(encoding="utf-8")
        chunks.append(text)
        for line_no, line in enumerate(text.splitlines(), start=1):
            lines.append({"path": str(path.relative_to(ROOT)), "line": line_no, "text": line})
    return "\n".join(chunks), lines


def canon_fio_command(argv: list[str]) -> list[str]:
    canonical: list[str] = []
    skip_next = False
    for token in argv:
        if skip_next:
            skip_next = False
            continue
        if token == "--directory":
            canonical.extend(["--directory", "${BENCH_DIR}"])
            skip_next = True
            continue
        if token.startswith("--directory="):
            canonical.append("--directory=${BENCH_DIR}")
            continue
        canonical.append(token)
    return canonical


def command_template_to_argv(template: str) -> list[str]:
    return [part if not part.startswith("--directory=") else "--directory=${BENCH_DIR}" for part in template.split()]


def grep_lines(lines: list[dict[str, Any]], pattern: str) -> list[dict[str, Any]]:
    regex = re.compile(pattern, re.IGNORECASE)
    return [line for line in lines if regex.search(line["text"])]


def evaluate_measured_row(
    row_name: str,
    row_path: Path,
    contract_id: str,
    contract_sha256: str,
    workload_profile: dict[str, Any],
    expected_command: list[str],
) -> dict[str, Any]:
    row = load_json(row_path)
    warm_summary = row.get("warm_cache_summary") or {}
    valid_repetitions = warm_summary.get("valid_repetitions", 0)
    expected_repetitions = workload_profile["repetition_count"]
    fio_command = row.get("fio_command") or []
    canonical_command = canon_fio_command(fio_command)

    command_artifacts = sorted(
        str(path.relative_to(ROOT))
        for path in row_path.parent.glob("fio_raw/warm_rep_*.command.json")
    )
    raw_fio_artifacts = sorted(
        str(path.relative_to(ROOT))
        for path in row_path.parent.glob("fio_raw/warm_rep_*.json")
    )
    cold_cache = row.get("cold_cache") or {}
    invalid_run_reasons: list[str] = []
    if cold_cache.get("status") and cold_cache.get("status") != "measured":
        invalid_run_reasons.append(f"cold_cache:{cold_cache.get('status')}")
    if not row.get("comparison_ready"):
        invalid_run_reasons.append("comparison_ready:false")

    checks = {
        "artifact_exists": row_path.exists(),
        "status_measured": row.get("overall_pass") is True,
        "contract_id_matches": row.get("contract_id") == contract_id,
        "contract_sha256_matches": row.get("contract_sha256_recorded") == contract_sha256,
        "workload_profile_matches": row.get("workload_profile") == workload_profile["profile_id"],
        "warm_cache_contract_compliant": row.get("contract_compliant_warm_cache") is True,
        "warm_repetition_count_matches": valid_repetitions == expected_repetitions,
        "fio_command_matches": canonical_command == expected_command,
        "command_artifacts_present": len(command_artifacts) >= expected_repetitions,
        "raw_fio_artifacts_present": len(raw_fio_artifacts) >= expected_repetitions,
    }

    return {
        "row": row_name,
        "status": "measured",
        "artifact": str(row_path.relative_to(ROOT)),
        "checks": checks,
        "pass": all(checks.values()),
        "contract_id": row.get("contract_id"),
        "contract_sha256_recorded": row.get("contract_sha256_recorded"),
        "workload_profile": row.get("workload_profile"),
        "cache_state": "warm",
        "warm_valid_repetitions": valid_repetitions,
        "expected_repetitions": expected_repetitions,
        "canonical_fio_command": canonical_command,
        "command_artifacts": command_artifacts,
        "raw_fio_artifacts": raw_fio_artifacts,
        "invalid_run_reasons": invalid_run_reasons,
        "cold_cache": cold_cache,
        "paper_table_status": "warm-cache measured row",
    }


def evaluate_blocked_row(
    row_name: str,
    row_info: dict[str, Any],
    contract: dict[str, Any],
) -> dict[str, Any]:
    artifact = row_info["status_artifact"]
    verdict = load_json(artifact)
    status = verdict.get(row_info["status_key"]) or {}
    workload = contract["workload_profiles"][0]
    mode = contract["filesystem_modes"][row_name]
    checks = {
        "artifact_exists": artifact.exists(),
        "environment_blocked": status.get("environment_blocked") is True,
        "not_measured": status.get("measured") is False,
        "blocking_reasons_present": bool(status.get("blocking_reasons")),
        "paper_marks_unavailable_with_proof": verdict.get(row_info["paper_proof_key"]) is True,
        "dangerous_claims_clear": verdict.get("dangerous_claims_clear") is True,
    }
    return {
        "row": row_name,
        "status": "environment-blocked",
        "artifact": str(artifact.relative_to(ROOT)),
        "checks": checks,
        "pass": all(checks.values()),
        "contract_id": contract["contract_id"],
        "workload_profile": workload["profile_id"],
        "cache_state": "not run",
        "warm_valid_repetitions": 0,
        "expected_repetitions": workload["repetition_count"],
        "canonical_fio_command": command_template_to_argv(
            workload["mount_options"]["command_template"].replace("${BENCH_DIR}", mode["benchmark_directory"])
        ),
        "mount_options": mode["mount_options"],
        "blocking_reasons": status.get("blocking_reasons", []),
        "invalid_run_reasons": status.get("blocking_reasons", []),
        "paper_table_status": "environment-blocked with proof",
    }


def evaluate_paper(lines: list[dict[str, Any]]) -> dict[str, Any]:
    frozen_lines = grep_lines(lines, r"frozen-contract|Frozen contract|filesystem contract|v2 filesystem contract")
    historical_lines = grep_lines(lines, r"historical .*fscrypt|historical .*dm-crypt|historical fscrypt|historical dm-crypt")
    bad_historical_lines = [
        line
        for line in historical_lines
        if not re.search(r"not current comparison evidence|only for traceability", line["text"], re.IGNORECASE)
    ]
    measured_kernel_claims = grep_lines(lines, r"measured fscrypt|fscrypt .*MiB/s")
    allowed_measured_kernel_lines = []
    bad_measured_kernel_lines = [
        line for line in measured_kernel_claims if line not in allowed_measured_kernel_lines
    ]
    status_lines = grep_lines(lines, r"fscrypt remains environment-blocked|fscrypt environment-blocked|fscrypt unavailable")
    measured_rows_lines = grep_lines(lines, r"plaintext/gocryptfs/dm-crypt/AEGIS-Q warm|warm-cache plaintext, gocryptfs, dm-crypt, AEGIS-Q|matched dm-crypt row")

    checks = {
        "frozen_contract_status_present": bool(frozen_lines),
        "measured_rows_present": bool(measured_rows_lines),
        "environment_blocked_rows_present": bool(status_lines),
        "historical_diagnostics_not_current_rows": not bad_historical_lines,
        "no_measured_fscrypt_throughput_claim": not bad_measured_kernel_lines,
    }
    return {
        "checks": checks,
        "pass": all(checks.values()),
        "frozen_contract_lines": frozen_lines,
        "measured_rows_lines": measured_rows_lines,
        "environment_blocked_lines": status_lines,
        "historical_lines": historical_lines,
        "bad_historical_lines": bad_historical_lines,
        "bad_measured_kernel_lines": bad_measured_kernel_lines,
    }


def build_report() -> dict[str, Any]:
    master = load_json(MASTER_CONTRACT)
    contract = master["contract"]
    workload = contract["workload_profiles"][0]
    expected_command = command_template_to_argv(workload["mount_options"]["command_template"])
    text, lines = paper_text()

    rows: dict[str, Any] = {}
    for name, path in MEASURED_ROWS.items():
        rows[name] = evaluate_measured_row(
            name,
            path,
            contract["contract_id"],
            master["contract_sha256"],
            workload,
            expected_command,
        )
    for name, info in BLOCKED_ROWS.items():
        rows[name] = evaluate_blocked_row(name, info, contract)

    paper = evaluate_paper(lines)
    statuses = {name: row["status"] for name, row in rows.items()}
    current_rows_only = all(status in {"measured", "environment-blocked", "not claimed"} for status in statuses.values())
    measured_ready = all(rows[name]["pass"] for name in MEASURED_ROWS)
    blocked_ready = all(rows[name]["pass"] for name in BLOCKED_ROWS)
    required_rows = set(contract["filesystem_modes"])
    observed_rows = set(rows)

    checks = {
        "master_contract_complete": master.get("overall_pass") is True,
        "all_required_rows_accounted": required_rows == observed_rows,
        "all_rows_have_valid_status": current_rows_only,
        "measured_rows_tied_to_same_contract": measured_ready,
        "blocked_rows_have_proof": blocked_ready,
        "paper_status_matches_rows": paper["pass"],
    }

    return {
        "schema_version": 1,
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "gate": "B4",
        "submilestone": "B4-S0",
        "source_contract": str(MASTER_CONTRACT.relative_to(ROOT)),
        "contract_id": contract["contract_id"],
        "contract_sha256": master["contract_sha256"],
        "workload_profile": workload["profile_id"],
        "expected_command_template": workload["mount_options"]["command_template"],
        "expected_repetitions": workload["repetition_count"],
        "row_statuses": statuses,
        "rows": rows,
        "paper": paper,
        "checks": checks,
        "overall_pass": all(checks.values()),
        "negative_claim_guard": {
            "historical_diagnostics_are_current_rows": False,
            "unsupported_fscrypt_speed_claim_allowed": False,
            "cold_cache_claim_allowed": False,
        },
    }


def markdown_report(report: dict[str, Any]) -> str:
    lines = [
        "# Baseline Contract Immutability Verdict",
        "",
        f"- Gate: `{report['gate']}`",
        f"- Submilestone: `{report['submilestone']}`",
        f"- Overall pass: `{str(report['overall_pass']).lower()}`",
        f"- Contract ID: `{report['contract_id']}`",
        f"- Contract SHA-256: `{report['contract_sha256']}`",
        f"- Workload profile: `{report['workload_profile']}`",
        f"- Expected repetitions: `{report['expected_repetitions']}`",
        f"- Command template: `{report['expected_command_template']}`",
        "",
        "## Rows",
        "",
    ]
    for name, row in report["rows"].items():
        lines.extend(
            [
                f"### `{name}`",
                "",
                f"- Status: `{row['status']}`",
                f"- Pass: `{str(row['pass']).lower()}`",
                f"- Artifact: `{row['artifact']}`",
                f"- Paper status: `{row['paper_table_status']}`",
                f"- Cache state: `{row['cache_state']}`",
                f"- Warm valid repetitions: `{row['warm_valid_repetitions']}` / `{row['expected_repetitions']}`",
            ]
        )
        if row.get("blocking_reasons"):
            lines.append(f"- Blocking reasons: `{', '.join(row['blocking_reasons'])}`")
        if row.get("invalid_run_reasons"):
            lines.append(f"- Invalid-run reasons: `{', '.join(row['invalid_run_reasons'])}`")
        failed = [key for key, value in row["checks"].items() if not value]
        lines.append(f"- Failed checks: `{', '.join(failed) if failed else 'none'}`")
        lines.append("")

    lines.extend(["## Paper Guard", ""])
    failed_paper = [key for key, value in report["paper"]["checks"].items() if not value]
    lines.append(f"- Failed paper checks: `{', '.join(failed_paper) if failed_paper else 'none'}`")
    lines.append(f"- Bad historical lines: `{len(report['paper']['bad_historical_lines'])}`")
    lines.append(f"- Bad measured-kernel lines: `{len(report['paper']['bad_measured_kernel_lines'])}`")
    lines.extend(["", "## Top-Level Checks", ""])
    for key, value in report["checks"].items():
        lines.append(f"- `{key}`: `{str(value).lower()}`")
    return "\n".join(lines) + "\n"


def main() -> int:
    report = build_report()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    json_path = OUT_DIR / "baseline_contract_immutability.json"
    md_path = OUT_DIR / "baseline_contract_immutability.md"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md_path.write_text(markdown_report(report), encoding="utf-8")
    print(
        json.dumps(
            {
                "json": str(json_path.relative_to(ROOT)),
                "markdown": str(md_path.relative_to(ROOT)),
                "overall_pass": report["overall_pass"],
                "row_statuses": report["row_statuses"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if report["overall_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
