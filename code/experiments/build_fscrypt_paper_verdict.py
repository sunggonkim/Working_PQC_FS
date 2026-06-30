#!/usr/bin/env python3
"""Build the Gate B1 paper-facing fscrypt verdict.

This script does not run an fscrypt benchmark. It verifies that the retained
kernel-baseline feasibility artifact blocks fscrypt with kernel/filesystem
evidence, then checks that paper text reflects that boundary without implying a
measured fscrypt speedup row.
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FEASIBILITY = ROOT / "artifacts" / "validation" / "kernel_baseline_feasibility" / "kernel_baseline_feasibility.json"
DEFAULT_OUT = ROOT / "artifacts" / "validation" / "kernel_baseline_feasibility"
DEFAULT_PAPER = ROOT / "Paper"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def paper_text(paper_dir: Path) -> tuple[str, list[tuple[Path, int, str]]]:
    joined: list[str] = []
    rows: list[tuple[Path, int, str]] = []
    for path in sorted(paper_dir.rglob("*.tex")):
        text = path.read_text(encoding="utf-8", errors="replace")
        joined.append(text)
        for lineno, line in enumerate(text.splitlines(), 1):
            if "fscrypt" in line:
                rows.append((path, lineno, line.strip()))
    return "\n".join(joined), rows


def scan_paper(paper_dir: Path) -> dict[str, Any]:
    text, rows = paper_text(paper_dir)
    dangerous_patterns = [
        r"speedup[^.\n]*fscrypt",
        r"fscrypt[^.\n]*speedup",
        r"faster than[^.\n]*fscrypt",
        r"beats?[^.\n]*fscrypt",
        r"outperform[^.\n]*fscrypt",
        r"fscrypt[^.\n]*MiB/s",
    ]
    dangerous: list[dict[str, Any]] = []
    for path, lineno, line in rows:
        for pattern in dangerous_patterns:
            if re.search(pattern, line, flags=re.IGNORECASE):
                dangerous.append({
                    "path": rel(path),
                    "line": lineno,
                    "pattern": pattern,
                    "text": line,
                })
    return {
        "paper_dir": rel(paper_dir),
        "fscrypt_lines": [
            {"path": rel(path), "line": lineno, "text": line}
            for path, lineno, line in rows
        ],
        "mentions_fscrypt": bool(rows),
        "mentions_environment_blocked": "environment-blocked" in text or "environment blocked" in text,
        "mentions_unavailable": "unavailable" in text,
        "mentions_kernel_proof": (
            "CONFIG_FS_ENCRYPTION" in text
            or "CONFIG\\_FS\\_ENCRYPTION" in text
            or "kernel config" in text
        ),
        "mentions_filesystem_proof": (
            "root ext4" in text
            or "encrypt feature" in text
            or "filesystem proof" in text
            or "fscrypt status" in text
        ),
        "dangerous_fscrypt_claims": dangerous,
    }


def fscrypt_status(feasibility: dict[str, Any]) -> dict[str, Any]:
    fscrypt = feasibility.get("fscrypt") or {}
    host = feasibility.get("host") or {}
    kernel_config = ((host.get("kernel_config") or {}).get("values") or {})
    status_repo = host.get("fscrypt_status_repo") or {}
    reasons = list(fscrypt.get("blocking_reasons") or [])
    measured = bool(
        fscrypt.get("runnable_without_interactive_root")
        or fscrypt.get("runnable_with_sudo_password")
    )
    environment_blocked = (
        not measured
        and "kernel_config_fs_encryption_disabled" in reasons
        and "root_ext4_encrypt_feature_not_enabled" in reasons
    )
    return {
        "measured": measured,
        "environment_blocked": environment_blocked,
        "blocking_reasons": reasons,
        "kernel_config_fs_encryption": kernel_config.get("CONFIG_FS_ENCRYPTION"),
        "fscrypt_status_returncode": status_repo.get("returncode"),
        "fscrypt_status_stderr": status_repo.get("stderr"),
        "runnable_without_interactive_root": bool(fscrypt.get("runnable_without_interactive_root")),
        "runnable_with_sudo_password": bool(fscrypt.get("runnable_with_sudo_password")),
    }


def write_markdown(report: dict[str, Any], path: Path) -> None:
    status = report["fscrypt_status"]
    paper = report["paper_scan"]
    lines = [
        "# fscrypt Paper Verdict",
        "",
        f"- Overall pass: `{str(report['overall_pass']).lower()}`",
        f"- Parent B1 gate closed: `{str(report['parent_b1_gate_closed']).lower()}`",
        f"- Measured row: `{str(status['measured']).lower()}`",
        f"- Environment-blocked row: `{str(status['environment_blocked']).lower()}`",
        f"- Blocking reasons: `{', '.join(status['blocking_reasons'])}`",
        f"- Kernel CONFIG_FS_ENCRYPTION: `{status['kernel_config_fs_encryption']}`",
        f"- Paper marks unavailable: `{str(report['paper_marks_fscrypt_unavailable_with_proof']).lower()}`",
        f"- Dangerous paper claims: `{len(paper['dangerous_fscrypt_claims'])}`",
        "",
        "## Paper fscrypt Lines",
        "",
    ]
    for row in paper["fscrypt_lines"]:
        lines.append(f"- `{row['path']}:{row['line']}` {row['text']}")
    lines.extend([
        "",
        "## Non-Claims",
        "",
    ])
    for claim in report["negative_claim_guard"]["forbidden_until_measured"]:
        lines.append(f"- {claim}")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feasibility", type=Path, default=DEFAULT_FEASIBILITY)
    parser.add_argument("--paper-dir", type=Path, default=DEFAULT_PAPER)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    args.feasibility = args.feasibility if args.feasibility.is_absolute() else ROOT / args.feasibility
    args.paper_dir = args.paper_dir if args.paper_dir.is_absolute() else ROOT / args.paper_dir
    args.out_dir = args.out_dir if args.out_dir.is_absolute() else ROOT / args.out_dir
    args.out_dir.mkdir(parents=True, exist_ok=True)

    feasibility = load_json(args.feasibility)
    status = fscrypt_status(feasibility)
    paper = scan_paper(args.paper_dir)
    paper_marks_unavailable = (
        paper["mentions_fscrypt"]
        and (paper["mentions_environment_blocked"] or paper["mentions_unavailable"])
        and paper["mentions_kernel_proof"]
        and paper["mentions_filesystem_proof"]
    )
    dangerous_claims_clear = len(paper["dangerous_fscrypt_claims"]) == 0
    parent_closed = (
        (status["measured"] or (status["environment_blocked"] and paper_marks_unavailable))
        and dangerous_claims_clear
    )
    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "artifact": rel(args.out_dir),
        "source_feasibility_artifact": rel(args.feasibility),
        "overall_pass": parent_closed,
        "parent_b1_gate_closed": parent_closed,
        "paper_marks_fscrypt_unavailable_with_proof": paper_marks_unavailable,
        "dangerous_claims_clear": dangerous_claims_clear,
        "fscrypt_status": status,
        "paper_scan": paper,
        "negative_claim_guard": {
            "forbidden_until_measured": [
                "AEGIS-Q is faster than fscrypt.",
                "AEGIS-Q has an fscrypt speedup.",
                "The paper reports an fscrypt MiB/s row without a measured frozen-contract artifact.",
            ],
        },
    }
    json_path = args.out_dir / "paper_fscrypt_verdict.json"
    md_path = args.out_dir / "paper_fscrypt_verdict.md"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown(report, md_path)
    print(json.dumps({
        "overall_pass": report["overall_pass"],
        "parent_b1_gate_closed": report["parent_b1_gate_closed"],
        "paper_marks_fscrypt_unavailable_with_proof": paper_marks_unavailable,
        "dangerous_claims_clear": dangerous_claims_clear,
        "json": rel(json_path),
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
