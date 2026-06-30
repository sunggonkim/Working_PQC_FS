#!/usr/bin/env python3
"""Build the Gate B2 paper-facing dm-crypt verdict.

This script does not run dm-crypt. It verifies that the retained dm-crypt
contract result is either measured or explicitly environment-blocked, then
checks that paper text does not imply a measured dm-crypt speedup row.
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DMCRYPT = ROOT / "artifacts" / "validation" / "frozen_dmcrypt_contract" / "frozen_dmcrypt_contract.json"
DEFAULT_OUT = ROOT / "artifacts" / "validation" / "frozen_dmcrypt_contract"
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
            if "dm-crypt" in line or "dmcrypt" in line:
                rows.append((path, lineno, line.strip()))
    return "\n".join(joined), rows


def scan_paper(paper_dir: Path) -> dict[str, Any]:
    text, rows = paper_text(paper_dir)
    dangerous_patterns = [
        r"speedup[^.\n]*dm-crypt",
        r"dm-crypt[^.\n]*speedup",
        r"faster than[^.\n]*dm-crypt",
        r"beats?[^.\n]*dm-crypt",
        r"outperform[^.\n]*dm-crypt",
        r"dm-crypt[^.\n]*MiB/s",
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
        "dmcrypt_lines": [
            {"path": rel(path), "line": lineno, "text": line}
            for path, lineno, line in rows
        ],
        "mentions_dmcrypt": bool(rows),
        "mentions_environment_blocked": "environment-blocked" in text or "environment blocked" in text,
        "mentions_unavailable": "unavailable" in text,
        "mentions_noninteractive_root": "noninteractive root" in text or "non-interactive root" in text,
        "mentions_sudo_root_proof": (
            "sudo -n true" in text
            or "sudo/root" in text
            or "root/device" in text
            or "root proof" in text
        ),
        "dangerous_dmcrypt_claims": dangerous,
    }


def dmcrypt_status(dmcrypt: dict[str, Any]) -> dict[str, Any]:
    verdict = dmcrypt.get("verdict")
    reasons = list(dmcrypt.get("blocking_reasons") or [])
    warm = dmcrypt.get("warm_cache_summary") or {}
    metrics = warm.get("metrics") or {}
    measured = (
        bool(dmcrypt.get("overall_pass"))
        and verdict == "measured"
        and int(warm.get("valid_repetitions") or 0) > 0
        and bool(metrics)
    )
    environment_blocked = (
        verdict == "environment-blocked"
        and "noninteractive_root_unavailable" in reasons
    )
    return {
        "verdict": verdict,
        "overall_pass": bool(dmcrypt.get("overall_pass")),
        "blocking_reasons": reasons,
        "measured": measured,
        "environment_blocked": environment_blocked,
        "warm_valid_repetitions": int(warm.get("valid_repetitions") or 0),
        "required_privilege": dmcrypt.get("required_privilege"),
        "non_claims": list(dmcrypt.get("non_claims") or []),
    }


def write_markdown(report: dict[str, Any], path: Path) -> None:
    status = report["dmcrypt_status"]
    paper = report["paper_scan"]
    lines = [
        "# dm-crypt Paper Verdict",
        "",
        f"- Overall pass: `{str(report['overall_pass']).lower()}`",
        f"- Parent B2 gate closed: `{str(report['parent_b2_gate_closed']).lower()}`",
        f"- dm-crypt verdict: `{status['verdict']}`",
        f"- Measured row: `{str(status['measured']).lower()}`",
        f"- Environment-blocked row: `{str(status['environment_blocked']).lower()}`",
        f"- Blocking reasons: `{', '.join(status['blocking_reasons'])}`",
        f"- Paper marks unavailable: `{str(report['paper_marks_dmcrypt_unavailable_with_proof']).lower()}`",
        f"- Dangerous paper claims: `{len(paper['dangerous_dmcrypt_claims'])}`",
        "",
        "## Paper dm-crypt Lines",
        "",
    ]
    for row in paper["dmcrypt_lines"]:
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
    parser.add_argument("--dmcrypt", type=Path, default=DEFAULT_DMCRYPT)
    parser.add_argument("--paper-dir", type=Path, default=DEFAULT_PAPER)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    args.dmcrypt = args.dmcrypt if args.dmcrypt.is_absolute() else ROOT / args.dmcrypt
    args.paper_dir = args.paper_dir if args.paper_dir.is_absolute() else ROOT / args.paper_dir
    args.out_dir = args.out_dir if args.out_dir.is_absolute() else ROOT / args.out_dir
    args.out_dir.mkdir(parents=True, exist_ok=True)

    dmcrypt = load_json(args.dmcrypt)
    status = dmcrypt_status(dmcrypt)
    paper = scan_paper(args.paper_dir)
    paper_marks_unavailable = (
        paper["mentions_dmcrypt"]
        and (paper["mentions_environment_blocked"] or paper["mentions_unavailable"])
        and (paper["mentions_noninteractive_root"] or paper["mentions_sudo_root_proof"])
    )
    dangerous_claims_clear = len(paper["dangerous_dmcrypt_claims"]) == 0
    parent_closed = (
        (
            status["measured"]
            or (status["environment_blocked"] and paper_marks_unavailable)
        )
        and dangerous_claims_clear
    )
    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "artifact": rel(args.out_dir),
        "source_dmcrypt_artifact": rel(args.dmcrypt),
        "overall_pass": parent_closed,
        "parent_b2_gate_closed": parent_closed,
        "paper_marks_dmcrypt_unavailable_with_proof": paper_marks_unavailable,
        "dangerous_claims_clear": dangerous_claims_clear,
        "dmcrypt_status": status,
        "paper_scan": paper,
        "negative_claim_guard": {
            "forbidden_until_measured": [
                "AEGIS-Q is faster than dm-crypt.",
                "AEGIS-Q has a dm-crypt speedup.",
                "The paper reports a dm-crypt MiB/s row without a measured frozen-contract artifact.",
            ],
        },
    }
    json_path = args.out_dir / "paper_dmcrypt_verdict.json"
    md_path = args.out_dir / "paper_dmcrypt_verdict.md"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown(report, md_path)
    print(json.dumps({
        "overall_pass": report["overall_pass"],
        "parent_b2_gate_closed": report["parent_b2_gate_closed"],
        "paper_marks_dmcrypt_unavailable_with_proof": paper_marks_unavailable,
        "dangerous_claims_clear": dangerous_claims_clear,
        "json": rel(json_path),
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
