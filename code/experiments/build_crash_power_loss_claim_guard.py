#!/usr/bin/env python3
"""C4 claim guard for crash, power-loss, kernel-crash, and drive-cache wording."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = ROOT / "artifacts" / "validation" / "crash_power_loss_claim_guard"

DANGER_PATTERNS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"power[- ]loss safe",
        r"power[- ]loss[- ]equivalent",
        r"crash certified",
        r"full crash certification",
        r"crash certification",
        r"power[- ]loss certification",
        r"complete (?:database|app(?:lication)?|crash).*certification",
        r"physical power[- ]loss",
        r"kernel[- ]crash",
        r"drive[- ]cache",
    )
]

GUARD_PATTERNS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"does not (?:claim|establish|prove|assert)",
        r"not (?:physical|a physical|complete|a formal|claimed|certification|.*proof|.*evidence)",
        r"no .*certification.*claimed",
        r"(?:remain|remains|still remains) open",
        r"(?:still )?lacks",
        r"(?:would|will|still) need",
        r"gap",
        r"outside",
        r"unsupported",
        r"boundary",
        r"does not establish",
        r"still missing",
        r"아직 더 검증해야 하는 것",
        r"앞으로 검증해야 할 주장",
        r"selected[- ](?:boundary|daemon|cutpoint)",
        r"cutpoint",
        r"not .*(?:power[- ]loss|kernel[- ]crash|drive[- ]cache)",
    )
]


@dataclass
class Finding:
    path: str
    line: int
    text: str
    matched: list[str]
    guarded: bool
    guard_context: str


def relpath(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def now_utc() -> str:
    return datetime.now().astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def scan_paths() -> list[Path]:
    paths: list[Path] = []
    paths.extend(sorted((ROOT / "Paper").glob("*.tex")))
    paths.append(ROOT / "README.md")
    docs = ROOT / "docs" / "architecture"
    if docs.exists():
        paths.extend(sorted(docs.glob("*.md")))
    return [path for path in paths if path.exists()]


def pattern_names(patterns: Iterable[re.Pattern[str]], text: str) -> list[str]:
    return [pattern.pattern for pattern in patterns if pattern.search(text)]


def guarded(context: str) -> bool:
    return bool(pattern_names(GUARD_PATTERNS, context))


def scan_file(path: Path) -> list[Finding]:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    findings: list[Finding] = []
    for index, line in enumerate(lines):
        matched = pattern_names(DANGER_PATTERNS, line)
        if not matched:
            continue
        start = max(0, index - 12)
        end = min(len(lines), index + 2)
        context = " ".join(lines[start:end])
        findings.append(Finding(
            path=relpath(path),
            line=index + 1,
            text=line.strip(),
            matched=matched,
            guarded=guarded(context),
            guard_context=context.strip(),
        ))
    return findings


def write_outputs(out_dir: Path, payload: dict[str, object]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "crash_power_loss_claim_guard.json"
    md_path = out_dir / "crash_power_loss_claim_guard.md"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n",
                         encoding="utf-8")
    lines = [
        "# Crash/Power-Loss Claim Guard",
        "",
        f"- Generated: `{payload['generated_utc']}`",
        f"- Overall pass: `{str(payload['overall_pass']).lower()}`",
        f"- Candidate lines: `{payload['candidate_count']}`",
        f"- Unguarded lines: `{payload['unguarded_count']}`",
        "",
        "## Unguarded Findings",
        "",
    ]
    unguarded = payload["unguarded_findings"]
    if isinstance(unguarded, list) and unguarded:
        for finding in unguarded:
            if isinstance(finding, dict):
                lines.append(
                    f"- `{finding['path']}:{finding['line']}`: "
                    f"{finding['text']}"
                )
    else:
        lines.append("- None.")
    lines.extend([
        "",
        "## Boundary",
        "",
        str(payload["boundary"]),
        "",
    ])
    md_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    findings: list[Finding] = []
    for path in scan_paths():
        findings.extend(scan_file(path))

    unguarded_findings = [finding for finding in findings if not finding.guarded]
    source_text = "\n".join(
        path.read_text(encoding="utf-8", errors="replace")
        for path in scan_paths()
    )
    required_boundary_terms = {
        "physical_power_loss_limit": "physical power-loss" in source_text,
        "kernel_crash_limit": "kernel-crash" in source_text,
        "drive_cache_limit": "drive-cache" in source_text,
        "selected_cutpoint_scope": "selected" in source_text and "cutpoint" in source_text,
    }
    no_power_loss_equivalent = not re.search(
        r"power[- ]loss[- ]equivalent", source_text, flags=re.IGNORECASE
    )

    payload: dict[str, object] = {
        "schema_version": 1,
        "generated_by": "code/experiments/build_crash_power_loss_claim_guard.py",
        "generated_utc": now_utc(),
        "scanned_paths": [relpath(path) for path in scan_paths()],
        "candidate_count": len(findings),
        "guarded_count": len(findings) - len(unguarded_findings),
        "unguarded_count": len(unguarded_findings),
        "no_power_loss_equivalent_wording": no_power_loss_equivalent,
        "required_boundary_terms": required_boundary_terms,
        "findings": [finding.__dict__ for finding in findings],
        "unguarded_findings": [finding.__dict__ for finding in unguarded_findings],
        "overall_pass": (
            not unguarded_findings and
            no_power_loss_equivalent and
            all(required_boundary_terms.values())
        ),
        "boundary": (
            "Crash/recovery wording is allowed only as selected final-binary "
            "daemon/app cutpoint evidence. Physical power loss, kernel crash, "
            "drive-cache behavior, arbitrary workloads, and complete crash or "
            "database certification must remain explicit non-claims unless a "
            "future runner supplies that evidence."
        ),
    }
    write_outputs(args.out_dir, payload)
    print(json.dumps({
        "overall_pass": payload["overall_pass"],
        "json": relpath(args.out_dir / "crash_power_loss_claim_guard.json"),
        "candidate_count": payload["candidate_count"],
        "unguarded_count": payload["unguarded_count"],
        "no_power_loss_equivalent_wording": no_power_loss_equivalent,
    }, indent=2, sort_keys=True))
    return 0 if payload["overall_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
