#!/usr/bin/env python3
"""Reject unprofiled production mutex locks and condition waits."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


FORBIDDEN = (
    r"\bpthread_mutex_lock\s*\(",
    r"\bpthread_mutex_unlock\s*\(",
    r"\bpthread_cond_wait\s*\(",
    r"\bpthread_cond_timedwait\s*\(",
)

ALLOWED_FILES = {
    "code/support/pqc_lock_profile.c",
    "code/support/pqc_selftest.c",
}

HELPER_ONLY_FILES = {
    "code/runtime/pqc_config.c": (
        "config_lock",
        "config_unlock",
    ),
}


def is_helper_body(line_no: int, lines: list[str], helper_names: tuple[str, ...]) -> bool:
    window_start = max(0, line_no - 12)
    text = "\n".join(lines[window_start:line_no])
    return any(re.search(rf"\b{re.escape(name)}\s*\(", text) for name in helper_names)


def scan_file(repo_root: Path, path: Path) -> list[dict[str, object]]:
    rel = path.relative_to(repo_root).as_posix()
    if rel in ALLOWED_FILES:
        return []

    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    findings: list[dict[str, object]] = []
    helper_names = HELPER_ONLY_FILES.get(rel)

    for idx, line in enumerate(lines, start=1):
        for pattern in FORBIDDEN:
            if not re.search(pattern, line):
                continue
            if helper_names and is_helper_body(idx, lines, helper_names):
                continue
            findings.append({
                "path": rel,
                "line": idx,
                "pattern": pattern,
                "text": line.strip(),
            })

    return findings


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=".", type=Path)
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    findings: list[dict[str, object]] = []
    checked: list[str] = []

    for path in sorted((repo_root / "code").rglob("*")):
        if path.suffix not in {".c", ".h"}:
            continue
        rel = path.relative_to(repo_root).as_posix()
        checked.append(rel)
        findings.extend(scan_file(repo_root, path))

    report = {
        "overall_pass": not findings,
        "checked_files": checked,
        "checked_file_count": len(checked),
        "forbidden_findings": findings,
        "allowed_files": sorted(ALLOWED_FILES),
        "helper_only_files": sorted(HELPER_ONLY_FILES),
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["overall_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
