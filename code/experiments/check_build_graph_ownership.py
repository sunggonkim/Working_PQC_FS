#!/usr/bin/env python3
"""Reject regressions to monolithic CMake ownership files."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


ROOT_RELATIVE_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("code/source_groups.cmake", (r"\bpqc_[A-Za-z0-9_]+\.c\b", r"\.cu\b")),
    ("code/include_paths.cmake", (r"\$\{CODE_DIR\}/(?:common|frontend|crypto|storage|runtime|fs|support|gpu)\s*(?:#.*)?$",)),
    ("code/build_targets.cmake", (r"\btarget_link_libraries\s*\(", r"\btarget_include_directories\s*\(")),
    ("code/cuda_backends.cmake", (r"\bpqc_fuse_core\b", r"\bSKIM_HAVE_CUDA\b", r"\btarget_sources\s*\(")),
    ("code/experiment_targets.cmake", (r"\badd_executable\s*\(", r"\btarget_link_libraries\s*\(", r"\bset_target_properties\s*\(")),
    ("code/tests.cmake", (r"\badd_test\s*\(", r"\bset_tests_properties\s*\(")),
    ("code/summary.cmake", (r"\bmessage\s*\(",)),
)


def scan_file(path: Path, patterns: tuple[str, ...]) -> list[dict[str, object]]:
    text = path.read_text(encoding="utf-8")
    findings: list[dict[str, object]] = []
    for line_no, line in enumerate(text.splitlines(), start=1):
        for pattern in patterns:
            if re.search(pattern, line):
                findings.append({
                    "path": str(path),
                    "line": line_no,
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
    missing: list[str] = []

    for relative, patterns in ROOT_RELATIVE_RULES:
        path = repo_root / relative
        if not path.exists():
            missing.append(relative)
            continue
        findings.extend(scan_file(path, patterns))

    report = {
        "overall_pass": not findings and not missing,
        "checked_files": [relative for relative, _ in ROOT_RELATIVE_RULES],
        "missing_files": missing,
        "forbidden_findings": findings,
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["overall_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
