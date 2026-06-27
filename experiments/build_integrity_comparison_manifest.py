#!/usr/bin/env python3
"""Build a non-numeric fs-verity/dm-integrity comparison manifest.

The checklist allows this item to close without throughput numbers when a
matched workload is invalid.  On this host the kernel disables both
CONFIG_FS_VERITY and CONFIG_DM_INTEGRITY, so the artifact records protection
boundaries, update models, and why a throughput number would be misleading.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "artifacts" / "validation" / "integrity_comparison_manifest"

KERNEL_BASELINE = ROOT / "artifacts" / "validation" / "kernel_baseline_feasibility" / "kernel_baseline_feasibility.json"
INTEGRITY_SCOPE = ROOT / "artifacts" / "validation" / "integrity_scope_audit" / "integrity_scope_audit.json"
RELATED_WORK = ROOT / "Paper" / "5_Related_Works.tex"
DISCUSSION = ROOT / "Paper" / "10_Discussion_and_Limitations.tex"


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
        for path in (RELATED_WORK, DISCUSSION)
        if path.exists()
    )


def build_manifest() -> dict[str, Any]:
    kernel = load_json(KERNEL_BASELINE)
    scope = load_json(INTEGRITY_SCOPE)
    kvalues = ((kernel.get("host") or {}).get("kernel_config") or {}).get("values") or {}

    systems = [
        {
            "system": "fs-verity",
            "required_kernel_config": "CONFIG_FS_VERITY",
            "kernel_config_value": kvalues.get("CONFIG_FS_VERITY"),
            "runnable_on_current_host": kvalues.get("CONFIG_FS_VERITY") in {"y", "m"},
            "protection_boundary": (
                "Per-file read-only authenticity for file contents using a Merkle tree "
                "whose root is supplied to the filesystem."
            ),
            "update_model": (
                "Files become immutable after verity is enabled; mutable overwrite, "
                "journal/checkpoint publication, and SQLite-style updates are not the "
                "same workload."
            ),
            "why_no_throughput_number": (
                "A fio overwrite/fdatasync number would measure an invalid update model: "
                "fs-verity protects sealed read-only files, while AEGIS-Q evaluates "
                "mutable encrypted records."
            ),
        },
        {
            "system": "dm-integrity",
            "required_kernel_config": "CONFIG_DM_INTEGRITY",
            "kernel_config_value": kvalues.get("CONFIG_DM_INTEGRITY"),
            "runnable_on_current_host": kvalues.get("CONFIG_DM_INTEGRITY") in {"y", "m"},
            "protection_boundary": (
                "Block-device integrity tags below the filesystem, optionally composed "
                "with dm-crypt/LUKS authenticated encryption."
            ),
            "update_model": (
                "Block-level tag updates and dm-integrity journal or bitmap policy; no "
                "per-file envelope, checkpoint, or application oracle semantics."
            ),
            "why_no_throughput_number": (
                "A number would depend on unavailable kernel support and on a chosen "
                "tag/journal mode, and would not isolate AEGIS-Q's per-file mutable "
                "FUSE publication protocol."
            ),
        },
        {
            "system": "AEGIS-Q",
            "required_kernel_config": "none",
            "kernel_config_value": "userspace FUSE prototype",
            "runnable_on_current_host": True,
            "protection_boundary": (
                "Per-file encrypted record format with AEAD tags, HMAC-authenticated "
                "envelope/checkpoint metadata, and optional external freshness anchor."
            ),
            "update_model": (
                "Mutable FUSE reads/writes with data-before-journal/checkpoint publication; "
                "selected recovery states use explicit application oracles."
            ),
            "why_no_throughput_number": (
                "The existing AEGIS-Q throughput row is a mutable secure-storage result, "
                "not an integrity-only comparison against read-only fs-verity or "
                "block-tag dm-integrity."
            ),
        },
    ]

    text = paper_text()
    paper_gate = {
        "fsverity_boundary_stated": "fs-verity is read-only per-file verification" in text,
        "dmintegrity_boundary_stated": "dm-integrity is block-layer tag/journaling" in text,
        "throughput_noncomparison_stated": "not throughput baselines for AEGIS-Q" in text,
    }
    checks = {
        "kernel_config_records_fsverity_disabled": kvalues.get("CONFIG_FS_VERITY") == "not_set",
        "kernel_config_records_dmintegrity_disabled": kvalues.get("CONFIG_DM_INTEGRITY") == "not_set",
        "aegisq_integrity_scope_audit_pass": scope.get("overall_pass") is True,
        "all_required_systems_present": {row["system"] for row in systems} == {"fs-verity", "dm-integrity", "AEGIS-Q"},
        "all_rows_explain_boundary_update_and_no_number": all(
            row.get("protection_boundary")
            and row.get("update_model")
            and row.get("why_no_throughput_number")
            for row in systems
        ),
        "paper_scope_gate_pass": all(paper_gate.values()),
    }

    return {
        "schema_version": 1,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "overall_pass": all(checks.values()),
        "scope": (
            "Non-numeric integrity-oriented comparison.  The current Thor kernel "
            "does not expose fs-verity or dm-integrity, and their protection/update "
            "models are not matched throughput baselines for mutable AEGIS-Q FUSE "
            "records."
        ),
        "checks": checks,
        "paper_scope_gate": paper_gate,
        "source_artifacts": [
            relpath(KERNEL_BASELINE),
            relpath(INTEGRITY_SCOPE),
        ],
        "systems": systems,
        "non_claims": [
            "no fs-verity throughput result",
            "no dm-integrity throughput result",
            "no persisted per-file content Merkle tree in AEGIS-Q",
            "no replacement claim for kernel integrity mechanisms",
        ],
    }


def markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Integrity Comparison Manifest",
        "",
        f"- Overall pass: `{str(report['overall_pass']).lower()}`",
        f"- Scope: {report['scope']}",
        "",
        "## Checks",
        "",
    ]
    for key, value in report["checks"].items():
        lines.append(f"- `{key}`: `{str(value).lower()}`")
    lines.extend(["", "## Systems", ""])
    for row in report["systems"]:
        lines.extend(
            [
                f"### {row['system']}",
                "",
                f"- Kernel config: `{row['required_kernel_config']}` = `{row['kernel_config_value']}`",
                f"- Runnable on current host: `{str(row['runnable_on_current_host']).lower()}`",
                f"- Protection boundary: {row['protection_boundary']}",
                f"- Update model: {row['update_model']}",
                f"- Why no throughput number: {row['why_no_throughput_number']}",
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
    json_path = out / "integrity_comparison_manifest.json"
    md_path = out / "integrity_comparison_manifest.md"
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
