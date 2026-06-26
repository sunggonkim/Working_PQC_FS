#!/usr/bin/env python3
"""Build a SQLite recovery-oracle and durable-boundary report from retained artifacts.

This script does not claim crash certification and does not inject faults.
It turns the existing SQLite WAL/commit samples plus the retained strace into a
concrete oracle contract for the next fault campaign:

  * which durable boundaries were observed,
  * which cut points should be injected,
  * which post-replay checks define an acceptable recovery verdict.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "artifacts" / "validation" / "sqlite_recovery_oracle"


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def digest_rows(rows: list[dict[str, str]]) -> str:
    blob = json.dumps(rows, sort_keys=True).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def summarize_sqlite_rows(path: Path) -> dict[str, Any]:
    rows = load_csv(path)
    return {
        "path": str(path.relative_to(ROOT)),
        "sha256": digest_rows(rows),
        "row_count": len(rows),
        "rows": [
            {
                "tier": row.get("tier", ""),
                "requested_mode": row.get("requested_mode", ""),
                "actual_mode": row.get("actual_mode", ""),
                "sync_mode": row.get("sync_mode", ""),
                "integrity_check": row.get("integrity_check", ""),
                "sample_count": len(json.loads(row.get("samples_ms") or "[]")),
                "median_ms": row.get("median_ms", ""),
                "p95_ms": row.get("p95_ms", ""),
                "fallback_error": row.get("fallback_error", ""),
                "error": row.get("error", ""),
            }
            for row in rows
        ],
    }


def parse_strace(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    interesting = []
    counts = {
        "open_db": 0,
        "open_journal": 0,
        "open_wal": 0,
        "open_shm": 0,
        "fdatasync": 0,
        "fsync": 0,
        "pwrite": 0,
        "unlink_journal": 0,
    }
    fd_paths: dict[str, str] = {}
    open_re = re.compile(r'openat\(.*?, "([^"]+)", .*?\) = (-?\d+)')
    for number, line in enumerate(lines, start=1):
        if "x.db" not in line and "fdatasync" not in line and "fsync" not in line and "pwrite" not in line:
            continue
        if "openat" in line:
            match = open_re.search(line)
            if match and int(match.group(2)) >= 0:
                path_name, fd = match.group(1), match.group(2)
                fd_paths[fd] = path_name
                if path_name.endswith("x.db"):
                    counts["open_db"] += 1
                elif path_name.endswith("x.db-journal"):
                    counts["open_journal"] += 1
                elif path_name.endswith("x.db-wal"):
                    counts["open_wal"] += 1
                elif path_name.endswith("x.db-shm"):
                    counts["open_shm"] += 1
        if "fdatasync(" in line:
            counts["fdatasync"] += 1
        if "fsync(" in line:
            counts["fsync"] += 1
        if "pwrite64(" in line:
            counts["pwrite"] += 1
        if "unlinkat" in line and "x.db-journal" in line:
            counts["unlink_journal"] += 1
        interesting.append({"line": number, "text": line})
    return {
        "path": str(path.relative_to(ROOT)),
        "line_count": len(lines),
        "interesting_count": len(interesting),
        "counts": counts,
        "fd_paths": fd_paths,
        "interesting_excerpt": interesting[:120],
    }


def build_cut_points(strace_summary: dict[str, Any]) -> list[dict[str, str]]:
    observed = strace_summary["counts"]
    return [
        {
            "id": "before_sqlite_journal_header_sync",
            "observed_basis": "journal file open + pwrite64 + first fdatasync",
            "required_observation": "open_journal>0, pwrite>0, fdatasync>0",
            "permitted_recovery": "previous committed DB state or fail closed",
            "bug_signal": "new DB contents reachable without durable journal header",
            "observed_in_strace": str(observed["open_journal"] > 0 and observed["pwrite"] > 0 and observed["fdatasync"] > 0),
        },
        {
            "id": "after_journal_header_before_db_sync",
            "observed_basis": "journal fdatasync before database fdatasync",
            "required_observation": "multiple fdatasync calls including journal and db fds",
            "permitted_recovery": "previous state, latest fully committed state, or fail closed",
            "bug_signal": "SQLite integrity_check != ok or unexpected row digest",
            "observed_in_strace": str(observed["fdatasync"] >= 2),
        },
        {
            "id": "after_db_fsync_before_journal_unlink",
            "observed_basis": "database pwrite/fdatasync followed by journal unlink",
            "required_observation": "fdatasync and unlink_journal",
            "permitted_recovery": "latest committed state or fail closed",
            "bug_signal": "journal removed while DB content is partial",
            "observed_in_strace": str(observed["fdatasync"] > 0 and observed["unlink_journal"] > 0),
        },
        {
            "id": "wal_file_created_before_checkpoint",
            "observed_basis": "WAL and SHM sidecar open after journal bootstrap",
            "required_observation": "open_wal>0 and open_shm>0",
            "permitted_recovery": "SQLite WAL recovery to a state with integrity_check=ok",
            "bug_signal": "WAL exists but integrity_check fails or expected rows disappear silently",
            "observed_in_strace": str(observed["open_wal"] > 0 and observed["open_shm"] > 0),
        },
    ]


def build_oracle(workload: dict[str, Any], contention: dict[str, Any]) -> dict[str, Any]:
    all_rows = workload["rows"] + contention["rows"]
    integrity_values = sorted({row["integrity_check"] for row in all_rows})
    return {
        "post_replay_checks": [
            "Open SQLite database read-only and read-write.",
            "Run PRAGMA integrity_check and require result exactly 'ok'.",
            "Record expected table names, row counts, and ORDER BY primary-key content digest before the fault.",
            "After replay, accept only previous committed digest, latest committed digest, or explicit fail-closed error.",
            "Classify any other readable digest as silent corruption.",
        ],
        "retained_sample_integrity_values": integrity_values,
        "retained_sample_rows": len(all_rows),
        "retained_sample_digests": {
            "sqlite_latency_csv": workload["sha256"],
            "sqlite_contention_latency_csv": contention["sha256"],
        },
    }


def write_markdown(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# SQLite recovery oracle",
        "",
        "This report defines durable-boundary cut points and the recovery oracle for a future SQLite fault-injection campaign.",
        "It is derived from retained SQLite samples and `artifacts/sqlite_strace.log`; it does not claim crash certification.",
        "",
        "## Retained SQLite samples",
        "",
    ]
    for section_name in ("workload", "contention"):
        section = report[section_name]
        lines.append(f"### {section_name}")
        lines.append(f"- Source: `{section['path']}`")
        lines.append(f"- Rows: `{section['row_count']}`")
        lines.append(f"- SHA-256: `{section['sha256']}`")
        for row in section["rows"]:
            lines.append(
                f"- {row['tier']}: {row['requested_mode']}/{row['actual_mode']} "
                f"{row['sync_mode']}, integrity={row['integrity_check']}, "
                f"samples={row['sample_count']}"
            )
        lines.append("")
    lines.extend(["## Strace durable-boundary observations", ""])
    counts = report["strace"]["counts"]
    for key, value in counts.items():
        lines.append(f"- {key}: `{value}`")
    lines.extend(["", "## Cut points", ""])
    for cut in report["cut_points"]:
        lines.append(f"### {cut['id']}")
        lines.append(f"- Observed basis: {cut['observed_basis']}")
        lines.append(f"- Observed in strace: `{cut['observed_in_strace']}`")
        lines.append(f"- Permitted recovery: {cut['permitted_recovery']}")
        lines.append(f"- Bug signal: {cut['bug_signal']}")
        lines.append("")
    lines.extend(["## Oracle", ""])
    for item in report["oracle"]["post_replay_checks"]:
        lines.append(f"- {item}")
    lines.extend(
        [
            "",
            "## Conservative interpretation",
            "",
            "- This closes the oracle/cut-point definition step for the current SQLite path.",
            "- It does not close the fault-injection campaign requirement because no per-cut replay was executed here.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    out_dir = args.out_dir if args.out_dir.is_absolute() else ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    workload = summarize_sqlite_rows(ROOT / "artifacts" / "motivation" / "sqlite_latency.csv")
    contention = summarize_sqlite_rows(ROOT / "artifacts" / "motivation" / "sqlite_contention_latency.csv")
    strace = parse_strace(ROOT / "artifacts" / "sqlite_strace.log")
    report = {
        "note": "Oracle-definition artifact only; not a crash-certification claim.",
        "workload": workload,
        "contention": contention,
        "strace": strace,
        "cut_points": build_cut_points(strace),
        "oracle": build_oracle(workload, contention),
    }
    json_path = out_dir / "sqlite_recovery_oracle.json"
    md_path = out_dir / "sqlite_recovery_oracle.md"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_markdown(report, md_path)
    print(json.dumps({"report": str(json_path.relative_to(ROOT)), "cut_points": len(report["cut_points"])}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
