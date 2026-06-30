#!/usr/bin/env python3
"""Audit UVM/migration/coherence counter availability for the retained UMA probe.

The goal is not to prove NVMe-to-UVM DMA semantics.  The goal is narrower:
record whether the retained Nsight reports expose Unified Memory, page fault,
migration, or coherence rows for the exact profile bundle that already contains
the same-buffer O_DIRECT checksum proof.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sqlite3
import subprocess
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_IN = ROOT / "artifacts" / "validation" / "uma_storage_dma_profile_combined"
DEFAULT_OUT = ROOT / "artifacts" / "validation" / "uma_counter_availability"


NSYS_REPORTS = [
    "um_sum",
    "um_total_sum",
    "um_cpu_page_faults_sum",
    "cuda_api_sum",
]


def run_nsys_report(nsys: str, report: str, rep_path: Path) -> dict[str, Any]:
    cmd = [
        nsys,
        "stats",
        "--force-export=true",
        "--report",
        report,
        "--format",
        "csv",
        str(rep_path),
    ]
    proc = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True, check=False)
    data_lines = [
        line
        for line in proc.stdout.splitlines()
        if line.strip()
        and not line.startswith("Generating ")
        and not line.startswith("Processing ")
        and not line.startswith("SKIPPED:")
    ]
    skipped = [line for line in proc.stdout.splitlines() + proc.stderr.splitlines() if "SKIPPED:" in line]
    return {
        "report": report,
        "command": cmd,
        "returncode": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
        "data_line_count": len(data_lines),
        "skipped_messages": skipped,
    }


def inspect_sqlite(sqlite_path: Path) -> dict[str, Any]:
    if not sqlite_path.exists():
        return {"path": str(sqlite_path), "exists": False}
    conn = sqlite3.connect(str(sqlite_path))
    try:
        tables = [
            row[0]
            for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
        ]
        keywords = ("UM", "UNIFIED", "MIGR", "PAGE", "FAULT", "COHER", "ATS")
        matched = [name for name in tables if any(k in name.upper() for k in keywords)]
        event_like = [name for name in matched if not name.startswith("ENUM_")]
        counts: dict[str, int | str] = {}
        for name in matched:
            try:
                counts[name] = int(conn.execute(f'SELECT COUNT(*) FROM "{name}"').fetchone()[0])
            except sqlite3.Error as exc:
                counts[name] = f"error: {exc}"
        return {
            "path": str(sqlite_path),
            "exists": True,
            "table_count": len(tables),
            "matched_tables": matched,
            "matched_event_tables": event_like,
            "matched_table_counts": counts,
        }
    finally:
        conn.close()


def ncu_metric_query() -> dict[str, Any]:
    ncu = shutil.which("ncu")
    if not ncu:
        return {"available": False}
    cmd = [ncu, "--query-metrics"]
    proc = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True, check=False)
    needles = ("uvm", "unified", "migration", "migrate", "page", "fault", "coher")
    matches = [
        line
        for line in proc.stdout.splitlines()
        if any(needle in line.lower() for needle in needles)
    ]
    return {
        "available": True,
        "command": cmd,
        "returncode": proc.returncode,
        "matched_line_count": len(matches),
        "matched_lines": matches[:200],
        "stderr": proc.stderr,
    }


def write_markdown(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# UMA counter availability audit",
        "",
        "This artifact records whether retained Nsight outputs expose UVM/page-fault/migration/coherence counters for the same-buffer storage-visible probe.",
        "It does not prove NVMe-to-UVM DMA semantics or migration suppression.",
        "",
        "## Nsight Systems reports",
        "",
    ]
    for run in report["nsys_runs"]:
        lines.append(f"### {run['name']}")
        lines.append("")
        lines.append(f"- Report file: `{run['rep_path']}`")
        for result in run["reports"]:
            skipped = "; ".join(result["skipped_messages"]) if result["skipped_messages"] else ""
            lines.append(
                f"- `{result['report']}`: rc={result['returncode']}, "
                f"data_lines={result['data_line_count']}"
                + (f", skipped=`{skipped}`" if skipped else "")
            )
        sqlite_info = run["sqlite"]
        lines.append(f"- SQLite matched event tables: `{sqlite_info.get('matched_event_tables', [])}`")
        lines.append(f"- SQLite matched enum tables: `{sqlite_info.get('matched_tables', [])}`")
        lines.append("")

    lines.extend(["## Nsight Compute metric query", ""])
    ncu = report["ncu_metric_query"]
    if ncu.get("available"):
        lines.append(f"- Command: `{ncu['command']}`")
        lines.append(f"- Return code: `{ncu['returncode']}`")
        lines.append(f"- Matching metric lines: `{ncu['matched_line_count']}`")
    else:
        lines.append("- `ncu` not available.")

    lines.extend(
        [
            "",
            "## Conservative interpretation",
            "",
            "- The retained raw probe and managed-storage probe have same-buffer GPU checksum evidence, but the Nsight Systems UM reports do not expose UVM transfer/page-fault rows for those runs.",
            "- If the UM reports are skipped or empty, the paper may document counter non-exposure for this bundle, not claim migration suppression.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-dir", type=Path, default=DEFAULT_IN)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    in_dir = args.in_dir if args.in_dir.is_absolute() else ROOT / args.in_dir
    out_dir = args.out_dir if args.out_dir.is_absolute() else ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    nsys = shutil.which("nsys")
    nsys_runs = []
    for name, rel in [
        ("raw_probe", Path("probe/raw_probe.nsys-rep")),
        ("um_smoke", Path("um_smoke/um_smoke.nsys-rep")),
        ("managed_storage_probe", Path("managed_storage/managed_storage.nsys-rep")),
    ]:
        rep = in_dir / rel
        sqlite_path = rep.with_suffix(".sqlite")
        entry: dict[str, Any] = {
            "name": name,
            "rep_path": str(rep.relative_to(ROOT) if rep.exists() else rep),
            "exists": rep.exists(),
            "reports": [],
            "sqlite": inspect_sqlite(sqlite_path),
        }
        if nsys and rep.exists():
            entry["reports"] = [run_nsys_report(nsys, report, rep) for report in NSYS_REPORTS]
        nsys_runs.append(entry)

    report = {
        "note": "Counter-availability audit only; not a migration-suppression proof.",
        "input_dir": str(in_dir.relative_to(ROOT) if in_dir.is_relative_to(ROOT) else in_dir),
        "nsys_available": bool(nsys),
        "nsys_runs": nsys_runs,
        "ncu_metric_query": ncu_metric_query(),
    }
    json_path = out_dir / "uma_counter_availability.json"
    md_path = out_dir / "uma_counter_availability.md"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_markdown(report, md_path)
    print(json.dumps({"report": str(json_path.relative_to(ROOT)), "runs": len(nsys_runs)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
