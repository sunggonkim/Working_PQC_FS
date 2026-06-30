#!/usr/bin/env python3
"""Mounted-path proof for Gate 0.9-S2 epoch redo-log append/barrier."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import struct
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from run_parallel_commit_epoch_path_smoke import (  # noqa: E402
    FUSE_BIN,
    ROOT,
    mounted_write_read,
    relpath,
    start_fuse,
    stop_fuse,
)


DEFAULT_OUT = ROOT / "artifacts" / "validation" / "publication_protocol_fault_matrix"
EPOCH_RECORD_SIZE = 136
EPOCH_MAGIC = 0x50514345504C3031
EPOCH_RECORD_BLOCK = 1
EPOCH_RECORD_COMMIT = 2


def now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_jsonl(path: Path) -> dict[str, Any]:
    events: list[dict[str, Any]] = []
    malformed = 0
    if path.exists():
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                malformed += 1
                continue
            if isinstance(payload, dict):
                events.append(payload)
    return {
        "path": relpath(path),
        "exists": path.exists(),
        "event_count": len(events),
        "malformed_line_count": malformed,
        "events": events,
        "event_names": [str(event.get("event", "")) for event in events],
        "modes": sorted({str(event.get("mode", "")) for event in events
                         if event.get("mode")}),
    }


def scan_epoch_logs(storage_dir: Path) -> dict[str, Any]:
    files: list[dict[str, Any]] = []
    total_records = 0
    total_block_records = 0
    total_commit_records = 0
    malformed_records = 0
    for path in sorted(storage_dir.rglob("*.pqcepoch")):
        data = path.read_bytes()
        record_count = len(data) // EPOCH_RECORD_SIZE
        file_block_records = 0
        file_commit_records = 0
        file_malformed = 0
        for idx in range(record_count):
            record = data[idx * EPOCH_RECORD_SIZE:(idx + 1) * EPOCH_RECORD_SIZE]
            magic = struct.unpack_from("<Q", record, 0)[0]
            record_type = struct.unpack_from("<I", record, 12)[0]
            if magic != EPOCH_MAGIC:
                file_malformed += 1
            elif record_type == EPOCH_RECORD_BLOCK:
                file_block_records += 1
            elif record_type == EPOCH_RECORD_COMMIT:
                file_commit_records += 1
            else:
                file_malformed += 1
        trailing_bytes = len(data) % EPOCH_RECORD_SIZE
        if trailing_bytes:
            file_malformed += 1
        total_records += record_count
        total_block_records += file_block_records
        total_commit_records += file_commit_records
        malformed_records += file_malformed
        files.append({
            "path": relpath(path),
            "size_bytes": len(data),
            "record_count": record_count,
            "block_records": file_block_records,
            "commit_records": file_commit_records,
            "malformed_records": file_malformed,
            "trailing_bytes": trailing_bytes,
        })
    return {
        "file_count": len(files),
        "files": files,
        "record_count": total_records,
        "block_records": total_block_records,
        "commit_records": total_commit_records,
        "malformed_records": malformed_records,
    }


def run_case(case_dir: Path, label: str,
             extra_env: dict[str, str] | None = None) -> dict[str, Any]:
    storage_dir = Path(tempfile.mkdtemp(prefix=f"aegisq_{label}_storage_"))
    mount_dir = Path(tempfile.mkdtemp(prefix=f"aegisq_{label}_mnt_"))
    case_dir.mkdir(parents=True, exist_ok=True)
    trace_path = case_dir / "publication_trace.jsonl"
    if trace_path.exists():
        trace_path.unlink()
    env = {
        "PQC_PUBLICATION_TRACE_PATH": str(trace_path),
    }
    if extra_env:
        env.update(extra_env)

    fuse = None
    unmount: dict[str, Any] = {}
    client: dict[str, Any] = {}
    listing: list[str] = []
    error: str | None = None
    try:
        fuse = start_fuse(storage_dir, mount_dir, case_dir,
                          f"epoch-redo-log-{label}", env)
        payload = (f"epoch-redo-log-smoke:{label}:".encode("ascii") +
                   bytes((i % 251 for i in range(16384))))
        client = mounted_write_read(mount_dir, "epoch-smoke.dat", payload)
        listing = sorted(os.listdir(mount_dir))
    except Exception as exc:  # noqa: BLE001 - retained as artifact evidence
        error = repr(exc)
    finally:
        try:
            unmount = stop_fuse(fuse, mount_dir, case_dir)
        finally:
            epoch_logs = scan_epoch_logs(storage_dir)
            shutil.rmtree(mount_dir, ignore_errors=True)
            shutil.rmtree(storage_dir, ignore_errors=True)

    trace = parse_jsonl(trace_path)
    hidden_epoch_sidecar_visible = any(name.endswith(".pqcepoch")
                                       for name in listing)
    strict_expected = label == "strict"
    if strict_expected:
        pass_condition = (
            error is None and
            client.get("matches") is True and
            unmount.get("returncode") == 0 and
            epoch_logs["file_count"] == 0 and
            not hidden_epoch_sidecar_visible
        )
    else:
        append_events = [
            event for event in trace["events"]
            if event.get("event") == "epoch_redo_log_append"
        ]
        pass_condition = (
            error is None and
            client.get("matches") is True and
            unmount.get("returncode") == 0 and
            epoch_logs["file_count"] >= 1 and
            epoch_logs["block_records"] >= 1 and
            epoch_logs["commit_records"] >= 1 and
            epoch_logs["malformed_records"] == 0 and
            append_events and
            all(int(event.get("rc", -1)) == 0 for event in append_events) and
            not hidden_epoch_sidecar_visible
        )
    return {
        "label": label,
        "client": client,
        "unmount": unmount,
        "mount_listing": listing,
        "hidden_epoch_sidecar_visible": hidden_epoch_sidecar_visible,
        "epoch_logs": epoch_logs,
        "trace": trace,
        "error": error,
        "pass": pass_condition,
    }


def write_outputs(out_dir: Path, payload: dict[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "epoch_redo_log_mounted_smoke.json"
    md_path = out_dir / "epoch_redo_log_mounted_smoke.md"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n",
                         encoding="utf-8")
    lines = [
        "# Epoch Redo-Log Mounted Smoke",
        "",
        f"- Generated: `{payload['generated_utc']}`",
        f"- Overall pass: `{str(payload['overall_pass']).lower()}`",
        "",
    ]
    for case in payload["cases"]:
        lines.append(f"## {case['label']}")
        lines.append(f"- Pass: `{str(case['pass']).lower()}`")
        lines.append(f"- Read matches write: `{str(case['client'].get('matches')).lower()}`")
        lines.append(f"- Epoch log files: `{case['epoch_logs']['file_count']}`")
        lines.append(f"- Block records: `{case['epoch_logs']['block_records']}`")
        lines.append(f"- Commit records: `{case['epoch_logs']['commit_records']}`")
        lines.append("")
    lines.extend([
        "## Negative Claim Guard",
        "",
        payload["negative_claim_guard"],
        "",
    ])
    md_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    if not FUSE_BIN.exists():
        raise SystemExit(f"missing build artifact: {relpath(FUSE_BIN)}")

    strict = run_case(args.out_dir / "strict_no_epoch_log", "strict")
    epoch = run_case(
        args.out_dir / "epoch_redo_log",
        "epoch_redo_log",
        {"PQC_PUBLICATION_MODE": "epoch-redo-log"},
    )
    payload = {
        "schema_version": 1,
        "generated_by": "experiments/run_epoch_redo_log_mounted_smoke.py",
        "generated_utc": now_utc(),
        "scope": "Gate 0.9-S2 mounted-path epoch redo-log append/barrier proof.",
        "cases": [strict, epoch],
        "overall_pass": strict["pass"] and epoch["pass"],
        "negative_claim_guard": (
            "This smoke proves that epoch-redo-log mode appends block and "
            "commit records to a per-file .pqcepoch sidecar and reaches a "
            "log fdatasync barrier before strict journal publication continues. "
            "It does not prove checkpoint compaction, crash replay, sync-count "
            "reduction, throughput improvement, or rollback resistance."
        ),
    }
    write_outputs(args.out_dir, payload)
    print(json.dumps({
        "overall_pass": payload["overall_pass"],
        "json": relpath(args.out_dir / "epoch_redo_log_mounted_smoke.json"),
        "strict_epoch_log_files": strict["epoch_logs"]["file_count"],
        "epoch_record_count": epoch["epoch_logs"]["record_count"],
    }, indent=2, sort_keys=True))
    return 0 if payload["overall_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
