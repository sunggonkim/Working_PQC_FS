#!/usr/bin/env python3
"""Gate 0.9-S3 fault/remount proof for epoch replay and compaction."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from run_epoch_redo_log_mounted_smoke import (  # noqa: E402
    EPOCH_RECORD_SIZE,
    DEFAULT_OUT,
    scan_epoch_logs,
)
from run_parallel_commit_epoch_path_smoke import (  # noqa: E402
    FUSE_BIN,
    ROOT,
    mounted_write_read,
    relpath,
    start_fuse,
    stop_fuse,
)


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
        "compact_events": [
            event for event in events
            if event.get("event") == "epoch_replay_compact"
        ],
    }


def mounted_read(mount_dir: Path, name: str, expected: bytes) -> dict[str, Any]:
    path = mount_dir / name
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError as exc:
        return {
            "path": str(path),
            "opened": False,
            "errno": exc.errno,
            "error": repr(exc),
            "matches": False,
        }
    try:
        recovered = os.read(fd, len(expected))
    finally:
        os.close(fd)
    return {
        "path": str(path),
        "opened": True,
        "read_len": len(recovered),
        "matches": recovered == expected,
    }


def epoch_log_paths(storage_dir: Path) -> list[Path]:
    return sorted(storage_dir.rglob("*.pqcepoch"))


def append_torn_tail(storage_dir: Path) -> dict[str, Any]:
    paths = epoch_log_paths(storage_dir)
    if not paths:
        return {"ok": False, "error": "no epoch log"}
    first = paths[0].read_bytes()[:17]
    with paths[0].open("ab") as fp:
        fp.write(first)
        fp.flush()
        os.fsync(fp.fileno())
    return {"ok": True, "path": relpath(paths[0]), "bytes_appended": len(first)}


def append_duplicate_record(storage_dir: Path) -> dict[str, Any]:
    paths = epoch_log_paths(storage_dir)
    if not paths:
        return {"ok": False, "error": "no epoch log"}
    first = paths[0].read_bytes()[:EPOCH_RECORD_SIZE]
    if len(first) != EPOCH_RECORD_SIZE:
        return {"ok": False, "error": "short first record"}
    with paths[0].open("ab") as fp:
        fp.write(first)
        fp.flush()
        os.fsync(fp.fileno())
    return {"ok": True, "path": relpath(paths[0]), "bytes_appended": len(first)}


def truncate_journal_sidecar(storage_dir: Path) -> dict[str, Any]:
    paths = epoch_log_paths(storage_dir)
    if not paths:
        return {"ok": False, "error": "no epoch log"}
    journal_path = Path(str(paths[0]).removesuffix(".pqcepoch") + ".pqcmeta")
    if not journal_path.exists():
        return {"ok": False, "error": "no journal sidecar",
                "path": relpath(journal_path)}
    before = journal_path.stat().st_size
    with journal_path.open("r+b") as fp:
        fp.truncate(0)
        fp.flush()
        os.fsync(fp.fileno())
    return {
        "ok": True,
        "path": relpath(journal_path),
        "bytes_before": before,
        "bytes_after": journal_path.stat().st_size,
    }


def run_case(case_dir: Path, label: str, mutation: str | None) -> dict[str, Any]:
    storage_dir = Path(tempfile.mkdtemp(prefix=f"aegisq_{label}_storage_"))
    mount_dir = Path(tempfile.mkdtemp(prefix=f"aegisq_{label}_mnt_"))
    remount_dir = Path(tempfile.mkdtemp(prefix=f"aegisq_{label}_remount_"))
    case_dir.mkdir(parents=True, exist_ok=True)
    trace_path = case_dir / "publication_trace.jsonl"
    if trace_path.exists():
        trace_path.unlink()
    payload = (f"epoch-replay:{label}:".encode("ascii") +
               bytes((i % 251 for i in range(12288))))
    env = {
        "PQC_PUBLICATION_MODE": "epoch-redo-log",
        "PQC_PUBLICATION_TRACE_PATH": str(trace_path),
    }

    fuse = None
    remount = None
    write_client: dict[str, Any] = {}
    read_client: dict[str, Any] = {}
    first_unmount: dict[str, Any] = {}
    second_unmount: dict[str, Any] = {}
    mutation_result: dict[str, Any] = {"ok": True, "mutation": mutation}
    error: str | None = None
    try:
        fuse = start_fuse(storage_dir, mount_dir, case_dir / "write_mount",
                          f"epoch-replay-{label}", env)
        write_client = mounted_write_read(mount_dir, "replay.dat", payload)
        first_unmount = stop_fuse(fuse, mount_dir, case_dir / "write_mount")
        fuse = None

        before_mutation = scan_epoch_logs(storage_dir)
        if mutation == "torn_tail":
            mutation_result = append_torn_tail(storage_dir)
        elif mutation == "duplicate_generation":
            mutation_result = append_duplicate_record(storage_dir)
        elif mutation == "journal_loss":
            mutation_result = truncate_journal_sidecar(storage_dir)
        after_mutation = scan_epoch_logs(storage_dir)

        remount = start_fuse(storage_dir, remount_dir, case_dir / "read_mount",
                             f"epoch-replay-{label}", env)
        read_client = mounted_read(remount_dir, "replay.dat", payload)
    except Exception as exc:  # noqa: BLE001 - retained as artifact evidence
        error = repr(exc)
        before_mutation = scan_epoch_logs(storage_dir)
        after_mutation = before_mutation
    finally:
        if remount is not None:
            second_unmount = stop_fuse(remount, remount_dir,
                                       case_dir / "read_mount")
        if fuse is not None:
            first_unmount = stop_fuse(fuse, mount_dir,
                                      case_dir / "write_mount")
        trace = parse_jsonl(trace_path)
        final_scan = scan_epoch_logs(storage_dir)
        shutil.rmtree(mount_dir, ignore_errors=True)
        shutil.rmtree(remount_dir, ignore_errors=True)
        shutil.rmtree(storage_dir, ignore_errors=True)

    compact_events = trace["compact_events"]
    if mutation == "duplicate_generation":
        pass_condition = (
            error is None and
            write_client.get("matches") is True and
            first_unmount.get("returncode") == 0 and
            mutation_result.get("ok") is True and
            read_client.get("opened") is False and
            any(int(event.get("rc", 0)) == -17 for event in compact_events) and
            any(int(event.get("duplicate_generation_records", 0)) >= 1
                for event in compact_events)
        )
    elif mutation == "journal_loss":
        pass_condition = (
            error is None and
            write_client.get("matches") is True and
            read_client.get("matches") is True and
            first_unmount.get("returncode") == 0 and
            second_unmount.get("returncode") == 0 and
            mutation_result.get("ok") is True and
            int(mutation_result.get("bytes_before", 0)) > 0 and
            int(mutation_result.get("bytes_after", -1)) == 0 and
            any(int(event.get("rc", -1)) == 0 for event in compact_events) and
            any(int(event.get("journal_repair_records", 0)) >= 1
                for event in compact_events)
        )
    else:
        expected_torn = 17 if mutation == "torn_tail" else 0
        pass_condition = (
            error is None and
            write_client.get("matches") is True and
            read_client.get("matches") is True and
            first_unmount.get("returncode") == 0 and
            second_unmount.get("returncode") == 0 and
            mutation_result.get("ok") is True and
            any(int(event.get("rc", -1)) == 0 for event in compact_events) and
            any(int(event.get("torn_tail_bytes", 0)) == expected_torn
                for event in compact_events)
        )
    return {
        "label": label,
        "mutation": mutation,
        "write_client": write_client,
        "read_client": read_client,
        "first_unmount": first_unmount,
        "second_unmount": second_unmount,
        "mutation_result": mutation_result,
        "before_mutation": before_mutation,
        "after_mutation": after_mutation,
        "final_scan": final_scan,
        "trace": trace,
        "error": error,
        "pass": pass_condition,
    }


def write_outputs(out_dir: Path, payload: dict[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "epoch_replay_fault_matrix.json"
    md_path = out_dir / "epoch_replay_fault_matrix.md"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n",
                         encoding="utf-8")
    lines = [
        "# Epoch Replay Fault Matrix",
        "",
        f"- Generated: `{payload['generated_utc']}`",
        f"- Overall pass: `{str(payload['overall_pass']).lower()}`",
        "",
    ]
    for case in payload["cases"]:
        compact_events = case["trace"]["compact_events"]
        last_rc = compact_events[-1].get("rc") if compact_events else None
        lines.append(f"## {case['label']}")
        lines.append(f"- Pass: `{str(case['pass']).lower()}`")
        lines.append(f"- Mutation: `{case['mutation']}`")
        lines.append(f"- Remount read matches: `{str(case['read_client'].get('matches')).lower()}`")
        lines.append(f"- Last compaction rc: `{last_rc}`")
        repair_records = [
            int(event.get("journal_repair_records", 0) or 0)
            for event in compact_events
        ]
        lines.append(f"- Journal repair records max: `{max(repair_records) if repair_records else 0}`")
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

    normal = run_case(args.out_dir / "replay_normal", "replay_normal", None)
    torn = run_case(args.out_dir / "replay_torn_tail", "replay_torn_tail",
                    "torn_tail")
    duplicate = run_case(args.out_dir / "replay_duplicate_generation",
                         "replay_duplicate_generation",
                         "duplicate_generation")
    journal_loss = run_case(args.out_dir / "replay_journal_loss",
                            "replay_journal_loss", "journal_loss")
    payload = {
        "schema_version": 1,
        "generated_by": "experiments/run_epoch_replay_fault_matrix.py",
        "generated_utc": now_utc(),
        "scope": "Gate 0.9-S3/S5 committed-prefix replay, checkpoint compaction, and recovery-time journal repair proof.",
        "cases": [normal, torn, duplicate, journal_loss],
        "overall_pass": (
            normal["pass"] and torn["pass"] and duplicate["pass"] and
            journal_loss["pass"]
        ),
        "negative_claim_guard": (
            "This matrix proves mounted remount compaction for journal-backed "
            "committed epoch prefixes, torn-tail ignore, and duplicate "
            "generation rejection. It also proves recovery-time journal repair "
            "from a committed epoch prefix after deliberate journal-sidecar "
            "loss. It does not prove physical power-loss certification, "
            "throughput improvement, group-commit amortization, or rollback "
            "resistance."
        ),
    }
    write_outputs(args.out_dir, payload)
    print(json.dumps({
        "overall_pass": payload["overall_pass"],
        "json": relpath(args.out_dir / "epoch_replay_fault_matrix.json"),
        "case_pass": {case["label"]: case["pass"] for case in payload["cases"]},
    }, indent=2, sort_keys=True))
    return 0 if payload["overall_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
