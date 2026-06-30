#!/usr/bin/env python3
"""Smoke-test production anchor root-transition tracing for C6.

This does not claim async Merkle+TPM epoch freshness.  It verifies the first
production-path code edit needed for that gate: committed-prefix root transitions
can now be observed from the anchor store path with `PQC_ANCHOR_TRACE_PATH`.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
FUSE_BIN = ROOT / "build" / "pqc_fuse"
DEFAULT_OUT = ROOT / "artifacts" / "validation" / "async_merkle_tpm_epoch"


def relpath(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def read_trace(path: Path) -> tuple[list[dict[str, Any]], int]:
    events: list[dict[str, Any]] = []
    malformed = 0
    if not path.exists():
        return events, malformed
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            malformed += 1
            continue
        if isinstance(value, dict):
            events.append(value)
        else:
            malformed += 1
    return events, malformed


def source_checks() -> dict[str, bool]:
    anchor = (ROOT / "code" / "storage" / "pqc_anchor.c").read_text(
        encoding="utf-8", errors="replace"
    )
    header = (ROOT / "code" / "storage" / "pqc_anchor.h").read_text(
        encoding="utf-8", errors="replace"
    )
    return {
        "trace_env_present": "PQC_ANCHOR_TRACE_PATH" in anchor,
        "commit_event_present": "anchor_commit_current_prefix" in anchor,
        "stage_event_present": "anchor_stage_pending" in anchor,
        "flush_event_present": "\"anchor_flush\"" in anchor,
        "epoch_record_type_present": "pqc_anchor_epoch_record_t" in header,
        "epoch_record_snapshot_api_present":
            "pqc_anchor_epoch_record_snapshot" in header,
        "epoch_record_event_present":
            "anchor_epoch_freshness_record" in anchor,
        "pending_status_present": "PQC_ANCHOR_EPOCH_STATUS_PENDING" in header,
        "hardware_flush_policy_present":
            "PQC_ANCHOR_EPOCH_FLUSH_HARDWARE_FORCE" in header,
        "root_hex_recorded": "prefix_root" in anchor and "root_hex" in anchor,
        "duration_recorded": "duration_ns" in anchor,
    }


def is_hex64(value: Any) -> bool:
    return (
        isinstance(value, str) and len(value) == 64 and
        all(ch in "0123456789abcdef" for ch in value)
    )


def valid_epoch_records(events: list[dict[str, Any]],
                        backend: str) -> list[dict[str, Any]]:
    valid: list[dict[str, Any]] = []
    for event in events:
        if event.get("event") != "anchor_epoch_freshness_record":
            continue
        if event.get("backend") != backend:
            continue
        if event.get("status") != "committed" or event.get("committed") is not True:
            continue
        if int(event.get("rc", -1)) != 0:
            continue
        if not is_hex64(event.get("prefix_root")):
            continue
        if int(event.get("global_sequence", 0)) <= 0:
            continue
        if int(event.get("file_count", 0)) <= 0:
            continue
        if int(event.get("epoch_interval", 0)) <= 0:
            continue
        valid.append(event)
    return valid


def mount_is_visible(mount_dir: Path) -> bool:
    mount_path = mount_dir.resolve()
    try:
        with open("/proc/mounts", "r", encoding="utf-8",
                  errors="replace") as proc_mounts:
            for line in proc_mounts:
                fields = line.split()
                if len(fields) >= 3 and fields[2].startswith("fuse"):
                    try:
                        if Path(fields[1]).resolve() == mount_path:
                            return True
                    except OSError:
                        pass
    except FileNotFoundError:
        pass
    return subprocess.run(
        ["mountpoint", "-q", str(mount_dir)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    ).returncode == 0


def stop_fuse(proc: subprocess.Popen[bytes] | None, mount_dir: Path,
              timeout_s: float = 5.0) -> dict[str, Any]:
    result: dict[str, Any] = {"unmounted": False, "returncode": None}
    if mount_is_visible(mount_dir):
        for tool in ("fusermount3", "fusermount"):
            exe = shutil.which(tool)
            if not exe:
                continue
            unmount = subprocess.run(
                [exe, "-uz", str(mount_dir)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
            if unmount.returncode == 0:
                result["unmounted"] = True
                break
    if proc is not None and proc.poll() is None:
        proc.send_signal(signal.SIGINT)
        try:
            proc.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=timeout_s)
    if proc is not None:
        result["returncode"] = proc.returncode
    return result


def run_mounted_smoke(out_dir: Path) -> dict[str, Any]:
    work_dir = Path(tempfile.mkdtemp(prefix="aegisq_anchor_trace_mount_"))
    storage_dir = work_dir / "store"
    mount_dir = work_dir / "mnt"
    storage_dir.mkdir()
    mount_dir.mkdir()
    trace_path = out_dir / "anchor_trace_mounted.jsonl"
    stdout_path = out_dir / "anchor_trace_mounted.stdout.txt"
    stderr_path = out_dir / "anchor_trace_mounted.stderr.txt"
    if trace_path.exists():
        trace_path.unlink()

    env = os.environ.copy()
    env.update({
        "PQC_MASTER_PASSWORD": "anchor-trace-smoke",
        "PQC_FRESHNESS_ANCHOR_BACKEND": "file",
        "PQC_FRESHNESS_ANCHOR_PATH": str(storage_dir / ".anchor"),
        "PQC_FRESHNESS_WINDOW_N": "1",
        "PQC_ANCHOR_TRACE_PATH": str(trace_path),
    })

    proc: subprocess.Popen[bytes] | None = None
    write_ok = False
    mount_error: str | None = None
    stop_result: dict[str, Any] = {}
    try:
        with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
            proc = subprocess.Popen(
                [str(FUSE_BIN), str(storage_dir), str(mount_dir), "-f"],
                cwd=ROOT,
                env=env,
                stdout=stdout,
                stderr=stderr,
            )
            deadline = time.monotonic() + 15.0
            while time.monotonic() < deadline:
                if mount_is_visible(mount_dir):
                    break
                if proc.poll() is not None:
                    mount_error = f"daemon exited before mount: rc={proc.returncode}"
                    break
                time.sleep(0.05)
            else:
                mount_error = "timed out waiting for FUSE mount"

            if mount_error is None:
                mounted_path = mount_dir / "mounted.dat"
                fd = os.open(mounted_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
                try:
                    payload = b"anchor epoch mounted smoke\n" * 8
                    written = os.write(fd, payload)
                    os.fsync(fd)
                    write_ok = written == len(payload)
                finally:
                    os.close(fd)
            stop_result = stop_fuse(proc, mount_dir)
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)

    events, malformed = read_trace(trace_path)
    valid_records = valid_epoch_records(events, "file")
    checks = {
        "mount_started": mount_error is None,
        "write_fsync_ok": write_ok,
        "trace_exists": trace_path.exists(),
        "trace_jsonl_well_formed": malformed == 0 and bool(events),
        "epoch_record_event_valid": bool(valid_records),
    }
    return {
        "command": [relpath(FUSE_BIN), "<storage_dir>", "<mount_dir>", "-f"],
        "stdout": relpath(stdout_path),
        "stderr": relpath(stderr_path),
        "trace_path": relpath(trace_path),
        "mount_error": mount_error,
        "stop_result": stop_result,
        "event_count": len(events),
        "malformed_trace_lines": malformed,
        "valid_epoch_record_count": len(valid_records),
        "events": events,
        "checks": checks,
        "overall_pass": all(checks.values()),
    }


def run_smoke(out_dir: Path) -> dict[str, Any]:
    if not FUSE_BIN.exists():
        raise SystemExit(f"missing build artifact: {relpath(FUSE_BIN)}")

    out_dir.mkdir(parents=True, exist_ok=True)
    work_dir = Path(tempfile.mkdtemp(prefix="aegisq_anchor_trace_"))
    anchor_path = work_dir / "anchor.bin"
    trace_path = out_dir / "anchor_trace_smoke.jsonl"
    stdout_path = out_dir / "anchor_trace_smoke.stdout.txt"
    stderr_path = out_dir / "anchor_trace_smoke.stderr.txt"
    if trace_path.exists():
        trace_path.unlink()

    env = os.environ.copy()
    env.update({
        "PQC_FRESHNESS_ANCHOR_BACKEND": "file",
        "PQC_FRESHNESS_ANCHOR_PATH": str(anchor_path),
        "PQC_ANCHOR_TRACE_PATH": str(trace_path),
    })

    try:
        with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
            proc = subprocess.run(
                [str(FUSE_BIN), "--anchor-self-test"],
                cwd=ROOT,
                env=env,
                stdout=stdout,
                stderr=stderr,
                check=False,
            )
        events, malformed = read_trace(trace_path)
        commit_events = [
            event for event in events
            if event.get("event") == "anchor_commit_current_prefix"
        ]
        valid_commit_events = [
            event for event in commit_events
            if event.get("backend") == "file" and
            int(event.get("rc", -1)) == 0 and
            isinstance(event.get("prefix_root"), str) and
            len(event.get("prefix_root", "")) == 64 and
            int(event.get("global_sequence", 0)) >= 11 and
            int(event.get("file_count", 0)) >= 2
        ]
        valid_epoch_events = valid_epoch_records(events, "file")
        mounted_smoke = run_mounted_smoke(out_dir)
        checks = {
            "self_test_passed": proc.returncode == 0,
            "trace_exists": trace_path.exists(),
            "trace_jsonl_well_formed": malformed == 0 and bool(events),
            "commit_event_valid": bool(valid_commit_events),
            "epoch_record_event_valid": bool(valid_epoch_events),
            "anchor_file_written": anchor_path.exists() and anchor_path.stat().st_size == 88,
            "source_checks_pass": all(source_checks().values()),
            "mounted_smoke_pass": mounted_smoke["overall_pass"],
        }
        return {
            "schema_version": 1,
            "generated_by": "code/experiments/run_anchor_trace_smoke.py",
            "generated_utc": now_utc(),
            "scope": (
                "C6 epoch-record smoke: production anchor store emits "
                "committed-prefix root transitions plus an explicit epoch "
                "freshness record carrying epoch interval, flush policy, "
                "pending/committed status, and prefix root.  This is still "
                "not async Merkle+TPM epoch freshness or PCR-bound rollback "
                "resistance."
            ),
            "command": [relpath(FUSE_BIN), "--anchor-self-test"],
            "returncode": proc.returncode,
            "stdout": relpath(stdout_path),
            "stderr": relpath(stderr_path),
            "trace_path": relpath(trace_path),
            "anchor_file_size": anchor_path.stat().st_size if anchor_path.exists() else 0,
            "event_count": len(events),
            "malformed_trace_lines": malformed,
            "valid_epoch_record_count": len(valid_epoch_events),
            "events": events,
            "mounted_smoke": mounted_smoke,
            "source_checks": source_checks(),
            "checks": checks,
            "overall_pass": all(checks.values()),
            "next_code_edit": (
                "Exercise the hardware TPM path with latency, anti-hammering, "
                "PCR drift, stale snapshot replay, reboot recovery, and mount "
                "refusal metadata before any rollback-resistance claim."
            ),
        }
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


def write_outputs(out_dir: Path, payload: dict[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "anchor_trace_smoke.json"
    md_path = out_dir / "anchor_trace_smoke.md"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n",
                         encoding="utf-8")
    lines = [
        "# Anchor Trace Smoke",
        "",
        f"- Overall pass: `{str(payload['overall_pass']).lower()}`",
        f"- Trace path: `{payload['trace_path']}`",
        f"- Event count: `{payload['event_count']}`",
        f"- Valid epoch records: `{payload['valid_epoch_record_count']}`",
        f"- Mounted smoke pass: `{str(payload['mounted_smoke']['overall_pass']).lower()}`",
        f"- Next code edit: {payload['next_code_edit']}",
        "",
        "## Checks",
        "",
    ]
    for key, value in payload["checks"].items():
        lines.append(f"- `{key}`: `{str(value).lower()}`")
    lines.extend(["", "## Boundary", "", payload["scope"], ""])
    md_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    payload = run_smoke(args.out_dir)
    write_outputs(args.out_dir, payload)
    print(json.dumps({
        "overall_pass": payload["overall_pass"],
        "json": relpath(args.out_dir / "anchor_trace_smoke.json"),
        "event_count": payload["event_count"],
        "failed_checks": [
            key for key, value in payload["checks"].items() if not value
        ],
        "next_code_edit": payload["next_code_edit"],
    }, indent=2, sort_keys=True))
    return 0 if payload["overall_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
